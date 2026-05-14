import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mapbox_vector_tile
from shapely import affinity, make_valid
from shapely.geometry import box, mapping

from .helpers import EXTENT, mercator_tile_bounds, explode_geom

logger = logging.getLogger(__name__)


def _render_one_tile(task):
    """
    Worker function: build a single MVT tile in memory and return encoded bytes.

    Returns:
        tuple[z, x, y, data_bytes, feature_count] or None if tile is empty
    """
    z, x, y, geoms = task

    if not geoms:
        return None

    tb_minx, tb_miny, tb_maxx, tb_maxy = mercator_tile_bounds(z, x, y)
    tb_width = tb_maxx - tb_minx
    tb_height = tb_maxy - tb_miny

    pad = tb_width * (256 / EXTENT)
    padded = box(
        tb_minx - pad,
        tb_miny - pad,
        tb_maxx + pad,
        tb_maxy + pad,
    )

    simplify_tol = tb_width * 0.0005

    x_scale = EXTENT / tb_width if tb_width != 0 else 0.0
    y_scale = EXTENT / tb_height if tb_height != 0 else 0.0
    affine_params = (
        x_scale,               # a
        0.0,                   # b
        0.0,                   # d
        y_scale,               # e
        -tb_minx * x_scale,    # xoff
        -tb_miny * y_scale,    # yoff
    )

    features = []

    for g, attrs in geoms:
        if g is None or g.is_empty:
            continue

        try:
            g2 = g.simplify(simplify_tol, preserve_topology=True)
        except Exception:
            g2 = g

        if g2.is_empty:
            continue

        if not g2.is_valid:
            try:
                g2 = make_valid(g2)
            except Exception:
                try:
                    g2 = g2.buffer(0)
                except Exception:
                    continue

        try:
            clipped = g2.intersection(padded)
        except Exception:
            try:
                repaired = g2.buffer(0)
                clipped = repaired.intersection(padded)
            except Exception:
                continue

        if clipped.is_empty:
            continue

        if not clipped.is_valid:
            try:
                clipped = make_valid(clipped)
            except Exception:
                try:
                    clipped = clipped.buffer(0)
                except Exception:
                    continue

        properties = {k: v for k, v in attrs.items() if v is not None}

        for part in explode_geom(clipped):
            if part.is_empty:
                continue

            try:
                transformed = affinity.affine_transform(part, affine_params)
            except Exception:
                continue

            if transformed.is_empty:
                continue

            features.append(
                {
                    "geometry": mapping(transformed),
                    "properties": properties,
                }
            )

    if not features:
        return None

    layer = {
        "name": "layer0",
        "features": features,
        "extent": EXTENT,
    }
    data = mapbox_vector_tile.encode([layer])

    return z, x, y, data, len(features)


class TileRenderer:
    def __init__(self, outdir, max_workers=None):
        logger.info("Initializing TileRenderer with outdir=%s", outdir)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        cpu_default = max(1, multiprocessing.cpu_count() - 1)
        self.max_workers = max(1, int(max_workers or cpu_default))

    def render(self, buckets):
        logger.info(
            "Starting tile rendering for %d zoom levels with up to %d workers",
            len(buckets),
            self.max_workers,
        )

        total = 0
        tasks = []

        # Flatten all work first
        for z, tiles in buckets.items():
            logger.info("Queueing zoom %s: %d tiles", z, len(tiles))
            zoom_dir = self.outdir / str(z)
            zoom_dir.mkdir(parents=True, exist_ok=True)

            for (x, y), geoms in tiles.items():
                tasks.append((z, x, y, geoms))

        if not tasks:
            logger.info("No tiles to render.")
            return

        # Small workloads do not benefit much from process overhead
        if len(tasks) == 1:
            result = _render_one_tile(tasks[0])
            if result is not None:
                z, x, y, data, feature_count = result
                x_dir = self.outdir / str(z) / str(x)
                x_dir.mkdir(parents=True, exist_ok=True)
                with open(x_dir / f"{y}.mvt", "wb") as f:
                    f.write(data)
                total += 1
                logger.debug(
                    "Wrote tile z=%s x=%s y=%s with %d features",
                    z, x, y, feature_count,
                )
            logger.info("Rendering complete. Wrote %d MVT tiles to %s", total, self.outdir)
            return

        # Parallel render, serial write
        zoom_counts = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_render_one_tile, task): task[:3] for task in tasks}

            for fut in as_completed(futures):
                tile_key = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    z, x, y = tile_key
                    logger.error(
                        "Failed rendering tile z=%s x=%s y=%s",
                        z, x, y,
                        exc_info=True,
                    )
                    continue

                if result is None:
                    continue

                z, x, y, data, feature_count = result

                x_dir = self.outdir / str(z) / str(x)
                x_dir.mkdir(parents=True, exist_ok=True)

                with open(x_dir / f"{y}.mvt", "wb") as f:
                    f.write(data)

                total += 1
                zoom_counts[z] = zoom_counts.get(z, 0) + 1

                logger.debug(
                    "Wrote tile z=%s x=%s y=%s with %d features",
                    z, x, y, feature_count,
                )

        for z in sorted(zoom_counts):
            logger.info("Finished zoom %s: wrote %d tiles", z, zoom_counts[z])

        logger.info("Rendering complete. Wrote %d MVT tiles to %s", total, self.outdir)