"""Tile assignment with lazy threshold checks and a safe zoom cap.

This version avoids the catastrophic bbox-to-full-tile-rectangle explosion at
very high zoom levels by capping assignment work to the histogram's native
resolution. For a 4096x4096 histogram, that native zoom is 12.

Why this helps:
- The old code enumerated every tile inside a geometry bbox for every zoom.
- For very large polygons (e.g. countries), that becomes enormous at z=20.
- The histogram query for z > hist_zoom already approximates by evenly dividing
  the parent histogram cell among subtile descendants, so doing explicit
  assignment above hist_zoom is usually wasted work in this pipeline.

Behavior:
- Tiles are checked lazily against the histogram threshold only when a streamed
  geometry actually touches them.
- Results are cached per tile.
- Assignment is capped at min(requested_zoom, hist_zoom).

If you later want true z>hist_zoom tile generation, the correct design is to
refine recursively by geometry/tile intersection, not by full bbox rectangle
enumeration.
"""

import logging
import math
import random
from collections import defaultdict

from .helpers import hist_value_from_prefix, mercator_bounds_to_tile_range

logger = logging.getLogger(__name__)

MAX_GEOMS_PER_TILE = 25000


class TileAssigner:
    def __init__(self, zooms, prefix, threshold):
        logger.debug(
            "Initializing TileAssigner: zooms=%s, threshold=%s",
            zooms,
            threshold,
        )
        self.zooms = list(zooms)
        self.prefix = prefix
        self.threshold = threshold

        # Histogram native zoom inferred from width (e.g. 4096 -> z12)
        h, w = prefix.shape
        if h != w:
            raise ValueError(f"Expected square prefix histogram, got shape={prefix.shape}")
        self.hist_zoom = int(round(math.log2(w)))

        # Cap assignment work to histogram native zoom.
        self.assignment_zooms = [z for z in self.zooms if z <= self.hist_zoom]

        if len(self.assignment_zooms) < len(self.zooms):
            skipped = [z for z in self.zooms if z > self.hist_zoom]
            logger.warning(
                "Capping tile assignment to histogram native zoom z=%s. "
                "Skipping explicit assignment for zooms %s to avoid bbox explosion.",
                self.hist_zoom,
                skipped,
            )

        # Lazy threshold cache: only tiles actually touched by geometries
        self._tile_ok_cache = {z: {} for z in self.assignment_zooms}

        # Only materialize buckets for zooms we actually assign
        self.buckets = {z: defaultdict(list) for z in self.assignment_zooms}

    def _reservoir_insert(self, z, x, y, geom_attrs):
        """Insert geom_attrs into tile bucket using reservoir sampling."""
        bucket = self.buckets[z][(x, y)]
        k = MAX_GEOMS_PER_TILE
        n = len(bucket)

        if n < k:
            bucket.append(geom_attrs)
            return

        # Algorithm R
        j = random.randint(0, n)
        if j < k:
            bucket[j] = geom_attrs

    def _tile_passes_threshold(self, z, x, y):
        """Check tile threshold lazily and cache the result."""
        if self.threshold <= 0:
            return True

        cache = self._tile_ok_cache[z]
        key = (x, y)

        if key in cache:
            return cache[key]

        ok = hist_value_from_prefix(self.prefix, z, x, y) >= self.threshold
        cache[key] = ok
        return ok

    def assign_geometry(self, geom, attrs):
        """Assign one geometry to all overlapping tiles up to hist_zoom."""
        if geom is None or geom.is_empty:
            return

        minx, miny, maxx, maxy = geom.bounds
        logger.debug("Assigning geometry with bounds %s", geom.bounds)

        for z in self.assignment_zooms:
            tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy)

            assigned = 0
            candidate_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)

            logger.debug(
                "Zoom %s bbox tile range: x=[%s,%s], y=[%s,%s], candidates=%s",
                z, tx0, tx1, ty0, ty1, candidate_tiles,
            )

            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    if self._tile_passes_threshold(z, x, y):
                        self._reservoir_insert(z, x, y, (geom, attrs))
                        assigned += 1

            if assigned > 0:
                logger.debug("Assigned geometry to %d tiles at zoom %s", assigned, z)