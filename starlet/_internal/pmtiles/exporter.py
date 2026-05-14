"""Export MVT tiles to PMTiles archive format."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from pmtiles.writer import Writer
from pmtiles.tile import TileType, Compression, zxy_to_tileid

logger = logging.getLogger(__name__)


def export_to_pmtiles(
    mvt_dir: str,
    output_path: str,
    tile_type: str = "mvt",
    compression: str = "gzip",
) -> str:
    """Export MVT tiles directory to a single PMTiles archive.

    Parameters
    ----------
    mvt_dir : str
        Directory containing MVT tiles in z/x/y.mvt structure.
    output_path : str
        Path to output .pmtiles file.
    tile_type : str
        Tile type: "mvt" (vector tiles) or "png"/"jpg"/"webp" (raster).
        Default "mvt".
    compression : str
        Compression format: "gzip", "none", "brotli", "zstd".
        Default "gzip".

    Returns
    -------
    str
        Path to created PMTiles file.

    Examples
    --------
    >>> export_to_pmtiles(
    ...     mvt_dir="datasets/mydata/mvt",
    ...     output_path="datasets/mydata.pmtiles"
    ... )
    """
    mvt_path = Path(mvt_dir)
    if not mvt_path.exists():
        raise FileNotFoundError(f"MVT directory not found: {mvt_dir}")

    # Map string args to pmtiles enums
    tile_type_map = {
        "mvt": TileType.MVT,
        "png": TileType.PNG,
        "jpg": TileType.JPEG,
        "jpeg": TileType.JPEG,
        "webp": TileType.WEBP,
    }
    compression_map = {
        "none": Compression.NONE,
        "gzip": Compression.GZIP,
        "brotli": Compression.BROTLI,
        "zstd": Compression.ZSTD,
    }

    tt = tile_type_map.get(tile_type.lower(), TileType.MVT)
    comp = compression_map.get(compression.lower(), Compression.GZIP)

    logger.info(f"Exporting MVT tiles from {mvt_dir} to {output_path}")
    logger.info(f"Tile type: {tile_type}, Compression: {compression}")

    # Collect all MVT tiles
    tiles = []
    for tile_file in mvt_path.rglob("*.mvt"):
        rel_path = tile_file.relative_to(mvt_path)
        parts = rel_path.parts

        if len(parts) != 3:
            logger.warning(f"Skipping invalid tile path: {tile_file}")
            continue

        try:
            z = int(parts[0])
            x = int(parts[1])
            y = int(parts[2].replace(".mvt", ""))
        except ValueError:
            logger.warning(f"Skipping non-numeric tile path: {tile_file}")
            continue

        tile_data = tile_file.read_bytes()
        tiles.append((z, x, y, tile_data))

    if not tiles:
        raise ValueError(f"No MVT tiles found in {mvt_dir}")

    logger.info(f"Found {len(tiles)} tiles to export")

    # Write PMTiles archive
    with open(output_path, "wb") as f:
        writer = Writer(f)

        for z, x, y, data in sorted(tiles):
            tileid = zxy_to_tileid(z, x, y)
            writer.write_tile(tileid, data)

        # Header with tile type and compression
        header = {
            "tile_type": tt,
            "tile_compression": comp,
        }
        writer.finalize(header, {})

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"PMTiles export complete: {output_path} ({file_size_mb:.2f} MB)")

    return output_path
