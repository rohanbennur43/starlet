"""Tile assignment with bounded-memory MVT generation."""
import logging
import random
from collections import defaultdict

from .helpers import hist_value_from_prefix, hist_zoom_from_prefix, mercator_bounds_to_tile_range

logger = logging.getLogger(__name__)

MAX_GEOMS_PER_TILE = 25000


class TileAssigner:
    def __init__(self, zooms, prefix, threshold):
        logger.debug("Initializing TileAssigner: zooms=%s threshold=%s", zooms, threshold)
        self.zooms = zooms
        self.prefix = prefix
        self.threshold = threshold
        self.hist_zoom = hist_zoom_from_prefix(prefix)

        # Lazy cache: only memoize tiles we actually touch.
        self._tile_ok_cache = {z: {} for z in zooms}

        # Only tiles that actually receive features are materialized.
        self.buckets = {z: defaultdict(list) for z in zooms}

    def _reservoir_insert(self, z, x, y, geom_attrs):
        bucket = self.buckets[z][(x, y)]
        k = MAX_GEOMS_PER_TILE
        n = len(bucket)

        if n < k:
            bucket.append(geom_attrs)
            return

        j = random.randint(0, n)
        if j < k:
            bucket[j] = geom_attrs

    def _tile_passes_threshold(self, z, x, y):
        if self.threshold <= 0:
            return True

        cache = self._tile_ok_cache[z]
        key = (x, y)
        cached = cache.get(key)
        if cached is not None:
            return cached

        ok = (
            hist_value_from_prefix(
                self.prefix,
                z,
                x,
                y,
                hist_zoom=self.hist_zoom,
            ) >= self.threshold
        )
        cache[key] = ok
        return ok

    def assign_geometry(self, geom, attrs):
        minx, miny, maxx, maxy = geom.bounds

        for z in self.zooms:
            tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy)

            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    if self._tile_passes_threshold(z, x, y):
                        self._reservoir_insert(z, x, y, (geom, attrs))