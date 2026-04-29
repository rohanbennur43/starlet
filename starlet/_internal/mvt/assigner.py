"""Tile assignment with priority-based sampling for cross-tile consistent MVT generation.

Each geometry receives a single random priority when it enters the pipeline.
Per-tile buckets are min-heaps of size ``MAX_GEOMS_PER_TILE`` ordered by
priority.  Because the same geometry carries the same priority into every
tile it overlaps, adjacent tiles make consistent keep/drop decisions for
shared boundary features — eliminating the seam artifacts that arise from
independent per-tile reservoir sampling.
"""

import heapq
import logging
import math
import random
from collections import defaultdict

import numpy as np

from .helpers import hist_value_from_prefix, mercator_bounds_to_tile_range

logger = logging.getLogger(__name__)

MAX_GEOMS_PER_TILE = 25000


class TileAssigner:
    def __init__(self, zooms, prefix, threshold):
        logger.debug(f"Initializing TileAssigner: zooms={zooms}, threshold={threshold}")
        self.zooms = zooms
        self.prefix = prefix
        self.threshold = threshold
        self.nonempty = {z: set() for z in zooms}

        # Each bucket is a min-heap of (priority, seq, (geom, attrs)).
        # The seq counter is a tiebreaker so that heap comparisons never
        # fall through to comparing geometry objects.
        # OPTIMIZATION: Use flat dict with (z, x, y) keys instead of nested {z: {(x,y): heap}}
        # for better data locality and cleaner code
        self._heaps = defaultdict(list)  # key: (z, x, y)
        self._seq = 0

    # ── sampling ──────────────────────────────────────────────────────

    def _priority_insert(self, z, x, y, priority, geom_tuple):
        """Insert into the tile's min-heap, keeping only the top-k by priority."""
        key = (z, x, y)
        heap = self._heaps[key]
        entry = (priority, self._seq, geom_tuple)
        self._seq += 1

        if len(heap) < MAX_GEOMS_PER_TILE:
            heapq.heappush(heap, entry)
        elif priority > heap[0][0]:
            heapq.heapreplace(heap, entry)

    # ── nonempty tile detection ───────────────────────────────────────

    def compute_nonempty(self):
        """Determine nonempty tiles using vectorised histogram lookups.

        Instead of iterating every (x, y) at each zoom (O(4^z)), we recover
        the raw histogram from the prefix-sum array once, then use numpy
        block-reduction or expansion to map histogram cells to tiles.
        """
        logger.debug("Computing nonempty tiles from histogram")
        H, W = self.prefix.shape
        hist_zoom = int(round(math.log2(W)))

        # Recover per-cell values from the prefix-sum table
        padded = np.pad(self.prefix, ((1, 0), (1, 0)), mode='constant')
        raw_hist = (
            padded[1:, 1:] - padded[:-1, 1:] - padded[1:, :-1] + padded[:-1, :-1]
        )

        for z in self.zooms:
            if z == hist_zoom:
                ys, xs = np.nonzero(raw_hist >= self.threshold)
                self.nonempty[z] = set(zip(xs.tolist(), ys.tolist()))

            elif z < hist_zoom:
                scale = 2 ** (hist_zoom - z)
                n = 2 ** z
                trimmed = raw_hist[:n * scale, :n * scale]
                block_sums = trimmed.reshape(n, scale, n, scale).sum(axis=(1, 3))
                ys, xs = np.nonzero(block_sums >= self.threshold)
                self.nonempty[z] = set(zip(xs.tolist(), ys.tolist()))

            else:
                scale = 2 ** (z - hist_zoom)
                divisor = scale * scale
                hy, hx = np.nonzero(raw_hist >= self.threshold * divisor)
                tiles = set()
                for cy, cx in zip(hy.tolist(), hx.tolist()):
                    for dx in range(scale):
                        for dy in range(scale):
                            tiles.add((cx * scale + dx, cy * scale + dy))
                self.nonempty[z] = tiles

            logger.debug(f"Zoom {z}: {len(self.nonempty[z])} nonempty tiles")

    def auto_detect_max_zoom(self, occupancy_threshold=0.01):
        """
        Analyze histogram density to find maximum useful zoom level.

        Returns the highest zoom level where tile occupancy is above the threshold.
        Occupancy = (nonempty_tiles / total_possible_tiles) at each zoom.

        Args:
            occupancy_threshold: Minimum fraction of tiles that must be nonempty (default 0.01 = 1%)

        Returns:
            int: Maximum zoom level with sufficient data density
        """
        H, W = self.prefix.shape
        hist_zoom = int(round(math.log2(W)))
        max_zoom = max(self.zooms)

        # Recover histogram if not already cached
        padded = np.pad(self.prefix, ((1, 0), (1, 0)), mode='constant')
        raw_hist = (
            padded[1:, 1:] - padded[:-1, 1:] - padded[1:, :-1] + padded[:-1, :-1]
        )

        logger.debug(f"Auto-detecting max zoom from histogram (threshold={occupancy_threshold})")

        for z in range(hist_zoom, max_zoom + 1):
            total_tiles = 2 ** (2 * z)  # 2^z × 2^z grid

            # Count nonempty tiles at this zoom
            if z <= hist_zoom:
                # Use already computed nonempty set
                nonempty_count = len(self.nonempty.get(z, set()))
            else:
                # For deeper zooms, estimate from histogram subdivision
                scale = 2 ** (z - hist_zoom)
                divisor = scale * scale
                hist_nonempty = np.count_nonzero(raw_hist >= self.threshold * divisor)
                nonempty_count = hist_nonempty * divisor

            occupancy = nonempty_count / total_tiles
            logger.debug(f"Zoom {z}: {nonempty_count}/{total_tiles} tiles = {occupancy:.6f} occupancy")

            # If occupancy drops below threshold, return previous zoom
            if occupancy < occupancy_threshold:
                detected_max = max(z - 1, 0)
                logger.info(
                    f"Auto-detected max zoom: {detected_max} "
                    f"(zoom {z} occupancy {occupancy:.4f} < threshold {occupancy_threshold})"
                )
                return detected_max

        # All zooms are dense enough
        logger.info(f"Auto-detected max zoom: {max_zoom} (all zooms above threshold)")
        return max_zoom

    # ── geometry assignment ───────────────────────────────────────────

    def assign_geometry(self, geom, attrs):
        """Assign a geometry to all overlapping nonempty tiles.

        A single random priority is drawn once and reused for every tile
        the geometry touches, so the keep/drop decision is consistent
        across tile boundaries.
        """
        minx, miny, maxx, maxy = geom.bounds
        priority = random.random()

        for z in self.zooms:
            tx0, ty0, tx1, ty1 = mercator_bounds_to_tile_range(z, minx, miny, maxx, maxy)
            assigned = 0
            for x in range(tx0, tx1 + 1):
                for y in range(ty0, ty1 + 1):
                    if (x, y) in self.nonempty[z]:
                        self._priority_insert(z, x, y, priority, (geom, attrs))
                        assigned += 1
            if assigned > 0:
                logger.debug(f"Assigned geometry to {assigned} tiles at zoom {z}")

    # ── output interface ──────────────────────────────────────────────

    @property
    def buckets(self):
        """Return tile contents in the format the renderer expects.

        ``{z: {(x, y): [(geom, attrs), ...]}}``

        Converts flat dict structure {(z, x, y): heap} back to nested format.
        """
        out = {}
        for (z, x, y), heap in self._heaps.items():
            tiles = out.setdefault(z, {})
            tiles[(x, y)] = [entry[2] for entry in heap]
        return out
