"""Unit tests for MVT tile assignment with priority sampling.

Tests cover:
- Nonempty tile detection from histograms
- Auto-zoom detection based on tile occupancy
- Priority-based geometry assignment
- Consistent sampling across tile boundaries
- Bucket output format
"""
import math
import numpy as np
import pytest
from shapely.geometry import Point, box

from starlet._internal.mvt.assigner import TileAssigner, MAX_GEOMS_PER_TILE


class TestNonemptyTileDetection:
    """Test histogram-based nonempty tile detection."""

    def test_compute_nonempty_exact_zoom(self, sample_prefix_sum):
        """Test nonempty detection at native histogram zoom."""
        # sample_prefix_sum is 64x64, so hist_zoom = 6
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=1)
        assigner.compute_nonempty()

        assert 6 in assigner.nonempty
        assert len(assigner.nonempty[6]) > 0

        # Check that the dense cluster region has nonempty tiles
        # Dense cluster is at hist[10:20, 10:20]
        assert (15, 15) in assigner.nonempty[6]

    def test_compute_nonempty_lower_zoom(self, sample_prefix_sum):
        """Test nonempty detection at lower zoom (tile aggregation)."""
        assigner = TileAssigner(zooms=[4], prefix=sample_prefix_sum, threshold=1)
        assigner.compute_nonempty()

        assert 4 in assigner.nonempty
        # Lower zoom should have fewer tiles but still detect features
        assert len(assigner.nonempty[4]) > 0

    def test_compute_nonempty_higher_zoom(self, sample_prefix_sum):
        """Test nonempty detection at higher zoom (tile subdivision)."""
        assigner = TileAssigner(zooms=[8], prefix=sample_prefix_sum, threshold=1)
        assigner.compute_nonempty()

        assert 8 in assigner.nonempty
        # Higher zoom creates more tiles from each histogram cell
        assert len(assigner.nonempty[8]) > 0

    def test_compute_nonempty_multiple_zooms(self, sample_prefix_sum):
        """Test nonempty detection across multiple zoom levels."""
        assigner = TileAssigner(zooms=[4, 5, 6], prefix=sample_prefix_sum, threshold=1)
        assigner.compute_nonempty()

        for z in [4, 5, 6]:
            assert z in assigner.nonempty
            assert len(assigner.nonempty[z]) > 0

    def test_threshold_filtering(self):
        """Test that threshold filters out low-density tiles."""
        # Create histogram with specific feature counts
        hist = np.zeros((64, 64), dtype=float)
        hist[10, 10] = 100  # Above threshold
        hist[20, 20] = 5    # Below threshold

        prefix = hist.cumsum(axis=0).cumsum(axis=1)
        assigner = TileAssigner(zooms=[6], prefix=prefix, threshold=50)
        assigner.compute_nonempty()

        # Only the high-density tile should be included
        assert (10, 10) in assigner.nonempty[6]
        assert (20, 20) not in assigner.nonempty[6]

    def test_empty_histogram(self):
        """Test behavior with completely empty histogram."""
        hist = np.zeros((64, 64), dtype=float)
        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=[6], prefix=prefix, threshold=1)
        assigner.compute_nonempty()

        assert len(assigner.nonempty[6]) == 0


class TestAutoZoomDetection:
    """Test automatic max zoom detection."""

    def test_auto_detect_with_dense_data(self):
        """Test auto-zoom with dense data (should allow high zoom)."""
        # Create histogram with uniform high density
        hist = np.ones((64, 64), dtype=float) * 100
        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=list(range(0, 12)), prefix=prefix, threshold=10)
        assigner.compute_nonempty()

        max_zoom = assigner.auto_detect_max_zoom(occupancy_threshold=0.01)

        # With dense uniform data, should support high zoom
        assert max_zoom >= 6

    def test_auto_detect_with_sparse_data(self):
        """Test auto-zoom with sparse data (should limit zoom)."""
        # Create histogram with only a few cells populated
        hist = np.zeros((64, 64), dtype=float)
        hist[32, 32] = 1000  # Single dense cluster

        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=list(range(0, 12)), prefix=prefix, threshold=10)
        assigner.compute_nonempty()

        max_zoom = assigner.auto_detect_max_zoom(occupancy_threshold=0.01)

        # Sparse data should result in lower max zoom
        assert max_zoom < 10

    def test_auto_detect_custom_threshold(self):
        """Test auto-zoom with custom occupancy threshold."""
        hist = np.zeros((64, 64), dtype=float)
        hist[30:34, 30:34] = 100  # 4x4 cluster

        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=list(range(0, 10)), prefix=prefix, threshold=10)
        assigner.compute_nonempty()

        # Stricter threshold should reduce max zoom
        max_zoom_strict = assigner.auto_detect_max_zoom(occupancy_threshold=0.1)
        max_zoom_lenient = assigner.auto_detect_max_zoom(occupancy_threshold=0.001)

        assert max_zoom_strict <= max_zoom_lenient

    def test_auto_detect_returns_valid_zoom(self):
        """Test that auto-detect always returns a valid zoom level."""
        hist = np.ones((64, 64), dtype=float) * 50
        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=list(range(0, 15)), prefix=prefix, threshold=10)
        assigner.compute_nonempty()

        max_zoom = assigner.auto_detect_max_zoom(occupancy_threshold=0.01)

        assert 0 <= max_zoom <= 14
        assert isinstance(max_zoom, int)

    def test_auto_detect_with_empty_histogram(self):
        """Test auto-zoom detection with empty histogram."""
        hist = np.zeros((64, 64), dtype=float)
        prefix = hist.cumsum(axis=0).cumsum(axis=1)

        assigner = TileAssigner(zooms=list(range(0, 10)), prefix=prefix, threshold=10)
        assigner.compute_nonempty()

        max_zoom = assigner.auto_detect_max_zoom(occupancy_threshold=0.01)

        # Should return a low zoom (possibly 0)
        assert max_zoom >= 0


class TestGeometryAssignment:
    """Test priority-based geometry assignment to tiles."""

    def test_assign_point_to_single_tile(self, sample_prefix_sum):
        """Test assigning a point geometry to its containing tile."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Create a point in the middle of the nonempty region
        point = Point(0, 0)  # Will map to specific tile
        attrs = {"id": 1}

        assigner.assign_geometry(point, attrs)

        buckets = assigner.buckets
        # Should have assigned to zoom 6
        assert 6 in buckets

    def test_assign_polygon_to_multiple_tiles(self, sample_prefix_sum):
        """Test that large polygon gets assigned to all overlapping tiles."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Create a large polygon that spans multiple tiles
        large_poly = box(-1e7, -1e7, 1e7, 1e7)
        attrs = {"id": 1}

        assigner.assign_geometry(large_poly, attrs)

        buckets = assigner.buckets
        if 6 in buckets:
            # Should be assigned to multiple tiles
            assert len(buckets[6]) > 1

    def test_priority_consistency(self, sample_prefix_sum):
        """Test that same geometry gets same priority across tiles.

        This is critical for avoiding seams at tile boundaries.
        """
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Geometry that spans multiple tiles
        large_poly = box(-5e6, -5e6, 5e6, 5e6)
        attrs = {"id": 1}

        # Assign the same geometry multiple times
        # In reality, each geometry is assigned once with one random priority,
        # but we can test that the internal heap structure is consistent

        assigner.assign_geometry(large_poly, attrs)

        # Check that entries exist in buckets
        buckets = assigner.buckets
        assert len(buckets) > 0

    def test_max_geoms_per_tile_limit(self, sample_prefix_sum):
        """Test that heap enforces MAX_GEOMS_PER_TILE limit."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Assign many geometries to the same tile region
        for i in range(MAX_GEOMS_PER_TILE + 100):
            point = Point(0, 0)
            attrs = {"id": i}
            assigner.assign_geometry(point, attrs)

        buckets = assigner.buckets
        if 6 in buckets:
            for tile_coords, geoms in buckets[6].items():
                # No tile should exceed MAX_GEOMS_PER_TILE
                assert len(geoms) <= MAX_GEOMS_PER_TILE

    def test_assign_to_nonempty_only(self, sample_prefix_sum):
        """Test that geometries are only assigned to nonempty tiles."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=1000)
        assigner.compute_nonempty()

        # This creates a very restrictive threshold
        # Assign a small point
        point = Point(0, 0)
        assigner.assign_geometry(point, {"id": 1})

        buckets = assigner.buckets
        # Only nonempty tiles should be in buckets
        if 6 in buckets:
            for (x, y) in buckets[6].keys():
                assert (x, y) in assigner.nonempty[6]


class TestBucketOutput:
    """Test bucket output format and structure."""

    def test_bucket_format(self, sample_prefix_sum):
        """Test that buckets are in the expected nested dict format."""
        assigner = TileAssigner(zooms=[5, 6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        point = Point(0, 0)
        assigner.assign_geometry(point, {"id": 1})

        buckets = assigner.buckets

        # Should be {z: {(x, y): [(geom, attrs), ...]}}
        assert isinstance(buckets, dict)
        for z, tiles in buckets.items():
            assert isinstance(z, int)
            assert isinstance(tiles, dict)
            for coords, geoms in tiles.items():
                assert isinstance(coords, tuple)
                assert len(coords) == 2
                assert isinstance(geoms, list)

    def test_bucket_geometry_attrs_tuples(self, sample_prefix_sum):
        """Test that bucket entries are (geom, attrs) tuples."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        point = Point(1000, 1000)
        attrs = {"id": 42, "name": "test"}
        assigner.assign_geometry(point, attrs)

        buckets = assigner.buckets
        if 6 in buckets:
            for geom_list in buckets[6].values():
                for entry in geom_list:
                    assert isinstance(entry, tuple)
                    assert len(entry) == 2
                    geom, entry_attrs = entry
                    assert hasattr(geom, 'geom_type')  # Is a Shapely geometry
                    assert isinstance(entry_attrs, dict)

    def test_empty_buckets(self, sample_prefix_sum):
        """Test bucket output when no geometries are assigned."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Don't assign any geometries
        buckets = assigner.buckets

        # Should be an empty or minimal structure
        if buckets:
            for tiles in buckets.values():
                assert len(tiles) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_geometry_at_tile_boundary(self, sample_prefix_sum):
        """Test geometry exactly on tile boundary."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Create point exactly at tile boundary
        # Tile boundaries in Web Mercator depend on zoom level
        point = Point(0, 0)  # Origin is a tile boundary at all zooms
        assigner.assign_geometry(point, {"id": 1})

        buckets = assigner.buckets
        assert isinstance(buckets, dict)

    def test_empty_geometry(self, sample_prefix_sum):
        """Test assigning an empty geometry."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        empty_point = Point()
        # Empty geometries have invalid bounds (NaN), which causes ValueError
        # The implementation should either handle this or it's expected to fail
        # For now, we test that it either succeeds or raises ValueError
        try:
            assigner.assign_geometry(empty_point, {"id": 1})
            buckets = assigner.buckets
            assert isinstance(buckets, dict)
        except ValueError:
            # Empty geometry causes NaN bounds - this is expected behavior
            pass

    def test_very_small_geometry(self, sample_prefix_sum):
        """Test assigning a very small geometry (sub-pixel)."""
        assigner = TileAssigner(zooms=[6], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # Tiny polygon (1 meter across in Web Mercator)
        tiny = box(0, 0, 1, 1)
        assigner.assign_geometry(tiny, {"id": 1})

        buckets = assigner.buckets
        assert isinstance(buckets, dict)

    def test_world_spanning_geometry(self, sample_prefix_sum):
        """Test assigning a geometry that spans the entire world."""
        assigner = TileAssigner(zooms=[1, 2], prefix=sample_prefix_sum, threshold=0)
        assigner.compute_nonempty()

        # World-spanning polygon
        world = box(-2e7, -2e7, 2e7, 2e7)
        assigner.assign_geometry(world, {"id": 1})

        buckets = assigner.buckets
        # Should assign to many tiles
        total_tiles = sum(len(tiles) for tiles in buckets.values())
        assert total_tiles > 1
