"""Unit tests for attribute statistics sketches.

Tests cover:
- SpaceSavingTopK frequency tracking
- NumericSketch (min, max, mean, stddev, cardinality)
- CategoricalSketch (distinct count, top-k)
- TextSketch (length statistics)
- GeometrySketch (MBR, type distribution, vertex counts)
"""
import math
import pytest
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely import wkb

from starlet._internal.stats.sketches import (
    SpaceSavingTopK,
    NumericSketch,
    CategoricalSketch,
    TextSketch,
    GeometrySketch,
    TOP_K,
)


class TestSpaceSavingTopK:
    """Test top-k frequency tracking."""

    def test_update_and_result(self):
        """Test basic update and retrieval."""
        tracker = SpaceSavingTopK(k=3)
        tracker.update([1, 2, 3, 1, 2, 1])

        result = tracker.result()
        # Should return top values if they represent >= 80% of data
        assert isinstance(result, list)

    def test_top_k_ordering(self):
        """Test that top-k are ordered by frequency."""
        tracker = SpaceSavingTopK(k=3)
        # Clear distribution: 'a' appears 10 times, 'b' 5 times, 'c' 2 times
        tracker.update(['a'] * 10 + ['b'] * 5 + ['c'] * 2)

        result = tracker.result()
        if result:  # If threshold is met
            # First item should be most frequent
            assert result[0]['value'] == 'a'
            assert result[0]['count'] == 10

    def test_threshold_filtering(self):
        """Test that sparse top-k is filtered when threshold not met."""
        tracker = SpaceSavingTopK(k=3)
        # Add enough unique values so counter has k+1 values (not yet pruned)
        # Each value appears once, so top-3 out of 4 = 75% < 80%
        for i in range(4):
            tracker.update([i])

        result = tracker.result()
        # Top-3 values represent 3/4 = 75% < 80%, so should return empty
        assert len(result) == 0

    def test_bounded_size(self):
        """Test that tracker stays within size bounds."""
        tracker = SpaceSavingTopK(k=5)
        # Add many unique values
        tracker.update(list(range(1000)))

        # Internal counter should be pruned
        assert len(tracker.counter) <= 5 * 2  # k * 2 based on implementation

    def test_empty_input(self):
        """Test behavior with no updates."""
        tracker = SpaceSavingTopK(k=3)
        result = tracker.result()
        assert result == []


class TestNumericSketch:
    """Test numeric attribute statistics."""

    def test_basic_statistics(self):
        """Test min, max, mean computation."""
        sketch = NumericSketch()
        values = [1, 2, 3, 4, 5]
        sketch.update(values)

        result = sketch.finalize()
        assert result['min'] == 1
        assert result['max'] == 5
        assert result['mean'] == 3.0
        assert result['non_null_count'] == 5

    def test_standard_deviation(self):
        """Test standard deviation computation (Welford's algorithm)."""
        sketch = NumericSketch()
        values = [2, 4, 6, 8, 10]  # Mean = 6, variance = 8
        sketch.update(values)

        result = sketch.finalize()
        expected_stddev = math.sqrt(8)
        assert abs(result['stddev'] - expected_stddev) < 1e-6

    def test_with_nulls(self):
        """Test handling of None/NaN values."""
        sketch = NumericSketch()
        values = [1, None, 3, float('nan'), 5]
        sketch.update(values)

        result = sketch.finalize()
        assert result['non_null_count'] == 3  # Only 1, 3, 5
        assert result['min'] == 1
        assert result['max'] == 5

    def test_cardinality_estimation(self):
        """Test approximate distinct count using HyperLogLog."""
        sketch = NumericSketch()
        values = [1, 2, 3, 1, 2, 3, 1, 2, 3]  # 3 distinct values
        sketch.update(values)

        result = sketch.finalize()
        # HLL should approximate 3 distinct values
        assert 2 <= result['approx_distinct'] <= 5

    def test_single_value(self):
        """Test statistics with a single value."""
        sketch = NumericSketch()
        sketch.update([42])

        result = sketch.finalize()
        assert result['min'] == 42
        assert result['max'] == 42
        assert result['mean'] == 42
        assert result['stddev'] == 0.0

    def test_negative_values(self):
        """Test with negative numbers."""
        sketch = NumericSketch()
        values = [-10, -5, 0, 5, 10]
        sketch.update(values)

        result = sketch.finalize()
        assert result['min'] == -10
        assert result['max'] == 10
        assert result['mean'] == 0.0

    def test_floating_point_values(self):
        """Test with floating point numbers."""
        sketch = NumericSketch()
        values = [1.5, 2.5, 3.5]
        sketch.update(values)

        result = sketch.finalize()
        assert result['mean'] == 2.5


class TestCategoricalSketch:
    """Test categorical attribute statistics."""

    def test_distinct_count(self):
        """Test approximate distinct count."""
        sketch = CategoricalSketch()
        values = ['a', 'b', 'c', 'a', 'b', 'c']
        sketch.update(values)

        result = sketch.finalize()
        # Should estimate 3 distinct values
        assert 2 <= result['approx_distinct'] <= 5

    def test_top_k_categorical(self):
        """Test top-k tracking for categorical data."""
        sketch = CategoricalSketch()
        values = ['red'] * 50 + ['blue'] * 30 + ['green'] * 15 + ['yellow'] * 5
        sketch.update(values)

        result = sketch.finalize()
        if result['top_k']:
            # Most common should be 'red'
            assert result['top_k'][0]['value'] == 'red'

    def test_with_nulls(self):
        """Test that nulls are excluded."""
        sketch = CategoricalSketch()
        values = ['a', None, 'b', None, 'c']
        sketch.update(values)

        result = sketch.finalize()
        assert result['non_null_count'] == 3

    def test_empty_sketch(self):
        """Test finalize on empty sketch."""
        sketch = CategoricalSketch()
        result = sketch.finalize()

        assert result['non_null_count'] == 0
        assert result['approx_distinct'] == 0


class TestTextSketch:
    """Test text attribute statistics (extends CategoricalSketch)."""

    def test_length_statistics(self):
        """Test min, max, and average length tracking."""
        sketch = TextSketch()
        values = ['a', 'abc', 'abcde']
        sketch.update(values)

        result = sketch.finalize()
        assert result['min_length'] == 1
        assert result['max_length'] == 5
        assert result['avg_length'] == 3.0

    def test_with_empty_strings(self):
        """Test handling of empty strings."""
        sketch = TextSketch()
        values = ['', 'a', 'abc']
        sketch.update(values)

        result = sketch.finalize()
        assert result['min_length'] == 0
        assert result['max_length'] == 3

    def test_unicode_strings(self):
        """Test with unicode characters."""
        sketch = TextSketch()
        values = ['hello', 'world', 'こんにちは']  # Japanese characters
        sketch.update(values)

        result = sketch.finalize()
        assert result['non_null_count'] == 3
        assert result['min_length'] >= 5

    def test_distinct_text_values(self):
        """Test distinct count for text."""
        sketch = TextSketch()
        values = ['apple', 'banana', 'apple', 'cherry', 'banana']
        sketch.update(values)

        result = sketch.finalize()
        # Should estimate ~3 distinct values
        assert 2 <= result['approx_distinct'] <= 5


class TestGeometrySketch:
    """Test geometry statistics."""

    def test_mbr_computation(self):
        """Test minimum bounding rectangle computation."""
        sketch = GeometrySketch()

        point1 = Point(0, 0)
        point2 = Point(10, 10)
        geoms = [wkb.dumps(point1), wkb.dumps(point2)]

        sketch.update(geoms)
        result = sketch.finalize()

        assert result['mbr'] == [0, 0, 10, 10]

    def test_mbr_with_precomputed_global(self):
        """Test that pre-computed MBR is used when provided."""
        global_mbr = (-180, -90, 180, 90)
        sketch = GeometrySketch(global_mbr=global_mbr)

        # Update with some geometries
        point = Point(50, 50)
        sketch.update([wkb.dumps(point)])

        result = sketch.finalize()
        # Should use pre-computed MBR
        assert result['mbr'] == list(global_mbr)

    def test_geometry_type_counting(self):
        """Test counting of different geometry types."""
        sketch = GeometrySketch()

        point = Point(0, 0)
        line = LineString([(0, 0), (1, 1)])
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])

        geoms = [wkb.dumps(point), wkb.dumps(line), wkb.dumps(poly)]
        sketch.update(geoms)

        result = sketch.finalize()
        assert result['geom_types']['Point'] == 1
        assert result['geom_types']['LineString'] == 1
        assert result['geom_types']['Polygon'] == 1

    def test_vertex_counting(self):
        """Test total vertex/coordinate counting."""
        sketch = GeometrySketch()

        # Point has 1 vertex
        point = Point(0, 0)
        # LineString with 3 vertices
        line = LineString([(0, 0), (1, 1), (2, 2)])

        geoms = [wkb.dumps(point), wkb.dumps(line)]
        sketch.update(geoms)

        result = sketch.finalize()
        assert result['total_points'] == 4  # 1 + 3

    def test_polygon_vertex_counting(self):
        """Test vertex counting for polygons with holes."""
        sketch = GeometrySketch()

        exterior = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]
        hole = [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]
        poly = Polygon(exterior, [hole])

        sketch.update([wkb.dumps(poly)])

        result = sketch.finalize()
        # 5 exterior + 5 hole = 10 vertices
        assert result['total_points'] == 10

    def test_multigeometry_type_counting(self):
        """Test counting for multi-geometries."""
        sketch = GeometrySketch()

        mp = MultiPoint([Point(0, 0), Point(1, 1)])

        sketch.update([wkb.dumps(mp)])

        result = sketch.finalize()
        assert result['geom_types']['MultiPoint'] == 1

    def test_with_empty_geometries(self):
        """Test handling of empty geometries."""
        sketch = GeometrySketch()

        empty_point = Point()
        valid_point = Point(5, 5)

        geoms = [wkb.dumps(empty_point), wkb.dumps(valid_point)]
        sketch.update(geoms)

        result = sketch.finalize()
        # Empty geometry should be skipped
        assert result['geom_types']['Point'] == 1

    def test_with_invalid_wkb(self):
        """Test handling of invalid WKB data."""
        sketch = GeometrySketch()

        # Invalid WKB bytes
        invalid_wkb = b'invalid'
        valid_geom = wkb.dumps(Point(0, 0))

        sketch.update([invalid_wkb, valid_geom])

        result = sketch.finalize()
        # Should skip invalid and count valid
        assert result['geom_types']['Point'] == 1

    def test_mbr_expansion(self):
        """Test that MBR expands to include all geometries."""
        sketch = GeometrySketch()

        point1 = Point(-100, -50)
        point2 = Point(100, 50)

        sketch.update([wkb.dumps(point1)])
        sketch.update([wkb.dumps(point2)])

        result = sketch.finalize()
        assert result['mbr'] == [-100, -50, 100, 50]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_numeric_sketch_all_nulls(self):
        """Test numeric sketch with all null values."""
        sketch = NumericSketch()
        sketch.update([None, None, float('nan')])

        result = sketch.finalize()
        assert result['non_null_count'] == 0
        assert result['min'] is None
        assert result['max'] is None

    def test_numeric_sketch_single_null(self):
        """Test numeric sketch with single null."""
        sketch = NumericSketch()
        sketch.update([None])

        result = sketch.finalize()
        assert result['non_null_count'] == 0

    def test_text_sketch_single_character(self):
        """Test text sketch with single characters."""
        sketch = TextSketch()
        sketch.update(['a', 'b', 'c'])

        result = sketch.finalize()
        assert result['min_length'] == 1
        assert result['max_length'] == 1
        assert result['avg_length'] == 1.0

    def test_geometry_sketch_single_point(self):
        """Test geometry sketch with single point."""
        sketch = GeometrySketch()
        point = Point(5, 5)
        sketch.update([wkb.dumps(point)])

        result = sketch.finalize()
        assert result['mbr'] == [5, 5, 5, 5]

    def test_large_value_cardinality(self):
        """Test cardinality estimation with many distinct values."""
        sketch = NumericSketch()
        values = list(range(10000))
        sketch.update(values)

        result = sketch.finalize()
        # HLL should approximate 10000 (within reasonable error)
        assert 8000 <= result['approx_distinct'] <= 12000
