"""Unit tests for RSGrove spatial partitioner.

Tests cover:
- EnvelopeNDLite operations (merge, area, margin)
- Spatial partitioning with different configurations
- Weighted partitioning
- Edge cases (empty inputs, single points)
- R*-tree style split selection
"""
import numpy as np
import pytest

from starlet._internal.tiling.RSGrove import (
    EnvelopeNDLite,
    AuxiliarySearchStructure,
    RSGrovePartitioner,
    BeastOptions,
    partition_points,
    partition_weighted_points,
)


class TestEnvelopeNDLite:
    """Test the N-dimensional envelope (bounding box) class."""

    def test_create_2d_envelope(self):
        """Test creating a 2D envelope from min/max arrays."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 10.0]))
        assert env.getCoordinateDimension() == 2
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0

    def test_envelope_normalization(self):
        """Test that envelopes auto-normalize swapped min/max."""
        env = EnvelopeNDLite(np.array([10.0, 10.0]), np.array([0.0, 0.0]))
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0

    def test_empty_envelope(self):
        """Test detection of empty envelopes."""
        env = EnvelopeNDLite(np.array([0.0]), np.array([0.0]))
        env.setEmpty()  # Use setEmpty() method
        assert env.isEmpty()

    def test_merge_point(self):
        """Test expanding envelope to include a point."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        env.merge_point(np.array([10.0, 3.0]))
        assert env.getMaxCoord(0) == 10.0
        assert env.getMaxCoord(1) == 5.0

    def test_merge_box(self):
        """Test merging two envelopes."""
        env1 = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([5.0, 5.0]))
        env2 = EnvelopeNDLite(np.array([3.0, 3.0]), np.array([10.0, 10.0]))
        env1.merge_box(env2)
        assert env1.getMinCoord(0) == 0.0
        assert env1.getMaxCoord(0) == 10.0

    def test_area_2d(self):
        """Test area calculation for 2D envelope."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 5.0]))
        assert env.area() == 50.0

    def test_margin_2d(self):
        """Test margin (perimeter proxy) calculation."""
        env = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 5.0]))
        # Margin is sum of side lengths
        assert env.margin() == 15.0

    def test_from_points(self):
        """Test creating envelope from point array."""
        coords = np.array([[0.0, 5.0, 10.0], [0.0, 3.0, 7.0]])
        env = EnvelopeNDLite.from_points(coords)
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 10.0
        assert env.getMinCoord(1) == 0.0
        assert env.getMaxCoord(1) == 7.0


class TestAuxiliarySearchStructure:
    """Test the overlap search structure for partition lookup."""

    def test_build_and_search(self):
        """Test building index and searching for overlapping partitions."""
        boxes = [
            EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
            EnvelopeNDLite(np.array([5.0, 5.0]), np.array([15.0, 15.0])),
            EnvelopeNDLite(np.array([20.0, 20.0]), np.array([30.0, 30.0])),
        ]

        aux = AuxiliarySearchStructure()
        aux.build(boxes)

        # Query overlapping first two boxes
        query = EnvelopeNDLite(np.array([7.0, 7.0]), np.array([12.0, 12.0]))
        result = aux.search(query)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_search_no_overlap(self):
        """Test searching for non-overlapping query."""
        boxes = [
            EnvelopeNDLite(np.array([0.0, 0.0]), np.array([10.0, 10.0])),
        ]

        aux = AuxiliarySearchStructure()
        aux.build(boxes)

        query = EnvelopeNDLite(np.array([20.0, 20.0]), np.array([30.0, 30.0]))
        result = aux.search(query)
        assert len(result) == 0


class TestPartitionFunctions:
    """Test the core partitioning algorithms."""

    def test_partition_points_simple(self):
        """Test partitioning a small set of points."""
        # Create 100 random points in 2D
        np.random.seed(42)
        coords = np.random.rand(2, 100) * 100.0

        boxes = partition_points(coords, min_cap=10, max_cap=30,
                                expand_to_inf=False, fraction_min_split=0.0)

        assert len(boxes) > 0
        # With 100 points and max_cap=30, expect 4-10 partitions
        assert 3 <= len(boxes) <= 12

    def test_partition_weighted_points(self):
        """Test weighted partitioning."""
        np.random.seed(42)
        coords = np.random.rand(2, 50) * 100.0
        weights = np.random.randint(1, 10, size=50).astype(float)

        boxes = partition_weighted_points(
            coords, weights,
            min_cap_w=20.0, max_cap_w=60.0,
            expand_to_inf=False, fraction_min_split=0.0
        )

        assert len(boxes) > 0

    def test_partition_preserves_all_points(self):
        """Test that all points fall within some partition."""
        np.random.seed(42)
        coords = np.random.rand(2, 50) * 100.0

        boxes = partition_points(coords, min_cap=5, max_cap=15,
                                expand_to_inf=True, fraction_min_split=0.0)

        # Check each point is covered by at least one box
        for i in range(coords.shape[1]):
            point = coords[:, i]
            covered = False
            for box in boxes:
                if np.all(box.mins <= point) and np.all(point <= box.maxs):
                    covered = True
                    break
            assert covered, f"Point {i} not covered by any partition"

    def test_partition_empty_input(self):
        """Test partitioning with no points."""
        coords = np.empty((2, 0), dtype=float)
        boxes = partition_points(coords, min_cap=1, max_cap=10,
                                expand_to_inf=False, fraction_min_split=0.0)
        # Should handle gracefully, may return empty or single box
        assert isinstance(boxes, list)


class TestRSGrovePartitioner:
    """Test the main RSGrove partitioner class."""

    def test_setup(self):
        """Test partitioner configuration."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions({
            "mmratio": 0.9,
            "RSGrove.MinSplitRatio": 0.1,
            "RSGrove.ExpandToInf": True
        })
        partitioner.setup(conf, disjoint=True)

        assert partitioner.mMRatio == 0.9
        assert partitioner.fractionMinSplitSize == 0.1
        assert partitioner.expandToInf is True
        assert partitioner.isDisjoint()

    def test_construct_unweighted(self, mock_summary, sample_coords_2d):
        """Test constructing partitions without histogram weighting."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)

        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        assert partitioner.numPartitions() > 0
        assert partitioner.getCoordinateDimension() == 2

    def test_construct_weighted(self, mock_summary, mock_histogram):
        """Test constructing partitions with histogram weighting."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)

        # Sample some points
        np.random.seed(42)
        sample = np.random.rand(2, 50) * 100.0

        partitioner.construct(mock_summary, sample,
                             histogram=mock_histogram, numPartitions=3)

        assert partitioner.numPartitions() > 0

    def test_overlap_partitions(self, mock_summary, sample_coords_2d):
        """Test finding partitions that overlap a query envelope."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=False)
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        # Query a region
        query = EnvelopeNDLite(np.array([20.0, 20.0]), np.array([30.0, 30.0]))
        result = partitioner.overlapPartitions(query)

        assert isinstance(result, list)
        assert len(result) >= 1  # Should overlap at least one partition

    def test_overlap_partition_single(self, mock_summary, sample_coords_2d):
        """Test selecting single best partition for an envelope."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=False)
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        query = EnvelopeNDLite(np.array([20.0, 20.0]), np.array([30.0, 30.0]))
        pid = partitioner.overlapPartition(query)

        assert 0 <= pid < partitioner.numPartitions()

    def test_empty_envelope_random_assignment(self, mock_summary, sample_coords_2d):
        """Test that empty envelopes get random partition assignment."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=5)

        empty = EnvelopeNDLite(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        empty.setEmpty()  # Use setEmpty() method to create proper empty envelope
        result = partitioner.overlapPartitions(empty)

        # Should assign to exactly one random partition
        assert len(result) == 1
        assert 0 <= result[0] < partitioner.numPartitions()

    def test_get_partition_mbr(self, mock_summary, sample_coords_2d):
        """Test retrieving partition bounding box."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=3)

        mbr_out = EnvelopeNDLite(np.array([0.0]), np.array([0.0]))
        partitioner.getPartitionMBR(0, mbr_out)

        assert mbr_out.getCoordinateDimension() == 2
        assert not mbr_out.isEmpty()

    def test_get_envelope(self, mock_summary, sample_coords_2d):
        """Test retrieving global envelope."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)
        partitioner.construct(mock_summary, sample_coords_2d,
                             histogram=None, numPartitions=3)

        env = partitioner.getEnvelope()
        assert env.getCoordinateDimension() == 2
        assert env.getMinCoord(0) == 0.0
        assert env.getMaxCoord(0) == 100.0

    def test_compute_point_weights(self, mock_histogram):
        """Test weight computation from histogram."""
        np.random.seed(42)
        sample = np.random.rand(2, 30) * 100.0

        weights = RSGrovePartitioner.computePointWeights(sample, mock_histogram)

        assert len(weights) == 30
        assert np.all(weights >= 0)
        assert weights.dtype == np.int64

    def test_partition_with_fabricated_points(self, mock_summary):
        """Test that empty sample triggers point fabrication."""
        partitioner = RSGrovePartitioner()
        conf = BeastOptions()
        partitioner.setup(conf, disjoint=True)

        empty_sample = np.empty((2, 0), dtype=float)
        partitioner.construct(mock_summary, empty_sample,
                             histogram=None, numPartitions=3)

        # Should still create partitions using fabricated points
        assert partitioner.numPartitions() > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_point_partition(self):
        """Test partitioning a single point."""
        coords = np.array([[5.0], [5.0]])
        boxes = partition_points(coords, min_cap=1, max_cap=10,
                                expand_to_inf=False, fraction_min_split=0.0)
        assert len(boxes) == 1

    def test_two_points_partition(self):
        """Test partitioning exactly two points."""
        coords = np.array([[0.0, 10.0], [0.0, 10.0]])
        boxes = partition_points(coords, min_cap=1, max_cap=1,
                                expand_to_inf=False, fraction_min_split=0.0)
        assert len(boxes) == 2

    def test_collinear_points(self):
        """Test partitioning collinear points."""
        coords = np.array([[float(i) for i in range(20)],
                          [0.0] * 20])
        boxes = partition_points(coords, min_cap=2, max_cap=5,
                                expand_to_inf=False, fraction_min_split=0.0)
        assert len(boxes) >= 4
