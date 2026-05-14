# Starlet Test Suite

Comprehensive pytest test suite for the Starlet geospatial MVT generation system.

## Overview

This test suite covers all major components of Starlet:

- **Tiling** (`test_tiling/`): RSGrove partitioner, orchestrator, data sources
- **MVT Generation** (`test_mvt/`): Tile assignment, rendering, helpers
- **Histograms** (`test_histogram/`): Spatial histogram pyramid building
- **Statistics** (`test_stats/`): Attribute sketches and statistics collection
- **Server** (`test_server/`): Flask tile server and API endpoints

## Installation

Install test dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` - Test framework
- `pytest-cov` - Coverage reporting
- `pytest-mock` - Mocking utilities

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test module

```bash
pytest tests/test_tiling/test_rsgrove.py
pytest tests/test_mvt/test_assigner.py
```

### Run specific test class or function

```bash
pytest tests/test_tiling/test_rsgrove.py::TestEnvelopeNDLite
pytest tests/test_mvt/test_helpers.py::TestHistogramQueries::test_hist_value_exact_zoom
```

### Run with coverage

```bash
pytest --cov=starlet --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`.

### Run with verbose output

```bash
pytest -v
```

### Skip slow tests

Some tests are marked as `@pytest.mark.slow` (e.g., full pipeline tests). Skip them:

```bash
pytest -m "not slow"
```

### Run only slow/integration tests

```bash
pytest -m slow
```

## Test Organization

### `conftest.py`

Shared fixtures used across all tests:

- `temp_dir` - Temporary directory for file I/O tests
- `sample_points`, `sample_polygons` - Geometry fixtures
- `sample_parquet_file` - Pre-built test Parquet data
- `sample_histogram`, `sample_prefix_sum` - Test histogram data
- `sample_dataset_dir` - Complete dataset directory structure
- `mock_summary`, `mock_histogram` - Mock objects for partitioner tests

### Tiling Tests

#### `test_rsgrove.py`

Tests for RSGrove spatial partitioner:

- **TestEnvelopeNDLite**: Bounding box operations (merge, area, margin)
- **TestAuxiliarySearchStructure**: Partition overlap search
- **TestPartitionFunctions**: Core partitioning algorithms
- **TestRSGrovePartitioner**: Full partitioner API
- **TestEdgeCases**: Boundary conditions (single point, collinear points)

#### `test_datasource.py`

Tests for data source readers:

- **TestGeoParquetSource**: GeoParquet reading and iteration
- **TestGeoJSONSource**: GeoJSON reading (if implemented)
- **TestDataSourceIntegration**: Cross-source consistency

#### `test_orchestrator.py`

Tests for tiling orchestrator (templates - adapt to actual API):

- Tiling coordination
- Parallel write management
- Sort modes (Z-order, Hilbert)

### MVT Tests

#### `test_helpers.py`

Tests for MVT helper functions:

- **TestHistogramQueries**: Prefix-sum histogram queries at different zoom levels
- **TestMercatorTileBounds**: Web Mercator tile bounds calculations
- **TestMercatorBoundsToTileRange**: Geometry bounds to tile range conversion
- **TestGeometryExplosion**: GeometryCollection handling
- **TestCoordinateConstants**: Web Mercator constants validation

#### `test_assigner.py`

Tests for tile assignment with priority sampling:

- **TestNonemptyTileDetection**: Histogram-based nonempty tile detection
- **TestAutoZoomDetection**: Auto-detection of maximum useful zoom level (IMPORTANT)
- **TestGeometryAssignment**: Priority-based geometry assignment
- **TestBucketOutput**: Output format validation
- **TestEdgeCases**: Boundary geometries, empty geometries

#### `test_renderer.py`

Tests for MVT tile rendering:

- **TestTileRenderer**: Tile rendering and MVT encoding
- **TestGeometryTransformation**: Coordinate transformation to tile space
- **TestErrorHandling**: Invalid geometry handling
- **TestEdgeCases**: Edge cases (zoom 0, high zoom, directory creation)

#### `test_generator.py`

Tests for MVT generation orchestration (templates - adapt to actual API):

- Generator initialization
- Full pipeline integration

### Histogram Tests

#### `test_hist_pyramid.py`

Tests for histogram pyramid generation:

- **TestGeometryVertexIteration**: Vertex extraction from various geometry types
- **TestHistConfig**: Configuration validation
- **TestHistogramBuilding**: Full histogram building pipeline (marked `@pytest.mark.slow`)
- **TestEdgeCases**: Small/large grid sizes, different dtypes

### Statistics Tests

#### `test_sketches.py`

Tests for attribute statistics sketches:

- **TestSpaceSavingTopK**: Frequency tracking for top-k values
- **TestNumericSketch**: Numeric statistics (min, max, mean, stddev, cardinality)
- **TestCategoricalSketch**: Categorical statistics (distinct count, top-k)
- **TestTextSketch**: Text statistics (length tracking)
- **TestGeometrySketch**: Geometry statistics (MBR, type distribution, vertex counts)
- **TestEdgeCases**: Null handling, single values, large cardinality

### Server Tests

#### `test_app.py`

Tests for Flask tile server:

- **TestApplicationFactory**: App creation and configuration
- **TestDatasetListingAPI**: `/api/datasets` endpoint
- **TestTileServing**: `/<dataset>/<z>/<x>/<y>.mvt` endpoint
- **TestIndexPage**: Index page rendering
- **TestFeatureDownload**: Feature download endpoints (CSV, GeoJSON)
- **TestErrorHandling**: Error cases (404, invalid requests)
- **TestIntegration**: End-to-end workflows

## Test Markers

Custom pytest markers defined in `pytest.ini`:

- `@pytest.mark.slow` - Long-running tests (full pipeline, large datasets)
- `@pytest.mark.integration` - Integration tests across modules
- `@pytest.mark.unit` - Unit tests (default)

## Writing New Tests

### Test Structure

Follow the Arrange-Act-Assert pattern:

```python
def test_feature(fixture):
    """Test description explaining what is being validated."""
    # Arrange - set up test data
    data = prepare_data()

    # Act - execute the code under test
    result = function_under_test(data)

    # Assert - verify the result
    assert result == expected_value
```

### Using Fixtures

Use fixtures from `conftest.py` for common test data:

```python
def test_with_temp_dir(temp_dir):
    """Test that uses temporary directory."""
    file_path = temp_dir / "output.txt"
    file_path.write_text("test")
    assert file_path.exists()
```

### Mocking

Use `pytest-mock` for mocking file I/O or external dependencies:

```python
def test_with_mock(mocker):
    """Test using mocked file I/O."""
    mock_open = mocker.patch("builtins.open")
    # ... test code
```

### Marking Slow Tests

Mark long-running tests as slow:

```python
@pytest.mark.slow
def test_full_pipeline(sample_dataset_dir):
    """Test the complete tiling and MVT generation pipeline."""
    # ... test code
```

## Coverage

Target coverage: **>80%** for core modules.

Check coverage report:

```bash
pytest --cov=starlet --cov-report=term-missing
```

Modules to prioritize:
- `starlet/_internal/tiling/RSGrove.py` - Core partitioning logic
- `starlet/_internal/mvt/assigner.py` - Tile assignment (including auto-zoom)
- `starlet/_internal/mvt/helpers.py` - MVT utilities
- `starlet/_internal/stats/sketches.py` - Statistics collection

## Continuous Integration

Tests should be run before:
- Creating a pull request
- Tagging a release
- Merging to main branch

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=starlet --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Troubleshooting

### Tests fail with import errors

Ensure package is installed in development mode:

```bash
pip install -e .
```

### Slow tests timeout

Increase timeout or skip slow tests:

```bash
pytest -m "not slow" --timeout=300
```

### Coverage not showing all files

Ensure `pytest.ini` has correct source paths and omit patterns.

### Fixtures not found

Check that `conftest.py` is in the correct location and properly formatted.

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure tests cover:
   - Happy path (normal operation)
   - Edge cases (empty inputs, boundary values)
   - Error cases (invalid inputs, missing files)
3. Add docstrings explaining what each test validates
4. Use descriptive test names: `test_<feature>_<condition>_<expected_result>`
5. Keep tests fast - mock slow operations
6. Mark slow tests with `@pytest.mark.slow`

## Important Test Cases

### Auto-Zoom Detection

The `test_assigner.py::TestAutoZoomDetection` class contains critical tests for the auto-zoom feature that determines optimal zoom levels based on data density. These tests validate:

- Dense data handling (high zoom support)
- Sparse data handling (zoom limitation)
- Custom occupancy thresholds
- Return value validation

This feature is essential for preventing over-tiling of sparse datasets.

### Priority-Based Sampling

The `test_assigner.py::TestGeometryAssignment` class tests the priority-based sampling that ensures consistent feature selection across tile boundaries, preventing visual seams in MVT output.

### Histogram Queries

The `test_helpers.py::TestHistogramQueries` class validates the O(1) prefix-sum queries that enable efficient nonempty tile detection without scanning the entire dataset.

## License

Tests are part of the Starlet project and follow the same MIT license.
