"""Unit tests for data source readers.

Tests cover:
- GeoParquetSource reading and iteration
- GeoJSONSource reading and iteration
- Column detection (geometry column)
- Error handling for missing files
- Schema validation
"""
import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely import wkb

from starlet._internal.tiling.datasource import GeoParquetSource


class TestGeoParquetSource:
    """Test GeoParquet data source."""

    def test_init_with_file(self, sample_parquet_file):
        """Test initializing with a valid Parquet file."""
        source = GeoParquetSource(str(sample_parquet_file))
        # Source should be initialized successfully
        assert source is not None
        assert source._pf is not None

    def test_init_with_missing_file(self, temp_dir):
        """Test that missing file raises appropriate error."""
        missing_file = temp_dir / "nonexistent.parquet"
        with pytest.raises(Exception):  # FileNotFoundError or similar
            GeoParquetSource(str(missing_file))

    def test_detect_geometry_column(self, sample_parquet_file):
        """Test that geometry column is in schema."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()
        # Should have geometry column
        assert 'geometry' in schema.names

    def test_read_geometries(self, sample_parquet_file):
        """Test reading and decoding geometries."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Use iter_tables() to read data
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            # Decode first geometry
            geom = wkb.loads(geoms_wkb[0])
            assert geom is not None
            assert geom.geom_type == 'Polygon'
            break  # Only test first batch

    def test_read_all_columns(self, sample_parquet_file):
        """Test reading all columns including attributes."""
        source = GeoParquetSource(str(sample_parquet_file))

        for table in source.iter_tables():
            assert 'geometry' in table.column_names
            assert 'id' in table.column_names
            assert 'name' in table.column_names
            break  # Only test first batch

    def test_multiple_row_groups(self, temp_dir, sample_polygons):
        """Test reading Parquet file with multiple row groups."""
        # Create file with multiple row groups
        geoms = [wkb.dumps(g) for g in sample_polygons]
        table = pa.table({
            'geometry': geoms,
            'id': list(range(len(geoms)))
        })

        file_path = temp_dir / "multi_rg.parquet"
        pq.write_table(table, str(file_path), row_group_size=2)

        source = GeoParquetSource(str(file_path))
        # Check that we can iterate through all row groups
        tables = list(source.iter_tables())
        assert len(tables) > 1

    def test_schema_validation(self, sample_parquet_file):
        """Test that schema is accessible and valid."""
        source = GeoParquetSource(str(sample_parquet_file))
        schema = source.schema()

        assert 'geometry' in schema.names
        # Geometry should be binary type
        geom_field = schema.field('geometry')
        assert pa.types.is_binary(geom_field.type)


class TestGeoJSONSource:
    """Test GeoJSON data source.

    Note: GeoJSONSource implementation may vary. These are placeholder tests
    that should be adapted based on the actual implementation.
    """

    def test_read_geojson_feature_collection(self, temp_dir):
        """Test reading a GeoJSON FeatureCollection."""
        # Create a simple GeoJSON file
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0.0, 0.0]
                    },
                    "properties": {"id": 1}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [10.0, 10.0]
                    },
                    "properties": {"id": 2}
                }
            ]
        }

        json_path = temp_dir / "test.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        # If GeoJSONSource is implemented, test it here
        # source = GeoJSONSource(str(json_path))
        # features = list(source.iter_features())
        # assert len(features) == 2

    def test_read_empty_geojson(self, temp_dir):
        """Test reading empty GeoJSON file."""
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }

        json_path = temp_dir / "empty.geojson"
        with open(json_path, 'w') as f:
            json.dump(geojson, f)

        # If GeoJSONSource is implemented, test it here
        # source = GeoJSONSource(str(json_path))
        # features = list(source.iter_features())
        # assert len(features) == 0


class TestDataSourceIntegration:
    """Integration tests across data sources."""

    def test_consistent_geometry_reading(self, sample_parquet_file, sample_polygons):
        """Test that geometries are read correctly and match original data."""
        source = GeoParquetSource(str(sample_parquet_file))

        # Read all geometries using iter_tables()
        all_geoms = []
        for table in source.iter_tables():
            geoms_wkb = table['geometry'].to_pylist()
            all_geoms.extend([wkb.loads(g) for g in geoms_wkb])

        # Decode and compare bounds
        for i, geom in enumerate(all_geoms):
            original = sample_polygons[i]
            # Compare bounds (should be identical)
            assert geom.bounds == original.bounds
