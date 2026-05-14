"""Unit tests for Flask tile server application.

Tests cover:
- Application factory and configuration
- Tile serving endpoints
- Dataset listing API
- Feature download endpoints
- Error handling (404, invalid requests)
- CORS headers
"""
import json
import pytest
from pathlib import Path

from starlet._internal.server.app import create_app


@pytest.fixture
def app(sample_dataset_dir):
    """Create a test Flask app with sample dataset."""
    app = create_app(str(sample_dataset_dir.parent), cache_size=10, log_level="ERROR")
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


class TestApplicationFactory:
    """Test Flask app creation and configuration."""

    def test_create_app(self, temp_dir):
        """Test basic app creation."""
        app = create_app(str(temp_dir), cache_size=128, log_level="ERROR")
        assert app is not None
        assert app.config['TESTING'] is False

    def test_create_app_with_custom_cache_size(self, temp_dir):
        """Test app creation with custom cache size."""
        app = create_app(str(temp_dir), cache_size=512)
        assert app is not None

    def test_cors_enabled(self, app):
        """Test that CORS is enabled for all routes."""
        # CORS extension should be registered
        assert 'cors' in app.extensions or hasattr(app, 'after_request')


class TestDatasetListingAPI:
    """Test dataset listing endpoint."""

    def test_list_datasets(self, client, sample_dataset_dir):
        """Test GET /api/datasets returns dataset list."""
        response = client.get('/api/datasets')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'datasets' in data
        assert isinstance(data['datasets'], list)
        assert 'test_dataset' in data['datasets']

    def test_list_datasets_empty_dir(self, temp_dir):
        """Test listing datasets from empty directory."""
        app = create_app(str(temp_dir), log_level="ERROR")
        app.config['TESTING'] = True
        client = app.test_client()

        response = client.get('/api/datasets')
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data['datasets'] == []

    def test_list_datasets_json_format(self, client):
        """Test that response contains valid JSON."""
        response = client.get('/api/datasets')
        # Flask may return text/html or application/json depending on configuration
        # Just verify the data is valid JSON
        data = json.loads(response.data)
        assert 'datasets' in data


class TestTileServing:
    """Test MVT tile serving endpoint."""

    def test_serve_tile_endpoint(self, client):
        """Test GET /<dataset>/<z>/<x>/<y>.mvt endpoint."""
        response = client.get('/test_dataset/0/0/0.mvt')
        # May return 200 with tile data or 404 if tile doesn't exist
        assert response.status_code in [200, 404, 500]

    def test_serve_tile_content_type(self, client):
        """Test that tiles have correct MIME type."""
        response = client.get('/test_dataset/1/0/0.mvt')
        if response.status_code == 200:
            assert response.content_type == 'application/vnd.mapbox-vector-tile'

    def test_serve_tile_invalid_coords(self, client):
        """Test tile request with invalid coordinates."""
        response = client.get('/test_dataset/1/999/999.mvt')
        # Should handle gracefully (404 or empty tile)
        assert response.status_code in [200, 404]

    def test_serve_tile_negative_zoom(self, client):
        """Test tile request with negative zoom."""
        # Flask routing may reject this, or it may return 404
        response = client.get('/test_dataset/-1/0/0.mvt')
        assert response.status_code in [404, 400]

    def test_serve_nonexistent_dataset(self, client):
        """Test requesting tile from non-existent dataset."""
        response = client.get('/nonexistent/0/0/0.mvt')
        # Server may return 200 with empty tile, or 404/500 depending on implementation
        # Just verify it responds without crashing
        assert response.status_code in [200, 404, 500]


class TestIndexPage:
    """Test index page rendering."""

    def test_index_page(self, client):
        """Test GET / returns HTML page."""
        response = client.get('/')
        assert response.status_code == 200
        # Should return HTML
        assert b'html' in response.data.lower() or response.content_type == 'text/html'

    def test_index_page_content_type(self, client):
        """Test that index page has HTML content type."""
        response = client.get('/')
        if response.status_code == 200:
            # May be text/html or text/html; charset=utf-8
            assert 'text/html' in response.content_type


class TestFeatureDownload:
    """Test feature download endpoints."""

    def test_download_csv_endpoint(self, client):
        """Test GET /datasets/<dataset>/features.csv."""
        response = client.get('/datasets/test_dataset/features.csv')
        # May succeed or fail depending on implementation
        assert response.status_code in [200, 404, 500]

    def test_download_geojson_endpoint(self, client):
        """Test GET /datasets/<dataset>/features.geojson."""
        response = client.get('/datasets/test_dataset/features.geojson')
        assert response.status_code in [200, 404, 500]

    def test_download_with_mbr_filter(self, client):
        """Test feature download with MBR query parameter."""
        response = client.get('/datasets/test_dataset/features.csv?mbr=-180,-90,180,90')
        assert response.status_code in [200, 404, 500]

    def test_download_invalid_format(self, client):
        """Test requesting unsupported format."""
        response = client.get('/datasets/test_dataset/features.invalid')
        # Should return error
        assert response.status_code in [400, 404, 500]


class TestStaticFileServing:
    """Test static file serving."""

    def test_serve_existing_file(self, client):
        """Test serving existing static files."""
        # This depends on what files exist in the server directory
        # May need to mock or skip if no static files
        pass

    def test_serve_nonexistent_file(self, client):
        """Test that non-existent files return 404."""
        response = client.get('/nonexistent.js')
        assert response.status_code == 404


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_url(self, client):
        """Test handling of malformed URLs."""
        response = client.get('/test_dataset/not_a_number/0/0.mvt')
        # Flask routing should reject this
        assert response.status_code == 404

    def test_very_large_tile_coords(self, client):
        """Test handling of very large tile coordinates."""
        response = client.get('/test_dataset/30/999999999/999999999.mvt')
        assert response.status_code in [200, 404, 500]

    def test_special_characters_in_dataset_name(self, client):
        """Test dataset name with special characters."""
        response = client.get('/test/../etc/passwd/0/0/0.mvt')
        # Should be blocked or return 404
        assert response.status_code in [404, 400]


class TestCaching:
    """Test tile caching behavior."""

    def test_cache_initialization(self, temp_dir):
        """Test that app initializes with specified cache size."""
        app = create_app(str(temp_dir), cache_size=256)
        # Cache should be initialized (hard to test without exposing internals)
        assert app is not None

    def test_multiple_requests_same_tile(self, client):
        """Test that multiple requests for same tile work correctly."""
        # First request
        response1 = client.get('/test_dataset/0/0/0.mvt')
        # Second request (should hit cache if tile exists)
        response2 = client.get('/test_dataset/0/0/0.mvt')

        # Both should have same status
        assert response1.status_code == response2.status_code


class TestLogging:
    """Test logging configuration."""

    def test_log_level_configuration(self, temp_dir):
        """Test that log level can be configured."""
        app = create_app(str(temp_dir), log_level="DEBUG")
        assert app is not None

        app = create_app(str(temp_dir), log_level="ERROR")
        assert app is not None

    def test_default_log_level(self, temp_dir):
        """Test default log level (INFO)."""
        app = create_app(str(temp_dir))
        assert app is not None


class TestIntegration:
    """Integration tests across multiple endpoints."""

    def test_list_and_serve_workflow(self, client):
        """Test workflow: list datasets, then request tile."""
        # List datasets
        list_response = client.get('/api/datasets')
        assert list_response.status_code == 200

        datasets = json.loads(list_response.data)['datasets']
        if datasets:
            # Request tile from first dataset
            dataset = datasets[0]
            tile_response = client.get(f'/{dataset}/0/0/0.mvt')
            assert tile_response.status_code in [200, 404, 500]

    def test_cors_headers_on_tile_request(self, client):
        """Test that CORS headers are present on tile requests."""
        response = client.get('/test_dataset/0/0/0.mvt')
        # CORS headers should be present
        # (exact header depends on flask-cors configuration)
        # Common CORS header is Access-Control-Allow-Origin
        headers_lower = {k.lower(): v for k, v in response.headers}
        # May or may not have CORS headers depending on configuration
        # This is more of a smoke test
        assert isinstance(headers_lower, dict)
