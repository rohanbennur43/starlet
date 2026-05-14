# Starlet Tile Server

Flask application that serves vector tiles (MVT) and dataset metadata from
pre-processed GeoParquet data.

## Running

```bash
# Install the package
pip install -e .

# Start the server
starlet serve --dir <data_directory> [--host 0.0.0.0] [--port 8765] [--cache-size 256]
```

Or via the Makefile (uses the legacy `server/server.py` entry point):

```bash
make server   # http://127.0.0.1:5000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive dataset selector page |
| `GET` | `/api/datasets` | List all available datasets |
| `GET` | `/datasets.json` | Search datasets by name |
| `GET` | `/datasets/<dataset>.json` | Dataset metadata |
| `GET` | `/datasets/<dataset>.html` | Dataset detail page |
| `GET` | `/<dataset>/<z>/<x>/<y>.mvt` | Mapbox Vector Tile |
| `GET` | `/datasets/<dataset>/features.<fmt>` | Download features (csv/geojson) |
| `POST` | `/datasets/<dataset>/features.<fmt>` | Download with geometry filter |
| `GET` | `/datasets/<dataset>/features/sample.json` | Sample non-geometry attributes |
| `GET` | `/datasets/<dataset>/features/sample.geojson` | Sample record with geometry |
| `GET` | `/api/datasets/<dataset>/stats` | Precomputed attribute statistics |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
