"""Flask application factory for the starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional
import json
import logging
import os
import re

from flask import Flask, Response, render_template, send_from_directory, request
from flask_cors import CORS

from .tiler.tiler import VectorTiler
from .download_service import DatasetFeatureService
from .catalog.embedder import GeminiTextEmbedder
from .catalog.index import CATALOG_FILENAME
from .catalog.router import CatalogRouter, SearchBackend
from .catalog.pgvector_store import PgVectorConfig, PgVectorStore

logger = logging.getLogger(__name__)


def create_app(
    data_dir: str,
    cache_size: int = 256,
    log_level: Optional[str] = None,
) -> Flask:
    level = log_level or os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    template_dir = str(Path(__file__).parent / "templates")
    app = Flask(__name__, template_folder=template_dir)
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir)
    tiler_cache: dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    _catalog_runtime: Dict[str, Any] = {
        "router": None,
        "mtime": None,
    }

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(str(data_root / dataset), memory_cache_size=cache_size)
        return tiler_cache[dataset]

    def _extract_first_json_array(text: str) -> List[Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in LLM response")
        parsed = json.loads(match.group())
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array from LLM response")
        return parsed

    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < 0 or end < start:
            raise ValueError("No JSON object found in LLM response")
        parsed = json.loads(cleaned[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object from LLM response")
        return parsed

    def _load_stats_for_dataset(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        stats_path = dataset_path / "stats" / "attributes.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found for dataset: {dataset}")

        with open(stats_path, "r") as f:
            return json.load(f)

    def _generate_styles_from_stats(
        dataset: str,
        stats: Dict[str, Any],
        requested_features: Optional[List[str]] = None,
        instruction_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        attributes = stats.get("attributes", [])
        if not attributes:
            return []

        if requested_features:
            attr_subset = [a for a in attributes if a.get("name") in requested_features]
            instruction = (
                instruction_override
                if instruction_override
                else "Generate styling suggestions for these specific attributes: "
                + ", ".join(requested_features)
            )
        else:
            attr_subset = attributes
            instruction = (
                instruction_override
                if instruction_override
                else "Analyze all attributes and suggest the best styling rules for map visualization."
            )

        if not attr_subset:
            return []

        prompt_path = Path(__file__).parent / "llm" / "prompt.md"
        prompt_template = prompt_path.read_text()

        prompt = prompt_template.replace(
            "{{ATTRIBUTES_JSON}}", json.dumps(attr_subset, indent=2)
        ).replace(
            "{{INSTRUCTION}}", instruction
        )

        from .llm.factory import LLMFactory

        provider = LLMFactory.get_default_provider()
        raw = provider.generate_response(prompt)
        styles = _extract_first_json_array(raw)

        if not isinstance(styles, list):
            return []
        return styles

    def _catalog_index_path() -> Path:
        return data_root / "_catalog" / CATALOG_FILENAME

    def _catalog_backend_from_env() -> SearchBackend:
        value = os.environ.get("CATALOG_SEARCH_BACKEND", "auto").strip().lower()
        if value == "pgvector":
            return SearchBackend.PGVECTOR
        if value == "npy":
            return SearchBackend.NPY
        return SearchBackend.AUTO

    def _build_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                f"Build it first with catalog/index.py."
            )

        embedder = GeminiTextEmbedder()
        backend = _catalog_backend_from_env()

        pg_store = None
        if backend in (SearchBackend.AUTO, SearchBackend.PGVECTOR):
            pg_store = PgVectorStore(PgVectorConfig())

        return CatalogRouter(
            index_dir_or_file=str(index_path.parent),
            embedder=embedder,
            backend=backend,
            pgvector_store=pg_store,
        )

    def _get_catalog_router() -> CatalogRouter:
        index_path = _catalog_index_path()
        if not index_path.exists():
            raise FileNotFoundError(
                f"Catalogue index not found at {index_path}. "
                f"Build it first with catalog/index.py."
            )

        mtime = index_path.stat().st_mtime
        if _catalog_runtime["router"] is None or _catalog_runtime["mtime"] != mtime:
            _catalog_runtime["router"] = _build_catalog_router()
            _catalog_runtime["mtime"] = mtime
        return _catalog_runtime["router"]

    def _candidate_payload_for_llm(candidates) -> List[Dict[str, Any]]:
        payload = []
        for c in candidates:
            payload.append({
                "dataset": c.dataset,
                "score": round(float(c.score), 6),
                "summary": c.summary,
            })
        return payload

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        data = tiler.get_tile(z, x, y)
        elapsed_ms = (perf_counter() - t0) * 1000
        logger.info(
            "[TileRequest] dataset=%s z=%d x=%d y=%d bytes=%d elapsed=%.1fms",
            dataset, z, x, y, len(data), elapsed_ms
        )
        return Response(data, mimetype="application/vnd.mapbox-vector-tile")

    @app.get("/api/datasets")
    def list_datasets():
        datasets = []
        if data_root.exists():
            datasets = sorted([d.name for d in data_root.iterdir() if d.is_dir()])
        return json.dumps({"datasets": datasets})

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html")

    @app.route("/<path:filename>")
    def serve_file(filename):
        server_dir = Path(__file__).parent
        file_path = server_dir / filename
        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), filename)
        return "File not found", 404

    @app.get("/datasets/<dataset>/features.<format>")
    def download_features(dataset, format):
        try:
            mbr_string = request.args.get("mbr", default=None)
            feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            if mbr_string:
                filename = f"{dataset}_{mbr_string.replace(',', '_')}.{format}"
            else:
                filename = f"{dataset}_full.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.post("/datasets/<dataset>/features.<format>")
    def download_features_with_geometry(dataset, format):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json()
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(dataset, format, geometry=geometry)
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/api/datasets/<dataset>/stats")
    def get_dataset_stats(dataset):
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {str(e)}"}, 500

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = []
        if data_root.exists():
            for d in data_root.iterdir():
                if d.is_dir():
                    dataset_metadata = {
                        "id": d.name,
                        "name": d.name.replace("_", " ").title(),
                        "size": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()),
                    }
                    if query is None or query.lower() in d.name.lower():
                        datasets.append(dataset_metadata)
        return json.dumps({"datasets": datasets}, indent=2)

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            metadata = {
                "id": dataset,
                "name": dataset.replace("_", " ").title(),
                "size": sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file()),
                "file_count": sum(1 for f in dataset_path.rglob("*") if f.is_file()),
            }
            return json.dumps(metadata, indent=2)
        except Exception as e:
            return {"error": f"Failed to retrieve metadata: {str(e)}"}, 500

    @app.get("/datasets/<dataset>.html")
    def visualize_dataset(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return "<h1>Dataset not found</h1>", 404
        try:
            return render_template(
                "view_dataset.html",
                dataset_id=dataset,
                dataset_name=dataset.replace("_", " ").title(),
            )
        except Exception as e:
            return f"<h1>Failed to render visualization: {str(e)}</h1>", 500

    @app.get("/datasets/<dataset>/features/sample.json")
    def get_sample_non_geometry_attributes(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=False)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return json.dumps(sample_record, indent=2)
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.get("/datasets/<dataset>/features/sample.geojson")
    def get_sample_with_geometry(dataset):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            mbr_string = request.args.get("mbr", default=None)
            if not mbr_string:
                return {"error": "MBR query parameter is required"}, 400
            sample_record = feature_service.get_sample_record(dataset, mbr_string, include_geometry=True)
            if not sample_record:
                return {"error": "No matching record found"}, 404
            return json.dumps(sample_record, indent=2)
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {str(e)}"}, 500

    @app.post("/datasets/<dataset>/styles.json")
    def generate_styles(dataset):
        empty = json.dumps([])
        json_ct = {"Content-Type": "application/json"}

        try:
            body = request.get_json(silent=True) or {}
            requested_features = body.get("features")
            stats = _load_stats_for_dataset(dataset)
            styles = _generate_styles_from_stats(
                dataset=dataset,
                stats=stats,
                requested_features=requested_features,
            )
            return json.dumps(styles), 200, json_ct
        except Exception as e:
            logger.error("[Styles] Failed for %s: %s", dataset, e)
            return empty, 200, json_ct

    @app.post("/api/query-styles")
    def query_styles():
        json_ct = {"Content-Type": "application/json"}
        body = request.get_json(silent=True) or {}

        user_query = str(body.get("query", "")).strip()
        if not user_query:
            return json.dumps({"error": "Request body must include a non-empty 'query'"}), 400, json_ct

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            router = _get_catalog_router()
            candidates = router.search(user_query, k=k)
        except FileNotFoundError as e:
            return json.dumps({"error": str(e)}), 503, json_ct
        except Exception as e:
            logger.exception("[QueryStyles] Failed during catalogue retrieval")
            return json.dumps({"error": f"Catalogue retrieval failed: {e}"}), 500, json_ct

        if not candidates:
            return json.dumps({"error": "No indexed datasets available"}), 404, json_ct

        candidate_payload = _candidate_payload_for_llm(candidates)

        try:
            prompt_path = Path(__file__).parent / "llm" / "query_routing_prompt.md"
            prompt_template = prompt_path.read_text()
            prompt = prompt_template.replace(
                "{{USER_QUERY}}", user_query
            ).replace(
                "{{CANDIDATES_JSON}}", json.dumps(candidate_payload, indent=2)
            )

            from .llm.factory import LLMFactory
            provider = LLMFactory.get_default_provider()
            raw = provider.generate_response(prompt)
            routing = _extract_first_json_object(raw)
        except Exception as e:
            logger.exception("[QueryStyles] LLM dataset routing failed")
            return json.dumps({"error": f"LLM dataset routing failed: {e}"}), 500, json_ct

        candidate_by_name = {c.dataset: c for c in candidates}
        selected_dataset = routing.get("selected_dataset")
        if selected_dataset not in candidate_by_name:
            logger.warning(
                "[QueryStyles] LLM selected invalid dataset '%s'; falling back to top candidate",
                selected_dataset,
            )
            selected_dataset = candidates[0].dataset

        selected_attributes = routing.get("selected_attributes")
        if not isinstance(selected_attributes, list):
            selected_attributes = []

        try:
            stats = _load_stats_for_dataset(selected_dataset)
            available_attributes = {
                attr.get("name")
                for attr in (stats.get("attributes") or [])
                if isinstance(attr, dict)
            }
            filtered_attributes = [
                a for a in selected_attributes
                if isinstance(a, str) and a in available_attributes
            ]

            style_intent = str(routing.get("style_intent", "")).strip()
            reason = str(routing.get("reason", "")).strip()

            if filtered_attributes:
                instruction = (
                    f"User query: {user_query}\n"
                    f"Routing reason: {reason}\n"
                    f"Style intent: {style_intent}\n"
                    f"Generate styling suggestions for these specific attributes: "
                    + ", ".join(filtered_attributes)
                )
            else:
                instruction = (
                    f"User query: {user_query}\n"
                    f"Routing reason: {reason}\n"
                    f"Style intent: {style_intent}\n"
                    f"Analyze the dataset and suggest the best styling rules for map visualization."
                )

            styles = _generate_styles_from_stats(
                dataset=selected_dataset,
                stats=stats,
                requested_features=filtered_attributes or None,
                instruction_override=instruction,
            )
        except Exception as e:
            logger.exception("[QueryStyles] Final style generation failed")
            return json.dumps({"error": f"Final style generation failed: {e}"}), 500, json_ct

        response = {
            "query": user_query,
            "top_k": candidate_payload,
            "routing": {
                "selected_dataset": selected_dataset,
                "reason": routing.get("reason"),
                "selected_attributes": filtered_attributes,
                "style_intent": routing.get("style_intent"),
                "selected_dataset_score": candidate_by_name[selected_dataset].score,
            },
            "styles": styles,
        }
        return json.dumps(response, indent=2), 200, json_ct

    return app