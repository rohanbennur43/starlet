"""Flask application factory for the Starlet tile server."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone
from threading import Lock, Thread
from uuid import uuid4
import json
import logging
import os
import re

from flask import Flask, Response, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS

from .catalog.embedder import GeminiTextEmbedder
from .catalog.index import CATALOG_FILENAME, build_catalog_index
from .catalog.pgvector_store import PgVectorConfig, PgVectorStore
from .catalog.router import CatalogRouter, SearchBackend
from .download_service import DatasetFeatureService
from .llm import continue_style_conversation, generate_map_code, start_style_conversation
from .tiler.tiler import VectorTiler

try:
    from ... import build as starlet_build
except Exception:  # pragma: no cover
    import starlet as starlet_api

    def starlet_build(*args, **kwargs):
        return starlet_api.build(*args, **kwargs)

logger = logging.getLogger(__name__)


def _normalize_unicode_text(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )


def _json_response(payload: Any, status: int = 200) -> Tuple[str, int, Dict[str, str]]:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False),
        status,
        {"Content-Type": "application/json; charset=utf-8"},
    )


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

    server_dir = Path(__file__).parent.resolve()
    template_dir = str(server_dir / "templates")

    app = Flask(__name__, template_folder=template_dir)
    app.config["JSON_AS_ASCII"] = False
    CORS(app, resources={r"/*": {"origins": "*"}})

    data_root = Path(data_dir).resolve()
    logger.info("Resolved data root: %s", data_root)
    print("DATA DIR =", data_root)

    tiler_cache: Dict[str, VectorTiler] = {}
    feature_service = DatasetFeatureService(data_root)

    _catalog_runtime: Dict[str, Any] = {
        "router": None,
        "mtime": None,
    }

    _build_jobs: Dict[str, Dict[str, Any]] = {}
    _build_jobs_lock = Lock()

    _tile_metrics: Dict[str, Dict[str, Any]] = {}
    _tile_metrics_lock = Lock()

    uploads_root = data_root / "_uploads"
    uploads_root.mkdir(parents=True, exist_ok=True)

    def _set_build_job(job_id: str, **updates: Any) -> None:
        with _build_jobs_lock:
            current = _build_jobs.get(job_id, {}).copy()
            current.update(updates)
            current["updated_at"] = datetime.now(timezone.utc).isoformat()
            _build_jobs[job_id] = current

    def _get_build_job(job_id: str) -> Optional[Dict[str, Any]]:
        with _build_jobs_lock:
            job = _build_jobs.get(job_id)
            return dict(job) if job else None

    def _set_tile_metric(dataset: str, metric: Dict[str, Any]) -> None:
        with _tile_metrics_lock:
            _tile_metrics[dataset] = dict(metric)

    def _get_tile_metric(dataset: str) -> Optional[Dict[str, Any]]:
        with _tile_metrics_lock:
            metric = _tile_metrics.get(dataset)
            return dict(metric) if metric else None

    def _safe_int(value: Any, fallback: int) -> int:
        try:
            return int(value)
        except Exception:
            return fallback

    def _safe_float(value: Any, fallback: float) -> float:
        try:
            return float(value)
        except Exception:
            return fallback

    def _slugify_dataset_name(name: str) -> str:
        cleaned = _normalize_unicode_text(name).strip()
        cleaned = re.sub(r"\.[A-Za-z0-9]+$", "", cleaned)
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", cleaned)
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or f"dataset_{uuid4().hex[:8]}"

    def _human_size(num_bytes: int) -> str:
        size = float(max(0, int(num_bytes)))
        units = ["B", "KB", "MB", "GB", "TB"]
        index = 0
        while size >= 1024.0 and index < len(units) - 1:
            size /= 1024.0
            index += 1
        if index == 0:
            return f"{int(size)} {units[index]}"
        return f"{size:.1f} {units[index]}"

    def _apply_pgvector_env_from_request(payload: Dict[str, Any]) -> bool:
        sync_pgvector = str(payload.get("sync_pgvector", "")).strip().lower() in {"1", "true", "yes", "on"}

        if not sync_pgvector:
            os.environ["CATALOG_PGVECTOR_ENABLED"] = "false"
            return False

        os.environ["CATALOG_PGVECTOR_ENABLED"] = "true"
        os.environ["PGVECTOR_HOST"] = _normalize_unicode_text(payload.get("pgvector_host", "localhost")).strip() or "localhost"
        os.environ["PGVECTOR_PORT"] = str(_safe_int(payload.get("pgvector_port", 5432), 5432))
        os.environ["PGVECTOR_DB"] = _normalize_unicode_text(payload.get("pgvector_db", "postgres")).strip() or "postgres"
        os.environ["PGVECTOR_USER"] = _normalize_unicode_text(payload.get("pgvector_user", "")).strip()
        os.environ["PGVECTOR_PASSWORD"] = _normalize_unicode_text(payload.get("pgvector_password", "")).strip()
        os.environ["PGVECTOR_TABLE"] = _normalize_unicode_text(payload.get("pgvector_table", "dataset_catalog_embeddings")).strip() or "dataset_catalog_embeddings"
        return True

    def _run_dataset_build_job(
        *,
        job_id: str,
        uploaded_file_path: Path,
        dataset_name: str,
        num_tiles: int,
        zoom: int,
        threshold: int,
        sync_pgvector: bool,
    ) -> None:
        try:
            _set_build_job(
                job_id,
                status="running",
                step="building_tiles",
                message="Starting Starlet build...",
            )

            outdir = data_root / dataset_name
            outdir.parent.mkdir(parents=True, exist_ok=True)

            tile_result, mvt_result = starlet_build(
                input=str(uploaded_file_path),
                outdir=str(outdir),
                num_tiles=num_tiles,
                zoom=zoom,
                threshold=threshold,
            )

            tiler_cache.pop(dataset_name, None)
            _catalog_runtime["router"] = None
            _catalog_runtime["mtime"] = None

            _set_build_job(
                job_id,
                step="building_index",
                message="Tiles generated. Rebuilding catalogue index...",
                tile_result={
                    "outdir": str(getattr(tile_result, "outdir", outdir)),
                    "num_files": getattr(tile_result, "num_files", None),
                    "total_rows": getattr(tile_result, "total_rows", None),
                    "bbox": getattr(tile_result, "bbox", None),
                },
                mvt_result={
                    "outdir": str(getattr(mvt_result, "outdir", outdir / "mvt")),
                    "zoom_levels": getattr(mvt_result, "zoom_levels", None),
                    "tile_count": getattr(mvt_result, "tile_count", None),
                },
            )

            catalog = build_catalog_index(
                data_root=data_root,
                out_dir=data_root / "_catalog",
                sync_pgvector=sync_pgvector,
            )

            _catalog_runtime["router"] = None
            _catalog_runtime["mtime"] = None

            _set_build_job(
                job_id,
                status="completed",
                step="done",
                message="Dataset uploaded, tiled, and indexed successfully.",
                dataset=dataset_name,
                output_dir=str(outdir),
                catalog_entry_count=catalog.get("entry_count", 0),
            )

        except Exception as e:
            logger.exception("[DatasetBuildJob] Failed for dataset=%s", dataset_name)
            _set_build_job(
                job_id,
                status="failed",
                step="error",
                message=str(e),
            )

    def get_tiler(dataset: str) -> VectorTiler:
        if dataset not in tiler_cache:
            tiler_cache[dataset] = VectorTiler(
                str(data_root / dataset),
                memory_cache_size=cache_size,
            )
        return tiler_cache[dataset]

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
                "Build it first with catalog/index.py."
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
                "Build it first with catalog/index.py."
            )

        mtime = index_path.stat().st_mtime
        if _catalog_runtime["router"] is None or _catalog_runtime["mtime"] != mtime:
            _catalog_runtime["router"] = _build_catalog_router()
            _catalog_runtime["mtime"] = mtime
        return _catalog_runtime["router"]

    def _load_catalog_index_json() -> Dict[str, Any]:
        index_path = _catalog_index_path()
        if not index_path.exists():
            return {"entries": [], "entry_count": 0}
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_catalog_entry_map() -> Dict[str, Dict[str, Any]]:
        catalog = _load_catalog_index_json()
        entries = catalog.get("entries") or []
        return {str(entry.get("dataset")): entry for entry in entries if isinstance(entry, dict)}

    def _load_stats_for_dataset(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        stats_path = dataset_path / "stats" / "attributes.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats not found for dataset: {dataset}")

        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _dataset_exists(dataset: str) -> bool:
        path = data_root / dataset
        return path.exists() and path.is_dir()

    def _summarize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        attributes = stats.get("attributes") or []
        geometry_types: List[str] = []
        geometry_attribute_count = 0
        non_geometry_attribute_count = 0
        sample_attributes: List[str] = []

        for attr in attributes:
            if not isinstance(attr, dict):
                continue

            attr_name = _normalize_unicode_text(attr.get("name"))
            stats_obj = attr.get("stats") or {}

            if "geom_types" in stats_obj:
                geometry_attribute_count += 1
                for geom_type in (stats_obj.get("geom_types") or {}).keys():
                    geom_type_lc = str(geom_type).lower()
                    if geom_type_lc not in geometry_types:
                        geometry_types.append(geom_type_lc)
            else:
                non_geometry_attribute_count += 1
                if attr_name:
                    sample_attributes.append(attr_name)

        return {
            "attribute_count": non_geometry_attribute_count,
            "geometry_attribute_count": geometry_attribute_count,
            "geometry_types": geometry_types,
            "sample_attributes": sample_attributes[:12],
        }

    def _dataset_metadata(dataset: str) -> Dict[str, Any]:
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset not found: {dataset}")

        size = sum(f.stat().st_size for f in dataset_path.rglob("*") if f.is_file())
        file_count = sum(1 for f in dataset_path.rglob("*") if f.is_file())

        return {
            "id": dataset,
            "name": dataset.replace("_", " ").title(),
            "size": size,
            "size_human": _human_size(size),
            "file_count": file_count,
            "path": str(dataset_path),
        }

    def _list_dataset_metadata(query: Optional[str] = None) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []
        if not data_root.exists():
            return datasets

        query_lc = _normalize_unicode_text(query).strip().lower()
        catalog_map = _load_catalog_entry_map()

        for d in sorted(data_root.iterdir()):
            if not d.is_dir():
                continue
            if d.name.startswith("."):
                continue
            if d.name in {"_catalog", "_uploads"}:
                continue

            size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            entry = catalog_map.get(d.name) or {}
            summary = entry.get("summary") or {}

            item = {
                "id": d.name,
                "name": d.name.replace("_", " ").title(),
                "size": size,
                "size_human": _human_size(size),
                "file_count": sum(1 for f in d.rglob("*") if f.is_file()),
                "attribute_count": summary.get("attribute_count"),
                "geometry_attribute_count": summary.get("geometry_attribute_count"),
                "geometry_types": [
                    geom_type
                    for geom in (summary.get("geometry") or [])
                    for geom_type in (geom.get("geom_types") or {}).keys()
                ],
            }

            if not query_lc or query_lc in item["id"].lower() or query_lc in item["name"].lower():
                datasets.append(item)

        return datasets

    def _infer_geometry_kind_from_summary(summary: Dict[str, Any]) -> str:
        geometry = summary.get("geometry") or []
        geom_types: List[str] = []

        for geom_attr in geometry:
            stats = geom_attr.get("geom_types") or {}
            geom_types.extend(str(k).lower() for k in stats.keys())

        joined = " ".join(geom_types)
        if any(x in joined for x in ("line", "multiline")):
            return "line"
        if any(x in joined for x in ("polygon", "multipolygon")):
            return "polygon"
        if any(x in joined for x in ("point", "multipoint")):
            return "point"

        dataset_name = str(summary.get("dataset", "")).lower()
        if "road" in dataset_name or "rail" in dataset_name:
            return "line"
        if any(x in dataset_name for x in ("county", "state", "tract")):
            return "polygon"
        if "point" in dataset_name:
            return "point"
        return "unknown"

    def _attribute_role_from_summary(attr: Dict[str, Any]) -> str:
        role = str(attr.get("role", "")).strip().lower()
        if role:
            return role
        if attr.get("min") is not None or attr.get("max") is not None:
            return "numeric"
        if attr.get("top_k"):
            return "categorical"
        return "unknown"

    def _find_attribute_summary(summary: Dict[str, Any], attr_name: str) -> Optional[Dict[str, Any]]:
        for attr in summary.get("attributes") or []:
            if str(attr.get("name")) == attr_name:
                return attr
        return None

    def _normalize_hex_color(color: str, fallback: str) -> str:
        color = str(color or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", color):
            return color
        return fallback

    def _categorical_palette() -> List[str]:
        return [
            "#1f78b4",
            "#33a02c",
            "#e31a1c",
            "#ff7f00",
            "#6a3d9a",
            "#b15928",
            "#a6cee3",
            "#b2df8a",
        ]

    def _gradient_palette(default_name: str, colors: List[str]) -> Dict[str, Any]:
        sane = [_normalize_hex_color(c, "#4682B4") for c in (colors or [])]
        sane = [c for c in sane if c]
        if not sane:
            sane = ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        return {
            "name": default_name,
            "colors": sane,
        }

    def _dataset_summary_for_llm(dataset: str) -> Dict[str, Any]:
        stats = _load_stats_for_dataset(dataset)

        selected_summary: Dict[str, Any] = {
            "dataset": dataset,
            "geometry": [],
            "attributes": [],
        }

        for attr in stats.get("attributes") or []:
            if not isinstance(attr, dict):
                continue

            stats_obj = attr.get("stats") or {}
            name = _normalize_unicode_text(attr.get("name"))
            entry: Dict[str, Any] = {"name": name}

            if "geom_types" in stats_obj:
                entry["geom_types"] = stats_obj.get("geom_types") or {}
                selected_summary["geometry"].append(entry)
                continue

            if any(k in stats_obj for k in ("min", "max", "mean", "stddev")):
                entry["role"] = "numeric"
                entry["min"] = stats_obj.get("min")
                entry["max"] = stats_obj.get("max")
                entry["top_k"] = stats_obj.get("top_k") or []
            elif stats_obj.get("top_k") is not None:
                entry["role"] = "categorical"
                entry["top_k"] = stats_obj.get("top_k") or []
            else:
                entry["role"] = "unknown"

            selected_summary["attributes"].append(entry)

        return selected_summary

    def _normalize_style_for_client(
        dataset: str,
        dataset_summary: Dict[str, Any],
        style: Dict[str, Any],
    ) -> Dict[str, Any]:
        geometry_kind = _infer_geometry_kind_from_summary(dataset_summary)
        style_type = str(style.get("style_type", "")).strip() or (
            "line-single-color"
            if geometry_kind == "line"
            else "fill-single-color"
            if geometry_kind == "polygon"
            else "circle-single-color"
            if geometry_kind == "point"
            else "line-single-color"
        )

        target_attribute = _normalize_unicode_text(style.get("target_attribute", "")).strip()
        attr_summary = _find_attribute_summary(dataset_summary, target_attribute) if target_attribute else None
        attr_role = _attribute_role_from_summary(attr_summary or {})
        is_categorical = "categorical" in style_type or attr_role in {"categorical", "categorical_text"}
        is_gradient = "gradient" in style_type

        color_theme = style.get("color_theme") or {}
        theme_name = _normalize_unicode_text(color_theme.get("name", "")).strip() or "custom"
        theme_colors = color_theme.get("colors") or []

        opacity = _safe_float(style.get("opacity", 1.0), 1.0)
        stroke_width = _safe_float(style.get("stroke_width", 2.0), 2.0)
        radius = _safe_float(style.get("radius", 4.0), 4.0)
        legend_title = _normalize_unicode_text(style.get("legend_title", "")).strip() or target_attribute or dataset
        notes = style.get("notes") or []
        if not isinstance(notes, list):
            notes = [str(notes)]

        if is_categorical:
            palette = [_normalize_hex_color(c, "#1f78b4") for c in theme_colors]
            palette = [c for c in palette if c]
            if not palette:
                palette = _categorical_palette()

            categorical_values: List[str] = []
            if attr_summary:
                for item in attr_summary.get("top_k") or []:
                    value = item.get("value") if isinstance(item, dict) else item
                    if value is None:
                        continue
                    categorical_values.append(_normalize_unicode_text(value))

            stops = [
                {"value": value, "color": palette[i % len(palette)]}
                for i, value in enumerate(categorical_values)
            ]

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": {
                    "name": theme_name,
                    "colors": palette,
                },
                "renderer": {
                    "mode": "categorical",
                    "attribute": target_attribute,
                    "fallback_color": palette[0],
                    "stops": stops,
                },
                "notes": [_normalize_unicode_text(n) for n in notes],
            }

        if is_gradient:
            palette = _gradient_palette(theme_name or "gradient", list(theme_colors))
            min_value = attr_summary.get("min") if attr_summary else None
            max_value = attr_summary.get("max") if attr_summary else None

            min_value = _safe_float(min_value, 0.0)
            max_value = _safe_float(max_value, 1.0)
            if max_value == min_value:
                max_value = min_value + 1.0

            return {
                "dataset": dataset,
                "geometry_kind": geometry_kind,
                "style_type": style_type,
                "target_attribute": target_attribute,
                "legend_title": legend_title,
                "opacity": opacity,
                "stroke_width": stroke_width,
                "radius": radius,
                "color_theme": palette,
                "renderer": {
                    "mode": "gradient",
                    "attribute": target_attribute,
                    "min": min_value,
                    "max": max_value,
                    "colors": palette["colors"],
                },
                "notes": [_normalize_unicode_text(n) for n in notes],
            }

        fallback_color = "#4682B4"
        if theme_colors:
            fallback_color = _normalize_hex_color(theme_colors[0], fallback_color)

        return {
            "dataset": dataset,
            "geometry_kind": geometry_kind,
            "style_type": style_type,
            "target_attribute": target_attribute,
            "legend_title": legend_title,
            "opacity": opacity,
            "stroke_width": stroke_width,
            "radius": radius,
            "color_theme": {
                "name": theme_name,
                "colors": [fallback_color],
            },
            "renderer": {
                "mode": "single",
                "attribute": target_attribute,
                "color": fallback_color,
            },
            "notes": [_normalize_unicode_text(n) for n in notes],
        }

    def _candidate_payload_for_llm(candidates) -> List[Dict[str, Any]]:
        payload = []
        for c in candidates:
            payload.append(
                {
                    "dataset": c.dataset,
                    "score": round(float(c.score), 6),
                    "summary": c.summary,
                }
            )
        return payload

    def _looks_like_new_dataset_request(
        user_query: str,
        current_dataset: Optional[str],
        current_style: Optional[Dict[str, Any]],
    ) -> bool:
        q = _normalize_unicode_text(user_query).strip().lower()
        if not q:
            return False

        if any(
            phrase in q
            for phrase in [
                "use a different dataset",
                "switch dataset",
                "another dataset",
                "different dataset",
                "new dataset",
            ]
        ):
            return True

        if current_dataset and current_dataset.lower() in q:
            return False

        domain_triggers = [
            "county", "counties", "state", "states", "tract", "tracts",
            "road", "roads", "rail", "rails", "building", "buildings",
            "point", "points", "landmark", "landmarks",
        ]
        if any(word in q for word in domain_triggers):
            target_attr = ""
            if current_style:
                target_attr = str(current_style.get("target_attribute", "")).lower()
            current_dataset_lc = (current_dataset or "").lower()
            if current_dataset_lc and current_dataset_lc not in q and target_attr and target_attr not in q:
                return True

        return False

    def _run_initial_chat_turn(user_query: str, k: int = 5) -> Tuple[Dict[str, Any], int]:
        total_t0 = perf_counter()
        normalized_query = _normalize_unicode_text(user_query)

        router = _get_catalog_router()

        search_t0 = perf_counter()
        candidates = router.search(normalized_query, k=k)
        semantic_search_ms = (perf_counter() - search_t0) * 1000.0

        if not candidates:
            raise LookupError("No indexed datasets available")

        candidate_payload = _candidate_payload_for_llm(candidates)

        llm_t0 = perf_counter()
        turn1 = start_style_conversation(
            dataset="__choose_from_candidates__",
            dataset_summary={
                "mode": "candidate_selection",
                "candidates": candidate_payload,
            },
            user_query=(
                f"{normalized_query}\n\n"
                "Choose the single best dataset from the provided candidates, "
                "choose the best attribute for styling, and generate the initial style."
            ),
            selected_attributes=None,
            style_intent=None,
            provider_name="gemini",
            temperature=0.2,
        )
        llm_ms = (perf_counter() - llm_t0) * 1000.0

        selected_dataset = _normalize_unicode_text(turn1.selected_dataset).strip()
        candidate_by_name = {c.dataset: c for c in candidates}
        if selected_dataset not in candidate_by_name:
            logger.warning(
                "[ChatStyle] LLM selected invalid dataset '%s'; falling back to top candidate '%s'",
                selected_dataset,
                candidates[0].dataset,
            )
            selected_dataset = candidates[0].dataset

        selected_candidate = candidate_by_name[selected_dataset]
        selected_summary = selected_candidate.summary
        normalized_style = _normalize_style_for_client(
            dataset=selected_dataset,
            dataset_summary=selected_summary,
            style=turn1.style,
        )

        tile_metric = _get_tile_metric(selected_dataset)
        total_ms = (perf_counter() - total_t0) * 1000.0

        response = {
            "mode": "initial",
            "query": normalized_query,
            "interaction_id": _normalize_unicode_text(turn1.interaction_id),
            "assistant_response": _normalize_unicode_text(turn1.assistant_response),
            "selected_dataset": selected_dataset,
            "selected_dataset_score": float(selected_candidate.score),
            "selected_attributes": [_normalize_unicode_text(x) for x in (turn1.selected_attributes or [])],
            "style_intent": _normalize_unicode_text(turn1.style_intent),
            "style": normalized_style,
            "top_k": candidate_payload,
            "timings": {
                "semantic_search_ms": round(semantic_search_ms, 3),
                "llm_ms": round(llm_ms, 3),
                "total_ms": round(total_ms, 3),
                "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
            },
        }
        return response, 200

    def _run_followup_chat_turn(
        *,
        user_query: str,
        interaction_id: str,
        current_dataset: str,
        current_attributes: Optional[List[str]] = None,
        current_style: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], int]:
        if not current_dataset:
            raise ValueError("current_dataset is required for follow-up turns")
        if not interaction_id:
            raise ValueError("interaction_id is required for follow-up turns")

        total_t0 = perf_counter()
        normalized_query = _normalize_unicode_text(user_query)
        selected_summary = _dataset_summary_for_llm(current_dataset)

        llm_t0 = perf_counter()
        turn = continue_style_conversation(
            dataset=current_dataset,
            user_query=normalized_query,
            previous_interaction_id=_normalize_unicode_text(interaction_id),
            selected_attributes_hint=[_normalize_unicode_text(x) for x in (current_attributes or [])],
            current_style_hint=current_style or {},
            provider_name="gemini",
            temperature=0.2,
        )
        llm_ms = (perf_counter() - llm_t0) * 1000.0

        returned_dataset = _normalize_unicode_text(turn.selected_dataset).strip() or current_dataset
        if returned_dataset != current_dataset:
            logger.info(
                "[ChatStyle] Follow-up requested dataset switch from '%s' to '%s'; restarting retrieval.",
                current_dataset,
                returned_dataset,
            )
            return _run_initial_chat_turn(user_query=normalized_query, k=5)

        normalized_style = _normalize_style_for_client(
            dataset=current_dataset,
            dataset_summary=selected_summary,
            style=turn.style,
        )

        tile_metric = _get_tile_metric(current_dataset)
        total_ms = (perf_counter() - total_t0) * 1000.0

        response = {
            "mode": "followup",
            "query": normalized_query,
            "interaction_id": _normalize_unicode_text(turn.interaction_id),
            "assistant_response": _normalize_unicode_text(turn.assistant_response),
            "selected_dataset": current_dataset,
            "selected_attributes": [_normalize_unicode_text(x) for x in (turn.selected_attributes or [])],
            "style_intent": _normalize_unicode_text(turn.style_intent),
            "style": normalized_style,
            "top_k": [],
            "timings": {
                "semantic_search_ms": 0.0,
                "llm_ms": round(llm_ms, 3),
                "total_ms": round(total_ms, 3),
                "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
            },
        }
        return response, 200

    @app.get("/<dataset>/<int:z>/<int:x>/<int:y>.mvt")
    def serve_tile(dataset, z, x, y):
        t0 = perf_counter()
        tiler = get_tiler(dataset)
        data = tiler.get_tile(z, x, y)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        metric = {
            "dataset": dataset,
            "z": z,
            "x": x,
            "y": y,
            "bytes": len(data),
            "elapsed_ms": elapsed_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _set_tile_metric(dataset, metric)

        logger.info(
            "[TileRequest] dataset=%s z=%d x=%d y=%d bytes=%d elapsed=%.1fms",
            dataset,
            z,
            x,
            y,
            len(data),
            elapsed_ms,
        )
        return Response(data, mimetype="application/vnd.mapbox-vector-tile")

    @app.get("/datasets/<path:filepath>")
    def serve_dataset_file(filepath):
        full_path = (data_root / filepath).resolve()

        try:
            full_path.relative_to(data_root)
        except ValueError:
            logger.warning("[DatasetFile] Blocked path traversal: %s", full_path)
            return "File not found", 404

        logger.info(
            "[DatasetFile] request=%s resolved=%s exists=%s",
            filepath,
            full_path,
            full_path.exists(),
        )

        if not full_path.exists() or not full_path.is_file():
            return "File not found", 404

        return send_from_directory(str(data_root), filepath)

    @app.get("/api/datasets")
    def list_datasets():
        datasets = sorted(
            [d.name for d in data_root.iterdir() if d.is_dir()]
        ) if data_root.exists() else []
        return app.response_class(
            response=json.dumps({"datasets": datasets}, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/api/catalog/entries")
    def api_catalog_entries():
        query = request.args.get("q", default=None)
        datasets = _list_dataset_metadata(query=query)
        return app.response_class(
            response=json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/datasets.json")
    def search_datasets():
        query = request.args.get("q", default=None)
        datasets = _list_dataset_metadata(query=query)
        return app.response_class(
            response=json.dumps({"datasets": datasets}, indent=2, ensure_ascii=False),
            mimetype="application/json",
        )

    @app.get("/datasets/<dataset>.json")
    def get_dataset_metadata(dataset):
        try:
            metadata = _dataset_metadata(dataset)
            return app.response_class(
                response=json.dumps(metadata, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except FileNotFoundError:
            return {"error": "Dataset not found"}, 404
        except Exception as e:
            return {"error": f"Failed to retrieve metadata: {e}"}, 500

    @app.get("/api/datasets/<dataset>/stats")
    def get_dataset_stats(dataset):
        stats_path = data_root / dataset / "stats" / "attributes.json"
        if not stats_path.exists():
            return {"error": "Stats not found for dataset"}, 404
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            return {"error": f"Failed to load stats: {e}"}, 500

    @app.get("/api/datasets/<dataset>/inspect")
    def inspect_dataset(dataset: str):
        try:
            metadata = _dataset_metadata(dataset)

            stats = None
            stats_summary = {}
            try:
                stats = _load_stats_for_dataset(dataset)
                stats_summary = _summarize_stats(stats)
            except Exception:
                stats = None
                stats_summary = {}

            catalog_entry = _load_catalog_entry_map().get(dataset) or {}
            catalog_summary = catalog_entry.get("summary") or {}

            recent_tile_request = _get_tile_metric(dataset)

            payload = {
                "dataset": dataset,
                "metadata": metadata,
                "catalog_summary": {
                    "dataset": catalog_summary.get("dataset"),
                    "description": catalog_summary.get("description"),
                    "attribute_count": catalog_summary.get("attribute_count"),
                    "geometry_attribute_count": catalog_summary.get("geometry_attribute_count"),
                    "geometry": catalog_summary.get("geometry") or [],
                    "attributes": catalog_summary.get("attributes") or [],
                },
                "stats_summary": stats_summary,
                "recent_tile_request": recent_tile_request,
                "stats_preview": stats,
            }
            return _json_response(payload, 200)
        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[InspectDataset] Failed for dataset=%s", dataset)
            return _json_response({"error": f"Failed to inspect dataset: {e}"}, 500)

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
            return f"<h1>Failed to render visualization: {e}</h1>", 500

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
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.post("/datasets/<dataset>/features.<format>")
    def download_features_with_geometry(dataset, format):
        dataset_path = data_root / dataset
        if not dataset_path.exists() or not dataset_path.is_dir():
            return {"error": "Dataset not found"}, 404
        try:
            geojson_payload = request.get_json(silent=True)
            mbr_string = request.args.get("mbr", default=None)
            if geojson_payload:
                geometry = geojson_payload.get("geometry")
                if not geometry:
                    return {"error": "Invalid GeoJSON payload: 'geometry' field is required"}, 400
                feature_stream = feature_service.get_features_stream(
                    dataset,
                    format,
                    geometry=geometry,
                )
            else:
                feature_stream = feature_service.get_features_stream(dataset, format, mbr_string)
            mime_type = feature_service.get_mime_type(format)
            filename = f"{dataset}_filtered.{format}" if geojson_payload else f"{dataset}_mbr.{format}"
            return Response(
                feature_stream,
                mimetype=mime_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

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
            return app.response_class(
                response=json.dumps(sample_record, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

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
            return app.response_class(
                response=json.dumps(sample_record, indent=2, ensure_ascii=False),
                mimetype="application/json",
            )
        except ValueError as e:
            return {"error": str(e)}, 400
        except FileNotFoundError as e:
            return {"error": str(e)}, 404
        except Exception as e:
            return {"error": f"Internal error: {e}"}, 500

    @app.get("/")
    def index():
        logger.info("Serving index page")
        return render_template("index.html")

    @app.get("/map.html")
    def map_page():
        logger.info("Serving map runtime page")
        return render_template("map.html")

    @app.route("/<path:filename>")
    def serve_file(filename):
        normalized = filename.strip("/")

        if normalized.startswith("datasets/") or normalized.startswith("api/"):
            return "File not found", 404

        file_path = (server_dir / normalized).resolve()

        try:
            file_path.relative_to(server_dir)
        except ValueError:
            return "File not found", 404

        if file_path.exists() and file_path.is_file():
            return send_from_directory(str(server_dir), normalized)

        return "File not found", 404

    @app.post("/api/chat-style")
    def chat_style():
        body = request.get_json(silent=True) or {}

        user_query = _normalize_unicode_text(body.get("query", "")).strip()
        if not user_query:
            return _json_response({"error": "Request body must include a non-empty 'query'"}, 400)

        interaction_id = _normalize_unicode_text(body.get("interaction_id", "")).strip()
        current_dataset = _normalize_unicode_text(body.get("current_dataset", "")).strip()

        current_attributes_raw = body.get("current_attributes")
        if isinstance(current_attributes_raw, list):
            current_attributes = [
                _normalize_unicode_text(x)
                for x in current_attributes_raw
                if isinstance(x, (str, int, float))
            ]
        else:
            current_attributes = []

        current_style = body.get("current_style")
        if not isinstance(current_style, dict):
            current_style = {}

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            if not interaction_id or not current_dataset:
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return _json_response(response, status)

            if _looks_like_new_dataset_request(
                user_query=user_query,
                current_dataset=current_dataset,
                current_style=current_style,
            ):
                response, status = _run_initial_chat_turn(user_query=user_query, k=k)
                return _json_response(response, status)

            if not _dataset_exists(current_dataset):
                return _json_response({"error": f"Current dataset not found: {current_dataset}"}, 404)

            response, status = _run_followup_chat_turn(
                user_query=user_query,
                interaction_id=interaction_id,
                current_dataset=current_dataset,
                current_attributes=current_attributes,
                current_style=current_style,
            )
            return _json_response(response, status)

        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 503)
        except LookupError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[ChatStyle] Failed for query=%r", user_query)
            return _json_response({"error": f"Chat styling failed: {e}"}, 500)

    @app.post("/api/upload-dataset")
    def upload_dataset_and_build():
        try:
            uploaded = request.files.get("file")
            if uploaded is None or not uploaded.filename:
                return _json_response({"error": "A dataset file is required under form field 'file'."}, 400)

            dataset_name_raw = _normalize_unicode_text(request.form.get("dataset_name", uploaded.filename)).strip()
            dataset_name = _slugify_dataset_name(dataset_name_raw)

            num_tiles = max(1, _safe_int(request.form.get("num_tiles", 40), 40))
            zoom = max(0, _safe_int(request.form.get("zoom", 7), 7))
            threshold = max(0, _safe_int(request.form.get("threshold", 0), 0))

            sync_pgvector = _apply_pgvector_env_from_request(request.form)

            dataset_upload_dir = uploads_root / dataset_name
            dataset_upload_dir.mkdir(parents=True, exist_ok=True)

            original_name = secure_filename(uploaded.filename) or f"{dataset_name}.data"
            uploaded_file_path = dataset_upload_dir / original_name
            uploaded.save(str(uploaded_file_path))

            job_id = uuid4().hex
            _set_build_job(
                job_id,
                status="queued",
                step="queued",
                message="Upload received. Waiting to start build...",
                dataset=dataset_name,
                uploaded_file=str(uploaded_file_path),
                num_tiles=num_tiles,
                zoom=zoom,
                threshold=threshold,
                sync_pgvector=sync_pgvector,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

            worker = Thread(
                target=_run_dataset_build_job,
                kwargs={
                    "job_id": job_id,
                    "uploaded_file_path": uploaded_file_path,
                    "dataset_name": dataset_name,
                    "num_tiles": num_tiles,
                    "zoom": zoom,
                    "threshold": threshold,
                    "sync_pgvector": sync_pgvector,
                },
                daemon=True,
            )
            worker.start()

            return _json_response(
                {
                    "ok": True,
                    "job_id": job_id,
                    "dataset": dataset_name,
                    "message": "Upload accepted. Build started.",
                },
                202,
            )

        except Exception as e:
            logger.exception("[UploadDataset] Failed")
            return _json_response({"error": f"Upload/build request failed: {e}"}, 500)

    @app.get("/api/upload-dataset/<job_id>")
    def get_upload_dataset_job(job_id: str):
        job = _get_build_job(job_id)
        if not job:
            return _json_response({"error": f"Job not found: {job_id}"}, 404)
        return _json_response(job, 200)

    @app.post("/api/query-styles")
    def query_styles():
        body = request.get_json(silent=True) or {}
        with app.test_request_context(
            "/api/chat-style",
            method="POST",
            json=body,
        ):
            return chat_style()

    @app.post("/api/generate-map-code")
    def generate_map_code_route():
        body = request.get_json(silent=True) or {}

        user_query = _normalize_unicode_text(body.get("query", "")).strip()
        if not user_query:
            return _json_response({"error": "Request body must include a non-empty 'query'"}, 400)

        interaction_id = _normalize_unicode_text(body.get("interaction_id", "")).strip()
        current_dataset = _normalize_unicode_text(body.get("current_dataset", "")).strip()

        current_attributes_raw = body.get("current_attributes")
        if isinstance(current_attributes_raw, list):
            current_attributes = [
                _normalize_unicode_text(x)
                for x in current_attributes_raw
                if isinstance(x, (str, int, float))
            ]
        else:
            current_attributes = []

        current_style = body.get("current_style")
        if not isinstance(current_style, dict):
            current_style = {}

        requested_k = body.get("k", 5)
        try:
            k = max(1, min(int(requested_k), 10))
        except Exception:
            k = 5

        try:
            total_t0 = perf_counter()
            top_k_payload = []
            semantic_search_ms = 0.0

            if not current_dataset:
                initial_t0 = perf_counter()
                initial_response, _ = _run_initial_chat_turn(user_query=user_query, k=k)
                bootstrap_ms = (perf_counter() - initial_t0) * 1000.0

                current_dataset = _normalize_unicode_text(initial_response.get("selected_dataset", "")).strip()
                interaction_id = _normalize_unicode_text(initial_response.get("interaction_id", "")).strip()
                current_style = initial_response.get("style") or {}
                current_attributes = initial_response.get("selected_attributes") or []
                top_k_payload = initial_response.get("top_k") or []
                semantic_search_ms = float((initial_response.get("timings") or {}).get("semantic_search_ms") or 0.0)

                if not current_dataset:
                    return _json_response({"error": "Could not determine a dataset for map-code generation."}, 500)

                if not _dataset_exists(current_dataset):
                    return _json_response({"error": f"Dataset not found: {current_dataset}"}, 404)

                dataset_summary = _dataset_summary_for_llm(current_dataset)

                if not current_style:
                    current_style = _normalize_style_for_client(
                        dataset=current_dataset,
                        dataset_summary=dataset_summary,
                        style={},
                    )

                llm_t0 = perf_counter()
                code_turn = generate_map_code(
                    dataset=current_dataset,
                    dataset_summary=dataset_summary,
                    user_query=user_query,
                    current_style=current_style,
                    previous_interaction_id=None,
                    provider_name="gemini",
                    temperature=0.2,
                )
                llm_ms = (perf_counter() - llm_t0) * 1000.0
                total_ms = (perf_counter() - total_t0) * 1000.0
                tile_metric = _get_tile_metric(current_dataset)

                response = {
                    "mode": "generated_code",
                    "query": user_query,
                    "interaction_id": _normalize_unicode_text(code_turn.interaction_id or interaction_id),
                    "assistant_response": _normalize_unicode_text(code_turn.assistant_response),
                    "selected_dataset": current_dataset,
                    "selected_attributes": [_normalize_unicode_text(x) for x in (current_attributes or [])],
                    "style": current_style,
                    "generated_code": _normalize_unicode_text(code_turn.code),
                    "top_k": top_k_payload,
                    "timings": {
                        "semantic_search_ms": round(semantic_search_ms, 3),
                        "llm_ms": round(llm_ms, 3),
                        "total_ms": round(total_ms, 3),
                        "bootstrap_request_ms": round(bootstrap_ms, 3),
                        "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
                    },
                }
                return _json_response(response, 200)

            if not _dataset_exists(current_dataset):
                return _json_response({"error": f"Dataset not found: {current_dataset}"}, 404)

            dataset_summary = _dataset_summary_for_llm(current_dataset)

            if not current_style:
                current_style = _normalize_style_for_client(
                    dataset=current_dataset,
                    dataset_summary=dataset_summary,
                    style={},
                )

            llm_t0 = perf_counter()
            code_turn = generate_map_code(
                dataset=current_dataset,
                dataset_summary=dataset_summary,
                user_query=user_query,
                current_style=current_style,
                previous_interaction_id=None,
                provider_name="gemini",
                temperature=0.2,
            )
            llm_ms = (perf_counter() - llm_t0) * 1000.0
            total_ms = (perf_counter() - total_t0) * 1000.0
            tile_metric = _get_tile_metric(current_dataset)

            response = {
                "mode": "generated_code",
                "query": user_query,
                "interaction_id": _normalize_unicode_text(code_turn.interaction_id or interaction_id),
                "assistant_response": _normalize_unicode_text(code_turn.assistant_response),
                "selected_dataset": current_dataset,
                "selected_attributes": [_normalize_unicode_text(x) for x in (current_attributes or [])],
                "style": current_style,
                "generated_code": _normalize_unicode_text(code_turn.code),
                "top_k": top_k_payload,
                "timings": {
                    "semantic_search_ms": round(semantic_search_ms, 3),
                    "llm_ms": round(llm_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "last_tile_request_ms": round(float(tile_metric.get("elapsed_ms")), 3) if tile_metric and tile_metric.get("elapsed_ms") is not None else None,
                },
            }
            return _json_response(response, 200)

        except FileNotFoundError as e:
            return _json_response({"error": str(e)}, 503)
        except LookupError as e:
            return _json_response({"error": str(e)}, 404)
        except Exception as e:
            logger.exception("[GenerateMapCode] Failed for query=%r", user_query)
            return _json_response({"error": f"Map code generation failed: {e}"}, 500)

    return app
