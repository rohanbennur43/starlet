from typing import Iterable, Optional, List, Dict, Any, Tuple
import logging
import json
from pathlib import Path
from decimal import Decimal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class DataSource:
    def schema(self) -> pa.Schema:
        raise NotImplementedError

    def iter_tables(self) -> Iterable[pa.Table]:
        raise NotImplementedError


# ------------------------- GeoParquet source ------------------------- #
class GeoParquetSource(DataSource):
    def __init__(self, path: str):
        self._pf = pq.ParquetFile(path)
        self._schema = self._pf.schema_arrow
        self._num_row_groups = self._pf.num_row_groups
        logger.info("GeoParquetSource opened %s with %d row groups", path, self._num_row_groups)

    def schema(self) -> pa.Schema:
        logger.debug("GeoParquet source schema metadata: %s", self._schema.metadata)
        return self._schema

    def iter_tables(self) -> Iterable[pa.Table]:
        for i in range(self._num_row_groups):
            logger.debug("Reading row group %d/%d", i, self._num_row_groups)
            yield self._pf.read_row_group(i)


# ------------------------- Helpers ------------------------- #
def is_geojson_path(path: str) -> bool:
    p = path.lower()
    return p.endswith((".geojson", ".geojsonl", ".json", ".jsonl"))


def _attach_geoparquet_metadata(schema: pa.Schema, crs_hint: Optional[str]) -> pa.Schema:
    """
    Return a copy of `schema` with a minimal GeoParquet 'geo' JSON block so
    downstream writers (WriterPool) can inject tile bbox.

    Includes:
      - version: 1.1.0
      - primary_column: geometry
      - columns.geometry.encoding: WKB
      - columns.geometry.crs: <crs_hint> (string hint if provided)
    """
    md = dict(schema.metadata or {})
    if b"geo" in md:
        return pa.schema(schema, metadata=md)

    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {"geometry": {"encoding": "WKB"}},
    }
    if crs_hint:
        try:
            geo["columns"]["geometry"]["crs"] = crs_hint
        except Exception:
            pass

    md[b"geo"] = json.dumps(geo, separators=(",", ":")).encode("utf-8")
    return pa.schema(schema, metadata=md)


def _normalize_decimal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns containing Decimal values into float64 so Arrow does
    not infer different decimal128 precision/scale across batches.
    """
    if df.empty:
        return df

    df = df.copy()

    for col in df.columns:
        s = df[col]

        # Only object dtype can hold Decimal values in pandas here.
        if s.dtype != "object":
            continue

        sample = None
        for v in s:
            if v is not None:
                sample = v
                break

        if isinstance(sample, Decimal):
            df[col] = s.map(lambda x: float(x) if x is not None else None)

    return df


# ------------------------- GeoJSON source (streaming → Arrow) ------------------------- #
class GeoJSONSource(DataSource):
    """
    Streams GeoJSON / GeoJSONL as Arrow Tables, converting geometry to WKB.

    - For standard FeatureCollection GeoJSON, streams features with `ijson`.
    - For GeoJSON Lines (one Feature per line), reads and batches by line.
    - Geometry dicts → shapely.shape → WKB bytes (binary Arrow column 'geometry').
    - Attaches minimal GeoParquet metadata (version, primary_column, encoding, crs hint).
    """

    def __init__(
        self,
        path: str,
        batch_rows: int = 1_000,
        src_crs: str = "EPSG:4326",
        target_crs: Optional[str] = None,
        keep_null_geoms: bool = False,
    ):
        self.path = path
        self.batch_rows = int(batch_rows)
        self.src_crs = src_crs
        self.target_crs = target_crs  # informational only here
        self.keep_null_geoms = keep_null_geoms

        if target_crs:
            logger.warning(
                "target_crs requested (%s) but GeoJSON reader does not reproject on the fly; data will be read as-is.",
                target_crs,
            )

        self._schema: Optional[pa.Schema] = None
        self._crs_hint: Optional[str] = None  # hint if detected
        self._use_geojsonl: bool = False  # detection result for reader type

        self._use_geojsonl, header_crs = _detect_geojson_mode_and_crs(self.path)
        if header_crs:
            self._crs_hint = header_crs

        logger.info(
            "GeoJSONSource opened %s (batch_rows=%d, src_crs=%s)",
            path, self.batch_rows, self.src_crs
        )

    # ---------------- schema ---------------- #
    def schema(self) -> pa.Schema:
        if self._schema is None:
            first = self._read_first_batch()
            if first is None or first.num_rows == 0:
                # Empty input file. Create a minimal schema with geometry column.
                base = pa.schema([("geometry", pa.binary())])
                self._schema = _attach_geoparquet_metadata(
                    base, self._crs_hint or self.target_crs or self.src_crs
                )
            else:
                # Lock schema with GeoParquet metadata
                self._schema = _attach_geoparquet_metadata(
                    first.schema, self._crs_hint or self.target_crs or self.src_crs
                )

        return self._schema

    # ---------------- iterator ---------------- #
    def iter_tables(self) -> Iterable[pa.Table]:
        batch_index = 0
        crs_value = self._crs_hint or self.target_crs or self.src_crs

        import geopandas as gpd

        for features in _iter_geojson_feature_batches(self.path, self.batch_rows, self._use_geojsonl):
            if not features:
                continue

            gdf = gpd.GeoDataFrame.from_features(features, crs=crs_value)
            geometry_col = pa.array(gdf.geometry.to_wkb(), type=pa.binary())

            props_df = gdf.drop(columns="geometry")
            props_df = _normalize_decimal_columns(props_df)
            props_table = pa.Table.from_pandas(props_df, preserve_index=False)

            table = (
                pa.table([geometry_col], names=["geometry"])
                if props_table.num_columns == 0
                else props_table.append_column("geometry", geometry_col)
            )

            # Attach GeoParquet metadata with CRS
            schema_with_geo = _attach_geoparquet_metadata(table.schema, crs_value)
            table = table.replace_schema_metadata(schema_with_geo.metadata)

            if not self._schema:
                self._schema = schema_with_geo

            table = self._coerce_to_schema(table, self.schema()).combine_chunks()

            logger.info(
                "GeoJSON batch %d (%d rows) -> %d columns (including 'geometry')",
                batch_index,
                table.num_rows,
                len(table.column_names),
            )
            batch_index += 1
            yield table

    # ---------------- internal helpers ---------------- #
    def _read_first_batch(self) -> Optional[pa.Table]:
        """Read the first batch of features to establish the schema."""
        feature_iter = _iter_geojson_feature_batches(
            self.path,
            max(1, self.batch_rows),
            self._use_geojsonl
        )
        try:
            features = next(feature_iter)
        except StopIteration:
            logger.info("GeoJSON read returned 0 rows when inferring schema")
            return None

        rows_props: List[Dict[str, Any]] = []
        geometries: List[Any] = []

        for feat in features:
            rows_props.append(feat.get("properties") or {})
            geometries.append(feat.get("geometry", None))

        props_df = pd.DataFrame.from_records(rows_props)
        props_df = _normalize_decimal_columns(props_df)
        props_table = pa.Table.from_pandas(props_df, preserve_index=False)

        wkb_list = _geometries_to_wkb(geometries)
        geometry_col = pa.array(wkb_list, type=pa.binary())

        if props_table.num_columns == 0:
            return pa.table([geometry_col], names=["geometry"])

        return props_table.append_column("geometry", geometry_col)

    def _coerce_to_schema(self, t: pa.Table, schema: pa.Schema) -> pa.Table:
        if t.schema.equals(schema):
            return t

        out_cols = []
        for fld in schema:
            name = fld.name
            if name in t.column_names:
                col = t[name]
                if not col.type.equals(fld.type):
                    try:
                        col = col.cast(fld.type)
                    except Exception:
                        logger.warning(
                            "Type mismatch for column '%s': %s -> %s (kept original)",
                            name, col.type, fld.type
                        )
                out_cols.append(col)
            else:
                out_cols.append(pa.nulls(t.num_rows, type=fld.type))

        return pa.table(out_cols, names=[f.name for f in schema])


def _geometries_to_wkb(geometries: List[Any]) -> List[Any]:
    """
    Vectorized geometry -> WKB conversion using shapely's GeoJSON reader.

    Converting via shapely.geometry.shape per-feature is expensive for large
    files. Using shapely.from_geojson on an array of compact JSON strings keeps
    the heavy work inside GEOS and removes most Python-level loops.
    """
    from shapely import from_geojson, to_wkb

    wkb: List[Any] = [None] * len(geometries)
    non_null_idx: List[int] = []
    geojson_strings: List[str] = []

    for idx, geom in enumerate(geometries):
        if geom is None:
            continue
        non_null_idx.append(idx)
        geojson_strings.append(json.dumps(geom, separators=(",", ":")))

    if not geojson_strings:
        return wkb

    shapely_geoms = from_geojson(geojson_strings)
    encoded = to_wkb(shapely_geoms, hex=False).tolist()

    for idx, val in zip(non_null_idx, encoded):
        wkb[idx] = val

    return wkb


def _iter_geojson_feature_batches_with_ijson(path: str, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    Stream a GeoJSON FeatureCollection in batches using ijson to avoid loading
    the entire file in memory.
    """
    try:
        import ijson
    except ImportError as e:
        raise ImportError("GeoJSON streaming requires 'ijson'. Install via: pip install ijson") from e

    batch: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fin:
        for feature in ijson.items(fin, "features.item"):
            batch.append(feature)
            if len(batch) >= batch_size:
                yield batch
                batch = []

    if batch:
        yield batch


def _iter_geojsonl_feature_batches(path: str, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    """
    Stream a GeoJSON Lines file (one Feature per line) in batches.
    """
    batch: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            feature = json.loads(line)
            batch.append(feature)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _iter_geojson_feature_batches(path: str, batch_size: int, use_geojsonl: bool) -> Iterable[List[Dict[str, Any]]]:
    """
    Dispatch to FeatureCollection or GeoJSON Lines batch readers based on detection result.
    """
    if use_geojsonl:
        logger.info("Detected GeoJSON Lines file for %s", path)
        yield from _iter_geojsonl_feature_batches(path, batch_size)
    else:
        logger.info("Detected FeatureCollection GeoJSON file for %s", path)
        yield from _iter_geojson_feature_batches_with_ijson(path, batch_size)


def _detect_geojson_mode_and_crs(path: str, sniff_bytes: int = 64 * 1024) -> Tuple[bool, Optional[str]]:
    """
    Inspect the beginning of a GeoJSON file to decide whether it is GeoJSONL and
    to extract a CRS hint from the header of a FeatureCollection (if present).
    """
    path = str(Path(path))
    try:
        with open(path, "r", encoding="utf-8") as fin:
            buffer = fin.read(sniff_bytes)
    except OSError:
        return False, None

    first_nonempty: Optional[str] = None
    for line in buffer.splitlines():
        stripped = line.strip()
        if stripped:
            first_nonempty = stripped
            break

    use_geojsonl = False
    if first_nonempty:
        try:
            obj = json.loads(first_nonempty)
            if isinstance(obj, dict) and obj.get("type") == "Feature":
                use_geojsonl = True
        except json.JSONDecodeError:
            pass

    crs_hint: Optional[str] = None
    if not use_geojsonl:
        crs_hint = _extract_feature_collection_crs_hint(buffer)

    return use_geojsonl, crs_hint


def _extract_feature_collection_crs_hint(buffer: str) -> Optional[str]:
    """
    Try to read the CRS from the header of a FeatureCollection without loading the whole file.
    Looks for a 'crs' object and returns its 'properties.name' if present.
    """
    if not buffer:
        return None

    idx = buffer.lower().find('"features"')
    if idx == -1:
        return None

    header = buffer[:idx]
    first_brace = header.find("{")
    if first_brace == -1:
        return None

    candidate = header[first_brace:]
    candidate = candidate.rstrip(", \r\n\t")
    candidate = candidate + "}"

    try:
        parsed = json.loads(candidate)
    except Exception:
        return None

    crs = parsed.get("crs")
    if isinstance(crs, dict):
        props = crs.get("properties") or {}
        name = props.get("name")
        if isinstance(name, str):
            return name

    return None