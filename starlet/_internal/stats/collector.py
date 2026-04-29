import pyarrow as pa
from collections import OrderedDict
from .sketches import (
    NumericSketch,
    CategoricalSketch,
    TextSketch,
    GeometrySketch,
)


class AttributeStatsCollector:
    def __init__(self, schema: pa.Schema, geometry_column="geometry", global_mbr=None):
        """
        Initialize AttributeStatsCollector with optional pre-computed global MBR.

        Args:
            schema: PyArrow schema
            geometry_column: Name of geometry column
            global_mbr: Optional tuple of (minx, miny, maxx, maxy) to avoid redundant MBR computation
        """
        self.schema = schema
        self.geometry_column = geometry_column
        self.sketches = OrderedDict()

        for field in schema:
            name = field.name
            if name == geometry_column:
                self.sketches[name] = GeometrySketch(global_mbr=global_mbr)
                continue

            t = field.type
            if pa.types.is_integer(t) or pa.types.is_floating(t):
                self.sketches[name] = NumericSketch()
            elif pa.types.is_boolean(t):
                self.sketches[name] = CategoricalSketch()
            elif pa.types.is_timestamp(t) or pa.types.is_date(t):
                self.sketches[name] = NumericSketch()
            elif pa.types.is_string(t) or pa.types.is_large_string(t):
                self.sketches[name] = TextSketch()
            else:
                self.sketches[name] = CategoricalSketch()

    def consume_table(self, table: pa.Table):
        for col_name, sketch in self.sketches.items():
            if col_name not in table.column_names:
                continue

            col = table[col_name]

            if isinstance(sketch, GeometrySketch):
                # geometry is already decoded upstream in orchestrator
                geoms = col.to_pylist()
                sketch.update(geoms)
                continue

            # fast path for primitive columns
            arr = col.combine_chunks()
            values = arr.to_pylist()
            sketch.update(values)

    def finalize(self):
        out = []

        for name, sketch in self.sketches.items():
            entry = {
                "name": name,
                "stats": sketch.finalize(),
            }
            out.append(entry)

        return {"attributes": out}
