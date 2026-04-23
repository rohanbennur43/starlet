"""Streaming MVT generation pipeline: histogram → assign → stream → render."""
import logging
from starlet._internal.histogram.loader import HistogramLoader
from .streamer import GeometryStreamer
from .assigner import TileAssigner
from .renderer import TileRenderer

logger = logging.getLogger(__name__)


class BucketMVTGenerator:
    """Streaming MVT generation pipeline.

    Orchestrates four stages:
      1. **HistogramLoader** — loads the 2D prefix-sum histogram from ``.npy``
      2. **TileAssigner** — determines which z/x/y tiles are nonempty and
         assigns each geometry to overlapping tiles (with reservoir sampling
         to cap features per tile)
      3. **GeometryStreamer** — decodes WKB geometries from GeoParquet row
         groups and reprojects EPSG:4326 → EPSG:3857
      4. **TileRenderer** — clips, simplifies, transforms to tile coords,
         and encodes each tile as a ``.mvt`` Protobuf file
    """

    def __init__(self, parquet_dir: str, hist_path: str, outdir: str, last_zoom: int, threshold: float) -> None:
        logger.info(f"Initializing BucketMVTGenerator: parquet_dir={parquet_dir}, outdir={outdir}, last_zoom={last_zoom}, threshold={threshold}")
        self.parquet_dir = parquet_dir
        self.hist_path = hist_path
        self.outdir = outdir
        self.last_zoom = last_zoom
        self.threshold = threshold

    def run(self) -> None:
        logger.info("Starting MVT generation pipeline")
        prefix = HistogramLoader(self.hist_path).load()
        zooms = list(range(0, self.last_zoom + 1))
        logger.debug("Zooms to process: %s", zooms)

        assigner = TileAssigner(zooms, prefix, self.threshold)
        logger.info("Using lazy histogram checks during assignment; no global nonempty-tile prepass")

        logger.info("Streaming geometries from %s", self.parquet_dir)
        geom_count = 0
        streamer = GeometryStreamer(self.parquet_dir)
        for geom, attrs in streamer.iter_geometries():
            assigner.assign_geometry(geom, attrs)
            geom_count += 1
            if geom_count % 10000 == 0:
                logger.debug("Processed %d geometries", geom_count)

        logger.info("Assigned %d geometries to tiles", geom_count)

        logger.info("Rendering tiles to %s", self.outdir)
        TileRenderer(self.outdir).render(assigner.buckets)
        logger.info("Pipeline execution complete")