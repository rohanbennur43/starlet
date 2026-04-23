import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("bucket_mvt")


class HistogramLoader:
    def __init__(self, hist_path):
        self.hist_path = Path(hist_path)
        self.prefix = None

    def load(self):
        logger.info("Loading histogram from %s", self.hist_path)
        arr = np.load(self.hist_path, allow_pickle=False)

        # If the provided file is already a prefix histogram, use it directly.
        if self.hist_path.stem.endswith("_prefix"):
            self.prefix = arr
        else:
            self.prefix = arr.cumsum(axis=0).cumsum(axis=1)

        return self.prefix