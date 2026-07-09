"""Central configuration for the spatial + phenology screening pipeline.

Holds thresholds, Earth Engine / Sentinel-2 parameters, output paths, and the
`EE_PROJECT` id loaded from the repo-root `.env` (via python-dotenv).
"""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Inputs / outputs
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# pre-computed site metadata (name/lat/lon/year + phenology scores)
# Step 1 reads lat/lon here and Step 2 reuses the cached NDVI vs GCC scores
# Override the input with PIPELINE_INPUT=<path> (e.g. to run only the GBOV subset)
_input_env = os.environ.get("PIPELINE_INPUT")
SITE_METADATA_JSON = Path(_input_env) if _input_env else INPUT_DIR / "site_metadata_clean.json"
YEAR_CHECK_JSON = INPUT_DIR / "year_check.json"  # kept for reference only

# PIPELINE_TAG=<name> suffixes the output files so a subset run (e.g. GBOV) does
# not overwrite the full run's step1/step2/results JSONs.
_tag_env = os.environ.get("PIPELINE_TAG")
_suffix = f"_{_tag_env}" if _tag_env else ""
STEP1_JSON = OUTPUT_DIR / f"step1_uniformity{_suffix}.json"
STEP2_JSON = OUTPUT_DIR / f"step2_phenology{_suffix}.json"
RESULTS_JSON = OUTPUT_DIR / f"pipeline_results{_suffix}.json"

# Step 1 pass/fail thresholds (spatial uniformity)
CV_MAX = 0.1      
WATER_MAX = 0.05  
URBAN_MAX = 0.05  

# Earth Engine / Sentinel-2 parameters
CLOUD_PCT = 5             # keep scenes with CLOUDY_PIXEL_PERCENTAGE below this
BOX_RADIUS_M = 2000       # buffer radius -> ~4 km box via .bounds()
NDVI_SCALE_M = 10         # reduceRegion scale (S2 10 m bands)
SUMMER_HALF_WINDOW_DAYS = 30  # half width of the peak summer search window
MAX_IMAGE_SEARCH_WIDENINGS = 1  # extra times to widen the window if no scenes found

# Fallback peak-summer DOY when phenology-derived MOS/DOS is unavailable
NORTH_SUMMER_DOY = 196  # ~mid-July (northern hemisphere)
SOUTH_SUMMER_DOY = 15   # ~mid-January (southern hemisphere)

# Threading
# Each site now makes a single bundled getInfo, so we can run more in parallel.
# Earth Engine allows a few dozen concurrent requests; lower this if you hit
# "Too many concurrent aggregations" / rate-limit errors.
STEP1_WORKERS = 24   # concurrent Earth Engine requests
MAX_WORKERS = 20     

# Optional subset for test runs. Set the PIPELINE_LIMIT env var to an int to only
# process the first N site-years end-to-end (leave unset to run everything).
_limit_env = os.environ.get("PIPELINE_LIMIT")
LIMIT = int(_limit_env) if _limit_env else None

# Earth Engine Cloud project id, read from repo-root .env: EE_PROJECT="your-id"
load_dotenv(find_dotenv())
EE_PROJECT = os.environ.get("EE_PROJECT")
