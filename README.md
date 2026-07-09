# NOAA Phenology Validation

Compare PhenoCam **GCC_90** and **NDVI_90** time series and mark seasonal phases (SOS, MOS, DOS, EOS).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
source .venv/bin/activate
```

## Project layout

- `test/` - loads CSVs from `test/data/ex--/`, runs phase detection and plots
  (`plotting.py`). Outputs go to `test/output/`.
- `src/` - API pipeline. `api/phenocam_api.py` fetches PhenoCam data
  `api/api_single.py` (single site) and `api/api_batch.py` (first N ROIs)
- `pipeline/` - self-contained spatial + phenology screening pipeline
  (Earth Engine uniformity, then NDVI-vs-GCC phenology agreement). Reads the
  pre-computed `site_metadata_clean.json`, so it makes no live PhenoCam calls.

## Run plots part 1 (local files)
```bash
cd test
python3 test.py
```

The script rn loops over every `test/data/ex--/` folder that contains one `*_ndvi_3day.csv` and writes:

- `test/output/ex1.png`
- `test/output/ex2.png`
- ....

## Run plots part 2 (live from the PhenoCam API)
```bash
cd src/api
python3 api_single.py   # single site -> src/output/api/API_data_<site>.png
python3 api_batch.py    # first N ROIs -> src/output/api/API_data_<roi>.png
```

## Run the screening pipeline (`pipeline/`)

A two step screen over every site in `pipeline/input/site_metadata_clean.json` (obtain
via 'src/' year_check.json):
1. **Step 1 (Earth Engine):** keep sites whose peak summer NDVI is spatially uniform
   (NDVI CV < 0.1, water < 5%, urban < 5%) inside a ~4 km box.
2. **Step 2:** attach the pre computed NDVI vs GCC agreement scores (phenophase gap,
   DTW/step, divergence) for the Step 1 survivors

Outputs land in `pipeline/output/` (`step1_uniformity.json`, `step2_phenology.json`,
and `pipeline_results.json` - survivors ranked by divergence)

```bash
cd pipeline
python3 run_pipeline.py     # run ALL site years (default)
```

### `PIPELINE_LIMIT` (optional test cap)

`PIPELINE_LIMIT` is an optional environment variable that caps how many site-years
are processed - useful for a quick test without waiting on the full run. It is
**not set by default**, so a plain `python3 run_pipeline.py` already processes every
site in `site_metadata_clean.json`.

```bash
PIPELINE_LIMIT=8 python3 run_pipeline.py   # only the first 8 site years
python3 run_pipeline.py                     # no limit -> all site years
```

Notes:
- Requires a one-time Earth Engine `localhost` auth and `EE_PROJECT` set in `.env`.
- A full run makes several Earth Engine calls per site, so it is network-bound and
  can take a while; tune `STEP1_WORKERS` and thresholds in `pipeline/config.py`.

## IMPORTANT NOTES: TESTING

Set the preferred year at the top of `test/test.py` (`YEAR = 2024`).

Each CSV may cover different years. The script uses `YEAR` when that year exists in the file; otherwise it uses the **latest year available** and prints a note. For example:

- `ex1` --> 2024 
- `ex2` --> for ex, fall back to 2022 bc there is no 2024 data

*Only the NDVI summary CSV is required (it includes `gcc_90`) from the phenocam website 

*Add a new example by creating `data/ex3/` with one `*_ndvi_3day.csv` inside.

*Other potential file is gcc 3day transition day files which can be download from gcc data tab in phenocam and it's in data_record_5 folder but don't need that right now.

## Data source

[PhenoCam Network](https://phenocam.nau.edu/webcam/)
