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

The entry scripts add the repo root to `sys.path` themselves, so the `shared`
package imports with no extra step --> just run the commands below.

## Project layout

Shared code lives in one place; the two stages import from it instead of keeping
their own copies.

- `shared/` - shared code (single source of truth), imported via the repo-root
  `sys.path` bootstrap in each entry script (no install step):
  - `PhenoloDates.py` - CCRmax phenophase date detection (SOS/MOS/DOS/EOS)
  - `phenocam_api.py` - PhenoCam metadata + 3 day summary fetch/parse
  - `DynamicTimeWrap.py` - NDVI vs GCC agreement scoring (gap + DTW -> divergence)
  - `plotting.py` - matplotlib time series figures with phase markers
- `prep/api/` - **Stage 1: data prep.** `api_year_check.py` finds sites with
  2023/2024 data (`year_check.json`); `main.py` scores each site-year and writes
  the metadata JSONs. Outputs go to `prep/output/api/`.
- `uniformity_pipeline/` - **Stage 2: uniformity + phenology screening** (Earth
  Engine uniformity, then NDVI vs GCC phenology agreement). Reads a pre computed
  metadata snapshot, so it makes no live PhenoCam calls. Outputs go to
  `uniformity_pipeline/output/<run>/`.
- `validation_pipeline/` - **Stage 3: plotting.** Drop a screening results JSON
  into `validation_pipeline/input/`; `main.py` re-fetches each site's PhenoCam
  curves and saves a labelled NDVI/GCC plot per site-year to
  `validation_pipeline/output/<json_stem>/`.
- `test/` - loads CSVs from `test/data/ex--/`, runs phase detection and plots via
  `shared.plotting`. Outputs go to `test/output/`.

### Data flow

```
prep/api (prep)  ->  prep/output/api/site_metadata_clean.json
                  ->  copy into uniformity_pipeline/input/  (frozen snapshot)
                  ->  uniformity_pipeline (screening)  ->  uniformity_pipeline/output/<run>/
                  ->  copy a results JSON into validation_pipeline/input/
                  ->  validation_pipeline (plots)  ->  validation_pipeline/output/<json_stem>/
```

Stage 1 writes `site_metadata_clean.json` under `prep/output/api/`; copy that file
into `uniformity_pipeline/input/` to hand it to Stage 2.

## Run plots part 1 (local files)
```bash
cd test
python3 test.py
```

The script rn loops over every `test/data/ex--/` folder that contains one `*_ndvi_3day.csv` and writes:

- `test/output/ex1.png`
- `test/output/ex2.png`
- ....

## Run the data-prep stage (`prep/api/`, live from the PhenoCam API)
```bash
cd prep/api
python3 api_year_check.py   # -> prep/output/api/year_check.json
python3 main.py             # scores site-years -> prep/output/api/site_metadata*.json
```

`main.py` exposes helpers you can toggle in its `main()` (e.g. `build_site_metadata`,
`build_gbov_metadata`, `plot_site`). Plots are written as
`prep/output/api/<roi>_<year>.png`.

## Run the screening pipeline (`uniformity_pipeline/`)

A two step screen over every site in `uniformity_pipeline/input/site_metadata_clean.json` (the
snapshot produced by Stage 1):
1. **Step 1 (Earth Engine):** keep sites whose peak summer NDVI is spatially uniform
   inside a ~4 km box: NDVI CV < 0.1, water < 5%, and non-vegetated (urban + bare)
   < 5%. Surface fractions come from the ESA WorldCover classification
   (water = 80, bare = 60, urban/built-up = 50).
2. **Step 2:** attach the pre computed NDVI vs GCC agreement scores (phenophase gap,
   DTW/step, divergence) for the Step 1 survivors.

Each run writes into its own subfolder `uniformity_pipeline/output/<run>/`
(`step1_uniformity.json`, `step2_phenology.json`, and `pipeline_results.json` -
survivors ranked by divergence).

```bash
cd uniformity_pipeline
python3 run_pipeline.py     # run ALL site years -> uniformity_pipeline/output/
```

### Optional env overrides

- `PIPELINE_INPUT=<path>` - screen a different metadata file (e.g. the GBOV subset).
- `PIPELINE_TAG=<name>` - suffix the output filenames (e.g. `_GBOV`).
- `PIPELINE_OUTDIR=<path>` - redirect outputs (e.g. `output/Full`, `output/GBOV`).
- `CV_MAX`, `WATER_MAX`, `NONVEG_MAX` - override the Step 1 thresholds for a sweep
  (defaults 0.1 / 0.05 / 0.05).

```bash
# example: strict full run into output/Full
PIPELINE_OUTDIR=output/Full python3 run_pipeline.py
# example: loose GBOV run into output/GBOV_loose
CV_MAX=0.2 WATER_MAX=0.10 NONVEG_MAX=0.10 \
  PIPELINE_INPUT=../prep/output/api/site_GBOV_clean.json \
  PIPELINE_TAG=GBOV PIPELINE_OUTDIR=output/GBOV_loose python3 run_pipeline.py
```

### `PIPELINE_LIMIT` (optional cap for running GEE Uniformality test)

`PIPELINE_LIMIT` is an optional environment variable that caps how many site-years
are processed --> useful for a quick test without waiting on the full run. It is
**not set by default**, so a plain `python3 run_pipeline.py` already processes every
site in `site_metadata_clean.json`.

```bash
PIPELINE_LIMIT=8 python3 run_pipeline.py   # only the first 8 site years
python3 run_pipeline.py                     # no limit -> all site years
```

Notes:
- Requires a one time Earth Engine `localhost` auth and `EE_PROJECT` set in `.env`.
- A full run makes several Earth Engine calls per site, can take a while.
- Tune `STEP1_WORKERS` and thresholds in `uniformity_pipeline/config.py`.

## Run the validation plots (`validation_pipeline/`)

Copy any screening results JSON (e.g. a run's `step2_phenology.json` or
`pipeline_results.json`) into `validation_pipeline/input/`, then:

```bash
cd validation_pipeline
python3 main.py             # plot every site-year in each input JSON
python3 main.py --limit 5   # quick preview: first 5 site-years per file
```

Those JSONs store only names/years/scores (not the curves), so `main.py`
re-fetches each site's PhenoCam NDVI/GCC series and saves a labelled plot (with
the cached divergence score) to `validation_pipeline/output/<json_stem>/`. The
fetch + plot logic lives in `shared/plot_json.py` (which reuses
`shared/plotting.py`), so the prep stage and this stage share one code path.

## Testing:

Set the preferred year at the top of `test/test.py` (`YEAR = 2024`).

Each CSV may cover different years. The script uses `YEAR` when that year exists in the file; otherwise it uses the **latest year available** and prints a note. For example:

- `ex1` --> 2024 
- `ex2` --> for ex, fall back to 2022 bc there is no 2024 data

*Only the NDVI summary CSV is required (it includes `gcc_90`) from the phenocam website 

*Add a new example by creating `data/ex3/` with one `*_ndvi_3day.csv` inside.

*Other potential file is gcc 3day transition day files which can be download from gcc data tab in phenocam and it's in data_record_5 folder but don't need that right now.

## Data source

[PhenoCam Network](https://phenocam.nau.edu/webcam/)
