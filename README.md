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

## Run plots
```bash
cd test
python3 test.py
```

The script rn loops over every `test/data/ex--/` folder that contains one `*_ndvi_3day.csv` and writes:

- `test/output/ex1.png`
- `test/output/ex2.png`
- ....

## IMPORTANT NOTES

Set the preferred year at the top of `test/test.py` (`YEAR = 2024`).

Each CSV may cover different years. The script uses `YEAR` when that year exists in the file; otherwise it uses the **latest year available** and prints a note. For example:

- `ex1` --> 2024 
- `ex2` --> for ex, fall back to 2022 bc there is no 2024 data

*Only the NDVI summary CSV is required (it includes `gcc_90`). 

*Add a new example by creating `data/ex3/` with one `*_ndvi_3day.csv` inside.

*Other potential file is gcc 3day transition day files which can be download from gcc data tab in phenocam and it's in data_record_5 folder but don't need that right now.

## Data source

[PhenoCam Network](https://phenocam.nau.edu/webcam/)
