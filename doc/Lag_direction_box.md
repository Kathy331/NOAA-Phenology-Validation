# How the lag box plot is made

This page explains `lag_direction_box.png` (one year) and
`Lag_direction_box_2023_2024.png` (both years), from the input files to the
stars printed beside each box.

The plot asks one question: **does the satellite (GVF) put a season stage
earlier or later than the ground camera (PhenoCam)?**

## 1. Input

Each site-year starts as one text file in
`plotting_pipeline/input/Satellite_GVF_timeseries_<year>/`. The file is that
site's satellite greenness (GVF) for the year. The folder is the site list.

The scorer matches the file to a PhenoCam ROI and downloads that camera's
NDVI and GCC series. It then finds four dates on each series, with the same
curve method (`shared/PhenoloDates.py`):

- **SOS** start of green-up
- **MOS** peak green
- **DOS** start of decline
- **EOS** end of the green season

The result is one row in
`anomaly_pipeline/output/Satellite_GVF_timeseries_<year>/scores.csv`.

## 2. Which sites get in

A row is drawn on the box plot only if it passes every check below. The code
is `score_one`, `build_folder_artifacts`, and `plot_lag_reference_box_years`
in `shared/data_collection.py`.

1. **Both series have the year.** If the GVF file has no data for that year,
   or the PhenoCam ROI has no NDVI/GCC for that year, the site is skipped and
   a message is printed. It never reaches `scores.csv`.
2. **Not a spin-up.** Rows with `gvf_sos = 1` are dropped. That date is day 1
   of the year, which means the curve fit did not find a real start.
3. **Known vegetation type.** Type `XX` (unknown) is dropped.
4. **Enough sites in the biome.** A biome (DB, GR, SH, ...) is drawn only when
   at least 5 sites remain in that year. Smaller groups are left off the
   figure because a box of 2 or 3 sites is not a distribution.

The uniformity screen used elsewhere in this repo (summer NDVI evenness,
water fraction, urban fraction) is **not** applied to this figure. Being in
the GVF folder is enough to be considered.

## 3. What a box is

**Lag = satellite date − camera date**, in days.

- Left of the 0 line: the satellite is earlier than the camera.
- Right of the 0 line: the satellite is later.
- On the line: they agree.

Each box is one group: one biome, one stage (SOS, MOS, DOS, or EOS), one
year, compared with one camera signal.

- **Blue** = satellite vs camera NDVI.
- **Orange** = satellite vs camera GCC.
- The black line is the typical (middle) site.
- The box is the middle half of the sites.
- Dots are unusual sites. They are listed in
  `Lag_box_outliers_by_site_*.csv`.
- `n` on the axis is how many sites are in that biome that year.
- Lags past ±90 days are drawn at the edge of the plot. The test below uses
  the real number of days.

## 4. The test (the label beside each box)

A box can sit off zero by chance, especially with few sites. So every box is
tested against the idea **"there is no shift"** (the typical lag is 0).

- `*`, `**`, `***` means the sites really do lean the same way. More stars
  means more certain.
- `ns` means not proven. Either there is no shift, or there are too few sites
  to tell. Groups of about 5 to 8 sites (WL, UN) almost never get a star, so
  `ns` there mostly means "not enough data".
- `q` is the number behind the stars. Below 0.05 gets a star. It is already
  corrected for the fact that the figure tests many boxes at once, so a few
  lucky boxes do not show up as stars.

The same numbers, plus the typical lag and its likely range, are in the
`*_stats.csv` saved next to the PNG.

**How much to trust a shift:**

1. Starred in both years, pointing the same way: a real, repeatable
   difference.
2. Starred in only one year: maybe. Check whether the other year leans the
   same way.
3. Starred in both years but pointing opposite ways: suspicious. A date was
   often picked from the wrong season. Look at those sites' images.
4. `ns`: unknown.

*For statisticians:* each box is a two-sided Wilcoxon signed-rank test of
median lag = 0, on the unclipped per-site lags, with exact-zero lags dropped.
The p-values are Benjamini-Hochberg adjusted across every box in that figure;
the label shows the adjusted value `q`. Sites are treated as independent,
which is a bit optimistic when one site has several ROIs. The helper is
`lag_shift_test` in `shared/data_collection.py`. Both box-plot functions call
it, so every regenerated PNG gets the labels and the CSV.

## 5. Using the same sites for 2025

There is no `Satellite_GVF_timeseries_2025` folder yet, so 2025 lags are not
computed. Until that folder exists, the site list to aim for is
`anomaly_pipeline/output/_combined/Sites_for_2025.xlsx`:

- **2024_rules:** every site in the 2024 boxes (the checks in section 2).
- **both_years:** those sites that also passed in 2023. `use_for_2025` is
  `yes` for this stable set.
- Both sheets include `lat` and `lon`, the PhenoCam ROI coordinates.
- **readme:** a one-line reminder that 2025 lags still need the GVF files.

To score 2025 for real, put the 2025 GVF text files in
`plotting_pipeline/input/Satellite_GVF_timeseries_2025/` and rerun the
`FOLDERS` cell in `anomaly_pipeline/anomaly_check.ipynb`. The same checks,
boxes, and tests run with no extra steps.
