# DynamicTimeWrap.py

Compares two vegetation index time series for one year (e.g. NDVI vs GCC, or
GVF vs either) with a **phenophase gap** plus **Dynamic Time Warping (DTW)**,
combined into one **divergence score** (lower = better).

## Method

### Phenophase gap (timing)

SOS, MOS, DOS, and EOS come from the CCRmax fit in
[PhenoloDates.md](PhenoloDates.md). The gap is the mean
absolute day difference across phases that exist on both series:

$$
\text{phenophase\_gap} =
\text{mean}\big(
  |\text{SOS}_a - \text{SOS}_b|,\;
  |\text{MOS}_a - \text{MOS}_b|,\;
  |\text{DOS}_a - \text{DOS}_b|,\;
  |\text{EOS}_a - \text{EOS}_b|
\big)
$$

### DTW per step (shape)

Both series are resampled daily, gap-filled, and min-max normalized to $[0,1]$.
DTW finds the optimal nonlinear alignment and sums the leftover distances along
that path (Baumann et al., 2017). Because the raw distance grows with series
length, we report an average per day:

$$
\text{dtw\_per\_step} = \frac{\text{DTW}(a, b)}{\max(n,\, m)}
$$

### Divergence score

$$
\text{divergence\_score}
  = \frac{\text{phenophase\_gap}}{\text{CANDIDATE\_GAP\_DAYS}}
  + \text{dtw\_per\_step}
$$

`CANDIDATE_GAP_DAYS` is currently **14**. Example: gap 15 d and
`dtw_per_step` 0.04 → $15/14 + 0.04 \approx 1.11$.

## API

```python
site_agreement_score(timeseries, year)   # NDVI vs GCC (PhenoCam frame)
pairwise_agreement(doy_a, values_a, dates_a,
                   doy_b, values_b, dates_b, year)  # any two series
# both return {year, phenophase_gap_days, dtw_per_step, divergence_score}
```

## Sources

**Baumann, M., Ozdogan, M., Richardson, A. D., & Radeloff, V. C. (2017).**
Phenology from Landsat when data is scarce: Using MODIS and Dynamic Time-Warping
to combine multi-year Landsat imagery to derive annual phenology curves.
*ISPRS Journal of Photogrammetry and Remote Sensing*, 123, 149–162.
https://www.sciencedirect.com/science/article/pii/S0303243416301623
