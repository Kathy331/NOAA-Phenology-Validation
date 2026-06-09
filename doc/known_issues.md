# Known Issues — CCRmax phase detection

Notes on cases where `PhenoloDates.py` might produces misleading dates

## Observation: MOS and DOS collapse onto the seasonal peak (ex2)

For the `ex2` New Zealand pasture site (2022), the phases came out as:

```
GCC_90 : SOS 2,  MOS 44,  DOS 44,  EOS 251
NDVI_90: SOS 2,  MOS 200, DOS 200, EOS 251
```

Two things look wrong:

1. **MOS equals DOS for each index** (44/44 for GCC, 200/200 for NDVI).
2. **GCC and NDVI are far apart** (DOY 44 vs 200) for those middle phases.

### Why this happens

The method splits the year at the single seasonal peak, `peak = argmax(y)`:

- `CCRmax_SOS` runs on the **green up half** (`start --> peak`) --> returns SOS, **MOS**.
- `CCRmax_EOS` runs on the **senescence half** (`peak --> end`) --> returns **DOS**, EOS.

MOS is the last meaningful point of the green up half and DOS is the first of
the senescence half, and **both halves meet at the peak day**. When the curve is
clean, MOS lands just before the peak and DOS just after. But when the curve is
noisy or nearly flat, the curvature extrema fall on the segment boundary, so
**MOS and DOS both collapse onto the peak day**.

The two indices are far apart simply because they peak at different times:

```
gcc_90 : peak DOY 44  (mid-Feb), amplitude 0.116  (weak, nearly flat)
ndvi_90: peak DOY 200 (mid-Jul), amplitude 0.540  (strong)
```

### Root causes

- **Low amplitude / noisy signal** — GCC here has a seasonal range of only ~0.12,
  so its fit and curvature extrema are not meaningful.
- **Single-peak assumption** — the year is cut at one `argmax`. This pasture has
  multiple humps and the two indices peak in different seasons.
- **Southern Hemisphere calendar slicing** — a Jan to Dec window splits the NZ
  growing season awkwardly (summer straddles the year boundary), which is why
  GCC's max lands in February.

`ex1` (a clean curve) does not have this problem.

## Potential solutions
Rough order of effort, simplest first:

1. **Amplitude guard** If a half season's range (95th − 5th percentile) is below
   a small threshold, return `nan` for that pair instead of a date. Stops weak,
   noisy signals (like GCC at ex2) from producing fake phases.

2. **Collapse guard** If MOS lands on (or within ~1 sample of) the peak, or DOS
   does, treat it as a failed detection and return `nan`. Makes the
   "everything pins to the peak" artifact visible instead of silent.

3. **Phenological year offset for Southern Hemisphere sites** Slice the season
   July to June instead of Jan to Dec so a summer peaking cycle is not cut in half.

4. **Multiple peak handling** Detect more than one seasonal cycle (e.g. with
   `find_peaks` and a prominence threshold) instead of assuming one `argmax`.
   Needed for double cropping and multi hump pasture sites.

(1) and (2) are pretty good: they don't improve detection, but they
stop the plot from showing confident looking dates that are really just the peak.
