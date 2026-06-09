# PhenoloDates.py

Computes phenology transition dates — **SOS, MOS, DOS, EOS** — from a vegetation
index (VI) time series for one growing season, using the **CCRmax** method.

## Method (Zhang et al., 2003 — *"Monitoring vegetation phenology using MODIS"*)

Each half of the seasonal trajectory is fitted to a piecewise logistic model:

$$ y(t) = \frac{c}{1 + e^{a + bt}} + d $$

where $t$ is day of year, $c + d$ is the maximum VI, and $d$ is the background
value. Transition dates are located at the extrema of the *rate of change of
curvature* $K'(t)$ of that fitted curve.

Because the dates come from the curve's geometry rather than a fixed greenness
threshold, the method is **threshold independent** and comparable across biomes.
The piecewise fitting also makes it flexible for sites with more
complex cycles.

## API

```python
sos, mos = CCRmax_SOS(doy, values)      # spring: Start / Maturity of Season
dos, eos = CCRmax_EOS(doy, values)      # fall:   Decline / End of Season
phases   = compute_phases(doy, values)  # {"SOS", "MOS", "DOS", "EOS"}
```

All functions return DOY floats, or `np.nan` for any season half that lacks
enough data or cannot be fitted
