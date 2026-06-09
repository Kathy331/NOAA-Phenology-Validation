"""Phenology transition dates from a vegetation index time series (CCRmax method).

Used to calculate the SOS, MOS, DOS, and EOS from a vegetation index time series.
Functions return DOY floats, or ``np.nan`` for any season half that lacks enough
data or cannot be fitted.

Used by test.py to create the plots:
    sos, mos = CCRmax_SOS(doy, values)      # spring: Start / Maturity of Season
    dos, eos = CCRmax_EOS(doy, values)      # fall:   Decline / End of Season
    phases   = compute_phases(doy, values)  # {"SOS", "MOS", "DOS", "EOS"} DOY floats

See doc/PhenoloDates.md for the method (CCRmax / Zhang et al., 2003)
"""

import numpy as np
from scipy.optimize import curve_fit # logistic function

# Returned for any phase that could not be computed
MISSING_PHASE = np.nan

# Minimum number of valid samples even before attempting fit
MIN_POINTS = 10

# The seasonal peak must sit at least this many samples from either end, so that
# both the green up and senescence halves have enough points to fit
EDGE_MARGIN = 5


def _clean(doy, values) -> tuple[np.ndarray, np.ndarray]:
	"""Clean the data by dropping not valid samples and sorting by day of year"""
	doy = np.asarray(doy, dtype=float)
	values = np.asarray(values, dtype=float)
	valid_mask = np.isfinite(doy) & np.isfinite(values)
	doy, values = doy[valid_mask], values[valid_mask]
	order = np.argsort(doy)
	return doy[order], values[order]


def _fit_logistic(t: np.ndarray, y: np.ndarray):
	"""Fit y(t) = c/(1+exp(a+b*t)) + d; return (a, b, c) or None on failure.

	The amplitude c and background d are estimated up front from robust 5th/95th
	percentiles, so only the two shape parameters a and b are optimized. Fixing
	the level this way keeps the fit stable on short or noisy half-seasons.
	"""
	background = np.percentile(y, 5)              # d: dormant / baseline VI
	amplitude = np.percentile(y, 95) - background  # c: seasonal range
	if amplitude <= 0:
		return None

	def logistic(x, a, b):
		return amplitude / (1 + np.exp(a + b * x)) + background

	try:
		(a, b), _ = curve_fit(logistic, t, y, p0=[0.6557, 0.00957])
	except (RuntimeError, ValueError):
		return None
	return a, b, amplitude


def _curvature_change_rate(t: np.ndarray, a: float, b: float, c: float):
	"""Analytic rate of change of curvature K'(t) for the fitted logistic.

	This is the closed-form expression derived in Zhang et al. (2003). Evaluating
	it on a 1-day grid lets us pick transition dates as extrema of K'.
	Returns (days, k_prime) over the integer DOY span of t.
	"""
	days = np.arange(t[0], t[-1] + 1)
	z = np.exp(a + b * days)

	# Shared denominator (1+z)^4 + (b*c*z)^2 appears at powers 1.5 and 2.5 
	base = np.power(1 + z, 4) + np.power(b * c * z, 2)
	term_grow = (
		3 * z * (1 - z) * np.power(1 + z, 3)
		* (2 * np.power(1 + z, 3) + np.power(b, 2) * np.power(c, 2) * z)
	)
	term_decay = np.power(1 + z, 2) * (1 + 2 * z - 5 * np.power(z, 2))
	k_prime = np.power(b, 3) * c * z * (
		term_grow / np.power(base, 2.5) - term_decay / np.power(base, 1.5)
	)
	return days, k_prime


def _green_up_half(doy, values):
	"""Fit the start of year → peak segment; return (days, k_prime) or None"""
	t, y = _clean(doy, values)
	if len(t) < MIN_POINTS:
		return None

	peak = int(np.argmax(y))
	if peak < EDGE_MARGIN or peak > len(y) - EDGE_MARGIN:
		return None

	fit = _fit_logistic(t[: peak + 1], y[: peak + 1])
	if fit is None:
		return None
	return _curvature_change_rate(t[: peak + 1], *fit)


def _senescence_half(doy, values):
	"""Fit the peak → end of year segment; return (days, k_prime) or None"""
	t, y = _clean(doy, values)
	if len(t) < MIN_POINTS:
		return None

	peak = int(np.argmax(y))
	if peak < EDGE_MARGIN or peak > len(y) - EDGE_MARGIN:
		return None

	fit = _fit_logistic(t[peak:], y[peak:])
	if fit is None:
		return None
	return _curvature_change_rate(t[peak:], *fit)


def CCRmax_SOS(doy, values) -> tuple[float, float]:
	"""Spring transitions: (SOS, MOS) as DOY floats.

	On the green up half, K' has two local maxima straddling its global minimum
	(the steepest point of green up). The earlier maximum marks the onset of leaf
	growth (SOS); the later one marks the onset of maximum leaf area (MOS)
	"""
	half = _green_up_half(doy, values)
	if half is None:
		return MISSING_PHASE, MISSING_PHASE

	days, k_prime = half
	trough = int(np.argmin(k_prime))                       # steepest green up
	sos = days[int(np.argmax(k_prime[: trough + 1]))]      # max before trough
	mos = days[trough + int(np.argmax(k_prime[trough:]))]  # max after trough
	return float(sos), float(mos)


def CCRmax_EOS(doy, values) -> tuple[float, float]:
	"""Fall transitions: (DOS, EOS) as DOY floats.

	On the senescence half, K' has two local minima straddling its global maximum
	(the steepest point of decline). The earlier minimum marks the onset of
	decline (DOS); the later one marks the end of season (EOS)
	"""
	half = _senescence_half(doy, values)
	if half is None:
		return MISSING_PHASE, MISSING_PHASE

	days, k_prime = half
	crest = int(np.argmax(k_prime))                       # steepest senescence
	dos = days[int(np.argmin(k_prime[: crest + 1]))]      # min before crest
	eos = days[crest + int(np.argmin(k_prime[crest:]))]   # min after crest
	return float(dos), float(eos)


def compute_phases(doy, values) -> dict[str, float]:
	"""All four phases as {"SOS", "MOS", "DOS", "EOS"} in DOY floats."""
	sos, mos = CCRmax_SOS(doy, values)
	dos, eos = CCRmax_EOS(doy, values)
	return {"SOS": sos, "MOS": mos, "DOS": dos, "EOS": eos}
