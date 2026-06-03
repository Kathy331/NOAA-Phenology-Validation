"""Phenology transition dates from vegetation index time series (CCRmax).

Used by test.py:
    sos, mos = CCRmax_SOS(doy, values)
    dos, eos = CCRmax_EOS(doy, values)

Each function returns two day of the year (DOY) floats.
based on the curvature change rate (CCR) in VIIRS
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter


def _prepare_series(doy: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	doy = np.asarray(doy, dtype=float)
	values = np.asarray(values, dtype=float)
	mask = np.isfinite(doy) & np.isfinite(values)
	doy, values = doy[mask], values[mask]
	order = np.argsort(doy)
	doy, values = doy[order], values[order]

	n = len(doy)
	window = min(11, n if n % 2 else n - 1)
	if window >= 5 and window % 2 == 1:
		values = savgol_filter(values, window_length=window, polyorder=3)
	return doy, values


def _curvature_change_rate(t: np.ndarray, y: np.ndarray) -> np.ndarray:
	dy = np.gradient(y, t)
	d2y = np.gradient(dy, t)
	curvature = d2y / np.power(1 + dy**2, 1.5)
	return np.gradient(curvature, t)


def _two_peak_doys(t: np.ndarray, signal: np.ndarray, start: int, end: int) -> tuple[float, float]:
	segment = signal[start:end]
	t_segment = t[start:end]
	peaks, _ = find_peaks(segment)
	if len(peaks) < 2:
		return np.nan, np.nan

	top_peaks = peaks[np.argsort(segment[peaks])[-2:]]
	top_peaks.sort()
	return float(t_segment[top_peaks[0]]), float(t_segment[top_peaks[1]])


def CCRmax_SOS(doy, values) -> tuple[float, float]:
	"""Start and middle of spring: SOS, MOS (DOY)."""
	t, y = _prepare_series(doy, values)
	k_prime = _curvature_change_rate(t, y)
	peak_index = int(np.argmax(y))
	return _two_peak_doys(t, k_prime, 0, peak_index + 1)


def CCRmax_EOS(doy, values) -> tuple[float, float]:
	"""Start and end of fall: DOS, EOS (DOY)."""
	t, y = _prepare_series(doy, values)
	k_prime = _curvature_change_rate(t, y)
	peak_index = int(np.argmax(y))
	return _two_peak_doys(t, k_prime, peak_index, len(t))


def compute_phases(doy, values) -> dict[str, float]:
	sos, mos = CCRmax_SOS(doy, values)
	dos, eos = CCRmax_EOS(doy, values)
	return {"SOS": sos, "MOS": mos, "DOS": dos, "EOS": eos}
