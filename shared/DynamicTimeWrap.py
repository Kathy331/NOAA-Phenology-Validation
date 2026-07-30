"""Compare how well each PhenoCam site's NDVI and GCC curves agree

All comparisons are done for a single year (`YEAR`, configurable / any year).
take a `year` argument, so callers like api_batch.py and api_single.py 
pass whichever year they are plotting. 

  1. Phenophase date comparison
     The per phase NDVI vs GCC day gaps are averaged into a mean phenophase gap.

  2. DTW (Dynamic Time Warping) comparison
     Both series are aligned to a daily axis, and fill in the gaps with the mean of the surrounding days
     min max normalized to [0, 1], then compared with DTW. Low distance = similar shape and timing.

The two are combined into a single divergence score per site (lower = better).
Running this file computes that score for every candidate site and saves them to
year output JSON; multi threading speeds up the per-site downloads/comparisons.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .PhenoloDates import compute_phases
from .phenocam_api import (
	fetch_ndvi_3day_for_roi,
	list_rois,
	load_timeseries,
)

# reads/writes the data prep JSONs under prep_pipeline/output/api
REPO_DIR = Path(__file__).resolve().parent.parent

YEAR = 2024  # year to compare NDVI vs GCC (may pass in any year w/ function: site_agreement_score(timeseries, year))
PHENOPHASE_KEYS = ("SOS", "MOS", "DOS", "EOS")   
CANDIDATE_GAP_DAYS = 14 						 # divergence score scaling factor				
MIN_POINTS = 10 								 # minimum number of samples, otherwise score is NaN
MAX_WORKERS = 20 

YEAR_CHECK_JSON = REPO_DIR / "prep_pipeline" / "output" / "api" / "year_check.json"
OUTPUT_JSON = REPO_DIR / "prep_pipeline" / "output" / "api" / "site_ranking.json"

 
# Phenophase date comparison (CCRmax, via PhenoloDates.compute_phases)
def phenophase_gaps(ndvi_phases: dict[str, float], gcc_phases: dict[str, float]) -> dict[str, float]:
	"""Absolute NDVI vs GCC day gaps per phase plus their mean.

	Keys: SOS_gap, MOS_gap, DOS_gap, EOS_gap, mean_gap. The mean ignores phases
	that are missing (np.nan) in either series.
	"""
	gaps: dict[str, float] = {}
	for phase in PHENOPHASE_KEYS:
		a, b = ndvi_phases.get(phase, np.nan), gcc_phases.get(phase, np.nan)
		gaps[f"{phase}_gap"] = abs(a - b) if (np.isfinite(a) and np.isfinite(b)) else np.nan

	valid = [gaps[f"{phase}_gap"] for phase in PHENOPHASE_KEYS if np.isfinite(gaps[f"{phase}_gap"])]
	gaps["mean_gap"] = float(np.mean(valid)) if valid else np.nan
	return gaps


def phenophase_score(timeseries: pd.DataFrame, year: int = YEAR) -> tuple[float, dict]:
	"""Mean NDVI vs GCC phenophase gap (days) for a single `year`.

	Returns (score, detail). score is np.nan if the year cannot be scored; detail
	holds the per-curve phase dates and the per-phase gaps.
	"""
	year_df = timeseries.loc[timeseries["year"] == year]
	if len(year_df) < MIN_POINTS:
		return np.nan, {}

	ndvi_phases = compute_phases(year_df["doy"].values, year_df["ndvi_90"].values)
	gcc_phases = compute_phases(year_df["doy"].values, year_df["gcc_90"].values)
	gaps = phenophase_gaps(ndvi_phases, gcc_phases)
	detail = {"ndvi": ndvi_phases, "gcc": gcc_phases, "gaps": gaps}
	return gaps["mean_gap"], detail


# DTW comparison
def normalize01(x) -> np.ndarray:
	"""Min-max scale to [0, 1]; returns zeros for a flat/empty series."""
	x = np.asarray(x, dtype=float)
	lo, hi = np.nanmin(x), np.nanmax(x)
	if not np.isfinite(lo) or hi <= lo:
		return np.zeros_like(x)
	return (x - lo) / (hi - lo)


def dtw_distance(a, b) -> float:
	"""Classic O(n*m) dynamic time warping distance between two 1-D series."""
	a = np.asarray(a, dtype=float)
	b = np.asarray(b, dtype=float)
	n, m = len(a), len(b)
	if n == 0 or m == 0:
		return np.nan

	cost = np.full((n + 1, m + 1), np.inf)
	cost[0, 0] = 0.0
	for i in range(1, n + 1):
		ai = a[i - 1]
		for j in range(1, m + 1):
			d = abs(ai - b[j - 1])
			cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
	return float(cost[n, m])


def _aligned_normalized_year(timeseries: pd.DataFrame, year: int):
	"""
	1. Daily resampling: Convert the raw data to daily averages, filling gaps with the mean of the surrounding days.
	2. Gap filling: Use linear interpolation to fill in missing data points.
	3. Forward/backward filling: Fill in missing data points at the start and end of the year.
	4. Normalization: Scale the data to [0,1] range.
	5. Return the normalized NDVI and GCC curves.
	"""
	year_df = (
		timeseries.loc[timeseries["year"] == year, ["date", "ndvi_90", "gcc_90"]]
		.dropna(subset=["date"])
		.set_index("date")
		.sort_index()
	)
	if year_df.empty:
		return None

	daily = year_df.resample("1D").mean().interpolate(method="time").ffill().bfill()
	daily = daily.dropna()
	if len(daily) < MIN_POINTS:
		return None
	return normalize01(daily["ndvi_90"].values), normalize01(daily["gcc_90"].values)


def dtw_score(timeseries: pd.DataFrame, year: int = YEAR) -> float:
	"""DTW distance between normalized NDVI/GCC curves for a single `year`."""
	prepared = _aligned_normalized_year(timeseries, year)
	if prepared is None:
		return np.nan
	ndvi_norm, gcc_norm = prepared
	return dtw_distance(ndvi_norm, gcc_norm)

 
# Divergence score (phenophase gap + per-step DTW)
def site_agreement_score(timeseries: pd.DataFrame, year: int = YEAR) -> dict[str, float]:
	"""NDVI vs GCC divergence score for a single site and `year`.

	Combines the phenophase gap (scaled by CANDIDATE_GAP_DAYS) with the per-step
	DTW cost into one self-contained number. Returns the year, phenophase gap days,
	DTW per step, and their sum as `divergence_score` (lower = better agreement).
	"""
	pheno, _ = phenophase_score(timeseries, year)

	prepared = _aligned_normalized_year(timeseries, year)
	if prepared is None:
		dtw_per_step = np.nan
	else:
		ndvi_norm, gcc_norm = prepared
		dtw_per_step = dtw_distance(ndvi_norm, gcc_norm) / max(len(ndvi_norm), 1)

	pheno_part = pheno / CANDIDATE_GAP_DAYS if np.isfinite(pheno) else np.nan
	divergence = (
		float(pheno_part + dtw_per_step)
		if (np.isfinite(pheno_part) and np.isfinite(dtw_per_step))
		else np.nan
	)
	return {
		"year": year,
		"phenophase_gap_days": pheno,
		"dtw_per_step": dtw_per_step,
		"divergence_score": divergence,
	}


# Generic pairwise agreement (any two series, e.g. GVF vs GCC / GVF vs NDVI)
def _aligned_normalized(dates, values, year: int):
	"""Daily-aligned, gap-filled, [0,1]-normalized values for one series in `year`.

	Same preprocessing as _aligned_normalized_year but for a single (dates,
	values) series from any source (a PhenoCam column or a GVF txt file), so it
	is not tied to the ndvi_90/gcc_90 column names. Returns the normalized 1-D
	array, or None if there are too few points.
	"""
	frame = pd.DataFrame(
		{
			"date": pd.to_datetime(dates, errors="coerce"),
			"value": pd.to_numeric(values, errors="coerce"),
		}
	).dropna(subset=["date"])
	frame = frame.loc[frame["date"].dt.year == year].set_index("date").sort_index()
	if frame.empty:
		return None

	daily = frame.resample("1D").mean().interpolate(method="time").ffill().bfill()
	daily = daily.dropna()
	if len(daily) < MIN_POINTS:
		return None
	return normalize01(daily["value"].values)


def pairwise_agreement(
	doy_a,
	values_a,
	dates_a,
	doy_b,
	values_b,
	dates_b,
	year: int,
) -> dict[str, float]:
	"""Divergence score between any two series (same recipe as site_agreement_score).

	Series A and B are each given as (doy, values, dates). Combines the mean
	phenophase gap (|DOY gap| across SOS/MOS/DOS/EOS, scaled by
	CANDIDATE_GAP_DAYS) with the per-step DTW distance between the [0,1]
	normalized daily curves into one `divergence_score` (lower = better). Because
	it takes raw arrays, it works for GVF-vs-GCC and GVF-vs-NDVI, not just the
	built-in NDVI-vs-GCC path.
	"""
	phases_a = compute_phases(np.asarray(doy_a, dtype=float), np.asarray(values_a, dtype=float))
	phases_b = compute_phases(np.asarray(doy_b, dtype=float), np.asarray(values_b, dtype=float))
	pheno = phenophase_gaps(phases_a, phases_b)["mean_gap"]

	norm_a = _aligned_normalized(dates_a, values_a, year)
	norm_b = _aligned_normalized(dates_b, values_b, year)
	if norm_a is None or norm_b is None:
		dtw_per_step = np.nan
	else:
		dtw_per_step = dtw_distance(norm_a, norm_b) / max(len(norm_a), len(norm_b), 1)

	pheno_part = pheno / CANDIDATE_GAP_DAYS if np.isfinite(pheno) else np.nan
	divergence = (
		float(pheno_part + dtw_per_step)
		if (np.isfinite(pheno_part) and np.isfinite(dtw_per_step))
		else np.nan
	)
	return {
		"year": year,
		"phenophase_gap_days": pheno,
		"dtw_per_step": dtw_per_step,
		"divergence_score": divergence,
	}


# load the candidate site names from the year check JSON (check input folder for year_check.json)
def load_candidate_names() -> list[str]:
	"""roi_names with both 2023 and 2024 data, from year_check.json."""
	if not YEAR_CHECK_JSON.exists():
		raise FileNotFoundError(
			f"{YEAR_CHECK_JSON} not found; run prep_pipeline/api/api_year_check.py first to generate it."
		)
	data = json.loads(YEAR_CHECK_JSON.read_text())
	return data["has_both_2023_and_2024"]


def _fetch_timeseries(roi: dict) -> pd.DataFrame:
	return load_timeseries(fetch_ndvi_3day_for_roi(roi))


def _jsonable(scores: dict) -> dict:
	"""Round floats and turn non-finite values into None for clean JSON."""
	clean: dict = {}
	for key, value in scores.items():
		if isinstance(value, float):
			clean[key] = round(float(value), 4) if np.isfinite(value) else None
		else:
			clean[key] = value
	return clean


def _score_site(roi: dict, year: int) -> tuple[str, dict | None, str | None]:
	name = roi["roi_name"]
	try:
		timeseries = _fetch_timeseries(roi)
		return (name, site_agreement_score(timeseries, year), None)
	except Exception as error:
		return (name, None, str(error))


def main(year: int = YEAR) -> dict:
	candidate_names = set(load_candidate_names())
	rois_by_name = {r["roi_name"]: r for r in list_rois() if r["roi_name"] in candidate_names}
	rois = list(rois_by_name.values())
	print(f"Comparing NDVI vs GCC for {len(rois)} sites (year {year})...\n")

	sites: dict[str, dict] = {}
	failed = 0
	with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = {executor.submit(_score_site, roi, year): roi for roi in rois}
		for future in as_completed(futures):
			name, scores, error = future.result()
			if error or scores is None:
				failed += 1
				continue
			sites[name] = _jsonable(scores)

	print(f"Scored {len(sites)} sites ({failed} failed).")

	results = {
		"params": {
			"year": year,
			"phenophase_keys": list(PHENOPHASE_KEYS),
			"candidate_gap_days": CANDIDATE_GAP_DAYS,
		},
		"counts": {
			"candidate_sites": len(rois),
			"scored": len(sites),
			"failed": failed,
		},
		"sites": sites,
	}

	OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
	print(f"\nSaved comparison to {OUTPUT_JSON}")
	return results


if __name__ == "__main__":
	main()
