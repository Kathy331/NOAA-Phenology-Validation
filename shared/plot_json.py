"""Plot PhenoCam NDVI/GCC time series for the sites named in a results JSON.

The pipeline results / metadata JSONs (site_metadata*.json, step2_phenology.json,
pipeline_results.json) store only site names, years, and agreement scores, not the
raw curves. So these helpers refetch each site's NDVI 3 day series from PhenoCam
and render it with shared.plotting, reusing the cached scores for the on plot
divergence annotation.

Shared by the data prep stage (prep_pipeline/api/main.py) and the plotting stage
(plotting_pipeline/main.py) so the fetch/plot logic lives in one place.
"""

import json
from pathlib import Path

from .DynamicTimeWrap import site_agreement_score
from .phenocam_api import (
	fetch_ndvi_3day_for_roi,
	find_roi,
	list_rois,
	load_timeseries,
	roi_ndvi_3day_url,
)
from .plotting import create_plot, save_plot


def resolve_roi(name: str, rois: list[dict] | None = None) -> dict:
	"""Resolve a roi_name (e.g. site_EN_1001) or bare site name to its ROI record."""
	rois = rois if rois is not None else list_rois()
	for roi in rois:
		if roi["roi_name"] == name:
			return roi
	return find_roi(name, rois=rois)


def plot_one(
	name: str,
	year: int,
	output_dir: Path,
	rois: list[dict] | None = None,
	scores: dict | None = None,
) -> Path:
	"""Fetch one site's NDVI/GCC series and save its plot; returns the PNG path.

	`name` may be a full roi_name (site_VEG_ROI) or a bare site name. Pass a
	pre-fetched `rois` list to avoid re downloading the ROI list per call, and a
	cached `scores` dict (from site_agreement_score / a results JSON) to annotate
	the divergence score without recomputing it.
	"""
	output_dir = Path(output_dir)
	roi = resolve_roi(name, rois=rois)
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))

	available = {int(y) for y in timeseries["year"].dropna().unique()}
	if year not in available:
		raise ValueError(f"{roi['roi_name']} has no {year} data (available: {sorted(available)})")

	if scores is None:
		scores = site_agreement_score(timeseries, year)

	print(f"Resolved {roi['roi_name']} -> {roi_ndvi_3day_url(roi)}")
	title = f"API: PhenoCam Time Series for GCC_90 and NDVI_90 ({roi['roi_name']}, {year})"
	output_file = output_dir / f"{roi['roi_name']}_{year}.png"

	output_dir.mkdir(parents=True, exist_ok=True)
	fig, gcc_phases, ndvi_phases = create_plot(timeseries, year, title, scores=scores)
	save_plot(fig, output_file)

	print(f"Saved {output_file}")
	print(f"  GCC_90 phases (DOY, {year}): {gcc_phases}")
	print(f"  NDVI_90 phases (DOY, {year}): {ndvi_phases}")
	return output_file


def plot_from_results(
	results_json: str | Path,
	output_dir: Path,
	limit: int | None = None,
) -> list[Path]:
	"""Plot every site year listed in a pipeline results / step2 JSON.

	Accepts either step2_phenology.json (scores nested under "metadata") or
	pipeline_results.json (scores nested under "phenology"); both store a top
	level "sites" list of {name, year, ...}. Those files hold only names, years,
	and scores (not the curves), so the NDVI/GCC series is re-fetched from
	PhenoCam for each entry and the cached scores are reused for the on-plot
	divergence annotation. `limit` caps how many entries are plotted (handy for a
	quick preview). Returns the list of written PNG paths; entries that fail
	(e.g. no data for that year) are skipped with a message.
	"""
	data = json.loads(Path(results_json).read_text())
	entries = data.get("sites", [])
	if limit is not None:
		entries = entries[:limit]

	rois = list_rois()  # fetch the ROI list once and reuse it for every resolve
	written: list[Path] = []
	for entry in entries:
		name, year = entry.get("name"), entry.get("year")
		if name is None or year is None:
			continue
		scores = entry.get("metadata") or entry.get("phenology")
		try:
			written.append(plot_one(name, int(year), output_dir, rois=rois, scores=scores))
		except Exception as error:
			print(f"  skip {name} ({year}): {error}")

	print(f"\nPlotted {len(written)}/{len(entries)} site years to {output_dir}")
	return written
