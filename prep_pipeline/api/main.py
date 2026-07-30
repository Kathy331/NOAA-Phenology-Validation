"""Entry point for the API data pipeline.

  - build_site_metadata(): reads year_check.json, and for each site fetches its
    lat/lon and NDVI-vs-GCC scores (phenophase gap, DTW/step, divergence score),
    plus the four phenophase dates (SOS/MOS/DOS/EOS) for each of the NDVI and GCC
    curves, then saves them grouped by year bucket to site_metadata.json.
  - clean_site_metadata(): drops any site entry that has a null score and saves
    the result to site_metadata_clean.json.
  - build_gbov_metadata(): same scoring but only for the NEON/GBOV sites listed in
    doc/PhenoCam GBOV sites.xlsx, saved (clean style) to site_GBOV_clean.json.
  - plot_site(name, year): saves one site's NDVI/GCC time series plot.
  - plot_from_results(json): plots every site-year in a pipeline results / step2
    JSON (re-fetches the curves from PhenoCam, reuses the cached scores).

Run with:  python3 prep_pipeline/api/main.py
"""

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

# Run with:  cd prep_pipeline/api && python3 main.py
# repo root for shared package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.DynamicTimeWrap import _fetch_timeseries, _jsonable, site_agreement_score
from shared.PhenoloDates import compute_phases
from shared.phenocam_api import list_rois
from shared.plot_json import plot_from_results as _plot_from_results, plot_one

MAX_WORKERS = 20

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "api"
YEAR_CHECK_JSON = OUTPUT_DIR / "year_check.json"
SITE_METADATA_JSON = OUTPUT_DIR / "site_metadata.json"
SITE_METADATA_CLEAN_JSON = OUTPUT_DIR / "site_metadata_clean.json"
SITE_GEE_CLEAN_JSON = OUTPUT_DIR / "site_GEE_clean.json"
SITE_TOP_JSON = OUTPUT_DIR / "site_top.json"
SITE_GBOV_CLEAN_JSON = OUTPUT_DIR / "site_GBOV_clean.json"

# NEON/GBOV validation sites (Site name + ROI Subset + year X marks).
GBOV_XLSX = Path(__file__).resolve().parent.parent.parent / "doc" / "PhenoCam GBOV sites.xlsx"


# Each year_check bucket and the year(s) to score its sites for.
GROUP_YEARS: dict[str, tuple[int, ...]] = {
	"has_2023_only": (2023,),
	"has_2024_only": (2024,),
	"has_both_2023_and_2024": (2023, 2024),
}

# Metrics that top_sites can rank by / threshold on.
RANK_METRICS = ("divergence_score", "dtw_per_step", "phenophase_gap_days")


def run_threaded(func: Callable, items: Iterable, max_workers: int = MAX_WORKERS) -> list:
	"""Run func over items concurrently and return results as they complete."""
	results = []
	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = [executor.submit(func, item) for item in items]
		for future in as_completed(futures):
			results.append(future.result())
	return results


def load_year_check_groups(json_path: Path = YEAR_CHECK_JSON) -> dict[str, list[str]]:
	"""Return {bucket: [roi_name, ...]} for each year_check bucket we score."""
	data = json.loads(Path(json_path).read_text())
	return {group: list(data.get(group, [])) for group in GROUP_YEARS}


def build_tasks(
	groups: dict[str, list[str]], rois_by_name: dict[str, dict]
) -> tuple[list[tuple], list[str]]:
	"""Pair every (group, site) with its ROI record and the years to score.

	Returns (tasks, missing) where each task is (group, name, roi, years) and
	missing lists names with no ROI metadata.
	"""
	tasks: list[tuple] = []
	missing: list[str] = []
	for group, names in groups.items():
		years = GROUP_YEARS[group]
		for name in names:
			roi = rois_by_name.get(name)
			if roi is None:
				missing.append(name)
				continue
			tasks.append((group, name, roi, years))
	return tasks, missing


def _clean_phases(phases: dict[str, float]) -> dict[str, float | None]:
	"""Round DOY phase values and turn not finite ones into null."""
	return {
		phase: (round(float(doy), 2) if doy is not None and math.isfinite(doy) else None)
		for phase, doy in phases.items()
	}


def phenophase_dates(timeseries, year: int) -> dict[str, dict[str, float | None]]:
	"""SOS/MOS/DOS/EOS (DOY) for both the NDVI and GCC curves for one year.

	Stored under a site's metadata as {"ndvi": {...}, "gcc": {...}} so both
	curves' four dates sit together; any phase that could not be fit is null.
	"""
	year_df = timeseries.loc[timeseries["year"] == year]
	return {
		"ndvi": _clean_phases(compute_phases(year_df["doy"].values, year_df["ndvi_90"].values)),
		"gcc": _clean_phases(compute_phases(year_df["doy"].values, year_df["gcc_90"].values)),
	}


def score_task(task: tuple) -> tuple[str, list[dict], str | None]:
	"""Fetch one site once and build a metadata entry per requested year."""
	group, name, roi, years = task
	try:
		timeseries = _fetch_timeseries(roi)
	except Exception as error:
		return (group, [], str(error))

	entries = []
	for year in years:
		metadata = _jsonable(site_agreement_score(timeseries, year))
		metadata["phenophases"] = phenophase_dates(timeseries, year)
		entries.append(
			{
				"name": name,
				"lat": roi.get("lat"),
				"lon": roi.get("lon"),
				"metadata": metadata,
			}
		)
	return (group, entries, None)


def build_site_metadata(year_check_json: Path = YEAR_CHECK_JSON) -> dict:
	"""Score every year_check site and save the grouped, geo-tagged metadata."""
	groups = load_year_check_groups(year_check_json)
	rois_by_name = {roi["roi_name"]: roi for roi in list_rois()}
	tasks, missing = build_tasks(groups, rois_by_name)

	total_entries = sum(len(GROUP_YEARS[g]) * len(n) for g, n in groups.items())
	print(f"Scoring {len(tasks)} sites -> ~{total_entries} site-year entries...\n")

	grouped: dict[str, list[dict]] = {group: [] for group in GROUP_YEARS}
	failed = 0
	for group, entries, error in run_threaded(score_task, tasks):
		if error:
			failed += 1
			continue
		grouped[group].extend(entries)

	results: dict = {}
	for group in GROUP_YEARS:
		sites = sorted(grouped[group], key=lambda e: (e["name"], e["metadata"]["year"]))
		results[group] = {"count": len(sites), "sites": sites}

	scored = sum(results[group]["count"] for group in GROUP_YEARS)
	print(f"Saved {scored} entries ({failed} sites failed, {len(missing)} missing metadata).")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	SITE_METADATA_JSON.write_text(json.dumps(results, indent=2) + "\n")
	print(f"\nSaved site metadata to {SITE_METADATA_JSON}")
	return results


def clean_site_metadata(
	data: dict | str | Path, output_path: Path = SITE_METADATA_CLEAN_JSON
) -> dict:
	"""Drop every site entry that has a null score in its metadata, then save.

	`data` may be the dict returned by build_site_metadata or a path to a
	site_metadata.json file. An entry is removed if any of its scalar scores is
	null (e.g. a missing phenophase gap or divergence score). The nested
	phenophases block is not null-checked, so a site is kept even if a single
	SOS/MOS/DOS/EOS date could not be fit.
	"""
	if isinstance(data, (str, Path)):
		data = json.loads(Path(data).read_text())

	cleaned: dict = {}
	removed = 0
	for group, group_data in data.items():
		kept = [
			site
			for site in group_data["sites"]
			if all(
				value is not None
				for value in site["metadata"].values()
				if not isinstance(value, dict)
			)
		]
		removed += len(group_data["sites"]) - len(kept)
		cleaned[group] = {"count": len(kept), "sites": kept}

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(cleaned, indent=2) + "\n")

	total = sum(group_data["count"] for group_data in cleaned.values())
	print(f"Cleaned metadata: kept {total} entries, removed {removed} with nulls.")
	print(f"Saved cleaned metadata to {output_path}")
	return cleaned


def _year_bucket(years: tuple[int, ...]) -> str:
	"""Map the requested year(s) to the matching GROUP_YEARS bucket."""
	if 2023 in years and 2024 in years:
		return "has_both_2023_and_2024"
	if 2023 in years:
		return "has_2023_only"
	return "has_2024_only"


def load_gbov_tasks(
	xlsx_path: Path, rois_by_name: dict[str, dict]
) -> tuple[list[tuple], list[str]]:
	"""Read the GBOV spreadsheet into score_task tuples (group, name, roi, years).

	Each row gives a Site name, an ROI Subset (e.g. DB_1000), and an 'X' under
	2023 and/or 2024 for the years to fetch. Rows without a site/subset (note-only
	rows) are skipped. Returns (tasks, missing) where missing lists ROI names with
	no PhenoCam metadata.
	"""
	df = pd.read_excel(xlsx_path)
	tasks: list[tuple] = []
	missing: list[str] = []
	for _, row in df.iterrows():
		name_cell, subset = row.get("Site name"), row.get("Subset")
		if pd.isna(name_cell) or pd.isna(subset):
			continue
		# Cells can carry stray whitespace/newlines; normalize before matching.
		roi_name = f"{str(name_cell).strip()}_{str(subset).strip()}"
		years = tuple(y for y in (2023, 2024) if str(row.get(y)).strip() == "X")
		if not years:
			continue
		roi = rois_by_name.get(roi_name)
		if roi is None:
			missing.append(roi_name)
			continue
		tasks.append((_year_bucket(years), roi_name, roi, years))
	return tasks, missing


def build_gbov_metadata(
	xlsx_path: Path = GBOV_XLSX, output_path: Path = SITE_GBOV_CLEAN_JSON
) -> dict:
	"""Score just the GBOV spreadsheet sites and save them in the clean-JSON style.

	Fetches PhenoCam NDVI/GCC for each Site name + ROI Subset and requested year,
	computes the same metadata as build_site_metadata (phenophase gap, DTW/step,
	divergence, plus the NDVI/GCC phase dates), groups by year bucket, and drops
	null-score entries, giving the same structure as site_metadata_clean.json.
	"""
	rois_by_name = {roi["roi_name"]: roi for roi in list_rois()}
	tasks, missing = load_gbov_tasks(xlsx_path, rois_by_name)
	print(f"GBOV: scoring {len(tasks)} ROIs from {xlsx_path.name} ({len(missing)} missing metadata)...")
	if missing:
		print(f"  missing ROIs: {missing}")

	grouped: dict[str, list[dict]] = {group: [] for group in GROUP_YEARS}
	failed = 0
	for group, entries, error in run_threaded(score_task, tasks):
		if error:
			failed += 1
			continue
		grouped[group].extend(entries)

	results: dict = {}
	for group in GROUP_YEARS:
		sites = sorted(grouped[group], key=lambda e: (e["name"], e["metadata"]["year"]))
		results[group] = {"count": len(sites), "sites": sites}

	scored = sum(results[group]["count"] for group in GROUP_YEARS)
	print(f"GBOV: scored {scored} site-years ({failed} failed).")

	# Drop null-score entries so the file matches the site_metadata_clean style.
	return clean_site_metadata(results, output_path=output_path)


def save_site_data(data: dict | str | Path, output_path: Path = SITE_TOP_JSON) -> dict:
	"""Write grouped site data to JSON keeping all metadata fields intact.

	`data` may be a dict (e.g. from top_sites) or a path to a json file. Unlike
	build_gee_clean, this keeps the full metadata (year, phenophase_gap_days,
	dtw_per_step, divergence_score).
	"""
	if isinstance(data, (str, Path)):
		data = json.loads(Path(data).read_text())

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(data, indent=2) + "\n")

	total = sum(len(group_data["sites"]) for group_data in data.values())
	print(f"Saved {total} entries (full metadata) to {output_path}")
	return data


def build_gee_clean(
	data: dict | str | Path, output_path: Path = SITE_GEE_CLEAN_JSON
) -> dict:
	"""Strip metadata down to name/lat/lon plus metadata.year, then save for GEE.

	`data` may be the dict returned by clean_site_metadata or a path to a
	site_metadata_clean.json file.
	"""
	if isinstance(data, (str, Path)):
		data = json.loads(Path(data).read_text())

	stripped: dict = {}
	for group, group_data in data.items():
		sites = [
			{
				"name": site["name"],
				"lat": site["lat"],
				"lon": site["lon"],
				"metadata": {"year": site["metadata"]["year"]},
			}
			for site in group_data["sites"]
		]
		stripped[group] = {"count": len(sites), "sites": sites}

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(stripped, indent=2) + "\n")

	total = sum(group_data["count"] for group_data in stripped.values())
	print(f"GEE-clean: kept {total} sites (name/lat/lon only).")
	print(f"Saved GEE-clean sites to {output_path}")
	return stripped


def top_sites(
	data: dict | str | Path,
	n: int = 10,
	sort_by: str = "divergence_score",
	max_divergence_score: float | None = None,
	max_dtw_per_step: float | None = None,
	max_phenophase_gap_days: float | None = None,
	output_path: Path | None = None,
) -> dict:
	"""Return the N site entries with the lowest `sort_by` metric.

	`data` may be the dict from clean_site_metadata or a path to a
	site_metadata_clean.json file. `sort_by` is one of RANK_METRICS. The optional
	max_* arguments keep only entries whose metric is at or below the given cap.

	The result keeps the site_metadata structure (a single ranked group), so it
	can be passed straight to build_gee_clean. If output_path is given, it is also
	saved there.
	"""
	if sort_by not in RANK_METRICS:
		raise ValueError(f"sort_by must be one of {RANK_METRICS}, got {sort_by!r}")

	if isinstance(data, (str, Path)):
		data = json.loads(Path(data).read_text())

	entries = [site for group_data in data.values() for site in group_data["sites"]]

	caps = {
		"divergence_score": max_divergence_score,
		"dtw_per_step": max_dtw_per_step,
		"phenophase_gap_days": max_phenophase_gap_days,
	}
	for metric, cap in caps.items():
		if cap is not None:
			entries = [
				e for e in entries
				if e["metadata"].get(metric) is not None and e["metadata"][metric] <= cap
			]

	ranked = sorted(
		(e for e in entries if e["metadata"].get(sort_by) is not None),
		key=lambda e: e["metadata"][sort_by],
	)[:n]

	result = {f"top_{n}_lowest_{sort_by}": {"count": len(ranked), "sites": ranked}}

	if output_path is not None:
		output_path.parent.mkdir(parents=True, exist_ok=True)
		output_path.write_text(json.dumps(result, indent=2) + "\n")
		print(f"Saved top {len(ranked)} sites by {sort_by} to {output_path}")

	return result


def plot_site(name: str, year: int, output_dir: Path = OUTPUT_DIR) -> Path:
	"""Save the NDVI/GCC time series plot for one site and year to output/api.

	`name` may be a full roi_name (site_VEG_ROI) or a bare site name. Returns the
	path of the written PNG. Thin wrapper over shared.plot_json.plot_one.
	"""
	return plot_one(name, year, output_dir)


def plot_from_results(
	results_json: str | Path,
	output_dir: Path = OUTPUT_DIR,
	limit: int | None = None,
) -> list[Path]:
	"""Plot every site-year in a results JSON to output/api (defaults to prep_pipeline).

	Thin wrapper over shared.plot_json.plot_from_results; see that function for
	the accepted JSON formats (step2_phenology.json / pipeline_results.json).
	"""
	return _plot_from_results(results_json, output_dir, limit=limit)


def main() -> None:
	build_gbov_metadata()

	# Full pipeline (uncomment to rebuild everything from year_check.json). This
	# scores every site-year, including SOS/MOS/DOS/EOS for NDVI and GCC, then
	# drops entries with null scores.
	# metadata = build_site_metadata(YEAR_CHECK_JSON)
	# clean_site_metadata(metadata)

	# top = top_sites(SITE_METADATA_CLEAN_JSON, n=40, sort_by="divergence_score", max_divergence_score=0.5)
	# save_site_data(top)
	# build_gee_clean(top)
	# plot_site("NEON.D08.TOMB.DP1.20002_DB_2000", 2023)

	# show the plot for 'NEON.D10.STER.DP1.00033'
	plot_site("NEON.D10.STER.DP1.00033", 2023)


if __name__ == "__main__":
	main()
