"""Entry point for the API data pipeline.

  - build_site_metadata(): reads year_check.json, and for each site fetches its
    lat/lon and NDVI-vs-GCC scores (phenophase gap, DTW/step, divergence score),
    then saves them grouped by year bucket to site_metadata.json.
  - clean_site_metadata(): drops any site entry that has a null score and saves
    the result to site_metadata_clean.json.
  - plot_site(name, year): saves one site's NDVI/GCC time series plot.

Run with:  python3 src/api/main.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from pathlib import Path

# src on the path so functions here can reach PhenoloDates / plotting too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DynamicTimeWrap import ( 
	_fetch_timeseries,
	_jsonable,
	site_agreement_score,
)
from phenocam_api import ( 
	fetch_ndvi_3day_for_roi,
	find_roi,
	list_rois,
	load_timeseries,
	roi_ndvi_3day_url,
)
from plotting import create_plot, save_plot  

MAX_WORKERS = 20

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "api"
YEAR_CHECK_JSON = OUTPUT_DIR / "year_check.json"
SITE_METADATA_JSON = OUTPUT_DIR / "site_metadata.json"
SITE_METADATA_CLEAN_JSON = OUTPUT_DIR / "site_metadata_clean.json"


# Each year_check bucket and the year(s) to score its sites for.
GROUP_YEARS: dict[str, tuple[int, ...]] = {
	"has_2023_only": (2023,),
	"has_2024_only": (2024,),
	"has_both_2023_and_2024": (2023, 2024),
}


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


def score_task(task: tuple) -> tuple[str, list[dict], str | None]:
	"""Fetch one site once and build a metadata entry per requested year."""
	group, name, roi, years = task
	try:
		timeseries = _fetch_timeseries(roi)
	except Exception as error:
		return (group, [], str(error))

	entries = []
	for year in years:
		entries.append(
			{
				"name": name,
				"lat": roi.get("lat"),
				"lon": roi.get("lon"),
				"metadata": _jsonable(site_agreement_score(timeseries, year)),
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
	"""Drop every site entry that has a null value in its metadata, then save.

	`data` may be the dict returned by build_site_metadata or a path to a
	site_metadata.json file. An entry is removed if any metadata value is null
	(e.g. a missing phenophase gap or divergence score).
	"""
	if isinstance(data, (str, Path)):
		data = json.loads(Path(data).read_text())

	cleaned: dict = {}
	removed = 0
	for group, group_data in data.items():
		kept = [
			site
			for site in group_data["sites"]
			if all(value is not None for value in site["metadata"].values())
		]
		removed += len(group_data["sites"]) - len(kept)
		cleaned[group] = {"count": len(kept), "sites": kept}

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(cleaned, indent=2) + "\n")

	total = sum(group_data["count"] for group_data in cleaned.values())
	print(f"Cleaned metadata: kept {total} entries, removed {removed} with nulls.")
	print(f"Saved cleaned metadata to {output_path}")
	return cleaned


def resolve_roi(name: str, rois: list[dict] | None = None) -> dict:
	"""Resolve a roi_name (e.g. site_EN_1001) or bare site name to its ROI record."""
	rois = rois if rois is not None else list_rois()
	for roi in rois:
		if roi["roi_name"] == name:
			return roi
	return find_roi(name, rois=rois)


def plot_site(name: str, year: int, output_dir: Path = OUTPUT_DIR) -> Path:
	"""Save the NDVI/GCC time series plot for one site and year to output/api.

	`name` may be a full roi_name (site_VEG_ROI) or a bare site name. Returns the
	path of the written PNG.
	"""
	roi = resolve_roi(name)
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))

	available = {int(y) for y in timeseries["year"].dropna().unique()}
	if year not in available:
		raise ValueError(f"{roi['roi_name']} has no {year} data (available: {sorted(available)})")

	print(f"Resolved {roi['roi_name']} -> {roi_ndvi_3day_url(roi)}")
	title = f"API: PhenoCam Time Series for GCC_90 and NDVI_90 ({roi['roi_name']}, {year})"
	output_file = output_dir / f"API_data_{roi['roi_name']}_{year}.png"

	output_dir.mkdir(parents=True, exist_ok=True)
	scores = site_agreement_score(timeseries, year)
	fig, gcc_phases, ndvi_phases = create_plot(timeseries, year, title, scores=scores)
	save_plot(fig, output_file)

	print(f"Saved {output_file}")
	print(f"  GCC_90 phases (DOY, {year}): {gcc_phases}")
	print(f"  NDVI_90 phases (DOY, {year}): {ndvi_phases}")
	return output_file


def main() -> None:
	metadata = build_site_metadata(YEAR_CHECK_JSON)
	clean_site_metadata(metadata)
	# plot_site("NEON.D05.STEI.DP1.00033_DB_1000", 2023)


if __name__ == "__main__":
	main()
