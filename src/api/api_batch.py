"""Batch-run the first N PhenoCam ROIs straight from the API.

Pulls the ROI list from /api/roilists/, takes the first N records, and for each
one downloads the NDVI 3 day series, runs CCRmax phase detection, and saves a
figure as src/output/api/API_data_<roi_name>.png. 

Failures (missing file, too little data, etc) 
are reported without stopping the batch.
"""

import sys
from pathlib import Path

#(plotting, PhenoloDates files)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DynamicTimeWrap import site_agreement_score 
from phenocam_api import (  # noqa: E402
	fetch_ndvi_3day_for_roi,
	list_rois,
	load_timeseries,
	pick_year,
	roi_ndvi_3day_url,
)
from plotting import create_plot, save_plot  

N_SITES = 10
YEAR = 2024  # preferred year, uses latest year in file if this year is missing

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "api"


def process_roi(roi: dict):
	"""Fetch, plot, and save one ROI. Returns (year, output_file, gcc, ndvi)."""
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))
	year = pick_year(timeseries, YEAR)
	title = f"API: PhenoCam Time Series for GCC_90 and NDVI_90 ({roi['roi_name']}, {year})"
	output_file = OUTPUT_DIR / f"API_data_{roi['roi_name']}.png"

	scores = site_agreement_score(timeseries, year)
	fig, gcc_phases, ndvi_phases = create_plot(timeseries, year, title, scores=scores)
	save_plot(fig, output_file)
	return year, output_file, gcc_phases, ndvi_phases


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	rois = list_rois()[:N_SITES]
	print(f"First {len(rois)} ROIs from /api/roilists/:")
	for i, roi in enumerate(rois, 1):
		print(f"  {i:2d}. {roi['roi_name']}  (site={roi['site']}, type={roi['roitype']})")
	print()

	saved, failed = 0, 0
	for i, roi in enumerate(rois, 1):
		name = roi["roi_name"]
		try:
			year, output_file, gcc_phases, ndvi_phases = process_roi(roi)
			saved += 1
			print(f"[{i:2d}/{len(rois)}] {name}: saved {output_file.name} (year {year})")
			print(f"          GCC_90  {gcc_phases}")
			print(f"          NDVI_90 {ndvi_phases}")
		except Exception as error:  # keep going if one ROI fails
			failed += 1
			print(f"[{i:2d}/{len(rois)}] {name}: FAILED ({error})")
			print(f"          url: {roi_ndvi_3day_url(roi)}")

	print(f"\nDone: {saved} saved, {failed} failed, into {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
