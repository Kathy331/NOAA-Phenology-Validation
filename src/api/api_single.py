"""Run the PhenoCam API pipeline for a single site.

Only the site name is hardcoded for now! 
the vegetation type, ROI sequence number, NDVI availability, and file URL 
are all from the /api/roilists/ metadata.

The series will runs through phase detection, 
and plotting, and is saved as src/output/api/API_data_<site>.png.
"""

import sys
from pathlib import Path

#(plotting, PhenoloDates files)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from DynamicTimeWrap import site_agreement_score  # noqa: E402
from phenocam_api import (  # noqa: E402
	fetch_ndvi_3day_for_roi,
	find_roi,
	load_timeseries,
	pick_year,
	roi_ndvi_3day_url,
)
from plotting import create_plot, save_plot  # noqa: E402

SITE = "aafcottawacfiaf14n"  # same site as test/data/ex1
YEAR = 2024  # preferred year, uses latest year in file if this year is missing

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "api"


def main() -> None:
	roi = find_roi(SITE)
	print(f"Resolved {roi['roi_name']} (type={roi['roitype']}, ir_flag={roi['ir_flag']})")
	print(f"Fetching {roi_ndvi_3day_url(roi)}")

	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))
	year = pick_year(timeseries, YEAR)

	title = f"API: PhenoCam Time Series for GCC_90 and NDVI_90 ({roi['roi_name']}, {year})"
	output_file = OUTPUT_DIR / f"API_data_{SITE}.png"

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	scores = site_agreement_score(timeseries, year)
	fig, gcc_phases, ndvi_phases = create_plot(timeseries, year, title, scores=scores)
	save_plot(fig, output_file)

	print(f"Saved {output_file}")
	print(f"  GCC_90 phases (DOY, {year}): {gcc_phases}")
	print(f"  NDVI_90 phases (DOY, {year}): {ndvi_phases}")


if __name__ == "__main__":
	main()
