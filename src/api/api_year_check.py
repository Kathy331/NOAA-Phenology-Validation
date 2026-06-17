"""Scan every PhenoCam ROI and report which sites have 2023 and/or 2024 NDVI data.

Pulls the full ROI list from /api/roilists/, downloads each NDVI 3 day summary
(only ROIs with an IR/NDVI product), and inspects the available years. Each ROI
is sorted into one of three buckets: has 2023 only, has 2024 only, or has both.

Run with:  python src/api/api_year_check.py

The resulting site lists are recorded in output/api/year_check.json
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

#(plotting, PhenoloDates files)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phenocam_api import (  # noqa: E402
	fetch_ndvi_3day_for_roi,
	list_rois,
	load_timeseries,
	roi_ndvi_3day_url,
)

TARGET_YEARS = (2023, 2024)
MAX_WORKERS = 20  # tune this -- more = faster up to a point, but be respectful to the API

OUTPUT_JSON = Path(__file__).resolve().parent.parent / "output" / "api" / "year_check.json"


def years_for_roi(roi: dict) -> set[int]:
	"""Return the set of years present in an ROI's NDVI 3-day series."""
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))
	return {int(y) for y in timeseries["year"].dropna().unique()}


def check_roi(roi: dict) -> tuple[str, set[int] | None, str | None]:
	"""Fetch one ROI and return (name, years, error_message)."""
	name = roi["roi_name"]
	try:
		years = years_for_roi(roi)
		return (name, years, None)
	except Exception as error:
		return (name, None, f"FAILED ({error})\n            url: {roi_ndvi_3day_url(roi)}")


def main() -> dict:
	rois = list_rois()
	ir_rois = [(i, roi) for i, roi in enumerate(rois, 1) if roi.get("ir_flag", False)]
	total = len(rois)

	print(f"Scanning {len(ir_rois)} IR-capable ROIs (of {total} total) for {TARGET_YEARS} data...\n")

	has_2023, has_2024, has_both = [], [], []
	checked, failed = 0, 0

	with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
		futures = {executor.submit(check_roi, roi): (i, roi) for i, roi in ir_rois}

		for future in as_completed(futures):
			i, roi = futures[future]
			name, years, error = future.result()

			if error:
				failed += 1
				print(f"[{i:4d}/{total}] {name}: {error}")
				continue

			checked += 1
			got_2023 = 2023 in years
			got_2024 = 2024 in years

			if got_2023 and got_2024:
				has_both.append(name)
			elif got_2023:
				has_2023.append(name)
			elif got_2024:
				has_2024.append(name)

			if got_2023 or got_2024:
				label = "2023+2024" if (got_2023 and got_2024) else ("2023" if got_2023 else "2024")
				print(f"[{i:4d}/{total}] {name}: {label}")

	has_2023.sort()
	has_2024.sort()
	has_both.sort()

	print(f"\nChecked {checked} ROIs ({failed} failed).")
	print(f"\nHas 2023 only ({len(has_2023)}): {has_2023}")
	print(f"\nHas 2024 only ({len(has_2024)}): {has_2024}")
	print(f"\nHas both 2023 and 2024 ({len(has_both)}): {has_both}")

	results = {
		"checked": checked,
		"failed": failed,
		"counts": {
			"has_2023_only": len(has_2023),
			"has_2024_only": len(has_2024),
			"has_both_2023_and_2024": len(has_both),
		},
		"has_2023_only": has_2023,
		"has_2024_only": has_2024,
		"has_both_2023_and_2024": has_both,
	}

	OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
	OUTPUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
	print(f"\nSaved results to {OUTPUT_JSON}")

	return results


if __name__ == "__main__":
	main()
