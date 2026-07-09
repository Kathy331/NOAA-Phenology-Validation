"""Step 1: spatial-uniformity screening with Earth Engine.

For each site-year, find a cloud-free Sentinel-2 composite from the middle of the
growing season (midpoint of the cached NDVI MOS..DOS dates, with a hemisphere
fallback), then measure how uniform peak-summer NDVI is inside a ~4 km box:

    passed = NDVI CV < CV_MAX  AND  water% < WATER_MAX  AND  urban% < URBAN_MAX

The MOS/DOS dates come pre-computed from site_metadata_clean.json, so no PhenoCam
call is made here. Results (every evaluated site) are written to
step1_uniformity.json.
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import ee

import config
import gee_uniformity as gee


def _finite(value) -> bool:
	return value is not None and math.isfinite(value)


def summer_center(year: int, lat: float, mos=None, dos=None) -> tuple[date, int, str]:
	"""Peak-summer center date for a site-year.

	Uses the midpoint of the (pre-computed) NDVI MOS and DOS DOYs when both are
	available; otherwise falls back to a hemisphere-appropriate mid-summer DOY.
	Returns (center_date, summer_doy, source).
	"""
	if _finite(mos) and _finite(dos):
		summer_doy, source = int(round((mos + dos) / 2)), "MOS-DOS"
	else:
		summer_doy = config.NORTH_SUMMER_DOY if lat >= 0 else config.SOUTH_SUMMER_DOY
		source = "fallback"

	summer_doy = max(1, min(365, summer_doy))
	center = date(year, 1, 1) + timedelta(days=summer_doy - 1)
	return center, summer_doy, source


def _window(center: date, half_days: int) -> tuple[str, str]:
	"""ISO [start, end) window of +/- half_days around center (end is exclusive)."""
	start = center - timedelta(days=half_days)
	end = center + timedelta(days=half_days + 1)
	return start.isoformat(), end.isoformat()


def _round(value, digits: int = 4):
	return round(float(value), digits) if value is not None else None


def evaluate_site(entry: dict) -> dict:
	"""Compute Step 1 metrics and pass/fail for one cached site-year entry.

	`entry` is {name, lat, lon, year, mos, dos, ...} from site_metadata_clean.json;
	no PhenoCam call is made -- MOS/DOS are used directly to center the window.
	"""
	name, year = entry["name"], entry["year"]
	lat, lon = entry.get("lat"), entry.get("lon")
	result = {
		"name": name,
		"lat": lat,
		"lon": lon,
		"year": year,
		"summer_doy": None,
		"summer_source": None,
		"window": None,
		"n_images": None,
		"cv_ndvi": None,
		"water_pct": None,
		"urban_pct": None,
		"passed": False,
		"status": "ok",
	}

	if lat is None or lon is None:
		result["status"] = "missing_latlon"
		return result

	try:
		center, summer_doy, source = summer_center(year, lat, entry.get("mos"), entry.get("dos"))
		result["summer_doy"] = summer_doy
		result["summer_source"] = source

		point = ee.Geometry.Point([lon, lat])
		box = point.buffer(config.BOX_RADIUS_M).bounds()

		# One getInfo per window: bundle image count + all metrics server-side.
		# Widen the window if the first pass finds no cloud-free scenes.
		half = config.SUMMER_HALF_WINDOW_DAYS
		info: dict = {}
		n_images = 0
		start = end = None
		for widen in range(config.MAX_IMAGE_SEARCH_WIDENINGS + 1):
			start, end = _window(center, half * (widen + 1))
			info = gee.summer_metrics(point, box, start, end).getInfo()
			n_images = int(info.get("n_images") or 0)
			if n_images > 0:
				break

		result["window"] = [start, end]
		result["n_images"] = n_images
		if n_images == 0:
			result["status"] = "insufficient_data"
			return result

		ndvi_mean, ndvi_std = info.get("ndvi_mean"), info.get("ndvi_std")
		water, urban = info.get("water_pct"), info.get("urban_pct")
		cv = ndvi_std / ndvi_mean if (ndvi_mean not in (None, 0) and ndvi_std is not None) else None
		result["cv_ndvi"] = _round(cv)
		result["water_pct"] = _round(water)
		result["urban_pct"] = _round(urban)

		if cv is None or water is None or urban is None:
			result["status"] = "no_valid_pixels"
			return result

		result["passed"] = bool(
			cv < config.CV_MAX and water < config.WATER_MAX and urban < config.URBAN_MAX
		)
	except Exception as error:  # noqa: BLE001 - record any per-site failure, keep going
		result["status"] = f"error: {error}"

	return result


def run_step1(entries: list[dict]) -> list[dict]:
	"""Evaluate all entries (threaded) and write step1_uniformity.json."""
	gee.init_ee()

	results: list[dict] = []
	with ThreadPoolExecutor(max_workers=config.STEP1_WORKERS) as executor:
		futures = [executor.submit(evaluate_site, entry) for entry in entries]
		for future in as_completed(futures):
			results.append(future.result())

	results.sort(key=lambda r: (r["name"], r["year"]))
	survivors = [r for r in results if r["passed"]]

	payload = {
		"thresholds": {
			"cv_max": config.CV_MAX,
			"water_max": config.WATER_MAX,
			"urban_max": config.URBAN_MAX,
		},
		"params": {
			"cloud_pct": config.CLOUD_PCT,
			"box_radius_m": config.BOX_RADIUS_M,
			"ndvi_scale_m": config.NDVI_SCALE_M,
			"summer_half_window_days": config.SUMMER_HALF_WINDOW_DAYS,
		},
		"counts": {"evaluated": len(results), "passed": len(survivors)},
		"sites": results,
	}

	config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	config.STEP1_JSON.write_text(json.dumps(payload, indent=2) + "\n")
	print(f"Step 1: {len(survivors)}/{len(results)} passed -> {config.STEP1_JSON}")
	return results
