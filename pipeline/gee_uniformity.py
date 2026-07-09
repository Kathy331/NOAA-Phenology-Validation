"""Earth Engine helpers for Step 1 spatial-uniformity screening.

Ports the index math from test/GEE/calcu_uniform.js to headless Python:
  - NDVI  = normalizedDifference(['B8','B4'])
  - MNDWI = normalizedDifference(['B3','B11']); water where MNDWI > 0
  - NDBI  = normalizedDifference(['B11','B8']); urban where NDBI > 0
  - spatial CV = stdDev(NDVI) / mean(NDVI) over the box (single peak-summer composite)

`init_ee()` initializes Earth Engine, falling back to a one-time localhost auth if
no cached credentials exist.
"""

import ee

import config

_initialized = False


def init_ee() -> None:
	"""Initialize Earth Engine, doing a one-time localhost auth if needed."""
	global _initialized
	if _initialized:
		return
	try:
		ee.Initialize(project=config.EE_PROJECT)
	except Exception:
		# No cached credentials (or expired): trigger the interactive flow once.
		ee.Authenticate(auth_mode="localhost")
		ee.Initialize(project=config.EE_PROJECT)
	_initialized = True


def summer_collection(point: "ee.Geometry", start: str, end: str) -> "ee.ImageCollection":
	"""Cloud-filtered S2_SR_HARMONIZED collection over the point for [start, end)."""
	return (
		ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
		.filterBounds(point)
		.filterDate(start, end)
		.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.CLOUD_PCT))
	)


def summer_composite(point: "ee.Geometry", start: str, end: str):
	"""Return (median composite image, source collection) for the summer window."""
	collection = summer_collection(point, start, end)
	return collection.median(), collection


def ndvi_of(s2: "ee.Image") -> "ee.Image":
	"""NDVI = (B8 - B4) / (B8 + B4)."""
	return s2.normalizedDifference(["B8", "B4"]).rename("NDVI")


def water_mask(s2: "ee.Image") -> "ee.Image":
	"""Binary water mask from MNDWI = (B3 - B11)/(B3 + B11) > 0."""
	return s2.normalizedDifference(["B3", "B11"]).gt(0.0).rename("water")


def urban_mask(s2: "ee.Image") -> "ee.Image":
	"""Binary urban mask from NDBI = (B11 - B8)/(B11 + B8) > 0."""
	return s2.normalizedDifference(["B11", "B8"]).gt(0.0).rename("urban")


def ndvi_cv(ndvi_image: "ee.Image", box: "ee.Geometry", scale: int = config.NDVI_SCALE_M) -> "ee.Number":
	"""Spatial coefficient of variation (stdDev/mean) of NDVI over the box."""
	stats = ndvi_image.reduceRegion(
		reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
		geometry=box,
		scale=scale,
		maxPixels=int(1e9),
	)
	mean = ee.Number(stats.get("NDVI_mean"))
	std = ee.Number(stats.get("NDVI_stdDev"))
	return std.divide(mean)


def mask_fraction(mask_image: "ee.Image", box: "ee.Geometry", scale: int = config.NDVI_SCALE_M) -> "ee.Number":
	"""Fraction (0..1) of unmasked pixels in the box where the binary mask is 1."""
	stats = mask_image.reduceRegion(
		reducer=ee.Reducer.mean(),
		geometry=box,
		scale=scale,
		maxPixels=int(1e9),
	)
	return ee.Number(stats.values().get(0))


def summer_metrics(point: "ee.Geometry", box: "ee.Geometry", start: str, end: str) -> "ee.Dictionary":
	"""All Step 1 numbers for one site-year as a single server-side dictionary.

	Bundles the image count and one combined reduceRegion (NDVI mean+stdDev, plus
	the water/urban mask means) so the whole site needs just one `.getInfo()`
	round-trip instead of four. Values are null when the composite is empty.
	"""
	collection = summer_collection(point, start, end)
	composite = collection.median()

	stacked = (
		composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
		.addBands(composite.normalizedDifference(["B3", "B11"]).gt(0.0).rename("water"))
		.addBands(composite.normalizedDifference(["B11", "B8"]).gt(0.0).rename("urban"))
	)
	stats = stacked.reduceRegion(
		reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
		geometry=box,
		scale=config.NDVI_SCALE_M,
		maxPixels=int(1e9),
	)
	return ee.Dictionary(
		{
			"n_images": collection.size(),
			"ndvi_mean": stats.get("NDVI_mean"),
			"ndvi_std": stats.get("NDVI_stdDev"),
			"water_pct": stats.get("water_mean"),
			"urban_pct": stats.get("urban_mean"),
		}
	)
