"""Earth Engine helpers for Step 1 spatial-uniformity screening.

Ports the index math from test/GEE/calcu_uniform.js to headless Python:
  - NDVI = normalizedDifference(['B8','B4']) on a peak-summer Sentinel-2 composite
  - spatial CV = stdDev(NDVI) / mean(NDVI) over the box (single peak-summer composite)

Surface type (water / bare / urban) no longer comes from self-computed MNDWI/NDBI
thresholds. Those indices can't cleanly separate bare soil from built-up land (both
are bright in SWIR, dark in NIR), so ambiguous pixels leaked into the wrong mask.
Instead we read the pre-classified ESA WorldCover v200 map, where each pixel already
has exactly one class:
  10 Trees | 20 Shrubland | 30 Grassland | 40 Cropland | 50 Built-up |
  60 Bare/sparse vegetation | 70 Snow/ice | 80 Water | 90 Wetland |
  95 Mangroves | 100 Moss/lichen
  -> water = class 80, bare = class 60, urban/built-up = class 50.
Trade-off: WorldCover is a static 2020/2021 snapshot, not computed live per YEAR, so
it won't reflect land changed after that.

`init_ee()` initializes Earth Engine, falling back to a one-time localhost auth if
no cached credentials exist.
"""

import ee

import config

_initialized = False

# ESA WorldCover class codes used for the surface-type flags.
WC_WATER = 80
WC_BARE = 60
WC_URBAN = 50


def worldcover_map() -> "ee.Image":
	"""Single-band ESA WorldCover v200 classification map ('Map' band)."""
	return ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")


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


def water_mask(worldcover: "ee.Image") -> "ee.Image":
	"""Binary water mask from ESA WorldCover (class 80)."""
	return worldcover.eq(WC_WATER).rename("water")


def bare_mask(worldcover: "ee.Image") -> "ee.Image":
	"""Binary bare/sparse-vegetation mask from ESA WorldCover (class 60)."""
	return worldcover.eq(WC_BARE).rename("bare")


def urban_mask(worldcover: "ee.Image") -> "ee.Image":
	"""Binary built-up mask from ESA WorldCover (class 50)."""
	return worldcover.eq(WC_URBAN).rename("urban")


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

	Bundles two reduceRegions into one `.getInfo()` round-trip:
	  - NDVI mean+stdDev from the peak-summer Sentinel-2 composite (null when the
	    composite is empty), and
	  - water/bare/urban pixel fractions from the static ESA WorldCover map.
	WorldCover is independent of the cloud window, so those fractions are returned
	even if no cloud-free scene is found.
	"""
	collection = summer_collection(point, start, end)
	composite = collection.median()

	ndvi_stats = (
		composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
		.reduceRegion(
			reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True),
			geometry=box,
			scale=config.NDVI_SCALE_M,
			maxPixels=int(1e9),
		)
	)

	worldcover = worldcover_map()
	surface = (
		water_mask(worldcover)
		.addBands(bare_mask(worldcover))
		.addBands(urban_mask(worldcover))
	)
	surface_stats = surface.reduceRegion(
		reducer=ee.Reducer.mean(),
		geometry=box,
		scale=config.NDVI_SCALE_M,
		maxPixels=int(1e9),
	)

	return ee.Dictionary(
		{
			"n_images": collection.size(),
			"ndvi_mean": ndvi_stats.get("NDVI_mean"),
			"ndvi_std": ndvi_stats.get("NDVI_stdDev"),
			"water_pct": surface_stats.get("water"),
			"bare_pct": surface_stats.get("bare"),
			"urban_pct": surface_stats.get("urban"),
		}
	)
