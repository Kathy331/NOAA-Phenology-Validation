"""Plot satellite GVF against PhenoCam GCC/NDVI for the GBOV sites.

The GBOV network ships a daily satellite Green Vegetation Fraction (GVF) series
as plain text files (`YYYYMMDD,GVF`, GVF an integer 0-100) under folders like
plotting_pipeline/input/GBOV_2023/. GCC and NDVI are not in those files, so
they are refetched from PhenoCam (same source as shared.plot_json) using the ROI
recorded for each GBOV site.

Each plot overlays all three curves (GVF + GCC + NDVI), draws SOS/MOS/DOS/EOS
markers for each, and annotates freshly computed GVF vs GCC and GVF vs NDVI
agreement scores. Used by plotting_pipeline/main.py to render the GBOV_<year>
input folders.
"""

import json
import re
from pathlib import Path

import pandas as pd

from .DynamicTimeWrap import pairwise_agreement
from .phenocam_api import fetch_ndvi_3day_for_roi, list_rois, load_timeseries
from .plot_json import resolve_roi
from .plotting import create_satellite_plot, save_plot

REPO_DIR = Path(__file__).resolve().parents[1]
GBOV_CLEAN_JSON = REPO_DIR / "prep_pipeline" / "output" / "api" / "site_GBOV_clean.json"


def load_gvf(txt_path: str | Path) -> pd.DataFrame:
	"""Parse a GBOV GVF text file into a date sorted DataFrame.

	The file is headerless `YYYYMMDD,GVF` with GVF an integer 0-100. Returns
	columns date, year, doy, gvf (dropping unparseable rows), mirroring the shape
	of the PhenoCam frame so it can flow through the same plotting/scoring code.
	"""
	frame = pd.read_csv(txt_path, header=None, names=["date", "gvf"], dtype={"date": str})
	frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="coerce")
	frame["gvf"] = pd.to_numeric(frame["gvf"], errors="coerce")
	frame = frame.dropna(subset=["date", "gvf"]).sort_values("date")
	frame["year"] = frame["date"].dt.year
	frame["doy"] = frame["date"].dt.dayofyear
	return frame


PLACEHOLDER_VEG = "XX"


def normalize_site_id(raw: str) -> str:
	"""Clean a GVF-derived site id: collapse repeated `_` and trim stray `_`.

	GVF filenames sometimes carry an empty separator field (``ibp__SH_1000``) or
	a trailing underscore (``NEON.D14.JORN.DP1.00033_GR_1000_``) that keep the id
	from matching a PhenoCam roi_name. Collapsing runs of underscores to one and
	stripping leading/trailing underscores restores the canonical
	``<site>_<VEG>_<seq>`` form (PhenoCam separators are single underscores).
	"""
	return re.sub(r"_+", "_", raw).strip("_")


def veg_code(site_id: str) -> str | None:
	"""Vegetation code from a `<site>_<VEG>_<seq>` id, or None if absent.

	Returns the second-to-last underscore token (``ibp_SH_1000`` -> ``SH``). Bare
	site ids without a veg/seq suffix (e.g. ``NEON.D01.BART.DP1.00033``) return
	None so they still resolve through the ROI map.
	"""
	parts = site_id.split("_")
	return parts[-2] if len(parts) >= 3 else None


def gvf_site_id(txt_path: str | Path) -> str:
	"""Site id from a GVF filename (drops the `.ops_GVF...` suffix).

	Also normalizes stray underscores (double/trailing) so the id matches a
	PhenoCam roi_name (see normalize_site_id).
	"""
	name = Path(txt_path).name
	return normalize_site_id(name.split(".ops_")[0])


def bare_site_from_roi(roi_name: str) -> str:
	"""Strip the `_VEG_ROI` suffix from a roi_name to get the bare site id."""
	return roi_name.rsplit("_", 2)[0]


def load_gbov_roi_map(path: str | Path = GBOV_CLEAN_JSON) -> dict[str, str]:
	"""Map each GBOV bare site id to its full PhenoCam roi_name.

	Reads site_GBOV_clean.json (all year buckets) so a GVF file named for the
	bare site (e.g. NEON.D01.BART.DP1.00033) resolves to the ROI that carries the
	GCC/NDVI curves (NEON.D01.BART.DP1.00033_DB_1000). Sites missing from the
	JSON (e.g. STER) simply won't appear and fall back to a bare site lookup.
	"""
	data = json.loads(Path(path).read_text())
	mapping: dict[str, str] = {}
	for bucket in data.values():
		if not isinstance(bucket, dict):
			continue
		for site in bucket.get("sites", []):
			roi_name = site.get("name")
			if roi_name:
				mapping.setdefault(bare_site_from_roi(roi_name), roi_name)
	return mapping


def _infer_year(folder_name: str) -> int:
	"""Pull the 4 digit year out of a folder name like GBOV_2023."""
	match = re.search(r"(\d{4})", folder_name)
	if not match:
		raise ValueError(f"Cannot infer year from folder name {folder_name!r}; pass year=")
	return int(match.group(1))


def plot_satellite(
	txt_path: str | Path,
	year: int,
	output_dir: str | Path,
	rois: list[dict] | None = None,
	roi_map: dict[str, str] | None = None,
) -> Path:
	"""Plot one GVF file overlaid with its PhenoCam GCC/NDVI; returns the PNG path.

	Resolves the GVF site to a PhenoCam ROI (via site_GBOV_clean.json, falling
	back to a bare site lookup), fetches the NDVI/GCC 3 day series, computes
	GVF vs GCC and GVF vs NDVI agreement scores, and saves `<roi_name>_<year>.png`.
	Pass prefetched `rois`/`roi_map` to avoid re downloading per call.
	"""
	txt_path = Path(txt_path)
	output_dir = Path(output_dir)
	bare = gvf_site_id(txt_path)

	veg = veg_code(bare)
	if veg is not None and veg.upper() == PLACEHOLDER_VEG:
		raise ValueError(f"{bare}: no veg code ({PLACEHOLDER_VEG} placeholder); skipping")

	roi_map = roi_map if roi_map is not None else load_gbov_roi_map()
	roi_name = roi_map.get(bare, bare)  # fall back to the bare site (resolve_roi -> find_roi)

	gvf = load_gvf(txt_path)
	if year not in {int(y) for y in gvf["year"].dropna().unique()}:
		raise ValueError(f"{bare}: GVF file has no {year} data")

	roi = resolve_roi(roi_name, rois=rois)
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))
	available = {int(y) for y in timeseries["year"].dropna().unique()}
	if year not in available:
		raise ValueError(f"{roi['roi_name']} has no {year} PhenoCam data (available: {sorted(available)})")

	year_pc = timeseries.loc[timeseries["year"] == year]
	year_gvf = gvf.loc[gvf["year"] == year]

	gvf_vs_gcc = pairwise_agreement(
		year_gvf["doy"].values, year_gvf["gvf"].values, year_gvf["date"].values,
		year_pc["doy"].values, year_pc["gcc_90"].values, year_pc["date"].values,
		year,
	)
	gvf_vs_ndvi = pairwise_agreement(
		year_gvf["doy"].values, year_gvf["gvf"].values, year_gvf["date"].values,
		year_pc["doy"].values, year_pc["ndvi_90"].values, year_pc["date"].values,
		year,
	)

	title = f"Satellite GVF vs PhenoCam GCC_90 & NDVI_90 ({roi['roi_name']}, {year})"
	fig, gcc_phases, ndvi_phases, gvf_phases = create_satellite_plot(
		timeseries, gvf, year, title, gvf_vs_gcc=gvf_vs_gcc, gvf_vs_ndvi=gvf_vs_ndvi
	)

	output_dir.mkdir(parents=True, exist_ok=True)
	output_file = output_dir / f"{roi['roi_name']}_{year}.png"
	save_plot(fig, output_file)

	print(f"Saved {output_file}")
	print(f"  GVF phases (DOY, {year}):  {gvf_phases}")
	print(f"  GCC phases (DOY, {year}):  {gcc_phases}")
	print(f"  NDVI phases (DOY, {year}): {ndvi_phases}")
	return output_file


def plot_satellite_folder(
	input_dir: str | Path,
	output_dir: str | Path,
	year: int | None = None,
	limit: int | None = None,
) -> list[Path]:
	"""Plot every GVF text file in a GBOV_<year> folder.

	The year is inferred from the folder name (GBOV_2023 -> 2023) unless given.
	Fetches the ROI list and GBOV ROI map once, then renders each
	`*_GVF*_timeseries.txt`; entries that fail (e.g. no PhenoCam data for that
	year) are skipped with a message. `limit` caps how many files are plotted.
	Returns the list of written PNG paths.
	"""
	input_dir = Path(input_dir)
	if year is None:
		year = _infer_year(input_dir.name)

	files = sorted(input_dir.glob("*_GVF*_timeseries.txt"))
	if limit is not None:
		files = files[:limit]

	rois = list_rois()  # fetch once and reuse for every ROI resolve
	roi_map = load_gbov_roi_map()
	written: list[Path] = []
	for txt in files:
		try:
			written.append(plot_satellite(txt, year, output_dir, rois=rois, roi_map=roi_map))
		except Exception as error:
			print(f"  skip {txt.name}: {error}")

	print(f"\nPlotted {len(written)}/{len(files)} GVF site years to {output_dir}")
	return written
