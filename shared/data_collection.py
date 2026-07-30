"""Collect GVF / GCC / NDVI agreement scores into a local CSV table.

Scores are recomputed from the GVF text files + PhenoCam fetch (same path as
shared.plot_satellite), not scraped from PNG pixels. Each row holds SOS/MOS/
DOS/EOS (DOY) for GVF, GCC, and NDVI, plus GVF-vs-GCC, GVF-vs-NDVI, and
GCC-vs-NDVI divergence / phenophase gap / DTW-per-step so you can sort, group,
and rank sites later with pandas.

Used by validation_pipeline/main.py via ``--table GBOV_2023``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .DynamicTimeWrap import pairwise_agreement, site_agreement_score
from .PhenoloDates import compute_phases
from .phenocam_api import fetch_ndvi_3day_for_roi, list_rois, load_timeseries
from .plot_json import resolve_roi
from .plot_satellite import (
	_infer_year,
	gvf_site_id,
	load_gbov_roi_map,
	load_gvf,
)

PHASE_KEYS = ("SOS", "MOS", "DOS", "EOS")

SCORE_COLUMNS = (
	"gvf_vs_gcc_div",
	"gvf_vs_gcc_gap",
	"gvf_vs_gcc_dtw",
	"gvf_vs_ndvi_div",
	"gvf_vs_ndvi_gap",
	"gvf_vs_ndvi_dtw",
	"gcc_vs_ndvi_div",
	"gcc_vs_ndvi_gap",
	"gcc_vs_ndvi_dtw",
)


def _short_site(roi_name: str) -> str:
	"""Short display name: NEON ... BART ... -> BART; arkansaswhitaker_AG_1000 -> arkansaswhitaker."""
	bare = roi_name.rsplit("_", 2)[0]
	if bare.startswith("NEON."):
		parts = bare.split(".")
		return parts[2] if len(parts) >= 3 else bare
	return bare


def _veg_from_roi(roi_name: str) -> str:
	"""Vegetation type token from a roi_name (e.g. DB from ..._DB_1000)."""
	parts = roi_name.rsplit("_", 2)
	return parts[1] if len(parts) >= 3 else ""


def _pack(prefix: str, scores: dict) -> dict[str, float]:
	return {
		f"{prefix}_div": scores.get("divergence_score"),
		f"{prefix}_gap": scores.get("phenophase_gap_days"),
		f"{prefix}_dtw": scores.get("dtw_per_step"),
	}


def _pack_phases(prefix: str, phases: dict) -> dict[str, float]:
	"""Flat DOY columns: gvf_sos, gcc_mos, ndvi_eos, etc."""
	return {f"{prefix}_{phase.lower()}": phases.get(phase) for phase in PHASE_KEYS}


def score_one(
	txt_path: str | Path,
	year: int,
	rois: list[dict] | None = None,
	roi_map: dict[str, str] | None = None,
) -> dict:
	"""Compute pairwise scores and SOS/MOS/DOS/EOS for GVF, GCC, and NDVI."""
	txt_path = Path(txt_path)
	bare = gvf_site_id(txt_path)
	roi_map = roi_map if roi_map is not None else load_gbov_roi_map()
	roi_name = roi_map.get(bare, bare)

	gvf = load_gvf(txt_path)
	if year not in {int(y) for y in gvf["year"].dropna().unique()}:
		raise ValueError(f"{bare}: GVF file has no {year} data")

	roi = resolve_roi(roi_name, rois=rois)
	timeseries = load_timeseries(fetch_ndvi_3day_for_roi(roi))
	available = {int(y) for y in timeseries["year"].dropna().unique()}
	if year not in available:
		raise ValueError(
			f"{roi['roi_name']} has no {year} PhenoCam data (available: {sorted(available)})"
		)

	year_pc = timeseries.loc[timeseries["year"] == year]
	year_gvf = gvf.loc[gvf["year"] == year]
	roi_full = roi["roi_name"]

	gvf_phases = compute_phases(year_gvf["doy"].values, year_gvf["gvf"].values)
	gcc_phases = compute_phases(year_pc["doy"].values, year_pc["gcc_90"].values)
	ndvi_phases = compute_phases(year_pc["doy"].values, year_pc["ndvi_90"].values)

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
	gcc_vs_ndvi = site_agreement_score(timeseries, year)

	row = {
		"site": _short_site(roi_full),
		"roi": roi_full,
		"veg": _veg_from_roi(roi_full),
		"year": year,
	}
	row.update(_pack_phases("gvf", gvf_phases))
	row.update(_pack_phases("gcc", gcc_phases))
	row.update(_pack_phases("ndvi", ndvi_phases))
	row.update(_pack("gvf_vs_gcc", gvf_vs_gcc))
	row.update(_pack("gvf_vs_ndvi", gvf_vs_ndvi))
	row.update(_pack("gcc_vs_ndvi", gcc_vs_ndvi))
	return row


def collect_folder(
	folder: str | Path,
	input_dir: str | Path,
	output_dir: str | Path,
	year: int | None = None,
	limit: int | None = None,
) -> Path:
	"""Score every GVF file in one input folder and write a CSV table.

	`folder` may be a bare name under `input_dir` (e.g. GBOV_2023) or a path.
	Writes `output_dir/<folder_name>/<folder_name>_scores.csv` and returns that
	path. Year is inferred from the folder name unless given. Failures are
	skipped with a message (same style as plot_satellite_folder).
	"""
	folder = Path(folder)
	input_dir = Path(input_dir)
	output_dir = Path(output_dir)

	src = folder if folder.is_dir() else input_dir / folder
	if not src.is_dir():
		raise FileNotFoundError(f"No folder at {src}")

	files = sorted(src.glob("*_GVF*_timeseries.txt"))
	if not files:
		raise FileNotFoundError(f"No *_GVF*_timeseries.txt files in {src}")
	if limit is not None:
		files = files[:limit]

	if year is None:
		year = _infer_year(src.name)

	rois = list_rois()
	roi_map = load_gbov_roi_map()
	rows: list[dict] = []
	for txt in files:
		try:
			row = score_one(txt, year, rois=rois, roi_map=roi_map)
			rows.append(row)

			def _fmt(v) -> str:
				return f"{v:.2f}" if v is not None and pd.notna(v) else "n/a"

			print(
				f"  {row['site']} ({row['veg']}, {year}): "
				f"GVF-GCC div={_fmt(row['gvf_vs_gcc_div'])}  "
				f"GVF-NDVI div={_fmt(row['gvf_vs_ndvi_div'])}  "
				f"GCC-NDVI div={_fmt(row['gcc_vs_ndvi_div'])}"
			)
		except Exception as error:
			print(f"  skip {txt.name}: {error}")

	frame = pd.DataFrame(rows)
	if not frame.empty:
		# lowest GVF-NDVI divergence first (best agreement at top)
		frame = frame.sort_values("gvf_vs_ndvi_div", ascending=True, na_position="last")
		frame = frame.reset_index(drop=True)

	dest_dir = output_dir / src.name
	dest_dir.mkdir(parents=True, exist_ok=True)
	csv_path = dest_dir / f"{src.name}_scores.csv"
	frame.to_csv(csv_path, index=False)
	print(f"\nWrote {len(frame)}/{len(files)} rows to {csv_path}")
	return csv_path


def load_table(csv_path: str | Path) -> pd.DataFrame:
	"""Load a scores CSV written by collect_folder."""
	return pd.read_csv(csv_path)


def top_n(
	df: pd.DataFrame,
	by: str = "gvf_vs_ndvi_div",
	n: int = 10,
	ascending: bool = True,
) -> pd.DataFrame:
	"""Return the top `n` rows sorted by column `by` (default: best = lowest div)."""
	if by not in df.columns:
		raise KeyError(f"Unknown column {by!r}; expected one of {list(df.columns)}")
	ordered = df.sort_values(by, ascending=ascending, na_position="last")
	return ordered.head(n).reset_index(drop=True)


def group_summary(df: pd.DataFrame, by: str = "veg") -> pd.DataFrame:
	"""Mean divergence / gap / DTW per group (default: vegetation type)."""
	metrics = [c for c in SCORE_COLUMNS if c in df.columns]
	if by not in df.columns:
		raise KeyError(f"Unknown group column {by!r}")
	return df.groupby(by, dropna=False)[metrics].mean().reset_index()
