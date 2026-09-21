"""Collect GVF / GCC / NDVI agreement scores into a local CSV table.

Scores are recomputed from the GVF text files + PhenoCam fetch (same path as
shared.plot_satellite), not scraped from PNG pixels. Each row holds SOS/MOS/
DOS/EOS (DOY) for GVF, GCC, and NDVI, plus GVF-vs-GCC, GVF-vs-NDVI, and
GCC-vs-NDVI divergence / phenophase gap / DTW-per-step so you can sort, group,
and rank sites later with pandas.

Used by anomaly_pipeline/anomaly_check.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .DynamicTimeWrap import pairwise_agreement, site_agreement_score
from .PhenoloDates import compute_phases
from .phenocam_api import fetch_ndvi_3day_for_roi, list_rois, load_timeseries
from .plot_json import resolve_roi
from .plot_satellite import (
	PLACEHOLDER_VEG,
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


def _round4(value: float | None) -> float | None:
	return None if value is None else round(float(value), 4)


def _add_lag_compression(row: dict) -> dict:
	"""Add phase lags (GVF−NDVI) plus greenup/senescence compression ratios.

	``lag`` / ``lag_sos`` = gvf_sos − ndvi_sos (kept as ``lag`` for older callers).
	Also ``lag_mos``, ``lag_dos``, ``lag_eos``. Compression ratios to 4 decimals.
	"""
	gvf_sos = row.get("gvf_sos")
	ndvi_sos = row.get("ndvi_sos")
	gvf_mos = row.get("gvf_mos")
	ndvi_mos = row.get("ndvi_mos")
	gcc_sos = row.get("gcc_sos")
	gcc_mos = row.get("gcc_mos")
	gvf_dos = row.get("gvf_dos")
	ndvi_dos = row.get("ndvi_dos")
	gvf_eos = row.get("gvf_eos")
	ndvi_eos = row.get("ndvi_eos")
	gcc_dos = row.get("gcc_dos")
	gcc_eos = row.get("gcc_eos")

	def _diff(a, b):
		if a is None or b is None:
			return None
		return float(a) - float(b)

	lag_sos = _diff(gvf_sos, ndvi_sos)
	lag_mos = _diff(gvf_mos, ndvi_mos)
	lag_dos = _diff(gvf_dos, ndvi_dos)
	lag_eos = _diff(gvf_eos, ndvi_eos)

	# GCC-referenced lags (gvf_* − gcc_*), alongside the NDVI-referenced ones above
	lag_sos_gcc = _diff(gvf_sos, gcc_sos)
	lag_mos_gcc = _diff(gvf_mos, gcc_mos)
	lag_dos_gcc = _diff(gvf_dos, gcc_dos)
	lag_eos_gcc = _diff(gvf_eos, gcc_eos)

	greenup_comp = None
	if (
		gvf_sos is not None and gvf_mos is not None
		and gcc_sos is not None and gcc_mos is not None
	):
		gcc_greenup = float(gcc_mos) - float(gcc_sos)
		if gcc_greenup != 0:
			greenup_comp = (float(gvf_mos) - float(gvf_sos)) / gcc_greenup

	senescence_comp = None
	if (
		gvf_dos is not None and gvf_eos is not None
		and gcc_dos is not None and gcc_eos is not None
	):
		gcc_sen = float(gcc_eos) - float(gcc_dos)
		if gcc_sen != 0:
			senescence_comp = (float(gvf_eos) - float(gvf_dos)) / gcc_sen

	# NDVI-referenced compression (GVF phase length / NDVI phase length)
	greenup_comp_ndvi = None
	if (
		gvf_sos is not None and gvf_mos is not None
		and ndvi_sos is not None and ndvi_mos is not None
	):
		ndvi_greenup = float(ndvi_mos) - float(ndvi_sos)
		if ndvi_greenup != 0:
			greenup_comp_ndvi = (float(gvf_mos) - float(gvf_sos)) / ndvi_greenup

	senescence_comp_ndvi = None
	if (
		gvf_dos is not None and gvf_eos is not None
		and ndvi_dos is not None and ndvi_eos is not None
	):
		ndvi_sen = float(ndvi_eos) - float(ndvi_dos)
		if ndvi_sen != 0:
			senescence_comp_ndvi = (float(gvf_eos) - float(gvf_dos)) / ndvi_sen

	# ``lag`` kept as SOS alias for ranking / older notebook cells
	row["lag"] = _round4(lag_sos)
	row["lag_sos"] = _round4(lag_sos)
	row["lag_mos"] = _round4(lag_mos)
	row["lag_dos"] = _round4(lag_dos)
	row["lag_eos"] = _round4(lag_eos)
	row["lag_sos_gcc"] = _round4(lag_sos_gcc)
	row["lag_mos_gcc"] = _round4(lag_mos_gcc)
	row["lag_dos_gcc"] = _round4(lag_dos_gcc)
	row["lag_eos_gcc"] = _round4(lag_eos_gcc)
	row["greenup_comp"] = _round4(greenup_comp)
	row["senescence_comp"] = _round4(senescence_comp)
	row["greenup_comp_ndvi"] = _round4(greenup_comp_ndvi)
	row["senescence_comp_ndvi"] = _round4(senescence_comp_ndvi)
	return row


def _order_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
	"""Put lag / compression columns immediately after ``site``."""
	preferred = [
		"site",
		"lag", "lag_sos", "lag_mos", "lag_dos", "lag_eos",
		"lag_sos_gcc", "lag_mos_gcc", "lag_dos_gcc", "lag_eos_gcc",
		"greenup_comp", "senescence_comp",
		"greenup_comp_ndvi", "senescence_comp_ndvi",
		"roi", "veg", "year",
	]
	front = [c for c in preferred if c in frame.columns]
	rest = [c for c in frame.columns if c not in front]
	return frame[front + rest]


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
	return _add_lag_compression(row)


def collect_folder(
	folder: str | Path,
	input_dir: str | Path,
	anomaly_dir: str | Path,
	year: int | None = None,
	limit: int | None = None,
) -> Path:
	"""Score every GVF file in one input folder and write a CSV table.

	`folder` may be a bare name under `input_dir` (e.g. GBOV_2023) or a path.
	Writes ``anomaly_dir/<folder_name>/scores.csv`` (per-folder layout, mirroring
	``plotting_pipeline/output/<folder>/``) and returns that path. Year is
	inferred from the folder name unless given. Failures are skipped with a
	message (same style as plot_satellite_folder).
	"""
	folder = Path(folder)
	input_dir = Path(input_dir)

	src = folder if folder.is_dir() else input_dir / folder
	if not src.is_dir():
		raise FileNotFoundError(f"No folder at {src}")

	out_dir = folder_output_dir(anomaly_dir, src.name)

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
		frame = _order_score_columns(frame)

	out_dir.mkdir(parents=True, exist_ok=True)
	csv_path = out_dir / "scores.csv"
	frame.to_csv(csv_path, index=False)
	print(f"\nWrote {len(frame)}/{len(files)} rows to {csv_path}")
	return csv_path


def enrich_scores_frame(df: pd.DataFrame) -> pd.DataFrame:
	"""Ensure phase lags + compression columns exist (4 decimals) after ``site``."""
	frame = df.copy()
	# migrate old column names if present
	if "greenup_comp" not in frame.columns and "compression" in frame.columns:
		frame = frame.rename(columns={"compression": "greenup_comp"})
	if "senescence_comp" not in frame.columns and "senescence" in frame.columns:
		frame = frame.rename(columns={"senescence": "senescence_comp"})
	frame = frame.drop(columns=[c for c in ("compression", "senescence") if c in frame.columns], errors="ignore")

	phase_lags = [
		("lag_sos", "gvf_sos", "ndvi_sos"),
		("lag_mos", "gvf_mos", "ndvi_mos"),
		("lag_dos", "gvf_dos", "ndvi_dos"),
		("lag_eos", "gvf_eos", "ndvi_eos"),
		("lag_sos_gcc", "gvf_sos", "gcc_sos"),
		("lag_mos_gcc", "gvf_mos", "gcc_mos"),
		("lag_dos_gcc", "gvf_dos", "gcc_dos"),
		("lag_eos_gcc", "gvf_eos", "gcc_eos"),
	]
	for out_col, gvf_col, ref_col in phase_lags:
		if out_col not in frame.columns and {gvf_col, ref_col} <= set(frame.columns):
			frame[out_col] = frame[gvf_col] - frame[ref_col]
	# ``lag`` = SOS alias
	if "lag_sos" in frame.columns:
		frame["lag"] = frame["lag_sos"]
	elif "lag" not in frame.columns and {"gvf_sos", "ndvi_sos"} <= set(frame.columns):
		frame["lag"] = frame["gvf_sos"] - frame["ndvi_sos"]
		frame["lag_sos"] = frame["lag"]

	if "greenup_comp" not in frame.columns and {"gvf_sos", "gvf_mos", "gcc_sos", "gcc_mos"} <= set(frame.columns):
		gcc_greenup = frame["gcc_mos"] - frame["gcc_sos"]
		frame["greenup_comp"] = (frame["gvf_mos"] - frame["gvf_sos"]) / gcc_greenup.replace(0, pd.NA)
	if "senescence_comp" not in frame.columns and {"gvf_dos", "gvf_eos", "gcc_dos", "gcc_eos"} <= set(frame.columns):
		gcc_sen = frame["gcc_eos"] - frame["gcc_dos"]
		frame["senescence_comp"] = (frame["gvf_eos"] - frame["gvf_dos"]) / gcc_sen.replace(0, pd.NA)
	if "greenup_comp_ndvi" not in frame.columns and {"gvf_sos", "gvf_mos", "ndvi_sos", "ndvi_mos"} <= set(frame.columns):
		ndvi_greenup = frame["ndvi_mos"] - frame["ndvi_sos"]
		frame["greenup_comp_ndvi"] = (frame["gvf_mos"] - frame["gvf_sos"]) / ndvi_greenup.replace(0, pd.NA)
	if "senescence_comp_ndvi" not in frame.columns and {"gvf_dos", "gvf_eos", "ndvi_dos", "ndvi_eos"} <= set(frame.columns):
		ndvi_sen = frame["ndvi_eos"] - frame["ndvi_dos"]
		frame["senescence_comp_ndvi"] = (frame["gvf_eos"] - frame["gvf_dos"]) / ndvi_sen.replace(0, pd.NA)

	for col in (
		"lag", "lag_sos", "lag_mos", "lag_dos", "lag_eos",
		"lag_sos_gcc", "lag_mos_gcc", "lag_dos_gcc", "lag_eos_gcc",
		"greenup_comp", "senescence_comp",
		"greenup_comp_ndvi", "senescence_comp_ndvi",
	):
		if col in frame.columns:
			frame[col] = pd.to_numeric(frame[col], errors="coerce").round(4)
	return _order_score_columns(frame)


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


def load_all_scores(anomaly_dir: str | Path) -> pd.DataFrame:
	"""Concatenate every per-folder ``<folder>/scores.csv`` under the output dir.

	New per-folder layout is ``anomaly_dir/<folder>/scores.csv`` (source = folder
	name). Legacy ``anomaly_dir/metadata/<folder>_scores.csv`` files are still
	picked up for back-compat; the ``_combined`` folder is ignored.
	"""
	base = _output_root(anomaly_dir)
	frames = []
	seen: set[str] = set()

	for path in sorted(base.glob("*/scores.csv")):
		source = path.parent.name
		if source == COMBINED_DIRNAME:
			continue
		frame = enrich_scores_frame(pd.read_csv(path))
		frame["source"] = source
		frames.append(frame)
		seen.add(source)

	legacy_meta = base / "metadata"
	if legacy_meta.is_dir():
		for path in sorted(legacy_meta.glob("*_scores.csv")):
			source = path.stem.removesuffix("_scores")
			if source in seen:
				continue
			frame = enrich_scores_frame(pd.read_csv(path))
			frame["source"] = source
			frames.append(frame)
			seen.add(source)

	if not frames:
		raise FileNotFoundError(f"No <folder>/scores.csv (or legacy metadata) under {base}/")
	return pd.concat(frames, ignore_index=True)


def anomaly_output_dir(start: str | Path | None = None) -> Path:
	"""``anomaly_pipeline/output`` under the repo root.

	``start`` may be the repo, ``anomaly_pipeline``, ``output``, or any
	path under the repo (walks parents until ``shared/`` is found).
	"""
	p = Path(start).resolve() if start is not None else Path.cwd().resolve()
	for candidate in [p, *p.parents]:
		if (candidate / "shared").is_dir():
			return candidate / "anomaly_pipeline" / "output"
		if candidate.name == "output" and candidate.parent.name == "anomaly_pipeline":
			return candidate
	raise FileNotFoundError(f"Could not locate repo root from {p}")


def metadata_dir(anomaly_dir: str | Path | None = None) -> Path:
	"""``anomaly_pipeline/output/metadata`` (legacy scores CSV location)."""
	base = Path(anomaly_dir) if anomaly_dir is not None else anomaly_output_dir()
	# accept either output or already-metadata
	if base.name == "metadata":
		return base
	return Path(base) / "metadata"


COMBINED_DIRNAME = "_combined"


def _output_root(anomaly_dir: str | Path | None = None) -> Path:
	"""Resolve the ``anomaly_pipeline/output`` dir from various inputs.

	Accepts the output dir itself, its ``metadata`` subfolder, or None (auto).
	"""
	base = Path(anomaly_dir) if anomaly_dir is not None else anomaly_output_dir()
	if base.name in ("metadata", COMBINED_DIRNAME):
		base = base.parent
	return base


def folder_output_dir(anomaly_dir: str | Path | None, folder: str | Path) -> Path:
	"""Per-folder output dir ``anomaly_pipeline/output/<folder>/``.

	Mirrors ``plotting_pipeline/output/<folder>/``: every input folder gets one
	subfolder holding all of its artifacts (scores + plots).
	"""
	name = Path(folder).name
	return _output_root(anomaly_dir) / name


def combined_output_dir(anomaly_dir: str | Path | None = None) -> Path:
	"""Cross-folder output dir ``anomaly_pipeline/output/_combined/``.

	Holds artifacts that span every folder (golden ranking, combined
	year-vs-year lag/compression, effect-size tests).
	"""
	return _output_root(anomaly_dir) / COMBINED_DIRNAME


def build_golden_ranking(
	anomaly_dir: str | Path,
	out_csv: str | Path | None = None,
	exclude_spinup: bool = True,
) -> Path:
	"""Rank site-years by combined GVF gap+DTW; mark DB golden-standard candidates.

	Combined score = mean of GVF-vs-GCC and GVF-vs-NDVI divergence
	(each divergence is already phenophase_gap / 14 + dtw_per_step). Spin-up
	sites (``gvf_sos == 1``) are dropped by default. DB rows get
	``golden_candidate=True`` as the closed-canopy control-group pool.

	Reads scores from every ``anomaly_dir/<folder>/scores.csv``. Writes
	``anomaly_dir/_combined/golden_standard_ranking.csv`` unless ``out_csv`` is
	given.
	"""
	anomaly_dir = Path(anomaly_dir)
	df = load_all_scores(anomaly_dir)
	df["spin_up"] = df["gvf_sos"].eq(1.0) if "gvf_sos" in df.columns else False

	ranked = df.copy()
	if exclude_spinup:
		ranked = ranked.loc[~ranked["spin_up"]].copy()

	ranked["combined_div"] = ranked[["gvf_vs_gcc_div", "gvf_vs_ndvi_div"]].mean(axis=1)
	ranked["combined_gap"] = ranked[["gvf_vs_gcc_gap", "gvf_vs_ndvi_gap"]].mean(axis=1)
	ranked["combined_dtw"] = ranked[["gvf_vs_gcc_dtw", "gvf_vs_ndvi_dtw"]].mean(axis=1)
	ranked["golden_candidate"] = ranked["veg"].eq("DB")

	ranked = ranked.sort_values("combined_div", ascending=True, na_position="last")
	ranked = ranked.reset_index(drop=True)
	ranked.insert(0, "rank", ranked.index + 1)

	cols = [
		"rank", "site", "lag", "lag_sos", "lag_mos", "lag_dos", "lag_eos",
		"greenup_comp", "senescence_comp", "roi", "veg", "year", "source",

		"golden_candidate", "spin_up",
		"combined_div", "combined_gap", "combined_dtw",
		"gvf_vs_ndvi_div", "gvf_vs_ndvi_gap", "gvf_vs_ndvi_dtw",
		"gvf_vs_gcc_div", "gvf_vs_gcc_gap", "gvf_vs_gcc_dtw",
		"gcc_vs_ndvi_div", "gcc_vs_ndvi_gap", "gcc_vs_ndvi_dtw",
		"gvf_sos", "gvf_mos", "gvf_dos", "gvf_eos",
		"gcc_sos", "gcc_mos", "gcc_dos", "gcc_eos",
		"ndvi_sos", "ndvi_mos", "ndvi_dos", "ndvi_eos",
	]
	cols = [c for c in cols if c in ranked.columns]
	ranked = ranked[cols]

	out_csv = (
		Path(out_csv)
		if out_csv is not None
		else combined_output_dir(anomaly_dir) / "golden_standard_ranking.csv"
	)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	ranked.to_csv(out_csv, index=False)

	n_db = int(ranked["golden_candidate"].sum()) if "golden_candidate" in ranked.columns else 0
	print(f"Ranked {len(ranked)} site-years ({n_db} DB golden candidates); excluded spin-up={exclude_spinup}")
	print(f"Wrote {out_csv}")
	return out_csv


def plot_gap_boxplot_by_veg(
	scores_csv: str | Path,
	anomaly_dir: str | Path,
	out_png: str | Path | None = None,
	title: str | None = None,
) -> Path:
	"""Boxplot of ``gvf_vs_ndvi_gap`` by veg; spin-up (``gvf_sos == 1``) as red diamonds.

	Writes ``boxplot.png`` next to the scores CSV (i.e. in that folder's output
	dir ``anomaly_dir/<folder>/``) by default. The folder name for the title is
	taken from the scores CSV's parent directory.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import numpy as np

	scores_csv = Path(scores_csv)
	df = pd.read_csv(scores_csv)
	if "gvf_sos" not in df.columns or "gvf_vs_ndvi_gap" not in df.columns:
		raise ValueError(f"{scores_csv} needs gvf_sos and gvf_vs_ndvi_gap columns")

	# drop the XX placeholder (no vegetation code) from the plot
	df = df[df["veg"].astype(str).str.upper() != PLACEHOLDER_VEG]

	spin = df[df["gvf_sos"] == 1.0]
	normal = df[df["gvf_sos"] != 1.0]
	folder = _folder_label(scores_csv)
	if out_png is None:
		out_png = scores_csv.parent / "boxplot.png"
	out_png = Path(out_png)
	out_png.parent.mkdir(parents=True, exist_ok=True)

	if len(normal):
		veg_order = (
			normal.groupby("veg")["gvf_vs_ndvi_gap"].median().sort_values().index.tolist()
		)
	else:
		veg_order = []
	for v in sorted(df["veg"].dropna().unique()):
		if v not in veg_order:
			veg_order.append(v)

	fig, ax = plt.subplots(figsize=(9, 5.5))
	positions = np.arange(1, len(veg_order) + 1)
	data = [normal.loc[normal["veg"] == v, "gvf_vs_ndvi_gap"].dropna().values for v in veg_order]
	ax.boxplot(
		data,
		positions=positions,
		widths=0.55,
		patch_artist=True,
		showfliers=False,
		medianprops={"color": "#1a1a1a", "linewidth": 1.5},
		whiskerprops={"color": "#555555"},
		capprops={"color": "#555555"},
		boxprops={"facecolor": "#a6cee3", "edgecolor": "#555555", "alpha": 0.85},
	)

	rng = np.random.default_rng(42)
	other_labeled = spin_labeled = False
	for i, v in enumerate(veg_order):
		x0 = positions[i]
		n = normal.loc[normal["veg"] == v]
		s = spin.loc[spin["veg"] == v]
		if len(n):
			ax.scatter(
				x0 + rng.uniform(-0.12, 0.12, size=len(n)),
				n["gvf_vs_ndvi_gap"],
				s=36, c="#333333", zorder=3,
				label="other sites" if not other_labeled else None,
				edgecolors="white", linewidths=0.4,
			)
			other_labeled = True
		if len(s):
			ax.scatter(
				x0 + rng.uniform(-0.12, 0.12, size=len(s)),
				s["gvf_vs_ndvi_gap"],
				s=70, c="#e41a1c", zorder=4, marker="D",
				label="spin-up (GVF SOS = DOY 1)" if not spin_labeled else None,
				edgecolors="white", linewidths=0.6,
			)
			spin_labeled = True

	ax.legend(
		handles=[
			Line2D([0], [0], marker="o", color="w", markerfacecolor="#333333", markersize=8, label="other sites"),
			Line2D([0], [0], marker="D", color="w", markerfacecolor="#e41a1c", markersize=9, label="spin-up (GVF SOS = DOY 1)"),
		],
		loc="upper left",
		frameon=True,
	)
	veg_counts = [int((df["veg"] == v).sum()) for v in veg_order]
	ax.set_xticks(positions)
	ax.set_xticklabels([f"{v}\n(n={n})" for v, n in zip(veg_order, veg_counts)])
	ax.set_xlabel("Vegetation type")
	ax.set_ylabel("GVF vs NDVI phenophase gap (days)")
	ax.set_title(title or f"{folder}: GVF-NDVI gap by land type (n_spinup={len(spin)})")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_png, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


# ---------------------------------------------------------------------------
# Per-folder plots (lollipop + divergence) and the build_folder_artifacts driver
#
# These mirror the notebook helpers but are headless (Agg, no IPython) so a
# single call can render every plot for one input folder into
# ``anomaly_pipeline/output/<folder>/``.
# ---------------------------------------------------------------------------

DIV_COLS = ("gvf_vs_gcc_div", "gvf_vs_ndvi_div", "gcc_vs_ndvi_div")
DIV_LABELS = {
	"gvf_vs_gcc_div": "GVF vs GCC",
	"gvf_vs_ndvi_div": "GVF vs NDVI",
	"gcc_vs_ndvi_div": "GCC vs NDVI",
}
DIV_COLORS = ("#4C78A8", "#F58518", "#54A24B")


def _folder_label(scores_csv: str | Path) -> str:
	"""Folder name for a scores CSV: parent dir for ``scores.csv``, else the stem."""
	scores_csv = Path(scores_csv)
	if scores_csv.name == "scores.csv":
		return scores_csv.parent.name
	return scores_csv.stem.removesuffix("_scores")


def _lollipop(ax, labels, values, color, ref_line=None):
	"""Horizontal lollipop chart (cleaner than thick bars for ranked site values)."""
	import numpy as np

	y = np.arange(len(labels))
	vals = np.asarray(values, dtype=float)
	ax.hlines(y, 0 if ref_line is None else ref_line, vals, color=color, alpha=0.55, linewidth=1.4)
	ax.scatter(vals, y, color=color, s=42, zorder=3, edgecolors="white", linewidths=0.4)
	ax.set_yticks(y)
	ax.set_yticklabels(labels, fontsize=7)
	if ref_line is not None:
		ax.axvline(ref_line, color="black", linestyle="--", linewidth=1, alpha=0.75)
	ax.grid(True, axis="x", alpha=0.3)


# Phase-lag reference definitions: which columns/colour/label each reference uses.
_LAG_REFS = {
	"ndvi": {
		"cols": {"SOS": "lag_sos", "MOS": "lag_mos", "DOS": "lag_dos", "EOS": "lag_eos"},
		"color": "#4C78A8", "label": "GVF - NDVI",
	},
	"gcc": {
		"cols": {"SOS": "lag_sos_gcc", "MOS": "lag_mos_gcc", "DOS": "lag_dos_gcc", "EOS": "lag_eos_gcc"},
		"color": "#F58518", "label": "GVF - GCC",
	},
}
_LAG_PHASE_ORDER = ["SOS", "MOS", "DOS", "EOS"]


def plot_lag_lollipop(
	df: pd.DataFrame, out_png: str | Path, title_prefix: str, refs=("ndvi",),
	shared_xlim: bool = False,
) -> Path:
	"""Lag by phase (SOS/MOS/DOS/EOS) x veg as lollipops.

	``refs`` selects the reference series: ``("ndvi",)`` for gvf-ndvi (default),
	``("gcc",)`` for gvf-gcc, or ``("ndvi", "gcc")`` to overlay both per site.
	Sites are sorted by the first reference; each reference is drawn in its own
	colour on the same site rows so the two references line up.

	``shared_xlim=True`` gives every panel the same symmetric x-axis (global
	max |lag|), so biomes/phases are directly comparable ("research" view).
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import numpy as np

	out_png = Path(out_png)
	refs = [r for r in refs if r in _LAG_REFS] or ["ndvi"]
	primary = refs[0]
	plot_df = df.copy()
	if "lag_sos" not in plot_df.columns and "lag" in plot_df.columns:
		plot_df["lag_sos"] = plot_df["lag"]

	def any_ref_col(phase):
		return [
			_LAG_REFS[r]["cols"][phase] for r in refs
			if _LAG_REFS[r]["cols"][phase] in plot_df.columns
			and plot_df[_LAG_REFS[r]["cols"][phase]].notna().any()
		]

	phases = [p for p in _LAG_PHASE_ORDER if any_ref_col(p)] or _LAG_PHASE_ORDER
	vegs = sorted(plot_df["veg"].dropna().unique())
	n_veg = max(len(vegs), 1)
	max_n = 4
	for phase in phases:
		for col in any_ref_col(phase):
			max_n = max(max_n, int(plot_df.dropna(subset=[col]).groupby("veg").size().max()))

	global_xlim = None
	if shared_xlim:
		allabs = []
		for phase in phases:
			for col in any_ref_col(phase):
				allabs.append(pd.to_numeric(plot_df[col], errors="coerce").dropna().abs())
		lim = float(pd.concat(allabs).max()) if allabs else 1.0
		global_xlim = (-lim * 1.05, lim * 1.05)

	fig, axes = plt.subplots(
		len(phases), n_veg,
		figsize=(5.8 * n_veg, max(3.4, 0.32 * max_n + 1.6) * len(phases)),
		sharex=False, squeeze=False,
	)

	for row_i, phase in enumerate(phases):
		for col_i, veg in enumerate(vegs):
			ax = axes[row_i][col_i]
			sub = plot_df.loc[plot_df["veg"].eq(veg)].dropna(subset=["site"]).copy()
			# order rows by the primary reference (fall back to any available)
			order_col = _LAG_REFS[primary]["cols"][phase]
			if order_col not in sub.columns or sub[order_col].notna().sum() == 0:
				avail = any_ref_col(phase)
				order_col = next((c for c in avail if c in sub.columns and sub[c].notna().any()), None)
				if order_col is None:
					ax.set_visible(False)
					continue
			sub = sub.dropna(subset=[order_col]).sort_values(order_col, ascending=True)
			if sub.empty:
				ax.set_visible(False)
				continue
			y = np.arange(len(sub))
			for r in refs:
				rc = _LAG_REFS[r]["cols"][phase]
				if rc not in sub.columns:
					continue
				vals = pd.to_numeric(sub[rc], errors="coerce").values
				color = _LAG_REFS[r]["color"]
				ax.hlines(y, 0, vals, color=color, alpha=0.45, linewidth=1.2)
				ax.scatter(vals, y, color=color, s=36, zorder=3, edgecolors="white", linewidths=0.4)
			ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.75)
			ax.set_yticks(y)
			ax.set_yticklabels(list(sub["site"]), fontsize=7)
			ax.grid(True, axis="x", alpha=0.3)
			ax.tick_params(axis="y", pad=6)
			if global_xlim is not None:
				ax.set_xlim(*global_xlim)
			if row_i == 0:
				ax.set_title(f"{veg} (n={len(sub)})")
			ax.set_xlabel(f"{phase} lag (days)")

		right_ax = axes[row_i][-1]
		right_ax.text(
			1.28, 0.95, phase, transform=right_ax.transAxes,
			ha="left", va="top", fontsize=12, fontweight="bold", clip_on=False,
		)

	ref_desc = " & ".join(_LAG_REFS[r]["label"] for r in refs)
	fig.suptitle(f"{title_prefix} — Phase lag ({ref_desc}) by veg", y=0.995)
	fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.22, hspace=0.75, wspace=1.15)
	ref_handles = [
		Line2D([0], [0], marker="o", color="w", markerfacecolor=_LAG_REFS[r]["color"],
			markersize=9, label=_LAG_REFS[r]["label"])
		for r in refs
	]
	fig.legend(
		handles=ref_handles + [
			Line2D([0], [0], color="none", label="lag = gvf_phase − reference_phase"),
			Line2D([0], [0], color="none", label="+ : GVF later   0 : same DOY   − : GVF earlier"),
			Line2D([0], [0], color="none", label=f"sites sorted by {_LAG_REFS[primary]['label']}"),
			Line2D([0], [0], color="none", label="|lag| guide: ~0–15 typical | 15–40 look | 40+ candidate"),
		],
		loc="upper center", bbox_to_anchor=(0.47, 0.18), ncol=1, frameon=True, fontsize=8,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.5)
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


_LAG_PHASES = [("lag_sos", "SOS"), ("lag_mos", "MOS"), ("lag_dos", "DOS"), ("lag_eos", "EOS")]


def _lag_matrix(df: pd.DataFrame, vegs: list, phases=_LAG_PHASES):
	"""Mean phase-lag matrix (rows=veg, cols=phase) from a scored frame."""
	import numpy as np

	mat = np.full((len(vegs), len(phases)), np.nan)
	for i, veg in enumerate(vegs):
		sub = df.loc[df["veg"].eq(veg)]
		for j, (col, _) in enumerate(phases):
			if col in sub.columns and sub[col].notna().any():
				mat[i, j] = sub[col].mean()
	return mat


def _annotate_heatmap(ax, mat, vmax):
	import numpy as np

	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			v = mat[i, j]
			if np.isnan(v):
				continue
			ax.text(
				j, i, f"{v:+.0f}", ha="center", va="center", fontsize=9,
				color="white" if abs(v) > 0.6 * vmax else "black",
			)


def _heatmap_grid(ax, n_rows, n_cols):
	import numpy as np

	ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
	ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
	ax.grid(which="minor", color="white", linewidth=1.5)
	ax.tick_params(which="minor", length=0)


def plot_lag_heatmap(
	df: pd.DataFrame, out_png: str | Path, title_prefix: str, min_n: int = 5
) -> Path:
	"""Compact veg x phase mean-lag heatmap (cross-biome pattern at a glance).

	Rows = biomes (>= ``min_n`` site-years, ``XX`` dropped) sorted by sample size;
	columns = SOS/MOS/DOS/EOS. Colour = mean ``lag_* = gvf_* - ndvi_*`` (days),
	diverging about 0 (blue = GVF earlier, red = GVF later). Spin-up should already
	be excluded upstream.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	counts = df.dropna(subset=["veg"]).groupby("veg").size()
	vegs = [v for v in counts.index if counts[v] >= min_n and str(v).upper() != "XX"]
	vegs = sorted(vegs, key=lambda v: counts[v], reverse=True) or sorted(counts.index)

	mat = _lag_matrix(df, vegs)
	finite = np.abs(mat[np.isfinite(mat)])
	vmax = float(min(max(finite.max() if finite.size else 1.0, 1.0), 60.0))

	fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(vegs) + 2.0))
	im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
	ax.set_xticks(range(len(_LAG_PHASES)))
	ax.set_xticklabels([p for _, p in _LAG_PHASES], fontsize=10)
	ax.set_yticks(range(len(vegs)))
	ax.set_yticklabels([f"{v} (n={int(counts[v])})" for v in vegs], fontsize=9)
	_annotate_heatmap(ax, mat, vmax)
	_heatmap_grid(ax, len(vegs), len(_LAG_PHASES))
	ax.set_title(f"{title_prefix} - mean phase lag by biome (days)", fontsize=11, fontweight="bold")
	ax.set_xlabel("phenophase", fontsize=9)
	cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
	cbar.set_label("mean lag (days):  + GVF later  /  - GVF earlier", fontsize=8)
	fig.text(
		0.5, 0.01,
		"Spin-up (DOY-1) excluded. Rows sorted by sample size. "
		"Blue = GVF earlier than NDVI, red = GVF later.",
		ha="center", fontsize=7.5, style="italic",
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_lag_heatmap_years(
	scores_by_year: dict, out_png: str | Path, min_n: int = 5
) -> Path:
	"""Side-by-side veg x phase mean-lag heatmaps per year on a shared colour scale."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	years = sorted(scores_by_year)
	counts = {y: scores_by_year[y].dropna(subset=["veg"]).groupby("veg").size() for y in years}
	total: dict = {}
	for y in years:
		for v, c in counts[y].items():
			total[v] = total.get(v, 0) + int(c)
	vegs = [
		v for v in total
		if str(v).upper() != "XX" and any(counts[y].get(v, 0) >= min_n for y in years)
	]
	vegs = sorted(vegs, key=lambda v: total[v], reverse=True)

	mats = {y: _lag_matrix(scores_by_year[y], vegs) for y in years}
	vmax = 1.0
	for y in years:
		finite = np.abs(mats[y][np.isfinite(mats[y])])
		if finite.size:
			vmax = max(vmax, float(finite.max()))
	vmax = min(vmax, 60.0)

	fig, axes = plt.subplots(
		1, len(years), figsize=(4.9 * len(years), 0.6 * len(vegs) + 2.2), squeeze=False,
	)
	im = None
	for k, y in enumerate(years):
		ax = axes[0][k]
		im = ax.imshow(mats[y], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
		ax.set_xticks(range(len(_LAG_PHASES)))
		ax.set_xticklabels([p for _, p in _LAG_PHASES], fontsize=10)
		ax.set_yticks(range(len(vegs)))
		if k == 0:
			ax.set_yticklabels(vegs, fontsize=9)
		else:
			ax.set_yticklabels([])
		_annotate_heatmap(ax, mats[y], vmax)
		_heatmap_grid(ax, len(vegs), len(_LAG_PHASES))
		ax.set_title(str(y), fontsize=12, fontweight="bold")

	fig.suptitle("Mean phase lag by biome (GVF - NDVI, days) - 2023 vs 2024", fontsize=12, fontweight="bold")
	cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.03)
	cbar.set_label("mean lag (days):  + GVF later  /  - GVF earlier", fontsize=8)
	fig.text(
		0.5, 0.01,
		"Spin-up (DOY-1) excluded. Blue = GVF earlier than NDVI, red = GVF later. "
		"Read across a row to see how a biome's lag changes SOS->EOS.",
		ha="center", fontsize=8, style="italic",
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def _lag_split_stats(df: pd.DataFrame, vegs: list, ref: str):
	"""Per veg x phase: (mean, n) of positive-lag and negative-lag sites separately."""
	import numpy as np

	cols = _LAG_REFS[ref]["cols"]
	out = {}
	for p in _LAG_PHASE_ORDER:
		col = cols[p]
		pos_mean, pos_n, neg_mean, neg_n = [], [], [], []
		for v in vegs:
			s = pd.to_numeric(df.loc[df["veg"].eq(v), col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)
			pos, neg = s[s > 0], s[s < 0]
			pos_mean.append(pos.mean() if len(pos) else np.nan)
			pos_n.append(int(len(pos)))
			neg_mean.append(neg.mean() if len(neg) else np.nan)
			neg_n.append(int(len(neg)))
		out[p] = (pos_mean, pos_n, neg_mean, neg_n)
	return out


def _draw_lag_split_grid(ax, vegs, data, counts, vmax=60.0, title=None):
	"""Draw a biome x (phase x sign) grid: red = GVF later, blue = GVF earlier."""
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.patches import Rectangle
	from matplotlib.colors import Normalize

	phases = _LAG_PHASE_ORDER
	reds, blues = plt.cm.Reds, plt.cm.Blues
	norm = Normalize(0, vmax)
	n_rows, ncol = len(vegs), len(phases) * 2

	for i, veg in enumerate(vegs):
		yy = n_rows - 1 - i
		for j, p in enumerate(phases):
			pos_mean, pos_n, neg_mean, neg_n = data[p]
			cells = [(pos_mean[i], pos_n[i], reds), (neg_mean[i], neg_n[i], blues)]
			for k, (mean, n, cmap) in enumerate(cells):
				x = 2 * j + k
				if n and not np.isnan(mean):
					inten = float(norm(min(abs(mean), vmax)))
					face = cmap(0.15 + 0.85 * inten)
					txtcol = "white" if inten > 0.55 else "black"
					label = f"n={n}\n{mean:+.0f}d"
				else:
					face, txtcol, label = "#f2f2f2", "#c0c0c0", "n=0"
				ax.add_patch(Rectangle((x, yy), 1, 1, facecolor=face, edgecolor="white", lw=1.3))
				ax.text(x + 0.5, yy + 0.5, label, ha="center", va="center", fontsize=7, color=txtcol)

	ax.set_xlim(0, ncol)
	ax.set_ylim(0, n_rows + 0.6)
	# phase name centred over each pair, + / - sub-headers
	for j, p in enumerate(phases):
		ax.text(2 * j + 1.0, n_rows + 0.45, p, ha="center", va="bottom", fontsize=11, fontweight="bold")
		ax.text(2 * j + 0.5, n_rows + 0.02, "+later", ha="center", va="bottom", fontsize=6.5, color="#b2182b")
		ax.text(2 * j + 1.5, n_rows + 0.02, "-early", ha="center", va="bottom", fontsize=6.5, color="#2166ac")
	ax.set_xticks([])
	ax.set_yticks([n_rows - 1 - i + 0.5 for i in range(n_rows)])
	ax.set_yticklabels([f"{v} (n={counts.get(v, 0)})" for v in vegs], fontsize=9)
	ax.tick_params(length=0)
	for spine in ax.spines.values():
		spine.set_visible(False)
	if title:
		ax.set_title(title, fontsize=11, fontweight="bold", pad=24)


def plot_lag_split_heatmap(
	df: pd.DataFrame, out_png: str | Path, title_prefix: str,
	ref: str = "ndvi", min_n: int = 5, vmax: float = 60.0,
) -> Path:
	"""Sign-split lag heatmap: each biome x phase shows the +lag group and the
	-lag group separately (count + mean each), so a single large outlier only
	colours its own side instead of saturating a shared diverging scale.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_png = Path(out_png)
	counts = df.dropna(subset=["veg"]).groupby("veg").size()
	vegs = [v for v in counts.index if counts[v] >= min_n and str(v).upper() != "XX"]
	vegs = sorted(vegs, key=lambda v: counts[v], reverse=True) or sorted(counts.index)
	data = _lag_split_stats(df, vegs, ref)

	fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(vegs) + 2.2))
	_draw_lag_split_grid(
		ax, vegs, data, {v: int(counts[v]) for v in vegs}, vmax=vmax,
		title=f"{title_prefix} - phase lag split by sign ({_LAG_REFS[ref]['label']})",
	)
	fig.text(
		0.5, 0.015,
		"Each phase is split: red '+later' = sites where GVF is LATER than the reference "
		"(count n, mean of those positives); blue '-early' = sites where GVF is EARLIER. "
		f"Spin-up excluded. Colour intensity ∝ |mean| (capped {int(vmax)} d); outliers stay on their own side.",
		ha="center", fontsize=8, style="italic", wrap=True,
	)
	fig.tight_layout(rect=[0, 0.05, 1, 1])
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_lag_split_heatmap_years(
	scores_by_year: dict, out_png: str | Path,
	ref: str = "ndvi", min_n: int = 5, vmax: float = 60.0,
) -> Path:
	"""Side-by-side sign-split lag heatmaps per year on a shared colour scale."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	out_png = Path(out_png)
	years = sorted(scores_by_year)
	counts = {y: scores_by_year[y].dropna(subset=["veg"]).groupby("veg").size() for y in years}
	total: dict = {}
	for y in years:
		for v, c in counts[y].items():
			total[v] = total.get(v, 0) + int(c)
	vegs = [
		v for v in total
		if str(v).upper() != "XX" and any(counts[y].get(v, 0) >= min_n for y in years)
	]
	vegs = sorted(vegs, key=lambda v: total[v], reverse=True)

	fig, axes = plt.subplots(
		1, len(years), figsize=(9.0 * len(years), 0.62 * len(vegs) + 2.4), squeeze=False,
	)
	for k, y in enumerate(years):
		data = _lag_split_stats(scores_by_year[y], vegs, ref)
		_draw_lag_split_grid(
			axes[0][k], vegs, data, {v: int(counts[y].get(v, 0)) for v in vegs},
			vmax=vmax, title=str(y),
		)
	fig.suptitle(
		f"Phase lag split by sign ({_LAG_REFS[ref]['label']}) - 2023 vs 2024",
		fontsize=13, fontweight="bold",
	)
	fig.text(
		0.5, 0.015,
		"red '+later' = GVF later than reference (n, mean); blue '-early' = GVF earlier. "
		f"Colour ∝ |mean| (cap {int(vmax)} d). A single outlier only shades its own +/- side.",
		ha="center", fontsize=8.5, style="italic",
	)
	fig.tight_layout(rect=[0, 0.05, 1, 0.96])
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


_REF_DOT_COLORS = {"ndvi": "#4C78A8", "gcc": "#F58518"}


def _dumbbell_legend(fig, refs):
	from matplotlib.patches import Patch

	handles = [
		Patch(facecolor=_REF_DOT_COLORS.get(r, "#555"), edgecolor="white",
			label=_LAG_REFS[r]["label"])
		for r in refs
	]
	fig.legend(handles=handles, loc="lower center", ncol=len(refs),
		bbox_to_anchor=(0.5, -0.005), frameon=True, fontsize=9)


def _draw_lag_boxes(ax, vegs, year_df, refs, xcap, phase):
	"""One phase panel: box plot of per-site lag per biome for each reference.

	Each biome gets one box per reference (GVF-NDVI, GVF-GCC), offset vertically.
	Box = IQR, line = median, whiskers = 1.5x IQR, dots = outlier sites; colour =
	reference; left of 0 = GVF earlier, right = GVF later.
	"""
	import numpy as np

	n_ref = max(len(refs), 1)
	offsets = np.linspace(0.2, -0.2, n_ref) if n_ref > 1 else [0.0]
	width = 0.66 / n_ref if n_ref > 1 else 0.5
	for k, r in enumerate(refs):
		col = _LAG_REFS[r]["cols"][phase]
		color = _REF_DOT_COLORS.get(r, "#555")
		data, pos = [], []
		for i, v in enumerate(vegs):
			s = (
				pd.to_numeric(year_df.loc[year_df["veg"].eq(v), col], errors="coerce").dropna()
				if col in year_df.columns else pd.Series(dtype=float)
			)
			if not len(s):
				continue
			data.append(np.clip(s.to_numpy(dtype=float), -xcap, xcap))
			pos.append(i + offsets[k])
		if not data:
			continue
		bp = ax.boxplot(
			data, positions=pos, widths=width, vert=False, patch_artist=True,
			showfliers=True,
			flierprops=dict(marker="o", markersize=2.6, markerfacecolor=color,
				markeredgecolor="none", alpha=0.6),
			medianprops=dict(color="black", lw=1.0),
			whiskerprops=dict(color=color, lw=1.0), capprops=dict(color=color, lw=1.0),
			boxprops=dict(edgecolor="white", lw=0.6), zorder=3,
		)
		for patch in bp["boxes"]:
			patch.set_facecolor(color)
			patch.set_alpha(0.75)
	ax.axvline(0, color="black", lw=0.9, zorder=2)
	ax.set_xlim(-(xcap + 8), xcap + 8)
	ax.grid(True, axis="x", alpha=0.25)


def plot_lag_reference_box(
	df: pd.DataFrame, out_png: str | Path, title_prefix: str,
	refs=("ndvi", "gcc"), min_n: int = 5, xcap: float = 90.0,
) -> Path:
	"""Distribution of phase lag vs NDVI and GCC as box plots (one panel per phase).

	Each biome shows the full spread of per-site lags as a box per reference, so
	you can see medians, IQR and asymmetry (outliers appear as dots).
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	refs = [r for r in refs if r in _LAG_REFS] or ["ndvi", "gcc"]
	counts = df.dropna(subset=["veg"]).groupby("veg").size()
	vegs = [v for v in counts.index if counts[v] >= min_n and str(v).upper() != "XX"]
	vegs = sorted(vegs, key=lambda v: counts[v], reverse=True) or sorted(counts.index)
	phases = _LAG_PHASE_ORDER

	fig, axes = plt.subplots(
		1, len(phases), figsize=(4.7 * len(phases), 0.62 * len(vegs) + 2.2), sharey=True,
	)
	y = np.arange(len(vegs))
	for ax, p in zip(axes, phases):
		_draw_lag_boxes(ax, vegs, df, refs, xcap, p)
		ax.set_title(p, fontsize=12, fontweight="bold")
		ax.set_xlabel("lag (days)", fontsize=8)
	axes[0].set_yticks(y)
	axes[0].set_yticklabels([f"{v} (n={int(counts[v])})" for v in vegs], fontsize=9)
	axes[0].set_ylim(-0.6, len(vegs) - 0.4)
	axes[0].invert_yaxis()

	fig.suptitle(f"{title_prefix} - phase lag distribution: GVF-NDVI vs GVF-GCC", fontsize=12, fontweight="bold")
	_dumbbell_legend(fig, refs)
	fig.text(
		0.5, 0.03,
		"Box = IQR of per-site lag, line = median, whiskers = 1.5x IQR, dots = outlier sites. "
		f"Left of 0 = GVF earlier, right = GVF later (clipped +/-{int(xcap)} d). "
		"NDVI = blue, GCC = orange; n per biome on the axis. Spin-up excluded.",
		ha="center", fontsize=8, style="italic",
	)
	fig.tight_layout(rect=[0, 0.06, 1, 0.96])
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_lag_reference_box_years(
	scores_by_year: dict, out_png: str | Path,
	refs=("ndvi", "gcc"), min_n: int = 5, xcap: float = 90.0,
) -> Path:
	"""GVF-NDVI vs GVF-GCC lag box plots, one row of phase panels per year."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	refs = [r for r in refs if r in _LAG_REFS] or ["ndvi", "gcc"]
	years = sorted(scores_by_year)
	counts = {y: scores_by_year[y].dropna(subset=["veg"]).groupby("veg").size() for y in years}
	total: dict = {}
	for y in years:
		for v, c in counts[y].items():
			total[v] = total.get(v, 0) + int(c)
	vegs = [
		v for v in total
		if str(v).upper() != "XX" and any(counts[y].get(v, 0) >= min_n for y in years)
	]
	vegs = sorted(vegs, key=lambda v: total[v], reverse=True)
	phases = _LAG_PHASE_ORDER

	fig, axes = plt.subplots(
		len(years), len(phases),
		figsize=(4.7 * len(phases), (0.55 * len(vegs) + 1.6) * len(years)),
		sharey=True, squeeze=False,
	)
	y = np.arange(len(vegs))
	for r, yr in enumerate(years):
		for c, p in enumerate(phases):
			ax = axes[r][c]
			_draw_lag_boxes(ax, vegs, scores_by_year[yr], refs, xcap, p)
			if r == 0:
				ax.set_title(p, fontsize=12, fontweight="bold")
			if r == len(years) - 1:
				ax.set_xlabel("lag (days)", fontsize=8)
		axes[r][0].set_ylabel(str(yr), fontsize=12, fontweight="bold")
		axes[r][0].set_yticks(y)
		axes[r][0].set_yticklabels([f"{v} (n={int(counts[yr].get(v, 0))})" for v in vegs], fontsize=8)
	axes[0][0].set_ylim(-0.6, len(vegs) - 0.4)
	axes[0][0].invert_yaxis()

	fig.suptitle("Phase lag distribution: GVF-NDVI vs GVF-GCC - 2023 vs 2024", fontsize=13, fontweight="bold")
	_dumbbell_legend(fig, refs)
	fig.text(
		0.5, 0.02,
		"Box = IQR of per-site lag, line = median, whiskers = 1.5x IQR, dots = outlier sites. "
		f"Left = GVF earlier, right = GVF later (clipped +/-{int(xcap)} d). NDVI = blue, GCC = orange; n per biome on the axis.",
		ha="center", fontsize=8.5, style="italic",
	)
	fig.tight_layout(rect=[0, 0.04, 1, 0.965])
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


# Site metadata (lat/lon + surface fractions) from the uniformity pipeline, used to
# describe the box-plot lag outliers.
_META_FILES = [
	"uniformity_pipeline/output/Full/pipeline_results.json",
	"uniformity_pipeline/output/Full_loose/pipeline_results.json",
]
_OUTLIER_DIV_COLS = {
	"gvf_vs_ndvi_div": "GVF-NDVI div",
	"gcc_vs_ndvi_div": "GCC-NDVI div",
	"gvf_vs_gcc_div": "GVF-GCC div",
}


def load_site_metadata(meta_files=None) -> dict:
	"""``roi name`` -> {lat, lon, water_frac, urban_frac} from the uniformity pipeline."""
	import json

	root = Path(__file__).resolve().parents[1]
	files = meta_files or [root / p for p in _META_FILES]
	meta: dict = {}
	for f in files:
		f = Path(f)
		if not f.exists():
			continue
		for s in json.load(open(f)).get("sites", []):
			sp = s.get("spatial") or {}
			meta.setdefault(s["name"], {
				"lat": s.get("lat"), "lon": s.get("lon"),
				"water_frac": sp.get("water_pct"), "urban_frac": sp.get("urban_pct"),
			})
	return meta


def find_lag_box_outliers(clean: pd.DataFrame, min_n: int = 5, xcap: float = 90.0) -> pd.DataFrame:
	"""Sites beyond the Tukey 1.5x IQR whisker per veg x phase x reference (the box dots).

	Reproduces the fence used by :func:`plot_lag_reference_box`; one row per
	(site, reference, phase) outlier flag.
	"""
	import numpy as np

	counts = clean.dropna(subset=["veg"]).groupby("veg").size()
	vegs = [v for v in counts.index if counts[v] >= min_n and str(v).upper() != "XX"]
	rows = []
	for ref in ("ndvi", "gcc"):
		for p in _LAG_PHASE_ORDER:
			col = _LAG_REFS[ref]["cols"][p]
			if col not in clean.columns:
				continue
			for v in vegs:
				sub = clean[clean["veg"].eq(v)]
				s = pd.to_numeric(sub[col], errors="coerce")
				arr = np.clip(s.dropna().to_numpy(float), -xcap, xcap)
				if len(arr) < min_n:
					continue
				q1, q3 = np.percentile(arr, [25, 75])
				iqr = q3 - q1
				lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
				mask = (s < lo) | (s > hi)
				for idx in sub.index[mask.reindex(sub.index, fill_value=False)]:
					rows.append({
						"site": sub.at[idx, "site"], "roi": sub.at[idx, "roi"], "veg": v,
						"reference": ref.upper(), "phase": p, "lag_days": float(s.at[idx]),
						"fence_lo": round(float(lo), 1), "fence_hi": round(float(hi), 1),
					})
	return pd.DataFrame(rows)


def _lookup_site_meta(meta: dict, roi, site) -> dict:
	if roi in meta:
		return meta[roi]
	for k, v in meta.items():
		if k.split("_")[0] == str(site):
			return v
	return {"lat": None, "lon": None, "water_frac": None, "urban_frac": None}


def write_lag_box_outliers_by_site(
	scores_by_year: dict, out_csv: str | Path,
	meta: dict | None = None, min_n: int = 5, xcap: float = 90.0,
) -> Path:
	"""Write a single cross-year table of ``lag_direction_box.png`` outlier sites.

	One row per site that is a Tukey 1.5x IQR flier in any year. The ``flagged_in``
	column groups the flags per year, e.g. ``2024:{NDVI:MOS(-75), GCC:DOS(-69)}
	2023:{}``. Static site context (lat/lon, water/urban fraction) plus the three
	pairwise divergences (per year) are included. Written to ``out_csv`` (meant to
	live in the combined folder, outside the per-year folders).
	"""
	meta = load_site_metadata() if meta is None else meta
	years = sorted(scores_by_year)

	static: dict = {}
	year_flags: dict = {}   # site -> {year: "NDVI:MOS(-75), ..."}
	year_counts: dict = {}  # site -> {year: n}
	year_div: dict = {}     # site -> {year: {divcol: val}}
	year_phases: dict = {}  # site -> {year: {gvf_sos, gcc_mos, ...}}
	for year in years:
		clean = scores_by_year[year]
		detail = find_lag_box_outliers(clean, min_n=min_n, xcap=xcap)
		for site, g in detail.groupby("site"):
			roi = g["roi"].iloc[0]
			info = _lookup_site_meta(meta, roi, site)
			static.setdefault(site, {
				"site": site, "veg": g["veg"].iloc[0], "roi": roi,
				"lat": info["lat"], "lon": info["lon"],
				"water_frac": info["water_frac"], "urban_frac": info["urban_frac"],
			})
			year_flags.setdefault(site, {})[year] = ", ".join(
				f"{r['reference']}:{r['phase']}({r['lag_days']:+.0f})" for _, r in g.iterrows()
			)
			year_counts.setdefault(site, {})[year] = len(g)
			score_row = clean.loc[clean["roi"].eq(roi)]
			if score_row.empty:
				score_row = clean.loc[clean["site"].eq(site)]
			score_row = score_row.iloc[0] if len(score_row) else None
			divs = {}
			for col in _OUTLIER_DIV_COLS:
				val = score_row[col] if (score_row is not None and col in score_row.index) else None
				divs[col] = float(val) if val is not None and pd.notna(val) else None
			year_div.setdefault(site, {})[year] = divs
			year_phases.setdefault(site, {})[year] = _phase_columns(clean, site, roi)

	rows = []
	for site, base in static.items():
		row = dict(base)
		counts = year_counts.get(site, {})
		row["total_flags"] = sum(counts.values())
		for y in years:
			row[f"flags_{y}"] = counts.get(y, 0)
		# newest year first, always show every year (empty braces when none)
		row["flagged_in"] = " ".join(
			f"{y}:{{{year_flags.get(site, {}).get(y, '')}}}" for y in sorted(years, reverse=True)
		)
		if "phenocam_site" not in row:
			row["phenocam_site"] = _phenocam_sitename(base.get("roi"))
			row["roi"] = base.get("roi")
		for y in years:
			for col in _OUTLIER_DIV_COLS:
				row[f"{col}_{y}"] = year_div.get(site, {}).get(y, {}).get(col)
			phases = year_phases.get(site, {}).get(y) or _phase_columns(
				scores_by_year[y], site, base.get("roi"),
			)
			for prefix in _PHASE_SERIES:
				for p in PHASE_KEYS:
					col = f"{prefix}_{p.lower()}"
					row[f"{col}_{y}"] = phases.get(col)
		rows.append(row)

	out = pd.DataFrame(rows).sort_values(
		["veg", "total_flags", "site"], ascending=[True, False, True]
	).reset_index(drop=True)
	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(out_csv, index=False)
	print(f"Wrote {out_csv}  ({len(out)} sites across {', '.join(map(str, years))})")
	return out_csv


_PHASE_SERIES = ("gvf", "gcc", "ndvi")


def _phase_columns(clean: pd.DataFrame, site, roi=None) -> dict:
	"""SOS/MOS/DOS/EOS DOYs for GVF, GCC, and NDVI from a scores row."""
	score_row = clean.loc[clean["roi"].eq(roi)] if roi is not None else clean.iloc[0:0]
	if score_row.empty:
		score_row = clean.loc[clean["site"].eq(site)]
	score_row = score_row.iloc[0] if len(score_row) else None
	out = {"roi": roi}
	if score_row is None:
		for prefix in _PHASE_SERIES:
			for p in PHASE_KEYS:
				out[f"{prefix}_{p.lower()}"] = None
		return out
	if out["roi"] is None:
		out["roi"] = score_row["roi"] if "roi" in score_row.index else None
	for prefix in _PHASE_SERIES:
		for p in PHASE_KEYS:
			col = f"{prefix}_{p.lower()}"
			val = score_row[col] if col in score_row.index else None
			out[col] = float(val) if val is not None and pd.notna(val) else None
	return out


def _phenocam_sitename(roi) -> str | None:
	"""Camera sitename encoded in a roi_name (``site_VEG_seq`` -> ``site``)."""
	if roi is None or (isinstance(roi, float) and pd.isna(roi)):
		return None
	parts = str(roi).rsplit("_", 2)
	return parts[0] if len(parts) >= 3 else str(roi)


_COMP_DIFF_METRICS = (
	("greenup", "greenup_comp", "greenup_comp_ndvi"),
	("senescence", "senescence_comp", "senescence_comp_ndvi"),
)


def find_compression_ref_outliers(
	clean: pd.DataFrame, min_n: int = 5, abs_cap: float = 5.0,
) -> pd.DataFrame:
	"""Sites where GVF/GCC vs GVF/NDVI compression disagree a lot.

	Per vegetation x metric (green-up, senescence), compute
	``d = comp_gcc − comp_ndvi`` and flag a site if ``d`` is a Tukey 1.5x IQR
	outlier **or** ``|d| >= abs_cap`` (default 5: one reference is ≥5x more
	stretched/compressed than the other).
	"""
	import numpy as np

	counts = clean.dropna(subset=["veg"]).groupby("veg").size()
	vegs = [v for v in counts.index if counts[v] >= min_n and str(v).upper() != "XX"]
	rows = []
	for metric, gcc_col, ndvi_col in _COMP_DIFF_METRICS:
		if gcc_col not in clean.columns or ndvi_col not in clean.columns:
			continue
		for v in vegs:
			sub = clean[clean["veg"].eq(v)].copy()
			gcc = pd.to_numeric(sub[gcc_col], errors="coerce")
			ndvi = pd.to_numeric(sub[ndvi_col], errors="coerce")
			d = gcc - ndvi
			ok = d.dropna()
			if len(ok) < min_n:
				continue
			q1, q3 = np.percentile(ok, [25, 75])
			iqr = q3 - q1
			lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
			mask = (d < lo) | (d > hi) | (d.abs() >= abs_cap)
			for idx in sub.index[mask.reindex(sub.index, fill_value=False)]:
				if pd.isna(d.at[idx]):
					continue
				rows.append({
					"site": sub.at[idx, "site"], "roi": sub.at[idx, "roi"], "veg": v,
					"metric": metric,
					"comp_gcc": float(gcc.at[idx]),
					"comp_ndvi": float(ndvi.at[idx]),
					"delta": float(d.at[idx]),
					"fence_lo": round(float(lo), 2),
					"fence_hi": round(float(hi), 2),
				})
	return pd.DataFrame(rows)


def _fmt_comp_flag(r) -> str:
	return f"{r['metric']}(gcc={r['comp_gcc']:.2f},ndvi={r['comp_ndvi']:.2f},d={r['delta']:+.2f})"


def plot_compression_ref_outliers(
	scores_by_year: dict, out_png: str | Path,
	min_n: int = 5, abs_cap: float = 5.0, xcap: float = 12.0,
) -> Path:
	"""Bar chart of GCC−NDVI compression difference for flagged outlier sites."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	years = sorted(scores_by_year)
	metrics = [m[0] for m in _COMP_DIFF_METRICS]
	flagged = {y: find_compression_ref_outliers(scores_by_year[y], min_n=min_n, abs_cap=abs_cap) for y in years}
	# keep the figure readable: sort each panel by |delta|
	n_rows = 1
	for y in years:
		d = flagged[y]
		if len(d):
			n_rows = max(n_rows, int(d.groupby("metric").size().max()))
	fig, axes = plt.subplots(
		len(years), len(metrics),
		figsize=(7.4 * len(metrics), 0.28 * n_rows * len(years) + 2.4),
		squeeze=False,
	)
	gcc_c, ndvi_c = "#4C78A8", "#F58518"
	for r, yr in enumerate(years):
		detail = flagged[yr]
		for c, metric in enumerate(metrics):
			ax = axes[r][c]
			g = detail[detail["metric"].eq(metric)].copy() if len(detail) else detail
			if len(g):
				g = g.assign(abs_d=g["delta"].abs()).sort_values(
					["abs_d", "site"], ascending=[False, True],
				).drop_duplicates("site", keep="first").sort_values("abs_d", ascending=True)
			labels = [f"{row.site} ({row.veg})" for row in g.itertuples()] if len(g) else []
			y = np.arange(len(g))
			for i, row in enumerate(g.itertuples() if len(g) else []):
				val = float(np.clip(row.delta, -xcap, xcap))
				color = gcc_c if row.delta >= 0 else ndvi_c
				ax.barh(i, val, height=0.62, color=color, alpha=0.88, edgecolor="white")
				ha = "left" if val >= 0 else "right"
				ax.text(val + (0.15 if val >= 0 else -0.15), i,
					f"d={row.delta:+.1f}  gcc={row.comp_gcc:.1f}  ndvi={row.comp_ndvi:.1f}",
					va="center", ha=ha, fontsize=5.6)
			ax.axvline(0, color="black", lw=0.8)
			ax.set_xlim(-(xcap + 4.5), xcap + 4.5)
			ax.set_yticks(y)
			ax.set_yticklabels(labels, fontsize=6.2)
			ax.grid(True, axis="x", alpha=0.25)
			if r == 0:
				ax.set_title(metric, fontsize=12, fontweight="bold")
			if r == len(years) - 1:
				ax.set_xlabel("comp_gcc − comp_ndvi", fontsize=8)
			if c == 0:
				ax.set_ylabel(str(yr), fontsize=12, fontweight="bold")
	fig.suptitle(
		"Compression reference outliers: GVF/GCC vs GVF/NDVI (flagged if Tukey 1.5x IQR or |d|≥"
		f"{abs_cap:g})",
		fontsize=12, fontweight="bold",
	)
	fig.text(
		0.5, 0.012,
		"d = (GVF length / GCC length) − (GVF length / NDVI length). "
		"Blue = GCC ratio larger (more stretched vs GCC than vs NDVI); orange = NDVI ratio larger. "
		f"Bars clipped +/-{int(xcap)}; true d + both ratios labelled. Spin-up excluded.",
		ha="center", fontsize=8, style="italic",
	)
	fig.tight_layout(rect=[0, 0.04, 1, 0.96])
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def write_compression_ref_outliers_by_site(
	scores_by_year: dict, out_csv: str | Path,
	out_png: str | Path | None = None,
	meta: dict | None = None, min_n: int = 5, abs_cap: float = 5.0,
) -> dict[str, Path]:
	"""Write a cross-year table (+ optional chart) of GCC-vs-NDVI compression outliers.

	One row per site flagged in any year. ``flagged_in`` groups flags per year, e.g.
	``2024:{greenup(gcc=14.06,ndvi=1.05,d=+13.01)} 2023:{}``.
	"""
	meta = load_site_metadata() if meta is None else meta
	years = sorted(scores_by_year)

	static: dict = {}
	year_flags: dict = {}
	year_counts: dict = {}
	year_vals: dict = {}
	year_phases: dict = {}
	for year in years:
		clean = scores_by_year[year]
		detail = find_compression_ref_outliers(clean, min_n=min_n, abs_cap=abs_cap)
		for site, g in detail.groupby("site"):
			roi = g["roi"].iloc[0]
			info = _lookup_site_meta(meta, roi, site)
			static.setdefault(site, {
				"site": site, "veg": g["veg"].iloc[0], "roi": roi,
				"lat": info["lat"], "lon": info["lon"],
				"water_frac": info["water_frac"], "urban_frac": info["urban_frac"],
			})
			year_flags.setdefault(site, {})[year] = ", ".join(_fmt_comp_flag(r) for _, r in g.iterrows())
			year_counts.setdefault(site, {})[year] = len(g)
			vals = {}
			for _, r in g.iterrows():
				vals[f"{r['metric']}_gcc"] = r["comp_gcc"]
				vals[f"{r['metric']}_ndvi"] = r["comp_ndvi"]
				vals[f"{r['metric']}_d"] = r["delta"]
			year_vals.setdefault(site, {})[year] = vals
			year_phases.setdefault(site, {})[year] = _phase_columns(clean, site, roi)

	rows = []
	for site, base in static.items():
		row = dict(base)
		counts = year_counts.get(site, {})
		row["total_flags"] = sum(counts.values())
		for y in years:
			row[f"flags_{y}"] = counts.get(y, 0)
		row["flagged_in"] = " ".join(
			f"{y}:{{{year_flags.get(site, {}).get(y, '')}}}" for y in sorted(years, reverse=True)
		)
		if "phenocam_site" not in row:
			row["phenocam_site"] = _phenocam_sitename(base.get("roi"))
			row["roi"] = base.get("roi")
		for y in years:
			vals = year_vals.get(site, {}).get(y, {})
			for metric, _, _ in _COMP_DIFF_METRICS:
				row[f"{metric}_gcc_{y}"] = vals.get(f"{metric}_gcc")
				row[f"{metric}_ndvi_{y}"] = vals.get(f"{metric}_ndvi")
				row[f"{metric}_d_{y}"] = vals.get(f"{metric}_d")
			phases = year_phases.get(site, {}).get(y) or _phase_columns(
				scores_by_year[y], site, base.get("roi"),
			)
			for prefix in _PHASE_SERIES:
				for p in PHASE_KEYS:
					col = f"{prefix}_{p.lower()}"
					row[f"{col}_{y}"] = phases.get(col)
		rows.append(row)

	out = pd.DataFrame(rows).sort_values(
		["veg", "total_flags", "site"], ascending=[True, False, True]
	).reset_index(drop=True)
	out_csv = Path(out_csv)
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(out_csv, index=False)
	print(f"Wrote {out_csv}  ({len(out)} sites across {', '.join(map(str, years))})")

	paths = {"csv": out_csv}
	if out_png is not None:
		paths["png"] = plot_compression_ref_outliers(
			scores_by_year, out_png, min_n=min_n, abs_cap=abs_cap,
		)
	return paths


# Compression reference definitions (ratio = GVF phase length / reference phase length).
_COMP_REFS = {
	"gcc": {
		"cols": {"greenup": "greenup_comp", "senescence": "senescence_comp"},
		"color": "#4C78A8", "label": "vs GCC",
	},
	"ndvi": {
		"cols": {"greenup": "greenup_comp_ndvi", "senescence": "senescence_comp_ndvi"},
		"color": "#54A24B", "label": "vs NDVI",
	},
}
_COMP_METRICS = [("greenup", "green-up"), ("senescence", "senescence")]


def plot_compression_lollipop(
	df: pd.DataFrame, out_png: str | Path, title_prefix: str, refs=("gcc",),
	shared_xlim: bool = False,
) -> Path:
	"""green-up | senescence compression lollipops per veg.

	``refs`` selects the reference: ``("gcc",)`` (default) for GVF/GCC ratios,
	``("ndvi",)`` for GVF/NDVI, or ``("gcc", "ndvi")`` to overlay both per site.
	ratio = (GVF phase length)/(reference phase length): 1 = same, <1 compressed,
	>1 stretched. ``shared_xlim=True`` uses one common x-axis on every panel.
	"""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import numpy as np

	out_png = Path(out_png)
	refs = [r for r in refs if r in _COMP_REFS] or ["gcc"]
	primary = refs[0]

	def any_ref_col(metric):
		return [
			_COMP_REFS[r]["cols"][metric] for r in refs
			if _COMP_REFS[r]["cols"][metric] in df.columns
			and df[_COMP_REFS[r]["cols"][metric]].notna().any()
		]

	vegs = sorted(df["veg"].dropna().unique())
	n_veg = max(len(vegs), 1)
	max_n = 4
	for metric, _ in _COMP_METRICS:
		for col in any_ref_col(metric):
			max_n = max(max_n, int(df.dropna(subset=[col]).groupby("veg").size().max()))

	global_xlim = None
	if shared_xlim:
		allvals = []
		for metric, _ in _COMP_METRICS:
			for col in any_ref_col(metric):
				allvals.append(pd.to_numeric(df[col], errors="coerce").dropna())
		if allvals:
			pool = pd.concat(allvals)
			lo, hi = min(0.0, float(pool.min())), float(pool.max())
			pad = 0.05 * (hi - lo if hi > lo else 1.0)
			global_xlim = (lo - pad, hi + pad)

	fig, axes = plt.subplots(
		n_veg, 2, figsize=(13, max(3.8, 0.40 * max_n + 1.8) * n_veg),
		sharex=False, squeeze=False,
	)
	for row, veg in enumerate(vegs):
		for col_i, (metric, mlabel) in enumerate(_COMP_METRICS):
			ax = axes[row][col_i]
			sub = df.loc[df["veg"].eq(veg)].dropna(subset=["site"]).copy()
			order_col = _COMP_REFS[primary]["cols"][metric]
			if order_col not in sub.columns or sub[order_col].notna().sum() == 0:
				avail = any_ref_col(metric)
				order_col = next((c for c in avail if c in sub.columns and sub[c].notna().any()), None)
				if order_col is None:
					ax.set_visible(False)
					continue
			sub = sub.dropna(subset=[order_col]).sort_values(order_col, ascending=True)
			if sub.empty:
				ax.set_visible(False)
				continue
			y = np.arange(len(sub))
			for r in refs:
				rc = _COMP_REFS[r]["cols"][metric]
				if rc not in sub.columns:
					continue
				vals = pd.to_numeric(sub[rc], errors="coerce").values
				color = _COMP_REFS[r]["color"]
				ax.hlines(y, 1, vals, color=color, alpha=0.45, linewidth=1.2)
				ax.scatter(vals, y, color=color, s=36, zorder=3, edgecolors="white", linewidths=0.4)
			ax.axvline(1, color="black", linestyle="--", linewidth=1, alpha=0.75)
			ax.set_yticks(y)
			ax.set_yticklabels(list(sub["site"]), fontsize=7)
			ax.grid(True, axis="x", alpha=0.3)
			ax.set_xlabel(f"{mlabel} compression ratio")
			ax.set_title(f"{veg} · {mlabel} (n={len(sub)})")
			if global_xlim is not None:
				ax.set_xlim(*global_xlim)

	ref_desc = " & ".join(_COMP_REFS[r]["label"] for r in refs)
	fig.suptitle(f"{title_prefix} — green-up (left) | senescence (right) compression [{ref_desc}]", y=1.01)
	ref_handles = [
		Line2D([0], [0], marker="o", color="w", markerfacecolor=_COMP_REFS[r]["color"],
			markersize=9, label=f"compression {_COMP_REFS[r]['label']}")
		for r in refs
	]
	fig.legend(
		handles=ref_handles + [
			Line2D([0], [0], color="none", label="ratio = (GVF phase length)/(reference phase length)"),
			Line2D([0], [0], color="none", label="= 1 same length | < 1 compressed | > 1 stretched"),
			Line2D([0], [0], color="none", label=f"sites sorted by {_COMP_REFS[primary]['label']}"),
		],
		loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=1, frameon=True, fontsize=8,
	)
	fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.97])
	fig.subplots_adjust(hspace=0.85, wspace=0.45)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.4)
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_divergence_bars_by_veg(df: pd.DataFrame, out_png: str | Path, title: str | None = None) -> Path:
	"""Grouped bars of mean GVF-GCC / GVF-NDVI / GCC-NDVI divergence per veg."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	vegs = sorted(df["veg"].dropna().unique())
	x = np.arange(len(vegs))
	width = 0.25

	fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(vegs) + 3), 4.8))
	for i, (col, color) in enumerate(zip(DIV_COLS, DIV_COLORS)):
		if col not in df.columns:
			continue
		means = [df.loc[df["veg"].eq(v), col].mean(skipna=True) for v in vegs]
		ax.bar(x + (i - 1) * width, means, width, label=DIV_LABELS[col], color=color)

	counts = [int(df["veg"].eq(v).sum()) for v in vegs]
	ax.set_xticks(x)
	ax.set_xticklabels([f"{v}\n(n={n})" for v, n in zip(vegs, counts)])
	ax.set_ylabel("mean divergence")
	ax.set_xlabel("veg")
	ax.set_title(title or "Mean pairwise divergence by veg")
	ax.legend(frameon=True)
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_divergence_by_site(
	df: pd.DataFrame,
	out_png: str | Path,
	veg_keep: list[str] | None = None,
	title: str | None = None,
) -> Path:
	"""One panel per veg; each site-year gets three divergence bars."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	out_png = Path(out_png)
	plot_df = df.copy()
	if veg_keep is not None:
		plot_df = plot_df.loc[plot_df["veg"].isin(veg_keep)].copy()
	plot_df["year_i"] = pd.to_numeric(plot_df.get("year"), errors="coerce").astype("Int64")
	plot_df["label"] = plot_df["site"].astype(str)
	if "year" in plot_df.columns:
		plot_df["label"] = plot_df.apply(
			lambda r: f"{r['label']} {int(r['year_i'])}" if pd.notna(r["year_i"]) else r["label"],
			axis=1,
		)

	vegs = sorted(plot_df["veg"].dropna().unique())
	n_veg = max(len(vegs), 1)
	max_n = max((int(plot_df["veg"].eq(v).sum()) for v in vegs), default=4)

	fig, axes = plt.subplots(
		1, n_veg, figsize=(4.2 * n_veg, max(5.0, 0.38 * max_n + 1.8)),
		sharex=False, squeeze=False,
	)

	bar_h = 0.22
	offsets = np.linspace(-(len(DIV_COLS) - 1) / 2, (len(DIV_COLS) - 1) / 2, len(DIV_COLS)) * bar_h

	for ax, veg in zip(axes[0], vegs):
		sub = (
			plot_df.loc[plot_df["veg"].eq(veg)]
			.sort_values(["gvf_vs_ndvi_div", "label"], ascending=[True, True])
			.reset_index(drop=True)
		)
		y = np.arange(len(sub))
		for col, color, dy in zip(DIV_COLS, DIV_COLORS, offsets):
			if col not in sub.columns:
				continue
			ax.barh(y + dy, sub[col].to_numpy(dtype=float), height=bar_h * 0.95, color=color, alpha=0.9)
		ax.set_yticks(y)
		ax.set_yticklabels(sub["label"], fontsize=7)
		ax.invert_yaxis()
		ax.set_xlabel("divergence")
		ax.set_title(f"{veg} (n={len(sub)})")
		ax.grid(True, axis="x", alpha=0.3)

	handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=DIV_LABELS[col]) for col, c in zip(DIV_COLS, DIV_COLORS)]
	fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
	fig.suptitle(title or "Per-site pairwise divergence", y=1.01)
	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def write_divergence_summary(df: pd.DataFrame, out_csv: str | Path) -> Path:
	"""Write a per-veg divergence summary (count/mean/median) CSV."""
	out_csv = Path(out_csv)
	cols = [c for c in DIV_COLS if c in df.columns]
	g = df.groupby("veg", dropna=False)[cols].agg(["count", "mean", "median"]).round(3)
	g.columns = [f"{a}_{b}" for a, b in g.columns]
	g = g.reset_index()
	out_csv.parent.mkdir(parents=True, exist_ok=True)
	g.to_csv(out_csv, index=False)
	print(f"Wrote {out_csv}")
	return out_csv


def cohens_d(a: pd.Series, b: pd.Series) -> float:
	"""Pooled-SD Cohen's d (b − a); NaN if either group has < 2 samples."""
	import numpy as np

	na, nb = len(a), len(b)
	if na < 2 or nb < 2:
		return float("nan")
	var_p = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
	return (b.mean() - a.mean()) / np.sqrt(var_p)


# ---------------------------------------------------------------------------
# Cross-folder ("_combined") figures: pair 2023 vs 2024 for the same site.
# ---------------------------------------------------------------------------

COMBINED_VEG_KEEP = ("AG", "DB", "GR", "SH")
_YEAR_COLOR = {2023: "#1f77b4", 2024: "#2ca02c"}  # blue / green
_PAIR_HALF = 0.20
_SITE_STEP = 1.55


def _combined_pool(clean: pd.DataFrame, veg_keep) -> pd.DataFrame:
	"""Prep the clean pool for combined figures: keep vegs, prefix GBOV, set year."""
	pool = clean.loc[clean["veg"].isin(list(veg_keep))].copy()
	pool["site_label"] = pool["site"].astype(str)
	gbov = pool["source"].astype(str).str.startswith("GBOV_")
	pool.loc[gbov, "site_label"] = "GBOV_" + pool.loc[gbov, "site_label"]
	pool["site"] = pool["site_label"]
	if "year" not in pool.columns or pool["year"].isna().all():
		pool["year"] = pool["source"].str.extract(r"(20\d{2})")[0].astype(float)
	return pool


def _sym_axis_lim(vals, pad=5.0, empty=5.0) -> float:
	"""Half-range: max(|neg|, |pos|) + pad; keeps 0 centered."""
	if not vals:
		return float(empty)
	return float(max(abs(v) for v in vals) + pad)


def _apply_centered_xlim(ax, lim: float, candidates=(5, 10, 25, 50, 100, 200, 500), default=5.0) -> None:
	"""Symmetric xlim around 0 with readable tick steps."""
	from matplotlib.ticker import MultipleLocator

	ax.set_xlim(-lim, lim)
	ax.set_autoscalex_on(False)
	step = float(default)
	for candidate in candidates:
		if lim / candidate <= 5:
			step = float(candidate)
			break
	ax.xaxis.set_major_locator(MultipleLocator(step))


def _combined_layout(plot_df: pd.DataFrame, vegs):
	"""Per-veg y-layout (site order + tick positions) shared across phase rows."""
	layout = {}
	max_sites = 1
	for veg in vegs:
		sub = plot_df.loc[plot_df["veg"].eq(veg)]
		sites = sorted(sub["site"].dropna().unique())
		max_sites = max(max_sites, len(sites))
		centers, ticks, labels, y = [], [], [], 0.0
		for site in sites:
			yrs = sorted({int(v) for v in sub.loc[sub["site"].eq(site), "year"].dropna()})
			centers.append((site, yrs, y))
			ticks.append(y)
			labels.append(site)
			y += _SITE_STEP
		layout[veg] = {"centers": centers, "ticks": ticks, "labels": labels, "ymax": max(y - _SITE_STEP, 0.0)}
	return layout, max_sites


def plot_combined_lag_years(
	clean: pd.DataFrame,
	out_png: str | Path,
	veg_keep=COMBINED_VEG_KEEP,
	title_prefix: str = "All folders",
) -> Path:
	"""Phase x veg lag lollipops across all folders; 2023/2024 paired per site."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import pandas as _pd

	out_png = Path(out_png)
	plot_df = _combined_pool(clean, veg_keep)
	if "lag_sos" not in plot_df.columns and "lag" in plot_df.columns:
		plot_df["lag_sos"] = plot_df["lag"]
	phases = [("lag_sos", "SOS"), ("lag_mos", "MOS"), ("lag_dos", "DOS"), ("lag_eos", "EOS")]

	vegs = [v for v in veg_keep if v in set(plot_df["veg"].dropna())]
	n_veg = max(len(vegs), 1)
	layout, max_sites = _combined_layout(plot_df, vegs)

	fig, axes = plt.subplots(
		len(phases), n_veg,
		figsize=(5.8 * n_veg, max(4.0, 0.48 * max_sites + 2.0) * len(phases)),
		sharex=False, squeeze=False,
	)

	for row_i, (col, phase) in enumerate(phases):
		for col_i, veg in enumerate(vegs):
			ax = axes[row_i][col_i]
			info = layout[veg]
			sub = plot_df.loc[plot_df["veg"].eq(veg)].copy()
			sub["_year_i"] = _pd.to_numeric(sub["year"], errors="coerce").astype("Int64")

			drawn = []
			for site, yrs, y0 in info["centers"]:
				rows = sub.loc[sub["site"].eq(site)]
				if 2023 in yrs and 2024 in yrs:
					year_y = {2023: y0 - _PAIR_HALF, 2024: y0 + _PAIR_HALF}
				elif yrs:
					year_y = {yrs[0]: y0}
				else:
					year_y = {}
				for year, yy in year_y.items():
					r = rows.loc[rows["_year_i"].eq(year)]
					if r.empty or col not in r.columns or _pd.isna(r.iloc[0][col]):
						continue
					drawn.append((yy, float(r.iloc[0][col]), year))

			for yy, val, year in drawn:
				color = _YEAR_COLOR.get(year, "#555555")
				ax.hlines(yy, 0, val, color=color, alpha=0.65, linewidth=1.6)
				ax.scatter([val], [yy], color=color, s=40, zorder=3, edgecolors="white", linewidths=0.4)

			ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)
			ax.set_yticks(info["ticks"])
			ax.set_yticklabels(info["labels"], fontsize=7)
			ax.set_ylim(-0.55, info["ymax"] + 0.55)
			ax.invert_yaxis()
			ax.grid(True, axis="x", alpha=0.3)
			_apply_centered_xlim(ax, _sym_axis_lim([v for _, v, _ in drawn], pad=5.0, empty=5.0))
			if row_i == 0:
				ax.set_title(f"{veg} (n_sites={len(info['labels'])})")
			ax.set_xlabel(f"{phase} lag (days)")

		right_ax = axes[row_i][-1]
		right_ax.text(1.20, 0.95, phase, transform=right_ax.transAxes, ha="left", va="top",
			fontsize=12, fontweight="bold", clip_on=False)

	fig.suptitle(f"{title_prefix} — Phase lag 2023 (blue) vs 2024 (green)", y=0.995)
	fig.subplots_adjust(left=0.09, right=0.90, top=0.93, bottom=0.20, hspace=0.80, wspace=1.05)
	fig.legend(
		handles=[
			Line2D([0], [0], color=_YEAR_COLOR[2023], lw=3, marker="o", label="2023"),
			Line2D([0], [0], color=_YEAR_COLOR[2024], lw=3, marker="o", label="2024"),
			Line2D([0], [0], color="none", label="same site: paired lines close together"),
			Line2D([0], [0], color="none", label="lag_* = gvf_* − ndvi_*  |  0 centered on each panel"),
			Line2D([0], [0], color="none", label="+ later GVF | − earlier GVF"),
		],
		loc="upper center", bbox_to_anchor=(0.47, 0.16), ncol=1, frameon=True, fontsize=8,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.45)
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_combined_compression_years(
	clean: pd.DataFrame,
	out_png: str | Path,
	veg_keep=COMBINED_VEG_KEEP,
	title_prefix: str = "All folders",
) -> Path:
	"""Compression x veg lollipops across all folders; plot (ratio-1) so 0 centers."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import pandas as _pd

	out_png = Path(out_png)
	plot_df = _combined_pool(clean, veg_keep)
	comp_metrics = [
		("greenup_comp", "greenup_comp", "greenup = (gvf_mos−gvf_sos)/(gcc_mos−gcc_sos)"),
		("senescence_comp", "senescence_comp", "senescence = (gvf_eos−gvf_dos)/(gcc_eos−gcc_dos)"),
	]

	vegs = [v for v in veg_keep if v in set(plot_df["veg"].dropna())]
	n_veg = max(len(vegs), 1)
	layout, max_sites = _combined_layout(plot_df, vegs)

	fig, axes = plt.subplots(
		len(comp_metrics), n_veg,
		figsize=(5.8 * n_veg, max(4.0, 0.48 * max_sites + 2.0) * len(comp_metrics)),
		sharex=False, squeeze=False,
	)

	for row_i, (col, phase, _) in enumerate(comp_metrics):
		for col_i, veg in enumerate(vegs):
			ax = axes[row_i][col_i]
			info = layout[veg]
			sub = plot_df.loc[plot_df["veg"].eq(veg)].copy()
			sub["_year_i"] = _pd.to_numeric(sub["year"], errors="coerce").astype("Int64")

			drawn = []
			for site, yrs, y0 in info["centers"]:
				rows = sub.loc[sub["site"].eq(site)]
				if 2023 in yrs and 2024 in yrs:
					year_y = {2023: y0 - _PAIR_HALF, 2024: y0 + _PAIR_HALF}
				elif yrs:
					year_y = {yrs[0]: y0}
				else:
					year_y = {}
				for year, yy in year_y.items():
					r = rows.loc[rows["_year_i"].eq(year)]
					if r.empty or col not in r.columns or _pd.isna(r.iloc[0][col]):
						continue
					drawn.append((yy, float(r.iloc[0][col]) - 1.0, year))

			for yy, delta, year in drawn:
				color = _YEAR_COLOR.get(year, "#555555")
				ax.hlines(yy, 0, delta, color=color, alpha=0.65, linewidth=1.6)
				ax.scatter([delta], [yy], color=color, s=40, zorder=3, edgecolors="white", linewidths=0.4)

			ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.8)
			ax.set_yticks(info["ticks"])
			ax.set_yticklabels(info["labels"], fontsize=7)
			ax.set_ylim(-0.55, info["ymax"] + 0.55)
			ax.invert_yaxis()
			ax.grid(True, axis="x", alpha=0.3)
			_apply_centered_xlim(
				ax, _sym_axis_lim([d for _, d, _ in drawn], pad=0.5, empty=0.5),
				candidates=(0.5, 1, 2, 5, 10, 25, 50), default=0.5,
			)
			if row_i == 0:
				ax.set_title(f"{veg} (n_sites={len(info['labels'])})")
			ax.set_xlabel(phase)

		right_ax = axes[row_i][-1]
		right_ax.text(1.20, 0.95, phase, transform=right_ax.transAxes, ha="left", va="top",
			fontsize=11, fontweight="bold", clip_on=False)

	fig.suptitle(f"{title_prefix} — Compression 2023 (blue) vs 2024 (green)", y=0.995)
	fig.subplots_adjust(left=0.09, right=0.90, top=0.92, bottom=0.22, hspace=0.80, wspace=1.05)
	fig.legend(
		handles=[
			Line2D([0], [0], color=_YEAR_COLOR[2023], lw=3, marker="o", label="2023"),
			Line2D([0], [0], color=_YEAR_COLOR[2024], lw=3, marker="o", label="2024"),
			Line2D([0], [0], color="none", label="0 : same length as GCC"),
			Line2D([0], [0], color="none", label="+ : GVF longer (stretched) | − : GVF shorter (compressed)"),
			Line2D([0], [0], color="none", label=comp_metrics[0][2]),
			Line2D([0], [0], color="none", label=comp_metrics[1][2]),
		],
		loc="upper center", bbox_to_anchor=(0.47, 0.18), ncol=1, frameon=True, fontsize=8,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.45)
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def build_folder_artifacts(
	folder: str | Path,
	input_dir: str | Path,
	anomaly_dir: str | Path,
	year: int | None = None,
	limit: int | None = None,
) -> dict[str, Path]:
	"""Score one input folder and render all its plots into ``output/<folder>/``.

	Writes (per folder): ``scores.csv``, ``boxplot.png``, ``lag_all_lollipop.png``
	(GVF vs NDVI & GCC), ``lag_all_lollipop_research.png`` (shared axes),
	``lag_direction_box.png`` (lag distribution box plots vs NDVI & GCC),
	``compression_all_lollipop.png`` (GVF/GCC & GVF/NDVI), ``divergence_bars_by_veg.png``,
	``divergence_by_site.png``, and ``divergence_by_veg.csv``. Spin-up sites
	(``gvf_sos == 1``) are excluded from the lag/compression/divergence views
	(the boxplot still shows them as red diamonds). Returns the written paths.
	"""
	name = Path(folder).name
	csv_path = collect_folder(folder, input_dir, anomaly_dir, year=year, limit=limit)
	out_dir = csv_path.parent

	df = enrich_scores_frame(load_table(csv_path))
	if df.empty:
		print(f"  {name}: no scored rows; only scores.csv written")
		return {"scores": csv_path}

	df["spin_up"] = df["gvf_sos"].eq(1.0) if "gvf_sos" in df.columns else False
	clean = df.loc[~df["spin_up"]].copy()

	paths: dict[str, Path] = {"scores": csv_path}
	paths["boxplot"] = plot_gap_boxplot_by_veg(
		csv_path, anomaly_dir, out_png=out_dir / "boxplot.png",
		title=f"{name}: GVF-NDVI gap by land type (n_spinup={int(df['spin_up'].sum())})",
	)
	paths["lag_all"] = plot_lag_lollipop(clean, out_dir / "lag_all_lollipop.png", name, refs=("ndvi", "gcc"))
	paths["lag_all_research"] = plot_lag_lollipop(
		clean, out_dir / "lag_all_lollipop_research.png", name, refs=("ndvi", "gcc"), shared_xlim=True,
	)
	paths["lag_box"] = plot_lag_reference_box(clean, out_dir / "lag_direction_box.png", name)
	paths["compression_all"] = plot_compression_lollipop(
		clean, out_dir / "compression_all_lollipop.png", name, refs=("gcc", "ndvi"),
	)
	paths["divergence_bars"] = plot_divergence_bars_by_veg(
		clean, out_dir / "divergence_bars_by_veg.png", title=f"{name}: mean divergence by veg",
	)
	paths["divergence_site"] = plot_divergence_by_site(
		clean, out_dir / "divergence_by_site.png", title=f"{name}: per-site divergence",
	)
	paths["divergence_csv"] = write_divergence_summary(clean, out_dir / "divergence_by_veg.csv")
	return paths
