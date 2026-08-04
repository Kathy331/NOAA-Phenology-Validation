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

	# ``lag`` kept as SOS alias for ranking / older notebook cells
	row["lag"] = _round4(lag_sos)
	row["lag_sos"] = _round4(lag_sos)
	row["lag_mos"] = _round4(lag_mos)
	row["lag_dos"] = _round4(lag_dos)
	row["lag_eos"] = _round4(lag_eos)
	row["greenup_comp"] = _round4(greenup_comp)
	row["senescence_comp"] = _round4(senescence_comp)
	return row


def _order_score_columns(frame: pd.DataFrame) -> pd.DataFrame:
	"""Put lag / compression columns immediately after ``site``."""
	preferred = [
		"site",
		"lag", "lag_sos", "lag_mos", "lag_dos", "lag_eos",
		"greenup_comp", "senescence_comp",
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
	Writes ``anomaly_dir/metadata/<folder_name>_scores.csv`` and returns that
	path. Year is inferred from the folder name unless given. Failures are
	skipped with a message (same style as plot_satellite_folder).
	"""
	folder = Path(folder)
	input_dir = Path(input_dir)
	meta = metadata_dir(anomaly_dir)

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
		frame = _order_score_columns(frame)

	meta.mkdir(parents=True, exist_ok=True)
	csv_path = meta / f"{src.name}_scores.csv"
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
	]
	for out_col, gvf_col, ndvi_col in phase_lags:
		if out_col not in frame.columns and {gvf_col, ndvi_col} <= set(frame.columns):
			frame[out_col] = frame[gvf_col] - frame[ndvi_col]
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

	for col in ("lag", "lag_sos", "lag_mos", "lag_dos", "lag_eos", "greenup_comp", "senescence_comp"):
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
	"""Concatenate every ``*_scores.csv`` under ``anomaly_dir/metadata/``."""
	meta = metadata_dir(anomaly_dir)
	paths = sorted(meta.glob("*_scores.csv"))
	if not paths:
		raise FileNotFoundError(f"No *_scores.csv under {meta}/")
	frames = []
	for path in paths:
		frame = enrich_scores_frame(pd.read_csv(path))
		frame["source"] = path.stem.removesuffix("_scores")
		frames.append(frame)
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
	"""``anomaly_pipeline/output/metadata`` (scores CSVs)."""
	base = Path(anomaly_dir) if anomaly_dir is not None else anomaly_output_dir()
	# accept either output or already-metadata
	if base.name == "metadata":
		return base
	return Path(base) / "metadata"


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

	Reads scores from ``anomaly_dir/metadata/*_scores.csv``. Writes
	``anomaly_dir/golden_standard_ranking.csv`` unless ``out_csv`` is given.
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

	out_csv = Path(out_csv) if out_csv is not None else anomaly_dir / "golden_standard_ranking.csv"
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

	Writes under ``anomaly_dir/boxplot/<folder>_BOXPLOT.png`` by default (folder
	name taken from the scores CSV stem, e.g. GoldenSites_2023_scores ->
	GoldenSites_2023_BOXPLOT.png).
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

	spin = df[df["gvf_sos"] == 1.0]
	normal = df[df["gvf_sos"] != 1.0]
	folder = scores_csv.stem.removesuffix("_scores")
	if out_png is None:
		out_png = Path(anomaly_dir) / "boxplot" / f"{folder}_BOXPLOT.png"
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
	ax.set_xticks(positions)
	ax.set_xticklabels(veg_order)
	ax.set_xlabel("Vegetation type")
	ax.set_ylabel("GVF vs NDVI phenophase gap (days)")
	ax.set_title(title or f"{folder}: GVF-NDVI gap by land type (n_spinup={len(spin)})")
	ax.grid(True, axis="y", alpha=0.3)
	fig.tight_layout()
	fig.savefig(out_png, dpi=200, bbox_inches="tight")
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png
