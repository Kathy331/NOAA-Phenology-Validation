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


def plot_lag_lollipop(df: pd.DataFrame, out_png: str | Path, title_prefix: str) -> Path:
	"""Lag by phase (SOS/MOS/DOS/EOS) x veg as lollipops; lag_* = gvf_* − ndvi_*."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	import numpy as np

	out_png = Path(out_png)
	phases = [("lag_sos", "SOS"), ("lag_mos", "MOS"), ("lag_dos", "DOS"), ("lag_eos", "EOS")]
	plot_df = df.copy()
	if "lag_sos" not in plot_df.columns and "lag" in plot_df.columns:
		plot_df["lag_sos"] = plot_df["lag"]

	vegs = sorted(plot_df["veg"].dropna().unique())
	n_veg = max(len(vegs), 1)
	max_n = 4
	for col, _ in phases:
		if col in plot_df.columns and plot_df[col].notna().any():
			max_n = max(max_n, int(plot_df.dropna(subset=[col]).groupby("veg").size().max()))

	fig, axes = plt.subplots(
		len(phases), n_veg,
		figsize=(5.8 * n_veg, max(3.4, 0.32 * max_n + 1.6) * len(phases)),
		sharex=False, squeeze=False,
	)
	colors = plt.cm.tab10(np.linspace(0, 1, max(n_veg, 1)))

	for row_i, (col, phase) in enumerate(phases):
		for col_i, (veg, color) in enumerate(zip(vegs, colors)):
			ax = axes[row_i][col_i]
			if col not in plot_df.columns:
				ax.set_visible(False)
				continue
			sub = (
				plot_df.loc[plot_df["veg"].eq(veg)]
				.dropna(subset=[col, "site"])
				.sort_values(col, ascending=True)
			)
			if sub.empty:
				ax.set_visible(False)
				continue
			_lollipop(ax, list(sub["site"]), sub[col].values, color, ref_line=0)
			ax.tick_params(axis="y", pad=6)
			if row_i == 0:
				ax.set_title(f"{veg} (n={len(sub)})")
			ax.set_xlabel(f"{phase} lag (days)")

		right_ax = axes[row_i][-1]
		right_ax.text(
			1.28, 0.95, phase, transform=right_ax.transAxes,
			ha="left", va="top", fontsize=12, fontweight="bold", clip_on=False,
		)

	fig.suptitle(f"{title_prefix} — Phase lag (GVF − NDVI) by veg", y=0.995)
	fig.subplots_adjust(left=0.08, right=0.90, top=0.93, bottom=0.22, hspace=0.75, wspace=1.15)
	fig.legend(
		handles=[
			Line2D([0], [0], color="none", label="lag_sos/mos/dos/eos = gvf_* − ndvi_*"),
			Line2D([0], [0], color="none", label="+ : GVF after NDVI (GVF later)"),
			Line2D([0], [0], color="none", label="0 : same DOY"),
			Line2D([0], [0], color="none", label="− : GVF before NDVI (GVF earlier)"),
			Line2D([0], [0], color="none", label="|lag| guide: ~0–15 typical | 15–40 look | 40+ candidate"),
		],
		loc="upper center", bbox_to_anchor=(0.47, 0.18), ncol=1, frameon=True, fontsize=8,
	)
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=160, bbox_inches="tight", pad_inches=0.5)
	plt.close(fig)
	print(f"Wrote {out_png}")
	return out_png


def plot_compression_lollipop(df: pd.DataFrame, out_png: str | Path, title_prefix: str) -> Path:
	"""greenup_comp | senescence_comp side-by-side lollipop panels per veg."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D

	out_png = Path(out_png)
	gu_color, sen_color = "#4C78A8", "#F58518"
	metrics = [
		("greenup_comp", gu_color, "greenup_comp = (gvf_mos−gvf_sos)/(gcc_mos−gcc_sos)"),
		("senescence_comp", sen_color, "senescence_comp = (gvf_eos−gvf_dos)/(gcc_eos−gcc_dos)"),
	]
	vegs = sorted(df["veg"].dropna().unique())
	n_veg = max(len(vegs), 1)
	max_n = 4
	for col, _, _ in metrics:
		if col in df.columns and df[col].notna().any():
			max_n = max(max_n, int(df.dropna(subset=[col]).groupby("veg").size().max()))

	fig, axes = plt.subplots(
		n_veg, 2, figsize=(13, max(3.8, 0.40 * max_n + 1.8) * n_veg),
		sharex=False, squeeze=False,
	)
	for row, veg in enumerate(vegs):
		for col_i, (col, color, _) in enumerate(metrics):
			ax = axes[row][col_i]
			sub = df.loc[df["veg"].eq(veg)].dropna(subset=[col, "site"]).sort_values(col, ascending=True)
			if sub.empty:
				ax.set_visible(False)
				continue
			_lollipop(ax, list(sub["site"]), sub[col].values, color, ref_line=1)
			ax.set_xlabel(col)
			if col_i == 0:
				ax.set_ylabel(veg)
			ax.set_title(f"{veg} · {col} (n={len(sub)})")

	fig.suptitle(f"{title_prefix} — greenup_comp (left) | senescence_comp (right)", y=1.01)
	fig.legend(
		handles=[
			Line2D([0], [0], color=gu_color, lw=6, label=metrics[0][2]),
			Line2D([0], [0], color=sen_color, lw=6, label=metrics[1][2]),
			Line2D([0], [0], color="none", label="ratio = 1 : same length as GCC"),
			Line2D([0], [0], color="none", label="ratio < 1 : GVF shorter (compressed vs GCC)"),
			Line2D([0], [0], color="none", label="ratio > 1 : GVF longer (stretched vs GCC)"),
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

	Writes (per folder): ``scores.csv``, ``boxplot.png``, ``lag_lollipop.png``,
	``compression_lollipop.png``, ``divergence_bars_by_veg.png``,
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
	paths["lag"] = plot_lag_lollipop(clean, out_dir / "lag_lollipop.png", name)
	paths["compression"] = plot_compression_lollipop(clean, out_dir / "compression_lollipop.png", name)
	paths["divergence_bars"] = plot_divergence_bars_by_veg(
		clean, out_dir / "divergence_bars_by_veg.png", title=f"{name}: mean divergence by veg",
	)
	paths["divergence_site"] = plot_divergence_by_site(
		clean, out_dir / "divergence_by_site.png", title=f"{name}: per-site divergence",
	)
	paths["divergence_csv"] = write_divergence_summary(clean, out_dir / "divergence_by_veg.csv")
	return paths
