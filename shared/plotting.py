"""Plotting and styling for the PhenoCam time series figures.

Keeps all matplotlib/figure styling concerns out of test.py.
"""

import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from .PhenoloDates import compute_phases

GCC_COLOR = "#1f77b4"
NDVI_COLOR = "#d62728"
GVF_COLOR = "#17becf"  # satellite Green Vegetation Fraction (third series, distinct from GCC blue)

PHASE_ORDER = ("SOS", "MOS", "DOS", "EOS")
PHASES = {
	"SOS": {"color": "#2ca02c", "linestyle": "--"},
	"MOS": {"color": "#98df8a", "linestyle": "--"},
	"DOS": {"color": "#ff7f0e", "linestyle": ":"},
	"EOS": {"color": "#9467bd", "linestyle": ":"},
}
LABEL_Y = 1.08
LABEL_X_PAD = pd.Timedelta(days=1)
# Distinct markers within this distance would overlap, so their labels are
# split: the earlier one is placed left of its line, the later one right of it.
LABEL_CLUSTER_GAP = pd.Timedelta(days=20)


def doy_to_date(year: int, doy: float) -> pd.Timestamp:
	if pd.isna(doy):
		return pd.NaT
	return pd.to_datetime(f"{year}-{int(round(doy))}", format="%Y-%j")


def phases_to_dates(year: int, phases: dict[str, float]) -> dict[str, pd.Timestamp]:
	return {phase: doy_to_date(year, doy) for phase, doy in phases.items()}


def format_phase_label(prefixes: list[str], phase: str) -> str:
	return f"{' & '.join(prefixes)} {phase}"


def collect_phase_markers(series_dates: dict[str, dict[str, pd.Timestamp]]) -> list[dict]:
	"""Group SOS/MOS/DOS/EOS markers across any number of labelled series.

	`series_dates` maps a series label (e.g. "GCC", "NDVI", "GVF") to its
	{phase: date} dict. Series that share a phase and land on the same date are
	merged onto one line with a combined label (e.g. "GCC & NDVI SOS"), so the
	function handles two series (GCC/NDVI) or three (adding GVF) unchanged.
	"""
	markers = []
	for phase in PHASE_ORDER:
		by_date: dict[pd.Timestamp, list[str]] = {}
		for label, dates in series_dates.items():
			date = dates.get(phase)
			if not pd.isna(date):
				by_date.setdefault(date, []).append(label)

		for date, prefixes in by_date.items():
			markers.append(
				{
					"date": date,
					"phase": phase,
					"prefixes": prefixes,
					"label": format_phase_label(prefixes, phase),
					"style": PHASES[phase],
				}
			)
	return markers


def assign_label_sides(markers: list[dict]) -> list[dict]:
	"""Order markers by date and tag each with a label "side".

	Markers that sit closer than LABEL_CLUSTER_GAP form a cluster. Within a
	cluster the earliest keeps its label on the left of its line and the rest go
	to the right, so overlapping markers stay individually readable. Isolated
	markers default to the right.
	"""
	ordered = sorted(markers, key=lambda m: m["date"])

	cluster: list[dict] = []
	for marker in ordered:
		if cluster and (marker["date"] - cluster[-1]["date"]) > LABEL_CLUSTER_GAP:
			_set_cluster_sides(cluster)
			cluster = []
		cluster.append(marker)
	_set_cluster_sides(cluster)
	return ordered


def _set_cluster_sides(cluster: list[dict]) -> None:
	for index, marker in enumerate(cluster):
		# Single marker or any after the first sits right; the earliest sits left.
		marker["side"] = "left" if len(cluster) > 1 and index == 0 else "right"


def draw_phases(ax, series_dates: dict[str, dict[str, pd.Timestamp]]) -> None:
	"""Draw vertical phase-transition lines for each labelled series on `ax`.

	`series_dates` maps a series label to its {phase: date} dict (e.g.
	{"GCC": ..., "NDVI": ..., "GVF": ...}). Lines are x-only, so they render
	correctly no matter which y-axis a curve is plotted against.
	"""
	label_transform = ax.get_xaxis_transform()

	markers = assign_label_sides(collect_phase_markers(series_dates))
	for marker in markers:
		style = marker["style"]
		line_date = marker["date"]
		on_left = marker["side"] == "left"
		label_x = line_date - LABEL_X_PAD if on_left else line_date + LABEL_X_PAD
		label_ha = "right" if on_left else "left"

		ax.axvline(
			line_date,
			color=style["color"],
			linestyle=style["linestyle"],
			linewidth=1.8,
			alpha=0.85,
		)
		ax.text(
			label_x,
			LABEL_Y,
			marker["label"],
			transform=label_transform,
			color=style["color"],
			rotation=90,
			verticalalignment="bottom",
			horizontalalignment=label_ha,
			fontsize=8,
			bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": style["color"], "alpha": 0.9},
			clip_on=False,
			zorder=5,
		)


def _format_score(value: float, suffix: str = "") -> str:
	return f"{value:.2f}{suffix}" if value is not None and math.isfinite(value) else "n/a"


def draw_agreement_score(ax, scores: dict) -> None:
	"""Annotate the bottom right corner with the NDVI-GCC Divergence Score.

	Lower = better. `scores` is the dict from
	DynamicTimeWrap.site_agreement_score: it has divergence_score,
	phenophase_gap_days, and dtw_per_step.
	"""
	text = (
		f"Divergence Score: {_format_score(scores.get('divergence_score'))}"
		"  (lower = better)\n"
		f"phenophase gap: {_format_score(scores.get('phenophase_gap_days'), ' d')}"
		f"   |   DTW/step: {_format_score(scores.get('dtw_per_step'))}"
	)
	ax.text(
		0.99,
		0.02,
		text,
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=9,
		bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fffbe6", "edgecolor": "#888888", "alpha": 0.95},
		zorder=6,
	)


def draw_satellite_scores(ax, gvf_vs_gcc: dict | None, gvf_vs_ndvi: dict | None) -> None:
	"""Annotate the bottom right with the GVF agreement scores (lower = better).

	One line per comparison; each argument is a DynamicTimeWrap.pairwise_agreement
	result (divergence_score, phenophase_gap_days, dtw_per_step) or None.
	"""

	def line(tag: str, scores: dict | None) -> str:
		scores = scores or {}
		return (
			f"{tag}  Divergence: {_format_score(scores.get('divergence_score'))}"
			f"   |   gap: {_format_score(scores.get('phenophase_gap_days'), ' d')}"
			f"   |   DTW/step: {_format_score(scores.get('dtw_per_step'))}"
		)

	text = (
		"GVF agreement (lower = better)\n"
		+ line("GVF vs GCC ", gvf_vs_gcc)
		+ "\n"
		+ line("GVF vs NDVI", gvf_vs_ndvi)
	)
	ax.text(
		0.99,
		0.02,
		text,
		transform=ax.transAxes,
		ha="right",
		va="bottom",
		fontsize=8.5,
		bbox={"boxstyle": "round,pad=0.4", "facecolor": "#fffbe6", "edgecolor": "#888888", "alpha": 0.95},
		zorder=6,
	)


def create_plot(timeseries: pd.DataFrame, year: int, title: str, scores: dict | None = None):
	"""Build the dual-axis GCC/NDVI figure with phase markers.

	If `scores` (from DynamicTimeWrap.site_agreement_score) is given, the NDVI-GCC
	Divergence Score is annotated on the plot. Returns (figure, gcc_phases,
	ndvi_phases) so the caller can also log the DOY phase values.
	"""
	year_df = timeseries.loc[timeseries["year"] == year]
	if year_df.empty:
		raise ValueError(f"No rows found for year {year}")

	gcc_phases = compute_phases(year_df["doy"].values, year_df["gcc_90"].values)
	ndvi_phases = compute_phases(year_df["doy"].values, year_df["ndvi_90"].values)
	gcc_dates = phases_to_dates(year, gcc_phases)
	ndvi_dates = phases_to_dates(year, ndvi_phases)

	fig, ax = plt.subplots(figsize=(14, 7))
	dates = year_df["date"]

	ax.plot(dates, year_df["gcc_90"], color=GCC_COLOR, linewidth=2.2, label="GCC_90")
	ax.set_xlabel("Date")
	ax.set_ylabel("GCC_90", color=GCC_COLOR)
	ax.tick_params(axis="y", labelcolor=GCC_COLOR)

	ax2 = ax.twinx()
	ax2.plot(dates, year_df["ndvi_90"], color=NDVI_COLOR, linewidth=2.2, label="NDVI_90")
	ax2.set_ylabel("NDVI_90", color=NDVI_COLOR)
	ax2.tick_params(axis="y", labelcolor=NDVI_COLOR)

	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	fig.autofmt_xdate()
	ax.grid(True, alpha=0.3)
	fig.suptitle(title, y=0.90, fontsize=14)

	draw_phases(ax, {"GCC": gcc_dates, "NDVI": ndvi_dates})
	if scores is not None:
		draw_agreement_score(ax2, scores)

	fig.subplots_adjust(top=0.86)
	handles, labels = ax.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax.legend(handles + handles2, labels + labels2, loc="upper left")
	return fig, gcc_phases, ndvi_phases


def create_satellite_plot(
	timeseries: pd.DataFrame,
	gvf: pd.DataFrame,
	year: int,
	title: str,
	gvf_vs_gcc: dict | None = None,
	gvf_vs_ndvi: dict | None = None,
):
	"""Overlay GVF (satellite) with PhenoCam GCC and NDVI for one site-year.

	`timeseries` is the PhenoCam dataframe (needs date/year/doy/gcc_90/ndvi_90);
	`gvf` is a dataframe with date/year/doy/gvf (0-100). Draws GCC on the left
	axis, NDVI on a right axis, and GVF on a third offset right axis (real 0-100
	units), phase markers for all three series, and the GVF-vs-GCC / GVF-vs-NDVI
	score box. Returns (figure, gcc_phases, ndvi_phases, gvf_phases).
	"""
	year_df = timeseries.loc[timeseries["year"] == year]
	gvf_df = gvf.loc[gvf["year"] == year]
	if year_df.empty:
		raise ValueError(f"No PhenoCam rows found for year {year}")
	if gvf_df.empty:
		raise ValueError(f"No GVF rows found for year {year}")

	gcc_phases = compute_phases(year_df["doy"].values, year_df["gcc_90"].values)
	ndvi_phases = compute_phases(year_df["doy"].values, year_df["ndvi_90"].values)
	gvf_phases = compute_phases(gvf_df["doy"].values, gvf_df["gvf"].values)
	gcc_dates = phases_to_dates(year, gcc_phases)
	ndvi_dates = phases_to_dates(year, ndvi_phases)
	gvf_dates = phases_to_dates(year, gvf_phases)

	fig, ax = plt.subplots(figsize=(14, 7))

	ax.plot(year_df["date"], year_df["gcc_90"], color=GCC_COLOR, linewidth=2.2, label="GCC_90")
	ax.set_xlabel("Date")
	ax.set_ylabel("GCC_90", color=GCC_COLOR)
	ax.tick_params(axis="y", labelcolor=GCC_COLOR)

	ax2 = ax.twinx()
	ax2.plot(year_df["date"], year_df["ndvi_90"], color=NDVI_COLOR, linewidth=2.2, label="NDVI_90")
	ax2.set_ylabel("NDVI_90", color=NDVI_COLOR)
	ax2.tick_params(axis="y", labelcolor=NDVI_COLOR)

	ax3 = ax.twinx()
	# Push the third axis outward so its spine/label clears the NDVI axis.
	ax3.spines["right"].set_position(("axes", 1.08))
	ax3.plot(gvf_df["date"], gvf_df["gvf"], color=GVF_COLOR, linewidth=2.2, label="GVF")
	ax3.set_ylabel("GVF (%)", color=GVF_COLOR)
	ax3.tick_params(axis="y", labelcolor=GVF_COLOR)
	ax3.set_ylim(0, 100)

	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
	fig.autofmt_xdate()
	ax.grid(True, alpha=0.3)
	fig.suptitle(title, y=0.90, fontsize=14)

	draw_phases(ax, {"GCC": gcc_dates, "NDVI": ndvi_dates, "GVF": gvf_dates})
	if gvf_vs_gcc is not None or gvf_vs_ndvi is not None:
		draw_satellite_scores(ax2, gvf_vs_gcc, gvf_vs_ndvi)

	fig.subplots_adjust(top=0.86, right=0.86)
	handles, labels = ax.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	handles3, labels3 = ax3.get_legend_handles_labels()
	ax.legend(handles + handles2 + handles3, labels + labels2 + labels3, loc="upper left")
	return fig, gcc_phases, ndvi_phases, gvf_phases


def save_plot(fig, output_file) -> None:
	fig.tight_layout(rect=[0, 0, 1, 0.88])
	fig.savefig(output_file, dpi=200, bbox_inches="tight")
	plt.close(fig)


def is_interactive_backend() -> bool:
	return "agg" not in plt.get_backend().lower()


def show_plots() -> None:
	plt.show()
