"""Build the Satellite GVF "Data Anomalies" findings PDF for 2023 + 2024.

Rebuilds Section 4 of the summer report on the newly processed, larger Satellite
GVF dataset. Numbers/tables come first (per-veg lag / compression / divergence
summaries + patterns), then the report's DB-vs-sparse statistics are replicated,
then figure-backed findings with each supporting figure embedded and its
filename cited.

Read-only inputs:
  anomaly_pipeline/output/Satellite_GVF_timeseries_<year>/scores.csv (+ plots)
  anomaly_pipeline/output/_combined/{Lag,Compression}_2023_2024.png
  plotting_pipeline/output/Satellite_GVF_timeseries_<year>/<roi>_<year>.png

Output:
  doc/Progress_Report/Satellite_GVF_Anomaly_Findings_2023_2024.pdf

Run:
  cd anomaly_pipeline && python3 make_findings_report.py
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

REPO = Path(__file__).resolve().parent
if REPO.name == "anomaly_pipeline":
    REPO = REPO.parent
sys.path.insert(0, str(REPO))

from shared.data_collection import cohens_d  # noqa: E402

YEARS = (2023, 2024)
ANOMALY_DIR = REPO / "anomaly_pipeline" / "output"
COMBINED_DIR = ANOMALY_DIR / "_combined"
PLOT_DIR = REPO / "plotting_pipeline" / "output"
OUT_PDF = REPO / "doc" / "Progress_Report" / "Satellite_GVF_Anomaly_Findings_2023_2024.pdf"

# sparse / mixed biomes tested against the DB baseline (matches the notebook)
SPARSE = ["GR", "SH", "EN"]
MAIN_VEGS = ["DB", "EN", "GR", "SH", "AG"]
TINY_NOTE_VEGS = ["EB", "DN", "TN", "UN", "WL", "XX"]

SUMMARY_COLS = [
    "veg", "n", "spin_n", "spin_rate",
    "lag_sos", "lag_mos", "lag_dos", "lag_eos",
    "greenup_med", "senes_med", "gvf_ndvi_div",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_scores(year: int) -> pd.DataFrame:
    df = pd.read_csv(ANOMALY_DIR / f"Satellite_GVF_timeseries_{year}" / "scores.csv")
    df["spin_up"] = df["gvf_sos"].eq(1.0)
    df["year"] = year
    return df


def per_veg_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-veg lag / compression / divergence summary (spin-up excluded)."""
    clean = df.loc[~df["spin_up"]]
    rows = []
    for veg, g in clean.groupby("veg"):
        allg = df.loc[df["veg"] == veg]
        rows.append({
            "veg": veg,
            "n": len(g),
            "spin_n": int(allg["spin_up"].sum()),
            "spin_rate": round(float(allg["spin_up"].mean()), 2),
            "lag_sos": round(float(g["lag_sos"].mean()), 1),
            "lag_mos": round(float(g["lag_mos"].mean()), 1),
            "lag_dos": round(float(g["lag_dos"].mean()), 1),
            "lag_eos": round(float(g["lag_eos"].mean()), 1),
            "greenup_med": round(float(g["greenup_comp"].median()), 2),
            "senes_med": round(float(g["senescence_comp"].median()), 2),
            "gvf_ndvi_div": round(float(g["gvf_vs_ndvi_div"].mean()), 2),
        })
    out = pd.DataFrame(rows, columns=SUMMARY_COLS)
    return out.sort_values("n", ascending=False).reset_index(drop=True)


def sparse_vs_db(df: pd.DataFrame):
    """Welch one-sided t-test + Cohen's d on gvf_vs_ndvi_gap (sparse > DB)."""
    clean = df.loc[~df["spin_up"]]
    db = clean.loc[clean["veg"].eq("DB"), "gvf_vs_ndvi_gap"].dropna()
    sparse = clean.loc[clean["veg"].isin(SPARSE), "gvf_vs_ndvi_gap"].dropna()
    if len(db) < 2 or len(sparse) < 2:
        return db, sparse, float("nan"), None
    d = cohens_d(db, sparse)
    tt = stats.ttest_ind(sparse, db, equal_var=False, alternative="greater")
    return db, sparse, d, tt


def val(summary: pd.DataFrame, veg: str, col: str):
    r = summary.loc[summary["veg"] == veg]
    return None if r.empty else r.iloc[0][col]


def _grouped_bars(ax, vegs, a, b, title, ref=None, ymax=None, fmt="{:.2f}", units=""):
    """One subplot: side-by-side 2023/2024 bars per veg, values annotated."""
    x = np.arange(len(vegs))
    w = 0.38

    def clip(v):
        return (min(v, ymax) if (ymax is not None and v is not None) else v)

    ax.bar(x - w / 2, [clip(v) for v in a], w, label="2023", color="#4682B4")
    ax.bar(x + w / 2, [clip(v) for v in b], w, label="2024", color="#C0504D")
    if ref is not None:
        ax.axhline(ref, color="gray", lw=0.9, ls="--", zorder=0)
    ax.axhline(0, color="black", lw=0.6, zorder=0)
    ax.set_title(f"{title}{units}", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(vegs, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.margins(y=0.22)
    if ymax is not None:
        ax.set_ylim(top=ymax * 1.05)
    for xi, (va_, vb_) in enumerate(zip(a, b)):
        for off, raw in ((-w / 2, va_), (w / 2, vb_)):
            if raw is None:
                continue
            disp = clip(raw)
            above = disp >= 0
            ax.annotate(
                fmt.format(raw), (xi + off, disp),
                ha="center", va="bottom" if above else "top",
                fontsize=6, xytext=(0, 2 if above else -2),
                textcoords="offset points",
                fontweight="bold" if (ymax is not None and raw > ymax) else "normal",
                color="#7a1f1f" if (ymax is not None and raw > ymax) else "black",
            )


def add_patterns_chart(pdf: PdfPages, summaries: dict) -> None:
    """Replace the text patterns block with a 6-panel grouped-bar chart."""
    vegs = MAIN_VEGS
    metrics = [
        ("greenup_med", "Median green-up compression", 1.0, 5.0, "{:.2f}", "  (>1 stretched)"),
        ("senes_med", "Median senescence compression", 1.0, None, "{:.2f}", ""),
        ("lag_sos", "Mean SOS lag", 0.0, None, "{:.0f}", "  (days, +=GVF later)"),
        ("lag_mos", "Mean MOS lag", 0.0, None, "{:.0f}", "  (days, +=GVF later)"),
        ("gvf_ndvi_div", "Mean GVF-NDVI divergence", None, None, "{:.2f}", ""),
        ("spin_rate", "Spin-up rate", None, None, "{:.2f}", "  (DOY-1 fits)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 11))
    fig.suptitle("Patterns by vegetation - 2023 vs 2024 (spin-up excluded)",
                 fontsize=14, fontweight="bold", y=0.985)
    for ax, (col, title, ref, ymax, fmt, units) in zip(axes.flat, metrics):
        a = [val(summaries[2023], v, col) for v in vegs]
        b = [val(summaries[2024], v, col) for v in vegs]
        _grouped_bars(ax, vegs, a, b, title, ref, ymax, fmt, units)
    axes.flat[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.text(0.5, 0.045,
             "Dashed line = reference (compression=1.0 -> GVF matches GCC; lag=0 -> no offset). "
             "Green-up panel capped at 5; SH 2023 true value 14.06 shown in red.",
             ha="center", fontsize=7.5, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# PDF helpers
# --------------------------------------------------------------------------- #
def add_text_page(pdf: PdfPages, title: str, lines: list[str], fontsize: int = 10,
                  landscape: bool = False) -> None:
    """Flow wrapped text across as many portrait/landscape pages as needed."""
    size = (11, 8.5) if landscape else (8.5, 11)
    width = 150 if landscape else 100

    def new_fig(first: bool):
        fig = plt.figure(figsize=size)
        if first:
            fig.text(0.07, 0.955, title, fontsize=15, fontweight="bold", va="top")
        return fig

    fig = new_fig(True)
    y = 0.90
    for line in lines:
        chunks = textwrap.wrap(line, width, subsequent_indent="    ") if line else [""]
        for w in chunks:
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = new_fig(False)
                y = 0.94
            fig.text(0.07, y, w, fontsize=fontsize, va="top", family="DejaVu Sans")
            y -= 0.023
        y -= 0.008
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.95, title, fontsize=14, fontweight="bold", va="top")
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.74])
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    for (r, _c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dbe4ee")
    if note:
        fig.text(0.06, 0.08, note, fontsize=8, va="top", style="italic")
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, image_path: Path, heading: str, caption: str) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.06, 0.965, heading, fontsize=12, fontweight="bold", va="top")
    fig.text(0.06, 0.94, f"figure: {image_path.name}", fontsize=9, va="top",
             family="DejaVu Sans Mono", color="#333333")
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.axis("off")
    try:
        ax.imshow(mpimg.imread(str(image_path)))
    except Exception as err:  # noqa: BLE001
        ax.text(0.5, 0.5, f"[could not load image]\n{image_path}\n{err}",
                ha="center", va="center", fontsize=9)
    for cap_line_y, cap in _wrapped_caption(caption):
        fig.text(0.06, cap_line_y, cap, fontsize=8, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _wrapped_caption(caption: str):
    y = 0.10
    for line in textwrap.wrap(caption, 110):
        yield y, line
        y -= 0.02


# --------------------------------------------------------------------------- #
# Example selection (DB early-SOS / late-MOS timeseries)
# --------------------------------------------------------------------------- #
def db_examples(df: pd.DataFrame, year: int):
    """(early_SOS_row, late_MOS_row) DB examples with an existing timeseries PNG."""
    db = df.loc[(df["veg"] == "DB") & (~df["spin_up"])].dropna(subset=["lag_sos", "lag_mos", "roi"])

    def png_for(row) -> Path:
        return PLOT_DIR / f"Satellite_GVF_timeseries_{year}" / f"{row['roi']}_{year}.png"

    def pick(sorted_df):
        for _, row in sorted_df.iterrows():
            if png_for(row).exists():
                return row, png_for(row)
        return None, None

    early_row, early_png = pick(db.sort_values("lag_sos", ascending=True))   # most negative
    late_row, late_png = pick(db.sort_values("lag_mos", ascending=False))    # most positive
    return (early_row, early_png), (late_row, late_png)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build() -> Path:
    data = {y: load_scores(y) for y in YEARS}
    summaries = {y: per_veg_summary(data[y]) for y in YEARS}
    pooled = pd.concat(data.values(), ignore_index=True)

    stats_by_year = {y: sparse_vs_db(data[y]) for y in YEARS}
    stats_pooled = sparse_vs_db(pooled)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        # ---- Title / method -------------------------------------------------
        add_text_page(pdf, "Satellite GVF Data Anomalies — 2023 & 2024", [
            "Rebuild of the summer report's Section 4 (Data Anomalies) on the newly processed,",
            "larger Satellite GVF dataset (Satellite_GVF_timeseries_2023 and _2024).",
            "",
            "Method (unchanged from the report):",
            "  - Phase dates (SOS/MOS/DOS/EOS) from the CCRmax piecewise-logistic fit on GVF, GCC, NDVI.",
            "  - Divergence = phenophase gap (days)/14 + DTW-per-step, computed pairwise (GVF-GCC, GVF-NDVI, GCC-NDVI).",
            "  - greenup_comp = (gvf_mos-gvf_sos)/(gcc_mos-gcc_sos); senescence_comp = (gvf_eos-gvf_dos)/(gcc_eos-gcc_dos).",
            "    ratio > 1 => GVF stretched vs GCC; < 1 => compressed.",
            "  - lag_<phase> = gvf_<phase> - ndvi_<phase> (positive => GVF later).",
            "",
            "What changed: the original report used 40 hand-picked 'Golden Sites'. This runs on the full",
            f"Satellite GVF set ({len(data[2023])} site-years in 2023, {len(data[2024])} in 2024), so per-biome",
            "sample sizes are much larger (the report's Future Work item).",
            "",
            "Spin-up (gvf_sos == 1): the phenology fit failed and pinned SOS on DOY 1. These are EXCLUDED",
            "from all lag / compression / divergence summaries below (still shown as red diamonds on the boxplots).",
            "",
            "Order of this report: numbers first (per-veg summary tables + patterns), then the replicated",
            "statistics, then figure-backed findings. Each finding names the figure(s) that support it.",
        ])

        # ---- Numbers-first per-veg summaries --------------------------------
        for y in YEARS:
            add_table_page(
                pdf, f"Per-vegetation summary — {y} (spin-up excluded)", summaries[y],
                note=("Columns: n=clean site-years, spin_n/spin_rate=DOY-1 fit failures, "
                      "lag_*=mean GVF-NDVI phase lag (days), greenup_med/senes_med=median compression "
                      "ratios (1=match GCC), gvf_ndvi_div=mean GVF-vs-NDVI divergence."),
            )
        add_patterns_chart(pdf, summaries)
        add_image_page(
            pdf, COMBINED_DIR / "Lag_direction_box_2023_2024.png",
            "Lag distribution box plots: GVF-NDVI vs GVF-GCC (2023 vs 2024)",
            "Box plot view of the same lags: per biome x phenophase, one box per reference (NDVI = blue, "
            "GCC = orange). Box = IQR, line = median, whiskers = 1.5x IQR, dots = outlier sites; left of 0 = GVF "
            "earlier, right = GVF later. Shows median and spread/asymmetry rather than just the +/- means. Spin-up excluded.",
        )
        add_text_page(pdf, "Patterns - cross-year read (from the chart)", [
            "Reading the per-vegetation chart on the previous page:",
            "  - DB green-up is consistently STRETCHED (median green-up compression ~3.3-3.4, well above the 1.0 line) in both years.",
            "  - Sparse biomes GR/SH carry the highest GVF-NDVI divergence (~3.4-4.8) vs DB (~1.9-2.1).",
            "  - EN is dominated by spin-up (rate ~0.38, i.e. ~8/21 site-years/yr), so its lag bars are the least trustworthy.",
            "  - AG lags swing sign across phases/years (instability), not a stable offset.",
            "  - SH 2023 green-up compression is an outlier (14.06, off the capped axis) driven by a few tiny GCC-greenup denominators; treat descriptively.",
        ])

        # ---- Replicated statistics -----------------------------------------
        stat_lines = [
            "Replicating the report's test: one-sided Welch t-test + Cohen's d on gvf_vs_ndvi_gap (days),",
            "sparse/mixed (GR+SH+EN) vs the DB baseline. H1: sparse > DB. Spin-up excluded.",
            "",
        ]
        for label, (db, sparse, d, tt) in [("2023", stats_by_year[2023]),
                                            ("2024", stats_by_year[2024]),
                                            ("2023+2024 pooled", stats_pooled)]:
            if tt is None:
                stat_lines.append(f"  {label}: too few sites for a stable test.")
                continue
            diff = sparse.mean() - db.mean()
            sig = "significant" if tt.pvalue < 0.05 else "not significant"
            stat_lines.append(
                f"  {label}: DB mean gap={db.mean():.1f}d (n={len(db)}) | "
                f"sparse mean gap={sparse.mean():.1f}d (n={len(sparse)}) | "
                f"excess={diff:+.1f}d | Cohen's d={d:.2f} | one-sided p={tt.pvalue:.3f} -> {sig}."
            )
        stat_lines += [
            "",
            "Report reference values (40 Golden Sites): p=0.007, Cohen's d=1.21, ~22 day excess for GR.",
            "Interpretation: only the excess gap BEYOND the DB baseline supports a land-cover-driven lag claim;",
            "|d|<0.2 negligible, ~0.5 medium, ~0.8 large. Small n can leave a year underpowered.",
        ]
        add_text_page(pdf, "Statistical test: sparse biomes vs DB baseline", stat_lines)

        # spin-up table
        spin_tbl = pd.DataFrame({
            "veg": summaries[2023]["veg"],
            "2023_spin_n": summaries[2023]["spin_n"],
            "2023_spin_rate": summaries[2023]["spin_rate"],
        }).merge(
            pd.DataFrame({
                "veg": summaries[2024]["veg"],
                "2024_spin_n": summaries[2024]["spin_n"],
                "2024_spin_rate": summaries[2024]["spin_rate"],
            }), on="veg", how="outer"
        ).fillna(0)
        add_table_page(pdf, "Spin-up (DOY-1 fit failure) rate by vegetation", spin_tbl,
                       note="High spin-up = phenology fit unreliable for that biome (flat/low-amplitude curves).")

        # ---- Findings -------------------------------------------------------
        def fig23(name):
            return ANOMALY_DIR / "Satellite_GVF_timeseries_2023" / name

        def fig24(name):
            return ANOMALY_DIR / "Satellite_GVF_timeseries_2024" / name

        # F1 — stretching
        gu23 = val(summaries[2023], "DB", "greenup_med")
        gu24 = val(summaries[2024], "DB", "greenup_med")
        add_text_page(pdf, "Finding 1 - DB green-up is 'stretched', not compressed", [
            f"Deciduous broadleaf (DB) green-up duration ratio (greenup_comp) is well above 1.0 in both years",
            f"(median {gu23} in 2023, {gu24} in 2024). GVF takes far LONGER than the ground camera to go from",
            "onset to peak: the satellite curve is smoothed/stretched and fails to capture the abrupt spring ramp.",
            "This matches the report's Finding 1, now confirmed on a larger DB sample.",
            "",
            "Supporting figures: compression_all_lollipop.png (per year) and _combined/Compression_2023_2024.png.",
        ])
        add_image_page(pdf, fig23("compression_all_lollipop.png"), "Finding 1 - 2023 compression by veg",
                       "green-up (left) / senescence (right) per site vs GCC and NDVI; DB green-up sits far to the right of the ratio=1 line (stretched).")
        add_image_page(pdf, COMBINED_DIR / "Compression_2023_2024.png", "Finding 1 - combined 2023 vs 2024 compression",
                       "0 = same length as GCC; positive = GVF stretched. DB rows are consistently positive across both years.")

        # F2 — early SOS / late MOS mechanism
        (early_row, early_png), (late_row, late_png) = db_examples(data[2023], 2023)
        if early_png is None:
            (early_row, early_png), (late_row, late_png) = db_examples(data[2024], 2024)
            ex_year = 2024
        else:
            ex_year = 2023
        f2_lines = [
            "The stretching has a mechanism: at DB sites GVF tends to detect SOS EARLIER than the camera",
            "(negative lag_sos) but MOS LATER (positive lag_mos). Early start + late peak = a longer apparent",
            "green-up = the stretching in Finding 1. Likely a nadir-view effect (satellite sees understory green-up",
            "before the canopy the camera watches).",
            "",
            "Supporting figures: lag_all_lollipop.png (SOS and MOS rows for DB) plus two DB timeseries examples.",
        ]
        add_text_page(pdf, "Finding 2 - early SOS / late MOS explains the stretch", f2_lines)
        add_image_page(pdf, fig23("lag_all_lollipop.png"), "Finding 2 - 2023 phase lag by veg",
                       "Rows SOS/MOS/DOS/EOS vs NDVI and GCC; DB SOS lags cluster negative (GVF earlier), DB MOS lags trend positive (GVF later).")
        if early_png is not None:
            add_image_page(pdf, early_png, f"Finding 2 - DB early-SOS example ({early_row['site']}, {ex_year})",
                           f"lag_sos={early_row['lag_sos']:.0f}d (GVF SOS earlier than NDVI/GCC). GVF green-up begins before the camera's.")
        if late_png is not None:
            add_image_page(pdf, late_png, f"Finding 2 - DB late-MOS example ({late_row['site']}, {ex_year})",
                           f"lag_mos={late_row['lag_mos']:.0f}d (GVF MOS later). Smoothed GVF pushes peak maturity later into summer.")

        # F3 — sparse biome lag / divergence
        _, _, d_pool, tt_pool = stats_pooled
        p_txt = f"{tt_pool.pvalue:.3f}" if tt_pool is not None else "n/a"
        d_txt = f"{d_pool:.2f}" if tt_pool is not None else "n/a"
        add_text_page(pdf, "Finding 3 - sparse biomes (GR/SH) lag and diverge most", [
            "Grassland (GR) and shrub (SH) carry the highest GVF-NDVI divergence (~3.4-4.8) versus the DB",
            "baseline (~1.9-2.1), with large positive SOS/MOS lags (e.g. GR 2024 mean lag_sos ~ +55d).",
            f"Pooled DB-vs-sparse test: Cohen's d={d_txt}, one-sided p={p_txt} (H1: sparse > DB).",
            "Direction matches the report's Finding 3 (sparse-biome detection delay); see the per-year magnitudes",
            "in the summary tables and statistics page above.",
            "",
            "Supporting figures: divergence_bars_by_veg.png (per year) and _combined/Lag_2023_2024.png.",
        ])
        add_image_page(pdf, fig24("divergence_bars_by_veg.png"), "Finding 3 - 2024 mean divergence by veg",
                       "GR and SH bars are tallest (largest GVF-NDVI / GVF-GCC divergence); DB is among the lowest.")
        add_image_page(pdf, COMBINED_DIR / "Lag_2023_2024.png", "Finding 3 - combined 2023 vs 2024 phase lag",
                       "GR/SH panels show large positive SOS/MOS lags (GVF later) relative to the DB panel.")

        # F4 — AG instability
        add_text_page(pdf, "Finding 4 - agricultural land is phenologically unstable", [
            "Agriculture (AG) lags swing sign across phases and years rather than holding a stable offset",
            "(e.g. mean lag_sos ~ -6d in 2023 vs ~ -36d in 2024). Human planting/harvest schedules violate the",
            "smooth double-logistic curve the pipeline assumes, so AG is unreliable for satellite validation.",
            "This matches the report's Finding 4.",
            "",
            "Supporting figure: lag_all_lollipop.png (AG column) - wide spread of per-site lag around 0.",
        ])
        add_image_page(pdf, fig24("lag_all_lollipop.png"), "Finding 4 - 2024 phase lag by veg (see AG column)",
                       "AG per-site lags spread widely on both sides of 0 across all four phases (instability, not a consistent bias).")

        # F5 — EN spin-up caveat
        en23 = int(val(summaries[2023], "EN", "spin_n") or 0)
        en24 = int(val(summaries[2024], "EN", "spin_n") or 0)
        add_text_page(pdf, "Finding 5 - evergreen needleleaf (EN) is spin-up dominated", [
            f"EN has the highest spin-up rate ({en23} of ~21 in 2023, {en24} of ~21 in 2024): flat, low-amplitude",
            "evergreen curves have no clear winter-to-summer swing, so the SOS fit collapses onto DOY 1.",
            "EN lag/divergence numbers are therefore the least trustworthy and EN should be interpreted with",
            "caution (a data-quality caveat, not necessarily a real satellite error).",
            "",
            "Supporting figure: boxplot.png - spin-up sites plotted as red diamonds.",
        ])
        add_image_page(pdf, fig23("boxplot.png"), "Finding 5 - 2023 GVF-NDVI gap by veg (spin-up = red diamonds)",
                       "Red diamonds mark DOY-1 spin-up fits; EN (and some GR) carry many, inflating their apparent gap.")

        # ---- Figure-to-finding map + caveats --------------------------------
        add_text_page(pdf, "Figure-to-finding map & caveats", [
            "Which figure supports which finding:",
            "  F1 stretching (DB)         -> compression_all_lollipop.png ; _combined/Compression_2023_2024.png",
            "  F2 early-SOS / late-MOS    -> lag_all_lollipop.png ; DB example timeseries (plotting_pipeline/output/...)",
            "  F3 sparse-biome lag        -> divergence_bars_by_veg.png ; _combined/Lag_2023_2024.png",
            "  F4 AG instability          -> lag_all_lollipop.png (AG column)",
            "  F5 EN spin-up caveat       -> boxplot.png",
            "",
            "Caveats:",
            f"  - Tiny-n vegetation types ({', '.join(TINY_NOTE_VEGS)}) are excluded from claims (n<=~10 or placeholder).",
            "  - XX = no vegetation code (placeholder); dropped from the boxplots and not interpreted.",
            "  - 2023 SH shows an inflated median greenup_comp / a very negative mean lag_sos driven by a few sites",
            "    with a tiny GCC green-up denominator; treat those SH ratios descriptively, not as a stable effect.",
            "  - Divergence blends gap and DTW; a large gap from a single mis-fit phase can dominate a site's score.",
            "",
            "Source data: anomaly_pipeline/output/Satellite_GVF_timeseries_<year>/scores.csv",
            "Regenerate: cd anomaly_pipeline && python3 make_findings_report.py",
        ])

    return OUT_PDF


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
