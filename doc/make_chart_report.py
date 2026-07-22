"""Chart-only progress report: the ranked passing-sites table with the updated
WorldCover-based water/urban values.

Reads pipeline/output/Full/pipeline_results.json and writes a single-purpose PDF
containing just the ranked site table (no pipeline write-up):
  - "*" after the site name = GBOV network validation site (mentor's list)
  - gold row = golden site (isolated, no close neighbor, in the USA, safe to pick)
"""

import json
import math
import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO = Path(__file__).resolve().parent.parent
# REPORT_INPUT / REPORT_OUTPUT let a threshold sweep point at a different results
# file and output PDF without editing this script.
RESULTS_JSON = Path(os.environ.get("REPORT_INPUT", REPO / "pipeline" / "output" / "Full" / "pipeline_results.json"))
OUT_PDF = Path(os.environ.get("REPORT_OUTPUT", REPO / "doc" / "Progress_Report" / "July_9_WorldCover.pdf"))

GOLDEN = colors.HexColor("#f5e6a8")
HEADER_BG = colors.HexColor("#404040")
CLOSE_KM = 4.0  # two sites closer than this share a ~4 km footprint

# NEON site prefixes the mentor flagged as GBOV network sites (both 2023 & 2024).
# ROI names look like "<prefix>_DB_1000", so match on the prefix.
GBOV_PREFIXES = (
	"NEON.D01.BART.DP1.00033",
	"NEON.D10.CPER.DP1.00033",
	"NEON.D08.DELA.DP1.00033",
	"NEON.D01.HARV.DP1.00033",
	"NEON.D14.JORN.DP1.00033",
	"NEON.D06.KONA.DP1.00033",
	"NEON.D13.MOAB.DP1.00033",
	"NEON.D15.ONAQ.DP1.00033",
	"NEON.D07.ORNL.DP1.00033",
	"NEON.D14.SRER.DP1.00033",
	"NEON.D08.TALL.DP1.00033",
)


def is_gbov(name: str) -> bool:
	return name.startswith(GBOV_PREFIXES)


def in_usa(lat: float, lon: float) -> bool:
	"""True if lat/lon falls in the continental USA, Alaska, or Hawaii boxes."""
	boxes = [
		(24.0, 50.0, -125.0, -66.0),   # CONUS
		(51.0, 72.0, -170.0, -129.0),  # Alaska
		(18.0, 23.5, -161.0, -154.0),  # Hawaii
	]
	return any(la0 <= lat <= la1 and lo0 <= lon <= lo1 for la0, la1, lo0, lo1 in boxes)


def haversine_km(a, b) -> float:
	R = 6371.0
	la1, lo1 = map(math.radians, a)
	la2, lo2 = map(math.radians, b)
	dla, dlo = la2 - la1, lo2 - lo1
	h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
	return 2 * R * math.asin(math.sqrt(h))


def isolated_locations(sites: list[dict]) -> set:
	"""Names whose location has no *different* site location within CLOSE_KM."""
	locs = {s["name"]: (s["lat"], s["lon"]) for s in sites}
	isolated = set()
	for name, loc in locs.items():
		close = False
		for other, oloc in locs.items():
			if other == name:
				continue
			if haversine_km(loc, oloc) < CLOSE_KM:
				close = True
				break
		if not close:
			isolated.add(name)
	return isolated


def fmt(value, digits, default="-"):
	return f"{value:.{digits}f}" if value is not None else default


def build_pdf() -> None:
	data = json.loads(RESULTS_JSON.read_text())
	sites = data["sites"]
	counts = data["counts"]

	isolated = isolated_locations(sites)
	usa_count = sum(1 for s in sites if in_usa(s["lat"], s["lon"]))
	th = data.get("thresholds", {})
	thresh_txt = (
		f"Thresholds: NDVI CV &lt; {th.get('cv_max')}, water &lt; {th.get('water_max', 0) * 100:.0f}%, "
		f"non-vegetated (urban + bare) &lt; {th.get('nonveg_max', 0) * 100:.0f}%."
	)

	styles = getSampleStyleSheet()
	title = ParagraphStyle("t", parent=styles["Title"], fontSize=15, spaceAfter=6)
	body = ParagraphStyle("b", parent=styles["Normal"], fontSize=8.5, leading=12)
	cell = ParagraphStyle("c", parent=styles["Normal"], fontSize=7, leading=8)
	head = ParagraphStyle("h", parent=styles["Normal"], fontSize=7, leading=8,
	                      textColor=colors.white, fontName="Helvetica-Bold")

	story = [
		Paragraph("Passing Validation Sites - Updated Surface Values (ESA WorldCover)", title),
		Paragraph(
			f"Sites that cleared both screens, ranked by divergence score (lower = stronger "
			f"NDVI-GCC agreement). Water% and Urban% are now read from the ESA WorldCover "
			f"classification (water = class 80, urban/built-up = class 50) instead of the old "
			f"MNDWI/NDBI index thresholds. Evaluated {counts['evaluated']:,} site-years; "
			f"{counts['step1_passed']} passed. {usa_count} are in the USA. {thresh_txt}",
			body,
		),
		Paragraph(
			"Key: * after a site name = GBOV network validation site (from mentor's list); "
			"gold row = golden site (isolated, no close neighbor, in the USA, safe to pick).",
			body,
		),
		Spacer(1, 0.15 * inch),
	]

	header = ["#", "Site (ROI)", "Yr", "Lat", "Lon", "USA", "NDVI CV", "Water%", "Urban%", "Divergence"]
	rows = [[Paragraph(h, head) for h in header]]

	style_cmds = [
		("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
		("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
		("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
		("ALIGN", (2, 1), (-1, -1), "CENTER"),
		("ALIGN", (0, 0), (0, -1), "CENTER"),
		("LEFTPADDING", (0, 0), (-1, -1), 3),
		("RIGHTPADDING", (0, 0), (-1, -1), 3),
		("TOPPADDING", (0, 0), (-1, -1), 2),
		("BOTTOMPADDING", (0, 0), (-1, -1), 2),
	]

	for i, s in enumerate(sites, start=1):
		name = s["name"]
		sp = s.get("spatial", {})
		ph = s.get("phenology") or {}
		usa = in_usa(s["lat"], s["lon"])
		gbov = is_gbov(name)
		water = sp.get("water_pct")
		urban = sp.get("urban_pct")
		display_name = f"{name} *" if gbov else name
		rows.append([
			Paragraph(str(i), cell),
			Paragraph(display_name, cell),
			Paragraph(str(s["year"]), cell),
			Paragraph(fmt(s["lat"], 4), cell),
			Paragraph(fmt(s["lon"], 4), cell),
			Paragraph("USA" if usa else "", cell),
			Paragraph(fmt(sp.get("cv_ndvi"), 3), cell),
			Paragraph(fmt(water * 100 if water is not None else None, 1), cell),
			Paragraph(fmt(urban * 100 if urban is not None else None, 1), cell),
			Paragraph(fmt(ph.get("divergence_score"), 3), cell),
		])

		is_golden = (name in isolated) and usa
		if is_golden:
			style_cmds.append(("BACKGROUND", (0, i), (-1, i), GOLDEN))

	col_widths = [16, 150, 26, 50, 55, 28, 48, 42, 42, 58]
	table = Table(rows, colWidths=col_widths, repeatRows=1)
	table.setStyle(TableStyle(style_cmds))
	story.append(table)

	OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
	doc = SimpleDocTemplate(
		str(OUT_PDF), pagesize=letter,
		leftMargin=0.5 * inch, rightMargin=0.5 * inch,
		topMargin=0.5 * inch, bottomMargin=0.5 * inch,
	)
	doc.build(story)
	print(f"Wrote {OUT_PDF} ({len(sites)} sites, {usa_count} USA, {len(isolated)} isolated).")


if __name__ == "__main__":
	build_pdf()
