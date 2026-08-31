"""Create an interactive map for pipeline selected site-years.

By default this reads the loose threshold pipeline results and writes a Leaflet
HTML map with one pin per surviving site-year. A golden site mode can filter the
map down to only isolated USA rows, matching the gold row logic used in the
report generator.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "Full_loose" / "pipeline_results.json"
CLOSE_KM = 4.0  # two sites closer than this count as not isolated
GOLDEN_SITE_COUNT = 40

# final golden-site list. Entries with "2023/2024" expand to both years.
GOLDEN_SITE_YEAR_SPEC = [
	"mandani2_AG_1000 2023",
	"bigtraillake_EN_1000 2023",
	"arkansaswhitaker_AG_1000 2023",
	"NEON.D02.SCBI.DP1.00033_DB_1000 2023",
	"blueoakheadquarters_GR_1000 2023/2024",
	"blackrockforest_DB_1000 2023",
	"uiefmiscanthus2_AG_1000 2023/2024",
	"dukehw_DB_1000 2023",
	"shalehillsczo_DB_2000 2023/2024",
	"NEON.D18.OKSR.DP1.20002_TN_1000 2024",
	"arsope3ltar_AG_1000 2023/2024",
	"macleish_DB_1000 2023",
	"blackrockforest_DB_1000 2024",
	"robinson2_DB_1000 2023",
	"arsmorris2_AG_1000 2024",
	"cafcookeastltar01_AG_1000 2023",
	"NEON.D11.BLUE.DP1.20002_DB_1000 2023",
	"willowcreek_DB_1000 2023/2024",
	"dangermondjalama_GR_1000 2024",
	"laupahoehoe_EB_1000 2023",
	"segawhitepockets_GR_1000 2023",
	"coweeta_DB_2000 2024",
	"morganmonroe2_DB_1000 2023",
	"ninemileprairie_DB_2000 2023",
	"lostcreek_WL_1000 2024",
	"grca1pj_EN_1000 2023",
	"sweetbriar_DB_1000 2023",
	"coweeta_DB_2000 2023",
	"russellsage_DB_1000 2024",
	"vallesmixedconifer_EN_1000 2023",
	"NEON.D05.STEI.DP1.00042_UN_1000 2024",
	"sevmettswl_SH_1000 2024",
	"vallesmixedconifer_EN_1000 2024",
	"usgseros_DB_2000 2023",
	"grca1pj_EN_1000 2024",
	"niwot3_EN_1000 2023/2024",
	"ecb4_AG_1000 2023",
	"turkeypointdbf_DB_2000 2023",
	"luckyhills_SH_4000 2024",
	"portal_SH_1000 2024",
]


# Site loading and value formatting
def _load_sites(input_path: Path) -> tuple[list[dict], dict]:
	"""Read a pipeline_results.json; return (list of site rows, full payload)."""
	data = json.loads(input_path.read_text())
	return data.get("sites", []), data


def _safe_text(value) -> str:
	"""HTML escape a value for safe embedding in a popup (None becomes empty)."""
	return html.escape("" if value is None else str(value), quote=True)


def _format_number(value, digits: int = 3) -> str:
	"""Fixed decimal string for a metric, or 'n/a' when missing/not numeric."""
	if value is None:
		return "n/a"
	try:
		return f"{float(value):.{digits}f}"
	except (TypeError, ValueError):
		return _safe_text(value)


def _marker_popup(site: dict) -> str:
	"""Build the HTML popup (name, year, status, and scores) for one map pin."""
	spatial = site.get("spatial") or {}
	phenology = site.get("phenology") or {}
	status = site.get("status") or "n/a"
	lines = [
		f"<strong>{_safe_text(site.get('name'))}</strong>",
		f"Year: {_safe_text(site.get('year'))}",
		f"Status: {_safe_text(status)}",
		f"Divergence: {_format_number(phenology.get('divergence_score'))}",
		f"NDVI CV: {_format_number(spatial.get('cv_ndvi'))}",
		f"Water%: {_format_number(spatial.get('water_pct'))}",
		f"Urban%: {_format_number(spatial.get('urban_pct'))}",
		f"Bare%: {_format_number(spatial.get('bare_pct'))}",
		f"Non-veg%: {_format_number(spatial.get('nonveg_pct'))}",
	]
	return "<br>".join(lines)


# Geo helpers (distance, USA test, golden-site test)
def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
	"""Great circle distance in km between two (lat, lon) points."""
	radius_km = 6371.0
	lat1, lon1 = map(math.radians, a)
	lat2, lon2 = map(math.radians, b)
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
	return 2 * radius_km * math.asin(math.sqrt(value))


def _is_in_usa(lat: float, lon: float) -> bool:
	"""Approximate USA bounds (CONUS, Alaska, Hawaii, Puerto Rico)."""
	regions = (
		(24.0, 49.5, -125.0, -66.0),   # CONUS
		(51.0, 72.5, -170.0, -129.0),  # Alaska
		(18.5, 22.6, -161.0, -154.0),  # Hawaii
		(17.8, 18.6, -67.4, -65.1),    # Puerto Rico
	)
	return any(min_lat <= lat <= max_lat and min_lon <= lon <= max_lon for min_lat, max_lat, min_lon, max_lon in regions)


def _expand_site_year_spec(spec: str) -> list[tuple[str, int]]:
	"""Expand one spec row into (site, year) tuples (supports 2023/2024 syntax)."""
	parts = spec.rsplit(" ", 1)
	if len(parts) != 2:
		return []
	name, year_token = parts[0].strip(), parts[1].strip()
	if "/" in year_token:
		years = [token.strip() for token in year_token.split("/")]
	else:
		years = [year_token]

	out: list[tuple[str, int]] = []
	for year_text in years:
		try:
			year_value = int(year_text)
		except ValueError:
			continue
		out.append((name, year_value))
	return out


def _golden_site_year_keys() -> list[tuple[str, int]]:
	"""Return ordered site/year keys from the curated golden-site spec."""
	keys: list[tuple[str, int]] = []
	for item in GOLDEN_SITE_YEAR_SPEC:
		keys.extend(_expand_site_year_spec(item))

	# Preserve order while removing accidental duplicates.
	seen: set[tuple[str, int]] = set()
	ordered: list[tuple[str, int]] = []
	for key in keys:
		if key in seen:
			continue
		seen.add(key)
		ordered.append(key)
	return ordered


def _select_golden_sites(sites: list[dict]) -> list[dict]:
	"""Select only the user-curated golden site/year entries, in list order."""
	by_key: dict[tuple[str, int], dict] = {}
	for site in sites:
		name = site.get("name")
		year = site.get("year")
		if name is None or year is None:
			continue
		try:
			year_int = int(year)
		except (TypeError, ValueError):
			continue
		by_key[(str(name), year_int)] = site

	selected: list[dict] = []
	for key in _golden_site_year_keys():
		site = by_key.get(key)
		if site is None:
			continue
		selected.append(site)
	return selected


def _selected_sites(sites: list[dict], golden_only: bool) -> list[dict]:
	"""Every site, or only the golden (isolated USA) ones when golden_only."""
	if not golden_only:
		return sites
	return _select_golden_sites(sites)


# HTML rendering
def _build_html(sites: list[dict], title: str, *, golden_only: bool = False) -> str:
	"""Render the full Leaflet page with one clustered pin per selected site.

	Raises ValueError if no selected site has usable lat/lon coordinates.
	"""
	selected_sites = _selected_sites(sites, golden_only)
	points = []
	for site in selected_sites:
		lat = site.get("lat")
		lon = site.get("lon")
		if lat is None or lon is None:
			continue
		phenology = site.get("phenology") or {}
		spatial = site.get("spatial") or {}
		points.append(
			{
				"lat": lat,
				"lon": lon,
				"popup": _marker_popup(site),
				"name": site.get("name"),
				"year": site.get("year"),
				"divergence": phenology.get("divergence_score"),
				"cv_ndvi": spatial.get("cv_ndvi"),
			}
		)

	if not points:
		raise ValueError("No sites with latitude/longitude were found in the input JSON.")

	center_lat = sum(float(point["lat"]) for point in points) / len(points)
	center_lon = sum(float(point["lon"]) for point in points) / len(points)
	point_json = json.dumps(points, indent=2)
	title_text = _safe_text(title)
	legend_label = "golden site pin" if golden_only else "site-year pin"
	subtitle = (
		"Final Golden Sites (curated list): 320 candidates reduced to the geographically diverse subset across the United States."
		if golden_only
		else "Each pin is one loose-threshold site-year that passed the spatial screen and was ranked by NDVI-GCC divergence."
	)

	return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title_text}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">
  <style>
    :root {{
      color-scheme: light;
      --panel: rgba(255, 255, 255, 0.94);
      --ink: #1f2933;
      --muted: #5b6770;
      --accent: #ad5a1f;
      --border: rgba(31, 41, 51, 0.12);
    }}
    html, body {{ height: 100%; margin: 0; }}
    body {{
      font-family: Georgia, "Times New Roman", serif;
      background: linear-gradient(180deg, #efe6d7 0%, #f8f5ef 40%, #f5f1e8 100%);
      color: var(--ink);
    }}
    .frame {{ position: relative; width: 100%; height: 100%; }}
    #map {{ width: 100%; height: 100%; }}
    .legend {{
      position: absolute;
      bottom: 16px;
      left: 16px;
      z-index: 1000;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 10px 28px rgba(31, 41, 51, 0.12);
      padding: 10px 12px;
      font-size: 13px;
      max-width: min(360px, calc(100vw - 32px));
    }}
    .legend strong {{ display: block; margin-bottom: 4px; }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 8px;
      vertical-align: middle;
      background: var(--accent);
    }}
		/* Override MarkerCluster default green bubbles for better contrast on vegetation. */
		.marker-cluster-small {{
			background-color: rgba(245, 158, 11, 0.36);
		}}
		.marker-cluster-small div {{
			background-color: rgba(245, 158, 11, 0.7);
			color: #1f2933;
		}}
		.marker-cluster-medium {{
			background-color: rgba(234, 179, 8, 0.38);
		}}
		.marker-cluster-medium div {{
			background-color: rgba(234, 179, 8, 0.74);
			color: #1f2933;
		}}
		.marker-cluster-large {{
			background-color: rgba(202, 138, 4, 0.4);
		}}
		.marker-cluster-large div {{
			background-color: rgba(202, 138, 4, 0.76);
			color: #1f2933;
		}}
  </style>
</head>
<body>
  <div class="frame">
    <div id="map"></div>
    <div class="legend">
      <strong>Map key</strong>
      <div><span class="swatch"></span>{legend_label}</div>
      <div>{_safe_text(subtitle)}</div>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    const points = {point_json};
    const map = L.map('map', {{ preferCanvas: true }});

    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const markers = L.markerClusterGroup({{ chunkedLoading: true }});
    const bounds = [];

    for (const point of points) {{
      const marker = L.marker([point.lat, point.lon]);
      marker.bindPopup(point.popup, {{ maxWidth: 320 }});
      markers.addLayer(marker);
      bounds.push([point.lat, point.lon]);
    }}

    map.addLayer(markers);
    map.fitBounds(bounds, {{ padding: [40, 40] }});
    if (!bounds.length) {{
      map.setView([{center_lat}, {center_lon}], 2);
    }}
  </script>
</body>
</html>
"""


# Public map builders + CLI
def build_map(input_path: Path, output_path: Path, title: str | None = None) -> Path:
	"""Write the HTML map of every surviving site-year and return its path."""
	sites, _payload = _load_sites(input_path)
	title = title or f"Loose-threshold validation sites ({input_path.stem})"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(_build_html(sites, title))
	return output_path


def build_golden_map(input_path: Path, output_path: Path, title: str | None = None) -> Path:
	"""Write the HTML map of only the golden (isolated USA) sites, return its path."""
	sites, _payload = _load_sites(input_path)
	title = title or f"Golden validation sites ({input_path.stem})"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(_build_html(sites, title, golden_only=True))
	return output_path


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments (input, output, title, golden-only)."""
	parser = argparse.ArgumentParser(description="Build an interactive map of selected site-years.")
	parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to pipeline_results.json")
	parser.add_argument(
		"--output",
		type=Path,
		default=None,
		help="Path to the HTML file to write. Defaults to a map next to the input file.",
	)
	parser.add_argument(
		"--title",
		type=str,
		default=None,
		help="Optional title shown in the map header.",
	)
	parser.add_argument(
		"--golden-only",
		action="store_true",
		help="Show only isolated USA sites (the gold rows).",
	)
	return parser.parse_args()


def main() -> None:
	"""CLI entry: build the full or golden only map from a pipeline_results.json."""
	args = parse_args()
	output_path = args.output or args.input.with_name(f"{args.input.stem}_map.html")
	if args.golden_only:
		written = build_golden_map(args.input, output_path, title=args.title)
	else:
		written = build_map(args.input, output_path, title=args.title)
	print(f"Saved map to {written}")


if __name__ == "__main__":
	main()