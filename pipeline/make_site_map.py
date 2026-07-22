"""Create an interactive map for pipeline-selected site-years.

By default this reads the loose-threshold pipeline results and writes a Leaflet
HTML map with one pin per surviving site-year. A golden-site mode can filter the
map down to only isolated USA rows, matching the gold-row logic used in the
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
CLOSE_KM = 4.0


def _load_sites(input_path: Path) -> tuple[list[dict], dict]:
	data = json.loads(input_path.read_text())
	return data.get("sites", []), data


def _safe_text(value) -> str:
	return html.escape("" if value is None else str(value), quote=True)


def _format_number(value, digits: int = 3) -> str:
	if value is None:
		return "n/a"
	try:
		return f"{float(value):.{digits}f}"
	except (TypeError, ValueError):
		return _safe_text(value)


def _marker_popup(site: dict) -> str:
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


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
	radius_km = 6371.0
	lat1, lon1 = map(math.radians, a)
	lat2, lon2 = map(math.radians, b)
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
	return 2 * radius_km * math.asin(math.sqrt(value))


def _is_in_usa(lat: float, lon: float) -> bool:
	return (
		24.0 <= lat <= 50.0
		or 51.0 <= lat <= 72.0
		or 18.0 <= lat <= 23.5
	)


def _is_golden_site(site: dict, all_sites: list[dict]) -> bool:
	lat = site.get("lat")
	lon = site.get("lon")
	if lat is None or lon is None or not _is_in_usa(float(lat), float(lon)):
		return False
	location = (float(lat), float(lon))
	for other in all_sites:
		if other is site:
			continue
		other_lat = other.get("lat")
		other_lon = other.get("lon")
		if other_lat is None or other_lon is None:
			continue
		if _haversine_km(location, (float(other_lat), float(other_lon))) < CLOSE_KM:
			return False
	return True


def _selected_sites(sites: list[dict], golden_only: bool) -> list[dict]:
	if not golden_only:
		return sites
	return [site for site in sites if _is_golden_site(site, sites)]


def _build_html(sites: list[dict], title: str, *, golden_only: bool = False) -> str:
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
		"Golden sites only: isolated within 4 km and inside the USA."
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


def build_map(input_path: Path, output_path: Path, title: str | None = None) -> Path:
	sites, _payload = _load_sites(input_path)
	title = title or f"Loose-threshold validation sites ({input_path.stem})"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(_build_html(sites, title))
	return output_path


def build_golden_map(input_path: Path, output_path: Path, title: str | None = None) -> Path:
	sites, _payload = _load_sites(input_path)
	title = title or f"Golden validation sites ({input_path.stem})"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(_build_html(sites, title, golden_only=True))
	return output_path


def parse_args() -> argparse.Namespace:
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
	args = parse_args()
	output_path = args.output or args.input.with_name(f"{args.input.stem}_map.html")
	if args.golden_only:
		written = build_golden_map(args.input, output_path, title=args.title)
	else:
		written = build_map(args.input, output_path, title=args.title)
	print(f"Saved map to {written}")


if __name__ == "__main__":
	main()