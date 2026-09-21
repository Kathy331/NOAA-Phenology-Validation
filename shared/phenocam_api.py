"""Fetch and load PhenoCam metadata and summary time series from the public API/archive.

The PhenoCam REST API (https://phenocam.nau.edu/api/) exposes metadata such as
the ROI list (/api/roilists/). The actual 3 day summary CSVs are served as static
files under the data archive:

    https://phenocam.nau.edu/data/archive/{site}/ROI/{site}_{veg}_{roi}_ndvi_3day.csv

Typical use: resolve a site name to its ROI metadata with find_roi(), then download
the NDVI series with fetch_ndvi_3day_for_roi(), and parse it with load_timeseries().

The same loader accepts a local path.
"""

import io
import json
import urllib.request

import pandas as pd

API_BASE = "https://phenocam.nau.edu/api"
ARCHIVE_BASE = "https://phenocam.nau.edu/data/archive"
ROILIST_PAGE_SIZE = 2000 # maximum number of records to fetch at once

NUMERIC_COLUMNS = ("doy", "year", "gcc_90", "ndvi_90")
REQUIRED_COLUMNS = ("date", *NUMERIC_COLUMNS)


def load_timeseries(source) -> pd.DataFrame:
	"""Read a PhenoCam summary CSV into a cleaned, date sorted DataFrame.
	"""
	df = pd.read_csv(source, comment="#")
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	for column in NUMERIC_COLUMNS:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	return df.dropna(subset=list(REQUIRED_COLUMNS)).sort_values("date")


def pick_year(timeseries: pd.DataFrame, preferred: int) -> int:
	"""Return the preferred year if present, else the latest available year."""
	years = sorted(int(y) for y in timeseries["year"].dropna().unique())
	if not years:
		raise ValueError("No valid years in timeseries")
	if preferred in years:
		return preferred
	print(f"  note: no data for {preferred}, using {years[-1]} (available: {years})")
	return years[-1]


def fetch_json(url: str, timeout: float = 30.0) -> dict:
	with urllib.request.urlopen(url, timeout=timeout) as response:
		return json.loads(response.read().decode("utf-8"))


def fetch_csv_text(url: str, timeout: float = 30.0) -> str:
	"""Download a CSV from the archive and return its raw text."""
	with urllib.request.urlopen(url, timeout=timeout) as response:
		return response.read().decode("utf-8")


def list_rois(timeout: float = 30.0) -> list[dict]:
	"""Return every ROI metadata record from /api/roilists/."""
	url = f"{API_BASE}/roilists/?format=json&limit={ROILIST_PAGE_SIZE}"
	return fetch_json(url, timeout=timeout)["results"]


def find_roi(
	site: str,
	roitype: str | None = None,
	sequence_number: int | None = None,
	rois: list[dict] | None = None,
	timeout: float = 30.0,
) -> dict:
	"""Resolve a site name to a single ROI metadata record.

	Optionally narrow by vegetation type and ROI sequence number. When several
	ROIs match, the lowest sequence number wins, so a bare site name resolves to
	its primary ROI. Pass a prefetched `rois` list to avoid repeat downloads.
	"""
	rois = rois if rois is not None else list_rois(timeout=timeout)
	matches = [r for r in rois if r["site"] == site]
	if roitype is not None:
		matches = [r for r in matches if r["roitype"] == roitype]
	if sequence_number is not None:
		matches = [r for r in matches if r["sequence_number"] == sequence_number]
	if not matches:
		raise ValueError(
			f"No ROI found for site={site!r} (roitype={roitype}, sequence_number={sequence_number})"
		)
	return min(matches, key=lambda r: r["sequence_number"])


def ndvi_3day_url(site: str, veg_type: str, roi_id: int | str) -> str:
	"""URL of the {site}_{veg}_{roi}_ndvi_3day.csv summary file."""
	filename = f"{site}_{veg_type}_{roi_id}_ndvi_3day.csv"
	return f"{ARCHIVE_BASE}/{site}/ROI/{filename}"


def roi_ndvi_3day_url(roi: dict) -> str:
	"""NDVI 3 day summary URL derived from an ROI metadata record."""
	return ndvi_3day_url(roi["site"], roi["roitype"], roi["sequence_number"])


def fetch_ndvi_3day(site: str, veg_type: str, roi_id: int | str, timeout: float = 30.0) -> io.StringIO:
	"""Download an NDVI 3 day summary by explicit site/veg/ROI as a text buffer."""
	text = fetch_csv_text(ndvi_3day_url(site, veg_type, roi_id), timeout=timeout)
	return io.StringIO(text)


def fetch_ndvi_3day_for_roi(roi: dict, timeout: float = 30.0) -> io.StringIO:
	"""Download the NDVI 3 day summary for a resolved ROI record.

	Raises if the ROI has no infrared/NDVI product (ir_flag is false).
	"""
	if not roi.get("ir_flag", False):
		raise ValueError(f"ROI {roi['roi_name']} has no IR/NDVI data (ir_flag is false)")
	text = fetch_csv_text(roi_ndvi_3day_url(roi), timeout=timeout)
	return io.StringIO(text)


def fetch_camera(site: str, timeout: float = 30.0) -> dict:
	"""Camera metadata from ``GET /api/cameras/{site}/``."""
	return fetch_json(f"{API_BASE}/cameras/{site}/", timeout=timeout)


def list_midday_images(
	site: str,
	*,
	date: str | None = None,
	limit: int = 20,
	offset: int = 0,
	timeout: float = 30.0,
) -> dict:
	"""Paginated midday-image records from ``GET /api/middayimages/?site=``.

	``date`` is ``YYYY-MM-DD`` (exact ``imgdate``). Each result has ``imgdate``,
	``site``, and ``imgpath`` (archive path under ``/data/archive/...``).
	"""
	from urllib.parse import urlencode

	params = {"site": site, "limit": limit, "offset": offset}
	if date:
		params["imgdate"] = date
	return fetch_json(f"{API_BASE}/middayimages/?{urlencode(params)}", timeout=timeout)


def archive_image_url(imgpath: str) -> str:
	"""Turn an API ``imgpath`` into a downloadable archive URL."""
	path = imgpath if imgpath.startswith("/") else f"/{imgpath}"
	return f"https://phenocam.nau.edu{path}"


def is_rgb_midday(imgpath: str) -> bool:
	"""True if the archive path is a visible-RGB midday JPEG (not IR)."""
	name = imgpath.rsplit("/", 1)[-1]
	return name.endswith(".jpg") and "_IR_" not in name


def _browse_html(site: str, date: str, timeout: float = 20.0) -> str | None:
	import urllib.error

	y, m, d = date.split("-")
	url = f"https://phenocam.nau.edu/webcam/browse/{site}/{y}/{m}/{d}/"
	try:
		with urllib.request.urlopen(url, timeout=timeout) as response:
			return response.read().decode("utf-8", errors="ignore")
	except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
		return None


def browse_rgb_urls(site: str, date: str, timeout: float = 20.0) -> list[str]:
	"""RGB archive JPEG URLs listed on the browse page for ``date``, noon-first."""
	import re

	html = _browse_html(site, date, timeout=timeout)
	if not html:
		return []
	names = re.findall(rf"{re.escape(site)}_\d{{4}}_\d{{2}}_\d{{2}}_(\d{{6}})\.jpg", html)
	times = []
	seen = set()
	for t in names:
		if t in seen:
			continue
		seen.add(t)
		times.append(t)
	if not times:
		clocks = re.findall(r"(\d{2}):(\d{2}):(\d{2})\s+UTC", html)
		times = [f"{h}{mi}{s}" for h, mi, s in clocks]
	if not times:
		return []

	def dist(t):
		return abs((int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])) - 12 * 3600)

	times.sort(key=dist)
	y, m, d = date.split("-")
	return [f"{ARCHIVE_BASE}/{site}/{y}/{m}/{site}_{y}_{m}_{d}_{t}.jpg" for t in times]


def browse_noon_timestamp(site: str, date: str, timeout: float = 20.0) -> str | None:
	"""HHMMSS closest to noon from the PhenoCam browse page for ``date``."""
	urls = browse_rgb_urls(site, date, timeout=timeout)
	if not urls:
		return None
	return urls[0].rsplit("_", 1)[-1].removesuffix(".jpg")


def _guessed_noon_urls(site: str, date: str) -> list[str]:
	y, m, d = date.split("-")
	times = []
	for hh in (12, 11, 13):
		for mm in (0, 30):
			for ss in (0, 5, 6, 7):
				times.append(f"{hh:02d}{mm:02d}{ss:02d}")
	return [f"{ARCHIVE_BASE}/{site}/{y}/{m}/{site}_{y}_{m}_{d}_{t}.jpg" for t in times]


def _neighbor_dates(date: str, radius: int) -> list[str]:
	from datetime import date as date_cls, timedelta

	y, m, d = (int(x) for x in date.split("-"))
	center = date_cls(y, m, d)
	out = []
	for delta in range(1, radius + 1):
		out.append((center - timedelta(days=delta)).isoformat())
		out.append((center + timedelta(days=delta)).isoformat())
	return out


def _dated_archive_candidates(site: str, date: str, timeout: float = 20.0, nearby: int = 2) -> list[str]:
	"""RGB archive URLs for ``date``, then nearby days if the camera skipped that DOY."""
	seen, urls = set(), []

	def add(items):
		for u in items:
			if u not in seen:
				seen.add(u)
				urls.append(u)

	add(browse_rgb_urls(site, date, timeout=timeout))
	add(_guessed_noon_urls(site, date))
	for alt in _neighbor_dates(date, nearby):
		add(browse_rgb_urls(site, alt, timeout=timeout))
	return urls


def _download_jpeg(url: str, timeout: float) -> bytes:
	with urllib.request.urlopen(url, timeout=timeout) as response:
		data = response.read()
		ctype = response.headers.get("Content-Type", "")
	if not data or ("jpeg" not in ctype.lower() and data[:2] != b"\xff\xd8"):
		raise ValueError(f"Download did not look like a JPEG: {url} content-type={ctype!r} n={len(data)}")
	return data


def fetch_one_midday_image(
	site: str,
	out_path,
	*,
	date: str | None = None,
	timeout: float = 60.0,
) -> dict:
	"""Download one RGB midday JPEG for ``site`` and write it to ``out_path``.

	If ``date`` is given, try constructed archive URLs around noon (the midday
	list API does not filter by date). Otherwise take the first RGB record the
	API returns (oldest first). Returns the chosen record plus ``url`` / ``bytes``.
	"""
	from pathlib import Path
	import urllib.error

	picked = None
	data = None
	url = None
	if date:
		for url in _dated_archive_candidates(site, date, timeout=min(timeout, 20.0)):
			try:
				data = _download_jpeg(url, timeout)
				picked = {
					"imgdate": date,
					"site": site,
					"imgpath": url.replace("https://phenocam.nau.edu", ""),
				}
				break
			except urllib.error.HTTPError as exc:
				if exc.code not in (404, 403):
					raise
		if picked is None:
			raise ValueError(f"No RGB midday JPEG at typical noon times for site={site!r} date={date!r}")
	else:
		page = list_midday_images(site, limit=20, timeout=timeout)
		picked = next((r for r in page.get("results", []) if is_rgb_midday(r.get("imgpath", ""))), None)
		if picked is None:
			raise ValueError(f"No RGB midday image for site={site!r} (api count={page.get('count')})")
		url = archive_image_url(picked["imgpath"])
		data = _download_jpeg(url, timeout)

	out_path = Path(out_path)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_bytes(data)
	return {**picked, "url": url, "bytes": len(data), "out_path": str(out_path)}
