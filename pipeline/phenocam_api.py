"""Fetch and load PhenoCam metadata and summary time series from the public API/archive.

The PhenoCam REST API (https://phenocam.nau.edu/api/) exposes metadata such as
the ROI list (/api/roilists/). The actual 3-day summary CSVs are served as static
files under the data archive:

    https://phenocam.nau.edu/data/archive/{site}/ROI/{site}_{veg}_{roi}_ndvi_3day.csv

Typical use: resolve a site name to its ROI metadata with find_roi(), then download
the NDVI series with fetch_ndvi_3day_for_roi(), and parse it with load_timeseries().
The same loader accepts a local path or an in-memory buffer, so downloaded CSVs are
handled exactly like on-disk ones.

Self-contained duplicate for the pipeline/ folder (originally src/api/phenocam_api.py).
"""

import io
import json
import urllib.request

import pandas as pd

API_BASE = "https://phenocam.nau.edu/api"
ARCHIVE_BASE = "https://phenocam.nau.edu/data/archive"
# The roilists endpoint has no server-side filtering, so we request every record
# in a single page and filter client-side.
ROILIST_PAGE_SIZE = 2000

NUMERIC_COLUMNS = ("doy", "year", "gcc_90", "ndvi_90")
REQUIRED_COLUMNS = ("date", *NUMERIC_COLUMNS)


def load_timeseries(source) -> pd.DataFrame:
	"""Read a PhenoCam summary CSV into a cleaned, date-sorted DataFrame.

	`source` is a path or a file-like object (e.g. io.StringIO of downloaded text).
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
	its primary ROI. Pass a pre-fetched `rois` list to avoid repeat downloads.
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
	"""NDVI 3-day summary URL derived from an ROI metadata record."""
	return ndvi_3day_url(roi["site"], roi["roitype"], roi["sequence_number"])


def fetch_ndvi_3day(site: str, veg_type: str, roi_id: int | str, timeout: float = 30.0) -> io.StringIO:
	"""Download an NDVI 3-day summary by explicit site/veg/ROI as a text buffer."""
	text = fetch_csv_text(ndvi_3day_url(site, veg_type, roi_id), timeout=timeout)
	return io.StringIO(text)


def fetch_ndvi_3day_for_roi(roi: dict, timeout: float = 30.0) -> io.StringIO:
	"""Download the NDVI 3-day summary for a resolved ROI record.

	Raises if the ROI has no infrared/NDVI product (ir_flag is false).
	"""
	if not roi.get("ir_flag", False):
		raise ValueError(f"ROI {roi['roi_name']} has no IR/NDVI data (ir_flag is false)")
	text = fetch_csv_text(roi_ndvi_3day_url(roi), timeout=timeout)
	return io.StringIO(text)
