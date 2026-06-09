"""Loading and cleaning of PhenoCam 3-day summary time series.

Shared by the local-file driver (test.py) and the API driver (api_test.py).
`load_timeseries` accepts anything pandas.read_csv accepts — a file path or an
in-memory text buffer — so the same parsing serves both on-disk and downloaded
CSVs.
"""

import pandas as pd

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
