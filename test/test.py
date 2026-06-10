from pathlib import Path

import pandas as pd

from plotting import create_plot, is_interactive_backend, save_plot, show_plots

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"
OUTPUT_DIR = DIR / "output"

YEAR = 2024  # preferred year, uses latest year in file if this year is missing

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


def find_timeseries_files(data_dir: Path) -> list[tuple[str, Path]]:
	"""Return (folder_name, csv_path) for each data/exN/ with one *_ndvi_3day.csv."""
	datasets = []
	for folder in sorted(data_dir.iterdir()):
		if not folder.is_dir():
			continue
		csv_files = sorted(folder.glob("*_ndvi_3day.csv"))
		if not csv_files:
			continue
		if len(csv_files) > 1:
			names = ", ".join(f.name for f in csv_files)
			raise ValueError(f"Expected one *_ndvi_3day.csv in {folder}, found: {names}")
		datasets.append((folder.name, csv_files[0]))
	return datasets


def main() -> None:
	datasets = find_timeseries_files(DATA_DIR)
	if not datasets:
		raise FileNotFoundError(f"No *_ndvi_3day.csv files found under {DATA_DIR}")

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	show_plot = len(datasets) == 1 and is_interactive_backend()

	for name, timeseries_file in datasets:
		timeseries = load_timeseries(timeseries_file)
		year = pick_year(timeseries, YEAR)
		title = f"PhenoCam Time Series for GCC_90 and NDVI_90 ({name}, {year})"
		output_file = OUTPUT_DIR / f"{name}.png"

		fig, gcc_phases, ndvi_phases = create_plot(timeseries, year, title)
		save_plot(fig, output_file)

		print(f"Saved {output_file}")
		print(f"  source: {timeseries_file.name}")
		print(f"  GCC_90 phases (DOY, {year}): {gcc_phases}")
		print(f"  NDVI_90 phases (DOY, {year}): {ndvi_phases}")

	if show_plot:
		show_plots()


if __name__ == "__main__":
	main()
