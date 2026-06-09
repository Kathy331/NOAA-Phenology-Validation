import pandas as pd
from pathlib import Path

from plotting import create_plot, is_interactive_backend, save_plot, show_plots

DIR = Path(__file__).parent
DATA_DIR = DIR / "data"
OUTPUT_DIR = DIR / "output"

YEAR = 2024  # preferred year, uses latest year in file if this year is missing


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


def load_timeseries(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path, comment="#")
	df["date"] = pd.to_datetime(df["date"], errors="coerce")
	df["doy"] = pd.to_numeric(df["doy"], errors="coerce")
	df["year"] = pd.to_numeric(df["year"], errors="coerce")
	df["gcc_90"] = pd.to_numeric(df["gcc_90"], errors="coerce")
	df["ndvi_90"] = pd.to_numeric(df["ndvi_90"], errors="coerce")
	return df.dropna(subset=["date", "doy", "year", "gcc_90", "ndvi_90"]).sort_values("date")


def pick_year(timeseries: pd.DataFrame, preferred: int) -> int:
	years = sorted(int(y) for y in timeseries["year"].dropna().unique())
	if not years:
		raise ValueError("No valid years in timeseries")
	if preferred in years:
		return preferred
	print(f"  note: no data for {preferred}, using {years[-1]} (available: {years})")
	return years[-1]


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
