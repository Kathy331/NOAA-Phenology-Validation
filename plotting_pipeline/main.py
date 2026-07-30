"""Plotting stage: plot time series for screened sites and GBOV satellite data.

Two kinds of input live under plotting_pipeline/input/:

1. Results JSONs (e.g. uniformity_pipeline/output/<run>/step2_phenology.json or
   pipeline_results.json). For every site-year listed, the PhenoCam NDVI 3-day
   series is refetched and saved as a labelled plot (with the cached divergence
   score) to plotting_pipeline/output/<json_stem>/.
2. GBOV_<year>/ (or GoldenSites_<year>/) subfolders of satellite GVF text files.
   Each GVF file is plotted against its PhenoCam GCC and NDVI curves to
   plotting_pipeline/output/<folder_name>/.

Run with:
    cd plotting_pipeline && python3 main.py
    python3 main.py --limit 5
    python3 main.py --gvf GBOV_2023
"""

import argparse
import sys
from pathlib import Path

# shared package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.plot_json import plot_from_results  
from shared.plot_satellite import plot_satellite_folder 

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"


def plot_all(
	input_dir: Path = INPUT_DIR,
	output_dir: Path = OUTPUT_DIR,
	limit: int | None = None,
) -> list[Path]:
	"""Plot every site-year in every *.json under input_dir.

	Each input file's plots go to output_dir/<file stem>/ so results from
	different JSONs stay separate. `limit` caps how many site-years are plotted
	per file (handy for a quick preview). Returns the written PNG paths.
	"""
	json_files = sorted(input_dir.glob("*.json"))
	if not json_files:
		print(f"No .json files found in {input_dir}. Drop a results JSON there first.")
		return []

	written: list[Path] = []
	for json_file in json_files:
		dest = output_dir / json_file.stem
		print(f"\n=== {json_file.name} -> {dest} ===")
		written.extend(plot_from_results(json_file, dest, limit=limit))

	print(f"\nDone: {len(written)} plots from {len(json_files)} file(s) under {output_dir}")
	return written


def _is_gvf_folder(path: Path) -> bool:
	"""True if `path` is a subfolder that holds GBOV GVF text files."""
	return path.is_dir() and any(path.glob("*_GVF*_timeseries.txt"))


def plot_satellite_all(
	input_dir: Path = INPUT_DIR,
	output_dir: Path = OUTPUT_DIR,
	limit: int | None = None,
) -> list[Path]:
	"""Plot every GBOV_<year> satellite folder found under input_dir.

	A subfolder qualifies if it contains `*_GVF*_timeseries.txt` files (year is
	inferred from its name, e.g. GBOV_2023 -> 2023). Each folder's plots go to
	output_dir/<folder name>/. `limit` caps how many GVF files are plotted per
	folder. Returns the written PNG paths.
	"""
	gvf_folders = sorted(p for p in input_dir.iterdir() if _is_gvf_folder(p))
	if not gvf_folders:
		return []

	written: list[Path] = []
	for folder in gvf_folders:
		dest = output_dir / folder.name
		print(f"\n=== {folder.name}/ (GVF) -> {dest} ===")
		written.extend(plot_satellite_folder(folder, dest, limit=limit))

	print(f"\nDone: {len(written)} GVF plots from {len(gvf_folders)} folder(s) under {output_dir}")
	return written


def plot_satellite_vs_phenocam(
	folder: str | Path,
	input_dir: Path = INPUT_DIR,
	output_dir: Path = OUTPUT_DIR,
	limit: int | None = None,
) -> list[Path]:
	"""Plot one GBOV_<year> folder (e.g. ``GBOV_2023``); skip everything else.

	`folder` may be a bare name under `input_dir` (`GBOV_2023`) or an absolute /
	relative path to the folder. Plots go to `output_dir/<folder name>/`. Year is
	inferred from the folder name. Returns the written PNG paths.
	"""
	folder = Path(folder)
	src = folder if folder.is_dir() else input_dir / folder
	if not _is_gvf_folder(src):
		raise FileNotFoundError(
			f"No GVF timeseries files in {src}. Pass a folder like GBOV_2023 under {input_dir}."
		)

	dest = output_dir / src.name
	print(f"\n=== {src.name}/ (GVF) -> {dest} ===")
	written = plot_satellite_folder(src, dest, limit=limit)
	print(f"\nDone: {len(written)} GVF plots from {src.name} under {output_dir}")
	return written


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments (limit, input/output, optional GVF folder)."""
	parser = argparse.ArgumentParser(
		description="Plot NDVI/GCC/GVF series for plotting_pipeline/input/."
	)
	parser.add_argument("--limit", type=int, default=None, help="Max site-years to plot per JSON/folder (default: all).")
	parser.add_argument("--input", type=Path, default=INPUT_DIR, help="Folder of results JSONs / GVF folders.")
	parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Folder for the plot subfolders.")
	parser.add_argument(
		"--gvf",
		type=str,
		default=None,
		help="Plot only this GVF folder (e.g. GBOV_2023). Default: every GVF folder under --input.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	plot_all(args.input, args.output, limit=args.limit)
	if args.gvf:
		plot_satellite_vs_phenocam(args.gvf, args.input, args.output, limit=args.limit)
	else:
		plot_satellite_all(args.input, args.output, limit=args.limit)


if __name__ == "__main__":
	main()
