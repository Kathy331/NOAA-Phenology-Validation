"""Validation stage: plot the NDVI/GCC time series for screened sites.

Drop any results JSON (e.g. uniformity_pipeline/output/<run>/step2_phenology.json
or pipeline_results.json) into validation_pipeline/input/, then run this script.
For every site-year listed in each input JSON it re-fetches the PhenoCam NDVI
3-day series and saves a labelled plot (with the cached divergence score) to
validation_pipeline/output/<json_stem>/.

The site JSONs store only names, years, and scores (not the curves), so the plot
logic is shared with the data-prep stage via shared.plot_json (which reuses
shared.plotting for the actual figure).

Run with:
    cd validation_pipeline && python3 main.py
    # optional: cap the number of plots per file while iterating
    python3 main.py --limit 5
"""

import argparse
import sys
from pathlib import Path

# repo root for shared package imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.plot_json import plot_from_results  # noqa: E402

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


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments (limit, input folder, output folder)."""
	parser = argparse.ArgumentParser(
		description="Plot NDVI/GCC series for the sites listed in validation_pipeline/input/*.json."
	)
	parser.add_argument("--limit", type=int, default=None, help="Max site-years to plot per JSON (default: all).")
	parser.add_argument("--input", type=Path, default=INPUT_DIR, help="Folder of results JSONs.")
	parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Folder for the plot subfolders.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	plot_all(args.input, args.output, limit=args.limit)



if __name__ == "__main__":
	main()
