"""Build combined outlier tables for lag box-plot fliers and GCC-vs-NDVI compression.

Thin CLI wrapper around ``write_lag_box_outliers_by_site`` and
``write_compression_ref_outliers_by_site``. Writes into
``anomaly_pipeline/output/_combined/``:

  - ``Lag_box_outliers_by_site_<years>.csv``
  - ``Compression_ref_outliers_by_site_<years>.csv``
  - ``Compression_ref_outliers_<years>.png``

Run:
    python anomaly_pipeline/make_box_outliers.py [FOLDER ...]
(defaults to the 2023 + 2024 satellite GVF folders)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.data_collection import (  # noqa: E402
    enrich_scores_frame,
    load_table,
    write_lag_box_outliers_by_site,
    write_compression_ref_outliers_by_site,
)

OUTPUT = ROOT / "anomaly_pipeline" / "output"
COMBINED = OUTPUT / "_combined"
DEFAULT_FOLDERS = ["Satellite_GVF_timeseries_2023", "Satellite_GVF_timeseries_2024"]


def _year_of(folder: str) -> int:
    return int(folder.split("_")[-1])


def main(folders: list[str]) -> None:
    scores_by_year = {}
    for folder in folders:
        clean = enrich_scores_frame(load_table(OUTPUT / folder / "scores.csv"))
        clean["spin_up"] = clean["gvf_sos"].eq(1.0) if "gvf_sos" in clean.columns else False
        scores_by_year[_year_of(folder)] = clean.loc[~clean["spin_up"]].copy()

    years = "_".join(str(y) for y in sorted(scores_by_year))
    write_lag_box_outliers_by_site(scores_by_year, COMBINED / f"Lag_box_outliers_by_site_{years}.csv")
    write_compression_ref_outliers_by_site(
        scores_by_year,
        COMBINED / f"Compression_ref_outliers_by_site_{years}.csv",
        COMBINED / f"Compression_ref_outliers_{years}.png",
    )


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_FOLDERS)
