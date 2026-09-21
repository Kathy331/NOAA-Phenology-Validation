"""Smoke-test: can we pull a midday camera JPEG from the PhenoCam API?

Uses one site from ``Lag_box_outliers_by_site_2023_2024.csv`` (default
``harvardfarmnorth``). Walks:

  GET /api/cameras/{site}/
  GET /api/middayimages/?site={site}
  GET https://phenocam.nau.edu{imgpath}   # the JPEG itself

Run:
    python anomaly_pipeline/test_phenocam_image.py
    python anomaly_pipeline/test_phenocam_image.py --site homesteadsprings --date 2023-07-15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.phenocam_api import (  # noqa: E402
    fetch_camera,
    fetch_one_midday_image,
    is_rgb_midday,
    list_midday_images,
)

OUTLIER_CSV = ROOT / "anomaly_pipeline" / "output" / "_combined" / "Lag_box_outliers_by_site_2023_2024.csv"
OUT_DIR = ROOT / "anomaly_pipeline" / "output" / "_combined" / "phenocam_image_test"
DEFAULT_SITE = "harvardfarmnorth"


def _site_from_csv(preferred: str) -> str:
    import pandas as pd

    if not OUTLIER_CSV.exists():
        return preferred
    sites = pd.read_csv(OUTLIER_CSV)["site"].dropna().astype(str)
    if preferred in set(sites):
        return preferred
    return sites.iloc[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--site", default=DEFAULT_SITE, help="PhenoCam sitename (must match the camera API)")
    p.add_argument("--date", default="2023-07-15", help="imgdate YYYY-MM-DD (empty = first RGB the API returns)")
    args = p.parse_args()
    site = _site_from_csv(args.site)
    date = args.date or None

    print(f"API root: https://phenocam.nau.edu/api/")
    print(f"test site: {site}  date: {date or '(first available)'}")

    cam = fetch_camera(site)
    print(
        f"camera OK  Sitename={cam.get('Sitename')}  "
        f"lat={cam.get('Lat')} lon={cam.get('Lon')}  "
        f"active={cam.get('active')}  date_first={cam.get('date_first')}  "
        f"date_last={cam.get('date_last')}"
    )

    page = list_midday_images(site, date=date, limit=6)
    print(f"middayimages OK  count={page.get('count')}  n_this_page={len(page.get('results') or [])}")
    print("  note: this endpoint only honours ?site=  (imgdate / year filters are ignored)")
    for rec in page.get("results") or []:
        kind = "RGB" if is_rgb_midday(rec.get("imgpath", "")) else "IR/other"
        print(f"  {rec.get('imgdate')}  {kind}  {rec.get('imgpath')}")

    out = OUT_DIR / f"{site}_{(date or 'first').replace('-', '')}.jpg"
    try:
        got = fetch_one_midday_image(site, out, date=date)
    except ValueError as exc:
        if date:
            print(f"no constructed noon image on {date} ({exc}); retrying first API RGB")
            out = OUT_DIR / f"{site}_first.jpg"
            got = fetch_one_midday_image(site, out, date=None)
        else:
            raise
    print(f"download OK  {got['bytes']} bytes  {got['imgdate']}  -> {got['out_path']}")
    print(f"url: {got['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
