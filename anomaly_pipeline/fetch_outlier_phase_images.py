"""Download PhenoCam RGB images at GVF/GCC/NDVI CCRmax phase dates for outlier sites.

Phenophase DOYs already live in ``scores.csv`` (computed with CCRmax for GVF,
GCC, and NDVI). This script does **not** refetch those time series; it joins
them onto the compression-outlier table, then pulls one noon-ish camera JPEG
per (site, year, series, phase).

GVF has no camera of its own: the image on a GVF phase date is the PhenoCam
RGB from that DOY, so you can see what the landscape looked like when GVF
said SOS/MOS/DOS/EOS.

Run (after make_box_outliers.py, or this will rewrite the CSV first):
    python anomaly_pipeline/fetch_outlier_phase_images.py
    python anomaly_pipeline/fetch_outlier_phase_images.py --limit 2
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.data_collection import (  # noqa: E402
    PHASE_KEYS,
    enrich_scores_frame,
    load_table,
    write_compression_ref_outliers_by_site,
    write_lag_box_outliers_by_site,
)
from shared.phenocam_api import fetch_one_midday_image  # noqa: E402

OUTPUT = ROOT / "anomaly_pipeline" / "output"
COMBINED = OUTPUT / "_combined"
DEFAULT_FOLDERS = ["Satellite_GVF_timeseries_2023", "Satellite_GVF_timeseries_2024"]
SERIES = ("gcc", "ndvi", "gvf")


def _year_of(folder: str) -> int:
    return int(folder.split("_")[-1])


def _doy_to_date(year: int, doy) -> str | None:
    if doy is None:
        return None
    try:
        n = int(round(float(doy)))
    except (TypeError, ValueError):
        return None
    if n < 1 or n > 366:
        return None
    return (date(int(year), 1, 1) + timedelta(days=n - 1)).isoformat()


def _load_scores(folders: list[str]) -> dict:
    scores_by_year = {}
    for folder in folders:
        clean = enrich_scores_frame(load_table(OUTPUT / folder / "scores.csv"))
        clean["spin_up"] = clean["gvf_sos"].eq(1.0) if "gvf_sos" in clean.columns else False
        scores_by_year[_year_of(folder)] = clean.loc[~clean["spin_up"]].copy()
    return scores_by_year


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="max sites to download (0 = all)")
    p.add_argument("--skip-download", action="store_true", help="only rewrite the CSVs with phase DOYs")
    p.add_argument("--no-rewrite", action="store_true", help="do not rewrite the outlier CSVs/PNG")
    args = p.parse_args()

    scores_by_year = _load_scores(DEFAULT_FOLDERS)
    years = "_".join(str(y) for y in sorted(scores_by_year))
    lag_csv = COMBINED / f"Lag_box_outliers_by_site_{years}.csv"
    comp_csv = COMBINED / f"Compression_ref_outliers_by_site_{years}.csv"
    if args.no_rewrite:
        if not comp_csv.exists() or not lag_csv.exists():
            raise SystemExit("CSVs missing; run without --no-rewrite first")
        paths = {"csv": comp_csv}
    else:
        write_lag_box_outliers_by_site(scores_by_year, lag_csv)
        paths = write_compression_ref_outliers_by_site(
            scores_by_year,
            comp_csv,
            COMBINED / f"Compression_ref_outliers_{years}.png",
        )
    table = load_table(paths["csv"])
    lag_table = load_table(lag_csv) if lag_csv.exists() else table.iloc[0:0]
    year_list = sorted(scores_by_year)
    img_dir = COMBINED / "phenocam_image_test"
    manifest_rows = []

    if args.skip_download:
        print("skip-download: CSVs updated with gvf/gcc/ndvi SOS-MOS-DOS-EOS")
        return 0

    by_site: dict[str, dict] = {}
    for src, frame in (("compression", table), ("lag", lag_table)):
        for recd in frame.to_dict("records"):
            site = recd["site"]
            cur = by_site.setdefault(site, dict(recd))
            cur.setdefault("sources", set()).add(src)
            for year in year_list:
                key = f"flags_{year}"
                a, b = cur.get(key), recd.get(key)
                cur[key] = max(
                    0 if a is None or pd.isna(a) else int(a),
                    0 if b is None or pd.isna(b) else int(b),
                )
                for series in SERIES:
                    for phase in PHASE_KEYS:
                        col = f"{series}_{phase.lower()}_{year}"
                        if (cur.get(col) is None or (not isinstance(cur.get(col), (set, dict)) and pd.isna(cur.get(col)))) and recd.get(col) is not None and not pd.isna(recd.get(col)):
                            cur[col] = recd[col]
            if not cur.get("phenocam_site"):
                cur["phenocam_site"] = recd.get("phenocam_site")

    sites = list(by_site.values())
    if args.limit:
        sites = sites[: args.limit]
    n_ok = n_miss = n_skip = 0
    jpeg_cache: dict[tuple[str, str], Path] = {}
    for i, recd in enumerate(sites, 1):
        site = recd["site"]
        cam = recd.get("phenocam_site") or site
        src = ",".join(sorted(recd.get("sources") or []))
        print(f"[{i}/{len(sites)}] {site}  camera={cam}  ({src})")
        for year in year_list:
            if not recd.get(f"flags_{year}"):
                continue
            for series in SERIES:
                for phase in PHASE_KEYS:
                    doy = recd.get(f"{series}_{phase.lower()}_{year}")
                    iso = _doy_to_date(year, doy)
                    if iso is None:
                        n_skip += 1
                        continue
                    out = img_dir / site / str(year) / f"{series}_{phase}_{iso}.jpg"
                    if out.exists() and out.stat().st_size > 0:
                        jpeg_cache[(cam, iso)] = out
                        n_ok += 1
                        continue
                    cached = jpeg_cache.get((cam, iso))
                    if cached is not None and cached.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(cached.read_bytes())
                        n_ok += 1
                        print(f"    {series} {phase} DOY {doy:.0f} {iso}  copy {cached.name}")
                        continue
                    try:
                        got = fetch_one_midday_image(cam, out, date=iso)
                        jpeg_cache[(cam, iso)] = out
                        n_ok += 1
                        manifest_rows.append({
                            "site": site, "phenocam_site": cam, "year": year,
                            "series": series, "phase": phase, "doy": doy,
                            "date": iso, "url": got["url"], "bytes": got["bytes"],
                            "path": got["out_path"], "status": "ok",
                        })
                        print(f"    {series} {phase} DOY {doy:.0f} {iso}  {got['bytes']} B")
                    except Exception as exc:
                        n_miss += 1
                        manifest_rows.append({
                            "site": site, "phenocam_site": cam, "year": year,
                            "series": series, "phase": phase, "doy": doy,
                            "date": iso, "url": "", "bytes": 0,
                            "path": str(out), "status": f"fail:{exc}",
                        })
                        print(f"    {series} {phase} DOY {doy} {iso}  FAIL {exc}")

    man = COMBINED / "phenocam_phase_images_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(man, index=False)
    print(f"done  ok={n_ok} miss={n_miss} skip_no_doy={n_skip}  manifest={man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
