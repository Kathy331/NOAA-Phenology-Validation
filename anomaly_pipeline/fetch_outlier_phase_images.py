"""Download PhenoCam RGB images at GVF/GCC/NDVI CCRmax phase dates.

Phenophase DOYs already live in ``scores.csv``. Writes:

  - outlier stills under ``phenocam_image_outlier/{VEG}/{site}/{year}/``
  - median-site stills under ``phenocam_image_good/{VEG}/{site}/{year}/``

Also moves the legacy ``phenocam_image_test/{site}/{year}/`` tree into the
outlier layout. Copies a JPEG already on disk when ``(site, date)`` matches.

Run:
    python anomaly_pipeline/fetch_outlier_phase_images.py --set good
    python anomaly_pipeline/fetch_outlier_phase_images.py --set both
    python anomaly_pipeline/fetch_outlier_phase_images.py --skip-download
"""
from __future__ import annotations

import argparse
import shutil
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
    write_lag_box_medians_by_site,
    write_lag_box_outliers_by_site,
)
from shared.phenocam_api import fetch_one_midday_image  # noqa: E402

OUTPUT = ROOT / "anomaly_pipeline" / "output"
COMBINED = OUTPUT / "_combined"
DEFAULT_FOLDERS = ["Satellite_GVF_timeseries_2023", "Satellite_GVF_timeseries_2024"]
SERIES = ("gcc", "ndvi", "gvf")
OUTLIER_DIR = COMBINED / "phenocam_image_outlier"
GOOD_DIR = COMBINED / "phenocam_image_good"
LEGACY_DIR = COMBINED / "phenocam_image_test"


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


def _as_int(val) -> int:
    if val is None or (not isinstance(val, (set, dict)) and pd.isna(val)):
        return 0
    return int(val)


def _veg_map(*frames, scores_by_year: dict | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for frame in frames:
        if frame is None or getattr(frame, "empty", True):
            continue
        for recd in frame.to_dict("records"):
            site, veg = recd.get("site"), recd.get("veg")
            if site and veg and (isinstance(veg, str) or not pd.isna(veg)):
                mapping[str(site)] = str(veg).upper()
    if scores_by_year:
        for df in scores_by_year.values():
            if "site" not in df.columns or "veg" not in df.columns:
                continue
            for recd in df[["site", "veg"]].dropna().drop_duplicates().to_dict("records"):
                mapping.setdefault(str(recd["site"]), str(recd["veg"]).upper())
    return mapping


def reorg_legacy_images(veg_map: dict[str, str]) -> int:
    """Move ``phenocam_image_test/{site}/{year}`` into ``phenocam_image_outlier/{VEG}/...``."""
    if not LEGACY_DIR.exists():
        return 0
    moved = 0
    for path in list(LEGACY_DIR.rglob("*.jpg")):
        rel = path.relative_to(LEGACY_DIR)
        parts = rel.parts
        if len(parts) >= 3 and parts[1].isdigit():
            site, year, name = parts[0], parts[1], parts[-1]
        elif len(parts) == 1:
            stem = path.stem
            site = stem.split("_")[0]
            year = stem.split("_")[1][:4] if "_" in stem and stem.split("_")[1][:4].isdigit() else "misc"
            name = path.name
        else:
            continue
        veg = veg_map.get(site, "XX")
        dest = OUTLIER_DIR / veg / site / str(year) / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.resolve() == path.resolve():
            continue
        if dest.exists() and dest.stat().st_size > 0:
            path.unlink()
        else:
            shutil.move(str(path), str(dest))
        moved += 1
    for folder in sorted(LEGACY_DIR.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    if LEGACY_DIR.exists() and not any(LEGACY_DIR.iterdir()):
        LEGACY_DIR.rmdir()
    print(f"reorg  moved={moved}  -> {OUTLIER_DIR}")
    return moved


def _merge_site_rows(frames: list[tuple[str, pd.DataFrame]], year_list: list[int]) -> list[dict]:
    by_site: dict[str, dict] = {}
    for src, frame in frames:
        if frame is None or frame.empty:
            continue
        for recd in frame.to_dict("records"):
            site = recd["site"]
            cur = by_site.setdefault(site, dict(recd))
            cur.setdefault("sources", set()).add(src)
            if recd.get("veg") and not cur.get("veg"):
                cur["veg"] = recd["veg"]
            if recd.get("phenocam_site") and not cur.get("phenocam_site"):
                cur["phenocam_site"] = recd["phenocam_site"]
            for year in year_list:
                key = f"flags_{year}"
                cur[key] = max(_as_int(cur.get(key)), _as_int(recd.get(key)))
                for series in SERIES:
                    for phase in PHASE_KEYS:
                        col = f"{series}_{phase.lower()}_{year}"
                        if _empty(cur.get(col)) and not _empty(recd.get(col)):
                            cur[col] = recd[col]
    return list(by_site.values())


def _empty(val) -> bool:
    if val is None or val == "":
        return True
    if isinstance(val, (set, dict)):
        return False
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _seed_cache(roots: list[Path], cache: dict[tuple[str, str], Path]) -> None:
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jpg"):
            name = path.stem
            bits = name.split("_")
            if len(bits) < 3:
                continue
            iso = bits[-1]
            if len(iso) != 10 or iso[4] != "-":
                continue
            if path.parent.name.isdigit():
                site = path.parent.parent.name
            else:
                site = path.parent.name
            cache.setdefault((site, iso), path)


def _download_set(
    sites: list[dict],
    dest_root: Path,
    year_list: list[int],
    *,
    jpeg_cache: dict[tuple[str, str], Path],
    limit: int = 0,
) -> tuple[int, int, int, list[dict]]:
    if limit:
        sites = sites[:limit]
    n_ok = n_miss = n_skip = 0
    manifest_rows: list[dict] = []
    dest_root.mkdir(parents=True, exist_ok=True)
    for i, recd in enumerate(sites, 1):
        site = recd["site"]
        veg = str(recd.get("veg") or "XX").upper()
        cam = recd.get("phenocam_site") or site
        src = ",".join(sorted(recd.get("sources") or []))
        print(f"[{i}/{len(sites)}] {veg}/{site}  camera={cam}  ({src})")
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
                    out = dest_root / veg / site / str(year) / f"{series}_{phase}_{iso}.jpg"
                    if out.exists() and out.stat().st_size > 0:
                        jpeg_cache[(cam, iso)] = out
                        jpeg_cache[(site, iso)] = out
                        n_ok += 1
                        continue
                    cached = jpeg_cache.get((cam, iso)) or jpeg_cache.get((site, iso))
                    if cached is not None and cached.exists():
                        out.parent.mkdir(parents=True, exist_ok=True)
                        if cached.resolve() != out.resolve():
                            out.write_bytes(cached.read_bytes())
                        jpeg_cache[(cam, iso)] = out
                        jpeg_cache[(site, iso)] = out
                        n_ok += 1
                        print(f"    {series} {phase} DOY {doy:.0f} {iso}  copy {cached.name}")
                        continue
                    try:
                        got = fetch_one_midday_image(cam, out, date=iso)
                        jpeg_cache[(cam, iso)] = out
                        jpeg_cache[(site, iso)] = out
                        n_ok += 1
                        manifest_rows.append({
                            "site": site, "veg": veg, "phenocam_site": cam, "year": year,
                            "series": series, "phase": phase, "doy": doy,
                            "date": iso, "url": got["url"], "bytes": got["bytes"],
                            "path": got["out_path"], "status": "ok",
                        })
                        print(f"    {series} {phase} DOY {doy:.0f} {iso}  {got['bytes']} B")
                    except Exception as exc:
                        n_miss += 1
                        manifest_rows.append({
                            "site": site, "veg": veg, "phenocam_site": cam, "year": year,
                            "series": series, "phase": phase, "doy": doy,
                            "date": iso, "url": "", "bytes": 0,
                            "path": str(out), "status": f"fail:{exc}",
                        })
                        print(f"    {series} {phase} DOY {doy} {iso}  FAIL {exc}")
    return n_ok, n_miss, n_skip, manifest_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="max sites to download per set (0 = all)")
    p.add_argument("--skip-download", action="store_true", help="only rewrite the CSVs with phase DOYs")
    p.add_argument("--no-rewrite", action="store_true", help="do not rewrite the outlier/median CSVs")
    p.add_argument("--set", dest="which", choices=("outlier", "good", "both"), default="both")
    args = p.parse_args()

    scores_by_year = _load_scores(DEFAULT_FOLDERS)
    years = "_".join(str(y) for y in sorted(scores_by_year))
    lag_csv = COMBINED / f"Lag_box_outliers_by_site_{years}.csv"
    median_csv = COMBINED / f"Lag_box_medians_by_site_{years}.csv"
    comp_csv = COMBINED / f"Compression_ref_outliers_by_site_{years}.csv"
    if args.no_rewrite:
        missing = [str(p) for p in (lag_csv, median_csv, comp_csv) if not p.exists()]
        if missing:
            raise SystemExit(f"CSVs missing; run without --no-rewrite first: {missing}")
    else:
        write_lag_box_outliers_by_site(scores_by_year, lag_csv)
        write_lag_box_medians_by_site(scores_by_year, median_csv)
        write_compression_ref_outliers_by_site(
            scores_by_year,
            comp_csv,
            COMBINED / f"Compression_ref_outliers_{years}.png",
        )

    lag_table = load_table(lag_csv) if lag_csv.exists() else pd.DataFrame()
    median_table = load_table(median_csv) if median_csv.exists() else pd.DataFrame()
    comp_table = load_table(comp_csv) if comp_csv.exists() else pd.DataFrame()
    year_list = sorted(scores_by_year)
    veg_map = _veg_map(lag_table, median_table, comp_table, scores_by_year=scores_by_year)
    reorg_legacy_images(veg_map)

    if args.skip_download:
        print("skip-download: CSVs updated; image folders reorganized")
        return 0

    jpeg_cache: dict[tuple[str, str], Path] = {}
    _seed_cache([OUTLIER_DIR, GOOD_DIR], jpeg_cache)
    all_rows: list[dict] = []
    n_ok = n_miss = n_skip = 0
    if args.which in ("outlier", "both"):
        sites = _merge_site_rows((("compression", comp_table), ("lag", lag_table)), year_list)
        print(f"outlier sites={len(sites)} -> {OUTLIER_DIR}")
        ok, miss, skip, rows = _download_set(
            sites, OUTLIER_DIR, year_list, jpeg_cache=jpeg_cache, limit=args.limit,
        )
        n_ok += ok
        n_miss += miss
        n_skip += skip
        all_rows.extend(rows)
    if args.which in ("good", "both"):
        sites = _merge_site_rows((("median", median_table),), year_list)
        print(f"median sites={len(sites)} -> {GOOD_DIR}")
        ok, miss, skip, rows = _download_set(
            sites, GOOD_DIR, year_list, jpeg_cache=jpeg_cache, limit=args.limit,
        )
        n_ok += ok
        n_miss += miss
        n_skip += skip
        all_rows.extend(rows)

    man = COMBINED / "phenocam_phase_images_manifest.csv"
    pd.DataFrame(all_rows).to_csv(man, index=False)
    print(f"done  ok={n_ok} miss={n_miss} skip_no_doy={n_skip}  manifest={man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
