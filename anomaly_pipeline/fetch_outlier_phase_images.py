"""Download PhenoCam RGB images at GVF/GCC/NDVI CCRmax phase dates.

Phenophase DOYs already live in ``scores.csv``. Writes:

  - outlier stills under ``phenocam_image_outlier/{VEG}/{site}/{year}/``
  - median-site stills under ``phenocam_image_good/{VEG}/{site}/{year}/``

Also moves the legacy ``phenocam_image_test/{site}/{year}/`` tree into the
outlier layout. Copies a JPEG already on disk when ``(site, date)`` matches.
Every outlier site-year folder gets an ``outlier_phases.txt`` saying which
phase(s) and which series (GVF / GCC / NDVI) were flagged. Outlier images are
fetched per flagged ROI, so multi-ROI sites get each ROI's own phase dates.
Dates with no photo on PhenoCam (within 2 days) are logged to
``phenocam_missing_images.csv`` and not retried.

Each site/year folder also gets ``roi_mask_{roi}.png`` (ROI white on black,
from PhenoCam's ``{roi}_roi.csv`` + ``.tif`` masks) and a side-by-side
``_preview.png``; the photos themselves are never drawn on.

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
    find_compression_ref_outliers,
    find_lag_box_outliers,
    load_table,
    _phenocam_sitename,
    write_compression_ref_outliers_by_site,
    write_lag_box_medians_by_site,
    write_lag_box_outliers_by_site,
)
from shared.phenocam_api import fetch_one_midday_image, fetch_roi_mask, roi_mask_list  # noqa: E402

OUTPUT = ROOT / "anomaly_pipeline" / "output"
COMBINED = OUTPUT / "_combined"
DEFAULT_FOLDERS = ["Satellite_GVF_timeseries_2023", "Satellite_GVF_timeseries_2024"]
SERIES = ("gcc", "ndvi", "gvf")
OUTLIER_DIR = COMBINED / "phenocam_image_outlier"
GOOD_DIR = COMBINED / "phenocam_image_good"
LEGACY_DIR = COMBINED / "phenocam_image_test"
MISSING_CSV = COMBINED / "phenocam_missing_images.csv"
MASK_CACHE = COMBINED / "phenocam_roi_masks"
MASK_PREFIX = "roi_mask_"
MASK_VERSIONS_NOTE = "roi_masks.txt"


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


def _outlier_roi_jobs(scores_by_year: dict, site_vegs: dict[str, str]) -> list[dict]:
    """One download job per flagged (site, ROI), carrying that ROI's own phase DOYs."""
    jobs: dict[tuple[str, str], dict] = {}
    for year, clean in sorted(scores_by_year.items()):
        lag = find_lag_box_outliers(clean)
        comp = find_compression_ref_outliers(clean)
        pairs = {
            (r["site"], r["roi"]): src
            for src, frame in (("lag", lag), ("compression", comp)) if len(frame)
            for r in frame[["site", "roi"]].drop_duplicates().to_dict("records")
        }
        for (site, roi), src in sorted(pairs.items()):
            row = _score_row(clean, site, roi)
            job = jobs.setdefault((site, roi), {
                "site": site,
                "roi": roi,
                "veg": site_vegs.get(site) or "XX",
                "phenocam_site": _phenocam_sitename(roi),
                "sources": set(),
            })
            job["sources"].add(src)
            job[f"flags_{year}"] = 1
            for series in SERIES:
                for phase in PHASE_KEYS:
                    job[f"{series}_{phase.lower()}_{year}"] = _doy(row, f"{series}_{phase.lower()}")
    return list(jobs.values())


def _load_missing() -> set[tuple[str, str]]:
    if not MISSING_CSV.exists():
        return set()
    df = pd.read_csv(MISSING_CSV, dtype=str)
    return set(zip(df["phenocam_site"], df["date"]))


def _save_missing(missing: set[tuple[str, str]]) -> None:
    pd.DataFrame(sorted(missing), columns=["phenocam_site", "date"]).to_csv(MISSING_CSV, index=False)


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
    missing: set[tuple[str, str]],
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
                    if (cam, iso) in missing:
                        n_miss += 1
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
                        if "No RGB midday JPEG" in str(exc):
                            missing.add((cam, iso))
                        manifest_rows.append({
                            "site": site, "veg": veg, "phenocam_site": cam, "year": year,
                            "series": series, "phase": phase, "doy": doy,
                            "date": iso, "url": "", "bytes": 0,
                            "path": str(out), "status": f"fail:{exc}",
                        })
                        print(f"    {series} {phase} DOY {doy} {iso}  FAIL {exc}")
    return n_ok, n_miss, n_skip, manifest_rows


def _mask_versions(cam: str, roi: str) -> list[dict]:
    """ROI mask versions, cached as ``phenocam_roi_masks/{camera}/{roi}_versions.csv``."""
    cached = MASK_CACHE / cam / f"{roi}_versions.csv"
    if cached.exists():
        df = pd.read_csv(cached, dtype=str)
        return df.to_dict("records")
    try:
        versions = roi_mask_list(cam, roi)
    except Exception as exc:
        print(f"  mask list FAIL {roi}: {exc}")
        return []
    cached.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(versions, columns=["start", "end", "maskfile"]).to_csv(cached, index=False)
    return versions


def masks_for_year(cam: str, roi: str, year: int) -> list[dict]:
    lo, hi = f"{year}-01-01", f"{year}-12-31"
    return [v for v in _mask_versions(cam, roi) if v["start"] <= hi and v["end"] >= lo]


def _mask_tif(cam: str, maskfile: str) -> Path | None:
    path = MASK_CACHE / cam / maskfile
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        fetch_roi_mask(cam, maskfile, path)
        return path
    except Exception as exc:
        print(f"  mask FAIL {cam}/{maskfile}: {exc}")
        return None


def _preview_dates(job: dict, year: int, version: dict) -> list[str]:
    """Candidate photo dates inside both ``year`` and the mask version's range, MOS first."""
    lo = max(f"{year}-01-01", version["start"])
    hi = min(f"{year}-12-31", version["end"])
    out: list[str] = []
    for phase in ("MOS", "DOS", "SOS", "EOS"):
        for series in ("gvf", "gcc", "ndvi"):
            iso = _doy_to_date(year, job.get(f"{series}_{phase.lower()}_{year}"))
            if iso and lo <= iso <= hi and iso not in out:
                out.append(iso)
    mid = date.fromisoformat(lo) + (date.fromisoformat(hi) - date.fromisoformat(lo)) / 2
    out.append(mid.isoformat())
    return out


def _preview_photo(cam: str, dates: list[str], missing: set[tuple[str, str]]) -> Path | None:
    """A photo from the ROI's own camera, cached under ``phenocam_roi_masks/{camera}/photos``."""
    for iso in dates[:4]:
        path = MASK_CACHE / cam / "photos" / f"{cam}_{iso}.jpg"
        if path.exists() and path.stat().st_size > 0:
            return path
        if (cam, iso) in missing:
            continue
        try:
            fetch_one_midday_image(cam, path, date=iso)
            return path
        except Exception as exc:
            if "No RGB midday JPEG" in str(exc):
                missing.add((cam, iso))
            print(f"  preview photo FAIL {cam} {iso}: {exc}")
    return None


def _write_mask_pngs(tif: Path, folder: Path, stem: str, roi: str, photo_path: Path | None) -> None:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np

    mask = Image.open(tif).convert("L")
    inside = np.array(mask) < 128
    Image.fromarray(np.where(inside, 255, 0).astype("uint8")).save(folder / f"{stem}.png")

    if photo_path is None:
        return
    photo = Image.open(photo_path).convert("RGB")
    roi_img = Image.fromarray(np.where(inside, 255, 0).astype("uint8"))
    if roi_img.size != photo.size:
        roi_img = roi_img.resize(photo.size, Image.NEAREST)
    roi_arr = np.array(roi_img) > 0
    edge = roi_arr & ~(np.array(roi_img.filter(ImageFilter.MinFilter(7))) > 0)

    shaded = np.array(photo).astype("float32")
    shaded[~roi_arr] *= 0.3
    shaded[edge] = (255, 220, 0)
    right = Image.fromarray(shaded.astype("uint8"))

    w, h = photo.size
    pad = 28
    canvas = Image.new("RGB", (2 * w, h + pad), "white")
    canvas.paste(photo, (0, pad))
    canvas.paste(right, (w, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), photo_path.name, fill="black")
    draw.text((w + 8, 8), f"ROI {roi} ({tif.name})", fill="black")
    if canvas.width > 1800:
        canvas = canvas.resize((1800, round(canvas.height * 1800 / canvas.width)))
    canvas.save(folder / f"{stem}_preview.png")


def write_roi_masks(jobs: list[dict], dest_root: Path, year_list: list[int]) -> tuple[int, int]:
    """Write ROI mask PNGs + side-by-side previews into each ``{VEG}/{site}/{year}`` folder."""
    missing = _load_missing()
    cleaned: set[Path] = set()
    multi: dict[Path, list[str]] = {}
    n_ok = n_miss = 0
    for job in jobs:
        site, roi = job["site"], job.get("roi")
        if not isinstance(roi, str) or not roi:
            continue
        veg = str(job.get("veg") or "XX").upper()
        cam = job.get("phenocam_site") or _phenocam_sitename(roi)
        for year in year_list:
            if not job.get(f"flags_{year}"):
                continue
            folder = dest_root / veg / site / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            if folder not in cleaned:
                for old in [*folder.glob(f"{MASK_PREFIX}*"), *folder.glob(MASK_VERSIONS_NOTE)]:
                    old.unlink()
                cleaned.add(folder)
            versions = masks_for_year(cam, roi, year)
            if not versions:
                n_miss += 1
                print(f"  no ROI mask for {roi} in {year}")
                continue
            for v in versions:
                tif = _mask_tif(cam, v["maskfile"])
                if tif is None:
                    n_miss += 1
                    continue
                suffix = f"_{Path(v['maskfile']).stem.rsplit('_', 1)[-1]}" if len(versions) > 1 else ""
                stem = f"{MASK_PREFIX}{roi}{suffix}"
                photo = _preview_photo(cam, _preview_dates(job, year, v), missing)
                _write_mask_pngs(tif, folder, stem, roi, photo)
                n_ok += 1
                if len(versions) > 1:
                    end = "still active" if v["end"].startswith("9999") else v["end"]
                    multi.setdefault(folder, []).append(f"{stem}.png  {v['maskfile']}  {v['start']} to {end}")
    for folder, lines in multi.items():
        (folder / MASK_VERSIONS_NOTE).write_text(
            "The ROI mask changed during this year; each mask applies to its date range.\n"
            + "\n".join(lines) + "\n"
        )
    _save_missing(missing)
    print(f"masks  wrote={n_ok} missing={n_miss} -> {dest_root}")
    return n_ok, n_miss


def _mask_note(folder: Path, roi: str) -> str:
    names = sorted(
        p.name for p in folder.glob(f"{MASK_PREFIX}{roi}*.png") if not p.stem.endswith("_preview")
    )
    return "mask: " + (", ".join(names) + " (+ _preview.png)" if names else "none available on PhenoCam")


NOTE_NAME = "outlier_phases.txt"
_COMP_PHASES = {"greenup": ("SOS", "MOS"), "senescence": ("DOS", "EOS")}


def _outlier_site_vegs(comp_table: pd.DataFrame, lag_table: pd.DataFrame, year_list) -> dict[str, str]:
    """Veg folder per outlier site, matching where ``_download_set`` puts the images."""
    return {
        r["site"]: str(r.get("veg") or "XX").upper()
        for r in _merge_site_rows((("compression", comp_table), ("lag", lag_table)), year_list)
    }


def _score_row(clean: pd.DataFrame, site, roi):
    row = clean.loc[clean["roi"].eq(roi)]
    if row.empty:
        row = clean.loc[clean["site"].eq(site)]
    return row.iloc[0] if len(row) else None


def _doy(row, col):
    if row is None or col not in row.index or pd.isna(row[col]):
        return None
    return int(round(float(row[col])))


def _image_name(folder: Path, series: str, phase: str, year: int, doy, cam: str, missing) -> str:
    iso = _doy_to_date(year, doy)
    if iso is None:
        return f"{series}_{phase}: (no date)"
    name = f"{series}_{phase}_{iso}.jpg"
    if (folder / name).exists():
        return name
    if (cam, iso) in missing:
        return f"{name} (no image: camera had no photo within 2 days)"
    return f"{name} (not downloaded yet: rerun without --skip-download)"


def _odd_span(span: dict[str, int | None]) -> str:
    vals = {k: v for k, v in span.items() if v is not None}
    if len(vals) < 3:
        return "unknown (missing a phase date)"
    mid = sorted(vals.values())[1]
    return max(vals, key=lambda k: abs(vals[k] - mid)).upper()


def _lag_suspect(flags_at_phase: set[str]) -> str:
    if {"NDVI", "GCC"} <= flags_at_phase:
        return "GVF (disagrees with both NDVI and GCC at this phase)"
    ref = next(iter(flags_at_phase))
    other = "GCC" if ref == "NDVI" else "NDVI"
    return f"{ref} (GVF still agrees with {other} at this phase)"


def write_outlier_notes(scores_by_year: dict, site_vegs: dict[str, str]) -> int:
    """Write ``{VEG}/{site}/{year}/outlier_phases.txt`` for every flagged site-year.

    Recomputed from scores on each run (same fences as the CSVs), so stale notes
    are removed first and new outliers always get one.
    """
    for old in OUTLIER_DIR.rglob(NOTE_NAME):
        old.unlink()

    missing = _load_missing()
    written = 0
    for year in sorted(scores_by_year):
        clean = scores_by_year[year]
        lag = find_lag_box_outliers(clean)
        comp = find_compression_ref_outliers(clean)
        sites = sorted(set(lag.get("site", [])) | set(comp.get("site", [])))
        for site in sites:
            veg = site_vegs.get(site) or "XX"
            folder = OUTLIER_DIR / veg / site / str(year)
            folder.mkdir(parents=True, exist_ok=True)
            lag_s = lag[lag["site"].eq(site)] if len(lag) else lag
            comp_s = comp[comp["site"].eq(site)] if len(comp) else comp
            rois = sorted(set(lag_s.get("roi", [])) | set(comp_s.get("roi", [])))

            lines = [f"{site} ({veg}) {year} - why this site-year is an outlier", ""]
            for roi in rois:
                row = _score_row(clean, site, roi)
                cam = _phenocam_sitename(roi)
                lines.append(f"ROI: {roi}   (camera: {cam})")
                lines.append(_mask_note(folder, roi))
                lines.append("")

                def imgs(ph):
                    return " | ".join(
                        _image_name(folder, s, ph, year, _doy(row, f"{s}_{ph.lower()}"), cam, missing)
                        for s in ("gvf", "gcc", "ndvi")
                    )

                lr = lag_s[lag_s["roi"].eq(roi)].drop_duplicates(["reference", "phase"]) if len(lag_s) else lag_s
                lines.append("LAG OUTLIERS (dot on the lag box plot; lag = GVF date - PhenoCam date)")
                if len(lr):
                    by_phase: dict[str, set[str]] = {}
                    for r in lr.itertuples():
                        by_phase.setdefault(r.phase, set()).add(r.reference)
                    for r in lr.sort_values(["phase", "reference"]).itertuples():
                        ref = r.reference.lower()
                        g, x = _doy(row, f"gvf_{r.phase.lower()}"), _doy(row, f"{ref}_{r.phase.lower()}")
                        lines.append(
                            f"  {r.phase} vs {r.reference}: lag {r.lag_days:+.0f} d "
                            f"(GVF DOY {g}, {r.reference} DOY {x}; normal range {r.fence_lo:+.0f} to {r.fence_hi:+.0f} d)"
                        )
                    for phase, refs in sorted(by_phase.items()):
                        lines.append(f"  -> {phase} likely odd series: {_lag_suspect(refs)}")
                        lines.append(f"     compare: {imgs(phase)}")
                else:
                    lines.append("  none")
                lines.append("")

                cr = comp_s[comp_s["roi"].eq(roi)].drop_duplicates("metric") if len(comp_s) else comp_s
                lines.append("COMPRESSION OUTLIERS (GVF season length / PhenoCam season length)")
                if len(cr):
                    for r in cr.itertuples():
                        p0, p1 = _COMP_PHASES[r.metric]
                        span = {
                            s: (None if _doy(row, f"{s}_{p0.lower()}") is None or _doy(row, f"{s}_{p1.lower()}") is None
                                else _doy(row, f"{s}_{p1.lower()}") - _doy(row, f"{s}_{p0.lower()}"))
                            for s in ("gvf", "gcc", "ndvi")
                        }
                        odd = _odd_span(span)
                        lines.append(
                            f"  {r.metric} ({p0}->{p1}): vs GCC {r.comp_gcc:.2f}x, vs NDVI {r.comp_ndvi:.2f}x, "
                            f"d = {r.delta:+.2f} (normal range {r.fence_lo:+.2f} to {r.fence_hi:+.2f}, or |d| >= 5)"
                        )
                        lines.append(
                            f"    {p0}->{p1} length: GVF {span['gvf']} d, GCC {span['gcc']} d, NDVI {span['ndvi']} d"
                        )
                        lines.append(f"  -> likely odd series: {odd} (its {p0}->{p1} span is farthest from the other two)")
                        for ph in (p0, p1):
                            lines.append(f"     {ph}: {imgs(ph)}")
                else:
                    lines.append("  none")
                lines.append("")

                lines.append("PHASE DOYs        GVF   GCC  NDVI")
                for ph in PHASE_KEYS:
                    vals = [_doy(row, f"{s}_{ph.lower()}") for s in ("gvf", "gcc", "ndvi")]
                    lines.append(f"  {ph:<14}" + "".join(f"{'-' if v is None else v:>6}" for v in vals))
                lines.append("")

            lines.append(
                "'Likely odd series' is a hint, not a verdict. Lag: flagged against both NDVI and GCC points at GVF; "
                "flagged against one points at that reference. Compression: the series whose season span is "
                "farthest from the other two. Check the images."
            )
            (folder / NOTE_NAME).write_text("\n".join(lines) + "\n")
            written += 1
    print(f"notes  wrote {written} {NOTE_NAME} files under {OUTLIER_DIR}")
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=0, help="max sites to download per set (0 = all)")
    p.add_argument("--skip-download", action="store_true", help="only rewrite the CSVs with phase DOYs")
    p.add_argument("--no-rewrite", action="store_true", help="do not rewrite the outlier/median CSVs")
    p.add_argument("--set", dest="which", choices=("outlier", "good", "both"), default="both")
    p.add_argument("--no-masks", action="store_true", help="do not write ROI mask PNGs / previews")
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
    site_vegs = _outlier_site_vegs(comp_table, lag_table, year_list)

    def masks() -> None:
        if args.no_masks:
            return
        if args.which in ("outlier", "both"):
            write_roi_masks(_outlier_roi_jobs(scores_by_year, site_vegs), OUTLIER_DIR, year_list)
        if args.which in ("good", "both"):
            write_roi_masks(_merge_site_rows((("median", median_table),), year_list), GOOD_DIR, year_list)

    if args.skip_download:
        masks()
        write_outlier_notes(scores_by_year, site_vegs)
        print("skip-download: CSVs updated; image folders reorganized")
        return 0

    jpeg_cache: dict[tuple[str, str], Path] = {}
    _seed_cache([OUTLIER_DIR, GOOD_DIR], jpeg_cache)
    missing = _load_missing()
    all_rows: list[dict] = []
    n_ok = n_miss = n_skip = 0
    if args.which in ("outlier", "both"):
        jobs = _outlier_roi_jobs(scores_by_year, site_vegs)
        print(f"outlier site-ROIs={len(jobs)} -> {OUTLIER_DIR}")
        ok, miss, skip, rows = _download_set(
            jobs, OUTLIER_DIR, year_list, jpeg_cache=jpeg_cache, missing=missing, limit=args.limit,
        )
        _save_missing(missing)
        n_ok += ok
        n_miss += miss
        n_skip += skip
        all_rows.extend(rows)
    if args.which in ("good", "both"):
        sites = _merge_site_rows((("median", median_table),), year_list)
        print(f"median sites={len(sites)} -> {GOOD_DIR}")
        ok, miss, skip, rows = _download_set(
            sites, GOOD_DIR, year_list, jpeg_cache=jpeg_cache, missing=missing, limit=args.limit,
        )
        _save_missing(missing)
        n_ok += ok
        n_miss += miss
        n_skip += skip
        all_rows.extend(rows)

    masks()
    write_outlier_notes(scores_by_year, site_vegs)
    man = COMBINED / "phenocam_phase_images_manifest.csv"
    pd.DataFrame(all_rows).to_csv(man, index=False)
    print(f"done  ok={n_ok} miss={n_miss} skip_no_doy={n_skip}  manifest={man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
