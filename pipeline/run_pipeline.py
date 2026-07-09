"""Orchestrator for spatial + phenology pipeline.

    site_metadata_clean.json  ->  entries (name/lat/lon/year + cached scores + MOS/DOS)
                              ->  Step 1: Earth Engine spatial uniformity (survivors)
                              ->  Step 2: reuse cached NDVI-vs-GCC scores on survivors
                              ->  pipeline_results.json (survivors ranked by divergence)

The input is the pre-computed site_metadata_clean.json (produced by src/api/main.py),
so the pipeline makes no live PhenoCam calls -- only Earth Engine (Step 1) hits the
network. Set PIPELINE_LIMIT=<N> to only process the first N site-years.
"""

import json

import config
import step1_spatial_uniformity as step1
import step2_phenology as step2


def load_entries() -> tuple[list[dict], list[str]]:
	"""Build site-year entries from the cached site_metadata_clean.json.

	Each entry carries name/lat/lon/year, the cached phenology scores, and the
	NDVI MOS/DOS DOYs used to center Step 1's peak-summer window. Entries missing
	lat/lon are reported as `missing`.
	"""
	data = json.loads(config.SITE_METADATA_JSON.read_text())

	entries: list[dict] = []
	missing: list[str] = []
	for group_data in data.values():
		for site in group_data["sites"]:
			metadata = site["metadata"]
			if site.get("lat") is None or site.get("lon") is None:
				missing.append(site["name"])
				continue
			ndvi_phases = (metadata.get("phenophases") or {}).get("ndvi", {})
			entries.append(
				{
					"name": site["name"],
					"lat": site["lat"],
					"lon": site["lon"],
					"year": metadata["year"],
					"mos": ndvi_phases.get("MOS"),
					"dos": ndvi_phases.get("DOS"),
					"phenology": {
						"year": metadata["year"],
						"phenophase_gap_days": metadata.get("phenophase_gap_days"),
						"dtw_per_step": metadata.get("dtw_per_step"),
						"divergence_score": metadata.get("divergence_score"),
					},
				}
			)

	if config.LIMIT:
		entries = entries[: config.LIMIT]
	return entries, missing


def _divergence(entry: dict) -> float:
	pheno = entry.get("phenology") or {}
	value = pheno.get("divergence_score")
	return value if value is not None else float("inf")


def combine(entries: list[dict], step1_results: list[dict], step2_results: list[dict]) -> list[dict]:
	"""Merge Step 1 spatial metrics with Step 2 phenology metrics per site-year."""
	step1_by_key = {(r["name"], r["year"]): r for r in step1_results}

	combined: list[dict] = []
	for r in step2_results:
		key = (r["name"], r["year"])
		s1 = step1_by_key.get(key, {})
		combined.append(
			{
				"name": r["name"],
				"lat": r["lat"],
				"lon": r["lon"],
				"year": r["year"],
				"spatial": {
					"cv_ndvi": s1.get("cv_ndvi"),
					"water_pct": s1.get("water_pct"),
					"urban_pct": s1.get("urban_pct"),
					"summer_doy": s1.get("summer_doy"),
					"n_images": s1.get("n_images"),
				},
				"phenology": r.get("metadata"),
				"status": r.get("status"),
			}
		)

	combined.sort(key=_divergence)
	return combined


def main() -> dict:
	entries, missing = load_entries()
	print(f"Loaded {len(entries)} site-years ({len(missing)} missing lat/lon).")
	if config.LIMIT:
		print(f"(PIPELINE_LIMIT active: capped to first {config.LIMIT} site-years.)")

	print("\n=== Step 1: Spatial uniformity (Earth Engine) ===")
	step1_results = step1.run_step1(entries)
	passed_keys = {(r["name"], r["year"]) for r in step1_results if r["passed"]}
	survivors = [e for e in entries if (e["name"], e["year"]) in passed_keys]
	print(f"Step 1 survivors: {len(survivors)}/{len(entries)}")

	print("\n=== Step 2: Phenology agreement (NDVI vs GCC) ===")
	step2_results = step2.run_step2(survivors)

	combined = combine(entries, step1_results, step2_results)

	payload = {
		"counts": {
			"evaluated": len(entries),
			"step1_passed": len(survivors),
			"step2_scored": sum(1 for c in combined if c["phenology"]),
			"missing_latlon": len(missing),
		},
		"thresholds": {
			"cv_max": config.CV_MAX,
			"water_max": config.WATER_MAX,
			"urban_max": config.URBAN_MAX,
		},
		"sites": combined,
	}

	config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	config.RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
	print(f"\nSaved final results to {config.RESULTS_JSON}")
	return payload


if __name__ == "__main__":
	main()
