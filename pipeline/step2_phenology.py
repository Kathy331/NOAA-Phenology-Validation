"""Step 2: phenology-agreement scores for Step 1 survivors.

The NDVI-vs-GCC agreement scores (phenophase gap + DTW/step + divergence) are
pre-computed in site_metadata_clean.json, so this step is a pass-through: it
attaches each survivor's cached scores and writes step2_phenology.json. No
PhenoCam call is made.
"""

import json

import config


def score_site(entry: dict) -> dict:
	"""Attach the cached phenology scores for one survivor site-year entry."""
	pheno = entry.get("phenology") or {}
	scored = pheno.get("divergence_score") is not None
	return {
		"name": entry["name"],
		"lat": entry.get("lat"),
		"lon": entry.get("lon"),
		"year": entry["year"],
		"metadata": pheno if scored else None,
		"status": "ok" if scored else "no_cached_score",
	}


def run_step2(entries: list[dict]) -> list[dict]:
	"""Collect cached scores for all survivor entries and write step2_phenology.json."""
	results = [score_site(entry) for entry in entries]
	results.sort(key=lambda r: (r["name"], r["year"]))
	scored = [r for r in results if r.get("metadata")]

	payload = {
		"counts": {"evaluated": len(results), "scored": len(scored)},
		"sites": results,
	}
	config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	config.STEP2_JSON.write_text(json.dumps(payload, indent=2) + "\n")
	print(f"Step 2: attached cached scores for {len(scored)}/{len(results)} survivors -> {config.STEP2_JSON}")
	return results
