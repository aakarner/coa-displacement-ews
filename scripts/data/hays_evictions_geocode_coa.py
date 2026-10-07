"""Cache official Austin geocoder responses for unresolved candidate addresses.

Only address strings are sent, without party names or case numbers. Each unique
street/ZIP key is requested once. Response acceptance happens in the R review.
"""
import csv
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://maps.austintexas.gov/arcgis/rest/services/Geocode/COA_Locator/GeocodeServer/findAddressCandidates"
cache = ROOT / "data/raw_hays_evictions/geocode_cache"
cache.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((ROOT / "output/hays_eviction_candidate_geocode_followup.csv").open()))
requests = {}
for row in rows:
    address = f'{row["street_key"]}, {row["postal_city"]}, TX {row["postal_zip"]}'
    requests.setdefault(address, []).append(row["address_id"])
results = []
for address, ids in requests.items():
    key = hashlib.sha256(address.encode()).hexdigest()
    path = cache / f"{key}.json"
    if not path.exists():
        if "--network" not in sys.argv:
            continue
        params = dict(f="json", SingleLine=address, outSR="4326", outFields="*", maxLocations="3")
        try:
            with urllib.request.urlopen(ENDPOINT+"?"+urllib.parse.urlencode(params), timeout=30) as response:
                data = json.load(response)
        except Exception as e:
            print(f"Request failed ({type(e).__name__}); completed requests remain cached.", flush=True)
            raise SystemExit(1)
        if "error" in data:
            raise RuntimeError("Geocoder returned an error; inspect cache/request configuration")
        path.write_text(json.dumps(dict(query_address=address, endpoint=ENDPOINT,
            retrieved_at=datetime.now(timezone.utc).isoformat(), response=data)))
    data = json.loads(path.read_text())
    for address_id in ids:
        for rank, candidate in enumerate(data["response"].get("candidates", []), 1):
            a = candidate.get("attributes", {})
            results.append(dict(address_id=address_id, rank=rank, query_address=address,
                match_address=candidate.get("address", ""), score=candidate.get("score"),
                address_type=a.get("Addr_type", ""), longitude=candidate.get("location", {}).get("x"),
                latitude=candidate.get("location", {}).get("y"), postal=a.get("Postal", ""),
                street_number=a.get("AddNum", ""), street_name=a.get("StName", ""),
                cache_file=str(path.relative_to(ROOT))))
    if len(results) % 20 == 0:
        print(f"Processed {len(results)} candidate responses.", flush=True)
fields = ["address_id", "rank", "query_address", "match_address", "score", "address_type", "longitude",
          "latitude", "postal", "street_number", "street_name", "cache_file"]
with (ROOT / "output/hays_eviction_coa_geocode_candidates.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)
print(f"Wrote {len(results)} candidate responses for {len(requests)} distinct queries.")
