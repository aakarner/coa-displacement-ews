"""Build address-review handoffs after importing observations and refreshing screening."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative):
    with (ROOT / relative).open(newline="") as f:
        return list(csv.DictReader(f))


def write(relative, rows, fields):
    with (ROOT / relative).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


review = read("data/hays_eviction_address_review.csv")
lookup = {r["case_key"]: r for r in review}
queue = read("output/hays_eviction_austin_lookup_queue.csv")
inventory = read("output/hays_eviction_cases_extracted.csv")
inventory_keys = {r["case_key"] for r in inventory}
deferred = read("output/hays_eviction_deferred_outside_property_groups.csv")
assert len(lookup) == len(review)
assert len(queue) == len({r["case_key"] for r in queue})
assert len(inventory_keys) == len(inventory) == len(queue) + len(deferred)
assert {r["case_key"] for r in queue}.isdisjoint(r["case_key"] for r in deferred)
assert inventory_keys == {r["case_key"] for r in queue + deferred}
source_fields = ("case_key", "court", "case_number", "filing_date", "plaintiff", "source_file", "source_page")
for source in inventory:
    assert all(lookup[source["case_key"]][key] == source[key] for key in source_fields)
extra = ["property_group_name", "property_group_id", "effective_filing_date", "current_window_status", "lookup_priority"]
fields = list(review[0]) + extra
active = [dict(lookup[r["case_key"]], **{k: r[k] for k in extra}) for r in queue]
current = [r for r in active if r["current_window_status"] == "in_current_two_year_window"]
exceptions = [r for r in active if r["portal_status"] != "addresses_captured" or
              r["premises_verification"] == "candidate_multiple_defendant_addresses"]
single = [r for r in active if r["portal_status"] == "addresses_captured" and
          r["premises_verification"] in ("candidate_party_address", "candidate_party_address_only")]
assert len(single) + len(exceptions) == len(active)
assert all(r["portal_status"] != "not_started" for r in active)
assert all(r["effective_filing_date"] for r in active)
write("output/hays_eviction_collected_active_cases.csv", active, fields)
write("output/hays_eviction_current_window_address_review.csv", current, fields)
write("output/hays_eviction_address_review_exceptions.csv", exceptions, fields)
write("output/hays_eviction_single_candidate_addresses.csv", single, fields)

baseline = read("data/raw_hays_evictions/portal_observations/review_before_portal_import.csv")
for old in baseline:
    new = lookup[old["case_key"]]
    if old["portal_status"] != "not_started":
        assert all(old[key] == new[key] for key in old if key not in source_fields)
    for key in ("case_key", "court", "case_number", "verified_premises_address", "verified_unit", "austin_boundary_status"):
        assert old[key] == new[key]
snapshot = ROOT / "data/raw_hays_evictions/snapshots/before_jp4_update_2026-10-07/hays_eviction_address_review.csv"
if snapshot.exists():
    for old in read(snapshot):
        if old["portal_status"] != "not_started":
            assert all(old[k] == lookup[old["case_key"]][k] for k in old if k not in source_fields)
observations = []
for path in sorted((ROOT / "data/raw_hays_evictions/portal_observations").glob("20??-??-??.jsonl")):
    observations.extend(json.loads(line) for line in path.read_text().splitlines())
assert len(observations) == len({(r["court"], r["case_number"]) for r in observations})
assert len({r["portal_url"] for r in observations}) == len(observations)
summary = dict(
    active_cases=len(active), active_case_statuses=dict(Counter(r["portal_status"] for r in active)),
    new_case_detail_observations=len(observations), active_single_candidate_cases=len(single),
    active_multiple_address_cases=sum(r["premises_verification"] == "candidate_multiple_defendant_addresses" for r in active),
    active_exceptions=len(exceptions), active_current_window_cases=len(current),
    current_window_review_statuses=dict(Counter(r["premises_verification"] for r in current)),
    active_outside_window_cases=len(active)-len(current), unstarted_active_cases=0,
    canonical_inventory_cases=len(inventory), historical_review_rows=len(review)-len(inventory),
    deferred_cases=len(deferred),
    deferred_unstarted_cases=sum(lookup[r["case_key"]]["portal_status"] == "not_started" for r in deferred),
    verification="Party addresses only; no new premises verification or boundary assignment.",
    checks=dict(unique_case_observations="pass", unique_case_detail_urls="pass",
                all_active_cases_attempted="pass", canonical_source_fields_match="pass",
                previous_manual_observations_preserved="pass", verified_fields_preserved="pass"),
    review_log_sha256=hashlib.sha256((ROOT / "data/hays_eviction_address_review.csv").read_bytes()).hexdigest())
(ROOT / "output/hays_eviction_collection_completion.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
