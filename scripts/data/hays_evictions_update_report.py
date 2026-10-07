"""Audit the canonical September JP4 update against the saved pre-update state."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "data/raw_hays_evictions/snapshots/before_jp4_update_2026-10-07"


def read(path):
    return list(csv.DictReader(path.open(newline="")))


def keyed(path, key="case_key"):
    records = read(path)
    result = {r[key]: r for r in records}
    assert len(result) == len(records), path
    return result


old = keyed(SNAPSHOT / "hays_eviction_cases_extracted.csv")
current = keyed(ROOT / "output/hays_eviction_cases_extracted.csv")
review = keyed(ROOT / "data/hays_eviction_address_review.csv")
screen = keyed(ROOT / "output/hays_eviction_case_geography_screen.csv")
geo = keyed(ROOT / "output/hays_eviction_current_window_geography_review.csv")
added = sorted(set(current) - set(old))
removed = sorted(set(old) - set(current))
assert len(added) == 15 and len(removed) == 2
assert {old[k]["case_number"] for k in removed} == {"F19-006J4", "F19-008J4"}
assert all(current[k] == row for k, row in old.items() if row["court"] == "JP5")
jp4 = [r for r in current.values() if r["court"] == "JP4"]
assert len(jp4) == 206 and len(current) == 1170
assert {r["report_selection"] for r in jp4} == {"filed_date"}
assert len({r["source_file"] for r in jp4}) == 1
assert all(r["filing_date"] for r in jp4)
checked = [r for r in jp4 if review[r["case_key"]]["portal_filing_date"]]
assert all(r["filing_date"] == review[r["case_key"]]["portal_filing_date"] for r in checked)
old_geo = keyed(SNAPSHOT / "hays_eviction_current_window_geography_review.csv")
assert all(geo[k]["candidate_boundary_status"] == r["candidate_boundary_status"]
           and geo[k]["geocode_method"] == r["geocode_method"] for k, r in old_geo.items())
old_ids = keyed(SNAPSHOT / "hays_eviction_candidate_addresses_geocoded_local.csv", "address_id")
new_ids = keyed(ROOT / "output/hays_eviction_candidate_addresses_geocoded_local.csv", "address_id")
assert all(new_ids[k]["candidate_premises_address"] == r["candidate_premises_address"]
           for k, r in old_ids.items())
handoff = []
for key in added:
    source, observed, triage = current[key], review[key], screen[key]
    assert observed["portal_status"] in {"addresses_captured", "no_defendant_address"}
    window = "2024-04-02" <= source["filing_date"] <= "2026-04-01"
    if observed["premises_verification"] == "candidate_multiple_defendant_addresses":
        result = "unresolved_conflicting_party_addresses"
    elif triage["triage_action"] == "defer_outside_property_group":
        result = "outside_property_group_screen"
    elif key in geo:
        result = geo[key]["candidate_boundary_status"]
    else:
        result = "outside_analysis_window_not_individually_geocoded"
    handoff.append(dict(case_number=source["case_number"], filing_date=source["filing_date"],
        in_analysis_window=window, source_page=source["source_page"],
        portal_status=observed["portal_status"], portal_url=observed["portal_url"],
        observed_on=observed["observed_on"], defendant_addresses_raw=observed["defendant_addresses_raw"],
        candidate_address=observed["candidate_premises_address"],
        premises_verification=observed["premises_verification"],
        property_group=triage["property_group_name"], preliminary_result=result,
        parcel_ids=geo.get(key, {}).get("parcel_ids", ""),
        distance_to_austin_m=geo.get(key, {}).get("distance_to_austin_m", "")))
with (ROOT / "output/hays_eviction_jp4_update_cases.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(handoff[0]))
    writer.writeheader()
    writer.writerows(handoff)
current_added = [r for r in handoff if r["in_analysis_window"]]
assert len(current_added) == 7
summary = dict(canonical_jp4_cases=len(jp4), canonical_combined_cases=len(current),
    added_cases=len(added), removed_out_of_period_cases=len(removed),
    new_cases_with_addresses=sum(r["portal_status"] == "addresses_captured" for r in handoff),
    new_cases_without_addresses=sum(r["portal_status"] == "no_defendant_address" for r in handoff),
    jp4_current_window_cases=sum("2024-04-02" <= r["filing_date"] <= "2026-04-01" for r in jp4),
    added_current_window_cases=len(current_added),
    added_current_window_results=dict(Counter(r["preliminary_result"] for r in current_added)),
    source_portal_filing_dates_checked=len(checked),
    prior_current_window_geography_preserved=len(old_geo),
    prior_address_ids_preserved=len(old_ids),
    source_sha256=hashlib.sha256((ROOT / jp4[0]["source_file"]).read_bytes()).hexdigest(),
    checks="pass", premises_remain_unverified=True)
(ROOT / "output/hays_eviction_jp4_update_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
parcels = read(ROOT / "output/hays_eviction_property_parcel_evidence.csv")
assert all(int(r["geometry_features"]) > 0 for r in parcels)
assert set(screen) == set(current)
deferred = [r for r in screen.values() if r["triage_action"] == "defer_outside_property_group"]
assert all(r["property_polygon_status"] == "outside_austin_full"
           and r["requires_identity_review"] == "FALSE"
           and r["premises_verification"] != "candidate_multiple_defendant_addresses" for r in deferred)
source_paths = {r["source_file"] for r in current.values()}
qa = dict(reviewed_on="2026-10-07", canonical_cases=len(current),
    case_partition="pass", unique_case_keys="pass",
    all_registered_parcels_have_geometry="pass",
    only_conflict_free_outside_groups_deferred="pass",
    canonical_source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sorted(source_paths)},
    inventory_sha256=hashlib.sha256((ROOT / "output/hays_eviction_cases_extracted.csv").read_bytes()).hexdigest(),
    review_log_sha256=hashlib.sha256((ROOT / "data/hays_eviction_address_review.csv").read_bytes()).hexdigest())
(ROOT / "output/hays_eviction_geography_screen_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
print(json.dumps(summary, indent=2))
