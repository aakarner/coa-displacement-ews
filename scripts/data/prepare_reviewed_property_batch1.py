"""Promote the completed October 7 review to immutable local pipeline inputs.

This is an explicit review-promotion step, not part of routine rebuilding.
Case details and screenshot evidence stay in the gitignored data directory.
"""
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing review input differs; create a new batch: {path}")
    else:
        path.write_bytes(content)


def main():
    audit = Path("output/residential_property_batch1")
    root = Path("data/reviewed_eviction_properties/batch1_20261007")
    evidence = {}

    def retain(path):
        path = Path(path)
        target = root / "evidence" / path
        write_once(target, path.read_bytes())
        evidence[str(target)] = {"path": str(target), "sha256": digest(target)}
        return str(target)

    notes = retain("docs/audits/residential-property-batch1-2026-10.md")
    common = [retain(audit / name) for name in (
        "case_address_evidence.csv", "bell_case_review.csv", "bell_unit_matches.csv",
        "bell_site_map.geojson", "bell_mapped_unit_counts.csv", "account_evidence.csv",
        "county_account_links.csv", "source_manifest.json")]
    ledger = {r["case_number"]: r for r in csv_rows(Path("output/part2/evictions/eviction_case_ledger.csv"))}
    automatic = {r["case_number"]: r for r in csv_rows(audit / "bell_case_review.csv")}
    reviews = {r["case_number"]: (r, p) for p in sorted((audit / "reviewed_cases").glob("*.json"))
               for r in [json.loads(p.read_text())]}
    assert len(automatic) == 54 and len(reviews) == 23
    addresses = {}
    for row in csv_rows(audit / "case_address_evidence.csv"):
        if row["st_addr"] in ("10500 S Interstate 35", "10505 S Interstate 35", "8515 S Interstate 35"):
            addresses.setdefault(row["case_number"], {})[row["address_for_geocoding"]] = row
    cases = []
    for case, raw in sorted(addresses.items()):
        rows = list(raw.values())
        streets = {r["st_addr"] for r in rows}
        assert len(streets) == 1
        street = next(iter(streets))
        meta = ledger[case]
        assert meta["assignment_status"] == "assigned_unique_hex" and meta["source_county"] == "Travis"
        conflict = False
        sources = [notes] + common
        if street == "10500 S Interstate 35":
            property_name = "Bell Southpark"
            if case in reviews:
                r, path = reviews[case]
                assert r["review_status"] == "property_verified_court_record"
                assert r["court_file_date"] == meta["file_date"]
                assert r["court_location"] == "Precinct Three" and meta["source_jp_district"] == "JP3"
                parcel, project, cell = r["resolved_parcel_id"], r["resolved_project_id"], r["proposed_unit_hex_id"]
                conflict = r.get("apartment_number_verified") is False
                basis = ("case_specific_court_phase_shared_street_address_apartment_unresolved" if conflict
                         else "case_specific_court_phase_and_operator_unit_map")
                sources.append(retain(path))
                for s in [r] + r.get("supplementary_sources", []):
                    p = Path(s["source_path"])
                    assert digest(p) == s["source_sha256"]
                    sources.append(retain(p))
            else:
                a = automatic[case]
                assert a["review_status"] == "unique_phase_candidate" and a["all_addresses_matched"] == "TRUE"
                assert a["parcel_ids"] == "878332"
                parcel, project, cell = "878332", "project:878332", 6670
                basis = "reviewed_unique_operator_unit_map_phase"
        elif street == "10505 S Interstate 35":
            property_name = "Bridge at Asher"
            parcel, project, cell = "513751", "project:513751", 6973
            basis = "reviewed_operator_address_and_county_property"
        else:
            property_name = "Bridge at Monarch Bluffs"
            parcel, project, cell = "533185", "project:533185", 6964
            basis = "reviewed_operator_address_and_county_land_improvement_project"
        cases.append({
            "review_id": "batch1_20261007:" + case, "case_number": case,
            "source_county": meta["source_county"], "source_jp_district": meta["source_jp_district"],
            "file_date": meta["file_date"], "property_name": property_name,
            "parcel_id": parcel, "project_id": project, "expected_unit_hex_id": cell,
            "review_basis": basis, "apartment_conflict": conflict,
            "addresses": [{"address_for_geocoding": r["address_for_geocoding"],
                           "longitude": float(r["longitude"]), "latitude": float(r["latitude"])}
                          for r in sorted(rows, key=lambda x: x["address_for_geocoding"])],
            "evidence_paths": sorted(set(sources))})
    assert len(cases) == 71 and sum(c["apartment_conflict"] for c in cases) == 1
    assert [sum(c["property_name"] == p for c in cases) for p in
            ("Bell Southpark", "Bridge at Asher", "Bridge at Monarch Bluffs")] == [54, 6, 11]
    bundle = {"schema_version": 1, "batch_id": "batch1_20261007", "reviewed_on": "2026-10-07",
              "scope": "Case locations only; original source addresses, inclusion decisions, and unit counts preserved",
              "evidence": sorted(evidence.values(), key=lambda x: x["path"]), "cases": cases}
    path = root / "cases.json"
    write_once(path, (json.dumps(bundle, indent=2) + "\n").encode())
    config = {"schema_version": 1, "batches": [{"batch_id": bundle["batch_id"],
        "path": str(path), "sha256": digest(path), "case_count": len(cases),
        "scope": "54 Bell, 6 Asher, 11 Monarch; case-specific location corrections only"}]}
    write_once(Path("config/eviction_property_reviews.json"), (json.dumps(config, indent=2) + "\n").encode())
    print(f"Promoted {len(cases)} reviewed cases and {len(evidence)} evidence files; unit counts unchanged.")


if __name__ == "__main__":
    main()
