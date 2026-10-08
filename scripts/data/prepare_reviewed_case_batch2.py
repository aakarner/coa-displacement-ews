"""Pin the 31 supported batch-2 cases without altering their source records."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path("data/reviewed_eviction_properties/batch2_20261007")
AUDIT = Path("output/residential_property_batch2")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    config_path = Path("config/eviction_property_reviews.json")
    config = json.loads(config_path.read_text())
    assert all(b["batch_id"] != "batch2_20261007" for b in config["batches"])
    assert not (ROOT / "cases.json").exists(), "Do not overwrite a completed review bundle"
    ROOT.mkdir(parents=True, exist_ok=True)
    evidence = []
    for path in [AUDIT / "case_address_evidence.csv", AUDIT / "case_review_queue.csv",
                 AUDIT / "canyon_creek_review.json", AUDIT / "canyon_creek_public_sources.json",
                 *sorted((AUDIT / "reviewed_cases").glob("*")),
                 Path("data/reviewed_unit_properties/batch2_20261007/public_sources.json"),
                 Path("data/reviewed_unit_properties/batch2_20261007/county_account_attributes.json")]:
        dest = ROOT / "evidence" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert sha(dest) == sha(path), "Partial evidence differs: " + str(dest)
        else:
            shutil.copy2(path, dest)
        evidence.append({"path": str(dest), "sha256": sha(dest)})
    specs = {
        "3319": ("498141", "Bridge at Canyon Creek", 3321, 18),
        "3261": ("WILLIAMSON:R500219", "Caliza", 3261, 9),
        "6929": ("911866", "Nexus at Goodnight Ranch", 6929, 3),
        "602": ("859326", "Ocotillo", 596, 1),
    }
    address_rows = rows(AUDIT / "case_address_evidence.csv")
    queue = rows(AUDIT / "case_review_queue.csv")
    cases = []
    for cell, (parcel, name, dest, count) in specs.items():
        group = [r for r in queue if r["hex_id"] == cell]
        assert len(group) == count
        for meta in group:
            assert meta["assignment_status"] == "assigned_unique_hex"
            raw = [r for r in address_rows if r["case_number"] == meta["case_number"]]
            addresses = {}
            for r in raw:
                address = {"address_for_geocoding": r["address_for_geocoding"],
                           "longitude": float(r["longitude"]), "latitude": float(r["latitude"])}
                if address["address_for_geocoding"] in addresses:
                    assert addresses[address["address_for_geocoding"]] == address
                addresses[address["address_for_geocoding"]] = address
            assert addresses
            basis = "reviewed_property_address_and_county_account"
            if meta["case_number"] == "TRAVIS:JP2:J2-CV-25-004162":
                basis += "_with_case_specific_court_name"
            if cell in ("3261", "6929"):
                basis += "_explicit_boundary_reference"
            cases.append({
                "review_id": "batch2_20261007:" + meta["case_number"],
                "case_number": meta["case_number"], "source_county": meta["source_county"],
                "source_jp_district": meta["source_jp_district"], "file_date": meta["file_date"],
                "property_name": name, "parcel_id": parcel, "project_id": "project:" + parcel,
                "expected_unit_hex_id": dest, "review_basis": basis,
                "apartment_conflict": False, "addresses": sorted(addresses.values(),
                    key=lambda r: r["address_for_geocoding"]),
                "evidence_paths": [e["path"] for e in evidence],
            })
    assert len(cases) == 31 and len({c["case_number"] for c in cases}) == 31
    bundle = {"schema_version": 1, "batch_id": "batch2_20261007", "reviewed_on": "2026-10-07",
              "scope": "18 Canyon Creek, 9 Caliza, 3 Nexus, 1 Ocotillo; only one case individually court-reviewed",
              "evidence": evidence, "cases": cases}
    path = ROOT / "cases.json"
    path.write_text(json.dumps(bundle, indent=2) + "\n")
    config["batches"].append({"batch_id": bundle["batch_id"], "path": str(path),
        "sha256": sha(path), "case_count": len(cases), "scope": bundle["scope"]})
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print("Prepared 31 pinned case reviews; 24 mobile-home cases remain unresolved.")


if __name__ == "__main__":
    main()
