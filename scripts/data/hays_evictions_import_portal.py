"""Merge browser-observed public party addresses into the persistent review log.

No networking. Input JSONL contains case identifiers, filing dates, source URLs,
and party names/addresses only. It excludes demographics and financial details.
Never promotes a party address to verified eviction premises.
"""
import csv
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
review_path = ROOT / "data/hays_eviction_address_review.csv"
source_dir = ROOT / "data/raw_hays_evictions/portal_observations"


def clean_address_lines(lines):
    """Portal identifiers sometimes share address cells; they are not addresses."""
    cleaned = [re.split(r"\b(?:SID|SSN|DOB|DL|FBI|CID)\s*:", line, flags=re.I)[0].strip(" ,")
               for line in lines]
    return [line for line in cleaned if line]


def main():
    with review_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    lookup = {r["case_key"]: r for r in rows}
    assert len(lookup) == len(rows)
    observations = {}
    for path in sorted(source_dir.glob("*.jsonl")):
        if path.name == "search_attempts.jsonl":
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            for party in r["parties"]:
                party["address_lines"] = clean_address_lines(party["address_lines"])
            key = f'Hays|{r["court"]}|{r["case_number"]}'
            assert key in lookup, key
            assert r["case_number"].endswith(r["court"].replace("P", ""))
            date.fromisoformat(r["portal_filing_date"])
            assert re.fullmatch(r"https://portal-txhays\.tylertech\.cloud/PublicAccess/CaseDetail\.aspx\?CaseID=\d+", r["portal_url"])
            if key in observations and observations[key] != r:
                raise ValueError(f"Conflicting observations for {key}; review manually")
            observations[key] = r
    added = 0
    conflicts = []
    for key, obs in observations.items():
        row = lookup[key]
        if row["filing_date"] and row["filing_date"] != obs["portal_filing_date"]:
            conflicts.append({"case_key": key, "source": row["filing_date"], "portal": obs["portal_filing_date"]})
        if row["portal_status"] != "not_started":
            # Idempotent reruns and previously reviewed manual rows stay intact.
            continue
        parties = [p for p in obs["parties"] if p["role"].strip().lower() == "defendant"]
        addresses = [", ".join(p["address_lines"]) for p in parties if p["address_lines"]]
        distinct = list(dict.fromkeys(addresses))
        row.update(portal_status="addresses_captured" if addresses else "no_defendant_address",
                   portal_url=obs["portal_url"], observed_on=obs["observed_on"],
                   portal_filing_date=obs["portal_filing_date"],
                   defendant_addresses_raw=" | ".join(addresses),
                   candidate_premises_address=distinct[0] if len(distinct) == 1 else "",
                   address_evidence=f"Portal Party Information: {len(parties)} defendant(s), {len(distinct)} distinct nonblank address(es).",
                   premises_verification=("candidate_party_address" if len(distinct) == 1 else
                                          "candidate_multiple_defendant_addresses" if distinct else "missing_address"),
                   notes="Party address observed in register of actions; eviction premises and historical address are unverified. Unit text is preserved in raw/candidate address. See dated JSONL for party-to-address associations.")
        if len(addresses) < len(parties):
            row["notes"] += " At least one defendant has no listed address."
        added += 1
    attempts_path = source_dir / "search_attempts.jsonl"
    if attempts_path.exists():
        for line in attempts_path.read_text().splitlines():
            attempt = json.loads(line)
            row = lookup[attempt["case_key"]]
            assert attempt["portal_status"] == "case_not_found"
            if row["portal_status"] == "not_started":
                row.update({k: attempt[k] for k in ("portal_status", "portal_url", "observed_on", "notes")})
                added += 1
    if added:
        backup = source_dir / "review_before_portal_import.csv"
        if not backup.exists():
            backup.write_bytes(review_path.read_bytes())
        temporary = review_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(review_path)
    summary = dict(observations=len(observations), added_this_run=added,
                   total_review_rows=len(rows), portal_status_counts=dict(Counter(r["portal_status"] for r in rows)),
                   filing_date_conflicts=conflicts)
    (ROOT / "output/hays_eviction_portal_collection_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
