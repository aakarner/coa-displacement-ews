"""Stage evidence-backed home links for the 24 remaining original-audit cases."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'tmp/oak_ranch_20261007'


def address(value):
    value = value.upper().replace('.', '').split(',')[0]
    value = re.sub(r'\s+#.*$', '', value)
    value = re.sub(r'\b(DRIVE|CIRCLE|LANE|BEND)\b', lambda m:
                   {'DRIVE':'DR','CIRCLE':'CIR','LANE':'LN','BEND':'BND'}[m[0]], value)
    return ' '.join(value.split())


def main():
    homes = defaultdict(list)
    for r in csv.DictReader((WORK/'home_spatial_review.csv').open()):
        homes[address(r['account_address'])].append(r)
    cases = {r['case_number']:r for r in csv.DictReader(
        (ROOT/'output/residential_property_batch2/case_review_results.csv').open())
        if r['review_status'] == 'unresolved_manufactured_home_inventory_and_geography'}
    evidence = defaultdict(list)
    for r in csv.DictReader((ROOT/'output/residential_property_batch2/case_address_evidence.csv').open()):
        if r['case_number'] in cases:
            evidence[r['case_number']].append(r)
    out = []
    for case, original in cases.items():
        candidates, issues = {}, []
        for r in evidence[case]:
            raw_address = r['address_for_geocoding']
            matches = homes[address(raw_address)]
            if len(matches) != 1:
                issues.append('nonunique_home_address'); continue
            home = matches[0]
            if home['spatial_ready'] != 'TRUE': issues.append('home_location_review_incomplete')
            unit = re.search(r'#\s*(\w+)', raw_address)
            if unit and home.get('space_fields_verified') == 'FALSE':
                issues.append('home_space_fields_unresolved')
            if unit and unit[1] not in {home['legal_space'],home['description_space'],home['address_space']}:
                issues.append('case_space_disagrees')
            candidates[home['parcel_id']] = home
        if len(candidates) != 1: issues.append('case_not_unique_home')
        home = next(iter(candidates.values())) if len(candidates) == 1 else {}
        out.append(dict(case_number=case, file_date=original['file_date'],
                        source_addresses='; '.join(sorted({r['address_for_geocoding'] for r in evidence[case]})),
                        home_parcel_id=home.get('parcel_id',''), original_hex=original['assigned_hex_key'],
                        reviewed_hex=home.get('point_hex_id',''), reviewed_address=home.get('city_address',''),
                        longitude=home.get('longitude',''), latitude=home.get('latitude',''),
                        status='ready_for_batched_integration' if not issues else ';'.join(sorted(set(issues))),
                        production_applied=False))
    with (WORK/'case_home_crosswalk.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    summary=dict(cases=len(out), ready=sum(r['status']=='ready_for_batched_integration' for r in out),
                 distinct_homes=len({r['home_parcel_id'] for r in out}),
                 changed_hexes=sum(r['original_hex']!=r['reviewed_hex'] for r in out),
                 production_applied=False)
    (WORK/'case_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
