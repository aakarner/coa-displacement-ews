"""Focused checks of staged Oak Ranch records; does not rebuild production."""
import csv
import json
from collections import Counter

from oak_ranch_inventory import ROOT, WORK, REVIEW, source_hash


def read(name, directory=WORK):
    return list(csv.DictReader((directory/name).open()))


def main():
    baseline = ROOT/'data/reviewed_manufactured_housing/oak_ranch_20261007'
    original = {r['parcel_id']: r for r in read('home_inventory_review.csv', baseline)}
    inventory = {r['parcel_id']: r for r in read('home_inventory_review.csv')}
    ready = read('ready_home_locations.csv')
    reviews = json.loads(REVIEW.read_text())['accounts']
    source = {str(r['pID']): r for r in json.loads((baseline/'park_source_records.json').read_text())}
    accepted = {r['parcel_id'] for r in reviews if r['disposition'] == 'accept_home_count_and_city_location'}
    assert len(reviews) == 44 and len(accepted) == 29
    assert len(inventory) == 832 and inventory.keys() == original.keys()
    for review in reviews:
        pid = review['parcel_id']
        assert review['source_record_sha256'] == source_hash(source[pid])
        assert review['source_issues'] == original[pid]['review_issues'].split(';')
    for pid, row in inventory.items():
        for field in ['home_serial', 'legal_space', 'description_space', 'address_space',
                      'account_address', 'legal_description', 'review_issues']:
            assert row[field] == original[pid][field], (pid, field)
        if pid not in accepted:
            assert row['ready_before_spatial_validation'] == original[pid]['ready_before_spatial_validation']
        if row['review_issues'] == 'space_fields_disagree':
            assert row['space_fields_verified'] == 'False'
    ready_ids = {r['parcel_id'] for r in ready}
    baseline_ids = {r['parcel_id'] for r in read('ready_home_locations.csv', baseline)}
    assert ready_ids == baseline_ids | accepted
    assert len(ready) == len(ready_ids) == len({r['city_place_id'] for r in ready}) == 793
    assert all(r['candidate_units'] == '1' and r['spatial_ready'] == 'TRUE' for r in ready)
    assert all(r['spatial_parent_id'] in {'464309', '909849'} and r['point_hex_id'] != 'NA' for r in ready)
    existing = [r for r in ready if r['already_promoted'] == 'TRUE']
    assert {r['parcel_id'] for r in existing} == {'988420', '996455'}
    assert all(float(r['existing_units']) == 1 for r in existing)
    assert sum(r['already_promoted'] == 'FALSE' for r in ready) == 791
    assert sum(r['ready_before_spatial_validation'] == 'True' for r in inventory.values()) - len(ready) == 24
    assert sum(r['exception_review_status'] == 'unresolved_home_location' for r in inventory.values()) == 15
    cases = read('case_home_crosswalk.csv')
    previous_cases = {r['case_number']: r for r in read('case_home_crosswalk.csv', baseline)}
    assert len(cases) == 24 and len({r['home_parcel_id'] for r in cases}) == 17
    for case in cases:
        assert case['status'] == 'ready_for_batched_integration'
        assert case['home_parcel_id'] in ready_ids
        assert case['reviewed_hex'] == case['original_hex']
        assert case == previous_cases[case['case_number']]
    totals = Counter(r['point_hex_id'] for r in ready)
    assert totals['3767'] == 194 and totals['3769'] == 150
    result = dict(status='passed', ready_homes=793, additions=791, existing_units_preserved=2,
                  accepted_exceptions=29, unresolved_locations=15, outside_grid=24,
                  unchanged_case_links=24, production_applied=False)
    (WORK/'focused_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
