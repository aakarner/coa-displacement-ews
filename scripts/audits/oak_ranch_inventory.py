"""Audit Oak Ranch home accounts against independent City address points.

Reads the retained 2025 county extraction and 2026-10-07 City response. Writes
review/staging artifacts only; never changes promoted units or case assignments.
One manufactured-home account is one candidate home, not one home per section.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'tmp/oak_ranch_20261007'
REVIEW = ROOT / 'data/reviewed_manufactured_housing/oak_ranch_20261007/exception_review_01/account_reviews.json'


def source_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def clean(value):
    return ' '.join(str(value or '').upper().strip().split())


def suffix(value):
    value = clean(value).rstrip('.')
    return {'DRIVE': 'DR', 'CIRCLE': 'CIR', 'LANE': 'LN', 'BEND': 'BND'}.get(value, value)


def key(number, street):
    return clean(number), clean(street).replace(' ', '')


def serial(text):
    m = re.search(r'\bSN(?:\d+(?:/\d+)?)?\s*[#: ]+\s*(.*?)(?=;|,|\s+HUD|$)', text, re.I)
    return re.sub('[^A-Z0-9/]', '', m[1].upper()) if m else ''


def write_csv(name, rows):
    with (WORK/name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows = [json.loads(line) for line in (WORK/'county_m1_source_records.jsonl').open()]
    homes = [r for r in all_rows if any(d.get('asCode') == 'M00109'
                                       for d in r['propertyLegalDescription'])]
    assert len({r['pID'] for r in homes}) == len(homes)
    ids_by_serial = defaultdict(set)
    for r in all_rows:
        for legal in r['propertyLegalDescription']:
            value = serial(legal.get('legalDescription') or '')
            if value:
                ids_by_serial[value].add(r['pID'])
    city = defaultdict(list)
    city_points = json.loads((WORK/'city_area_addresses.json').read_text())['features']
    points_by_id = {str(r['attributes']['PLACE_ID']): r for r in city_points}
    assert len(points_by_id) == len(city_points)
    reviews = json.loads(REVIEW.read_text())['accounts']
    assert len({r['parcel_id'] for r in reviews}) == len(reviews)
    reviews = {r['parcel_id']: r for r in reviews}
    assert set(reviews) <= {str(r['pID']) for r in homes}
    for r in city_points:
        a = r['attributes']
        city[key(a['ADDRESS'], a['STREET_NAME'])].append(r)
    address_counts = Counter(key(r['situses'][0]['streetNum'], r['situses'][0]['streetName']) for r in homes)
    rows = []
    for r in homes:
        legal = r['propertyLegalDescription'][0]
        situs = r['situses'][0]
        profile = r['propertyProfile'][0]
        text = legal['legalDescription']
        sn = serial(text)
        k = key(situs['streetNum'], situs['streetName'])
        matches = city[k]
        point = matches[0] if len(matches) == 1 else None
        typed_space = clean(legal.get('mhSpaceNum'))
        match = re.search(r'\bSPACE\s+(\d+)', text)
        text_space = match[1] if match else ''
        secondary = clean(situs.get('streetSecondary'))
        issues = []
        if r['inactive'] != 0: issues.append('inactive_account')
        if sn and len(ids_by_serial[sn]) > 1: issues.append('serial_on_multiple_accounts')
        if not sn: issues.append('serial_missing')
        if address_counts[k] > 1: issues.append('address_shared_by_home_accounts')
        if len({v for v in [typed_space, text_space, secondary] if v}) > 1:
            issues.append('space_fields_disagree')
        if not point: issues.append('city_address_not_unique_or_missing')
        if point and suffix(situs.get('streetSuffix')) != suffix(point['attributes']['STREET_TYPE']):
            issues.append('street_suffix_differs')
        # Reviews are bound to the exact retained county record and source issues.
        # Preserve raw flags even when independent evidence resolves the location.
        review = reviews.get(str(r['pID']))
        accepted = False
        if review:
            assert review['source_record_sha256'] == source_hash(r), r['pID']
            assert review['source_issues'] == issues, r['pID']
            accepted = review['disposition'] == 'accept_home_count_and_city_location'
            if accepted:
                assert not set(issues) & {'inactive_account', 'serial_on_multiple_accounts', 'address_shared_by_home_accounts'}
                point = points_by_id[review['accepted_city_place_id']]
                assert point['attributes']['FULL_STREET_NAME'] == review['accepted_city_address']
        ready = (not issues or accepted) and r['propType'] == 'MH' and profile['stateCd'] in {'M1', 'A2'}
        row = dict(parcel_id=str(r['pID']), tax_year=r['pYear'], active=r['inactive'] == 0,
                   property_type=r['propType'], state_code=profile['stateCd'],
                   home_serial=sn, legal_space=typed_space, description_space=text_space,
                   address_space=secondary, account_address=' '.join(v for v in
                       [clean(situs['streetNum']), clean(situs.get('streetName')), clean(situs.get('streetSuffix'))] if v),
                   legal_description=text, source_geo_id=r['propertyIdentification'][0]['geoID'],
                   candidate_units=1, ready_before_spatial_validation=ready, review_issues=';'.join(issues),
                   exception_review_status=review['disposition'] if review else 'not_required',
                   unresolved_location_issues='' if accepted else ';'.join(issues),
                   space_fields_verified='space_fields_disagree' not in issues,
                   city_address=point['attributes']['FULL_STREET_NAME'] if point else '',
                   city_place_id=point['attributes']['PLACE_ID'] if point else '',
                   city_created_date=point['attributes']['CREATED_DATE'] if point else '',
                   city_modified_date=point['attributes']['MODIFIED_DATE'] if point else '',
                   longitude=point['geometry']['x'] if point else '', latitude=point['geometry']['y'] if point else '')
        rows.append(row)
    write_csv('home_inventory_review.csv', rows)
    summary = {'homes':len(rows), 'active':sum(r['active'] for r in rows),
                      'serial_present':sum(bool(r['home_serial']) for r in rows),
                      'unique_city_point_candidates':sum(bool(r['city_place_id']) for r in rows),
                      'ready_before_spatial_validation':sum(r['ready_before_spatial_validation'] for r in rows),
                      'accepted_exception_reviews':sum(r['exception_review_status']=='accept_home_count_and_city_location' for r in rows),
                      'issue_counts':dict(Counter(i for r in rows for i in r['review_issues'].split(';') if i))}
    (WORK/'inventory_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
