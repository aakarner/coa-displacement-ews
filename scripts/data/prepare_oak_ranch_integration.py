"""Prepare pinned Oak Ranch unit/case integration from completed local reviews.

Run from the repository root. This writes review configuration only; it never
runs production stages or fits clusters.
"""
import csv,json,hashlib
from pathlib import Path
b=Path('data/reviewed_manufactured_housing/oak_ranch_20261007'); d=b/'integration'
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p, x):
    payload = json.dumps(x, indent=2) + "\n"
    if p.exists() and p.read_text() != payload:
        raise ValueError("Existing review differs; re-review before replacing: " + str(p))
    p.write_text(payload)
homes=read(b/'exception_review_01/ready_home_locations.csv'); owners={r['parcel_id']:r for r in read(d/'home_owner_snapshots.csv') if r['tax_year']=='2025'}
raw={str(r['pID']):r for r in json.load((b/'park_source_records.json').open())}
# Footprint schema shared with the unit-review validator.
x=json.load((b/'park_parent_polygons.geojson').open())
print('footprint props',x['features'][0]['properties'])
for f in x['features']:f['properties']['polygon_parcel_id']=str(f['properties'].get('polygon_parcel_id',f['properties'].get('parcel_id',f['properties'].get('PROP_ID'))))
write(d/'home_parent_footprints.geojson',x)
paths=[b/'exception_review_01/ready_home_locations.csv',b/'exception_review_01/home_inventory_review.csv',b/'exception_review_01/case_home_crosswalk.csv',b/'park_source_records.json',b/'city_area_addresses.json',d/'home_parent_footprints.geojson',d/'home_owner_snapshots.csv',d/'ownership_manifest.json',d/'owner_rows_2024.json',d/'owner_rows_2025.json']
projects=[]
for h in homes:
 id=h['parcel_id'];o=owners[id];p=raw[id]['propertyProfile'][0]
 corp=o['is_corporate_owned']=='TRUE'; area=p.get('imprvMainArea') or 0
 supplement=None
 if h['already_promoted']=='FALSE':
  supplement=dict(parcel_id=id,source_county='Travis',situs_address=h['account_address']+' DEL VALLE TX 78617',situs_city='DEL VALLE',situs_state='TX',situs_zip='78617',propertyProf_imprvStateCd='M1',propertyProf_landStateCd='',propertyProf_imprvActualYearBuilt=str(p.get('imprvActualYearBuilt') or ''),improvement_sqft=area,land_sqft=0,units_raw=0,is_residential=True,is_owner_occupied=o['is_owner_occupied']=='TRUE',is_corporate_owned=corp,has_financialized_owner=o['has_financialized_owner']=='TRUE',n_owner_rows=o['n_owner_rows'],parcel_count=1,corporate_parcel_count=int(corp),corporate_improvement_sqft=area if corp else 0,county_unit_exclude_from_unit_universe=False,coord_source='reviewed_city_address_point')
 projects.append(dict(review_id='oak_ranch_home_20261007_'+id,project_id='project:'+id,name='Oak Ranch home '+id,source_county='Travis',parcel_ids=[id],units=1,inventory_group='oak_ranch_20261007',omission_reason='omitted_manufactured_home_account',basis='One active 2025 TCAD manufactured-home account; distinct City address point within a reviewed Oak Ranch parent parcel and the fixed grid. Home ownership classified independently from source-year records; no park-owner inheritance.',evidence_paths=[str(paths[0]),str(paths[3]),str(paths[4]),str(paths[5]),str(paths[6])],geometry=dict(lon=float(h['longitude']),lat=float(h['latitude']),polygon_parcel_id=h['spatial_parent_id'],expected_hex_id=int(h['point_hex_id']),evidence_path=str(paths[5]),method='unique_city_address_with_reviewed_parent_containment',coord_source='reviewed_city_address_point'),supplement=supplement))
write(Path('config/manufactured_home_property_reviews.json'),dict(schema_version=1,projects=projects,evidence=[dict(path=str(p),sha256=sha(p)) for p in paths],ownership_snapshot_path=str(d/'home_owner_snapshots.csv')))
# Case-specific evidence, preserving the exact source addresses/geocodes.
cs=read(b/'exception_review_01/case_home_crosswalk.csv'); ev=read(Path('output/residential_property_batch2/case_address_evidence.csv')); cases=[]
for c in cs:
 rows=[r for r in ev if r['case_number']==c['case_number']];assert rows
 addrs={r['address_for_geocoding']:dict(address_for_geocoding=r['address_for_geocoding'],longitude=float(r['longitude']),latitude=float(r['latitude'])) for r in rows}
 cases.append(dict(case_number=c['case_number'],source_county='Travis',source_jp_district='JP4',file_date=c['file_date'],parcel_id=c['home_parcel_id'],project_id='project:'+c['home_parcel_id'],review_basis='Exact filing street address matched to the reviewed individual TCAD home account and City address point; no park-wide address alias.',review_id='oak_ranch_case_20261007_'+c['case_number'].split(':')[-1],addresses=list(addrs.values()),apartment_conflict=False,expected_unit_hex_id=int(c['reviewed_hex'])))
casepath=Path('data/reviewed_eviction_properties/oak_ranch_20261007/cases.json');casepath.parent.mkdir(parents=True,exist_ok=True)
write(casepath,dict(schema_version=1,batch_id='oak_ranch_20261007',cases=cases,evidence=[dict(path=str(p),sha256=sha(p)) for p in [paths[0],paths[2]]]))
cfg=Path('config/eviction_property_reviews.json');x=json.load(cfg.open());x['batches']=[v for v in x['batches'] if v['batch_id']!='oak_ranch_20261007'];x['batches'].append(dict(batch_id='oak_ranch_20261007',path=str(casepath),sha256=sha(casepath),case_count=24,scope='24 exact-address filings linked to 17 individually reviewed Oak Ranch homes'));write(cfg,x)
