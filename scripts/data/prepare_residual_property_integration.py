"""Prepare the reviewed apartment geography and five-park supplement (no rebuild)."""
import csv,json,hashlib,re
from pathlib import Path
B=Path('data/reviewed_unit_properties/residual_20261007')
read=lambda p:list(csv.DictReader(p.open()))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
public=[dict(url='https://www.thebennettaustin.com/',retrieved_on='2026-10-07',fact='Operator identifies 7301 S IH-35 SVRD NB, Austin TX 78744.'),dict(url='https://www.theolivineaustin.com/',retrieved_on='2026-10-07',fact='Operator identifies The Olivine at 3201 Century Park Blvd, Austin TX 78727.'),dict(url='https://services.austintexas.gov/edims/document.cfm?id=184161',retrieved_on='2026-10-07',fact='Austin Energy identifies Hudson Miramont at 8818 Travis Hills Dr and 276 apartments.'),dict(url='https://www.weidner.com/blog/2025/12/30/weidner-apartment-homes-acquires-panorama-villas-apartments-in-southwest-austin-tx/',retrieved_on='2026-10-07',fact='December 30, 2025 acquisition announcement corroborates 276 apartments, formerly Hudson Miramont. Not used to classify 2025 ownership.')]
write(B/'public_sources.json',public)
homes=read(B/'ready_home_locations.csv');owners={r['parcel_id']:r for r in read(B/'ownership/home_owner_snapshots.csv') if r['tax_year']=='2025'}
raw={str(r['pID']):r for r in json.load((B/'park_home_source_records.json').open())}
refs={r['polygon_parcel_id']:r for r in read(B/'reference_candidates.csv')}
paths=[B/'park_home_source_records.json',B/'county_full_records.json',B/'county_links.json',B/'ready_home_locations.csv',B/'home_inventory_review.csv',B/'home_spatial_review.csv',B/'footprints.geojson',B/'reference_candidates.csv',B/'city_scope.csv',B/'public_sources.json',B/'hudson_energy.pdf',*sorted(B.glob('city_*.json')),*sorted((B/'ownership').glob('*')),Path('data/BOUNDARIES_jurisdictions_20260429.geojson')]
projects=[]
for h in homes:
 id=h['parcel_id'];o=owners[id];p=raw[id]['propertyProfile'][0];corp=o['is_corporate_owned']=='TRUE';area=p.get('imprvMainArea') or 0
 assert h['already_promoted']=='FALSE' and o['classification_status']=='matched_classified'
 shared=h['geometry_method']=='reviewed_park_phase_reference'
 supplement=dict(parcel_id=id,source_county='Travis',situs_address=h['account_address']+' AUSTIN TX '+str(raw[id]['situses'][0]['zip']),situs_city='AUSTIN',situs_state='TX',situs_zip=str(raw[id]['situses'][0]['zip']),propertyProf_imprvStateCd='M1',propertyProf_landStateCd='',propertyProf_imprvActualYearBuilt=str(p.get('imprvActualYearBuilt') or ''),improvement_sqft=area,land_sqft=0,units_raw=0,is_residential=True,is_owner_occupied=o['is_owner_occupied']=='TRUE',is_corporate_owned=corp,has_financialized_owner=o['has_financialized_owner']=='TRUE',n_owner_rows=o['n_owner_rows'],parcel_count=1,corporate_parcel_count=int(corp),corporate_improvement_sqft=area if corp else 0,county_unit_exclude_from_unit_universe=False,coord_source=h['geometry_method'])
 projects.append(dict(review_id='residual_home_20261007_'+id,project_id='project:'+id,name=h['park']+' home '+id,source_county='Travis',parcel_ids=[id],units=1,inventory_group='residual_parks_20261007',property_group_id='park_phase:'+h['parent_id'] if shared else None,omission_reason='omitted_manufactured_home_account',basis='One active 2025 manufactured-home account; duplicate serial/space accounts withheld. '+('Analytical reference shared by the verified park phase; not an individual home coordinate.' if shared else 'Unique individual City house address within the reviewed park footprint.')+' Home-level ownership retained independently; source flags: '+h['flags'],evidence_paths=[str(B/'park_home_source_records.json'),str(B/'ready_home_locations.csv'),str(B/'footprints.geojson')],geometry=dict(lon=float(h['longitude']),lat=float(h['latitude']),polygon_parcel_id=h['parent_id'],expected_hex_id=int(h['point_hex_id']),evidence_path=str(B/'footprints.geojson'),method=h['geometry_method'],coord_source=h['geometry_method'],location_precision='park_phase' if shared else 'individual_home'),supplement=supplement))
for id,parent,name,units,oldlon,oldlat in [('549351','549351','The Olivine',317.49477806788514,-97.6950379588,30.4332392733),('942518','942517','The Bennett',271,-97.702931869018,30.333786104977)]:
 ref=refs[parent];projects.append(dict(review_id='residual_geometry_20261007_'+id,project_id='project:'+id,name=name,source_county='Travis',parcel_ids=[id],units=units,count_status='reviewed_geometry_only',basis='Move the analytical unit reference inside the verified property footprint; retain the existing unit total and estimation method. '+('County source point lies outside its own parcel.' if id=='549351' else 'County improvement account links to parent 942517; source situs ZIP 78752 produced a geocoder point 18 km north. Operator confirms southern 78744 property. Preserve raw situs/ZIP for audit.'),evidence_paths=[str(B/'county_full_records.json'),str(B/'county_links.json'),str(B/'footprints.geojson'),str(B/'public_sources.json')],geometry=dict(lon=float(ref['lon']),lat=float(ref['lat']),polygon_parcel_id=parent,expected_hex_id=int(ref['hex_id']),expected_original_lon=oldlon,expected_original_lat=oldlat,evidence_path=str(B/'footprints.geojson'),method='point_on_surface_of_verified_property_footprint',coord_source='reviewed_county_footprint_reference',location_precision='property',city_boundary_evidence_path='data/BOUNDARIES_jurisdictions_20260429.geojson',minimum_city_overlap_fraction=.999),supplement=None))
write(Path('config/residual_property_reviews.json'),dict(schema_version=1,projects=projects,evidence=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(paths)],ownership_snapshot_path=str(B/'ownership/home_owner_snapshots.csv')))
# Narrow address aliases only for single-property base addresses. Pecan phases
# remain case-specific because the street address is shared between phases.
cfg=Path('config/eviction_property_address_reviews.json');x=json.load(cfg.open());x['reviews']=[r for r in x['reviews'] if not r['review_id'].startswith('residual_')]
for parent,project,parcel,pattern in [
 ('549351','project:549351','549351',r'^3201 CENTURY PARK (BLVD|BOULEVARD)( (APT |UNIT |#)[A-Z0-9-]+)? AUSTIN TX 78727$'),
 ('942517','project:942518','942518',r'^7301 S (IH-?35|IH 35|INTERSTATE 35)( SVRD NB)?( (APT |UNIT |#)[A-Z0-9-]+)? AUSTIN TX 78744$'),
 ('292158','park_phase:292158',None,r'^1308 THORNBERRY (RD|ROAD)( (APT |UNIT |LOT |#)[A-Z0-9-]+)? AUSTIN TX 78721$'),
 ('291921','park_phase:291921',None,r'^2705 HOEKE (LN|LANE)( (APT |UNIT |LOT |#)[A-Z0-9-]+)? AUSTIN TX 78744$'),
 ('191248','park_phase:191248',None,r'^841 AIRPORT (BLVD|BOULEVARD)( (APT |UNIT |LOT |#)[A-Z0-9-]+)? AUSTIN TX 78702$')]:
 ref=refs[parent];member=parcel or next(h['parcel_id'] for h in homes if h['parent_id']==parent)
 x['reviews'].append(dict(review_id='residual_address_20261007_'+parent,property_name=next(p['name'] for p in projects if p['parcel_ids']==[member]),source_county='Travis',parcel_id=member,project_id=project,expected_unit_hex_id=int(ref['hex_id']),reference_longitude=float(ref['lon']),reference_latitude=float(ref['lat']),normalized_address_pattern=pattern,review_basis='Verified single property/park-phase base address and reviewed denominator geography. Does not independently verify an apartment/space. County and City evidence retained.',evidence=[dict(path=str(p),sha256=sha(p)) for p in [B/'public_sources.json',B/'county_full_records.json',B/'ready_home_locations.csv',B/'footprints.geojson']]))
write(cfg,x)
# 12 Pecan Park cases: county phase/space identifies the home; City confirms unit identity.
# City registers some phase-2 units under the common phase-1 address point.
e=read(Path('tmp/residential_residual_triage_20261007/case_address_evidence.csv'));groups={}
for r in e:
 if r['address_key'] in ['5701 JOHNNY MORRIS RD','6008 OLEANDER TRL']:groups.setdefault(r['case_number'],[]).append(r)
city=[f['attributes'] for f in json.load((B/'city_subaddresses_5701.json').open())['features']]
cases=[]
for cid,rows in groups.items():
 matches=[];city_conflict=False
 for row in rows:
  if row['address_key']=='6008 OLEANDER TRL':hs=[h for h in homes if h['account_address']=='6008 OLEANDER TRL']
  else:
   m=re.search(r'(?:#|APT\s*|UNIT\s*)0*(\d+)\b',row['address_for_geocoding']);assert m,row
   space=str(int(m[1]));hs=[h for h in homes if h['park_code'] in ['M137','M00107'] and h['space'].lstrip('0')==space]
   cr=[r for r in city if r['PLACE_TYPE']=='UNIT' and str(r['UNIT_NAME']).lstrip('0')==space and r['DISCONTINUE_DATE'] is None]
   assert len(hs)==1 and cr,(space,hs,cr)
   city_conflict = city_conflict or any(r['PARCEL_ID']!=('0217300201' if hs[0]['park_code']=='M137' else '0217300202') for r in cr)
  assert len(hs)==1;matches.append(hs[0])
 assert len({h['parcel_id'] for h in matches})==1
 h=matches[0];shared=h['geometry_method']=='reviewed_park_phase_reference'
 addresses={r['address_for_geocoding']:dict(address_for_geocoding=r['address_for_geocoding'],longitude=float(r['longitude']),latitude=float(r['latitude'])) for r in rows}
 cases.append(dict(case_number=cid,source_county='Travis',source_jp_district=rows[0]['jp_district'],file_date=rows[0]['file_date'],parcel_id=h['parcel_id'],project_id='park_phase:'+h['parent_id'] if shared else 'project:'+h['parcel_id'],city_phase_reference_conflict=city_conflict,review_basis='Unique county home address/space and county park-phase code identify the reviewed home or phase. City registered unit confirms unit identity only: its shared address-point parcel sometimes disagrees with the county phase. No dwelling coordinate inferred from a park centroid.',review_id='residual_case_20261007_'+cid,addresses=list(addresses.values()),apartment_conflict=False,expected_unit_hex_id=int(h['point_hex_id'])))
casepath=Path('data/reviewed_eviction_properties/residual_20261007/cases.json');write(casepath,dict(schema_version=1,batch_id='residual_20261007',cases=cases,evidence=[dict(path=str(p),sha256=sha(p)) for p in [B/'ready_home_locations.csv',B/'park_home_source_records.json',B/'city_subaddresses_5701.json']]))
p=Path('config/eviction_property_reviews.json');cfg=json.load(p.open());cfg['batches']=[b for b in cfg['batches'] if b['batch_id']!='residual_20261007'];cfg['batches'].append(dict(batch_id='residual_20261007',path=str(casepath),sha256=sha(casepath),case_count=len(cases),scope='12 Pecan Park phase/space reviews and six exact-home Trails of Oak Hill reviews'));write(p,cfg)
print('Prepared',len(projects),'projects;',len(cases),'case reviews; geometry-only apartment corrections preserve units.')
