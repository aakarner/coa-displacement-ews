"""Review five park inventories from retained county and City evidence."""
import csv,json,re,sys
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path('scripts/audits').resolve()))
from oak_ranch_inventory import serial,clean,suffix,key
ROOT=Path('data/reviewed_unit_properties/residual_20261007')
PARKS={'M00019':('292158','Capitol View',1308),'M137':('545554','Pecan Park phase 1',5701),'M00107':('545754','Pecan Park phase 2',5701),'M00099':('291921','Village Park / Hoeke',2705),'M00009':('191248','Bel Aire',841),'M00094':('304763','Trails of Oak Hill',None)}
rows=json.load((ROOT/'park_home_source_records.json').open());allrows=[json.loads(s) for s in open('tmp/oak_ranch_20261007/county_m1_source_records.jsonl')]
snids=defaultdict(set)
for r in allrows:
 for d in r['propertyLegalDescription']:
  sn=serial(d.get('legalDescription') or '')
  if sn:snids[sn].add(r['pID'])
refs={r['polygon_parcel_id']:r for r in csv.DictReader((ROOT/'reference_candidates.csv').open())}
spaces=Counter((d['asCode'],clean(d.get('mhSpaceNum'))) for r in rows for d in r['propertyLegalDescription'])
city=defaultdict(list)
for f in json.load((ROOT/'city_points_304763.json').open())['features']:
 a=f['attributes']
 if a['ADDRESS_TYPE']==1:city[key(a['ADDRESS'],a['STREET_NAME'])].append(f)
output=[]
for r in rows:
 d=r['propertyLegalDescription'][0];s=r['situses'][0];p=r['propertyProfile'][0];code=d['asCode'];parent,name,num=PARKS[code]
 space=clean(d.get('mhSpaceNum'));sn=serial(d.get('legalDescription') or '');issues=[]
 if r['inactive']!=0 or r['propType']!='MH' or p['stateCd'] not in ['M1','A2']:issues.append('inactive_or_not_home')
 if sn and len(snids[sn])>1:issues.append('serial_on_multiple_accounts')
 if space and spaces[code,space]>1:issues.append('multiple_active_accounts_same_space')
 if not sn and not (space and (p.get('imprvMainArea') or 0)>0):issues.append('insufficient_home_identity')
 if code=='M00094':
  matches=city[key(s['streetNum'],s['streetName'])]
  if len(matches)!=1:issues.append('individual_city_point_not_unique_or_missing')
  point=matches[0] if len(matches)==1 else None
  lon=point['geometry']['x'] if point else '';lat=point['geometry']['y'] if point else '';method='individual_city_address_point'
  place=str(point['attributes']['PLACE_ID']) if point else ''
 else:
  lon=float(refs[parent]['lon']);lat=float(refs[parent]['lat']);method='reviewed_park_phase_reference';place=''
 secondary=clean(s.get('streetSecondary'));textspace=re.search(r'\bSPACE\s*#?\s*([A-Z0-9-]+)',d.get('legalDescription') or '')
 values={x for x in [space,secondary,textspace[1] if textspace else ''] if x}
 flags=[]
 if len(values)>1:flags.append('source_space_fields_disagree')
 if not sn:flags.append('serial_missing')
 output.append(dict(parcel_id=str(r['pID']),park=name,park_code=code,parent_id=parent,space=space,
 account_address=' '.join(x for x in [clean(s['streetNum']),clean(s['streetName']),suffix(s.get('streetSuffix'))] if x),
 serial=sn,flags=';'.join(flags),issues=';'.join(issues),ready=not issues,longitude=lon,latitude=lat,city_place_id=place,geometry_method=method))
with (ROOT/'home_inventory_review.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
print('Inventory',len(output),'accepted candidates',sum(r['ready'] for r in output),'issues',Counter(r['issues'] for r in output if r['issues']))
for code in PARKS:print(PARKS[code][1],sum(r['ready'] and r['park_code']==code for r in output))
