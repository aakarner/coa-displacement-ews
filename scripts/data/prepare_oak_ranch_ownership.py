"""Classify reviewed homes from pinned source-year owner rows, never park owners."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict,Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_other_county_ownership as adapter
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/reviewed_manufactured_housing/oak_ranch_20261007'
OUT=BASE/'integration'

def sha(p):return adapter.digest(p)

def main():
 OUT.mkdir(exist_ok=True)
 lm=adapter.load_classifier(ROOT.parent/'landlord-mapper/historical_ownership.py')
 manifest_path=ROOT.parent/'landlord-mapper/output/historical_ownership/travis_owner_snapshot_manifest.json'
 manifest=json.loads(manifest_path.read_text())
 assert sha(ROOT.parent/'landlord-mapper/historical_ownership.py')==manifest['pipeline']['script']['sha256']
 homes=list(csv.DictReader((BASE/'exception_review_01/ready_home_locations.csv').open()))
 ids={r['parcel_id'] for r in homes};assert len(ids)==len(homes)==793
 targets=[dict(parcel_id=r['parcel_id'],property_units_numeric=1,residential_use_category='manufactured_home') for r in homes]
 snapshots=[];sources=[]
 for year in [2024,2025]:
  info=manifest['sources'][str(year)]['intermediate' if year==2024 else 'standardized_intermediate']
  path=Path(info['path']);assert sha(path)==info['sha256']
  selected=defaultdict(list)
  with path.open() as f:
   for row in csv.DictReader(f):
    if row['parcel_id'] in ids:
     assert int(row['tax_year'])==year
     selected[row['parcel_id']].append(row)
  # Missing owner rows remain unknown; the retained rows never use park ownership.
  rows=lm.build_snapshot_rows(tax_year=year,ews_rows=targets,rows_by_target=selected,source_property_ids=set(selected),
   source_snapshot_id=lm.SOURCE_2024_SNAPSHOT_ID if year==2024 else lm.SOURCE_2025_SNAPSHOT_ID,
   source_owner_field='property_year_owner' if year==2024 else 'current_special_export_owner',source_supplement_number='0' if year==2024 else '1')
  snapshots.extend(rows)
  retained=OUT/f'owner_rows_{year}.json';retained.write_text(json.dumps(selected,indent=2)+'\n')
  sources.append(dict(tax_year=year,source=info,retained_path=str(retained.relative_to(ROOT)),retained_sha256=sha(retained)))
  print(year,Counter(r['classification_status'] for r in rows))
 path=OUT/'home_owner_snapshots.csv'
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(snapshots[0]));w.writeheader();w.writerows(snapshots)
 out=dict(production_applied=False,classifier_sha256=sha(ROOT.parent/'landlord-mapper/historical_ownership.py'),
  classifier_rule_version=lm.CLASSIFICATION_RULE_VERSION,upstream_manifest_sha256=sha(manifest_path),sources=sources,
  snapshots_path=str(path.relative_to(ROOT)),snapshots_sha256=sha(path),rows=len(snapshots))
 (OUT/'ownership_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
