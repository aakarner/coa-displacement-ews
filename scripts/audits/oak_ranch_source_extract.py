import csv, json, re, zipfile, time
from pathlib import Path

root = Path('tmp/oak_ranch_20261007')
profiles = {int(r['propertyProf_pID']) for r in csv.DictReader(open('../landlord-mapper/output/property_profile.csv'))
            if r['propertyProf_stateCd'] == 'M1'}
targets = profiles | {464309, 909849}
seed = {int(r['situs_pID']) for r in csv.DictReader(open('output/residential_property_batch2/remaining_address_accounts.csv'))}
targets |= seed
archive = Path('../landlord-mapper/tcad_special_export.zip')
count = selected = 0
start = time.time()
fields = ['pID', 'pYear', 'propType', 'inactive', 'inactiveDt', 'inactiveReason', 'inactiveNotes',
          'reactivateDt', 'propCreateDt', 'geometry', 'propertyLegalDescription', 'propertyIdentification',
          'propertyCharacteristics', 'propertyProfile', 'situses']
with zipfile.ZipFile(archive) as z, z.open(z.namelist()[0]) as f, (root/'county_m1_source_records.jsonl').open('w') as out:
    buffer = b''
    def consume(piece):
        global count, selected
        count += 1
        match = re.search(rb'"pID": (\d+)', piece[:100])
        if not match or (int(match[1]) not in targets and b'"asCode": "M00109"' not in piece):
            return
        obj = json.loads(b'{' + piece + b'}')
        if selected == 0:
            print('Source top-level keys:', list(obj), flush=True)
        keep = {k:v for k,v in obj.items() if k in fields or 'mobile' in k.lower() or 'improv' in k.lower()}
        out.write(json.dumps(keep)+'\n')
        selected += 1
    first = True
    while chunk := f.read(16*1024*1024):
        buffer += chunk
        if first:
            buffer = buffer[buffer.index(b'{')+1:]
            first = False
        parts = buffer.split(b'\n },\n {')
        buffer = parts.pop()
        for piece in parts:
            consume(piece)
    consume(buffer[:buffer.rfind(b'}')])
print('Done', count, 'source records;', selected, 'retained;', round(time.time()-start), 'seconds', flush=True)
