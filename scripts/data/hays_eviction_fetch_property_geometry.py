"""Cache a focused public Hays CAD geometry query for the reviewed property IDs.

Does not refresh or overwrite the project's existing countywide parcel cache.
Only parcel IDs are transmitted. This is a separate, dated triage source.
"""
import json
import hashlib
import sys
import zipfile
from pathlib import Path
import urllib.parse
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/EXTERNAL_hcad_parcels/FeatureServer/0"
registry = json.loads((ROOT / "config/hays_eviction_property_groups.json").read_text())
ids = sorted({p for g in registry["groups"] for p in g["parcel_ids"]})
params = dict(f="geojson", where="REFNAME IN (" + ",".join("'" + p + "'" for p in ids) + ")",
              outFields="OBJECTID,REFNAME,TEXT", returnGeometry="true", outSR="4326")
url = ENDPOINT + "/query?" + urllib.parse.urlencode(params)
out = ROOT / "output/hays_eviction_property_geometry_20260910.geojson"
if not out.exists():
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    assert data.get("type") == "FeatureCollection", data
    assert not data.get("exceededTransferLimit"), "Incomplete response"
    out.write_text(json.dumps(data))
    out.with_suffix(".metadata.json").write_text(json.dumps(dict(
        requested_ids=ids, endpoint=ENDPOINT, parameters=params,
        retrieved_at=datetime.now(timezone.utc).isoformat()), indent=2))
data = json.loads(out.read_text())
found = {f["properties"]["REFNAME"].strip() for f in data["features"]}
print(json.dumps(dict(requested=len(ids), returned=len(data["features"]),
                     missing=sorted(set(ids) - found)), indent=2))

if "--state-archive" in sys.argv:
    # Public TxGIO origin; the resource listing identifies the Hays archive.
    collection = "0fa04328-872e-481c-b453-126a74777593"
    metadata_url = f"https://api.tnris.org/api/v1/resources?collection_id={collection}"
    folder = ROOT / "data/raw_parcels/hays/txgio_2025"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "resources.json"
    if not metadata.exists():
        with urllib.request.urlopen(metadata_url, timeout=60) as response:
            metadata.write_bytes(response.read())
    resources = json.loads(metadata.read_text())["results"]
    hays = [r for r in resources if r["area_type_name"] == "Hays"]
    assert len(hays) == 1
    filename = hays[0]["resource"].rsplit("/", 1)[1]
    assert filename == "stratmap25-landparcels_48209_lp.zip"
    archive_url = f"https://s3.amazonaws.com/data.tnris.org/{collection}/resources/{filename}"
    archive = folder / filename
    if not archive.exists():
        with urllib.request.urlopen(archive_url, timeout=60) as response:
            archive.write_bytes(response.read())
    assert archive.stat().st_size == hays[0]["filesize"]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for member in z.infolist():
            if member.filename.startswith("fgdb/"):
                target = (folder / member.filename).resolve()
                assert target.is_relative_to(folder.resolve())
                z.extract(member, folder)
    verification = folder / "verification.json"
    if not verification.exists():
        verification.write_text(json.dumps(dict(
            resource=hays[0], metadata_url=metadata_url, downloaded_from=archive_url,
            verified_at=datetime.now(timezone.utc).isoformat(),
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest()), indent=2))
    print(f"Verified state parcel archive: {archive}")
