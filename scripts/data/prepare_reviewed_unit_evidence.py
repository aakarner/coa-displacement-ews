"""Save the public documents supporting the October 2026 unit review."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path("data/reviewed_unit_properties/batch1_20261007")
SOURCES = {
    "asher_haca_2019_20.pdf": "https://www.hacanet.org/wp-content/uploads/2020/09/HACA-AnnualReport_2019-20_FINAL_PAGES.pdf",
    "asher_association.html": "https://www.austinaptassoc.com/aisd-property-directory/bridge-at-asher",
    "monarch_austin_energy_2023.pdf": "https://services.austintexas.gov/edims/document.cfm?id=421606",
    "domain_compliance_2009.pdf": "https://www.austintexas.gov/sites/default/files/files/Redevelopment/domain-report2009.pdf",
    "ben_white_reentry_2018.pdf": "https://www.austintexas.gov/sites/default/files/files/HR/TravisCountyReentryGuidebook2018.pdf",
}


def fetch(item):
    name, url = item
    path = ROOT / name
    if not path.exists():
        try:
            content = subprocess.check_output([
                "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
                "--max-time", "45", url,
            ], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            return {"path": str(path), "url": url, "status": "download_failed",
                    "error": error.stderr.decode("utf-8", errors="replace").strip()}
        if name.endswith(".pdf") and not content.startswith(b"%PDF"):
            return {"path": str(path), "url": url, "status": "download_failed",
                    "error": "Response is not a PDF"}
        path.write_bytes(content)
    return {"path": str(path), "url": url, "status": "saved",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "download_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved and hashed {sum(r['status'] == 'saved' for r in records)} of {len(records)} source documents; see manifest for failures.")
