#!/usr/bin/env python3
"""Acquire immutable, checksummed historical and current amenity source extracts.

No eligibility, event reconciliation, or geocoding is performed here. The full
February archive is checksum-pinned before normalization. The live extract has
no event-date filter, so reconciliation can inspect dates outside either window.
Its county/NAICS scope remains explicit: a missing ID does not prove closure.

Examples:
  python3 scripts/data/prepare_historical_amenity_sources.py
  python3 scripts/data/prepare_historical_amenity_sources.py --live-id texas_sales_tax_live_20260908T103000Z
  python3 scripts/data/prepare_historical_amenity_sources.py --archive-only

Existing completed snapshots are verified and reused, never refreshed in place.
Run without --live-id to acquire a new timestamped live snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/amenity_historical_sources.json"
VERSION = "historical-amenity-sources-v1"
ARCHIVE_COLUMNS = {
    "tp_number": "Taxpayer Number", "loc_number": "Location Number",
    "loc_name": "Location Name", "address_number": "Location Address Number",
    "address_text": "Location Address Text", "permit_date": "Permit Date",
    "juris_city": "Jurisdiction City", "loc_city": "Postal City",
    "loc_state": "Location State", "loc_zip": "Location Zip Code",
    "loc_county": "Location County", "naics": "NAICS Code",
    "first_sale_date": "First Sales Date", "out_of_business_date": "Out of Business Date",
}
DATE_FIELDS = {"permit_date", "first_sale_date", "out_of_business_date"}


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def read_headers(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith("HTTP/"):
            result = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            result[key.lower().strip()] = value.strip()
    return result


def http_date(value):
    return (parsedate_to_datetime(value).astimezone(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ")) if value else None


def download(url, destination, headers_path):
    """Publish only successful downloads; preserve partial files on failure."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    partial = destination.with_suffix(destination.suffix + ".download")
    if partial.exists():
        raise FileExistsError(f"Inspect or explicitly adopt existing partial: {partial}")
    started = utc_now()
    command = ["curl", "--location", "--fail", "--silent", "--show-error",
               "--max-time", "1200", "--retry", "3", "--retry-delay", "3",
               "--dump-header", str(headers_path), "--output", str(partial), url]
    subprocess.run(command, check=True)
    partial.rename(destination)
    return started, utc_now()


def iso_date(value):
    value = value.strip()
    if not value or value.upper() in {"NA", "NULL"}:
        return ""
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
        return datetime.strptime(value, "%m/%d/%Y").date().isoformat()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:T.*)?", value):
        return datetime.strptime(value[:10], "%Y-%m-%d").date().isoformat()
    raise ValueError(f"Unexpected date representation: {value!r}")


def normalize(raw_path, output, config, archive=False):
    """Stream the statewide archive; preserve identifiers as text."""
    counts = Counter()
    fields = config["columns"]
    source_fields = [ARCHIVE_COLUMNS[field] if archive else field for field in fields]
    keys = set()
    with raw_path.open(newline="", encoding="utf-8-sig") as src, \
            output.open("x", newline="", encoding="utf-8") as dest:
        reader = csv.DictReader(src)
        missing = set(source_fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing source fields: {sorted(missing)}")
        source_column_count = len(reader.fieldnames)
        writer = csv.DictWriter(dest, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for raw in reader:
            counts["raw_rows"] += 1
            row = dict(zip(fields, (raw[field].strip() for field in source_fields)))
            if row["loc_county"] not in config["counties"] or row["naics"] not in config["naics"]:
                continue
            for field in DATE_FIELDS:
                row[field] = iso_date(row[field])
            key = (row["tp_number"], row["loc_number"])
            if not all(key):
                raise ValueError("Empty taxpayer/location identifier")
            if key in keys:
                counts["duplicate_stable_id_rows"] += 1
            keys.add(key)
            writer.writerow(row)
            counts["normalized_rows"] += 1
            counts[f"county_{row['loc_county']}"] += 1
            counts[f"naics_{row['naics']}"] += 1
    return {**counts, "raw_columns": source_column_count,
            "normalized_columns": len(fields), "distinct_stable_ids": len(keys)}


def check_completed(directory):
    path = directory / "manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for label in ("raw", "normalized"):
        details = manifest[label]
        file = ROOT / details["path"]
        if sha256(file) != details["sha256"] or file.stat().st_size != details["bytes"]:
            raise ValueError(f"Pinned {label} content changed: {file}")
    return manifest


def base_manifest(source_id, config, directory, timestamps, headers, stats):
    raw, normalized = directory / "raw.csv", directory / "normalized.csv"
    return {
        "schema_version": 1, "preparation_version": VERSION,
        "source_id": source_id, "dataset_id": config["dataset_id"],
        "retrieval_started_utc": timestamps[0], "retrieval_completed_utc": timestamps[1],
        "prepared_utc": utc_now(), "config_sha256": sha256(CONFIG),
        "script_sha256": sha256(Path(__file__)),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "taxonomy_path": "config/amenity_categories.csv",
        "taxonomy_sha256": sha256(ROOT / "config/amenity_categories.csv"),
        "county_filter": config["counties"], "naics_filter": config["naics"],
        "date_filter": None, "eligibility_filter": "Not applied; source normalization only",
        "normalization": "14 API-compatible fields, text identifiers, ISO YYYY-MM-DD dates; empty missing dates",
        "http_headers_path": relative(directory / "retrieval.headers.txt"),
        "http_source_last_modified_utc": http_date(headers.get("x-archive-orig-last-modified") or headers.get("last-modified")),
        "raw": {"path": relative(raw), "sha256": sha256(raw), "bytes": raw.stat().st_size,
                "rows": stats["raw_rows"], "columns": stats["raw_columns"]},
        "normalized": {"path": relative(normalized), "sha256": sha256(normalized),
                       "bytes": normalized.stat().st_size, "rows": stats["normalized_rows"],
                       "columns": config["columns"]},
        "counts": stats,
    }


def prepare_archive(config, adopt_download=False):
    source = config["archive"]
    directory = ROOT / config["raw_directory"] / source["source_id"]
    directory.mkdir(parents=True, exist_ok=True)
    existing = check_completed(directory)
    if existing:
        return existing
    raw = directory / "raw.csv"
    partial = directory / "raw.csv.download"
    headers_path = directory / "retrieval.headers.txt"
    url = source["replay_urls"][0]
    if adopt_download:
        if not partial.exists() or raw.exists():
            raise ValueError("--adopt-archive-download requires only raw.csv.download")
        if sha256(partial) != source["expected_sha256"]:
            raise ValueError("Downloaded archive checksum does not match pinned source")
        partial.rename(raw)
        headers = read_headers(headers_path)
        # An externally started curl has no separately logged start time. Do not
        # invent one from file mtime; preserve response time as its own field.
        times = (None, utc_now())
    else:
        times = download(url, raw, headers_path)
        headers = read_headers(headers_path)
    if sha256(raw) != source["expected_sha256"] or raw.stat().st_size != source["expected_bytes"]:
        raise ValueError("Archive failed pinned checksum/byte validation")
    capture = http_date(headers.get("memento-datetime"))
    if capture != source["archive_capture_utc"]:
        raise ValueError(f"Unexpected archive capture: {capture}")
    stats = normalize(raw, directory / "normalized.csv", config, archive=True)
    for actual, expected in (("raw_rows", "expected_rows"), ("raw_columns", "expected_columns"),
                             ("normalized_rows", "expected_core_rows")):
        if stats[actual] != source[expected]:
            raise ValueError(f"Archive dimension mismatch: {actual}={stats[actual]}")
    result = base_manifest(source["source_id"], config, directory, times, headers, stats)
    result.update({"source_type": "archived_official_export", "source_url": source["source_url"],
                   "retrieval_url": url, "archive_capture_utc": capture,
                   "expected_source_last_modified_utc": source["source_last_modified_utc"],
                   "retrieval_response_date_utc": http_date(headers.get("date")),
                   "archive_content_validated_against_prior_audit": True})
    write_new_json(directory / "manifest.json", result)
    return result


def prepare_live(config, source_id):
    if not re.fullmatch(r"texas_sales_tax_live_[A-Za-z0-9_-]+", source_id):
        raise ValueError("Live source ID must start texas_sales_tax_live_ and use safe filename characters")
    directory = ROOT / config["raw_directory"] / source_id
    directory.mkdir(parents=True, exist_ok=True)
    existing = check_completed(directory)
    if existing:
        return existing
    source = config["live"]
    quoted = lambda values: ",".join(f"'{value}'" for value in values)
    query = {"$select": ",".join(config["columns"]),
             "$where": f"loc_county in ({quoted(config['counties'])}) and naics in ({quoted(config['naics'])})",
             "$order": source["order"], "$limit": str(source["limit"])}
    url = source["endpoint"] + "?" + urlencode(query)
    times = download(url, directory / "raw.csv", directory / "retrieval.headers.txt")
    stats = normalize(directory / "raw.csv", directory / "normalized.csv", config)
    if stats["raw_rows"] >= source["limit"]:
        raise ValueError("Query reached row limit; increase limit or implement pinned pagination")
    if stats["raw_rows"] != stats["normalized_rows"]:
        raise ValueError("API query returned rows outside requested scope")
    headers = read_headers(directory / "retrieval.headers.txt")
    result = base_manifest(source_id, config, directory, times, headers, stats)
    result.update({"source_type": "live_official_api_extract", "source_url": source["endpoint"],
                   "retrieval_url": url, "query": query, "scope_note": source["scope_note"],
                   "archive_capture_utc": None,
                   "retention_note": "Live rolling file is not a complete historical archive."})
    write_new_json(directory / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-id", default="texas_sales_tax_live_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--adopt-archive-download", action="store_true")
    args = parser.parse_args()
    if args.archive_only and args.live_only:
        parser.error("--archive-only and --live-only are mutually exclusive")
    config = json.loads(CONFIG.read_text())
    results = []
    if not args.live_only:
        results.append(prepare_archive(config, args.adopt_archive_download))
    if not args.archive_only:
        results.append(prepare_live(config, args.live_id))
    for result in results:
        print(json.dumps({"source_id": result["source_id"], "raw": result["raw"],
                          "normalized": result["normalized"], "manifest_path":
                          f"{config['raw_directory']}/{result['source_id']}/manifest.json"}, indent=2))


if __name__ == "__main__":
    main()
