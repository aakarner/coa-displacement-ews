#!/usr/bin/env python3
"""Adapt source-year Hays/WCAD ownership to the pinned landlord-mapper rules.

Only annual source evidence is used for classification. The supplied EWS target
defines fixed parcel support and residential eligibility; its current owner,
situs, and exemption fields are never substituted for missing historical data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import williamson_ownership_reconciliation as reconciliation

ROOT = Path(__file__).resolve().parents[2]
PINNED_CLASSIFIER_COMMIT = "cd68639"
HAYS_URL = "https://hayscad.com/wp-content/uploads/{year}/07/{year}-Certified-Data-Export.zip"
WCAD_REPORT_URL = "https://www.wcad.org/wp-content/uploads/2025/08/Certification-Report-{year}.zip"
ADAPTER_VERSION = "ews-other-county-owner-v3"
YEARS = (2024, 2025)
COUNTIES = ("Hays", "Williamson")
EXTRA_FIELDS = [
    "source_name_may_be_truncated", "source_address_may_be_truncated",
    "source_situs_may_be_truncated", "source_reported_owner_names",
    "source_report_exemption_codes", "source_year_verified",
    "source_reconciliation_status", "source_reconciliation_conflict",
    "source_reconciliation_version", "source_name_extended",
    "source_name_completeness_confirmed", "source_mailing_extended", "source_situs_extended",
]
BOOL_FIELDS = {
    "owner_name_available", "owner_address_available", "is_owner_occupied",
    "has_financialized_owner", "is_corporate_owned", "name_evidence_complete",
    "address_evidence_complete", "homestead_evidence_available",
    "homestead_positive", "address_match_positive",
    "situs_state_imputed_address_match", "source_name_may_be_truncated",
    "source_address_may_be_truncated", "source_situs_may_be_truncated",
    "source_year_verified",
    "source_reconciliation_conflict", "source_name_extended",
    "source_name_completeness_confirmed", "source_mailing_extended", "source_situs_extended",
}


def input_paths(root=ROOT):
    config_path = root / "config/williamson_ownership_sources.json"
    config = json.loads(config_path.read_text())
    return [config_path, *(root / item["path"] for item in config.values() if isinstance(item, dict)),
            *[root / "data/raw_parcels/appraisal_history" / county.lower() /
            str(year) / f"{county.lower()}_{year}.zip"
            for county in COUNTIES for year in YEARS if county != "Williamson"]]


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_classifier(path, verify=True):
    path = Path(path).resolve()
    if verify:
        expected = subprocess.check_output(
            ["git", "-C", str(path.parent), "show",
             f"{PINNED_CLASSIFIER_COMMIT}:{path.name}"], stderr=subprocess.PIPE)
        if hashlib.sha256(expected).hexdigest() != digest(path):
            raise ValueError("Classifier differs from pinned landlord-mapper commit " +
                             PINNED_CLASSIFIER_COMMIT)
    spec = importlib.util.spec_from_file_location("ews_pinned_ownership_classifier", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_target(path):
    targets = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"parcel_id", "source_county", "is_residential", "property_units"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Target file lacks fields: " + str(required - set(reader.fieldnames or [])))
        for row in reader:
            if row["source_county"] not in COUNTIES:
                continue
            key = row["parcel_id"]
            if not key.startswith(row["source_county"].upper() + ":"):
                raise ValueError("Unexpected county parcel identifier: " + key)
            if key in targets:
                raise ValueError("Duplicate target parcel: " + key)
            targets[key] = row
    return targets


def nested_csv(archive, token):
    """Stream the single table from one nested Hays ZIP (small inner ZIP in RAM)."""
    with zipfile.ZipFile(archive) as outer:
        matches = [name for name in outer.namelist()
                   if name.lower().endswith(".zip") and
                   re.search(r"(?:^|[-_])" + token + r"\.ZIP$", Path(name).name.upper())]
        if len(matches) != 1:
            raise ValueError(f"Expected one {token} archive in {archive}, found {matches}")
        with zipfile.ZipFile(io.BytesIO(outer.read(matches[0]))) as inner:
            tables = [name for name in inner.namelist()
                      if Path(name).suffix.lower() in {".txt", ".csv"}]
            if len(tables) != 1:
                raise ValueError(f"Expected one table in {matches[0]}")
            with inner.open(tables[0]) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="cp1252", newline=""))
                for row in reader:
                    if None in row:
                        raise ValueError(f"Malformed {token} row in {archive}")
                    yield row


def hays_standard(owner, prop, year, lm):
    """Map Hays annual owner/property exports into the upstream row contract."""
    return {
        "tax_year": str(year), "parcel_id": "HAYS:" + owner["QuickRefID"].strip(),
        "owner_id": owner.get("OwnerID", ""), "owner_name": owner.get("OwnerName", ""),
        "owner_share": owner.get("OwnershipPercent", ""),
        "owner_addr_line1": owner.get("Address1", ""),
        "owner_addr_line2": owner.get("Address2", ""),
        "owner_addr_line3": owner.get("Address3", ""),
        "owner_addr_city": owner.get("City", ""),
        "owner_addr_state": owner.get("State", ""),
        "owner_addr_zip": owner.get("Zip", ""),
        "owner_confidential_flag": owner.get("ConfidentialOwner", ""),
        "owner_address_suppressed_flag": owner.get("ConfidentialOwner", ""),
        "homestead_flag": lm.bool_csv(lm.exemption_list_has_homestead(owner.get("ExemptionList", ""))),
        "situs_number": prop.get("SitusStreetNumber", ""),
        "situs_prefix": prop.get("SitusPreDirectional", ""),
        "situs_street": prop.get("SitusStreetName", ""),
        "situs_suffix": " ".join(filter(None, [prop.get("SitusStreetSuffix", ""),
                                                prop.get("SitusPostDirectional", "")])),
        "situs_city": prop.get("SitusCity", ""),
        "situs_state": prop.get("SitusState", ""),
        "situs_zip": prop.get("SitusZip", ""),
        "source_situs_state_imputed": "FALSE",
        "property_type_code": "", "improvement_state_code": "", "land_state_code": "",
        "source_snapshot_id": f"hays-{year}-certified-data-export",
        "source_owner_field": "OwnerName", "source_supplement_number": "CERT",
        "source_partial_owner_flag": "FALSE",
        "source_reported_owner_name": owner.get("OwnerName", ""),
        "source_exemption_codes": owner.get("ExemptionList", ""),
        "source_year_verified": True,
    }


def parse_hays(archive, year, target_ids, lm):
    with zipfile.ZipFile(archive) as zipped:
        if any(not re.match(rf"^{year}(?:[-_\s]|$)", Path(name).name) for name in zipped.namelist()
               if name.lower().endswith(".zip")):
            raise ValueError("Unexpected source year in Hays archive member names")
    props = {}
    for row in nested_csv(archive, "PROPERTY"):
        key = "HAYS:" + row["QuickRefID"].strip()
        if key in target_ids:
            if key in props and props[key] != row:
                raise ValueError("Conflicting Hays property rows: " + key)
            props[key] = row
    owners = defaultdict(list)
    for row in nested_csv(archive, "OWNER"):
        key = "HAYS:" + row["QuickRefID"].strip()
        if key in target_ids:
            owners[key].append(hays_standard(row, props.get(key, {}), year, lm))
    return owners, {"property_rows_matched": len(props), "owner_parcels_matched": len(owners)}


PID_RE = re.compile(r"^PID:\s+(\S+)\s+(?:\(([^)]+)\))?")
CITY_RE = re.compile(r"^(.+?),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)$")
SUPPRESSED_RE = re.compile(r"CONFIDENTIAL|SUPPRESSED|NOT AVAILABLE|WITHHELD", re.I)


def report_block_to_standard(lines, year):
    """Decode the observed WCAD printed roll's three fixed columns.

    Owner column occupies positions 1-30, description 32-66, exemptions 67-82.
    Owner names do not wrap; names filling their column are potentially clipped.
    Printed agent information is excluded from owner mailing-address lines.
    """
    if len(lines) < 5 or not PID_RE.match(lines[0]):
        raise ValueError("Incomplete Williamson property block")
    pid = PID_RE.match(lines[0])
    name = lines[3][:30].strip()
    delivery = []
    for line in lines[4:]:
        left = line[:30].strip()
        if left.startswith("AGENT:") or not left:
            break
        delivery.append(left)
    city = state = zipcode = ""
    if delivery:
        city_match = CITY_RE.match(delivery[-1])
        if city_match:
            city, state, zipcode = city_match.groups()
            city = city.rstrip(",")
            delivery = delivery[:-1]
    description = [line[31:66].strip() for line in lines]
    situs_street = situs_city = situs_state = situs_zip = ""
    situs_truncated = False
    for index, segment in enumerate(description):
        if segment.startswith("SITUS:"):
            situs_street = segment.removeprefix("SITUS:").strip()
            situs_truncated = len(segment) >= 35
            if index + 1 < len(description):
                city_match = CITY_RE.match(description[index + 1])
                if city_match:
                    situs_city, situs_state, situs_zip = city_match.groups()
                    situs_city = situs_city.rstrip(",")
            break
    codes = set()
    for line in lines:
        match = re.match(r"\s*([A-Z0-9]+)\s+\$", line[66:82])
        if match and match.group(1) != "TOT":
            codes.add(match.group(1))
    joined = " ".join(description)
    land_match = re.search(r"LAND SPTB:\s*([A-Z0-9]+)", joined)
    imp_match = re.search(r"IMP SPTB:\s*([A-Z0-9]+)", joined)
    suppressed = bool(SUPPRESSED_RE.search(name))
    # All tax-unit exemption sections are present in the full property block.
    # A listed HS is positive evidence; absence is explicit only within a
    # complete report block. Homestead exemption amounts can be zero.
    return {
        "tax_year": str(year), "parcel_id": "WILLIAMSON:" + pid.group(1),
        "owner_id": pid.group(2) or "", "owner_name": name,
        "owner_share": "", "owner_addr_line1": delivery[0] if delivery else "",
        "owner_addr_line2": delivery[1] if len(delivery) > 1 else "",
        "owner_addr_line3": " ".join(delivery[2:]),
        "owner_addr_city": city, "owner_addr_state": state, "owner_addr_zip": zipcode,
        "owner_confidential_flag": "TRUE" if suppressed else "",
        "owner_address_suppressed_flag": "TRUE" if suppressed else "",
        "homestead_flag": "TRUE" if "HS" in codes else "FALSE",
        "situs_street": situs_street, "situs_city": situs_city,
        "situs_state": situs_state, "situs_zip": situs_zip,
        "source_situs_state_imputed": "FALSE", "property_type_code": "",
        "improvement_state_code": imp_match.group(1) if imp_match else "",
        "land_state_code": land_match.group(1) if land_match else "",
        "source_snapshot_id": f"wcad-{year}-certification-printed-report",
        "source_owner_field": "printed_owner_name_column_1_30",
        "source_supplement_number": "CERT", "source_partial_owner_flag": "FALSE",
        "source_name_may_be_truncated": len(name) >= 29,
        "source_address_may_be_truncated": any(len(value) >= 29 for value in delivery),
        "source_situs_may_be_truncated": situs_truncated,
        "source_reported_owner_name": name,
        "source_exemption_codes": ";".join(sorted(codes)), "source_year_verified": True,
    }


def parse_williamson_report(archive, year, target_ids):
    owners = defaultdict(list)
    counters = Counter()
    verified_years = set()
    with zipfile.ZipFile(archive) as zipped:
        members = [name for name in zipped.namelist() if name.lower().endswith(".txt")]
        if len(members) != 1:
            raise ValueError("Expected a single text member in WCAD report")
        with zipped.open(members[0]) as raw:
            block = []
            keep = False
            def flush():
                if block:
                    row = report_block_to_standard(block, year)
                    owners[row["parcel_id"]].append(row)
                    counters["owner_rows_matched"] += 1
            for raw_line in raw:
                line = raw_line.decode("cp1252").rstrip("\r\n")
                year_match = re.match(r"(\d{4}) CERTIFIED ROLL", line)
                if year_match:
                    verified_years.add(int(year_match.group(1)))
                pid = PID_RE.match(line)
                if pid:
                    flush()
                    block = []
                    keep = "WILLIAMSON:" + pid.group(1) in target_ids
                    counters["report_owner_rows"] += 1
                # Page headers may appear between tax-unit sections; they do
                # not enter the owner or exemption fields and are harmless.
                if keep:
                    block.append(line)
            flush()
    if verified_years != {year}:
        raise ValueError(f"WCAD report years {verified_years}; expected {year}")
    counters["owner_parcels_matched"] = len(owners)
    return owners, dict(counters)


def prepare_evidence(row, lm):
    """Withhold negative evidence when printed fields may have lost a suffix."""
    row = dict(row)
    notes = []
    if row.get("source_name_may_be_truncated") and not lm.name_is_financialized(row.get("owner_name")):
        row["owner_name"] = ""
        notes.append("printed name may truncate an entity marker; negative name classification withheld")
    no_situs = not (lm.clean_text(row.get("situs_street")) or lm.clean_text(row.get("situs_number"))) or not (row.get("situs_city") and row.get("situs_state") and row.get("situs_zip"))
    no_mail_locality = not (row.get("owner_addr_city") and row.get("owner_addr_state") and row.get("owner_addr_zip"))
    clipped = row.get("source_address_may_be_truncated") or row.get("source_situs_may_be_truncated")
    mismatch = lm.classify_owner_row(row)["address_match"] is not True
    if no_situs or no_mail_locality or (clipped and mismatch):
        for field in ("owner_addr_line1", "owner_addr_line2", "owner_addr_line3"):
            row[field] = ""
        notes.append("incomplete source-year mailing/situs address; negative address match withheld")
    row["adapter_evidence_notes"] = notes
    return row


def snapshot(target, year, rows, lm):
    adapted = [prepare_evidence(row, lm) for row in rows]
    result = lm.aggregate_owner_rows(adapted,
        is_residential=lm.parse_optional_bool(target["is_residential"]))
    notes = sorted({note for row in adapted for note in row["adapter_evidence_notes"]})
    if not rows:
        result["classification_status"] = "source_parcel_not_found"
        result["classification_note"] = "no source-year owner record matched the fixed target parcel"
        for field in ("is_owner_occupied", "has_financialized_owner", "is_corporate_owned"):
            result[field] = None
    elif notes:
        result["classification_note"] = "; ".join(filter(None, [result["classification_note"], *notes]))
        if result["classification_status"] not in {"matched_ambiguous", "matched_owner_suppressed"} and any(result[key] is None for key in ["is_owner_occupied", "has_financialized_owner", "is_corporate_owned"]):
            result["classification_status"] = "matched_evidence_insufficient"
    result.update({
        "source_county": target["source_county"], "tax_year": year,
        "parcel_id": target["parcel_id"],
        "classification_rule_version": getattr(lm, "ews_rule_version", None) or lm.classifier_rule_version(),
        "source_snapshot_id": "; ".join(sorted({row["source_snapshot_id"] for row in rows})),
        "source_owner_field": "; ".join(sorted({row["source_owner_field"] for row in rows})),
        "source_supplement_number": "; ".join(sorted({row["source_supplement_number"] for row in rows})),
        "residential_use_category": target.get("residential_use_category", ""),
        "source_reported_owner_names": "; ".join(sorted({row["source_reported_owner_name"] for row in rows})),
        "source_report_exemption_codes": "; ".join(sorted({row["source_exemption_codes"] for row in rows})),
    })
    for field in EXTRA_FIELDS:
        if field not in result:
            result[field] = any(row.get(field, False) for row in rows) if field in BOOL_FIELDS else ""
    return result


def write_table(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("TRUE" if value else "FALSE") if isinstance(value, bool)
                             else value for key, value in row.items()})


def verified_gis(args, config, lm, year=2025):
    if int(config.get("tax_year", year)) != year:
        raise ValueError("GIS source configuration disagrees with requested tax year")
    evidence = args.output_dir / f"williamson_txgio_{year}_evidence.csv"
    manifest_path = args.output_dir / f"williamson_txgio_{year}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config_record = manifest["inputs"].get("source_config")
    if config_record and config_record["sha256"] != digest(args.root / "config/williamson_ownership_sources.json"):
        raise ValueError("GIS source configuration checksum mismatch; rerun preparation")
    if int(manifest.get("source", {}).get("tax_year", year)) != year:
        raise ValueError("GIS preparation manifest disagrees with requested tax year")
    archive = args.root / config["path"]
    if (digest(archive) != config["sha256"] or
            manifest["inputs"]["source_zip"]["sha256"] != config["sha256"] or
            manifest["inputs"]["target_csv"]["sha256"] != digest(args.target) or
            manifest["output"]["sha256"] != digest(evidence)):
        raise ValueError("GIS source, target, or prepared evidence checksum mismatch; rerun preparation")
    with evidence.open(newline="", encoding="utf-8-sig") as handle:
        rows, qa = reconciliation.deduplicate_gis_rows(csv.DictReader(handle), lm,
            year=year, acquisition_date=config.get("acquisition_date"))
    if manifest.get("output", {}).get("rows", qa["raw_rows"]) != qa["raw_rows"]:
        raise ValueError("GIS evidence row count disagrees with preparation manifest")
    if manifest.get("target_coverage", {}).get("matched_parcels", len(rows)) != len(rows):
        raise ValueError("GIS parcel coverage disagrees with preparation manifest")
    return rows, {"configuration": config, "preparation_manifest": manifest,
                  "preparation_manifest_sha256": digest(manifest_path), "deduplication": qa}


def reconcile_snapshots(selected, owners, gis, lm, year=2025):
    primary, certified_only, review = [], [], []
    flags = ("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")
    for key, target in sorted(selected.items()):
        cert_rows, gis_rows = owners.get(key, []), gis.get(key, [])
        rows, diagnostic = reconciliation.reconcile_year(cert_rows, gis_rows, lm, year=year)
        result = snapshot(target, year, rows, lm)
        result.update({"source_" + field: diagnostic[field] for field in
            ("reconciliation_status", "reconciliation_conflict", "reconciliation_version",
             "name_extended", "name_completeness_confirmed", "mailing_extended", "situs_extended")})
        cert = snapshot(target, year, cert_rows, lm)
        # The GIS-only variant must not inherit certified homesteads or owner IDs.
        gis_selected, _ = reconciliation.reconcile_year([], gis_rows, lm, year=year)
        gis_result = snapshot(target, year, gis_selected, lm)
        item = {"parcel_id": key, "source_county": "Williamson", "tax_year": year,
                "residential_units": float(target["property_units"]), **diagnostic,
                "certified_owner_names": "; ".join(row["owner_name"] for row in cert_rows),
                "gis_owner_names": "; ".join(row["owner_name"] for row in gis_rows),
                "certified_mailing": "; ".join(lm.source_owner_address(row) for row in cert_rows),
                "gis_mailing": "; ".join(lm.source_owner_address(row) for row in gis_rows)}
        for variant, values in (("primary", result), ("certified_only", cert), ("gis_only", gis_result)):
            item[variant + "_matched"] = values["n_owner_rows"] > 0
            for field in flags:
                item[variant + "_" + field] = values[field]
        primary.append(result)
        certified_only.append(cert)
        review.append(item)
    return primary, certified_only, review


def reconciliation_qa(review):
    grouped = defaultdict(lambda: {"parcels": 0, "units": 0.0})
    for row in review:
        units = row["residential_units"]
        metrics = [("reconciliation_status", row["reconciliation_status"]),
                   ("source_conflict", str(row["reconciliation_conflict"]))]
        for field in ("name_extended", "name_completeness_confirmed", "mailing_extended", "situs_extended"):
            metrics.append((field, str(row[field])))
        for variant in ("primary", "certified_only", "gis_only"):
            metrics.append((variant + "_matched", str(row[variant + "_matched"])))
            for flag in ("is_corporate_owned", "has_financialized_owner", "is_owner_occupied"):
                value = row[variant + "_" + flag]
                metrics.append((variant + "_" + flag, "unknown" if value is None else str(value)))
        for metric, category in metrics:
            grouped[metric, category]["parcels"] += 1
            grouped[metric, category]["units"] += units
    return [{"metric": metric, "category": category, **values}
            for (metric, category), values in sorted(grouped.items())]


def run(args):
    lm = load_classifier(args.classifier)
    lm.ews_rule_version = lm.classifier_rule_version()
    targets = read_target(args.target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.root / "config/williamson_ownership_sources.json"
    config = json.loads(config_path.read_text())
    gis_by_year = {year: verified_gis(args, config[f"gis_{year}"], lm, year=year)
                   for year in YEARS}
    sources = []
    snapshots = []
    qa = []
    supplements = {}
    for county in COUNTIES:
        selected = {key: row for key, row in targets.items() if row["source_county"] == county}
        for year in YEARS:
            archive = args.root / "data/raw_parcels/appraisal_history" / county.lower() / str(year) / f"{county.lower()}_{year}.zip"
            if county == "Williamson":
                archive = args.root / config[f"certified_{year}"]["path"]
                if not archive.is_file() or digest(archive) != config[f"certified_{year}"]["sha256"]:
                    raise ValueError(f"Missing or altered pinned Williamson {year} certified source")
            unavailable = not archive.exists()
            if unavailable:
                owners, counts = {}, {"owner_parcels_matched": 0}
            elif county == "Hays":
                owners, counts = parse_hays(archive, year, selected, lm)
            else:
                owners, counts = parse_williamson_report(archive, year, selected)
            if county == "Williamson":
                gis, gis_provenance = gis_by_year[year]
                # Earlier vintages may legitimately omit newer fixed-support
                # parcels. Keep certified-only/missing branches, never borrow
                # the other year's GIS to fill a missing ID.
                if not set(gis).issubset(selected):
                    raise ValueError("Prepared GIS evidence contains non-target Williamson IDs")
                current, certified_only, review = reconcile_snapshots(selected, owners, gis, lm, year=year)
                fields = [name for name in lm.SNAPSHOT_FIELDS if name != "property_units"] + EXTRA_FIELDS
                supplementary_tables = {
                    f"williamson_{year}_certified_only_snapshots.csv": (certified_only, fields),
                    f"williamson_{year}_source_reconciliation.csv": (review, None),
                    f"williamson_{year}_source_reconciliation_qa.csv": (reconciliation_qa(review), None),
                    f"williamson_{year}_source_conflict_review.csv": (
                        [row for row in review if row["reconciliation_conflict"]], list(review[0])),
                }
                for name, (table, columns) in supplementary_tables.items():
                    path = args.output_dir / name
                    write_table(path, table, columns)
                    supplements[name] = {"path": str(path.resolve()), "sha256": digest(path), "rows": len(table)}
            else:
                current = [snapshot(target, year, owners.get(key, []), lm)
                           for key, target in sorted(selected.items())]
            if unavailable:
                for row in current:
                    row["classification_status"] = "source_snapshot_unavailable"
                    row["classification_note"] = f"No verified source-year {county} {year} owner snapshot is available; current ownership was not substituted."
                    for field in BOOL_FIELDS:
                        row[field] = None
            snapshots.extend(current)
            status_counts = Counter(row["classification_status"] for row in current)
            source = {
                "county": county, "tax_year": year, "path": str(archive.resolve()),
                "sha256": digest(archive) if not unavailable else None,
                "bytes": archive.stat().st_size if not unavailable else None,
                "available": not unavailable,
                "source_url": (HAYS_URL.format(year=year) if county == "Hays" else WCAD_REPORT_URL.format(year=year)) if not unavailable else None,
                "temporal_evidence": ("annual certified archive and embedded member names" if county == "Hays" else "embedded report CERTIFIED ROLL year checked on every page") if not unavailable else "source unavailable; current owner extract explicitly excluded",
                "target_parcels": len(selected), **counts,
                "classification_status_counts": dict(status_counts),
            }
            if not unavailable:
                with zipfile.ZipFile(archive) as zipped:
                    source["archive_members"] = [
                        {"name": item.filename, "uncompressed_bytes": item.file_size,
                         "crc32": f"{item.CRC:08x}",
                         "member_timestamp_local_unspecified": "%04d-%02d-%02dT%02d:%02d:%02d" % item.date_time}
                        for item in zipped.infolist()]
            if county == "Williamson":
                source["source_url"] = config[f"certified_{year}"]["source_url"]
                source["certified_configuration"] = config[f"certified_{year}"]
                source["gis_supplement"] = gis_provenance
                source["reconciliation_version"] = reconciliation.RECONCILIATION_VERSION
                source["reconciliation_status_counts"] = dict(Counter(row["source_reconciliation_status"] for row in current))
                source["reconciliation_conflict_parcels"] = sum(row["source_reconciliation_conflict"] for row in current)
            sources.append(source)
            qa.append({"source_county": county, "tax_year": year, "target_parcels": len(selected),
                       "matched_parcels": sum(row["n_owner_rows"] > 0 for row in current),
                       "corporate_known_parcels": sum(row["is_corporate_owned"] is not None for row in current),
                       "financialized_known_parcels": sum(row["has_financialized_owner"] is not None for row in current),
                       "fully_classified_parcels": status_counts["matched_classified"]})
            print(json.dumps(qa[-1]), flush=True)
    output = args.output_dir / "other_county_owner_snapshots_2024_2025.csv"
    fields = [name for name in lm.SNAPSHOT_FIELDS if name != "property_units"] + EXTRA_FIELDS
    write_table(output, snapshots, fields)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_version": ADAPTER_VERSION, "adapter_sha256": digest(Path(__file__)),
        "reconciliation_module_sha256": digest(Path(reconciliation.__file__)),
        "source_configuration_sha256": digest(config_path),
        "classifier_commit": subprocess.check_output(["git", "-C", str(args.classifier.parent), "rev-parse", PINNED_CLASSIFIER_COMMIT], text=True).strip(),
        "classifier_rule_version": lm.classifier_rule_version(), "classifier_sha256": digest(args.classifier),
        "target_path": str(args.target.resolve()), "target_sha256": digest(args.target),
        "sources": sources, "qa": qa,
        "supplementary_outputs": supplements,
        "output": {"path": str(output.resolve()), "sha256": digest(output), "rows": len(snapshots)},
        "coverage_limitations": [
            "Current fixed EWS residential eligibility and parcel support are used; historical owner, situs and exemption evidence are annual.",
            "WCAD printed names/addresses can be truncated. Positive entity markers survive clipping; ambiguous negative markers and clipped address mismatches are withheld as unknown.",
            "WCAD printed reports expose one printed owner record per matched parcel; they do not establish a complete separate co-owner roster or ownership shares.",
            "Each year's GIS evidence supplements only that year's certified roll: matching parcel ID, exact or clipped-prefix owner name, and corroborating mailing street/city/state/ZIP are required for extensions or completeness confirmation. No ownership evidence is carried across years.",
            "Owner or mailing disagreement retains certified evidence. Situs disagreement retains the certified address pair but may allow independently corroborated name confirmation; every such parcel is flagged for source-agreement sensitivity.",
            "GIS-only parcels use GIS name and its own mailing/situs evidence, with no homestead or owner ID inferred. Identical duplicate features collapse before classification and unit aggregation; placeholders remain unknown.",
            "WCAD current wcad_owners.csv is a 2026 extract and is not valid as either 2024 or 2025 ownership.",
            "These are retrospective certified-year snapshots, not an exact April 1 publication-state reconstruction.",
        ],
    }
    with (args.output_dir / "other_county_sources_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--classifier", type=Path, default=ROOT.parent / "landlord-mapper/historical_ownership.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/part2/ownership")
    parser.add_argument("--list-inputs", action="store_true")
    args = parser.parse_args()
    if args.list_inputs:
        print(json.dumps([str(path) for path in input_paths(args.root)]))
        return
    if args.target is None:
        parser.error("--target is required unless --list-inputs is used")
    run(args)


if __name__ == "__main__":
    main()
