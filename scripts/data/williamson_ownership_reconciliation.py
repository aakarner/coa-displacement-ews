"""Conservative, source-year-only WCAD certified/GIS evidence reconciliation.

This changes source preparation, not landlord-mapper's pinned classifier.
Certified evidence is primary. GIS extensions need corroborating owner and
mailing evidence; homesteads are never transplanted to a different GIS owner.
Name corroboration can stand independently when situs disagrees, but that
disagreement remains flagged and the certified mailing/situs pair is retained.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

RECONCILIATION_VERSION = "ews-wcad-reconciliation-v2"
GIS_SNAPSHOT_ID = "txgio-williamson-2025-07-wcad-parcels"
UNAVAILABLE = {"", "UNAVAILABLE", "NOT AVAILABLE", "CONFIDENTIAL", "WITHHELD", "SUPPRESSED"}


def clean(value):
    return " ".join(str(value or "").upper().split()).strip()


def unavailable(value):
    return clean(value) in UNAVAILABLE


def validated_year(year):
    value = clean(year)
    if not re.fullmatch(r"\d{4}", value):
        raise ValueError("An explicit four-digit ownership tax year is required")
    return value


def acquisition_date_text(value):
    value = clean(value)
    if not re.fullmatch(r"\d{8}|\d{4}-\d{2}-\d{2}", value):
        raise ValueError("GIS acquisition date must be YYYYMMDD or YYYY-MM-DD")
    compact = value.replace("-", "")
    datetime.strptime(compact, "%Y%m%d")
    return compact


def source_vintage(year, acquisition_date):
    year = validated_year(year)
    if acquisition_date is None:
        if year != "2025":
            raise ValueError("An explicit verified acquisition date is required outside the 2025 default")
        acquisition_date = "20250701"
    acquisition_date = acquisition_date_text(acquisition_date)
    if acquisition_date[:4] != year:
        raise ValueError("GIS acquisition date must belong to the requested tax year")
    return year, acquisition_date


def standardize_gis(raw, lm, *, year=2025, acquisition_date=None):
    """Validate the caller's verified source vintage before mapping any fields.

    The historical July 2025 default remains available to existing callers;
    other annual sources require an explicit, independently verified date.
    """
    year, acquisition_date = source_vintage(year, acquisition_date)
    if (clean(raw.get("TAX_YEAR")) != year
            or acquisition_date_text(raw.get("DATE_ACQ")) != acquisition_date):
        raise ValueError(f"GIS ownership evidence must match verified {year}/{acquisition_date} vintage")
    if clean(raw.get("SOURCE")) != "WILLIAMSON APPRAISAL DISTRICT":
        raise ValueError("Unexpected GIS ownership source")
    key = clean(raw.get("Prop_ID"))
    if not key:
        raise ValueError("Missing GIS parcel ID")
    name = str(raw.get("OWNER_NAME") or "").strip()
    suppressed = unavailable(name)
    mail_suppressed = suppressed or unavailable(raw.get("MAIL_LINE1"))
    full_situs = str(raw.get("SITUS_ADDR") or "").strip()
    city, state, zipcode = (clean(raw.get(f)) for f in ("SITUS_CITY", "SITUS_STAT", "SITUS_ZIP"))
    # Parse the terminal locality from this source's own full situs; commas
    # within street/unit text are allowed. Never borrow current or mailing city.
    match = re.match(r"^(.*),\s*([^,]+?),?\s+([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\s*$",
                     full_situs, re.I)
    street = full_situs
    parse_conflict = False
    if match:
        street, parsed_city, parsed_state, parsed_zip = match.groups()
        for existing, parsed in ((city, parsed_city), (state, parsed_state), (zipcode[:5], parsed_zip[:5])):
            if existing and clean(existing) != clean(parsed):
                parse_conflict = True
        city, state, zipcode = city or clean(parsed_city), state or clean(parsed_state), zipcode or parsed_zip
    elif city and state and zipcode:
        # A full GIS address can omit commas. Its separately supplied locality
        # permits stripping only that exact terminal suffix, without guessing
        # a city boundary or borrowing another source's address components.
        city_pattern = re.escape(city).replace(r"\ ", r"\s+")
        terminal_locality = re.search(
            r"(?:,\s*|\s+)" + city_pattern + r",?\s+" + re.escape(state)
            + r"\s+" + re.escape(zipcode[:5]) + r"(?:-\d{4})?\s*$",
            full_situs, re.I,
        )
        if terminal_locality:
            street = full_situs[:terminal_locality.start()]
    if parse_conflict:
        street = ""
    return {
        "tax_year": year, "parcel_id": "WILLIAMSON:" + key,
        "owner_id": "",  # A property ID is not an owner ID.
        "owner_name": "" if suppressed else name, "owner_share": "",
        "owner_addr_line1": "" if mail_suppressed else str(raw.get("MAIL_LINE1") or "").strip(),
        "owner_addr_line2": "" if mail_suppressed else str(raw.get("MAIL_LINE2") or "").strip(),
        "owner_addr_line3": "", "owner_addr_city": str(raw.get("MAIL_CITY") or "").strip(),
        "owner_addr_state": str(raw.get("MAIL_STAT") or "").strip(),
        "owner_addr_zip": str(raw.get("MAIL_ZIP") or "").strip(),
        "owner_confidential_flag": "TRUE" if suppressed else "FALSE",
        "owner_address_suppressed_flag": "TRUE" if mail_suppressed else "FALSE",
        "homestead_flag": "",  # GIS has no homestead/exemption field.
        "situs_number": "", "situs_prefix": "", "situs_street": street.strip(" ,"),
        "situs_suffix": "", "situs_city": city.strip(" ,"), "situs_state": state,
        "situs_zip": zipcode, "source_situs_state_imputed": "FALSE",
        "source_snapshot_id": f"txgio-williamson-{year}-{acquisition_date[4:6]}-wcad-parcels",
        "source_owner_field": "TxGIO_OWNER_NAME",
        "source_supplement_number": "", "source_partial_owner_flag": "FALSE",
        "property_type_code": "", "improvement_state_code": "", "land_state_code": "",
        "source_name_may_be_truncated": len(name) >= 80,
        "source_address_may_be_truncated": False,
        "source_situs_may_be_truncated": False,
        "source_reported_owner_name": name, "source_exemption_codes": "",
        "source_year_verified": True, "source_situs_component_conflict": parse_conflict,
        "source_gis_name_care": str(raw.get("NAME_CARE") or "").strip(),
    }


def deduplicate_gis_rows(raw_rows, lm, *, year=2025, acquisition_date=None):
    year, acquisition_date = source_vintage(year, acquisition_date)
    result = defaultdict(list)
    signatures = defaultdict(set)
    count = 0
    # Geometry IDs and QA/export metadata cannot create additional owners.
    fields = ("OWNER_NAME", "NAME_CARE", "MAIL_ADDR", "MAIL_LINE1", "MAIL_LINE2",
              "MAIL_CITY", "MAIL_STAT", "MAIL_ZIP", "SITUS_ADDR", "SITUS_NUM",
              "SITUS_STRE", "SITUS_ST_1", "SITUS_ST_2", "SITUS_CITY", "SITUS_STAT",
              "SITUS_ZIP", "SOURCE", "DATE_ACQ", "TAX_YEAR")
    for raw in raw_rows:
        row = standardize_gis(raw, lm, year=year, acquisition_date=acquisition_date)
        count += 1
        key = row["parcel_id"]
        signature = tuple(clean(raw.get(field)) for field in fields)
        if signature not in signatures[key]:
            result[key].append(row)
            signatures[key].add(signature)
    qa = {"raw_rows": count, "parcels": len(result),
          "unique_evidence_rows": sum(map(len, result.values())),
          "duplicate_rows_removed": count - sum(map(len, result.values())),
          "conflicting_evidence_parcels": sum(len(rows) > 1 for rows in result.values())}
    return dict(result), qa


def name_relation(cert, gis):
    c, g = clean(cert.get("owner_name")), clean(gis.get("owner_name"))
    if unavailable(c) or unavailable(g):
        return "unavailable"
    if c == g:
        return "exact"
    if cert.get("source_name_may_be_truncated") and len(c) >= 29 and g.startswith(c):
        return "certified_prefix"
    return "disagree"


def address_relation(cert, gis, lm, *, situs=False):
    if situs:
        components = ("situs_city", "situs_state", "situs_zip")
        delivery = ("situs_number", "situs_prefix", "situs_street", "situs_suffix")
        clipped = cert.get("source_situs_may_be_truncated", False)
    else:
        components = ("owner_addr_city", "owner_addr_state", "owner_addr_zip")
        delivery = ("owner_addr_line1", "owner_addr_line2", "owner_addr_line3")
        clipped = cert.get("source_address_may_be_truncated", False)
    for field in components:
        c, g = clean(cert.get(field)), clean(gis.get(field))
        if unavailable(c) or unavailable(g):
            return "unavailable"
        if field.endswith("zip"):
            c, g = c[:5], g[:5]
        if c != g:
            return "disagree"
    c = lm.normalize_address(" ".join(str(cert.get(f) or "") for f in delivery))
    g = lm.normalize_address(" ".join(str(gis.get(f) or "") for f in delivery))
    if not c or not g:
        return "unavailable"
    if c == g:
        return "exact"
    if clipped and len(c) >= 10 and g.startswith(c):
        return "certified_prefix"
    return "disagree"


def reconcile_year(certified_rows, gis_rows, lm, *, year):
    """Reconcile one explicitly requested tax year without crossing vintages."""
    year = validated_year(year)
    for row in [*certified_rows, *gis_rows]:
        if str(row.get("tax_year")) != year:
            raise ValueError(f"{year} reconciliation cannot supplement another tax year")
    keys = {row.get("parcel_id") for row in [*certified_rows, *gis_rows]}
    if len(keys) > 1:
        raise ValueError("Cannot reconcile different parcel IDs")
    diag = dict(reconciliation_status="certified_only", reconciliation_conflict=False,
                reconciliation_version=RECONCILIATION_VERSION,
                name_relation="unavailable", mailing_relation="unavailable", situs_relation="unavailable",
                name_extended=False, name_completeness_confirmed=False,
                mailing_extended=False, situs_extended=False,
                certified_present=bool(certified_rows), gis_present=bool(gis_rows))
    selected = [dict(row) for row in certified_rows]
    if not gis_rows:
        if not certified_rows:
            diag["reconciliation_status"] = "no_source_record"
        return selected, diag
    if len(gis_rows) > 1:
        diag.update(reconciliation_status="gis_duplicate_conflict", reconciliation_conflict=True)
        if not selected:
            ambiguous = dict(gis_rows[0], owner_id="", owner_name="", homestead_flag="",
                             owner_confidential_flag="TRUE", owner_address_suppressed_flag="TRUE",
                             source_partial_owner_flag="TRUE")
            selected = [ambiguous]
        return selected, diag
    gis = gis_rows[0]
    if not certified_rows:
        diag["reconciliation_status"] = "gis_only_no_homestead"
        diag["reconciliation_conflict"] = bool(gis.get("source_situs_component_conflict"))
        return [dict(gis, owner_id="", homestead_flag="")], diag
    if len(certified_rows) != 1:
        diag.update(reconciliation_status="certified_multiple_owner_rows", reconciliation_conflict=True)
        return selected, diag
    cert = certified_rows[0]
    if lm.parse_optional_bool(cert.get("owner_confidential_flag")) is True:
        diag["reconciliation_status"] = "certified_suppressed_no_merge"
        return selected, diag
    if unavailable(gis.get("owner_name")):
        diag["reconciliation_status"] = "gis_unavailable_certified_retained"
        return selected, diag
    diag.update(name_relation=name_relation(cert, gis),
                mailing_relation=address_relation(cert, gis, lm),
                situs_relation=address_relation(cert, gis, lm, situs=True))
    agree = {"exact", "certified_prefix"}
    if diag["name_relation"] not in agree:
        diag.update(reconciliation_status="owner_disagreement_certified_retained", reconciliation_conflict=True)
        return selected, diag
    if diag["mailing_relation"] not in agree:
        diag.update(reconciliation_status="mailing_unconfirmed_certified_retained", reconciliation_conflict=True)
        return selected, diag
    # Name extension alone is safe under owner/mailing corroboration, but use
    # no GIS address pair if its situs disagrees with the certified property.
    merged = selected[0]
    if diag["name_relation"] == "certified_prefix":
        merged["owner_name"] = gis["owner_name"]
        diag["name_extended"] = True
    # An exact 29/30-character GIS name can establish that the printed column
    # ended at the actual name boundary. Retain possible clipping at GIS's
    # observed 80-character boundary, including after a prefix extension.
    merged["source_name_may_be_truncated"] = gis["source_name_may_be_truncated"]
    diag["name_completeness_confirmed"] = bool(
        cert.get("source_name_may_be_truncated")
        and not gis["source_name_may_be_truncated"]
    )
    if diag["situs_relation"] in agree and not gis.get("source_situs_component_conflict"):
        mail_fields = ("owner_addr_line1", "owner_addr_line2", "owner_addr_line3", "owner_addr_city",
                       "owner_addr_state", "owner_addr_zip", "owner_address_suppressed_flag")
        situs_fields = ("situs_number", "situs_prefix", "situs_street", "situs_suffix",
                        "situs_city", "situs_state", "situs_zip")
        # Use one coherent GIS mailing/situs pair, only after both corroborate.
        # The certified HS/owner ID remain attached to the corroborated owner.
        for field in (*mail_fields, *situs_fields):
            merged[field] = gis.get(field, "")
        merged["source_address_may_be_truncated"] = gis["source_address_may_be_truncated"]
        merged["source_situs_may_be_truncated"] = gis["source_situs_may_be_truncated"]
        diag["mailing_extended"] = bool(cert.get("source_address_may_be_truncated"))
        diag["situs_extended"] = bool(cert.get("source_situs_may_be_truncated"))
        diag["reconciliation_status"] = "same_owner_corroborated"
    else:
        diag.update(reconciliation_status="situs_unconfirmed_certified_address_retained", reconciliation_conflict=True)
    if (diag["name_extended"] or diag["name_completeness_confirmed"]
            or diag["reconciliation_status"] == "same_owner_corroborated"):
        merged["source_snapshot_id"] = cert["source_snapshot_id"] + "; " + gis["source_snapshot_id"]
        merged["source_owner_field"] = cert["source_owner_field"] + "; TxGIO_corroborated_evidence"
    return selected, diag


def reconcile_2025(certified_rows, gis_rows, lm):
    """Compatibility wrapper for the original verified 2025 workflow."""
    return reconcile_year(certified_rows, gis_rows, lm, year=2025)
