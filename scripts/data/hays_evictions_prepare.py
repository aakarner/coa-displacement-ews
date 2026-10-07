"""Extract supplied Hays PDF case indices; keep portal review separate from sources.

Run with Python containing pdfplumber and pypdf. Outputs are an intake inventory,
not evidence of complete filing coverage and not inputs to the scored pipeline.
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
CASE = re.compile(r"F\d{2}-\d+J[45]")
DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
JP4_DATE = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")
SOURCES = {
    "EV_2020-2026.pdf": ("JP5", [122, 312, 396, 512, 590], 737),
    "Odyssey-JobOutput-September_29_2026_10-36-25-2795841-1.pdf":
        ("JP4", None, 550),
}

SOURCE_REVIEW_FIELDS = ("case_key", "court", "case_number", "filing_date",
                        "plaintiff", "source_file", "source_page")


def extract_jp4_filing_report(path):
    """Canonical September report: filing-date selection and all case statuses."""
    rows = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            words = page.extract_words()
            starts = sorted((w for w in words if CASE.fullmatch(w["text"])),
                            key=lambda w: w["top"])
            for index, start in enumerate(starts):
                bottom = starts[index + 1]["top"] - 1 if index + 1 < len(starts) else 550
                row_words = [w for w in words if start["top"] - 1 <= w["top"] < bottom]
                totals = [w["top"] for w in row_words if w["text"] == "Grand"]
                if totals:
                    row_words = [w for w in row_words if w["top"] < min(totals) - 1]
                filed = text_column(row_words, 30, 120)
                style = text_column(row_words, 120, 350)
                status_text = text_column(row_words, 600, 760)
                assert text_column(row_words, 350, 500) == "Evictions"
                assert JP4_DATE.fullmatch(filed), (page_number, filed)
                parties = re.split(r"\s+vs\.\s*", style, maxsplit=1, flags=re.I)
                assert len(parties) == 2 and all(parties), style
                number = start["text"]
                assert number.endswith("J4")
                rows.append(dict(case_key=f"Hays|JP4|{number}", county="Hays", court="JP4",
                    case_number=number, case_number_year=2000 + int(number[1:3]),
                    filing_date=iso_date(filed), case_style=style, plaintiff=parties[0],
                    defendants=parties[1], case_type="Evictions",
                    case_status=JP4_DATE.sub("", status_text).strip(),
                    case_status_date=iso_date(status_text), statistical_closure_date="",
                    statistical_closure="", report_selection="filed_date",
                    report_filter_start="2020-01-01", report_filter_end="2026-07-30",
                    source_file=str(path.relative_to(ROOT)), source_page=page_number,
                    source_page_end=page_number, source_row_on_page=index + 1))
    reference = "\n".join(p.extract_text() for p in PdfReader(path).pages)
    assert Counter(CASE.findall(reference)) == Counter(r["case_number"] for r in rows)
    reference_dates = dict(re.findall(r"(F\d{2}-\d+J4)(\d{1,2}/\d{1,2}/\d{4})", reference))
    assert len(reference_dates) == len(rows) == 206
    for row in rows:
        assert iso_date(reference_dates[row["case_number"]]) == row["filing_date"]
        assert "2020-01-01" <= row["filing_date"] <= "2026-07-30"
    return rows


def text_column(words, left, right):
    selected = sorted((w for w in words if left <= w["x0"] < right),
                      key=lambda w: (round(w["top"], 1), w["x0"]))
    lines = []
    for word in selected:
        if not lines or abs(word["top"] - lines[-1][0]) > 2:
            lines.append((word["top"], []))
        lines[-1][1].append(word)
    return " ".join(" ".join(w["text"] for w in sorted(line, key=lambda w: w["x0"]))
                    for _, line in lines).strip()


def iso_date(text):
    matches = JP4_DATE.findall(text)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one date: {text!r}")
    return datetime.strptime(matches[0], "%m/%d/%Y").date().isoformat()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def extract_source(path, court, columns, page_bottom):
    rows = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            words = page.extract_words()
            starts = sorted((w for w in words if CASE.fullmatch(w["text"])
                             and w["x0"] < columns[0]), key=lambda w: w["top"])
            for index, start in enumerate(starts):
                bottom = starts[index + 1]["top"] - 1 if index + 1 < len(starts) else page_bottom
                row_words = [w for w in words if start["top"] - 1 <= w["top"] < bottom]
                end_page = page_number
                if court == "JP5" and index + 1 == len(starts) and page_number < len(pdf.pages):
                    next_words = pdf.pages[page_number].extract_words()
                    next_start = min(w["top"] for w in next_words
                                     if CASE.fullmatch(w["text"]) and w["x0"] < columns[0])
                    continuation = [w for w in next_words if 60 <= w["top"] < next_start - 1]
                    if continuation:
                        row_words.extend(dict(w, top=w["top"] + page.height) for w in continuation)
                        end_page = page_number + 1
                # The JP4 final-page grand total is below the final case.
                totals = [w["top"] for w in row_words if w["text"] == "Grand"]
                if totals:
                    row_words = [w for w in row_words if w["top"] < min(totals) - 1]
                style, type_text, officer_or_status, status_or_closure = [
                    text_column(row_words, a, b) for a, b in zip(columns, columns[1:])]
                if not type_text.startswith("Evictions"):
                    raise ValueError((path.name, page_number, start["text"], type_text))
                parties = re.split(r"\s+vs\.\s*", style, maxsplit=1, flags=re.I)
                if len(parties) != 2 or not all(parties):
                    raise ValueError(f"Unparsed case style: {style}")
                if court == "JP5":
                    filing_date = iso_date(type_text)
                    status_date = iso_date(status_or_closure)
                    status = DATE.sub("", status_or_closure).strip()
                    closure_date = closure = ""
                else:
                    filing_date = ""  # This report contains status dates, not filing dates.
                    status_date = iso_date(officer_or_status)
                    status = DATE.sub("", officer_or_status).strip()
                    closure_date = iso_date(status_or_closure) if DATE.search(status_or_closure) else ""
                    closure = DATE.sub("", status_or_closure).strip()
                case_number = start["text"]
                if not case_number.endswith(court.replace("JP", "J")):
                    raise ValueError(f"Court mismatch: {case_number}")
                rows.append(dict(
                    case_key=f"Hays|{court}|{case_number}", county="Hays", court=court,
                    case_number=case_number, case_number_year=2000 + int(case_number[1:3]),
                    filing_date=filing_date, case_style=style, plaintiff=parties[0],
                    defendants=parties[1], case_type="Evictions", case_status=status,
                    case_status_date=status_date, statistical_closure_date=closure_date,
                    statistical_closure=closure,
                    report_selection=("not_stated_in_pdf" if court == "JP5" else "inactive_case_status_date"),
                    report_filter_start=("" if court == "JP5" else "2020-01-01"),
                    report_filter_end=("" if court == "JP5" else "2026-07-30"),
                    source_file=str(path.relative_to(ROOT)), source_page=page_number,
                    source_page_end=end_page,
                    source_row_on_page=index + 1,
                ))
    # Independent text engine verifies identifiers across every source page.
    reference_ids = Counter(case for page in PdfReader(path).pages
                            for case in CASE.findall(page.extract_text()))
    assert reference_ids == Counter(row["case_number"] for row in rows), path.name
    if court == "JP5":
        reference_text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
        chunks = re.split(r"(F\d{2}-\d+J5)", reference_text)
        by_number = {row["case_number"]: row for row in rows}
        for index in range(1, len(chunks), 2):
            dates = DATE.findall(chunks[index + 1])
            row = by_number[chunks[index]]
            assert len(dates) >= 2
            assert (iso_date(dates[0]), iso_date(dates[1])) == (
                row["filing_date"], row["case_status_date"]), row["case_number"]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize-review", action="store_true",
                        help="Create a manual review CSV only if it does not exist.")
    parser.add_argument("--sync-review", action="store_true",
                        help="Refresh source columns and append new cases; preserve all manual observations.")
    args = parser.parse_args()
    rows, source_qa = [], []
    for filename, (court, columns, bottom) in SOURCES.items():
        path = ROOT / "data/raw_hays_evictions" / filename
        records = (extract_jp4_filing_report(path) if court == "JP4" else
                   extract_source(path, court, columns, bottom))
        assert len(records) == len({r["case_key"] for r in records}), filename
        if court == "JP4":
            assert len(records) == 206, "Mismatch with printed JP4 Grand Total"
        rows.extend(records)
        dates = [r["filing_date"] for r in records if r["filing_date"]]
        source_qa.append(dict(source_file=str(path.relative_to(ROOT)), court=court,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             pages=len(PdfReader(path).pages), cases=len(records),
                             missing_filing_dates=sum(not r["filing_date"] for r in records),
                             earliest_observed_filing_date=min(dates) if dates else "",
                             latest_observed_filing_date=max(dates) if dates else ""))
    assert len(rows) == len({r["case_key"] for r in rows})
    out = ROOT / "output"
    write_csv(out / "hays_eviction_cases_extracted.csv", rows)
    write_csv(out / "hays_eviction_source_qa.csv", source_qa)
    years = Counter((r["court"], r["case_number_year"]) for r in rows)
    write_csv(out / "hays_eviction_cases_by_case_number_year.csv", [
        dict(court=court, case_number_year=year, cases=count)
        for (court, year), count in sorted(years.items())])
    plaintiffs = Counter((r["court"], r["plaintiff"]) for r in rows)
    write_csv(out / "hays_eviction_plaintiff_review_groups.csv", [
        dict(court=court, plaintiff_raw=plaintiff, cases=count,
             verified_property_address="", property_evidence="",
             geography_review="unreviewed")
        for (court, plaintiff), count in sorted(plaintiffs.items(), key=lambda p: (-p[1], p[0]))])
    review_path = ROOT / "data/hays_eviction_address_review.csv"
    if (args.initialize_review and not review_path.exists()) or args.sync_review:
        queue = []
        if review_path.exists():
            with review_path.open(newline="") as f:
                queue = list(csv.DictReader(f))
        existing = {r["case_key"]: r for r in queue}
        assert len(existing) == len(queue)
        for row in rows:
            if row["case_key"] in existing:
                existing[row["case_key"]].update({k: row[k] for k in SOURCE_REVIEW_FIELDS})
                continue
            queue.append(dict(
                case_key=row["case_key"], court=row["court"], case_number=row["case_number"],
                filing_date=row["filing_date"], plaintiff=row["plaintiff"],
                source_file=row["source_file"], source_page=row["source_page"],
                portal_status="not_started", portal_url="", observed_on="", portal_filing_date="",
                defendant_addresses_raw="", candidate_premises_address="",
                candidate_unit="", address_evidence="", premises_verification="unreviewed",
                verified_premises_address="", verified_unit="",
                austin_boundary_status="unreviewed", notes=""))
        write_csv(review_path, queue)
    summary = dict(total_cases=len(rows), sources=source_qa,
                   exact_plaintiff_court_groups=len(plaintiffs),
                   jp5_filings_through_2026_04_01=sum(r["court"] == "JP5" and
                       r["filing_date"] <= "2026-04-01" for r in rows),
                   jp5_filings_2024_04_02_through_2026_04_01=sum(r["court"] == "JP5" and
                       "2024-04-02" <= r["filing_date"] <= "2026-04-01" for r in rows))
    (out / "hays_eviction_intake_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
