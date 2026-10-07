# Hays eviction records intake and address-collection pilot

Current JP4 source and refreshed results: [October canonical-report update](hays-eviction-jp4-update-2026-10.md). This document preserves the September snapshot.

Reviewed September 10, 2026. The user supplied two PDFs in
`data/raw_hays_evictions`, identified in the conversation as records Hays sent
to the City of Austin. The PDFs themselves do not establish their recipient or
that the records were filtered to Austin municipal limits.

## Supplied records

| Source | Court | Pages | Unique cases | Date information |
| --- | --- | ---: | ---: | --- |
| `EV_2020-2026.pdf` | JP5 | 51 | 964 | Observed filing dates January 3, 2020–July 29, 2026. The report does not print its selection criteria. |
| `Odyssey-JobOutput-August_11__2026_10-20-50-2741869-1.pdf` | JP4 | 17 | 193 | Selected by inactive case status date, January 1, 2020–July 30, 2026. Filing dates are absent. |
| Total | JP4 and JP5 | 68 | 1,157 | No duplicate court/case keys. |

Both reports were printed August 11, 2026. JP4's 193 extracted records match
its printed grand total. Its selection includes two case numbers from 2019,
and the supplied report cannot establish a complete filing cohort: cases that
do not satisfy its inactive-status filter are outside the selection.
The JP4 source also includes business defendants, so residential use requires
review. Its F24-010J4 statistical closure date is genuinely blank in the PDF.

| Case-number year (not a substitute for filing date) | JP4 | JP5 |
| --- | ---: | ---: |
| 2019 | 2 | 0 |
| 2020 | 16 | 52 |
| 2021 | 14 | 45 |
| 2022 | 20 | 89 |
| 2023 | 30 | 99 |
| 2024 | 50 | 157 |
| 2025 | 42 | 308 |
| 2026 | 19 | 214 |

There are 849 supplied JP5 filings through the project's April 1, 2026
reference date. Of those, 532 were filed April 2, 2024–April 1, 2026, the two
adjacent annual windows used for the current eviction measure. These are
counts before address, residential-use, or municipal-boundary review.
The same filing-date filters cannot be applied to JP4 until its filing dates
are recovered. Case-number year can help order work but cannot supply a date.

## Extraction and checks

`scripts/data/hays_evictions_prepare.py` uses PDF text coordinates to keep the
columns separate and joins the 27 JP5 records that span pages. It preserves
original party text, case status, source page and end page, and report date
semantics. A second PDF text engine independently reconciles all case numbers
and the filing/status dates of all 964 JP5 cases. Source hashes are recorded.
Representative source layouts were rendered for visual inspection.

Run with a Python environment containing `pdfplumber` and `pypdf`:

```sh
python scripts/data/hays_evictions_prepare.py --initialize-review
```

Generated intake files:

- `output/hays_eviction_cases_extracted.csv`: all case records and provenance.
- `output/hays_eviction_source_qa.csv`: source hashes, counts and date availability.
- `output/hays_eviction_cases_by_case_number_year.csv`: court/year inventory.
- `output/hays_eviction_plaintiff_review_groups.csv`: 452 exact plaintiff/court
  groups, including spelling/capitalization variants; these are not 452 verified
  properties. This is a generated grouping aid, not the authoritative review log.
- `output/hays_eviction_intake_summary.json`: intake summary and window counts.

`data/hays_eviction_address_review.csv` is the persistent manual review queue.
The initializer creates it only if absent; rerunning extraction does not
overwrite reviewed addresses. Each row retains its stable court/case key and
source reference. Source filing dates and portal filing dates are separate.
Candidate addresses and verified premises addresses are also separate.
No scored pipeline, coverage registry, or existing eviction output was changed.

## Live portal pilot

Started at the county's [portal landing page](https://portal-txhays.tylertech.cloud/PublicAccess/default.aspx),
selected Civil, Family & Probate Case Records, completed the CAPTCHA with
explicit user authorization, and searched by exact case number. Both pilot
lookups returned one matching result. Addresses are selectable page text under
Party Information, so browser-assisted transcription is feasible.

| Case | Portal finding | Review disposition |
| --- | --- | --- |
| [F20-001J5](https://portal-txhays.tylertech.cloud/PublicAccess/CaseDetail.aspx?CaseID=13129355) | Two different defendant addresses. One is 5500 Overpass Rd., unit 0810, Buda, TX 78610, matching the plaintiff street address. | Candidate premises address; conflicting party addresses require review. |
| [F26-020J4](https://portal-txhays.tylertech.cloud/PublicAccess/CaseDetail.aspx?CaseID=13379298) | Defendant address is 383 Rocky Ridge Trail, apartment 9026, Austin, TX 78737. Filing date is June 30, 2026; the PDF's July 23, 2026 date is a status date. | Candidate party address; premises identity and municipal inclusion remain unverified. |

The two samples were captured September 10, 2026 in the review CSV. Neither
is marked as a verified eviction-premises address. The first sample's other
address has a San Antonio/78723 inconsistency; its raw text is preserved rather
than silently corrected. DOB, physical descriptors and financial details are
not needed for this collection and were not added to the review dataset.
The observed event lists mention petitions and other documents, but those
events were not downloadable document links in the visible public pages.

## Collection strategy and limitations

1. Establish the target geography and time period. A small Austin analysis
   footprint does not make the supplied court lists small. An Austin postal
   city or ZIP is not evidence of inclusion in the project's exact FULL-purpose
   municipal boundary.
2. Use plaintiff/property groups to prioritize geographic checks. Verify the
   property and its location before treating repeated filings as one location.
   A landlord can have multiple properties, and variants can name one property.
   Do not exclude a case solely because of a plaintiff's name or mailing city.
3. For retained and unresolved cases, record raw party addresses, candidate
   premises address/unit, portal URL, observation date, portal filing date,
   verification evidence and review status. Resolve contradictory or absent
   addresses from case documents or the relevant court.
4. Geocode reviewed premises and apply the existing city boundary. Preserve
   unresolved cases as missing, not zero. Portal addresses may reflect updates
   after filing; capture date and source matter for historical analysis.
5. Request a JP4 filing-date-selected report including all statuses, or another
   demonstrated complete source, before claiming complete annual filing
   coverage. Recovering addresses for the 193 supplied cases does not recover
   cases omitted by the report's selection.

Only two live portal lookups are complete in this intake pilot; the other 1,155
queue entries are unstarted. At an illustrative one to two minutes per lookup,
collecting all 1,157 would take about 19–39 hours before difficult address
verification. This is a planning assumption, not measured throughput. Geographic
triage may reduce that workload. The subsequent
[property geography screening](hays-eviction-geography-screen-2026-09.md)
retains 506 cases for lookup and defers 651 cases in identified outside property
groups. These are preliminary property-level decisions, not verified premises.
