# Hays JP4 canonical report update

Completed October 7, 2026. The September 29 report is the sole canonical JP4
source, as requested. It replaces the August report in the extractor and all
current inventory and screening outputs. The supplied JP5 report is unchanged.

## Canonical inventory

`data/raw_hays_evictions/Odyssey-JobOutput-September_29_2026_10-36-25-2795841-1.pdf`
contains 206 unique JP4 eviction cases selected by **Filed Date: January 1, 2020
through July 30, 2026**. Its printed grand total is 206; the last page contains
only a footer. All cases have filing dates. Statuses are 77 Dismissed, 114
Disposed, 13 Filed, and 2 Judgment. This replaces the previous inactive-status
selection and resolves that specific JP4 coverage concern.

The combined canonical inventory has **1,170 cases: 206 JP4 and 964 JP5**.
There are 15 JP4 additions and two old-only 2019 cases excluded from the current
inventory. April 2, 2024–April 1, 2026 contains 99 JP4 filings and 532 JP5 filings,
before geographic screening. Seven of the 15 additions fall in this window.

Source columns were refreshed by court/case key. Previously collected portal
addresses, source URLs, observation dates, and manual-review fields were
preserved. The persistent review log retains two historical 2019 observations
for audit only; neither contributes to the canonical inventory or analysis.
A pre-update data snapshot is under
`data/raw_hays_evictions/snapshots/before_jp4_update_2026-10-07/`.
The obsolete PDF is not an input to any current extraction or report.

## Added cases and addresses

All 15 added cases were found by exact case-number search in the Hays public
portal. Thirteen have defendant addresses. F20-003J4 and F20-005J4 have no listed
defendant address; both are outside the analysis window. All seven additions
within the window have address information.

| Added current-window case | Preliminary result | Evidence |
| --- | --- | --- |
| F24-016J4 | Unresolved | Multiple distinct defendant addresses, including addresses at The Springs and another address with an Austin postal city. No premises selected. |
| F24-043J4 | Outside property group | The Springs / Dripping Springs Apartments, previously mapped outside Austin. |
| F25-006J4 | Candidate address outside Austin | CAD parcel R165669, approximately 11.0 km outside the FULL-purpose boundary. |
| F25-017J4 | Candidate address outside Austin | CAD parcel R53181, approximately 13.3 km outside the FULL-purpose boundary. |
| F25-031J4 | Outside property group | Western Springs Apartments, previously mapped outside Austin. |
| F25-047J4 | Outside property group | The Springs / Dripping Springs Apartments, previously mapped outside Austin. The portal's street text differs from the registered property address; this remains property-level triage. |
| F26-002J4 | Candidate address outside Austin | CAD parcel R138121, approximately 8.2 km outside the FULL-purpose boundary, despite its Austin mailing city. |

These are candidate party-address and property-group results, not verified
historical eviction premises. The conflicting-address case remains in active
review even though its plaintiff matches an outside-Austin property group.
The same rule is applied to all collected cases. No address is inferred from a
plaintiff where defendant information is missing.

Raw observations are saved in
`data/raw_hays_evictions/portal_observations/2026-10-07.jsonl`, with case numbers,
filing dates, exact URLs, observation date, and party-to-address associations.
Demographic and government identifier fields are excluded. The handoff CSV
`output/hays_eviction_jp4_update_cases.csv` contains all 15 additions, their
addresses, and separate preliminary screening results.

## Refreshed geography and checks

The current canonical cases partition into 516 active-review cases and 654
property-group deferrals. All active cases have been looked up. Of the 235
active cases within the analysis window:

| Candidate geography / unresolved category | Cases |
| --- | ---: |
| Outside Austin, local CAD parcel match | 88 |
| Outside Austin, previously accepted Census street match | 67 |
| Single address without an accepted location | 57 |
| Multiple distinct defendant addresses | 22 |
| Missing defendant address | 1 |
| Total | 235 |

This leaves **155 outside candidate locations and 80 unresolved cases** in the
active current-window queue. Another 396 current-window cases are deferred by
property-group screening. No accepted candidate location falls inside Austin;
this does not establish zero Austin evictions.

Geography uses the existing April 29, 2026 boundary snapshot, filtered to
CITY OF AUSTIN / FULL, and the existing county/state parcel geometry. Matching
normalizes PLACE to PL as well as previously supported street suffixes. Entire
matching parcels are checked; near-boundary or crossing parcels remain review
items. All three new single-address cases requiring individual checks matched
local parcels, so no new external geocoder request was necessary. An attempted
Census request was blocked by automatic approval review before execution;
the existing Census cache is unchanged.

Validation confirmed:

- 206 JP4 identifiers and filing dates independently reconciled between
  pdfplumber and pypdf; the printed total also matches.
- All 155 JP4 filing dates with saved portal observations agree with the PDF.
- All 964 JP5 source rows are unchanged.
- All prior manual observations are preserved; all 231 previous current-window
  geography decisions are unchanged.
- All 182 prior address IDs still refer to the same address strings. A persistent
  ID registry prevents reordered cases from receiving another address's cache.
- All 15 additions have a saved portal observation; source and review keys are
  unique, and the active/deferred partition reconciles to the canonical inventory.

The corrected JP4 inventory is ready for continued premises/address review.
Hays has not yet been promoted into the scored pipeline or the analytical
coverage registry: unresolved geography and premises verification still matter.
No further clarification of the old inactive-status report is needed for this
workflow now that the replacement is canonical.

## Rebuild

With the supplied PDF and saved observations available locally:

```sh
python3 scripts/data/hays_evictions_prepare.py --sync-review
python3 scripts/data/hays_evictions_import_portal.py
Rscript scripts/data/hays_evictions_screen_properties.R
python3 scripts/data/hays_evictions_collection_report.py
Rscript scripts/data/hays_evictions_geocode_review.R
python3 scripts/data/hays_evictions_update_report.py
```

The extraction command needs pdfplumber and pypdf. Current detail outputs include
`output/hays_eviction_cases_extracted.csv`,
`output/hays_eviction_current_window_geography_review.csv`, and
`output/hays_eviction_current_window_geography_followup.csv`. The update audit's
machine-readable counts and checks are in
`output/hays_eviction_jp4_update_summary.json`.
