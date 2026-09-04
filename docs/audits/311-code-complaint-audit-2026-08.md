# Austin Code Complaint Linkage Audit: August 2026

## Purpose

This audit tests whether the City of Austin's [Austin Code Complaint Cases
dataset](https://data.austintexas.gov/Public-Safety/Austin-Code-Complaint-Cases/6wtj-zbtb)
can refine the EWS 311 smoke signal. The complaint dataset supplies case
descriptions and workflow outcomes; the existing EWS extract supplies the
original 311 intake record and its location. The audit uses the April 1, 2026
analysis cutoff and does not change the Part 1 feature or clusters.

The audit distinguishes three records that are easy to conflate:

- a **Code case** is a record in the complaint-case dataset;
- a **311 request** is the original public service request; and
- a **residentially linked request** is a 311 request whose Code case parcel ID
  matches a parcel in the promoted EWS residential-unit surface.

## Source and Linkage

The source contains 69,364 unique Code cases from January 1, 2020 through the
cutoff. Cases are classified from the dataset's published `DESCRIPTION` field.
The current EWS 311 cache is not the universe of all Austin 311 requests: it is
the 148,469 geocoded records with one of the three versioned "Request Code
Officer" intake descriptions.

The audit first matches the reported service-request number to that EWS cache.
It then checks every missing identifier against the unfiltered Austin 311
dataset. Residential linkage first uses the Code parcel ID and the City land-use
inventory to crosswalk to county appraisal IDs. Address and distance evidence
remain labeled as fallbacks and are not counted as exact parcel matches.

| Complaint category | Code cases | Cases with a 311 ID | Found in unfiltered 311 | Linked to EWS Code Officer extract | Exact residential parcel match | EWS extract and exact residential match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Structure condition | 19,000 | 15,024 | 15,018 | 14,735 | 14,717 | 11,248 |
| Property abatement | 28,903 | 26,992 | 26,981 | 26,979 | 19,198 | 18,068 |
| Land use | 18,717 | 16,738 | 16,703 | 13,745 | 11,000 | 7,788 |
| Work without permit | 2,744 | 2,494 | 2,492 | 2,489 | 1,865 | 1,731 |
| **All categories** | **69,364** | **61,248** | **61,194** | **57,948** | **46,780** | **38,835** |

Among cases with a reported 311 ID, 99.9% are found in the unfiltered 311
dataset. The apparent linkage loss is therefore mostly definitional: many cases
do not report a service-request number, and some originated under a 311 intake
type outside the current EWS selection. Only 54 unique reported identifiers are
not found in the full source.

## Structure-Condition Findings

Structure condition is the leading candidate for a tenant- or
habitability-oriented smoke signal:

- 14,735 cases, representing 14,679 unique requests, link to the current EWS
  Code Officer intake extract;
- 14,717 cases (77.5%) exactly match the promoted residential parcel surface;
- 11,248 cases, representing 11,226 unique requests, satisfy both tests;
- 11,428 exact residential matches (77.7%) occur in a multi-unit context; and
- conservative address or 25-meter fallbacks add only 61 cases beyond the
  exact parcel matches.

Of the 289 structure cases with a 311 ID but no match in the selected EWS
extract, 283 are found in the unfiltered source: 282 originated as private-
property graffiti requests and one as a billboard complaint. This supports
retaining the original Code Officer intake test when constructing a narrower
structure-condition measure rather than treating every case labeled
"structure condition" as equivalent.

A complaint is not a verified violation. The source reports an associated
violation-case number for 7,499 structure cases (39.5%), and the latest workflow
description is "No Violation(s) Found/Inspection Performed" for 7,887 (41.5%).
These fields can support a separate substantiation sensitivity but should not
be silently folded into the complaint count.

## Time Coverage

The public Code case series contains only isolated records before August 2023,
then begins sustained monthly coverage. The audit does not infer the cause of
this discontinuity. From August 1, 2023 forward, 97.1% of the EWS Code Officer
requests link to a Code case; across the two current 12-month windows the share
is 99.5%.

The dataset is therefore well suited to the current baseline windows (April 2,
2024 through April 1, 2026), but it should not be used as if it were a complete
2020-2023 history.

## Spatial Coverage

The strict candidate below counts unique Code Officer requests that become a
structure-condition case and exactly match a promoted residential parcel.

| Measure | 12-month window | Requests in 3,250 eligible hexes | Hexes with at least one | Share of eligible hexes |
| --- | --- | ---: | ---: | ---: |
| Strict structure-condition candidate | Previous | 3,963 | 1,165 | 35.8% |
| Strict structure-condition candidate | Latest | 4,294 | 1,229 | 37.8% |
| Current configured Code Officer intake | Previous | 19,082 | 2,460 | 75.7% |
| Current configured Code Officer intake | Latest | 20,448 | 2,524 | 77.7% |

The narrower measure is usable but sparse. A zero means that no qualifying
linked request was observed in the window; it does not establish that the hex
had no unsafe housing or tenant distress.

## Recommendation

Use the strict 11,226-request intersection as the leading structure-condition
sensitivity: unique selected Code Officer requests, a structure-condition Code
case, and an exact promoted residential parcel match. Assign requests spatially
from the original 311 coordinates and calculate the same per-unit, density, and
equal-window change components used by the current 311 index.

Compare that candidate with the current broad Code Officer feature and with a
verified-outcome variant. Do not replace the canonical Part 1 feature until the
resulting cluster composition, stability, and substantive interpretation have
been reviewed. Do not use the Code case categories for historical windows that
begin before August 2023.

## Reproduction

Run the optional review target:

```r
targets::tar_make(
  code_complaint_audit,
  script = "_targets_review.R",
  store = "_targets_review"
)
```

Generated tables are written under `output/311_code_complaint_*`; the raw API
snapshots are cached under `data/raw_311/`. Both locations are intentionally
ignored by Git.
