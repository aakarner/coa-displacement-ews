# Hays current-window candidate address geography

Current JP4 source and refreshed results: [October canonical-report update](hays-eviction-jp4-update-2026-10.md). This document preserves the September snapshot.

Reviewed September 10, 2026. This is a conservative first pass on the
[collected party addresses](hays-eviction-address-collection-2026-09.md).

## Results

Of 231 cases in April 2, 2024–April 1, 2026, **152 candidate addresses locate
outside Austin's FULL-purpose boundary** and **79 cases remain unresolved**.
No accepted match falls inside the city. This is not evidence of zero Austin
evictions: the unresolved records and unverified premises remain outstanding,
as does the JP4 source-coverage issue.

| Method or unresolved category | Cases |
| --- | ---: |
| Normalized street and ZIP matched to local CAD parcel geometry; outside Austin | 85 |
| Normalized street and ZIP matched by Census, using interpolated street locations; outside Austin | 67 |
| Single-address candidates without an accepted location | 57 |
| Multiple distinct defendant addresses | 21 |
| Missing defendant address | 1 |
| Total | 231 |

Address cleanup removed non-address identifier labels from portal address
blocks. Two apparent conflicts differed only by those identifiers, leaving
209 single-address candidates (182 distinct strings), 21 conflicting-address
cases, and one missing address. No raw portal or source-PDF address was silently
replaced with a geocoder's suggestion. Geocoder suggestions live in separate
review columns.

## Acceptance rules

Local matching uses house number/street plus exact ZIP against CAD situs
addresses, restricted to real-property IDs. It normalizes punctuation, case,
common street suffixes, directions, and FM/RR/CR spacing, and removes unit labels
for parcel-level matching. All matching parcel IDs must have geometry. Their
entire union is compared with the existing April 29, 2026 city-boundary snapshot,
filtered to CITY OF AUSTIN / FULL. Missing geometry, crossing parcels, and
outside parcels within 100 m of the boundary remain review cases. County geometry
is preferred; the state March 2025 Hays archive fills missing IDs.

Austin's official locator returned 126 candidate responses for 69 distinct
unresolved street/city/ZIP queries. None were address-point or subaddress
matches. All were retained as unaccepted suggestions; some matched the wrong
street or only a street name. The results are not used for geographic assignment.

The Census batch submitted 109 unresolved address strings, using
`Public_AR_Current`. Acceptance requires a unique Match response, Exact or
Non_Exact match type, identical normalized street/house-number string, identical
ZIP, and coordinates within Texas. Only 56 distinct address strings passed,
covering 67 cases. The other 53 strings consist of 35 street discrepancies,
17 No_Match responses, and one Tie. These are held for manual review even when
the discrepancy might be a plausible abbreviation or typo.

Census coordinates are interpolated street locations, not parcel or apartment
coordinates. A point within 100 m of any city boundary is reserved for review.
Both the point/parcel method and raw matched address are retained in outputs.
No newly accepted geography is promoted to a verified eviction-premises address,
case-level final exclusion, or index score.

## Files and reproducibility

- `output/hays_eviction_current_window_geography_review.csv`: all 231 cases,
  with candidate geography and evidence.
- `output/hays_eviction_current_window_geography_followup.csv`: the 79 unresolved
  cases for further review.
- `output/hays_eviction_candidate_addresses_geocoded.csv`: unique candidate
  strings, accepted/rejected status, matching method, coordinates, and evidence.
- `output/hays_eviction_census_geocode_review.csv`: all Census response decisions.
- `output/hays_eviction_coa_geocode_candidates.csv`: unaccepted Austin responses.
- `output/hays_eviction_geography_review_summary.json`: counts and hashes.

Address-bearing API caches and request/response files are stored in the ignored
`data/raw_hays_evictions/geocode_cache` directory. Requests contained only
address strings and opaque address IDs, without party names, case numbers, or
government identifiers. Census metadata records the endpoint, benchmark,
retrieval date, and request/response hashes. County/state geometry sources and
the Austin boundary follow the earlier property-screen audit.

Run `Rscript scripts/data/hays_evictions_geocode_review.R` to rebuild the local
stage and reviewed Census results from the saved data. The separate
`hays_evictions_geocode_coa.py --network` fetcher fills missing Austin caches.
Census was requested by multipart POST to
`https://geocoding.geo.census.gov/geocoder/locations/addressbatch` with
`addressFile=census_request.csv` and `benchmark=Public_AR_Current`; it need not
be requested again to reproduce this pass.

The authoritative manual review log and source case fields are unchanged by
geocoding. All 231 case keys are retained exactly once. The 506/651 active/deferred
case partition and analytical coverage registry are unchanged.

## Outstanding evidence

Review the 57 unmatched single-address cases and the 21 conflicting-address
cases before deciding city membership or eviction premises. Standard address
abbreviations, missing street suffixes, and unit formatting explain some rejected
matches, but no such correction was assumed. The missing defendant address
for F24-089J5 still requires the court or a case document.

A separate [portal-reply draft](../hays-jp4-coverage-clarification-draft.md)
asks the county to explain the JP4 inactive-status-date filter and provide a
complete filing-date-selected export if needed. Send that clarification while
the remaining address review proceeds; the current data cannot establish complete
annual JP4 eviction filing counts.
