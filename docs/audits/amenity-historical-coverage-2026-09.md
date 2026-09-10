# Historical Amenity Coverage Audit: September 2026

- **Status:** Feasibility assessment; no production input or method changed
- **Research and validation date:** September 6, 2026
- **Question:** Can the amenity-change feature be reconstructed for a
  one-year-back Part 2 proof of concept using a 2024 appraisal vintage?

**Implementation follow-up (September 8, 2026):** The user selected a simple
retrospective one-year vintage comparison, without an initial exact-as-known
replay or extra sensitivity branches. Paired amenity processing is now
documented in [historical amenity methods](../methods/historical-amenities.md)
and the [execution audit](amenity-historical-processing-2026-09.md). The
feasibility findings below remain a dated research record, not the final
production counts or an outstanding requirement to run both source variants.

**Count correction:** The preliminary eligible totals below are reproduced when
the production home-business exclusion is omitted. The shared executable
classifier gives 1,278 archive events in the historical 36-month window and
459 calendar-2024 events before exact-opening alias deduplication, rather than
1,292 and 461. The processing audit reconciles the source-ID and final counts.

## Conclusion

The historical amenity feature is feasible for the proof of concept. Two
Internet Archive captures preserve downloadable copies of the Texas
Comptroller export that supplies the opening events. The February 2025 capture
appears sufficient for the older portion of the proposed historical window,
which the live rolling source no longer preserves reliably.

The remaining limitation is temporal provenance, not a general absence of
data. No complete export captured on or after April 1, 2025 was found in the
known Socrata URL patterns. Records added between the early-February source
vintage and April 1 can be reconstructed from the live dataset by event and
permit dates, but that supplement may also contain later corrections,
backfills, or reclassifications. It therefore supports a retrospective
event-date reconstruction, not an exact representation of information
published on April 1, 2025.

The recommended proof of concept is to build both an archive-first version and
an archive-plus-live retrospective version, then measure whether the source
choice materially changes the hex feature or cluster results. An exact agency
extract is worth requesting only if that sensitivity is consequential or if a
strict information-available-at-the-time interpretation is required.

## Current Feature Contract

The amenity feature measures selected **openings**, not the stock of amenities
or overall commercial density. `first_sale_date` is used as an opening proxy;
it is not independent confirmation of the date a business opened its doors.

The versioned taxonomy in
[`config/amenity_categories.csv`](../../config/amenity_categories.csv) admits:

- NAICS 722410 drinking places;
- NAICS 722511 full-service restaurants; and
- NAICS 722515 establishments whose names pass the configured cafe filter.

For each category, the pipeline compares two equal 18-month opening windows.
It calculates linearly declining exposure from each hex centroid to events
within 800 meters, combines the recent opening level with positive change from
the preceding window, and gives the three category scores equal weight.
Mixed-beverage and food-inspection matches are corroboration flags only; they
do not create additional opening events.

The executable feature definition remains in
[`scripts/audits/amenity_sources.R`](../../scripts/audits/amenity_sources.R) and
[`scripts/data/amenities.R`](../../scripts/data/amenities.R). The current
method is summarized in
[`docs/methods/cluster-methodology.md`](../methods/cluster-methodology.md).

## Time Alignment

The phrase "2024 amenity data" is potentially misleading. If the historical
cluster uses the 2024 appraisal ownership vintage as the one-year-back analog
to the current 2025 appraisal vintage, its amenity cutoff should move back one
year with the overall analysis cutoff. It should not be limited to openings
during calendar 2024.

| Vintage | Analysis cutoff | Appraisal vintage | Previous amenity window | Recent amenity window |
| --- | --- | --- | --- | --- |
| Current `T` | April 1, 2026 | 2025 | April 2, 2023-October 1, 2024 | October 2, 2024-April 1, 2026 |
| Proposed `T-1` | April 1, 2025 | 2024 | April 2, 2022-October 1, 2023 | October 2, 2023-April 1, 2025 |

This mirrors the configured April 1 cutoff and equal 18-month windows in
[`R/analysis_config.R`](../../R/analysis_config.R). A separate December 31,
2024 feature could be constructed, but it would change the comparison design
rather than provide the direct one-year-back vintage.

## Primary Source and Retention Limit

The classification backbone is the Texas Comptroller's
[All Permitted Sales Tax Locations and Local Sales Tax Responsibility](https://data.texas.gov/dataset/All-Permitted-Sales-Tax-Locations-and-Local-Sales-/3kx8-uryv)
dataset, Socrata ID `3kx8-uryv`. Relevant fields include stable taxpayer and
location numbers, name, address, county, NAICS, permit date, first-sale date,
and out-of-business date.

The Comptroller describes the live file as including locations currently
active or active at any time during the preceding 48 months. It is therefore a
rolling operational file, not a complete historical archive. The current
project extract has no closed target record earlier than May 4, 2022. A
current-file-only reconstruction cannot demonstrate that short-lived locations
in the older historical window are complete.

## Validated Archived Exports

These are archived copies of the official export, not an archive maintained or
certified by the Comptroller. The capture timestamp and the source's observed
`Last-Modified` timestamp are distinct and neither, by itself, guarantees that
every row was known on that date.

| Archive capture | Observed source modification | Bytes | Data rows | Columns | SHA-256 | Intended use |
| --- | --- | ---: | ---: | ---: | --- | --- |
| [December 17, 2024, 10:41 UTC](https://web.archive.org/web/20241217104100id_/https://data.texas.gov/views/3kx8-uryv/rows.csv?accessType=DOWNLOAD) | December 16, 2024, 16:57 UTC | 304,402,386 | 1,391,249 | 33 | `4cfad9ec4551af9be4b3fd3802d610457f62812b2dabdd4c3f0dea3cb90bfb51` | Contemporaneous audit reference for 2024 |
| [February 8, 2025, 11:37 UTC](https://web.archive.org/web/20250208113718id_/https://data.texas.gov/views/3kx8-uryv/rows.csv) | February 3, 2025, 19:45 UTC | 300,879,159 | 1,375,129 | 33 | `930aacba3293c9c626f1d9bba494b5c7f5f1075177afc6881f31873b412c5b9f` | Preferred base for the proposed April 1, 2025 reconstruction |

Both exports were downloaded and parsed successfully. The December capture
contains 5,792 raw records in Hays, Travis, and Williamson Counties for NAICS
722410, 722511, and 722515. The February capture contains 5,808 such records,
with no missing business name, address, or first-sale date in that inspected
subset.

No full export after April 1, 2025 was found while searching known `/views`,
`/api/views`, `/api/v3/views`, and `/resource` variants. Absence from those
captures does not prove that no other copy exists.

## Coverage Checks

### Calendar 2024

The February capture contains 559 raw three-county records in the three core
NAICS categories with a 2024 first-sale date. Applying the current project's
name, home-business, institutional, and category rules leaves 461 eligible
opening candidates:

| County | Drinking places | Full-service restaurants | Cafes | Total |
| --- | ---: | ---: | ---: | ---: |
| Hays | 7 | 48 | 5 | 60 |
| Travis | 46 | 206 | 19 | 271 |
| Williamson | 15 | 102 | 13 | 130 |
| **Total** | **68** | **356** | **37** | **461** |

These are county-level candidates before geocoding and the final study-area
buffer. They are not a count of City of Austin businesses or events ultimately
contributing to a hex.

### Proposed 36-Month `T-1` Window

Applying the same eligibility rules to April 2, 2022-April 1, 2025 shows that
the archive and the current source are similar but not interchangeable:

| Comparison | Eligible event IDs |
| --- | ---: |
| February 2025 archived export | 1,292 |
| Live API extract retrieved September 6, 2026 | 1,300 |
| Present in both | 1,222 |
| Archive only | 70 |
| Current only | 78 |

The mutable live comparison extract contained 1,688 raw rows after restricting
the API request to the three counties, the three core NAICS codes, and
first-sale dates from April 2, 2022 through April 1, 2025. Its SHA-256 checksum
at retrieval was
`a51dcd791df0485a7e670474bec854a299378e4d3ba414f5d7a4bffb32261efa`.
The 1,300 count above is after applying the additional project eligibility
rules.

Of the 78 current-only records, 45 have first-sale dates after the February
source modification and 33 have earlier dates. The latter demonstrate that
later backfills or revisions can affect even earlier event periods. Of the 70
archive-only records, eight were recorded as closed by April 1, 2025. Possible
record removal, identifier changes, and source-field revisions require
investigation before combining versions.

These counts document modest but real source-version differences; they do not
prove that either version is complete. A simple unrestricted union would also
be inappropriate because changed records could be double counted.

## Corroborating Sources

### Mixed-Beverage Reports

The Comptroller's
[Mixed Beverage Gross Receipts](https://data.texas.gov/dataset/Mixed-Beverage-Gross-Receipts/naix-2893)
dataset, Socrata ID `naix-2893`, is cumulative back to 2007. The live API has
all 12 reporting months in 2024, with 25,082 Hays, Travis, and Williamson rows
covering 2,343 distinct permits. Date filtering is sufficient for the
corroboration role; these records do not replace the sales-tax classification
backbone.

### Austin Food Inspections

The live Austin
[Food Establishment Inspection Scores](https://data.austintexas.gov/api/views/ecmv-9xxi)
dataset, Socrata ID `ecmv-9xxi`, retains only the most recent three years.
Historical copies nevertheless exist:

- an [October 1, 2024 capture](https://web.archive.org/web/20241001225920id_/https://data.austintexas.gov/api/views/ecmv-9xxi/rows.csv?accessType=DOWNLOAD)
  contains 23,533 inspections for 6,670 facilities, dated September 24,
  2021-August 30, 2024;
- a [January 2, 2025 capture](https://web.archive.org/web/20250102100031id_/https://data.austintexas.gov/api/views/ecmv-9xxi/rows.csv?accessType=DOWNLOAD)
  contains 23,225 inspections for 6,663 facilities and extends through
  December 7, 2024; and
- a [December 10, 2023 capture](https://web.archive.org/web/20231210192044id_/https://data.austintexas.gov/api/views/ecmv-9xxi/rows.csv?accessType=DOWNLOAD)
  extends the available inspection history back to December 5, 2020.

First inspection is not treated as an opening date. These records remain local
corroboration and do not control event eligibility.

## Geocoding Evidence

The current amenity run geocoded 1,452 of 1,456 eligible events, or about 99.7
percent. Census matched 1,244 and the ArcGIS fallback matched another 208;
four remained unresolved. This demonstrates that the existing cascade is
effective, but it does not guarantee the same result for the additional
historical addresses. Historical geocoding coverage and the spatial pattern of
unmatched records must be reported separately.

## Current Implementation Risks

The current scripts were written for one active vintage and should not yet be
used to certify a historical run:

- the primary source cache has a fixed filename, so a refresh can overwrite a
  prior extract and a no-refresh run can silently reuse the wrong vintage;
- `source_download_date` records the processing date rather than the source or
  archive vintage;
- output artifacts are not namespaced by analysis cutoff and source snapshot;
- `amenity_window_complete` is set to `TRUE` without testing the source's
  retained-history boundary; and
- there is no machine-readable manifest binding a run to archive URL, source
  modification timestamp, dimensions, checksum, taxonomy version, and code
  commit.

These limitations do not affect the documented current April 2026 result, but
they could make two historical runs appear comparable when they consumed
different source vintages.

## Recommended Proof-of-Concept Plan

1. Define the historical vintage as April 1, 2025 and pair it with the 2024
   appraisal ownership snapshot. Preserve April 1, 2026 and the 2025 appraisal
   snapshot as the current comparison.
2. Store the December and February source files outside Git as immutable raw
   inputs. Add a versioned manifest containing the URLs, timestamps, byte and
   row counts, checksums, filter scope, taxonomy version, and retrieval date.
3. Change amenity cache and output paths to include the analysis cutoff and
   source-vintage identifier. Replace the unconditional completeness flag with
   an explicit coverage contract.
4. Construct an **archive-first** event table from the February export and an
   **archive-plus-live retrospective** table that reconciles stable taxpayer
   and location IDs and admits only records with permit and first-sale dates no
   later than April 1, 2025. Do not combine versions with an unrestricted row
   union.
5. Apply the current taxonomy, exclusions, 18-month windows, geocoding cascade,
   study-area rules, and 800-meter exposure calculation identically to both
   variants.
6. Compare event counts, archive/current ID dispositions, geocoding coverage,
   hex-level score distributions, rank correlations, and maps of meaningful
   differences. Carry both variants into the initial cluster sensitivity.
7. Assemble the remaining `T-1` features with their own point-in-time source
   contracts. Do not mix a historical ownership and amenity surface with
   otherwise current feature values.
8. Evaluate two distinct temporal questions: independently refit and
   label-match `T-1` and `T` clusters to assess structural stability, and apply
   the frozen `T-1` transform and centroids to `T` to measure individual hex
   movement relative to fixed definitions.

## Decision Gate

Proceed with the retrospective reconstruction if the archive-first and
archive-plus-live variants produce substantively similar amenity surfaces and
cluster conclusions. If they do not, either request the complete Comptroller
export published on or nearest April 1, 2025 or move the proof-of-concept cutoff
to a date supported by a contemporaneous archived export. The chosen temporal
semantics and reconstruction rule would then warrant a separate analytical
decision record and updates to the current methods documentation.

## Reproducibility Requirements

A production implementation should record:

- source and replay URLs, dataset IDs, archive-capture timestamp, observed
  source-modification timestamp, and retrieval timestamp;
- raw byte, row, and column counts plus SHA-256 checksum;
- exact county, NAICS, date, and eligibility filters;
- taxonomy checksum and code commit;
- stable-ID reconciliation dispositions across source versions;
- event-date, category/window, geocoding, and spatial-coverage QA; and
- an explicit label distinguishing contemporaneous source evidence from later
  retrospective reconstruction.

The large validation downloads used for this investigation were kept outside
the repository. The archive URLs and checksums above are the durable retrieval
contract; the files should be copied to approved project storage before an
implemented historical run depends on them.
