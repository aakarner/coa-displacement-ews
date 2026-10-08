# Ben White provisional denominator, October 7, 2026

Following explicit user authorization, the production inventory uses **170
provisional housing-unit equivalents** for South Austin Marketplace, 2101 E Ben
White Blvd, account 291453, cell 6260. This resolves the denominator decision
under an assumption; it does not establish a verified dwelling count or formal
SRO classification. See [decision 0018](../decisions/0018-ben-white-provisional-denominator.md).

## Evidence and interpretation

The [July 1, 2019 property listing](https://www.loopnet.com/Listing/2101-E-Ben-White-Blvd-Austin-TX/16495941/)
reports approximately **178 rooms**, temporary/transitional housing and shared
kitchen facilities. The [2018 Travis County reentry guide](https://www.austintexas.gov/sites/default/files/files/HR/TravisCountyReentryGuidebook2018.pdf)
reports **100 beds** and weekly single/double room rentals. Neither source
verifies 170 independent dwellings. The selected 170 is the user's authorized
analytical assumption, recorded separately from those source quantities in
[`config/ben_white_170_assumption.json`](../../config/ben_white_170_assumption.json).
The review configuration pins that record's checksum.

## Implementation

The reviewed promotion layer adds the omitted account once, preserving its raw
F1 commercial classification and zero reported units. Its chosen count has
status `provisional_assumption`, method `reviewed_assumed_project_total`, and
confidence `low`. A county reference point inside the verified property
footprint places its units and filings in the same cell. The filing ledgers
retain a `provisional_assumed_170_units` flag.

All other parcel counts and coordinates, all filing locations and source-case
exclusions remain unchanged. The historical ownership layer retains explicit
unknowns for both retrospective vintages; current owner information does not
backfill historical evidence.

The immutable incremental baseline and generated comparison tables are under
`output/ben_white_170_20261007/`. Rebuild and validation logs are under
`tmp/ben_white_170_20261007/`. These local output bundles are ignored by Git.

## Filing-rate result

The current rolling year contains **60 filings** at this property, producing
**35.29 filings per 100 assumed units** (60 / 170 × 100). This is a filing rate,
not a percentage of households evicted. All **249 property-flagged cases** in
the longer paired ledger retain their original assigned cell. No other parcel
count or location changes in this incremental update.

| Current measurement | Before | After |
| --- | ---: | ---: |
| Units on the analytical grid | 511,798.9 | 511,968.9 |
| Covered recent filings | 11,784 | 11,784 |
| Cells below 20 units with recent filings | 82 | 81 |
| Recent filings in cells below 20 units | 306 | 246 |
| Original audited filings still in cells below 20 units | 115 | 55 |
| Cluster-eligible cells | 2,676 | 2,676 |
| Recent filings in eligible cells | 10,606 | 10,606 |

Ben White passes the unit threshold but remains outside Parts 1 and 2 clustering
because historical ownership coverage is zero. Resolving the denominator does
not waive this separate completeness requirement. The remaining 55 low-unit
filings in the original audit are in cells 602 (1), 3261 (9), 3319 (18),
3767 (10), 3769 (14) and 6929 (3).

The new usable denominator slightly changes the eviction and 311 scaling
references. Among the 2,676 eligible Part 1 cells, five of seven feature indices
are exactly unchanged; eviction scores change by at most 0.647 points and 311
scores by at most 0.00214 points on their 0–100 scales. The dependent cluster
outputs are rebuilt to use these updated measures.

## Part 1 result

The complete 100-bootstrap/100-subsample refit retains seven recognizable
profiles and the same reviewed labels. **2,675 of 2,676 eligible cells keep
their profile**. Cell 5703 changes from Lower Measured Pressure to Corporate
Ownership + Vulnerability. The respective profile sizes change from 901 to
900 and 356 to 357; all other sizes remain unchanged.

Using the previous frozen classifier with the updated measures changes no
eligible cell's assignment. The single change therefore arises through
refitting the cluster definitions. No neighborhood's population-weighted
predominant profile changes. All 19 Part 1 lock checks pass.

## Part 2 result

The common comparison sample remains **2,504 cells**. After matching numeric
IDs to their substantive profiles, the 2025 baseline assignments are unchanged;
one fixed-definition 2026 assignment changes. The number of cells changing
profile between 2025 and 2026 increases from **914 to 915**. The standard
20-seed fits, 50 random holdouts and 20 spatial holdouts were rerun. Interpretation
hashes and the independent semantic-profile checks use the newly reviewed IDs.

Both paired and annual case-assignment comparisons report **zero changes**.
The original case-specific Bell, Asher and Monarch corrections remain applied.

## Rebuild and validation

The completed rebuild covers promotion, corporate aggregation, property
geography, paired and annual evictions, annual forward labels, ownership,
311 rates, ACS support, current measurement, both cluster analyses and maps.
No forecast model was trained or externally published.

The focused Ben White test verifies the single-account/170-unit change,
preserved raw attributes, low-confidence assumption, distinct source quantities,
unchanged case assignments, expected filing rate and unknown historical owners.
Reviewed-unit, prior case-review, residential-repair, ambiguity/annual-label,
ACS, ownership, 311, eviction and current-measurement checks also pass.

Independent Part 1 checks reproduce all assignments, scaling and labels, in
addition to its 19 lock checks. The Part 2 audit independently verifies 40
optimizer fits, 140 paired-date holdout fits and 1,472 current checksums. The
concern audit passes 278 checks; the figure audit verifies hashes and dimensions.
Both map sets were visually inspected. All 25 incremental baseline files retain
their original checksums, and `git diff --check` passes.
