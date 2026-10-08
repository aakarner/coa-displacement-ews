# Residential property review, batch 2 — October 7, 2026

**Status update:** this document preserves the completed batch-2 production
results. Subsequent Oak Ranch reviews support all 24 remaining original-audit
filing links, but their inventory integration remains staged. See the
[consolidated status report](residential-geography-closeout-2026-10.md) for
current applied, pending and deferred work.

This follows the Ben White update and investigates the remaining 55 filings in
six cells from the original low-denominator audit. The immutable pre-batch
snapshot is `output/residential_property_batch2/before/`; private case evidence
and the public source transcriptions are separately pinned in the review inputs.

## Supported repairs

| Property | Audited filings | Reviewed unit cell | Units | Repair |
| --- | ---: | ---: | ---: | --- |
| Bridge at Canyon Creek | 18 | 3321 | 332 | Correct the adjacent-parcel filing geocode; replace modeled units |
| Caliza | 9 | 3261 | 270 | Add omitted active WCAD account and reviewed boundary reference |
| Nexus at Goodnight Ranch | 3 | 6929 | 294 | Move the existing unit reference into the reviewed parcel/grid portion |
| Ocotillo | 1 | 596 | 308 | Correct the filing location using the exact property address; replace modeled units |

The [Austin Apartment Association directory](https://www.austinaptassoc.com/aisd-property-directory/bridge-at-canyon-creek)
identifies 332 units at 9009 FM 620. County account 498141 has the matching DBA
and address. The user-provided court record for J2-CV-25-004162 corroborates
the property name. Only that case was individually court-reviewed; the other
17 rely on the shared property address and independent county/operator evidence.

The [JLL April 19, 2023 announcement](https://www.jll.com/en-us/newsroom/sale-of-northwest-austin-multihousing-community-closes)
reports 270 apartments at 12638 Ridgeline. WCAD R500219 is active, has DBA
CALIZA and 341,389 living square feet. Nexus account 911866 reports 294 units.
Both are almost entirely inside full-purpose Austin, despite their original
reference points falling outside the fixed grid. See
[decision 0019](../decisions/0019-reviewed-boundary-property-references.md).

[Ardent's project list](https://ardent-residential.com/projects) reports 308
Ocotillo units completed in May 2017. Its
[project page](https://ardent-residential.com/projects/ocotillo) identifies
8000 US 290 Highway W. County account 859326 has the same base address and DBA
OCOTILLO, and its existing unit reference is in cell 596. The audited filing's
exported address includes apartment 3110 at that base address. This supports a
property-level correction without claiming independent apartment verification.
The current parcel-polygon source does not include account 859326; the manual
review relies on the account's existing reference and corroborated address,
not a claimed polygon match to that account.

## Mobile-home cells remain unresolved

The 24 filings in cells 3767 and 3769 concern the Oak Ranch area. County
parent accounts 464309 and 909849 have commercial F1 classifications and
manufactured-home-park use code 97MHP. Account 464309 has DBA OAK RANCH MH
COMMTY; 909849 reports 377 improvement units, which is not independently
verified as a count of dwellings rather than sites or another quantity.

An address-based diagnostic across the affected street names found 237 M1
manufactured-home accounts, absent from the promoted residential inventory.
All 237 have only two distinct coordinates, matching the park-level references
(217 at one reference and 20 at the other). This is a diagnostic subset, not
a verified total for either park. Adding those accounts at their existing
coordinates would not repair dwelling geography in the two audited cells.

The [operator identifies Oak Ranch as manufactured housing](https://robertscommunities.com/our-communities/oak-ranch/),
but this does not reconcile park sites with individual-home accounts. The next
repair needs current active-account/serial-number deduplication, a dated
park/site inventory and address-level locations independent of the filing
sample. Count each home once, exclude vacant pads and superseded records, and
avoid adding a parent total on top of individual homes. Retain the 24 filing
locations and their unresolved-property flags pending that work. No speculative
park denominator is adopted in this batch.

## Rebuild and outcomes

Production rebuild and outcome checks are recorded in
`tmp/residential_batch2_20261007/` and the batch audit output directory.

The paired ledger changes exactly 19 assigned cells: 18 Canyon Creek cases
move from 3319 to 3321, and the Ocotillo case moves from 602 to 596. Caliza's
nine and Nexus's three audited filings stay in their original cells and gain
aligned unit denominators. All 31 have usable denominators after this repair;
the original audit's remaining 24 cases are the two mobile-home cells. The
earlier 71 case reviews remain applied, for 102 explicit reviews in total.

| Current measurement | Before | After |
| --- | ---: | ---: |
| Units on the fixed analytical grid | 511,968.9 | 512,415.3 |
| Covered recent filings | 11,784 | 11,784 |
| Cells below 20 units with recent filings | 81 | 77 |
| Filings in cells below 20 units | 246 | 215 |
| Original audited filings still below 20 units | 55 | 24 |
| Part 1 eligible cells | 2,676 | 2,677 |

Caliza's historical ownership records are matched but classified as
`matched_evidence_insufficient` in both years. Its usable denominator does
not waive that separate completeness screen: it remains outside clustering.
Nexus passes the screens and enters Part 1. Other low-unit cells outside this
original audit remain in the broader diagnostic universe; this batch does not
claim that every remaining low denominator is resolved or erroneous.

The full Part 1 refit uses 100 gap bootstraps and 100 stability subsamples,
and all 19 lock checks pass. Match nominal IDs to substantive profiles before
comparison: of 2,676 previously eligible cells, two change profile. Canyon
Creek's cell 3321 changes from Lower Measured Pressure to Eviction-Filing
Concentration; cell 5703 changes from Corporate Ownership + Vulnerability to
Lower Measured Pressure. Newly eligible Nexus cell 6929 enters Corporate
Ownership + Vulnerability. No neighborhood population-weighted predominant
profile changes.

Using the previous frozen Part 1 classifier with the repaired measurements
changes only Canyon Creek cell 3321. The additional change at 5703 arises
through refitting the cluster definitions.

Part 2's common sample increases from 2,504 to 2,505 cells. Among the 2,504
shared cells, the substantive 2025 profiles are unchanged and one fixed-definition
2026 profile changes. The number changing profile between years increases from
915 to 916. The run repeats the full 20-seed/100-start fits and 50 random plus
20 spatial paired holdouts; the independent audit reproduces 40 optimizer fits,
140 paired-date holdout fits and 1,537 current checksum checks. The concern
audit passes 278 checks, and the figure checks validate hashes and dimensions.

Validation also passes for the reviewed points and drift rejection, exact
four-property totals, 102 case reviews, original exclusions, paired/annual
agreement, historical repairs, raw-source preservation, ownership, ACS, 311,
current measurement and target dependencies. The first-batch and Ben White
incremental tests replay their preserved pre-batch-2 endpoint; the new batch-2
test checks current production against that endpoint. The canonical grid and
the 20-unit threshold are unchanged. Both map sets were visually inspected.

Current review results are in
`output/residential_property_batch2/case_review_results.csv`; the original
queue and source-review proposals remain unchanged as dated evidence.
