# Reviewed housing counts and property geography

Production review, October 7, 2026. Analytical cutoff: April 1, 2026.

**Subsequent update:** the user adopted a provisional 170-unit denominator for
Ben White under [decision 0018](../decisions/0018-ben-white-provisional-denominator.md).
The results below preserve the completed five-project review immediately before
that assumption was applied.

This update follows the 71 case-location corrections in
[decision 0016](../decisions/0016-reviewed-case-property-locations.md).
It implements [decision 0017](../decisions/0017-reviewed-unit-counts-and-geography.md)
in the canonical local production pipeline. Raw appraisal records, source
filings, geocodes and the unit estimation model remain unchanged. Reviewed
direct evidence overrides selected project estimates at promotion.

## Property decisions

| Property | Previous project units | Reviewed units | Geography/action |
| --- | ---: | ---: | --- |
| Bell Southpark Springs | 330.860831 | 400 | Existing account 878332, cell 6670 |
| Bridge at Monarch Bluffs | 361.292200 | 330 | Improvement account 975264 carries the total; land account 533185 remains zero, cell 6964 |
| Bridge at Asher | 449 | 452 | Existing account 513751, cell 6973; the cell also contains 12 other units |
| Villages at the Domain, south | 405.325991 | 412 | Move four existing accounts from misplaced cell 3486 to verified cell 3485 |
| Villages at the Domain, Building P | Omitted | 26 | Add verified account 774412 in northern cell 3459 |

The net change is **73.520979 units** in six cells. Fractional previous counts
are model estimates. The reviewed development counts are integers. The
neighboring 390-unit Residences at the Domain remains a separate project with
an unchanged count.

Bell's operator map identifies 400 distinct homes in the Springs footprint.
[Austin Energy's December 28, 2023 inspection fact sheet](https://services.austintexas.gov/edims/document.cfm?id=421606)
reports 330 rentable units for Monarch at the matching address (page 5).
Asher's 452 total is repeated in the
[2019 HACA acquisition packet](https://www.hacanet.org/wp-content/uploads/2019/04/20190418_HACA-Packet.pdf),
the [2019–2020 annual report](https://www.hacanet.org/wp-content/uploads/2020/09/HACA-AnnualReport_2019-20_FINAL_PAGES.pdf)
and the [apartment association listing](https://www.austinaptassoc.com/aisd-property-directory/bridge-at-asher).
The acquisition packet's bedroom breakdown is internally inconsistent; the
review uses the repeated explicit project total, not a sum of that breakdown.

The Domain operator map identifies 412 southern and 26 northern homes. The
[City's October 23, 2009 compliance report](https://www.austintexas.gov/sites/default/files/files/Redevelopment/domain-report2009.pdf)
independently identifies Building P as 26 apartments (page 6). County DBA and
ownership attributes identify its missing account. Its original county link
to unmapped account 866401 is retained as unresolved; the physical association
to footprint 737155 rests on independently reviewed evidence. The operator's
438-home total differs by two from the City's later 436-unit inventory; this
discrepancy is retained, not averaged away. The older report's 390-plus-26
description is not a count of the entire present-day Villages development.

Each Domain reference is an actual mapped home nearest the mean projected home
location, checked inside its reviewed footprint and expected cell. A mean point
alone can fall in a courtyard or polygon hole. Existing southern account shares
are preserved when allocating their replacement total.

## Filings and coverage

All prior 71 Bell/Asher/Monarch case reviews remain applied. No source exclusion,
ambiguous-case decision or accepted case total changes. The unresolved apartment
number discrepancy in case J3-EV-25-001704 remains recorded.

Adding Building P reveals that county polygon 737155 contains more than one
residential project. Its former blanket match to the neighboring 390-unit
project is withdrawn. **36 cases in the paired ledger** return to their original
accepted geocodes: eight to cell 3455 and 28 to cell 3459. The longer annual
ledger contains one additional affected 2020 filing, for **37 cases** total.
Cases already geocoded to cell 3454 keep that cell but lose the unsupported
single-project attribution. No nearest-property guess replaces these links.

Four of these changes fall in the current rolling year: three filings return
to zero-unit cell 3455 and one to the newly recovered 26-unit northern cell.
That northern cell remains ineligible for clustering because the pinned
historical ownership extract has no record for the recovered account.
Historical ownership is explicitly unknown in 2024 and 2025; the current owner
is not backfilled into those vintages.

| Current measurement | Before this update | After |
| --- | ---: | ---: |
| Units on the analytical grid | 511,725.4 | 511,798.9 |
| Covered recent filings | 11,784 | 11,784 |
| Cluster-eligible cells | 2,675 | 2,676 |
| Recent filings in eligible cells | 10,579 | 10,606 |
| Cells below 20 units with recent filings | 82 | 82 |
| Recent filings in cells below 20 units | 334 | 306 |
| Unresolved filings from the original 217-case audit | 146 | 115 |

The Domain repair gives all 31 southern audited filings a supported denominator
of 412 units: 7.52 filings per 100 units. The net reduction in low-unit filings
is 28 because the mixed-parcel correction returns three other recent filings
to a zero-unit cell. The count of affected low-unit cells stays 82: one cell
is repaired and another becomes visible.

The 115 remaining original audited filings occupy seven cells: 60 at Ben White,
18 in cell 3319, 14 in 3769, ten in 3767, nine in 3261, three in 6929 and one in
602. The 306 total low-unit filings include additional cases outside that
original audit cohort.

## Substantive cluster results

Part 1 retains seven recognizable profiles and their existing qualitative
concern tiers. Of the 2,675 previously eligible cells, **2,667 (99.7%) keep their
profile** and eight change after the full refit. The southern Domain cell joins
the eviction-filing concentration profile; no previously eligible cell is lost.

| Part 1 profile | Before | After |
| --- | ---: | ---: |
| Lower measured pressure | 903 | 901 |
| Higher rents / lower vulnerability | 685 | 687 |
| Nearby amenity activity | 137 | 137 |
| Corporate ownership + vulnerability | 360 | 356 |
| Selected 311 + vulnerability | 244 | 244 |
| Demolition-permit concentration | 236 | 240 |
| Eviction-filing concentration | 110 | 111 |

No neighborhood's population-weighted predominant profile changes. Assigning
the updated features with the *previous frozen classifier* changes none of the
previously eligible cells, so the eight changes above arise through refitting
the definitions rather than crossing the old cluster boundaries. The final
seven-feature solution has mean silhouette 0.222 and mean subsample adjusted
Rand agreement 0.945, using the full 100-replicate stability run and 100 gap
bootstraps. Numeric IDs were reviewed and remapped to the measured profiles.

Part 2's common sample increases from **2,503 to 2,504 cells**. Among the 2,503
cells common to both runs, 16 have a different substantive 2025 profile and 15
have a different fixed-2026 profile after the corrected refit. Fixed-baseline
2025-to-2026 transitions increase from 906 to 914 cells. The comparison uses
the reviewed profile names rather than treating nominal cluster IDs as stable.
Its standard 20-seed fits, 50 random holdouts and 20 spatial holdouts were rerun.

At the reviewed properties, Springs' current cell filing rate falls from 16.02
to 13.25 per 100 units; Monarch's rises from 4.98 to 5.45; Asher's cell rate
changes from 2.82 to 2.80. These corrections improve the local denominators
without substantially changing the citywide typology.

## Ben White remains an explicit unresolved denominator

The 60 recent filings at 2101 E Ben White Blvd remain included. The shared
crosswalk and case ledgers flag the property as a transitional residential
facility whose independent-housing-unit count is unverified. The paired ledger
contains 249 flagged cases across its longer source period.

The historical reentry guide describes 100 beds; the marketing evidence
describes approximately 178 rooms and shared facilities. Neither establishes
an April 2026 stock of independent housing units. County account 291453 has
commercial classification and no usable housing count. A match in the local
demolition-permit extract was a contractor/applicant mailing address for work
at 2607 Wilson Street, not evidence about the Ben White property.

The next useful evidence is an occupancy record or operating record that
distinguishes independent dwellings from beds/rooms and establishes capacity
for the relevant period. Until then, retain filings and the uncertainty flag,
assign no invented denominator, and suppress no cells. A separate facility
rate would require an explicit measurement decision about an appropriate
population at risk; it cannot be substituted into the ordinary housing-unit rate.

## Rebuild and validation

The rebuild refreshes the canonical unit surface, corporate aggregation,
property crosswalk, annual and paired filing outputs, historical ownership,
311 rates, ACS residential allocation and rent crosswalks, current measurement,
cluster products and annual forward labels. Both retrospective Part 2 vintages
use the same reviewed inventory. No forecast model is trained or deployed.

The targets graph explicitly propagates the new reviewed evidence through
unit promotion and the unit-dependent paired measures. The production layer
rejects changed evidence, project membership, overlapping future supplements,
changed original geometry and references outside the reviewed footprint/cell.

Local evidence is under `data/reviewed_unit_properties/batch1_20261007/`, pinned
by `config/residential_unit_property_reviews.json`. Some old PDFs were readable
through indexed web text but not downloadable from their current URLs; their
local records are explicitly labeled review transcriptions. Only successfully
downloaded PDFs are represented as archived PDFs. The evidence bundle and
case-level diagnostics are excluded from Git.

Before/after products are preserved under `output/reviewed_units_20261007/`.
`scripts/audits/reviewed_unit_properties.R` generates the comparisons;
`tests/test_reviewed_unit_properties.R` and `tests/test_reviewed_unit_outputs.R`
test drift rejection, exact property/count changes, original source hashes,
case conservation and retained uncertainty. The prior case-review regression
also verifies all 71 assignments in both annual and paired products.

Completed validation: all 19 Part 1 lock checks; reviewed-unit and prior
case-review regressions; residential-repair, ambiguity and annual-label checks;
independent current-measurement, ACS, 311, ownership and eviction audits;
Part 2 matrix, optimizer/holdout, concern and figure audits; and the acyclic
dependency/invalidation test. The Part 2 concern audit passed 278 checks, and
the cluster audit verified 1,433 current checksums. Both regenerated map sets
were visually inspected. Raw source hashes and the pre-review snapshots remain
preserved. Changes are local; no commit, push or external publication was made.
