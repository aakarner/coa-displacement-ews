# Residential geography and eviction-denominator repair: status report


**October 7, 2026. This is a closeout of the intensive review, not a claim that
every location or coverage problem is fixed.** Recent-filing figures cover
**April 2, 2025–April 1, 2026**, not calendar 2026. The last completed production
rebuild is the property batch-2 run. Ownership reconciliation and Oak Ranch
changes described below have not yet reached production outputs.

The work addresses review **#3 (ambiguity suppressing entire cells), #7
(residential denominators and property geography), and part of #26
(manufactured-home omissions)**. The Williamson ownership reconciliation repair
is a related eligibility correction. Oak Ranch does not constitute a citywide
manufactured-housing inventory repair.

## Answer to “is it all fixed?”

- The **whole-cell ambiguity veto is fixed in production**. Ambiguous cases
  remain flagged and unassigned; their candidate cells remain usable.
- Of **1,219 filings in the original 41-cell audit**, **1,195 (98.0%)** now have
  at least 20 units in their assigned production cell. This includes **60 Ben
  White filings supported by a provisional assumption**, not a verified
  dwelling count. Most other unit counts also remain modeled unless replaced
  by documented project totals.
- The remaining **24 original-audit filings** now have supported links to
  **17 Oak Ranch homes**, with no change to their assigned cells. Their home
  inventory and links are **staged, not production-integrated**.
- Across the broader study, **215 filings in 77 cells** remain below the
  20-unit floor in current production: those 24 plus **191 other filings**.
  These residuals have not all been adjudicated as either errors or genuinely
  small residential denominators.
- Neither usable unit counts nor a property match guarantees cluster
  eligibility. Rent, ownership and other completeness requirements still apply.

## Problems found and how they were repaired

| Problem | Implemented response | Status and measured effect |
| --- | --- | --- |
| One ambiguous filing suppressed every candidate cell, including its valid filings | Keep the case unassigned and retain its candidate-location flags; remove the cell veto from current/paired counts and annual outcomes | Applied: restored 103 Part 1 cells containing 2,619 recent accepted filings; restored 139 Part 2 paired cells |
| Williamson residential properties were omitted by restrictive classification/text selection | Add an evidence-based residential supplement using active accounts, positive living area, City housing use or documented review; reconcile reference/active accounts and retain commercial negative controls | Applied: 35 accounts recovered, 34 with positive operational units; initial net gridwide gain about 9,468 units |
| Filing geocodes and the unit reference for the same development landed in different cells | Use a shared verified property-to-unit-reference crosswalk in paired and annual counts; preserve original coordinates and all case exclusions | Applied: the initial combined repair gave 1,002 of the 1,219 audited filings denominator support; citywide low-unit filings fell from 1,614 to 405 |
| Geocodes missed parcel boundaries or confused nearby developments/phases | Apply source-pinned, case-specific property reviews using county records, operator maps, exact addresses and available court plaintiff/DBA evidence | Applied: 71 Bell/Asher/Monarch case relocations, followed by 19 Canyon Creek/Ocotillo relocations; no nearest-property guess |
| Existing unit estimates, omitted accounts or misplaced unit references distorted project denominators | Replace supported project totals, add documented missing accounts and use reviewed references inside the property/grid | Applied: Domain, Caliza, Nexus and the other projects listed below; remove unsupported single-project links on a mixed Domain parcel |
| Ben White residential facility had no ordinary dwelling-unit count | Adopt the user's explicit 170-unit-equivalent assumption, separately label its low confidence and retain source quantities | Applied: 60 filings gain a provisional denominator; no claim of verified 170 dwellings or formal SRO status |
| Matching Williamson ownership evidence was rejected because mailing fields were clipped or contained non-delivery headings | Reconcile supported heading variants and recover a truncated GIS delivery line from its corroborated full address in the same year | Code fixed; 36 targeted tests passed. Twenty of 202 flagged parcel-years recover usable evidence. Production rebuild deferred |
| Oak Ranch M1 home accounts were omitted, and county points located homes at park-level references | Review the complete 2025 subdivision account inventory, count each active home once, and use independent City address points | Staged: 793 accepted in-grid home locations, comprising 791 additions and two existing-unit relocations; 24 filing links ready |

The shared property reference is an explicit aggregation convention. It aligns
verified numerator and denominator geography; it does not establish the exact
building or apartment for every filing or distribute every development's units
across its footprint. Where evidence is insufficient, accepted original
geocodes remain and the property link is flagged.

The residential supplement is implemented in this EWS pipeline. This does not
claim that the upstream sibling repository's source selection was rewritten
or that every omitted residential account citywide has been discovered.

## Reviewed properties

These counts describe the audited cohort; they are not each property's entire
filing history. The 217 follow-up filings reconcile as
71 + 31 + 60 + 31 + 24.

| Property | Audited filings | Resolution |
| --- | ---: | --- |
| Bell Southpark | 54 | 53 Springs filings moved to cell 6670; one Meadows filing to 6974. Springs total corrected from about 330.9 to 400. One case retains conflicting apartment numbers, while its property assignment is supported. |
| Bridge at Asher | 6 | Moved to cell 6973; project total 449 → 452. |
| Bridge at Monarch Bluffs | 11 | Moved to cell 6964; total about 361.3 → 330. Residential improvement account carries the units; land account remains zero. |
| Villages at the Domain | 31 | Southern units moved from 3486 to 3485 and corrected from about 405.3 to 412; added 26 Building P units in 3459. All 31 southern audited filings gain denominator support. |
| South Austin Marketplace / Ben White | 60 | Cell 6260 receives 170 provisional unit equivalents; filings stay put. The resulting rate is 35.29 filings per 100 assumed units. |
| Bridge at Canyon Creek | 18 | Moved from adjacent-parcel cell 3319 to 3321; reviewed total 332. One court record corroborates the property; the other cases rely on common-address and property evidence. |
| Caliza | 9 | Added omitted active account with 270 units at reviewed cell 3261; filings stay put. Ownership reconciliation is a separate staged correction. |
| Nexus at Goodnight Ranch | 3 | Moved its existing 294-unit reference into reviewed cell 6929; filings stay put. |
| Ocotillo | 1 | Moved from cell 602 to 596; reviewed total 308. Property-level evidence does not claim independent apartment verification. |
| Oak Ranch | 24 | Staged links for ten filings in 3767 and fourteen in 3769; all remain in their original cells. Accepted staged home counts in those cells are 194 and 150. |

The Domain correction also **withdrew unsupported links** on a mixed northern
parcel: 36 cases in the longer paired ledger (37 in the annual ledger) reverted
to accepted original geocodes. Three recent filings consequently became visible
in a zero-unit cell. Thus resolving 31 southern filings reduced the broader
low-unit total by 28, not 31. The 438-home reviewed Domain total retains its
documented discrepancy with a separate 436-unit source; these are not averaged.

Ben White's sources report approximately **178 rooms** (2019 listing) and
**100 beds** (2018 reentry guide). Neither verifies 170 dwellings. The assumption,
source links and retained flags are documented in the
[Ben White audit](ben-white-provisional-units-2026-10.md).

## Measured production effects

All completed production stages through batch 2 retain **11,784 accepted recent filings** in covered study cells.
The repairs improve their denominators and eligibility rather than deleting
filings to improve rates. This conserved total excludes unresolved or
out-of-study source cases discussed below.

| Completed stage | Original audited filings still below 20 units | All low-unit filings | Low-unit cells with filings | Part 1 eligible cells |
| --- | ---: | ---: | ---: | ---: |
| After ambiguity policy, before residential repair | 1,219 | 1,614 | 170 | 2,660 |
| Initial inventory + shared property geography | 217 | 405 | 84 | 2,675 |
| Bell / Asher / Monarch case links | 146 | 334 | 82 | 2,675 |
| Reviewed unit totals + Domain geography | 115 | 306 | 82 | 2,676 |
| Ben White provisional denominator | 55 | 246 | 81 | 2,676 |
| Canyon Creek / Caliza / Nexus / Ocotillo | 24 | 215 | 77 | 2,677 |

The broader low-unit burden falls **1,614 → 215 (86.7%)**. Grid units rise
**502,256.9 → 512,415.3**, a net increase of about **10,158.4** after the
production residential repairs, including modeled units and the Ben White
assumption. Filings in Part 1 eligible cells rise **9,528 → 10,628** over those
stages.

Including the earlier ambiguity-policy change, Part 1 eligibility rises
**2,557 → 2,677**, Part 2 paired eligibility **2,351 → 2,505**, and recent
filings represented in Part 1 eligible cells **6,909 → 10,628**. These are
same-period data/measurement corrections, not evidence of a temporal increase
in displacement.

The largest substantive refit followed the initial inventory/geography repair:
145 of 2,659 commonly eligible cells changed profile; the eviction-concentration
group grew from 85 to 111 cells. Its mean filing rate fell from 27.9 to 14.7 per
100 units, comparing different refitted groups. Eight neighborhood plurality
labels changed at that stage. Later property batches had smaller effects; the
last batch changed two existing Part 1 profiles and added Nexus, with no
neighborhood plurality changes. Incremental profile-change counts must not be
summed as a cumulative number of unique affected cells. Seven recognizable
profiles remain. No Oak Ranch or ownership-fix cluster effect has yet been run.

## Remaining coverage and interpretation limits

**Low-unit cells:** the 215 current-production filings comprise the 24 staged
Oak Ranch cases and **191 other cases in 75 cells**. Of those other cases,
96 are in 38 zero-unit cells; 95 are in 37 cells with positive denominators
below 20. A low denominator is a review flag, not proof of a location error.
We have not eliminated all citywide inventory omissions or geocode problems.

**Other cluster exclusions:** **941 accepted recent filings** have at least
20 units in their cells but remain outside Part 1 because other requirements
fail. Adding the 215 low-unit filings gives 1,156 accepted filings outside
the current clustered sample. For example, Ben White and Domain Building P
still lack usable ownership evidence. The Williamson fixes address false
mailing-address mismatches at Caliza and other properties; they do not remove
the ownership completeness requirement. Part 1 needs 2025 evidence; a missing
2024 observation alone is not its exclusion rule. Cell 2326's diagnostic
ownership-unit coverage rises from 19.4% to 99.7%, pending rebuild.

**Property linkage:** 10,812 of the 11,784 assigned recent filings currently
have verified property links (2,508 reassigned, 8,304 already in the unit cell).
The other **972 keep accepted original geocodes without a verified property
link**. They are not all unlocated, below the unit floor, or excluded from
clustering. This count overlaps other categories and must not be added to them.

**Cases with no assigned study cell:** the current source ledger, restricted
to cases with a filing date in the recent window, still includes:

| Assignment disposition | Cases | Interpretation |
| --- | ---: | --- |
| Multiple candidate cells or mixed inside/outside evidence | 45 | Still flagged and unassigned; do not suppress candidate cells |
| No reliable location | 497 | Location unresolved; cannot assume all belong inside Austin |
| Reliable geocode outside the grid | 5,300 | Outside analytical grid; not automatically geocoding errors or in-city omissions |
| Reliable geocode outside study geography | 610 | Outside the defined study scope |

These are source-wide counts, not additional known Austin evictions omitted
from the 11,784 total. Missing/conflicting filing dates cannot all be assigned
to this recent window. The audit did not resolve every source case.

**Oak Ranch:** 832 active accounts were reviewed from the 2025 county snapshot.
The 793 accepted locations leave **15 unresolved home locations and 24 homes
outside the fixed grid**. Further investigation of these 39 is deferred by
the user. The 24 outside-grid homes are not the same quantity as the 24 audited
filings. County parent units are not added on top of individual homes, serial
sections are not counted as separate homes, and park ownership is not copied
onto residents' homes. City points retrieved in 2026 corroborate locations;
they do not prove a complete 2026 home stock. See the
[Oak Ranch evidence and decision record](../../data/reviewed_manufactured_housing/oak_ranch_20261007/README.md).

## Pending implementation and verification

### Follow-up now implemented, pending the combined rebuild

The [follow-up audit](residential-followup-2026-10.md) documents the completed
code/data integration and focused checks for the geocode precision rule,
Ocotillo address alias, two Williamson source-year ownership references, and
793 Oak Ranch home records with 24 case links.

The staging replay maps **11,716 recent filings**, compared with 11,784 in
current production: 76 unsupported coarse locations become unassigned and
eight false ambiguities resolve. Nine additional Ocotillo filings move to the
reviewed property cell. Oak Ranch adds **791 net units**, with the two existing
homes preserved, and supplies adequate denominators for 41 filings. The
combined low-unit residual is **161 filings in 65 cells**, compared with the
production baseline of 215 in 77. These are staging diagnostics, not new
cluster results. All 793 homes have classified 2025 ownership; 208 retain
unknown 2024 ownership.

The 39 deferred Oak Ranch records, the new 76-case coarse-location recovery
queue, and other unresolved property links remain explicit. No production
outputs, cluster fits or maps were rebuilt; full integration validation stays
at the end of the repair batch. The production tables above remain accurate
for their labeled batch-2 baseline.

Earlier completed batches passed their documented source-conservation,
assignment, unit-total, annual/paired agreement and model-output checks. The
staged ownership work passed 36 focused tests; Oak Ranch passed focused checks
of record identity, one home per account, spatial containment, existing-unit
preservation and all 24 case links. Those checks do not substitute for the
deferred production integration tests.

This report was independently reconciled against the **current saved
measurement and case ledger**, rather than only repeating earlier narratives.
Run `Rscript scripts/audits/residential_repair_closeout.R` from the repository
root to regenerate the aggregate report, source hashes and consistency checks
in `output/residential_repair_closeout_20261007/`. It writes audit summaries
only. No production rebuild or full test suite was run for this report.

Supporting dated audits preserve incremental baselines and source citations:

- [Ambiguity policy and restored coverage](eviction-ambiguity-policy-2026-09.md).
- [Original 41-cell diagnosis](eviction-low-unit-cells-2026-09.md).
- [Residential selection and shared property geography](residential-property-repair-2026-10.md).
- [Bell, Asher and Monarch case corrections](reviewed-property-locations-2026-10.md).
- [Reviewed counts and Domain repair](reviewed-unit-properties-2026-10.md).
- [Ben White assumption](ben-white-provisional-units-2026-10.md).
- [Second property batch](residential-property-batch2-2026-10.md).
- [Staged residential follow-up](residential-followup-2026-10.md).
- [Analytical changelog](../../CHANGELOG.md), including staged ownership fixes.

## Further staged repair batch — October 7

The [apartment and manufactured-home batch](residential-residual-repairs-2026-10.md)
extends the follow-up repairs with 676 homes and two apartment reference
corrections. The latest staged count is **108 filings in 56 below-threshold
cells** (58 filings in zero-unit cells; 50 in positive-count cells below 20).
Production is still unchanged, and this does not mean all location/denominator
issues are resolved. See the linked audit for the Hudson boundary hold and
remaining review queues.
