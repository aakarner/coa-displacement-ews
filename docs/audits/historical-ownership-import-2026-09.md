# Historical ownership import: September 7, 2026

**Subsequent source-search correction (September 7):** the official 2025
Williamson certified owner roll has now been located, downloaded, and verified.
See the [source-discovery audit](williamson-2025-ownership-source-search-2026-09.md).
The results below preserve the earlier import run. The subsequent
[Williamson integration audit](williamson-2025-ownership-integration-2026-09.md)
records the rerun with certified/GIS evidence and sensitivity checks. Missing
Williamson 2025 is no longer a source-availability problem. The later
[2024 integration audit](williamson-2024-ownership-integration-2026-09.md)
records supplementation of the existing 2024 roll and substantially improved
two-year classification coverage.

## Bottom line

The EWS ownership stage is implemented and has run end to end. Travis and Hays
have usable 2024/2025 snapshots on the fixed validated EWS surface. Williamson
has a partial-evidence 2024 snapshot but no verified 2025 owner snapshot; its
missing year remains explicit. This is a substantial Part 2 input, not a
complete citywide historical feature matrix or a cluster-stability result.

The run produced 474,610 parcel-year rows, retained all 7,027 grid cells, and
identified 3,055 cells passing the initial common-support coverage screen.
No Part 1 outputs were replaced, no cluster model was refit, and no ML work
was resumed. The upstream classifier remains in landlord-mapper; this does
not close the full parcel-reconstruction/classifier-migration issue #5.

## What was integrated

- The Travis snapshot and shared classifier from landlord-mapper commit
  `cd686394e51467541d5463422dcf7bed4aa9fee9`, with SHA-256 verification.
- Annual Hays owner/property archives for 2024 and 2025 and the Williamson
  2024 printed roll, adapted through that same classifier.
- Fixed EWS promoted units (`v2_2026-07-31_land_use_validated`), coordinates,
  land-use exclusions, and canonical H3 assignments.
- Separate full-support uncertainty, common-parcel change, old-method parity,
  source-coverage, and large-change review outputs.

See [`historical-ownership.md`](../methods/historical-ownership.md) for the
source contract, missing-data rules, reproducible commands, and output meanings.

## Coverage in the actual hex analysis

These denominators include eligible parcels assigned to the computational H3
grid. They are not countywide totals or the later Part 3 FULL-polygon sample.
Source matches and usable classifications are different: a matched record
may still have insufficient ownership evidence.

| County/year | Target parcels | Source-matched parcels | Parcel match | Unit-weighted match | Units with unknown corporate status |
| --- | ---: | ---: | ---: | ---: | ---: |
| Travis 2024 | 220,042 | 219,221 | 99.63% | 99.30% | 3,380.74 |
| Travis 2025 | 220,042 | 220,042 | 100% | 100% | 1.00 |
| Hays 2024 | 278 | 278 | 100% | 100% | 0 |
| Hays 2025 | 278 | 278 | 100% | 100% | 0 |
| Williamson 2024 | 13,014 | 12,911 | 99.21% | 97.57% | 2,308.96 |
| Williamson 2025 | 13,014 | Unavailable | Unavailable | Unavailable | 18,917.23 |

Hays has four mapped parcels with unknown financialized status in each year,
even though corporate status is determined. Williamson's printed 2024 source
leaves 4,534 mapped financialized flags unknown, largely because negative
classification cannot safely be made from potentially clipped names. It
contains one printed owner record per parcel, not a complete co-owner roster.

The full promoted target has 237,305 parcels. The fixed land-use rule excludes
142; another 3,829 eligible parcels fall outside the grid (Travis 3,197,
Williamson 612, Hays 20). All remain in the support audit. The 233,334 mapped
parcels sum to approximately 502,257 validated units, identically in both years.
No parcel maps to multiple cells. Counts and weights reconcile to the existing
canonical EWS parcel and hex outputs.

## What the comparison currently shows

Of 4,091 grid cells with residential support, 3,879 have some common ownership
support and 3,055 pass the initial screen: at least 20 common units and at least
95% of both fixed parcels and units observed in both years. The remaining cells
are retained, with coverage and explicit missingness. Passing is a completeness
screen, not final approval for clustering or a guarantee of representativeness.

For the common mapped Travis cohort (219,220 parcels and approximately 479,680
fixed validated units), measured corporate-unit share is 47.24% in 2024 and
46.04% in 2025: a decline of 1.19 percentage points. There are 1,329 parcels
changing from noncorporate to corporate and 1,643 changing in the other
direction. These are classification transitions, not confirmed sales or
displacement events. In particular, 247 changing Travis parcels retain the
same owner names and IDs; their homestead or address-match evidence changes.

Hays' common mapped cohort has 274 parcels/units, with two corporate units in
2024 and one in 2025. No Williamson parcel enters the two-year common cohort.
Mixed-county hex cells remain subject to the same coverage screen, so a small
unobserved Williamson share can be present in an otherwise screened hex.

There are 52 screened cells with a corporate-unit-share swing of at least
10 percentage points. Their diagnostic table and underlying parcel transition
table are saved for review. A targeted inspection of the ten largest Travis
parcel transitions found important classification sensitivities:

- Nine of the ten (about 3,364 units) become noncorporate with changed owner
  names/IDs; the new names contain housing-finance or public-housing wording
  not matched by the existing financialized-name rules. Across all transitions,
  24 parcels representing about 5,777 units have new names containing
  housing-finance, public-facility, or housing-authority wording. This warrants
  separate substantive interpretation, not a conclusion of corporate
  disinvestment. The name wording does not by itself establish beneficial
  ownership or a particular housing program.
- Parcel `820453` (328.53 units) becomes corporate with the same owner ID,
  owner name, and normalized mailing address. Its source situs address changes,
  causing the inferred owner-occupancy flag to change. The raw standardized
  upstream records confirm this is an address-driven classification change,
  not an observed owner transfer.
- No Travis corporate transition has unchanged owner ID, name, homestead
  evidence, and address-match evidence simultaneously. No unmistakable parser
  defect was found. A 100-point cell swing can nevertheless reflect one large
  multifamily parcel and be sensitive to its classification.

The shared classifier was not redefined in response to these findings. Before
clustering, decide how to interpret housing-related public entities and review
address-driven outliers; any rule change should be applied to both years with
a new version. A normalized ownership-pressure index and cluster
movement should wait for a common transformation recipe and the other
historical feature domains.

## Why the 2025 parity check matters

The harmonized 2025 classifier differs from the old current Travis flags for
412 eligible parcels: corporate TRUE becomes FALSE, affecting 433.23 validated
units; one additional unit becomes unknown. Financialized flags agree on all
223,239 eligible Travis parcels. Owner-occupancy changes for 38,276 parcels,
with another 11 now unknown.

These are the documented upstream corrections for recovered homestead and
address evidence, not annual changes. They are kept separate from the
2024-to-2025 comparison. The old upstream unit surface totals about 792,121
Travis units, versus 488,174 validated EWS units, so upstream unit-weighted
statistics must not be used as EWS results.

## Williamson 2025 source investigation

A bounded follow-up search found no usable complete 2025 ownership source:

- The cached owner table is for 2026; its companion certified property table
  describes 2025. Joining them would not create a 2025 owner snapshot.
- The [WCAD public catalog](https://data.wcad.org/browse) has live owner data.
  The [PropOwner export](https://data.wcad.org/d/absk-uy9g) inspected in this
  search contained 324,599 records, all with tax year 2027, not 2025.
- The [certified appraisal report](https://documents.wcad.org/cert/CertifiedAppraisalReport.txt)
  is a 2025 property-only table; it does not supply the needed owner evidence.
- The [Entity Portal](https://www.wcad.org/entity-portal/) was inspected through
  `City of Austin - CAU/2025`. Its supporting documents contain aggregate
  assessment/valuation information and top taxpayers, not a complete
  owner-and-mailing population.
- The direct 2025 counterpart to the known 2024 printed-roll ZIP returned 404;
  the searched WordPress upload catalog and Wayback index did not supply it.

This does not prove that no public copy exists. If Williamson must be included
in the first proof of concept, the focused follow-up is to obtain **2025
certified owner records for City of Austin accounts**: parcel/QuickRefID,
owner ID/name, mailing address, homestead/exemption evidence, and source
date/supplement. The [WCAD public-information page](https://www.wcad.org/public-information-open-records-request/)
provides the route. No request has been submitted. Alternatively, scope an
initial cluster comparison to a consistently observed geography and clearly
report the missing Williamson coverage; that sample decision is still open.

## Validation and next steps

The new R synthetic tests and all 10 Python adapter tests pass. The upstream
classifier's 27 tests also pass. The direct EWS stage completed successfully;
the target graph parses, and the ownership target's outputs exist. The full
canonical pipeline was not rerun. Independent review confirmed fixed parcel
assignments, year-invariant denominators, and explicit unknown handling.

Reproducible outputs live under the Git-ignored `output/part2/ownership/`:

- `ownership_snapshot_manifest.json`: hashes, versions, coverage and checks;
- `ownership_county_qa.csv` and `ownership_support_qa.csv`: coverage/support;
- `ownership_features_by_hex_year.rds/.csv`: full-support estimates and bounds;
- `ownership_common_support_by_hex_year.rds/.csv` and `ownership_hex_change.csv`:
  common-cohort levels, differences and screening flags;
- `ownership_2025_parity_qa.csv`: old-method comparison;
- `ownership_transition_evidence_qa.csv`, `ownership_transition_review.csv`,
  `ownership_large_hex_changes.csv`, and `ownership_unknown_review.csv`: review
  diagnostics. Parcel-level files contain owner information and stay ignored.

Next, settle the Williamson/common-geography treatment and the substantive
handling of housing-finance/public-entity and address-driven changes, complete
the large-change review, and reconstruct the historical amenity feature using the
already documented archive-first and archive-plus-live alternatives. Assemble
the other domains at the same two cutoffs, then freeze a shared scaling recipe
and sample before comparing fixed-cluster reassignment with cluster refitting.
