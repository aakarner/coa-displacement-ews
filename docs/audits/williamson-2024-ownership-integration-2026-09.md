# Williamson 2024 ownership supplementation: September 7, 2026

## Clarification and result

**We already had the 2024 certified roll.** The limitation was incomplete
classification evidence from clipped printed names and addresses, not a
missing 2024 file. The earlier description of a "2024 bottleneck" should not
be read as a newly discovered absence of that vintage.

A comparable public 2024 GIS archive has now been found, verified, and
reconciled with that existing roll using the same conservative rules applied
to 2025. The ownership-only Part 2 stage was rebuilt successfully.

For Williamson parcels in the analysis grid:

| Measure | Before 2024 GIS supplementation | After |
| --- | ---: | ---: |
| 2024 units with known corporate status | 87.8% | **99.4%** |
| 2024 units with known financialized/entity status | 69.0% | **98.9%** |
| Units usable in both 2024 and 2025 | 67.2% | **97.0%** |
| Common observed parcels | 8,435 | **12,745** |
| Common observed units | 12,707.07 | **18,358.79** |

The denominator is 13,014 eligible mapped Williamson parcels / 18,917.23
validated units. These are fixed parcel points in the computational H3 grid,
not countywide totals or an exact City of Austin polygon clip. A known flag
does not establish a complete co-owner roster.

## Verified source and timing

- [Public 2024 Williamson GIS ZIP](https://s3.amazonaws.com/data.tnris.org/d0f7da13-ab09-4994-a16f-d52589e2476e/resources/stratmap24-landparcels_48491_lp.zip).
- [Official collection metadata](https://api.tnris.org/api/v1/collections?collection_id=d0f7da13-ab09-4994-a16f-d52589e2476e).
- [Official county resource catalog](https://api.tnris.org/api/v1/resources?collection_id=d0f7da13-ab09-4994-a16f-d52589e2476e),
  Williamson resource ID `bcf49541-a5b9-4227-9897-5574dd0986ca`.
- ZIP size: 116,623,566 bytes; integrity test passes.
- SHA-256: `8603a705b80fb9f22c2037b32a7935176fc736bde01f3ed1c8010a49fff7a4ad`.
- Public object's Last-Modified: August 22, 2024, 14:24:14 UTC.
- GDB layer: `stratmap24_landparcels_48491_williamson_202406`, containing
  269,465 county features and 37 attributes.
- Every county record has `TAX_YEAR = 2024`, `DATE_ACQ = 20240601`, and source
  `WILLIAMSON APPRAISAL DISTRICT`.

The date metadata need qualification. The file/attribute date is coded June
2024, but the embedded XML describes July acquisition. Its processing history
records a direct extraction from `WilliamsonCAD.dbo.svewTNRISWilliamsonParcels`
on July 12, 2024, at 15:44:47 (timezone unspecified), followed by a July 19
copy. The existing certified report is dated July 12, 2024, at 13:52
(timezone unspecified). This establishes a contemporaneous 2024 source, not
an exact June 1 ownership inventory. The collection-level February date is
not a county snapshot date.

The source and those distinctions are pinned in
[`config/williamson_ownership_sources.json`](../../config/williamson_ownership_sources.json).
The Git-ignored ZIP is saved at
`data/raw_parcels/ownership_research/williamson_2024_txgio/stratmap24-landparcels_48491_lp.zip`.
No account login or public-information request was needed.

## What was reconciled

The 2024 certified report supplies one printed record for 13,519 of the
13,626 full Williamson targets. The GIS archive covers all 13,626 through
13,738 features. All 112 repeated feature IDs agree across the extracted
evidence fields and collapse before classification and unit aggregation.

The same source-adaptation rules now run independently for each year:

- Same parcel ID, exact/clipped-prefix owner-name agreement, and corroborating
  mailing street/city/state/ZIP are required for name extension or completeness
  confirmation. Certified evidence remains primary when owner or mailing
  evidence disagrees.
- Mailing and situs fields are replaced as one coherent GIS pair only when
  both corroborate certified evidence. Situs disagreement retains certified
  addresses, although independent name/mailing agreement can still verify a name.
- GIS-only parcels receive no inferred homestead or owner ID. Source-local
  full situs addresses supply their own locality; neither current target
  addresses nor the other year's evidence are borrowed.
- Missing, suppressed, possibly clipped, and conflicting evidence remain
  explicit. Original certified names are retained for longitudinal review.

The 2024 reconciliation yields:

| Source-evidence result | Full target parcels |
| --- | ---: |
| Corroborated name and address pair | 12,799 |
| GIS-only; no certified homestead inferred | 107 |
| Owner disagreement; certified retained | 190 |
| Mailing unconfirmed; certified retained | 121 |
| Situs unconfirmed; certified address pair retained | 409 |

There are **3,836 name extensions** and **4,561 completeness confirmations**,
including exact names at the printed column boundary. The 720 flagged
disagreements/unconfirmed comparisons are not verified ownership transfers.
No target GIS name is a suppression placeholder in 2024. Sixty-eight reach
the observed 80-character boundary and remain potentially clipped; 19 target
records lack full mailing locality. Field capacity alone is not evidence of
complete names.

Corporate status is now known for 13,503 of 13,626 full targets; 123 parcels
representing 122 units remain unknown. On mapped support, 116 parcels / 115
units remain corporate-unknown, and 194 parcels / 203.54 units remain
financialized-unknown. The 2025 classifications are unchanged: corporate
status covers 98.1% of its mapped units and financialized status covers 99.2%.

## Common support and sensitivity

The common cohort requires known corporate and financialized flags in both
years. It now includes 97.9% of Williamson mapped parcels and 97.0% of units.
Full fixed support remains unchanged in every variant. The full panel retains
237,305 parcels in each year, totaling 474,610 parcel-years including exclusions
and outside-grid records. Within the grid, each year has 233,334 eligible
residential parcels / 502,256.94 units across all counties.

The provisional cell screen remains at least 20 common units and at least
95% coverage of both parcels and units. This is a completeness screen, not
the final cluster sample or a guarantee of representativeness.

| Variant | Williamson common parcels | Williamson common units | All-county screened cells |
| --- | ---: | ---: | ---: |
| Reconciled GIS/certified evidence in both years | 12,745 | 18,358.79 | **3,227** |
| Exclude flagged parcel-years in either year | 12,167 | 17,780.79 | **3,196** |
| 2024 certified-only, 2025 reconciled | 8,435 | 12,707.07 | 3,063 |
| Both years certified-only | 7,980 | 12,094.53 | 3,063 |

The 2024-certified-only variant exactly reproduces the previous run's hex
comparison, isolating the new 2024 evidence. There are 164 additional passing
cells in the main result. Of 228 cells containing Williamson residential
parcels, 173 now pass, versus nine previously; these include mixed-county
boundary cells. The conflict-exclusion variant retains 142 of those cells.

Thirty-one cells' eligibility is sensitive to source discrepancies. Even
among cells passing both main and conflict-exclusion screens, corporate-share
changes can differ by up to 4.76 percentage points. Do not describe this
reconciliation as sensitivity-free or treat source-sensitive cells as equally
secure inputs to a later cluster comparison. The cohorts differ between
variants, so their numerical differences combine sample composition and
classification effects.

The bounded change review found 30 corporate-share deltas differing among
cells passing both screens, with no sign reversals. Two one-unit signals
disappear under conflict exclusion (hex 2465: -4.762 points to zero; hex 3242:
+1.754 points to zero). The other discrepancies are at most 0.114 points and
reflect denominator adjustments. Exclusion removes seven of Williamson's 81
corporate-status transition rows, all one-unit parcels. These are
small-denominator/source-evidence effects, not a new multifamily transfer signal.

The `source_agreement` output label means excluding flagged conflicts, not
requiring affirmative corroboration from two sources for every parcel.
GIS-only and unconflicted certified-only rows can remain eligible. Exclusion
counts distinguish 1,216 flagged parcel-years from 752 unique parcels to
avoid counting housing units twice across years.

Within the main common Williamson cohort, 48 parcels / 48 units enter the
corporate category and 33 parcels / 32 units leave it: a net increase of 16
units. Corporate-unit share changes from 29.034% to 29.121%. This is a
classification change on a fixed observed cohort, not proof of acquisitions,
sales, or displacement. All 81 transitions involve one-unit or zero-unit
parcels; seven retain the same reported name and owner ID and differ only in
homestead or address-match evidence. The different level from the previous run reflects
the much broader cohort; it is not itself a change in the neighborhood.

## Checks and outputs

The full stage, 52 Python tests, R synthetic tests, and output regression
checks pass. The output checks verify 57 declared file hashes, unique
parcel-year keys, unchanged validated coordinates/units and canonical hex
denominators, preserved original certified names, and all four source variants.
They also confirm unchanged Travis, Hays, and Williamson 2025 classifications.
The optimized geometry-free source extraction reproduces the prior 2025
evidence bytes exactly. No clusters were rerun, Part 1 outputs were not
replaced, and ML remains paused.

```sh
Rscript scripts/part2/build_ownership_snapshots.R
python3 -B -m unittest discover -s tests -p 'test_*ownership*.py'
Rscript tests/test_williamson_txgio.R
Rscript tests/test_ownership_snapshots.R
Rscript tests/test_williamson_ownership_outputs.R
```

The targets graph also parses and tracks both annual inputs/outputs. The
classifier remains pinned to landlord-mapper; this does not close issue #5's
full upstream parcel-reconstruction/classifier-migration work.

Under the ignored `output/part2/ownership/`, start with:

- `ownership_county_qa.csv`, `ownership_hex_change.csv`, and
  `ownership_snapshot_manifest.json` for primary results/provenance.
- `williamson_2024_source_reconciliation_qa.csv` and
  `williamson_2024_source_conflict_review.csv` for the new source resolution.
- `ownership_source_variant_summary.csv`,
  `ownership_source_variant_county_qa.csv`, and the certified-only,
  source-agreement, and 2024-certified-only hex-change files for sensitivities.
- `ownership_transition_review.csv` for annual classification-change evidence.

The previous run's summary/change tables and other-county snapshots are
preserved in `pre_williamson_2024_integration/`. The
[method document](../methods/historical-ownership.md) is the current contract;
the [2025 integration audit](williamson-2025-ownership-integration-2026-09.md)
preserves the earlier evidence state.

## Implication for Part 2

The earlier widespread 2024 clipping limitation is substantially resolved;
we do not need another broad ownership-data search before progressing.
Retain the remaining unknowns and explicitly consider the 31 source-sensitive
cells when choosing the comparison sample. The next distinct analysis task is
assembling the remaining historical smoke signals, including amenities, and
then deciding the transformations/sample for the actual cluster-stability
proof of concept. This ownership-only run does not construct that matrix or
estimate cluster movements.
