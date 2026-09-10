# Historical ownership inputs for Part 2

This stage builds a retrospective 2024/2025 ownership comparison on fixed EWS
parcel geography and validated housing-unit counts. It supplies one input to
the proposed cluster-stability proof of concept; it does not build the full
historical feature matrix or change existing Part 1 outputs.

## Sources and classifier

[`config/ownership_snapshot_spec.json`](../../config/ownership_snapshot_spec.json)
pins landlord-mapper commit `cd686394e51467541d5463422dcf7bed4aa9fee9`, the
classifier SHA-256, its rule version, and the Travis snapshot checksum. The
code must be available in the sibling `../landlord-mapper` checkout. Its
committed classifier blob and working file must agree; unrelated upstream
changes do not invalidate the import.

Travis snapshots are imported from that upstream workflow. The EWS Hays and
Williamson adapters call the same upstream Python classifier without copying
its entity-name rules or writing into landlord-mapper.

Corporate ownership means residential, financialized ownership, and not
owner-occupied. The classifier combines entity-name markers, owner mailing
and situs address matches, and homestead evidence. Among available owner rows,
any financialized owner makes the financialized flag positive; any occupancy
evidence rules out corporate ownership. Shares are not apportioned. Unknown
evidence remains unknown unless other evidence logically determines the flag.

| County | 2024 source | 2025 source |
| --- | --- | --- |
| Travis | Certified supplement 0 rerun; property-year owner | Pinned special-export owner and associated homestead evidence |
| Hays | Certified annual owner and property archives | Certified annual owner and property archives |
| Williamson | Certified roll, supplemented conservatively with the verified 2024 TxGIO/WCAD parcel archive | Certified roll, supplemented conservatively with July 2025 TxGIO/WCAD parcel evidence |

The cached Williamson owner extract is for 2026 and cannot substitute for 2025.
The 2024 printed roll has one displayed owner record per parcel, not a verified
complete co-owner roster. Observed entity markers and homesteads are usable
positive evidence; potentially clipped negative names and address comparisons
are withheld. Annual Hays archive member names and Williamson printed report
years are checked before classification. This stage never downloads live data
to replace missing historical files.

### Williamson annual source reconciliation

[`config/williamson_ownership_sources.json`](../../config/williamson_ownership_sources.json)
pins the certified and GIS ZIPs for both years and their checksums. The 2024
certified report was already available and dated July 12, 2024; its limitation
was clipped fields, not a missing vintage. The 2024 GIS is coded `DATE_ACQ =
20240601`, but its XML says acquired in July and records extraction from the
WCAD database on July 12, with public distribution August 22. Preserve those
distinctions; do not call it an exact June 1 inventory. The 2025 certified
report is dated July 11, 2025, but posted online July 16, 2026. Its GIS source
has July 1 acquisition metadata and July 28 processing.
All four local ZIPs must be present and unaltered; the pipeline does not fetch a
live substitute. Their provenance and retrieval routes are documented in the
[2025 source-search audit](../audits/williamson-2025-ownership-source-search-2026-09.md)
and [2024 integration audit](../audits/williamson-2024-ownership-integration-2026-09.md).

The certified roll is primary. The adapter supplements it as follows:

- Require the same parcel ID, matching owner name (exact or a literal prefix
  at the printed clipping boundary), and corroborating mailing street, city,
  state and ZIP before extending a name or confirming that a 29/30-character
  printed name is complete. Names reaching the GIS's observed 80-character
  boundary remain potentially clipped.
- Replace mailing and situs evidence only as a coherent same-source pair,
  after both agree with the certified evidence. A clipped printed address may
  match a longer GIS prefix under the same locality. A situs disagreement
  retains the certified address pair and is flagged, even when independent
  owner/mailing agreement allows name confirmation.
- Retain certified ownership when owner or mailing evidence disagrees. Never
  attach a certified homestead to a different GIS owner.
- Where the certified roll has no parcel, use the GIS source alone: no
  homestead or owner ID is inferred. Parse situs locality from that source's
  own full address; do not borrow the mailing address or current EWS address.
  Missing/suppressed evidence remains unknown. Duplicate geometry features
  with identical evidence collapse to one parcel before any unit weighting;
  conflicting repeated evidence is retained for review, not silently selected.

These same matching rules are applied independently to both years, improving
source evidence without changing the pinned classifier. No names, addresses
or homesteads are carried across years. Neither annual source
establishes a complete co-owner roster or an exact common-day owner inventory.

The intended mapping is 2024 appraisal data for April 1, 2025 and 2025 data for
April 1, 2026. These are appraisal vintages, not proof of ownership or published
information on those exact dates. This is a retrospective comparison, not a
leakage-free predictive backtest.

## Fixed geography and denominators

Both years use `output/residential_parcels_unit_promoted.rds`, its coordinates,
and `units_calibrated_targeted`. Upstream ownership-output unit weights predate
EWS validation and are ignored. Current land-use exclusions apply to both years.

Parcel points use the canonical `sf::st_within` join to `output/hex_grid.rds`.
Excluded and outside-grid parcels stay in the parcel audit but not analytical
hex summaries. Eligible parcel weights and every hex's parcel/unit totals must
agree with the current canonical surface. Historical construction, demolition,
parcel splits, and changing unit counts are not reconstructed by this design.

## Two complementary summaries

1. **Full fixed support:** retain every hex/year and count unknown ownership.
   Corporate-unit share and density have lower/upper bounds obtained by
   treating unknown units as all noncorporate or all corporate. Strict estimates
   are missing wherever required evidence is incomplete. Financialized-parcel
   share is withheld when any financialized flag is unknown. Cells without
   residential support have no ownership estimate, not a zero-risk estimate.
2. **Common observed support:** use exactly the same parcels in both years,
   requiring known corporate and financialized flags in both. Occupancy need
   not itself be known if those two flags are determined. Calculate corporate
   unit share, corporate unit density, and financialized parcel share and their
   changes on this fixed subset.

Common-support coverage is measured against full fixed support. The initial
screen requires at least 20 common units and at least 95% coverage of both
parcels and units. These configurable thresholds screen completeness; they do
not guarantee representativeness or define final cluster eligibility. Values
below the screen remain inspectable; select `comparison_ready` for screened
comparisons. Common-support densities and shares describe the observed cohort,
not an imputed full-cell estimate.

This source-snapshot stage does not calculate the normalized `ownership_pressure_index`.
The separate downstream index stage below now performs that calculation.
Cross-vintage transformation rules and the final cluster sample must be chosen
with the other historical smoke signals.

## Separate paired-index stage

`scripts/part2/build_ownership_index.R` verifies the existing snapshot manifest
and its recursively pinned files, then scores the three selected common-support
components for 2024/2025 (mapped to April 2025/2026 cutoffs). Corporate-unit and
financialized-parcel shares are percentages; coverage measures are proportions.
Corporate shares/densities use common units, while financialized share uses
common parcels. The original 20-common-unit and 95%-unit/parcel-coverage screen
is retained. Inputs outside it are missing, not zero.

The earlier component 1st/99th-percentile bounds are frozen for the later
snapshot. The resulting index is the mean of its three 0–100 scores. All
3,227 screened cells have all three components at both dates; this is not the
final seven-feature sample. Full-support unknowns and bounds, common-support
coverage, and source-variant sensitivity flags remain alongside the scores.

Outputs go only to `output/part2/ownership_index/`; the original snapshot
directory and its manifest are verified unchanged. Reproduce with
`Rscript scripts/part2/build_ownership_index.R`, then run
`tests/test_part2_ownership_index.R` and
`tests/test_part2_ownership_index_outputs.R` using Rscript. See the
[paired-matrix methods](historical-feature-matrix.md) for final sample assembly.

## QA and interpretation

`ownership_2025_parity_qa.csv` compares harmonized 2025 flags with existing
Part 1 flags. These method/evidence corrections are separate from annual
changes. `ownership_common_support_transitions.csv` records annual transitions;
the evidence and review tables distinguish owner name/ID changes from changes
in homestead or mailing-match evidence. A transition does not independently
establish a sale, acquisition, or displacement.

For Williamson rows present in both certified rolls, `comparison_name_changed`
uses the original printed names in both years. `owner_names_changed` still
reports differences in the names used for classification, but may reflect
longer GIS evidence rather than a different owner. Equal printed names also
cannot rule out changes hidden by clipping.

The adapter exports parcel-level source comparisons and aggregate reconciliation
QA. Three sensitivity comparisons accompany the primary hex changes:

- `ownership_certified_only_hex_change.csv`: use only the certified rolls for
  Williamson in both years, holding all other inputs fixed.
- `ownership_2024_certified_only_hex_change.csv`: use only the 2024 certified
  roll but retain the reconciled 2025 evidence. This isolates the effect of
  supplementing 2024 and reproduces the prior run's source design.
- `ownership_source_agreement_hex_change.csv`: withhold all three ownership
  flags on each parcel-year flagged for source conflict or unconfirmed correspondence, then
  rebuild the same two-year common-support rule. GIS-only rows without conflicts
  remain eligible; this is a conflict-exclusion test, not a dual-source-only sample.

`ownership_source_variant_county_qa.csv` and
`ownership_source_variant_summary.csv` compare coverage and screened cells
across these variants. Their common cohorts can differ: changes in estimated
shares are not purely classification effects on an identical subset.

The pinned name-marker rules can classify housing-finance/public-entity names
as nonfinancialized, and a changed source situs can change inferred occupancy
without an owner transfer. The run audit documents such cases. Preserve the
versioned definitions in this import; resolve any substantive reclassification
as an explicit revision applied to both years before interpreting clusters.

`ownership_large_hex_changes.csv` flags screened corporate-unit share changes
of at least 10 percentage points for follow-up. Unknown records, clipping,
outside-grid records, and source gaps remain explicit. The run manifest records
source/output hashes, unit-surface version, classifier provenance, coverage,
and reconciliation checks. Generated owner information stays Git-ignored.

## Running and testing

From the EWS root with the pinned sibling checkout and cached sources:

```sh
Rscript scripts/part2/build_ownership_snapshots.R
Rscript tests/test_ownership_snapshots.R
python3 -B -m unittest discover -s tests -p test_other_county_ownership.py
python3 -B -m unittest discover -s tests -p test_williamson_ownership_reconciliation.py
Rscript tests/test_williamson_txgio.R
Rscript tests/test_williamson_ownership_outputs.R
```

The direct script uses existing validated inputs and writes only to
`output/part2/ownership/`. `--prepare-only` exports the target CSV without
classification or aggregation. `EWS_PYTHON` selects the Python executable.

The pipeline equivalent resolves normal upstream dependencies too:

```sh
Rscript run_analysis.R part2_ownership_snapshots
```

For an isolated refresh use the direct script. Keep
`EWS_TARGETS_ADOPT_EXISTING` unset/false for a fresh rebuild. Missing county
snapshots become explicit unavailable vintages. Missing required Travis/EWS
inputs, any pinned Williamson ZIP, or invalid/wrong-year present archives
stop the run. GIS preparation checks the annual county source and writes a
geometry-free, target-filtered evidence file plus checksum manifest.

This speed-first arrangement leaves full upstream parcel reconstruction and
classifier migration in issue #5 open. See the
[`2024 Williamson integration audit`](../audits/williamson-2024-ownership-integration-2026-09.md)
for the updated two-year reconciliation and remaining evidence limitations; the
[`2025 integration audit`](../audits/williamson-2025-ownership-integration-2026-09.md)
preserves the intermediate run, and the
[`initial run audit`](../audits/historical-ownership-import-2026-09.md) preserves
the pre-integration baseline.
