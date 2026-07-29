# Residential Unit Count Modeling

## Purpose

The residential unit workflow separates source evidence, project construction,
model training, and production integration. This prevents a project total from
being repeated on every parcel account and allows disagreements to remain
visible.

WCAD residential eligibility is a production preprocessing rule.
`scripts/data/parcel_units_calibrate.R` applies it before unit calibration and writes
excluded and review records separately. Scripts `02p` through `02t` build and
validate the project hierarchy. `scripts/data/unit_counts/promote_integration.R` converts that
reviewed hierarchy into the canonical parcel input while retaining the `02e`
targeted surface as an immutable, parcel-linked baseline.

The EWS repository owns this definition. It reads local WCAD raw files through
`R/wcad_unit_eligibility.R`; it does not import or execute `landlord-mapper`
code. County CSV exports are treated as broad candidate inputs under a
documented schema, not as authoritative residential classifications.

## Source hierarchy

`02d` first joins the compact WCAD legal, use, unit, and property-type fields to
the Williamson candidate parcels. It writes canonical eligibility audit,
exclusion, review, and QA outputs before excluded rows can enter calibration,
corporate aggregation, ACS allocation, or feature engineering.

`scripts/data/unit_counts/prepare_sources.R` then stores each project or account count once in
`output/residential_unit_source_records.rds` and stores parcel relationships
separately in `output/residential_unit_source_parcel_links.rds`.

The evidence classes, in selection order, are:

1. Completed City affordable-housing project totals, high-confidence CoStar
   project totals, and plausible TCAD `imprvUnits` on B1 properties. These can
   supply strict model labels.
2. Deterministic appraisal-account evidence. TCAD uses one unit for A1-A4,
   two for B2, three for B3, and four for B4. Hays uses the corresponding
   A/B2/B3/B4 state-code rules. WCAD contributes explicit residential
   condominium/unit accounts and duplex/triplex/fourplex legal descriptions.
   These counts are not floor-area model labels.
3. A one-unit WCAD rule for positive-area residential accounts that contain no
   condominium, small-multifamily, or apartment signal. This is retained as a
   separate rule-based tier rather than treated as direct evidence.
4. City Universal Recycling Ordinance multifamily counts. The City calls these
   estimated counts, so they are retained as sensitivity labels.
5. Future floor-area model predictions for projects still unresolved after the
   preceding sources.

No lower-tier source overwrites a higher-tier source. Source conflicts remain
unclassified pending review.

The Williamson cleanup distinguishes primary property evidence (legal
description, DBA, and use description) from historical property comments.
Incidental `APT` references in an owner's former mailing address do not make a
parcel multifamily. A comment-only apartment signal is accepted only for C3/C5
properties, which retains a documented apartment project without misclassifying
ordinary homes.

The following WCAD records remain visible in audit tables but are excluded from
the unit-bearing parcel universe:

- `REFERENCE ONLY` condominium common-interest or master accounts;
- nonresidential condominium units identified by WCAD property and use codes;
- park or amenity parcels with no residential building area;
- transitional-commercial land with no building area; and
- other explicitly nonresidential accounts carried into the upstream extract.

## Project construction

`scripts/data/unit_counts/build_projects.R` connects parcels only when they share:

- a high-confidence source record; or
- the same normalized address within 250 meters and at least one multifamily
  signal.

It does not group projects by owner name alone. Complete condominium and
small-multifamily account groups are summed, while repeated TCAD project totals
on multiple B1 parcels are treated as ambiguous.

Direct sources must agree within 20 percent. A complete account enumeration is
also used as a conflict check. The script does not average sources that fail
this test.

Two apartment properties at the Travis-Williamson boundary have parcel
records in both appraisal districts. They remain unified when a shared,
high-confidence project source establishes that they are the same physical
property. Outputs include `project_counties` and `project_cross_county`; county
QA counts these projects once in each involved county, while regional project
and unit totals count them only once.

Main/living floor area is the canonical model exposure. For a cross-county
project represented in both appraisal systems, `02q` uses the maximum
county-specific area total instead of summing duplicate county representations.
The raw sums and aggregation method remain in the project table for audit.
Reviewed WCAD companion accounts for Lantern at Westwood and Lakeline Crossing
are also connected upstream, before model candidates are constructed.

## Promoted hierarchy

After production eligibility filtering, the source hierarchy produces:

- 237,305 eligible parcel rows grouped into 206,305 project records;
- 264 excluded candidate parcels retained outside the unit universe;
- 874 strict, model-eligible multifamily projects;
- 853 unique unresolved multifamily model candidates; and
- 141 direct-source or direct-versus-account conflicts held out for review.

The mutually exclusive selected hierarchy currently contains:

- 116,434 units from 892 strict direct project totals;
- 230,518 units from 188,497 deterministic appraisal-account projects; and
- 10,762 units from 10,762 WCAD single-unit-rule projects.

Together these selections account for 357,714 units. Only the first
116,434 are direct reported project totals; the appraisal and single-unit
tiers must remain separately identified in reporting.

All current model-eligible projects are in Travis County. The first model must
therefore use a common regional form without an estimated county effect.
County remains a validation and out-of-domain flag. A county effect should be
considered only after obtaining defensible project labels in Hays and
Williamson.

Current county handling is:

- Hays: all 298 currently linked residential parcels receive deterministic
  one-unit appraisal-code counts. No Hays multifamily model candidate is
  present in the current extract.
- Williamson: 2,742 explicit residential unit accounts supply 2,742 units;
  96 small-multifamily legal descriptions supply 194 units; and 10,762
  ordinary residential accounts, including 378 supplemented records, receive
  the separate one-unit rule.
- Williamson has 23 apartment model candidates after companion accounts are
  grouped upstream. Nineteen have URO sensitivity counts, and documented
  project sources cover the four remaining developments. Two of the 23 are
  cross-county projects also represented in TCAD.
- Williamson excludes 202 nonresidential condominium accounts, 55
  reference-only common-interest accounts, four park/amenity parcels, two
  transitional-commercial land records, and one other nonresidential account.
- One record remains under manual review: a 1,886-square-foot residential lot
  whose DBA says `SPRINGWOODS APTS` but whose appraisal and legal fields do not
  establish a unit count.

The populated zero-unit review exposed a source-version mismatch in
Williamson. The current certified roll contains 407 active residential records
with positive living area and unique parcel/address keys that were absent from
the broad input despite having geometry inside the study grid. One additional
active certified property uses a reviewed legacy parcel solely as its geometry
proxy. `02d` now adds these 408 records through
`R/wcad_residential_supplement.R`, assigns one unit under the existing WCAD
rules, and writes a complete source crosswalk. The supplement includes 30
explicit residential condominium accounts, 378 ordinary one-unit records, and
16 corporate-owned parcels. It resolves all initially flagged WCAD proxy gaps.

The 264 excluded parcels carried 1,344.823 units under the former production
rules, including 1,275.6 corporate-owned units. The integrated eligibility rule
now removes them before calibration. A full refresh produces 534,861 primary
units and 519,504 targeted units regionwide; the H3 parcel/ACS audit contains
514,263 targeted parcel units versus 479,614 ACS housing units. The complete
refresh also holds a pre-existing Travis CoStar mismatch out of training: its
address says 308 E 34th while its geocode falls near 308 W 34th. The repaired
project construction now yields 874 training projects and 853 project-level
candidates.

Affordable Housing Inventory records require special care. Some completed
records describe a subsidized subset within a larger condominium property.
Complete account enumerations and other direct sources are therefore used to
detect, rather than silently accept, those mismatches.

## Current model comparison

`scripts/data/unit_counts/fit_models.R` compares:

1. a fold-refit global square-feet-per-unit ratio;
2. square-feet-per-unit ratios stratified by project floor-area and story bands;
3. a negative-binomial GAM with log residential floor area as an offset; and
4. monotonic gradient boosting as a performance benchmark.

The script uses five project-stratified folds, five spatial-cluster holdouts,
and three source holdouts. No demographic, ACS, URO, or displacement-outcome
variables enter model training. Current project-fold results are:

| Method | Pooled WAPE | Median fold WAPE | Large-project WAPE |
|---|---:|---:|---:|
| Global ratio | 27.2% | 27.0% | 26.4% |
| Stratified ratio | 17.5% | 17.3% | 15.0% |
| Negative-binomial GAM | 20.6% | 20.2% | 18.1% |
| Monotonic boosting | 16.5% | 16.3% | 13.1% |

The stratified ratio is the current conservative recommendation. It improves
median fold WAPE by 36.1 percent relative to the global ratio, keeps maximum
absolute fold bias below 3.3 percent, and keeps size-band bias below the
predeclared 20 percent threshold. Its leave-one-fold-out empirical 80 percent
intervals cover 79.3 percent of held-out projects.

Boosting is more accurate in aggregate and has near-zero total bias, but it
exceeds the size-band bias gate at 24.4 percent. It remains a benchmark pending
small-project calibration. The GAM overpredicts overall, exceeds the fold-bias
gate, and is materially worse than boosting for large properties.

Across all 853 unresolved candidates, the stratified method predicts 146,107
units, compared with 184,896 current primary and 116,326 current conservative
units. A total of 767 projects, containing 129,915 predicted units, pass the
model-domain screen. The review flags include 23 projects that touch
Williamson County, 57 projects outside at least one observed training range,
23 projects with a missing core predictor, and 41 predictions below the
five-unit training scope; these categories overlap.

Using the stratified prediction for the 767 in-scope projects and retaining
current primary estimates for the 86 review projects would produce 147,556
candidate units. Using current conservative estimates for the review group
would produce 142,281. These were pre-promotion scenario bounds. The narrower
validated hierarchy described below was subsequently promoted through `02v`.

Across all 409 candidate projects with URO estimates, the main-area stratified
ratio has 15.7 percent WAPE, boosting has 17.3 percent, and the current
conservative estimate has 46.7 percent. URO remains a sensitivity source rather
than an authoritative training label.

Among the 19 Williamson-transfer projects with URO estimates, the main-area
stratified model has 13.3 percent WAPE, compared with 25.2 percent for the
current conservative estimate. The earlier transfer failure was primarily a
floor-area definition and appraisal-structure problem, not evidence that
Williamson needs a separate housing-production model.

## Williamson validation and comparable floor area

`scripts/data/unit_counts/validate_williamson.R` audits 23 candidate projects touching
Williamson. Two pairs of WCAD companion accounts, Lantern at Westwood and
Lakeline Crossing, are grouped upstream in `02q`, so each development enters
validation and prediction once. The two cross-county projects also contain
TCAD and WCAD appraisal records for the same development; their areas are not
summed as though the records represented independent buildings.

All 23 developments now have an external unit reference: 19 have City URO
estimates, and documented project sources fill the four remaining development
gaps. Nine developments have a documented source in addition to or instead of
URO. URO remains an estimate and is not promoted to a direct training label.
The AEGB and TDHCA inventories produce no exact candidate matches, which means
the properties are not in those limited program inventories, not that they
contain zero units. The HUD multifamily service timed out and contributes no
negative evidence.

The key measurement result is that TCAD `main_area` and WCAD
`TotalSqFtLivingArea` are substantially more comparable than total improvement
area. On the existing project-grouped folds:

| Floor-area definition | Ratio method | Pooled WAPE | Bias |
|---|---|---:|---:|
| Main/living area | Stratified ratio | 17.5% | 0.2% |
| Total improvement area | Stratified ratio | 24.1% | -0.3% |
| Main/living area | Global ratio | 27.2% | 15.5% |
| Total improvement area | Global ratio | 39.4% | 10.5% |

For the 19 pure-Williamson developments with plausible main-area records, the
main-area stratified estimate has 9.7 percent WAPE against the available
references. Restricting the check to five comparable-area developments with
independently documented counts gives 14.3 percent WAPE. Across all nine
documented developments, including the two known partial-area anomalies, WAPE
is 23.5 percent. These are encouraging but small validation samples.

Four developments remain unsuitable for a floor-area model even though their
unit counts are supported: Hidden Timber and SoNA have clearly partial WCAD
living-area records, while Balcones Club and Terrazzo have cross-county area
overlap or incompatible county measures. The recommended next integration is
therefore simple and conditional: use the main-area stratified estimate only
for comparable-area projects, retain source-backed counts where available, and
send the four structural exceptions to explicit fallback/review. Applying the
original Travis total-improvement-area rules wholesale to Williamson is not
recommended.

## Validated integration and promotion

`scripts/data/unit_counts/build_integration.R` applies the validated hierarchy first to
the 892 strict selected direct projects and then to the 853 disjoint unresolved
candidates:

- 892 projects use strict selected direct totals, totaling 116,434 units;
- 9 projects use documented counts, totaling 2,346 units;
- 14 comparable Williamson projects use the validated main-area stratified
  estimate, totaling 3,705 units;
- 767 other in-domain projects use that estimate, totaling 129,915 units;
- 62 review or out-of-domain projects retain 8,478 targeted units; and
- one out-of-domain project explicitly retains its current zero-unit fallback.

The direct-project layer rises from 96,141 targeted parcel units to 116,434
selected units, a 20,293-unit correction. The unresolved-candidate subtotal
falls from 169,593 targeted units to 144,444 shadow units. Together, those
changes reduce the full regional parcel total from 519,504 to 514,648 units, a
0.9 percent change. Hays is unchanged; Travis accounts for 4,794 fewer units
and Williamson for 62 fewer units. The hierarchy changes 2,075 parcel rows and
967 mapped hex totals.

The pre-promotion unit-threshold comparison gained 56 eligible hexes and lost
3. After all seven clustering domains were required, the candidate solution
contained 3,261 assigned hexes versus 3,212 under the prior surface. Among
3,209 hexes classified in both scenarios, 96.9 percent retained the same
aligned cluster and the adjusted Rand index was 0.946. The candidate classified
about 92.1 percent of allocated population, compared with 91.0 percent before
promotion.

These results supported promotion of the conditional hierarchy on July 28,
2026. `02v` preserves the original targeted count and selection provenance on
every parcel, archives the pre-promotion canonical analytical outputs, and
writes the promoted parcel table consumed by `02c`.

The rebuilt canonical amenity solution now contains 3,261 hexes. At the
substantively selected k=6, repeated-subsample stability is 0.974 and the
smallest cluster contains 173 hexes. Silhouette still favors k=3 and gap
statistics favor larger solutions, so k=6 remains an explicit substantive
choice rather than an automatic optimum.

The subsequent populated zero-unit audit now verifies direct counts at the
project level and finds zero omitted strict direct projects. It leaves 293
populated zero-unit hexes containing 29,822 allocated residents, or 3.07 percent
of allocated population. Most of that residual reflects ACS point-fallback
allocation rather than positive independent parcel-unit evidence. The
certified-roll supplement reduces the full-residential-proxy gap count to zero.
The ten former multifamily-signal hexes are now documented as TCAD land-only
records whose signal came from zoning rather than current improvements. See
the [July 2026 parcel/ACS unit audit](../audits/parcel-acs-unit-audit-2026-07.md)
and `scripts/audits/populated_zero_unit_hexes.R`.

## Outputs

`02d` writes the canonical eligibility artifacts:

- `output/residential_unit_eligibility_audit.rds/.csv`;
- `output/residential_unit_eligibility_exclusions.rds/.csv`;
- `output/residential_unit_eligibility_reviews.rds/.csv`;
- `output/residential_unit_eligibility_qa.csv`; and
- `output/residential_unit_county_exclusion_audit.csv`.

It also writes the certified-roll source repair:

- `output/williamson_certified_residential_supplement.csv`;
- `output/williamson_certified_residential_supplement_audit.csv`; and
- `output/williamson_certified_residential_supplement_summary.csv`.

`02p` writes:

- `output/residential_parcels_unit_source_attributes.rds`;
- `output/residential_unit_source_records.rds/.csv`;
- `output/residential_unit_source_parcel_links.rds/.csv`;
- `output/residential_unit_source_qa.csv`;
- `output/residential_unit_county_classification_qa.csv`;
- `output/residential_unit_source_manifest.csv`; and
- `output/residential_unit_unmatched_source_records.csv`.

`02q` writes:

- `output/residential_unit_project_membership.rds/.csv`;
- `output/residential_unit_projects.rds/.csv`;
- `output/residential_unit_training_table.rds/.csv`;
- `output/residential_unit_model_candidates.rds/.csv`;
- `output/residential_unit_project_source_comparison.csv`;
- `output/residential_unit_source_conflicts.csv`;
- `output/residential_unit_cross_county_projects.csv`;
- `output/residential_unit_reviewed_group_audit.csv`;
- `output/residential_unit_excluded_projects.csv`; and
- `output/residential_unit_project_qa.csv`.

`02r` writes:

- `output/residential_unit_count_models.rds`;
- `output/residential_unit_model_cv_predictions.rds/.csv`;
- `output/residential_unit_model_predictions.rds/.csv`;
- fold, pooled, size-band, source, interval, and recommendation diagnostics;
- URO sensitivity diagnostics overall and by county-transfer status;
- model-domain, review, integration-scenario, and monotonicity QA tables; and
- `figures/02r_unit_model_cv_observed_predicted.png` and
  `figures/02r_unit_model_validation_wape.png`.

`02s` writes:

- `output/residential_unit_williamson_validation.rds/.csv`;
- `output/residential_unit_williamson_candidate_audit.csv`;
- `output/residential_unit_williamson_manual_sources.csv`;
- `output/residential_unit_williamson_official_matches.csv`;
- `output/residential_unit_williamson_source_coverage.csv`;
- `output/residential_unit_williamson_measurement_qa.csv`;
- `output/residential_unit_williamson_strategy_comparison.csv`;
- `output/residential_unit_williamson_manual_review.csv`;
- `output/residential_unit_main_area_model_cv_predictions.csv`; and
- `output/residential_unit_floor_area_cv_comparison.csv`.

`02t` writes:

- `output/residential_parcels_unit_shadow_integrated.rds`;
- `output/corporate_ownership_by_hex_unit_shadow.rds`;
- project-selection and parcel-allocation audit tables;
- allocation QA and strategy/county comparisons; and
- hex-level unit and eligibility comparisons.

`02v` writes:

- `output/residential_parcels_unit_promoted.rds`;
- `output/residential_unit_promotion_manifest.csv`; and
- one-time pre-promotion canonical artifacts under
  `output/pre_unit_model_promotion/`.

After `03_feature_engineering.R` creates `hex_features_unit_shadow.rds`, `03f`
writes:

- `output/unit_shadow_cluster_comparison.rds`;
- `output/unit_shadow_cluster_assignments.csv`;
- aligned-label, transition, population-coverage, and profile tables; and
- `output/unit_shadow_cluster_metrics.csv`.

`02u` then writes the populated zero-unit residual audit:

- `output/populated_zero_unit_hex_audit.rds/.csv`;
- category, jurisdiction, transition, and summary tables;
- direct-project, unit-parcel, exclusion, and full-parcel review tables; and
- `figures/02u_populated_zero_unit_hex_audit.png`.
