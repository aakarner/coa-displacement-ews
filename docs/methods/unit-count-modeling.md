# Residential Unit Count Modeling

## Why Unit Counts Must Be Estimated

The Early Warning System (EWS) needs an estimate of the number of dwellings in
each hex. Unit counts determine whether a hex has enough housing to enter the
cluster analysis, form denominators for rates such as eviction filings per 100
units, measure the share of housing under corporate ownership, and help
allocate Census and American Community Survey (ACS) estimates from larger
geographies to the hex grid.

County appraisal records are the closest available source to a census of
property, but they do not report housing units consistently. One appraisal row
may represent a house, an individual condominium, an apartment parcel, a
common-area account, or vacant land. A multifamily development may span
several parcels, and a unit total may be repeated on more than one appraisal
record. Treating every row as one dwelling would undercount many apartment
properties and overcount some condominium and master accounts.

This workflow reconstructs a defensible count while preserving how each number
was obtained. Reported project totals, appraisal-code rules, one-unit rules,
and model estimates remain distinguishable. Conflicting sources are not
silently averaged, and the parcel totals are not forced to equal ACS survey
estimates.

## Key Terms

| Term | Meaning in this workflow |
| --- | --- |
| Residential unit or housing unit | One dwelling intended for one household. A building may contain one unit or many units. |
| Parcel | A mapped piece of land. Appraisal data often use one row per tax account, so a row is not always equivalent to a parcel or dwelling. |
| Appraisal account | A county record created for property-tax administration. Several accounts may describe one physical property. |
| Project or development | One real-world residential property after related parcel and account records have been grouped. |
| Source evidence | Any reported field, inventory record, legal description, or reviewed document that helps determine a unit count. |
| Direct count | A project total reported by a source judged reliable enough for the stated use. |
| Strict direct count | A direct count that also passes the source-agreement, plausibility, and project-linkage checks required for selection. |
| Deterministic count | A count produced by an explicit appraisal rule, such as two units for a duplex code or one unit for each condominium account. |
| Rule-based estimate | A count assigned by a documented assumption when no direct count exists, such as the Williamson one-unit rule. |
| Model estimate | A prediction based on floor area and building characteristics learned from projects with known counts. |
| Training project | A project with a reliable count and usable predictors that can teach or test the model. |
| Model candidate | An unresolved multifamily project for which the model might supply a count. |
| Project construction | The process of deciding which parcel and appraisal-account rows belong to the same physical development. |
| Model training | Fitting a relationship between known unit counts and property characteristics, then testing it on held-out projects. |
| Production integration | Selecting one count for each project, allocating it back to parcel records, and making the resulting surface available to the main pipeline. |
| Unit-count surface | The parcel-level table containing the selected count and its source for every eligible residential record. |
| Primary surface | The higher pre-model parcel calibration retained for comparison. |
| Conservative surface | A deliberately cautious lower pre-model alternative retained for comparison. |
| Targeted baseline | The earlier parcel-linked estimate retained as a comparison and fallback during model integration. |
| Shadow surface | A candidate unit-count surface evaluated beside the current pipeline before it is adopted. |
| Promoted or canonical surface | The reviewed surface currently used by downstream EWS analysis. |

## Workflow Overview

The workflow proceeds in six substantive steps. The names in parentheses are
the corresponding `{targets}` pipeline steps.

1. **Identify eligible residential records** (`unit_calibration`). Remove
   appraisal accounts that represent nonresidential condominiums, common
   areas, parks, vacant transitional land, or other non-housing property.
2. **Collect unit-count evidence** (`unit_sources`). Extract reported counts,
   appraisal fields, legal descriptions, and external inventory records while
   keeping source totals separate from parcel links.
3. **Construct physical projects** (`unit_projects`). Group related parcel and
   appraisal-account rows so each development and each source total is counted
   once.
4. **Estimate unresolved projects** (`unit_models`). Compare floor-area models
   using held-out projects and identify candidates that fall within the
   model's supported range.
5. **Test transfer to Williamson County** (`williamson_validation`). Confirm
   that the selected floor-area definition and model relationship remain
   reasonable when applied to WCAD records.
6. **Integrate and promote one hierarchy** (`unit_integration` and
   `promoted_unit_surface`). Prefer reliable observed counts, use the validated
   model only where appropriate, retain explicit fallbacks, and write the
   parcel surface used by the rest of the EWS.

The EWS repository owns these rules. It reads local Williamson County
Appraisal District (WCAD) files through `R/wcad_unit_eligibility.R` and Travis
County Appraisal District (TCAD) files through the configured local inputs. It
does not import or execute code from the sibling `landlord-mapper` repository.
County CSV exports are broad candidate inputs, not automatically authoritative
lists of residential property.

## How Source Evidence Is Ranked

The eligibility step first joins WCAD legal descriptions, use codes, unit
fields, and property types to the Williamson candidate parcels. Exclusion,
manual-review, and quality-control tables are written before any excluded
record can affect unit calibration, corporate-ownership summaries, ACS
allocation, or clustering features.

`scripts/data/unit_counts/prepare_sources.R` stores each reported project or
account count once in `output/residential_unit_source_records.rds`. A separate
table, `output/residential_unit_source_parcel_links.rds`, records which parcels
belong to that source record. This separation prevents a project total from
being repeated on every linked parcel.

Evidence is selected in the following order:

1. **Reported project counts.** Completed City affordable-housing project
   totals, high-confidence CoStar totals, and plausible TCAD `imprvUnits`
   values for B1 multifamily properties can serve as direct counts. A subset
   with complete model inputs also supplies the known outcomes used to train
   and validate the floor-area model.
2. **Counts implied by appraisal codes or complete account enumeration.** TCAD
   assigns one unit to A1-A4 records, two to B2, three to B3, and four to B4.
   Hays uses corresponding state-code rules. WCAD contributes explicit
   residential condominium accounts and duplex, triplex, or fourplex legal
   descriptions. These are transparent deterministic counts, but they are not
   treated as reported project totals for model training.
3. **The separate WCAD one-unit rule.** A positive-area residential account
   receives one unit when it contains no condominium, small-multifamily, or
   apartment signal. This is a documented assumption, not direct evidence.
4. **City URO estimates.** The Universal Recycling Ordinance inventory contains
   useful multifamily unit estimates. Because the City describes them as
   estimates, they are used for sensitivity and external validation rather
   than as authoritative training outcomes.
5. **Floor-area model estimates.** These are considered only for multifamily
   projects still unresolved after the preceding evidence.

A lower-priority estimate never overwrites stronger evidence. When two direct
sources disagree beyond the allowed tolerance, the project remains unresolved
and is sent to review.

The Williamson cleanup also distinguishes current property evidence from
historical text. Legal descriptions, doing-business-as (DBA) names, and use
descriptions can indicate an apartment property. An incidental `APT` in an
owner's former mailing address cannot. A comment-only apartment signal is
accepted only for the reviewed C3/C5 property classes where that evidence
identifies a documented apartment development.

The following WCAD records remain visible in audit tables but are excluded from
the eligible residential parcel set:

- `REFERENCE ONLY` condominium common-interest or master accounts;
- nonresidential condominium units identified by WCAD property and use codes;
- park or amenity parcels with no residential building area;
- transitional-commercial land with no building area; and
- other explicitly nonresidential accounts carried into the upstream extract.

## How Parcel Records Become Projects

`scripts/data/unit_counts/build_projects.R` groups records only when they share:

- a high-confidence source record; or
- the same normalized address within 250 meters and at least one multifamily
  signal.

Address normalization removes superficial formatting differences so addresses
can be compared. The 250-meter condition guards against joining identically
named or numbered properties in different places. Owner name alone is never
enough because one owner may hold several unrelated developments.

Complete condominium and small-multifamily account groups are summed because
each account represents part of the total. In contrast, a TCAD project total
repeated on several B1 parcels is treated as ambiguous because summing it would
multiply the development's units.

Direct sources must agree within 20 percent. A complete account enumeration is
also used as an independent check. Sources that fail this comparison are not
averaged; the disagreement remains visible for review.

Two apartment properties at the Travis-Williamson boundary have parcel
records in both appraisal districts. They remain unified when a shared,
high-confidence project source establishes that they are the same physical
property. County quality-control tables identify the project in both counties,
but regional totals count the project and its units once.

The model uses **main/living floor area**, meaning space intended for ordinary
residential use, rather than total improvement area that may include parking,
storage, or other structures. When both counties describe the same
cross-county development, the project uses the larger county-specific
main/living-area total instead of adding duplicate representations. The raw
values and the selection rule remain in the project table. Reviewed companion
accounts for Lantern at Westwood and Lakeline Crossing are joined before model
candidates are created.

## Current Evidence Coverage

After nonresidential and reference records are removed, the workflow contains:

- 237,305 eligible parcel rows grouped into 206,305 project records;
- 264 excluded candidate parcels retained for audit but not counted as
  housing;
- 892 projects with selected direct totals, 874 of which have the complete,
  consistent data needed for model training;
- 853 unique unresolved multifamily model candidates; and
- 141 direct-source or direct-versus-account conflicts held out for review.

Before the unresolved model candidates are filled, the mutually exclusive
evidence tiers account for:

- 116,434 units from 892 strict direct project totals;
- 230,518 units from 188,497 deterministic appraisal-account projects; and
- 10,762 units from 10,762 WCAD single-unit-rule projects.

Together, these tiers account for 357,714 units. This is not one uniform
confidence class. Only 116,434 are direct reported project totals; 230,518
come from explicit appraisal rules, and 10,762 come from the separate one-unit
assumption. Reports should retain those distinctions.

All current projects eligible to train the model are in Travis County. The
model therefore cannot estimate a separate county effect from the available
training data. County is instead used to flag transfer and validation risk. A
county-specific coefficient should be considered only after reliable training
counts are obtained in Hays and Williamson.

The current county rules and coverage are:

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
  whose DBA is `SPRINGWOODS APTS` but whose appraisal and legal fields do not
  establish whether it contains housing or how many units it has.

An audit of populated hexes with zero parcel units exposed a source-version
mismatch in Williamson. The current certified roll contained 407 active,
uniquely addressed residential records with positive living area that were
inside the study grid but absent from the broad input. One additional active
property was linked to the mapped shape of a reviewed legacy parcel. Target
`unit_calibration` adds these 408 records through
`R/wcad_residential_supplement.R`. The supplement includes 30 explicit
residential condominium accounts, 378 ordinary one-unit records, and 16
corporate-owned parcels. It resolves all initially identified WCAD geometry
and source gaps.

The 264 excluded parcels had carried about 1,345 units under the former rules,
including about 1,276 corporate-owned units. They are now removed before
calibration. A pre-model refresh produced 534,861 primary units and 519,504
targeted-baseline units regionwide. These are retained as comparison surfaces;
the promoted result is described under
[Integration Results](#integration-results).

The refresh also excludes one Travis CoStar record from model training because
its listed address is on East 34th Street while its coordinates fall near the
corresponding West 34th Street address. This prevents a likely location
mismatch from becoming a training outcome.

Affordable Housing Inventory records require special care. Some completed
records describe a subsidized subset within a larger condominium property.
Complete account enumerations and other direct sources are therefore used to
detect, rather than silently accept, those mismatches.

## How the Floor-Area Models Were Compared

`scripts/data/unit_counts/fit_models.R` compares four methods:

1. **Global ratio.** Estimate one regional square-feet-per-unit value from the
   training projects and apply it to every candidate.
2. **Stratified ratio.** Estimate separate square-feet-per-unit values for
   groups defined by project floor area and number of stories. This allows a
   small low-rise development and a large apartment complex to have different
   typical unit sizes.
3. **Negative-binomial generalized additive model (GAM).** Fit a flexible count
   regression that relates unit totals to floor area and other building
   characteristics.
4. **Monotonic gradient boosting.** Combine many constrained prediction trees
   while requiring predicted units not to decrease as residential floor area
   increases. This is retained as a higher-complexity benchmark.

The comparison uses **cross-validation**. Training projects are divided into
groups called folds; each fold is predicted by a model fit without that fold.
This tests performance on projects the model did not see during fitting. The
workflow repeats the test with five general project folds, five spatial
holdouts, and three source holdouts to check sensitivity to location and source
type.

No demographic, ACS, URO, or displacement-outcome variables enter model
training. This prevents the unit model from borrowing information from the
variables it will later help construct or analyze.

The main error measure is **weighted absolute percentage error (WAPE)**: the
sum of absolute unit-count errors divided by the sum of observed units. Lower
values are better. **Bias** reports whether predictions are systematically too
high or too low in total. Current project-fold results are:

| Method | Pooled WAPE | Median fold WAPE | Large-project WAPE |
|---|---:|---:|---:|
| Global ratio | 27.2% | 27.0% | 26.4% |
| Stratified ratio | 17.5% | 17.3% | 15.0% |
| Negative-binomial GAM | 20.6% | 20.2% | 18.1% |
| Monotonic boosting | 16.5% | 16.3% | 13.1% |

The stratified ratio is the selected method. It is easy to inspect, improves
median fold WAPE by 36.1 percent relative to the global ratio, keeps maximum
absolute fold bias below 3.3 percent, and keeps bias within each project-size
group below the predeclared 20 percent threshold. Its empirical 80 percent
prediction intervals, which are ranges intended to contain about 80 percent of
true project counts, contain 79.3 percent of held-out counts.

Boosting has slightly lower overall WAPE and near-zero total bias, but its
errors are less balanced across project sizes: size-band bias reaches 24.4
percent. It remains a benchmark pending better calibration for small projects.
The GAM tends to overpredict, fails the fold-bias criterion, and performs worse
than boosting on large developments. The simpler stratified method therefore
offers the best current balance of accuracy, transparency, and consistent
performance.

Across all 853 unresolved candidates, the stratified method predicts 146,107
units. The primary and conservative pre-model alternatives contain 184,896 and
116,326 units, respectively. Of those candidates, 767 projects containing
129,915 predicted units pass the **model-domain screen**, meaning their core
inputs fall within the conditions represented in training.

The remaining projects receive review flags. These include 23 touching
Williamson County, 57 outside at least one training range, 23 missing a core
predictor, and 41 with predictions below the five-unit training scope. These
categories overlap. A flag does not mean the property has zero units; it means
the model should not automatically replace the existing estimate.

Before promotion, applying the model to the 767 supported projects and
retaining the primary estimates for the 86 review projects produced 147,556
candidate units. Retaining conservative estimates for the review group
produced 142,281. These two totals were comparison scenarios, not the final
integrated surface.

The model was also compared with City URO estimates that were not used for
training. Across 409 candidate projects with URO estimates, the main-area
stratified ratio has 15.7 percent WAPE, compared with 17.3 percent for boosting
and 46.7 percent for the conservative parcel estimate. URO remains an
independent sensitivity source rather than an authoritative count.

## Williamson Validation and Comparable Floor Area

Travis and Williamson appraisal systems do not define building-area fields in
exactly the same way. `scripts/data/unit_counts/validate_williamson.R`
therefore tests whether the selected relationship can transfer before the
Travis-trained model is used on Williamson projects.

The audit covers 23 candidate developments touching Williamson. Companion
accounts for Lantern at Westwood and Lakeline Crossing are grouped before
validation, so each physical development enters once. Two projects also cross
the county line and have TCAD and WCAD records for the same property; their
areas are not added as though they described different buildings.

All 23 developments have an external reference: 19 have City URO estimates,
and documented property sources cover the other four. Nine have a documented
count in addition to or instead of URO. Austin Energy Green Building and Texas
Department of Housing and Community Affairs inventories produce no exact
matches. That means the developments are absent from those limited program
inventories, not that they contain zero units. The HUD multifamily service
timed out and supplies neither positive nor negative evidence.

The central measurement result is that TCAD `main_area` and WCAD
`TotalSqFtLivingArea`, both intended to represent usable living space, are much
more comparable than each county's total improvement area. On the existing
project-grouped folds:

| Floor-area definition | Ratio method | Pooled WAPE | Bias |
|---|---|---:|---:|
| Main/living area | Stratified ratio | 17.5% | 0.2% |
| Total improvement area | Stratified ratio | 24.1% | -0.3% |
| Main/living area | Global ratio | 27.2% | 15.5% |
| Total improvement area | Global ratio | 39.4% | 10.5% |

For the 19 Williamson-only developments with plausible main/living-area
records, the stratified estimate has 9.7 percent WAPE against available
references. Among five comparable-area developments with independently
documented counts, WAPE is 14.3 percent. Across all nine documented
developments, including known partial-area problems, WAPE rises to 23.5
percent. These results support cautious transfer, but the validation samples
are small.

Four developments remain unsuitable for a floor-area model even though their
unit counts are supported: Hidden Timber and SoNA have clearly partial WCAD
living-area records, while Balcones Club and Terrazzo have cross-county area
overlap or incompatible county measures. The integration therefore uses the
main-area stratified estimate only for comparable projects, preserves
source-backed counts where available, and sends structural exceptions to an
explicit fallback. A separate Williamson model is not currently recommended:
the main problem was making the floor-area definitions comparable, not evidence
of a fundamentally different unit-production process.

## Integration Results

`scripts/data/unit_counts/build_integration.R` selects one count for each
project in priority order. The final project-level hierarchy is:

- 892 projects use strict selected direct totals, totaling 116,434 units;
- 9 projects use documented counts, totaling 2,346 units;
- 14 comparable Williamson projects use the validated main-area stratified
  estimate, totaling 3,705 units;
- 767 other projects within the model's supported range use that estimate,
  totaling 129,915 units;
- 62 review projects or projects outside the supported range retain 8,478
  targeted-baseline units; and
- one project outside the supported range explicitly retains its existing
  zero-unit fallback.

Replacing the targeted baseline with selected direct counts adds 20,293 units
to the direct-project layer. Replacing supported unresolved estimates with the
validated model reduces that candidate layer by 25,149 units. The combined
effect reduces the full regional parcel total from 519,504 to 514,648 units, a
0.9 percent change. Hays is unchanged; Travis has 4,794 fewer units and
Williamson has 62 fewer. The selected hierarchy changes 2,075 parcel rows and
933 mapped hex totals.

Changing the unit surface moved 52 hexes above the 20-unit eligibility
threshold and moved three below it. After requiring complete data for all seven
clustering domains, the candidate solution classified 3,261 hexes, compared
with 3,212 under the prior surface. Among the 3,209 hexes classified in both
versions, 96.9 percent retained the same matched cluster. The adjusted Rand
index, a measure of agreement between two cluster assignments, was 0.946 on a
scale where 1 indicates identical assignments.

These results supported promotion of the conditional hierarchy on July 28,
2026. Target `promoted_unit_surface` retains the old targeted count, the new
selected count, and the reason for that selection on every parcel. It archives
the pre-promotion analytical outputs and writes the promoted parcel table used
to calculate corporate-ownership denominators, ACS allocation weights, rate
denominators, and cluster eligibility.

The locked Part 1 solution now classifies 3,261 hexes into six clusters and
covers about 92.1 percent of allocated population. Unit modeling is not the
reason six clusters were selected; it changes which hexes are eligible and the
rate denominators used by several cluster variables.

## Relationship to ACS Housing Estimates

The parcel and ACS totals should be compared, but they should not be forced to
match. The parcel surface combines property records, documented counts, rules,
and model estimates. The ACS is a survey estimate for larger geographies with
published sampling uncertainty. Allocating that estimate to hexes adds further
spatial uncertainty.

In the July 2026 audit, 508,843 promoted parcel units fall on the exact study
grid, compared with 479,614 allocated ACS housing units. The parcel total is
29,229 units, or 6.1 percent, higher. The full regional parcel total of 514,648
is larger because not every source record contributes to the exact mapped-grid
comparison.

Inside the Austin full-purpose boundary, the area under the City's full
municipal jurisdiction, 514,241 promoted parcel units are 0.8 percent below the
2024 one-year ACS city benchmark of 518,574. This close citywide result does
not imply local agreement. At the block-group level, the promoted parcel total
is 11.5 percent above ACS in the audit sample, and 55.2 percent of block groups
fall within the published ACS margin of error (MOE).

The subsequent populated zero-unit audit now verifies direct counts at the
project level and finds zero omitted strict direct projects. It leaves 293
populated zero-unit hexes containing 29,822 allocated residents, or 3.07
percent of allocated population. Most of those residents were placed through
the Census-block point fallback because no independent unit-bearing parcel was
found. That pattern is a warning about small-area allocation; it is not
evidence that 29,822 parcel units should be added.

The certified-roll supplement resolves all previously identified cases where a
full residential parcel proxy was missing. Ten former multifamily-signal hexes
remain at zero because TCAD identifies the records as land-only with no
improvement units or building area; their earlier apartment signal came from
zoning rather than a current structure.

See the
[July 2026 parcel/ACS unit audit](../audits/parcel-acs-unit-audit-2026-07.md)
and `scripts/audits/populated_zero_unit_hexes.R` for the full comparison. ACS
disagreement remains a diagnostic and review tool, not an automatic parcel
replacement.

## Technical Output Reference

The sections below list the main files created at each pipeline step. `.rds`
files preserve R data types for the pipeline; `.csv` files provide a
human-readable tabular version. Files containing `qa`, `audit`, `review`, or
`comparison` are diagnostics rather than main pipeline inputs.

Target `unit_calibration` (`scripts/data/parcel_units_calibrate.R`) writes the
current eligibility files:

- `output/residential_unit_eligibility_audit.rds/.csv`;
- `output/residential_unit_eligibility_exclusions.rds/.csv`;
- `output/residential_unit_eligibility_reviews.rds/.csv`;
- `output/residential_unit_eligibility_qa.csv`; and
- `output/residential_unit_county_exclusion_audit.csv`.

It also writes the certified-roll source repair:

- `output/williamson_certified_residential_supplement.csv`;
- `output/williamson_certified_residential_supplement_audit.csv`; and
- `output/williamson_certified_residential_supplement_summary.csv`.

Target `unit_sources` (`scripts/data/unit_counts/prepare_sources.R`) writes:

- `output/residential_parcels_unit_source_attributes.rds`;
- `output/residential_unit_source_records.rds/.csv`;
- `output/residential_unit_source_parcel_links.rds/.csv`;
- `output/residential_unit_source_qa.csv`;
- `output/residential_unit_county_classification_qa.csv`;
- `output/residential_unit_source_manifest.csv`; and
- `output/residential_unit_unmatched_source_records.csv`.

Target `unit_projects` (`scripts/data/unit_counts/build_projects.R`) writes:

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

Target `unit_models` (`scripts/data/unit_counts/fit_models.R`) writes:

- `output/residential_unit_count_models.rds`;
- `output/residential_unit_model_cv_predictions.rds/.csv`;
- `output/residential_unit_model_predictions.rds/.csv`;
- fold, pooled, size-band, source, interval, and recommendation diagnostics;
- URO sensitivity diagnostics overall and by county-transfer status;
- model-domain, review, integration-scenario, and monotonicity QA tables; and
- `figures/unit_model_cv_observed_predicted.png` and
  `figures/unit_model_validation_wape.png`.

Target `williamson_validation`
(`scripts/data/unit_counts/validate_williamson.R`) writes:

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

Target `unit_integration` (`scripts/data/unit_counts/build_integration.R`)
writes:

- `output/residential_parcels_unit_shadow_integrated.rds`;
- `output/corporate_ownership_by_hex_unit_shadow.rds`;
- project-selection and parcel-allocation audit tables;
- allocation QA and strategy/county comparisons; and
- hex-level unit and eligibility comparisons.

Target `promoted_unit_surface`
(`scripts/data/unit_counts/promote_integration.R`) writes:

- `output/residential_parcels_unit_promoted.rds`;
- `output/residential_unit_promotion_manifest.csv`; and
- one-time copies of the pre-promotion analytical files under
  `output/pre_unit_model_promotion/`.

Before promotion, the manual unit-surface sensitivity workflow used
`output/hex_features_unit_shadow.rds` and
`scripts/audits/unit_surface_clusters.R` to write:

- `output/unit_shadow_cluster_comparison.rds`;
- `output/unit_shadow_cluster_assignments.csv`;
- aligned-label, transition, population-coverage, and profile tables; and
- `output/unit_shadow_cluster_metrics.csv`.

The manual diagnostic `scripts/audits/populated_zero_unit_hexes.R` writes the
populated zero-unit residual audit:

- `output/populated_zero_unit_hex_audit.rds/.csv`;
- category, jurisdiction, transition, and summary tables;
- direct-project, unit-parcel, exclusion, and full-parcel review tables; and
- `figures/populated_zero_unit_hex_audit.png`.
