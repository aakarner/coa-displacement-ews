# Residential Unit Count Modeling

## Purpose

The residential unit workflow separates source evidence, project construction,
model training, and production integration. This prevents a project total from
being repeated on every parcel account and allows disagreements to remain
visible.

The current `02p` and `02q` scripts are a shadow stage. They do not change
`units_calibrated`, the ACS allocation, clustering eligibility, features, or
cluster assignments.

## Source hierarchy

`02p_prepare_unit_sources.R` stores each project or account count once in
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

WCAD records explicitly marked `REFERENCE ONLY` are condominium common-interest
or master accounts, not additional dwellings. They remain visible in the audit
tables but are excluded from the number of unit-bearing parcels that an account
enumeration must cover.

## Project construction

`02q_build_residential_projects.R` connects parcels only when they share:

- a high-confidence source record; or
- the same normalized address within 250 meters and at least one multifamily
  signal.

It does not group projects by owner name alone. Complete condominium and
small-multifamily account groups are summed, while repeated TCAD project totals
on multiple B1 parcels are treated as ambiguous.

Direct sources must agree within 20 percent. A complete account enumeration is
also used as a conflict check. The script does not average sources that fail
this test.

Three apartment properties at the Travis-Williamson boundary have parcel
records in both appraisal districts. They remain unified when a shared,
high-confidence project source establishes that they are the same physical
property. Outputs include `project_counties` and `project_cross_county`; county
QA counts these projects once in each involved county, while regional project
and unit totals count them only once.

## Current shadow results

The repaired source hierarchy produces:

- 237,161 parcel rows grouped into 206,162 residential projects;
- 869 strict, model-eligible multifamily projects;
- 855 unique unresolved multifamily model candidates; and
- 140 direct-source or direct-versus-account conflicts held out for review.

The mutually exclusive selected hierarchy currently contains:

- 116,418 units from 891 strict direct project totals;
- 230,486 units from 188,468 deterministic appraisal-account projects; and
- 10,381 units from 10,381 WCAD single-unit-rule projects.

Together these shadow selections account for 357,285 units. Only the first
116,418 are direct reported project totals; the appraisal and single-unit
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
- Williamson: 2,712 explicit residential unit accounts supply 2,712 units;
  96 small-multifamily legal descriptions supply 194 units; and 10,381
  ordinary residential accounts receive the separate one-unit rule.
- Williamson has 25 apartment model candidates. Nineteen have URO sensitivity
  counts and six currently have no external project total. Two of the 25 are
  cross-county projects also represented in TCAD.
- Fifty-five WCAD reference-only common-interest accounts are excluded from
  unit-bearing parcel coverage. Another 213 records remain flagged for source
  review, primarily commercial condominiums carried into the upstream
  residential extract.

Affordable Housing Inventory records require special care. Some completed
records describe a subsidized subset within a larger condominium property.
Complete account enumerations and other direct sources are therefore used to
detect, rather than silently accept, those mismatches.

## Next model stage

The next script should compare:

1. the existing fixed square-feet-per-unit ratio;
2. stratified square-feet-per-unit ratios;
3. a negative-binomial generalized additive model with log residential floor
   area as an offset; and
4. monotonic gradient boosting as a performance benchmark.

Evaluation must use grouped project folds, spatial folds, source holdouts, and
size-band summaries. The GAM should replace the fixed ratio only if it:

- improves median held-out WAPE by approximately 10 percent;
- does not introduce material bias by project size or label source;
- produces reasonably calibrated prediction intervals; and
- is not materially worse than the benchmark on large multifamily projects.

URO estimates should be used for sensitivity validation, not for primary model
training. ACS housing estimates remain independent block-group and tract
validation data; they do not calibrate parcel predictions.

## Outputs

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
- `output/residential_unit_cross_county_projects.csv`; and
- `output/residential_unit_project_qa.csv`.
