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

The evidence classes are:

1. Completed City affordable-housing project totals, high-confidence CoStar
   project totals, and plausible TCAD `imprvUnits` on B1 properties. These can
   supply strict model labels.
2. Appraisal-account rules: one unit for A1-A4, two for B2, three for B3, and
   four for B4. These are deterministic counts, not floor-area model labels.
3. City Universal Recycling Ordinance multifamily counts. The City calls these
   estimated counts, so they are retained as sensitivity labels.
4. Future floor-area model predictions for projects still unresolved after the
   preceding sources.

No lower-tier source overwrites a higher-tier source. Source conflicts remain
unclassified pending review.

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

## Current shadow results

The first run produced:

- 237,161 parcel rows grouped into 206,116 residential projects;
- 867 strict, model-eligible multifamily projects;
- 833 unresolved multifamily model candidates representing about 180,000
  current floor-area-estimated units; and
- 140 direct-source or direct-versus-account conflicts held out for review.

All current model-eligible projects are in Travis County. The first model must
therefore use a common regional form without an estimated county effect.
County remains a validation and out-of-domain flag. A county effect should be
considered only after obtaining defensible project labels in Hays and
Williamson.

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
- `output/residential_unit_source_manifest.csv`; and
- `output/residential_unit_unmatched_source_records.csv`.

`02q` writes:

- `output/residential_unit_project_membership.rds/.csv`;
- `output/residential_unit_projects.rds/.csv`;
- `output/residential_unit_training_table.rds/.csv`;
- `output/residential_unit_model_candidates.rds/.csv`;
- `output/residential_unit_project_source_comparison.csv`;
- `output/residential_unit_source_conflicts.csv`; and
- `output/residential_unit_project_qa.csv`.
