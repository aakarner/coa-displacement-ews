# Residential Unit Count Modeling

## Purpose

The residential unit workflow separates source evidence, project construction,
model training, and production integration. This prevents a project total from
being repeated on every parcel account and allows disagreements to remain
visible.

WCAD residential eligibility is now a production preprocessing rule.
`02d_calibrate_parcel_units.R` applies it before unit calibration and writes
excluded and review records separately. Scripts `02p` and `02q` remain the
source-modeling shadow stage: they do not replace calibrated units with the
source hierarchy or future model predictions.

The EWS repository owns this definition. It reads local WCAD raw files through
`R/wcad_unit_eligibility.R`; it does not import or execute `landlord-mapper`
code. County CSV exports are treated as broad candidate inputs under a
documented schema, not as authoritative residential classifications.

## Source hierarchy

`02d` first joins the compact WCAD legal, use, unit, and property-type fields to
the Williamson candidate parcels. It writes canonical eligibility audit,
exclusion, review, and QA outputs before excluded rows can enter calibration,
corporate aggregation, ACS allocation, or feature engineering.

`02p_prepare_unit_sources.R` then stores each project or account count once in
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

After production eligibility filtering, the source hierarchy produces:

- 236,897 eligible parcel rows grouped into 205,899 project records;
- 264 excluded candidate parcels retained outside the unit universe;
- 868 strict, model-eligible multifamily projects;
- 855 unique unresolved multifamily model candidates; and
- 141 direct-source or direct-versus-account conflicts held out for review.

The mutually exclusive selected hierarchy currently contains:

- 116,412 units from 890 strict direct project totals;
- 230,488 units from 188,467 deterministic appraisal-account projects; and
- 10,384 units from 10,384 WCAD single-unit-rule projects.

Together these shadow selections account for 357,284 units. Only the first
116,412 are direct reported project totals; the appraisal and single-unit
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
  96 small-multifamily legal descriptions supply 194 units; and 10,384
  ordinary residential accounts receive the separate one-unit rule.
- Williamson has 25 apartment model candidates. Nineteen have URO sensitivity
  counts and six currently have no external project total. Two of the 25 are
  cross-county projects also represented in TCAD.
- Williamson excludes 202 nonresidential condominium accounts, 55
  reference-only common-interest accounts, four park/amenity parcels, two
  transitional-commercial land records, and one other nonresidential account.
- One record remains under manual review: a 1,886-square-foot residential lot
  whose DBA says `SPRINGWOODS APTS` but whose appraisal and legal fields do not
  establish a unit count.

The 264 excluded parcels carried 1,344.823 units under the former production
rules, including 1,275.6 corporate-owned units. The integrated eligibility rule
now removes them before calibration. A full refresh produces 534,966 primary
units and 519,533 targeted units regionwide; the H3 parcel/ACS audit contains
514,292 targeted parcel units versus 477,406 ACS housing units. The complete
refresh also updated a pre-existing Travis CoStar match: a six-unit record whose
address says 308 E 34th but whose geocode falls near 308 W 34th is now held as a
direct-versus-account conflict. This explains the change from 869 to 868
training projects; all 855 model-candidate project IDs are unchanged.

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

`02d` writes the canonical eligibility artifacts:

- `output/residential_unit_eligibility_audit.rds/.csv`;
- `output/residential_unit_eligibility_exclusions.rds/.csv`;
- `output/residential_unit_eligibility_reviews.rds/.csv`;
- `output/residential_unit_eligibility_qa.csv`; and
- `output/residential_unit_county_exclusion_audit.csv`.

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
- `output/residential_unit_excluded_projects.csv`; and
- `output/residential_unit_project_qa.csv`.
