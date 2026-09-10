# Handoff: Build Comparable 2024 and 2025 Travis Ownership Snapshots

## Instruction to the receiving agent

Work in the `landlord-mapper` repository at
`/Users/alexkarner/Repositories/landlord-mapper`. Implement and validate the
speed-first historical-ownership work described below. The resulting files will
be consumed by the sibling `coa-displacement-ews` repository for a Part 2
cluster-stability proof of concept.

Complete the work end to end unless an input is unavailable or a source
ambiguity would materially change the ownership definition. Do not commit or
push unless the user explicitly requests it.

## Objective

Produce comparable parcel-level Travis County corporate-ownership snapshots
for tax years 2024 and 2025 using one pinned version of the `landlord-mapper`
classification logic.

The EWS proof of concept will hold its parcel geography and residential-unit
surface fixed. Therefore, this task does **not** need to rebuild the full parcel
universe, redo geocoding, or complete EWS GitHub issue #5. It only needs to
provide reliable annual ownership classifications that can be joined to the
existing EWS parcel IDs.

## Repositories and existing state

- Upstream processing repository:
  `/Users/alexkarner/Repositories/landlord-mapper`
- Consuming analysis repository:
  `/Users/alexkarner/Repositories/coa-displacement-ews`
- Current EWS Travis parcel surface:
  `/Users/alexkarner/Repositories/coa-displacement-ews/data/residential_parcels_for_hex.csv`
- Relevant upstream scripts include:
  `standalone_corporate_parcels.R`, `target_helper_functions.R`, and
  `TCAD_parse.py`.

The `landlord-mapper` working tree already contains user changes and untracked
data/output files. Preserve all of them. In particular, review and retain the
working-tree classifier changes, including the added `LLP` patterns. Do not
reset, discard, or overwrite unrelated work.

Before implementation, record:

- `git status --short --branch`;
- the current branch and `HEAD` commit;
- a hash or saved patch of the relevant uncommitted classifier changes; and
- the exact classifier code used for the two snapshots.

## Source contract

### 2024 source

Use the official TCAD certified appraisal export, supplement 0, rerun dated
August 21, 2024:

`https://traviscad.org/wp-content/largefiles/2024%20Certified%20Appraisal%20Export%20Supp%200_08212024_Rerun.zip`

Use TCAD's accompanying fixed-width layout:

`https://traviscad.org/wp-content/largefiles/Website_Legacy8.0.30-AppraisalExportLayout.zip`

The previously inspected temporary download no longer exists. Download the
source directly into a durable, Git-ignored location under the
`landlord-mapper/data/` tree. Do not use `/tmp` or `/private/tmp` as its only
copy. Record the final URL, retrieval timestamp, byte count, SHA-256 checksum,
ZIP member list, and relevant member sizes.

The archive should contain the large fixed-width `PROP.TXT` property table.
Validate that the records represent tax year 2024 and certified supplement 0
before using them.

### 2025 source

Use the same current 2025 owner source that generated the Travis parcel file
currently consumed by EWS. Prefer existing cached upstream extracts rather than
silently downloading a newer rolling TCAD file. Identify and document the raw
source/capture date and intermediate files used.

If the exact 2025 source underlying the EWS file cannot be established, stop
and report that ambiguity before substituting another vintage.

### Snapshot semantics

For 2024, use the property-year owner as the primary annual owner concept.
Retain January 1 and current-appraisal owner fields when available so their
agreement can be audited. Do not claim that the certified 2024 roll represents
ownership on exactly April 1, 2025.

## Implementation requirements

### 1. Parse the 2024 certified export efficiently

Do not expand the entire archive merely to read `PROP.TXT`; the archive is
roughly 15.8 GB uncompressed. Stream the member from the ZIP and extract only
the required fixed-width fields. It is acceptable to add a small Python parser
using the standard library or an equally memory-safe R implementation.

At minimum, retain the available versions of:

- property/account ID;
- appraisal or tax year;
- supplement number and action;
- property-year owner ID, name, share, and mailing address;
- January 1 owner ID, name, and mailing address;
- current-appraisal owner ID, name, and mailing address;
- confidentiality and address-suppression flags;
- situs address components;
- homestead/exemption evidence needed for owner-occupancy classification; and
- property-use fields needed to check linkage to the fixed residential surface.

Derive field offsets from the official layout and encode them in one documented
schema object. Add a validation that rejects unexpected record lengths, years,
or layouts instead of silently shifting columns.

Keep a row-level parsed owner table as a Git-ignored intermediate. Never commit
raw owner names or mailing addresses as test fixtures; use synthetic records in
tests.

### 2. Apply one classifier to both years

Standardize the 2024 and 2025 owner records into the same intermediate schema,
then run one shared classifier implementation over both vintages.

The classifier must preserve the existing project meanings:

- `is_owner_occupied`: owner mailing address matches the situs address or
  homestead evidence is present;
- `has_financialized_owner`: at least one owner name matches the versioned
  entity/real-estate marker rules; and
- `is_corporate_owned`: residential, not owner-occupied, and has a
  financialized owner.

Use the current fork's marker rules, including uncommitted intended changes,
unless testing identifies a clear defect. Normalize owner names and addresses
identically in both years. Treat missing or suppressed owner evidence as
unknown, not automatically noncorporate.

For parcels with multiple owner rows, retain the row-level evidence and create
one deterministic parcel-level record. Document the aggregation rule. Do not
let an ordering change in `PROP.TXT` alter the result.

### 3. Join to the fixed EWS parcel surface

Normalize TCAD property IDs and join each vintage to the current Travis rows in
the EWS parcel surface. Do not rebuild coordinates, geocode addresses, change
residential eligibility, or recalculate unit counts for this proof of concept.

Retain one row for every target EWS Travis parcel, including unmatched parcels,
with an explicit status such as:

- `matched_classified`;
- `matched_owner_missing`;
- `matched_owner_suppressed`;
- `matched_ambiguous`;
- `source_parcel_not_found`; or
- another documented, mutually exclusive status.

An unmatched or unclassifiable parcel must never receive a default
`is_corporate_owned = FALSE` value.

## Required consumer output

Write a combined Git-ignored CSV at:

`output/historical_ownership/travis_owner_snapshots_2024_2025.csv`

It must contain exactly one row per `tax_year` and target EWS Travis
`parcel_id`, with at least these columns:

| Column | Meaning |
|---|---|
| `source_county` | Constant `Travis` |
| `tax_year` | `2024` or `2025` |
| `parcel_id` | Normalized EWS/TCAD parcel identifier |
| `owner_ids` | Deterministically ordered owner-ID rollup, if available |
| `owner_names` | Deterministically ordered owner-name rollup, if available |
| `n_owner_rows` | Number of source owner rows represented |
| `owner_name_available` | Whether usable owner-name evidence exists |
| `owner_address_available` | Whether usable mailing-address evidence exists |
| `is_owner_occupied` | `TRUE`, `FALSE`, or `NA` when evidence is insufficient |
| `has_financialized_owner` | `TRUE`, `FALSE`, or `NA` when evidence is insufficient |
| `is_corporate_owned` | `TRUE`, `FALSE`, or `NA` when evidence is insufficient |
| `classification_status` | Explicit match/classification status |
| `classification_rule_version` | Stable identifier for the exact shared rules |
| `source_snapshot_id` | Stable source-vintage identifier |
| `source_owner_field` | Owner concept used, such as `property_year_owner` |
| `source_supplement_number` | Certified/supplement version where applicable |

Additional diagnostic columns are welcome, but do not change these meanings.
Keep address fields in a separate ignored intermediate unless the downstream
consumer demonstrably needs them.

Also write:

- `output/historical_ownership/travis_owner_snapshot_qa.csv`;
- `output/historical_ownership/travis_owner_classifier_parity_2025.csv`;
- `output/historical_ownership/travis_owner_snapshot_manifest.json`; and
- a concise tracked methods note explaining how to reproduce the outputs.

## QA and acceptance gates

### Source and parser QA

Report for each source vintage:

- raw rows and unique property IDs;
- rows by tax year and supplement number;
- invalid record-length and parsing failures;
- duplicate property-owner keys;
- parcels with multiple owners;
- missing, blank, confidential, or suppressed owner names and addresses; and
- agreement among property-year, January 1, and current-appraisal owner fields
  where more than one is available.

### EWS linkage QA

Report both parcel-weighted and EWS-unit-weighted:

- match rate;
- usable owner-name coverage;
- complete-classification coverage;
- unmatched and ambiguous counts; and
- coverage by residential property/use category where available.

Write review tables containing the unmatched and ambiguous parcel IDs. Treat
coverage below 95 percent as a blocking result that must be explained rather
than silently accepted. Do not reuse the historical EARS value-match rate as
evidence of owner-record coverage.

### Classifier parity QA

Reclassify the 2025 records with the shared implementation and compare them to
the current EWS parcel flags for:

- `is_owner_occupied`;
- `has_financialized_owner`; and
- `is_corporate_owned`.

Provide confusion matrices, row- and unit-weighted agreement, and a review file
for every discrepancy. Zero unexplained differences is the acceptance
criterion; intentional corrections may remain if they are individually or
categorically documented.

### Tests

Add lightweight automated tests for:

- fixed-width extraction using synthetic lines;
- unexpected record length/year/supplement;
- owner-name and address normalization;
- every supported entity-marker variant, including `LLP`;
- natural-person and missing-owner negative cases;
- homestead and address-match owner occupancy;
- multiple-owner aggregation;
- suppressed/insufficient evidence producing `NA`; and
- deterministic output ordering and repeat runs.

## Non-goals

Do not expand this task to:

- complete EWS GitHub issue #5;
- rebuild the full Travis, Hays, or Williamson residential parcel datasets;
- replace the EWS parcel geography or unit model;
- geocode historical owner or situs addresses;
- infer an exact April 1, 2025 ownership event state;
- construct the Part 2 feature index or clusters;
- redesign corporate-ownership definitions; or
- publish, commit, or transmit raw owner-address data.

## Completion report

When finished, report:

1. Files added or changed and the exact command used to reproduce the run.
2. Raw source URLs, capture semantics, sizes, and SHA-256 checksums.
3. The `landlord-mapper` commit plus the relevant working-tree classifier diff
   used for the run.
4. 2024 and 2025 source row counts and unique property counts.
5. Parcel- and unit-weighted linkage and classification coverage.
6. Corporate-owned parcel and unit totals for each year.
7. The number and nature of 2025 parity differences.
8. Any unresolved source, matching, or semantic limitations.
9. Whether the output is ready for import into `coa-displacement-ews`.

Stop short of Part 2 modeling. The next EWS task will review these outputs and
decide how to construct the frozen 2025 cluster baseline.
