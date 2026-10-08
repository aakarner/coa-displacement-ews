# Residential repair follow-up: staged implementation

> **Subsequent production update — October 7, 2026:** These staged repairs
> have now been incorporated into the expanded-grid production rebuild.
> See the [completed rebuild report](residential-cluster-rebuild-2026-10.md).
> Counts and pending-work language below describe this report's earlier endpoint.

October 7, 2026. These changes are wired into the production pipeline, but **no
production rebuild, cluster fit, map regeneration or full test suite has run**.
The last published/saved production baseline remains property batch 2. All
filing counts below cover April 2, 2025–April 1, 2026.

## Changes and measured staging effects

| Repair | Implemented behavior | Staging result |
| --- | --- | --- |
| Coarse geocodes | Both paired/current and annual builders use one precision rule. Matched status and score ≥90 are necessary but insufficient: Postal, StreetName, POI and unknown types cannot locate a filing in a cell. Address points and street-segment interpolations remain distinct usable categories. | 76 formerly assigned filings become explicitly unassigned (68 Postal, eight StreetName). Eight formerly ambiguous filings become assignable after a coarse candidate is removed. Net mapped total: 11,784 → 11,716. No cells are suppressed. |
| Ocotillo address evidence | A pinned, exact base-address alias links 8000 US-290 W variants to the already-reviewed 308-unit project in cell 596. County, project, parcel, coordinates, unit cell and evidence hashes must match. | Nine additional filings move from cell 630 to 596. Eleven registry address variants match the alias across the longer history. |
| Williamson ownership | Two explicit 2025 reference-account reviews supply source-year ownership evidence while preserving original source rows and certified-only sensitivity outputs. | R577451: about 177.6 units in cell 2325; R605025: 128 units in cell 2516. Both classify successfully for 2025. Each cell has 20 recent filings. Cluster eligibility is not yet recomputed. |
| Oak Ranch homes | Add 791 individually reviewed homes, retain and relocate two existing homes, and link 24 reviewed cases to 17 individual home accounts. Use home-level ownership, never the park owner's classification. | 793 accepted homes across 11 cells; net +791 units. These denominators lift 41 filings above the 20-unit floor, including all 24 original-audit Oak Ranch filings. |

The combined diagnostic changes the broader low-unit residual from **215
filings in 77 cells to 161 filings in 65 cells**. Precision and Ocotillo alone
leave 202 low-unit filings; the home inventory reduces that by 41. Grid units
increase from about **512,415.313 to 513,206.313**. These are counterfactual
counts using the fixed grid and saved measurement baseline, **not newly fitted
Part 1 results**. Adequate denominators do not establish complete ownership,
rent or other clustering inputs.

The precision change also relabels 50 previously unassigned outside-grid/study
cases as insufficient-precision cases; two mixed in/out cases retain only a
reliable outside-grid location. Neither category adds mapped cases. Every
source filing remains in the ledger. The 76 newly unassigned cases and their
original address/geocode evidence are saved as a recovery queue, not declared
resolved residential locations.

## Evidence and safeguards

**Ocotillo.** The existing county account/DBA and Ardent development evidence
in `data/reviewed_unit_properties/batch2_20261007/` identify one project at this
base address. The new rule is in `config/eviction_property_address_reviews.json`.
It does not infer individual apartments, match neighboring house numbers, or
extend to shared Bell phases or mixed Domain addresses. Original case
ambiguity, conflicting reliable property links, county, City and court coverage
checks remain in force. Address matching does not authorize a coarse geocode
to bypass the first-stage precision rule.

**Williamson.** The retained [WCAD 2025 certified roll](https://www.wcad.org/historical-data/)
and [TxGIO 2025 Williamson parcel extract](https://s3.amazonaws.com/data.tnris.org/0fa04328-872e-481c-b453-126a74777593/resources/stratmap25-landparcels_48491_lp.zip)
are pinned in `config/williamson_ownership_sources.json`. Their source dates and
later public-posting limitations are retained there; this is source-year
reconstruction, not a claim that every file was publicly available at the
retrospective cutoff.

- **R577451, Bridge at Balcones:** the certified active account has the clipped
  name `HOUSING AUTHORITY OF THE CITY`. Certified NON-REF links and the same
  owner ID connect reference accounts R081221/R401725; the 2025 GIS reference
  completes the name as `HOUSING AUTHORITY OF THE CITY OF AUSTIN`. Keep the
  active account's certified mailing/situs fields and complete only the name.
- **R605025, Lakeline Station:** the active account is absent from the retained
  2025 extracts. Reference R072533 has the exact 13635 Rutledge Spur address,
  DBA and FC RUTLEDGE HOUSING LP owner. The already-reviewed account crosswalk
  and matching 2024 active/reference records support the association. Apply the
  2025 reference evidence to the active account, with provenance; add no units
  for the reference account.

The reviews, expected original target rows and retained source extracts are
hash-bound by `config/williamson_ownership_reference_reviews.json`. A changed
target, year, evidence file or source configuration fails rather than silently
reusing the review. No 2026 owner is substituted into an earlier year. The
pinned classifier classifies the housing authority as noncorporate and
nonfinancialized, and FC Rutledge as corporate and financialized; this batch
does not alter that taxonomy.

**Oak Ranch.** The accepted inventory is the July 20, 2025 TCAD subdivision
extract and the previously reviewed [City address-point service](https://awgisadaptor.austintexas.gov/awgisago/rest/services/AGO/Address_Points_with_SubAddresses/MapServer).
See `data/reviewed_manufactured_housing/oak_ranch_20261007/README.md` for source
and exception details. New configuration:
`config/manufactured_home_property_reviews.json`; case batch:
`data/reviewed_eviction_properties/oak_ranch_20261007/cases.json`.

Ownership is independently classified using the pinned upstream classifier
and the full-county normalized 2024/2025 owner rows whose hashes are recorded
in the upstream historical-ownership manifest. The compact home-level extracts,
classification outputs and source hashes are retained under `integration/`.
All **793 homes have classified 2025 ownership**. For 2024, **585 classify and
208 remain `source_parcel_not_found`**. The pipeline imports 1,582 new parcel-year
rows for the 791 additions, checks the two existing homes' classifications
against the upstream snapshot, and preserves unknown 2024 flags. Thus Part 1
can use complete 2025 home ownership while historical common support still
reflects the missing 2024 evidence.

The original two homes move from cell 3766 to their individual points in cell
3769. They are not counted twice. The park parent's 377 improvement units are
not added. Source address/space conflicts remain in the retained review files.
The **39 deferred homes** (15 unresolved locations and 24 outside the grid)
remain outside the accepted supplement; no coordinates are snapped or invented.

## Validation and replay

Passed focused checks:

- 38 Williamson reconciliation tests, including the two source-year reviews,
  unchanged certified-only results and rejection of drift/cross-year backfill.
- Geocode precision and paired eviction tests: coarse candidates do not assign
  or suppress cells; precise alternatives resolve; genuine conflicts persist.
- Narrow alias matching, reference drift, overlapping alias and evidence-hash
  rejection; existing case-review identity/coverage safeguards.
- Reviewed-unit and dependency-graph tests: source conservation, expected supplements, no duplicate
  accounts, unchanged other parcels, allocation/geometry drift rejection.
- Oak Ranch staging integration: 793 one-unit homes, +791 units, 24 case links,
  independent ownership vintages and unchanged production-file hashes.

Review-only replays, from the repository root:

```sh
Rscript scripts/audits/oak_ranch_integration.R
Rscript scripts/audits/residential_followup.R
```

Results are in `tmp/residential_followup_20261007/`, especially
`oak_integration_validation.json`, `staging_summary.json`,
`combined_diagnostic.json`, `staged_low_unit_cells.csv`, and
`coarse_geocode_recovery_cases.csv` / `coarse_geocode_recovery_addresses.csv`.
Private case and owner evidence remains local. These scripts do not write
canonical products. The geography replay deliberately uses the saved baseline
crosswalk plus the reviewed additions; the full geometry rebuild and annual /
paired output reconciliation remain part of the final integration run.

The batch-2 endpoint is preserved under
`output/residential_followup_20261007/before/` with source hashes. Its incremental
regression test now reads that endpoint, so later repairs do not erase the
earlier conservation checks. The current reviewed-case output contract expects
126 case-specific reviews and allows only documented precision/alias changes;
that production-output check awaits the rebuild.

At the end of the larger repair batch, rebuild units, ownership, property
geography, paired and annual eviction outputs, measures and clusters in
pipeline dependency order; then run the full tests and refresh the closeout
report. Current products are expected to fail freshness checks until rebuilt.
Demolition-specific issues remain outside this batch.

## Subsequent residual triage: next repair candidates

A read-only review of the staged residual finds **75 filings in 31 zero-unit
cells** and **86 in 34 cells with positive counts below 20**. A below-threshold
cell is not automatically a data error: legitimate small residential inventories
and grid edges can produce these outcomes. However, the following **55 distinct
filings** have concrete local evidence warranting further review. None of the
candidate corrections in this section has been applied.

| Candidate | Recent residual filings | Evidence and next check |
| --- | ---: | --- |
| Five other mobile-home park addresses | 37 | 5701 Johnny Morris Rd (12), 1308 Thornberry Rd (8), 2705 Hoeke Ln (7), 6008 Oleander Trl (6), and 841 Airport Blvd (4). County DBAs identify Pecan Park phases, Capitol View, Village Park, Trails of Oak Hill and Bel Aire. Parent parcels are commercially classified F1. The City recycling inventory lists 103 units at Capitol View and 105 at the Hoeke address (named Pecan Park in that inventory). Reconcile names, active home accounts, individual points and existing units using the Oak Ranch method. These are separate from the 39 deferred Oak Ranch records. |
| 3201 Century Park Blvd | 5 | City inventory: Madison at Wells Branch, 300 units; county DBA: The Olivine. The pipeline already carries about 317.5 modeled units on parcel 549351 in cell 2682, while filings fall in cell 2683. Filing points intersect the named parcel, but the existing unit point does not anchor it in the polygon crosswalk. Verify the property footprint/reference and align numerator and denominator before applying a link. |
| 7301 S IH 35 / The Bennett | 5 | City inventory: 267 units. Improvement account 942518 already carries 271 units, but its ArcGIS-derived reference is in cell 5326, about 18.06 km from the filing points in cell 6789 on parcel 942517. The unit account's retained situs ZIP is 78752; filings say 78744. Resolve the parent/improvement relationship and bad-address/geocode possibility; do not move cases to the existing reference merely because base-address text matches. |
| 8818 Travis Hills Dr / Hudson Miramont | 4 | Filing points intersect parcel 103824. The county profile classifies it B1 and reports 278,740 square feet, but that parcel ID is absent from the promoted unit surface and upstream residential export. Trace active-account links, source selection and coordinates before deciding whether housing is missing or represented by another account. |
| 3220 Feathergrass Ct / Maravilla at the Domain | 4 | City inventory reports 238 units. Filing points intersect commercial parcel 823214, whose DBA is Domain II Shopping Center. Resolve the residential improvement account and housing/facility type within the mixed-use property; do not assign all shopping-center activity to the senior-housing inventory. |

This shortlist accounts for **55/161 (34%)** of remaining low-unit filings.
The other 106 are not certified correct or irreparable; they remain in the
address-level queue. Other promising smaller candidates include McKinney Falls
Apartments (two filings) and apartment/condominium address matches, while some
one-off cases may correctly belong to low-density cells. Repeated filings are
not a count of unique households and must not be used to infer housing units.

Source records: the existing normalized City affordable-housing/recycling and
CoStar inventories in `output/residential_unit_source_records.rds`; county
polygons and account links in `output/property_geography/`; and the retained
upstream 2025 `property_profile.csv`, `property_characteristics.csv`, `situses.csv`
and `residential_parcels_for_hex.csv`. The inventory counts above are review
leads, not newly verified dwelling totals. `scripts/audits/residential_residual_triage.R`
reproduces the candidate joins without changing production. Address groups,
case evidence, inventory/parcel matches and a source-hash summary are saved in
`tmp/residential_residual_triage_20261007/`.

Recommended next batch: verify the three apartment references/omissions first
(Century Park, Bennett and Hudson Miramont), then recover home-level records
for the five parks; review Maravilla's mixed-use/facility relationship
separately. In parallel, audit whether other large residential unit references
fall outside their own verified footprints. This addresses unit-side location
errors that a filing-only review misses. Keep the 20-unit threshold and defer
cluster rebuilding until these data repairs are batched.

## Implemented next batch (staged, October 7)

The apartment/park shortlist above has now been reviewed in
[Apartment references and manufactured-home recovery](residential-residual-repairs-2026-10.md).
This adds 676 homes and repairs The Olivine and The Bennett references. The
combined staged residual falls from 161 filings/65 cells to **108 filings/56
cells**. Hudson Miramont remains unresolved because its parcel crosses the
City boundary; Maravilla remains a separate mixed-use review. Production
outputs and clusters are still unchanged.
