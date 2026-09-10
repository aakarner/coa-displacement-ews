# Retrospective amenity snapshots for Part 2

## Scope and interpretation

This stage constructs the amenity feature for April 1, 2025 and April 1, 2026.
It implements the agreed retrospective, one-year data-vintage comparison. It
does not build the full feature-readiness table, estimate clusters, or resume
Part 3 modeling. Existing Part 1 outputs remain separate and unchanged.

The feature measures selected **opening events**, using sales-tax first-sale
dates as proxies. It does not measure the total stock of amenities, verify that
a business opened its doors on that date, or establish exactly what information
was published by either cutoff. Businesses subsequently recorded as closed
still contribute their historical opening events.

| Cutoff | Previous opening window | Recent opening window |
| --- | --- | --- |
| April 1, 2025 | April 2, 2022–October 1, 2023 | October 2, 2023–April 1, 2025 |
| April 1, 2026 | April 2, 2023–October 1, 2024 | October 2, 2024–April 1, 2026 |

## Sources and reconciliation

[`config/amenity_historical_sources.json`](../../config/amenity_historical_sources.json)
pins the sources, normalized input paths, retrieval timestamps, dimensions and
SHA-256 checksums:

- February 8, 2025 Internet Archive capture of the official Texas Comptroller
  sales-tax location export, with source modification February 3, 2025. The full
  downloaded export matches the previously audited checksum exactly.
- September 8, 2026 live API extract, with source modification September 7, 2026.
  It includes all retained records for the three core NAICS categories in Hays,
  Travis and Williamson counties, without an event-date filter.

The same reconstructed event ledger supplies both cutoffs. Records match on
taxpayer number plus location number. When an ID occurs in both sources, the
archived record is retained as a coherent record; live-only IDs supplement it.
This preserves older establishments and avoids silently moving, renaming or
retiming historical openings using a later version of an existing ID. Every
shared-ID field difference is exported for audit.

Archive-first is an explicit precedence rule, not a claim that archived values
are invariably more accurate. It deliberately does not incorporate later
corrections to shared records. A missing live ID does not prove closure: the
live query is limited by retention, county and NAICS coverage. Closure flags in
the event QA reflect the selected source, not independently verified operation
at the cutoff; they do not control index eligibility.

After applying each cutoff's eligibility rules, exact duplicate opening proxies
under different IDs are counted once. This requires the same business name,
full street including unit, ZIP, NAICS and first-sale date. Different names,
units or dates are not merged; incomplete fingerprints remain separate. The
representative is chosen deterministically: archived evidence first, then a
known/earlier permit date, then stable ID. All aliases remain in audit outputs.
This additional cleaning is applied identically to both Part 2 vintages and
does not rewrite the original Part 1 event tables.

## Classification, geocoding and scoring

[`R/amenity_classification.R`](../../R/amenity_classification.R) supplies the
same sales-tax classification function to Part 1 and Part 2. The categories and
exclusions remain in [`config/amenity_categories.csv`](../../config/amenity_categories.csv):
drinking places, full-service restaurants and name-filtered cafes, with the
existing home-business and institutional exclusions. First-sale dates must be
inside the appropriate window; known permit dates must not exceed its cutoff.
A missing permit date remains eligible under the existing rule and is counted
in date QA. Permit eligibility is not proof of publication timing.

The existing Census-then-ArcGIS geocoding cascade is reused, including the
ArcGIS score threshold of 90. Original street text is preserved; mechanical
cleanup removes a duplicated leading street number or a literal missing-number
`NA` prefix. It does not infer another street or lower the match threshold.
A separate Part 2 cache is initially seeded from the Part 1 cache. Unmatched
records stay in an all-event audit, with spatial scope marked unknown rather
than outside Austin.

Both vintages use the same 7,027-cell grid and projected hex centroids. Openings
within 800 meters receive linearly declining distance weights. Each category
combines recent weighted openings with the positive portion of the change from
the previous window; the three category scores are equally weighted.

The six component transformations use the earlier vintage's 1st/99th percentile
bounds, fitted on the fixed grid and **held unchanged for the later vintage**.
Raw exposures, bounds and clipping QA are saved. This prevents rescaling each
year from introducing an additional source of apparent change. These are
amenity-component bounds, not the later full-model standardization or centroids.

Mixed-beverage and food-inspection corroboration is not rebuilt in this scoped
stage. Those sources do not determine opening eligibility or the index; their
confirmation flags remain explicitly unknown, not negative matches.

## Coverage contract

Each candidate and feature artifact explicitly records:

- `amenity_retrospective_usable = TRUE` after event and geocoding QA;
- `amenity_window_complete = NA`, because reconstruction cannot prove exhaustive
  historical coverage; and
- `amenity_coverage_status = retrospective_reconstructed_completeness_unverified`.

This distinguishes a usable proof-of-concept reconstruction from a claim of a
complete historical census. The later readiness table should read the explicit
retrospective contract instead of treating an unknown completeness flag as zero
amenities. Geocoding rates use all eligible three-county events; they are not
City-of-Austin-only match rates. Events outside the city may contribute when
they lie within the established 800-meter access radius.

## Reproduction

From the repository root, verify/reuse the pinned source files (or download
them if absent):

```bash
python3 scripts/data/prepare_historical_amenity_sources.py \
  --live-id texas_sales_tax_live_20260908T102000Z
Rscript scripts/part2/build_amenity_snapshots.R
```

An archived input can be recovered by its pinned checksum. The live endpoint is
mutable: an exact rerun requires the preserved raw/normalized snapshot, not a
new download that merely uses the same filename. The R runner checks the
registry's normalized checksums and rejects changed inputs.

The runner accepts an alternate pinned manifest path as its sole argument.
`EWS_AMENITY_PREPARE_ONLY=true` builds candidate and reconciliation tables without
geocoding. Network access is required only for missing raw sources or newly
encountered business addresses. Existing completed source directories are
verified, never refreshed in place.

```bash
Rscript tests/test_amenity_classification.R
Rscript tests/test_amenity_scoring.R
Rscript tests/test_amenity_snapshot_outputs.R
```

## Outputs

All new generated artifacts live under ignored `output/part2/amenities/`:

- `amenity_features_paired.rds/.csv`: one row per hex and cutoff;
- `amenity_feature_changes_by_hex.csv`: paired feature values and differences;
- `amenity_snapshot_summary.csv` and `amenity_paired_spatial_qa.csv`: event and
  spatial coverage summaries, not the full workflow readiness table;
- reconciliation, field-change, cross-ID and paired event-membership audits;
- date-specific directories with candidates, all-event geocoding audits,
  feature tables, source/alias/address-cleaning QA and scoring references; and
- `amenity_run_manifest.json`: source registry hash, code and taxonomy hashes,
  grid hash, cutoffs, method label and generated artifact hashes.

Raw snapshots and per-source manifests remain in ignored
`data/raw_amenities/historical/`. Preserve them with the project's data backups;
they are not stored in Git. The tracked registry and preparation code document
their provenance. The dated execution results are in the
[`September 2026 processing audit`](../audits/amenity-historical-processing-2026-09.md).
