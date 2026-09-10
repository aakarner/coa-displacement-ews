# Paired historical ACS inputs for Part 2

This standalone build reconstructs rent pressure and demographic vulnerability
for April 1, 2025 and April 1, 2026. It does not rebuild Part 1, fit clusters or
resume Part 3 ML. Outputs live under `output/part2/acs/`.

## Source and comparison contract

| Item | April 2025 snapshot | April 2026 snapshot |
| --- | --- | --- |
| Demographics | 2019–2023 ACS 5-year, labeled 2023 | 2020–2024 ACS 5-year, labeled 2024 |
| Rent releases | 2013, 2018, 2023 | 2014, 2019, 2024 |
| Dollar base | 2024 dollars | 2024 dollars |
| Spatial support | Same 7,027 hexes, 2020 Census blocks and current parcel support | Identical support |
| Scoring reference | Fit component bounds on earlier observed inputs | Apply the earlier bounds unchanged |

The 2023 release was published [December 12, 2024](https://www.census.gov/programs-surveys/acs/news/data-releases/2023/release.html);
the 2024 release was published [January 29, 2026](https://www.census.gov/programs-surveys/acs/news/data-releases/2024/release.html).
Both precede their respective cutoffs. This remains a **retrospective
reconstruction** using the retrieved releases and fixed modern spatial support,
not an exact reproduction of the information and geography used at the time.
Adjacent five-year releases share four survey years. Differences describe a
one-year data refresh, not independent annual demographic change or a test of
statistical significance.

Year-specific block-group and tract extracts cover Travis, Hays and Williamson.
The run manifest hashes the 16 ACS extract caches, three 2020 block caches,
fixed support, processing code and generated outputs. Older source boundaries
are mapped spatially to the fixed block support, not joined to newer block
groups by assuming GEOIDs are unchanged.

## Allocation, dollars and uncertainty

Counts use the existing dasymetric allocation: distribute 2020 block population
or housing controls across residential parcel support, then allocate each ACS
source zone's counts to hexes. Parcel floor area is the primary within-block
weight, with units/parcel-count fallbacks. Out-of-grid blocks remain in source
denominators. This is a fixed allocation surface, not reconstructed historical
parcel populations or dwelling totals.

Medians are **assigned, never averaged**. Demographic medians use the block group
supplying the largest residential ancillary share, or the dominant tract when
its estimate is unavailable. Rent instead uses the complete-series selection
below. Unpopulated/unallocated hexes may have a geographic
median fallback without usable count-based percentages. The estimate, margin
of error (MOE), and source metadata must all refer to the same selected geography.
A missing block-group MOE cannot be repaired by borrowing a tract's MOE.

Rent and income, including their dollar-valued MOEs, use the same 2024-dollar
conversion: nominal estimate × CPI-U(2024) / CPI-U(source year).
The annual all-items U.S.-city-average CPI-U values are 232.957 (2013), 236.736
(2014), 251.107 (2018), 255.657 (2019), 304.702 (2023), and 313.689 (2024).
Sources: [BLS historical CPI-U](https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202308.pdf)
and [annual averages](https://www.bls.gov/regions/mid-atlantic/data/ConsumerPriceIndexAnnualandSemiAnnual_Table.htm).
The conversion is between the dollar bases of published ACS estimates, not
between individual survey observations.

Each rent profile contains exactly three releases spaced five years apart.
Recent/prior growth is `100 × log(new real rent / old real rent) / 5`;
acceleration is recent minus prior growth, in percentage points. A trend is
usable only with three positive finite rent estimates and three finite,
nonnegative relative MOEs at or below 30%. For the pair, select block-group
data only if all six required releases pass; otherwise select tract data if
all six pass. Use that same geographic level for both profiles. If neither
series passes, all three rent components and the rent index are missing at
both dates. A reliable tract series is preferred to a level-only block-group
score; there is no per-vintage mixing or level-only fallback.

The twelve raw source extracts and original per-vintage dominant assignments
are reused. Every selected estimate retains its own MOE and source GEOID.
Geographic level is fixed, but historical Census boundaries are not harmonized.
Selection uses reliability at both snapshots and is explicitly retrospective,
not an operational rule based only on information available at the earlier date.

## The two indices

Rent pressure requires and equally averages all three components: real current
median gross rent, reliable recent growth, and reliable acceleration.
Vulnerability requires and equally averages all five components: lower real median income, renter share, poverty
share, rent-burden share, and lower college share. Variable definitions are
retained from the current feature build; specifically the existing education
numerator uses `B15003_022 + B15003_023`, and is not expanded here to add the
professional/doctoral categories. The legacy four-component `vulnerability_index`
in the intermediate demographic file is **not** the selected Part 2 index.

Each component preserves the original earlier full-grid observed-value 1st/99th percentiles
(R quantile type 7), clipping and scaling to 0–100. Later values reuse these
exact bounds. Component directions and equal weights are unchanged. A
degenerate earlier range yields zero for observed inputs and is explicitly
flagged; an entirely unavailable earlier component stops the build.

Missing components remain missing. A composite with any missing required
component remains `NA`; remaining terms are never reweighted. Income reliability is
reported, but is not an additional vulnerability eligibility gate. Paired
outputs retain component counts and flags for changed availability, all eight
components at both dates, and both indices at both dates. Final clustering
requires all eight ACS components plus the other required domains at both dates.
These ACS flags alone **do not define the eventual seven-feature cluster sample**.

## Reproduction and artifacts

Run from the repository root, with R dependencies installed and a Census API
key configured for missing extracts (do not put the key in a command or log):

```sh
Rscript scripts/part2/build_acs_snapshots.R
Rscript tests/test_acs_dasymetric.R
Rscript tests/test_acs_rent_history.R
Rscript tests/test_acs_snapshot_scoring.R
Rscript tests/test_acs_snapshot_outputs.R
```

Both modes rebuild the fixed rent histories directly from cached source
candidates, not from old processed rent trends. `--assemble-only` reuses
date-specific demographics after validating metadata; the full mode reruns
demographic allocation. Existing dominant-source crosswalks and the pinned
original normalization reference are required. It is separate from the default
current-vintage targets graph to protect the existing benchmark.

- `acs_features_paired.rds/.csv`: two rows per hex, one per cutoff, with both
  indices, raw components, scores and availability/uncertainty metadata.
- `acs_feature_changes_by_hex.csv`: later-minus-earlier differences and paired
  component-availability flags.
- `acs_scaling.rds`, `acs_scaling_bounds.csv`: frozen earlier scoring recipe.
- `acs_rent_source_candidates.rds`, `acs_rent_source_selection.rds`, and
  `acs_rent_fixed_series_features.rds`: six-vintage quality, selection, and
  estimate/MOE provenance for the fixed-geography histories.
- `acs_snapshot_summary.csv`, `acs_run_manifest.json`: coverage and provenance.
- Date folders: rent histories/trends, demographic counts/medians and sources,
  allocation/conservation, scoring and component-availability QA.
- `part1_preservation_before.csv` and `part1_preservation_after.csv`: checksums
  confirming the existing generic ACS outputs and fixed input files are intact.

The paired runner sets `EWS_ACS_OUTPUT_DIR`, `EWS_ACS_PRESERVE_MISSING=true`,
`EWS_ACS_DOLLAR_BASE_YEAR=2024`, the date, and each separate three-year profile.
Default Part 1 paths, nominal fields and its legacy count-fill setting remain
compatible; the shared MOE-source correction and stricter rent reliability
will apply when Part 1 is intentionally rebuilt in the future.
