# Historical 311 and demolition inputs for Part 2

These two standalone stages reconstruct the selected 311 and residential
demolition indices for April 1, 2025 and April 1, 2026. They read pinned local
histories, retain the fixed 7,027-cell audit grid, and write only separate
Part 2 outputs. They do not run the eviction build, clusters or Part 3 ML.

## Exact windows and sources

Both endpoints in this table are inclusive. Records are selected by their
event dates, not relabeled from annual counts or current summaries.

| Stream / cutoff | Previous window | Recent window |
| --- | --- | --- |
| 311 / April 2025 | Apr 2, 2023–Apr 1, 2024 | Apr 2, 2024–Apr 1, 2025 |
| 311 / April 2026 | Apr 2, 2024–Apr 1, 2025 | Apr 2, 2025–Apr 1, 2026 |
| Demolitions / April 2025 | Apr 2, 2021–Apr 1, 2023 | Apr 2, 2023–Apr 1, 2025 |
| Demolitions / April 2026 | Apr 2, 2022–Apr 1, 2024 | Apr 2, 2024–Apr 1, 2026 |

311 reuses `data/raw_311/austin_311_selected_20200101_20260401.rds` and the three
configured Code Officer intake types in `config/311_smoke_signal_types.csv`.
Those descriptions include department-name transitions; monthly type QA
retains all three. The narrower linked structure-condition series is **not**
substituted, because its sustained coverage starts within the earlier window.
Source floating timestamps retain their calendar dates without timezone shifts.

The cache's `complete=TRUE` means complete pagination for the selected types
**with non-null latitude/longitude**, January 2020 through April 1, 2026. It
does not establish that all complaints have coordinates, that every complaint
was reported, or that a recorded complaint was a confirmed violation. Invalid
coordinates and out-of-grid/city records are retained in the audit ledger.

Demolitions use `data/Issued_Construction_Permits_20260401.csv`, with the
existing residential-demolition selection and unique permit IDs. Issuance is
not proof of a completed demolition. The third index component counts the
subset whose description matches `total\s+demo` case-insensitively; it is not
the number of all demolition permits across residential and commercial uses.
Duplicate conflicts and missing identifiers/dates are checked explicitly;
description-missing and classification-difference QA are retained. The source
contract ends April 1, 2026, even though the latest qualifying issue date is
March 31. The source file is checksum-pinned.

## Fixed geography and coverage

Both streams use the existing City of Austin FULL-purpose boundary snapshot
dated April 29, 2026. Among the 7,027 grid cells, 6,060 have their representative
points inside that fixed boundary. Qualifying events must also have coordinates
inside the exact current FULL boundary. Events outside the grid or in city
slivers whose cell centers lie outside remain explicit exclusions. These
counts therefore describe **the fixed city/grid study footprint**, not every
part of current Austin. The same footprint applies to both dates.

This modern boundary and current unit surface are deliberately fixed for the
retrospective proof of concept. They are not a reconstruction of the exact
footprint or information known at each cutoff. Current membership alone does
not establish historical coverage: pinned historical baseline polygons and
effective-dated jurisdiction actions are replayed at the beginning and every
change date throughout the requested periods.

- **Demolitions:** retain the existing permit-source contract for continuously
  resolved `FULL`, `LTD` or `2MILE` jurisdiction. Check each complete 24-month
  window independently; both must pass before exposing the index. Unsupported
  or unresolved geography remains unavailable.
- **311:** use a conservative, continuously `FULL` geographic screen across
  both 12-month windows. This is explicitly a **proof-of-concept geographic
  assumption**, not independently verified 311 service coverage. The permit
  source's LTD/2MILE coverage rule is not transferred to Code Officer requests.

Observed mapped counts remain available as diagnostics outside usable coverage,
but modeled components and indices are `NA` there. A point falling in multiple
hexes is not counted repeatedly or assigned arbitrarily: its candidate cells
are flagged and withheld for the affected comparison windows.

There are 909 city-center cells that straddle the current boundary. Full
canonical hex areas and promoted residential-unit counts are kept fixed;
they are not clipped or re-estimated for city-only slivers. Boundary-cell
densities and per-unit rates are therefore **hex-scale approximations**, not
exact city-clipped rates. Straddling flags are retained for the final sample
audit; this stage does not automatically exclude them.

## Features and fixed scoring

The corrected 311 index averages three required components with fixed equal weights:

1. Recent selected requests per 100 promoted residential units, available only
   with at least 20 units in the fixed denominator.
2. Recent selected requests per square kilometer of full hex area.
3. Signed difference between recent and previous selected-request rates per
   100 of the same fixed units. A zero previous count is valid; unknown
   coverage remains missing. Retaining both current rate and density means
   two thirds of this composite measures current activity, not three
   independent signals.

The demolition index averages recent residential-demolition permit density,
`max(log(1 + recent permits) - log(1 + previous permits), 0)`, and recent
total-demolition-description permit density.

Unchanged components retain their earlier-observed 1st/99th percentile bounds
(R quantile type 7), applied at both dates. The new signed rate change uses
`B=q99(abs(2025 rate change))`, symmetric clipping at +/-B and zero change
anchored at 50 on a 0–100 score. B=0 yields neutral 50 for observed values and
an explicit flag. Every composite requires all three components, with fixed
equal weights; any missing term makes the entire index missing. Unchanged
components with degenerate earlier ranges retain their zero-score policy.
The 311 raw rates are **not** subjected to the
extra post-index cap found in the legacy current-feature output.

A usable zero means no qualifying mapped source events were observed, not no
displacement, unsafe housing or complaints. The final common seven-feature
clustering sample still must intersect residential eligibility, ownership and
eviction coverage, and feature availability at both dates.

## Reproduction and outputs

From the repository root, with the existing R dependencies and local sources:

```sh
Rscript scripts/part2/build_311_snapshots.R
Rscript scripts/part2/build_demolition_snapshots.R
Rscript tests/test_part2_event_scoring.R
Rscript tests/test_part2_311.R
Rscript tests/test_part2_demolitions.R
Rscript tests/test_part2_311_outputs.R
Rscript tests/test_part2_demolition_outputs.R
Rscript tests/test_part2_event_snapshot_outputs.R
```

No network requests are made. The builders do not invoke the default targets
graph or overwrite current/Part 3 outputs.

Each of `output/part2/311/` and `output/part2/demolitions/` contains:

- Date-specific features, raw counts, coverage flags and component scores.
- A paired `.rds/.csv` table with canonical integer `hex_id`, and a by-hex
  changes table using the frozen earlier scoring reference.
- Event-location/window-membership audits, source/coverage summaries and
  boundary flags.
- Saved normalization bounds and a run manifest hashing inputs, processing
  code and outputs.

`output/part2/events/existing_outputs_before.rds` records the pre-build
preservation audit; the corrected cross-stream output test checks 298 protected
non-Part2 files and excludes the explicitly overwritten Part2 outputs. This
is a dated audit baseline, not permission to overwrite
files if a later intentional Part 1 rebuild changes them.
