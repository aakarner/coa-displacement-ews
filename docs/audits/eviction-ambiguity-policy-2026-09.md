# Eviction ambiguity policy: September 11, 2026

[Decision 0014](../decisions/0014-eviction-ambiguity-keeps-cells.md) removes the
candidate-cell veto across current/paired snapshots and annual outcomes.
Ambiguous cases retain explicit assignment-status flags and no assigned hex.
Candidate locations remain audit evidence. Covered cells retain the tally of
accepted uniquely mapped filings, including proxy zeros.

## Verified effect on eligibility

| Measure | Before | After | Restored |
| --- | ---: | ---: | ---: |
| April 2025 cells with usable eviction counts | 5,845 | 5,977 | 132 |
| April 2025 cells with usable eviction scores | 3,123 | 3,241 | 118 |
| April 2026 cells with usable eviction counts | 5,854 | 5,977 | 123 |
| April 2026 cells with usable eviction scores | 3,135 | 3,241 | 106 |
| Part 1 complete current sample | 2,557 | 2,660 | 103 |
| Part 2 complete paired sample | 2,351 | 2,490 | 139 |

There are no eligibility losses. The restored 103 Part 1 cells contain 2,217
accepted filings in the previous scored year and 2,619 in the recent scored
year: 4,836 valid filings across the two years. The classified sample's recent
filings increase from 6,909 to 9,528. These are April 2–April 1 windows, not
calendar years. The restored filings are uniquely mapped cases previously
hidden by their cells' veto, not newly assigned ambiguous cases.

The entire supported city-study source still has 11,784 accepted filings in
the recent April 2026 window. Raw mapped counts, unresolved-candidate counts,
court coverage, geocodes and units are unchanged. Remaining exclusions follow
other source, denominator and complete-component requirements.

## Annual outcomes

| Calendar year | Restored complete hex-years | Accepted filings restored |
| --- | ---: | ---: |
| 2020 | 26 | 202 |
| 2021 | 18 | 64 |
| 2022 | 69 | 762 |
| 2023 | 69 | 890 |
| 2024 | 66 | 1,301 |
| 2025 | 72 | 1,771 |
| 2026 | 0 | 0 |

2026 remains partial. Observed-to-date counts are retained, but no complete-year
2026 outcome is created. Existing forward labels are rebuilt from accepted
counts; Part 3 ML remains paused.

## Verification and local artifacts

Synthetic regressions cover ambiguity spanning two candidate cells, valid
filings in both, zero accepted filings, missing/conflicting dates, invalid
identifiers, source gaps, and incomplete years. Independent persisted-output
checks reproduce snapshot counts and flags, annual counts and forward-label
sums, current measurement and paired eligibility. Source-manifest checks
verify that the rebuild used the same inputs.

- `output/part1/eviction_ambiguity_change_summary.csv`: before/after eligibility.
- Companion `eviction_ambiguity_change_events.csv` and
  `eviction_ambiguity_change_eligibility.csv`: full-grid masks and raw counts.
- `output/part3/eviction_ambiguity_change_summary.csv`: annual restoration.
- `output/part2/evictions/eviction_case_assignment_issues.csv`: case exclusion reasons.
- Dated `eviction_localizable_uncertainty.csv` tables: candidate case/hex links,
  retained solely for audit.

The reference scales and downstream cluster models are refitted on the revised
samples. Their updated numerical results are documented in the current Part 1
and Part 2 audits. Changed assignments combine restored support with refreshed
scales and centroids; they are not a temporal displacement estimate.
