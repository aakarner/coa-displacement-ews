# 0014: Flag ambiguous eviction cases without suppressing candidate cells

- **Status:** Accepted
- **Decision date:** September 11, 2026
- **Scope:** Part 1 current features, Part 2 retrospective snapshots, and Part 3 annual eviction counts and labels

## Context and decision

A case can have several reliable defendant-address geocodes without a unique
residential location. The prior rule excluded that case and also made every
candidate cell's eviction count unavailable. A small number of ambiguous
cases consequently removed thousands of other, uniquely mapped filings from
otherwise usable cells. Restricting the veto to the scored two years in
[decision 0013](0013-harmonized-measurement.md) reduced but did not resolve this
problem. The user explicitly directed that ambiguous cases remain flagged and
unassigned, with no suppression of their candidate cells.

Keep the existing case-resolution requirements. Multiple candidate hexes,
mixed inside/outside locations, inconsistent dates or jurisdictions, and
invalid identifiers remain explicit case-level exclusion reasons. Do not pick
an address, split a filing between cells, or count it more than once. Retain
candidate locations and original dates for audit only.

An otherwise covered cell uses the count of accepted, uniquely mapped filings,
even when an excluded case could belong there. `eviction_unresolved_candidate_cases`
and `eviction_has_unassigned_ambiguous_cases` identify potentially relevant
candidate cases in each snapshot. Neither field gates counts or indices.
The snapshot contract is `rolling_scored_24_months_v2` and the ambiguity policy
is `flag_unassigned_cases_keep_cells_v1`.

## Consequences

A zero means zero accepted uniquely mapped filings, not proof that no filings
or displacement occurred there. Source/court coverage, exact point and city
geography, scored time windows, the 20-unit rate floor, and complete-component
requirements still apply. Uncovered cells remain missing.

Annual Part 3 panels apply the same rule. `measurement_complete` now describes
the tally of accepted uniquely mapped filings; `all_filing_locations_complete`
remains false. Separate `has_unassigned_ambiguous_cases` and
`unresolved_candidate_cases` fields retain location uncertainty. Source and
full-year coverage still gate annual counts, so partial 2026 counts do not
become complete-year outcomes. Rebuilding existing labels does not resume ML.

Rebuild the canonical affected outputs in place. Refit both eviction component
scales on the revised 2025 reference sample for Part 2 and freeze them for
2026; Part 1 continues to fit its separate current reference. Refit affected
clusters and review labels against their new profiles. Rate formulas, feature
weights, unit denominators and qualitative concern criteria are unchanged.

## Validation and reconsideration

Regression tests require unassigned ambiguous cases, retained valid counts in
both candidate cells, valid proxy zeros, explicit ambiguity flags, and missing
counts where source or full-year coverage fails. The before/after audit must
show unchanged raw mapped counts and no loss of eligibility from this change.

Revisit assignment when better court premises information or an adjudicated
case-level address review can resolve uncertainty. Candidate locations alone
are not evidence sufficient to assign a case. Assess remaining undercount
through the case and court/window audits rather than suppressing valid cells.
