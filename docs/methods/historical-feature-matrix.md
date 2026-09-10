# Part 2: eviction reconstruction and the paired feature matrix

This stage completes the retrospective April 1, 2025 / April 1, 2026 inputs.
It does **not** fit clusters, update the Part 1 benchmark, or resume Part 3 ML.
The same cells and seven feature definitions are used at both dates. Source
vintages advance one year, but this is not a prediction backtest using only
information available at the earlier date.

## Eviction pressure

`scripts/part2/build_eviction_snapshots.R` reuses the prepared Travis and
Williamson filings and reviewed geocode registries. It does not acquire new
court records or send addresses for new geocoding. County-namespaced case IDs
prevent cross-county collisions; multiple defendant/address rows do not become
multiple filings. Reliable registry locations require status M/T and score
at least 90 under the existing resolver contract.

Eligibility is checked over **the two scored annual windows only**. Older
records since January 2022 remain available for unscored diagnostics and full
case-conflict review; their coverage or older-only ambiguities do not exclude
a cell. The two scored components are:

- Recent 12-month mapped filings per 100 fixed promoted residential units;
  rates require at least 20 units.
- Signed change in that rate: recent minus previous 12-month filings per 100
  of the same fixed residential units. Zero previous filings are valid.

For April 2025, the recent window is April 2, 2024–April 1, 2025 and the previous
window is April 2, 2023–April 1, 2024. Both shift forward one year for April
2026. Endpoints are inclusive. History length and window-day counts remain
diagnostics, but percentage change and expanding-history recent share no longer
enter the index. The eligibility contract is `rolling_scored_24_months_v1`.

Source coverage is checked continuously from April 2, 2023 through April 1,
2025 for the earlier snapshot, and April 2, 2024 through April 1, 2026 for the
later snapshot, using supplied court intervals and effective-dated Williamson JP
references. Calendar-year 2026 need not be complete to cover April 1. Travis
JP1–5 and supplied Williamson JP1/JP2 are supported; Hays and unsupported
Williamson court geography are unknown, never filled with zero.

Events must lie inside the fixed April 29, 2026 Austin FULL-purpose boundary
and be assigned to a canonical hex whose center is inside that boundary.
Known ambiguous locations or conflicting/missing filing dates mask candidate
hexes if any retained date could affect either scored window, or a missing date
cannot exclude that possibility. All rows of a potentially relevant case are
retained: a conflicting older or post-cutoff date is not discarded to manufacture
a clean case. An ambiguity known to be older than the scored window is not a
gate. Malformed
case IDs remain audit-only records, not invented valid filings. An entirely
unlocated record cannot identify a particular hex to mask; court/window QA
reports those records without suppressing every cell in the court.

Consequently, this is a **reliably mapped filing proxy**, not complete eviction
incidence, completed displacement, or a verified City-of-Austin match rate.
A usable zero means no qualifying mapped filing in the local source universe.
Defendant names/addresses and case-level audits remain local and Git-ignored.

## Ownership pressure

The separate `scripts/part2/build_ownership_index.R` consumes the reviewed
2024/2025 ownership summaries without rerunning the classifier or parcel
pipeline. See [historical ownership](historical-ownership.md).

Corporate-unit share and density, plus financialized-owner parcel share, use
the same jointly observed parcels at both dates. The existing screen requires
20 common units and at least 95% coverage of both units and parcels. All three
analytical inputs remain missing outside that screen; full-support unknowns,
coverage and source-variant flags remain in the audit outputs. This common
cohort uses both years' evidence and is explicitly retrospective.

## Scoring and common-sample assembly

Unchanged components retain the original earlier-observed 1st/99th-percentile
bounds. The new eviction and 311 signed rate-change components use an earlier
bound `B = q99(abs(rate change))`: clip to `[-B, B]` and map to 0–100, with zero
change at 50. If B is zero, observed changes receive the documented neutral 50
score and a degeneracy flag. Later observations never refit these scales.

Every index uses its complete fixed recipe with equal weights. Missing any
required component makes the whole index unavailable; there is no automatic
reweighting. The recipes contain 3 rent, 5 vulnerability, 3 demolition,
2 eviction, 3 selected-311, 3 ownership and 3 amenity-category scores. Amenity
categories themselves require complete exposure inputs. A zero-event/zero-change
eviction composite can be 25 because its change term is neutral at 50: these
relative indices are not risk probabilities or literal no-risk-zero measures.

`scripts/part2/build_feature_matrix.R` verifies source-manifest checksums,
validates the full integer hex/date keys, checks fixed support and the earlier
scoring reference, and reconciles each composite with its component/category
scores. It then retains a cell only if **both snapshots** pass:

1. The fixed current-city-center mask and at least 20 promoted units.
2. The ownership common-support screen.
3. The selected 311, demolition and mapped-eviction coverage screens.
4. Amenity retrospective usability and finite values for all seven indices.

Every required subcomponent must be available. Availability signatures remain
diagnostics outside the sample; all included cells have complete, unchanged
signatures at both dates. Rent uses a complete reliable block-group series
across all six releases, or a complete reliable tract series across all six;
the selected geographic level cannot change between snapshots. See
[historical ACS](historical-acs.md).
Amenity retrospective usability is not a claim of exhaustive historical source
coverage; its unknown completeness flag is not treated as FALSE or TRUE.

All 7,027 hexes remain in the audit table at each date. The final two analysis
matrices have identical eligible IDs and order. Exclusion reasons are provided
both as overlapping flags and as a single priority-ordered reason, so totals
can be reconciled without double-counting. Source details stay in their
respective paired domain artifacts, linked through a source registry.

The matrix does **not** refit the component scales on the smaller common
sample, fit final cluster z-scores/centroids, or assign clusters. Full-hex areas
and fixed promoted units remain denominators even at boundary-straddling cells;
these are hex-scale rates, not exact city-clipped rates. Adjacent ACS five-year
releases overlap, 311 excludes requests without source coordinates, and the
common sample is a covered subset rather than a complete Austin census.

## Outputs and reproduction

With the previously validated local source caches and paired amenity, ACS, 311
and demolition outputs present:

```sh
Rscript scripts/part2/build_ownership_index.R
Rscript scripts/part2/build_eviction_snapshots.R
Rscript scripts/part2/build_feature_matrix.R
Rscript tests/test_part2_index_scoring.R
Rscript tests/test_part2_ownership_index.R
Rscript tests/test_part2_ownership_index_outputs.R
Rscript tests/test_part2_evictions.R
Rscript tests/test_part2_eviction_outputs.R
Rscript tests/test_part2_feature_matrix.R
Rscript tests/test_part2_feature_matrix_outputs.R
```

New outputs are isolated in `output/part2/ownership_index/`,
`output/part2/evictions/` and `output/part2/matrix/`. The matrix directory contains
the full paired feature table, per-hex eligibility, date-specific analysis
matrices, exclusion summaries, county/index QA, source registry, checksum
verification and run manifest. Non-Part2 outputs/figures are checked for
preservation. At the user's request, corrected Part2 results overwrite the
failed run at the same paths; that run is not archived.

The downstream [historical cluster comparison](historical-cluster-comparison.md)
is implemented separately: fit an earlier k=7 baseline, freeze its
standardization and centroids, assign the later data, and separately refit/align
the later solution to examine structural stability. Those operations are not
part of data assembly and do not replace the canonical Part 1 model.
