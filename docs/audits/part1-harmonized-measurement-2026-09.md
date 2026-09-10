# Part 1: harmonized current measurement

Updated September 10, 2026; retrospective April 1, 2026 cutoff. This is the
current Part 1 baseline, replacing the earlier available-component specification.
The [measurement contract](../methods/current-measurement.md) and
[decision 0013](../decisions/0013-harmonized-measurement.md) give the full rules.

## What changed

Part 1 now uses the same complete, fixed component recipes as Part 2. Rent
retains level, growth and acceleration using a coherent reliable three-vintage
BG history or tract fallback. Vulnerability requires all five terms. Evictions
use recent rate and signed rate change, not percentage change or an
expanding-history share; 311 uses rate, density and signed rate change.
Ownership uses jointly known corporate/financialized evidence with explicit
coverage. Unknown evidence is not zero pressure.

Follow-up: eviction eligibility now checks only April 2, 2024–April 1, 2026,
the two scored years, not all history since January 2022. Older-only ambiguity
no longer removes a cell; missing/conflicting dates potentially in-window and
all other safeguards remain. This restores 109 current eviction scores and
101 fully eligible Part 1 cells. Raw filing counts, geocodes and actual court
coverage are unchanged; no previously eligible cell is lost.

Part 1 uses current-only support: the 2014/2019/2024 rent triplet and 2025
ownership evidence. It does not require the corresponding prior snapshot.
Component bounds use 2026 domain support; cluster means/SDs use the eligible
2026 sample. Part 2 additionally fixes rent source level and ownership cohort
across dates, requires both dates to pass, and freezes 2025 scales. Consequently
the two parts share measurement formulas, not identical scores or cluster IDs.

## Sample and coverage

- **2,557 current cells:** 2,410 Travis and 147 Williamson.
- Includes **all 2,351 paired Part 2 cells**, plus 206 current-only eligible cells.
- Included rent histories: 1,630 block-group and 927 tract.
- Classified cells contain approximately 399,593 fixed promoted residential
  units and 727,183 allocated people: 74.9% of population and 77.3% of ACS
  housing units allocated to the full 7,027-cell audit grid. These denominators
  are not a claim of complete City of Austin coverage.

The old 3,250-cell sample is not the valid current full sample under the
corrected rules. Its reduction is a measurement/coverage correction, not
observed temporal change. Whole-hex denominators are retained; 165 included
cells straddle the current boundary.

Sequential exclusions (each cell appears only at its first failing gate):

| Gate | Excluded | Remaining |
| --- | ---: | ---: |
| Current full-purpose city-center scope | 967 | 6,060 |
| At least 20 fixed residential units | 2,786 | 3,274 |
| Current ownership evidence | 10 | 3,264 |
| 311 coverage | 7 | 3,257 |
| Demolition coverage | 0 | 3,257 |
| Observed eviction coverage | 136 | 3,121 |
| Amenity usability | 0 | 3,121 |
| All required components | 564 | **2,557** |

All 7,027 cells remain in the feature, eligibility and fixed-assignment audit
outputs; excluded cells are not assigned a concern category. The compact
Part 1 assignment/profile table contains classified cells only.

## Refit and interpretation

The routine six-domain/amenity-augmented comparison was rerun across k=2–12,
with 100 gap bootstraps and 100 paired 80% subsamples. This sensitivity compares
domain inclusion under the corrected measurement; it is not a retained old recipe.

For the selected seven-domain, seven-cluster fit: mean silhouette is 0.222,
mean subsample adjusted Rand index is 0.949, and cluster sizes range from 87
to 877. The diagnostics do not identify a unique optimal k: silhouette favors
four, stability favors eight, and gap recommendations diverge (including the
upper search boundary). Seven remains a provisional interpretable typology.
The August spatially blocked review was **not** repeated; its old results do
not validate this new fit. Partner review remains outstanding.

Numeric model IDs were reassigned by the refit. Names and qualitative concern
tiers were reviewed against the new centroids and raw profiles, then pinned
to centroid and label-file hashes. Display order is below; IDs are not scores.

| Display | Current profile | Qualitative concern | Cells |
| --- | --- | --- | ---: |
| 1 | Lower Measured Pressure | Low | 877 |
| 2 | Higher Rents / Lower Vulnerability | Low | 690 |
| 3 | Nearby Amenity Activity | Moderate | 125 |
| 4 | Corporate Ownership + Vulnerability | Moderate | 311 |
| 5 | Selected 311 + Vulnerability | Moderate | 234 |
| 6 | Demolition-Permit Concentration | High | 233 |
| 7 | Eviction-Filing Concentration | Very high | 87 |

Every cell in the eviction-concentration group has a recent filing; its mean
rate is 19.46 per 100 units. Every cell in the demolition-concentration group
has a recent permit. These anchors support the descriptions, not predicted
displacement probabilities. Low concern does not mean no vulnerable residents.

Compared with the immediately preceding harmonized 2,456-cell fit, the broad
profiles remain recognizable; the eviction group increases from 81 to 87.
Sample-based bounds/scaling and centroids are refreshed, so changed assignments
are not themselves a pure estimate of the eligibility rule's effect.

## Products and verification

Canonical current features, model, diagnostics, labels, static/interactive maps,
local site map, neighborhood summaries and frozen-model self-assignment audit
have been regenerated in place. No website deployment, commit or push was made.

- Main map: `figures/03e_amenity_clusters_tentative.png`.
- Neighborhood map: `figures/03g_neighborhood_cluster_plurality.png`.
- Measurement decisions, bounds, eligibility and source hashes:
  `output/part1/measurement/`.
- Model/summary/validation: `output/part1/baseline_cluster_*`.
- Before/after decision audit: `output/part1/eviction_window_change_summary.csv`,
  with full-grid masks and unchanged raw-window counts in companion tables.

Validation includes pure current-source adapter tests, an independent raw-input
reconstruction of all seven recipes and eligibility, independent current model
scaling/centroid/assignment/margin/silhouette/label checks, and the routine
baseline audit. Part 2 matrix, event, model, concern and figure audits verify
the regenerated paired results and source hashes. Current Part 1 replacement
is explicitly distinguished from the earlier run's as-run preservation checks;
old checksums were not rewritten to disguise the rebuild.

The obsolete 14 cluster-selection and 10 high-risk-island generated review
artifacts were removed without creating another archive. Their processing code
and source evidence remain. The short dated August decision narrative is
clearly superseded; its numerical claims are not current. Raw source vintages
and provenance needed for reconstruction are retained. The
[running changelog](../../CHANGELOG.md) records future material changes.

Part 2 is also regenerated on 2,351 paired cells; Part 3 ML remains paused. Remaining
limitations include Hays/Williamson JP3 eviction coverage, address linkage,
coordinate-required 311 extraction, retrospective amenity completeness,
overlapping ACS periods and whole-hex boundary denominators.
