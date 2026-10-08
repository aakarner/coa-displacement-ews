# Part 1: harmonized current measurement

The run statistics below precede the October residential/property repair.
See the [current repair audit](residential-property-repair-2026-10.md) for the
rebuilt inventory, 2,675-cell model and updated profile comparison.

Updated September 11, 2026; retrospective April 1, 2026 cutoff. This is the
current Part 1 baseline. The [measurement contract](../methods/current-measurement.md),
[decision 0013](../decisions/0013-harmonized-measurement.md), and
[ambiguity policy](../decisions/0014-eviction-ambiguity-keeps-cells.md) specify
its measurement and eligibility rules.

## Change and coverage

Ambiguous eviction cases stay flagged and unassigned. Their candidate cells
retain accepted uniquely mapped filings and audit flags. Source coverage still
spans the two scored years, April 2, 2024–April 1, 2026. Source records,
geocodes, accepted raw counts, unit denominators and other eligibility gates
are unchanged. The change restores 103 fully eligible cells without losing
any previously eligible cell. Those cells contain 2,619 recent and 2,217
previous-window accepted filings; see the [decision audit](eviction-ambiguity-policy-2026-09.md).

- **2,660 current cells:** 2,512 Travis and 148 Williamson, up from 2,557.
- Includes **all 2,490 paired Part 2 cells**, plus 170 current-only cells.
- Rent histories: 1,702 block-group and 958 tract.
- Approximately 435,358 fixed promoted residential units and 785,949 allocated
  people in classified cells; 80.9% of population and 84.1% of ACS housing
  allocated to the full 7,027-cell audit grid. These are not exact City-clipped
  coverage measures.
- 169 included cells straddle the current city boundary; whole-hex units and
  areas remain the denominators.

Sequential exclusions assign each cell to its first failing gate:

| Gate | Excluded | Remaining |
| --- | ---: | ---: |
| Current full-purpose city-center scope | 967 | 6,060 |
| At least 20 fixed residential units | 2,786 | 3,274 |
| Current ownership evidence | 10 | 3,264 |
| 311 coverage | 7 | 3,257 |
| Demolition coverage | 0 | 3,257 |
| Eviction source coverage | 30 | 3,227 |
| Amenity usability | 0 | 3,227 |
| All required components | 567 | **2,660** |

All 7,027 cells remain in feature, eligibility and fixed-assignment audits.
Excluded cells receive no cluster or concern category.

## Refit and interpretation

Complete fixed recipes remain in force. Part 1 uses current-only source
support and component scales estimated from 2026 domain support. Part 2 uses
paired source support and its frozen 2025 scales. The two parts therefore
share formulas, not identical scores or cluster IDs.

The six-domain/amenity-augmented sensitivity was rerun across k=2–12 with
100 gap bootstraps and 100 paired 80% subsamples. For the selected seven-domain,
seven-cluster model, mean silhouette is 0.221 and mean subsample adjusted Rand
index is 0.941. Silhouette favors four, stability favors eight, and gap criteria
favor eight or the upper search boundary. Seven remains a provisional
interpretable typology. The prior August spatial-holdout review was not repeated
and does not validate this refit; partner review remains outstanding.

Names and existing qualitative tiers were matched to the new profiles using
centroids and raw events, then pinned to centroid and label-file hashes.
Numeric model IDs are nominal; the display order below carries the labels.

| Display | Model ID | Profile | Concern | Cells |
| --- | --- | --- | --- | ---: |
| 1 | 1 | Lower Measured Pressure | Low | 913 |
| 2 | 7 | Higher Rents / Lower Vulnerability | Low | 689 |
| 3 | 4 | Nearby Amenity Activity | Moderate | 129 |
| 4 | 6 | Corporate Ownership + Vulnerability | Moderate | 359 |
| 5 | 2 | Selected 311 + Vulnerability | Moderate | 250 |
| 6 | 3 | Demolition-Permit Concentration | High | 235 |
| 7 | 5 | Eviction-Filing Concentration | Very high | 85 |

Every member of the eviction-concentration profile has a recent mapped filing;
its mean rate is 27.9 per 100 units. Every member of the demolition-concentration
profile has a recent permit. These descriptions are not displacement
probabilities. Low concern does not mean no vulnerable residents or displacement.

The broad profiles remain recognizable after expanding support. The eviction
profile changes from 87 to 85 cells as its mean rate rises from 19.5 to 27.9;
restored cells do not all join that profile. Refitted scales and boundaries
also affect assignment, so the differences are not a temporal estimate.

## Products and verification

Canonical features, model, diagnostics, labels, maps, local site map,
neighborhood summaries and frozen-model self-assignment audits are regenerated
in place. No website deployment is part of this change.

Independent checks reconstruct all seven recipes, eligibility, scaling,
centroids, assignments, margins, silhouette, labels and frozen-model
reproduction. The routine Part 1 audit checks the selected baseline. Part 2
has its own source, matrix, model, concern and figure audits. Original dated
preservation hashes remain intact; authorized derived-product replacements
are distinguished from source changes.

- Main map: `figures/03e_amenity_clusters_tentative.png`.
- Neighborhood map: `figures/03g_neighborhood_cluster_plurality.png`.
- Current source decisions, scales and hashes: `output/part1/measurement/`.
- Model, summary, validation and assignments: `output/part1/baseline_cluster_*`.
- Before/after masks: `output/part1/eviction_ambiguity_change_*`.

Part 3 labels are rebuilt under the same ambiguity policy; ML remains paused.
Remaining limitations include Hays/Williamson JP3 filing gaps, incomplete
address linkage, coordinate-required 311 extraction, retrospective amenity
completeness, overlapping ACS releases and whole-hex denominators.
