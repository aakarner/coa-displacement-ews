# 0019: Reviewed property references at the fixed grid boundary

- **Status:** Accepted; user authorized implementation October 7, 2026
- **Scope:** Two named properties, Caliza and Nexus at Goodnight Ranch

## Decision

Use an explicitly reviewed analytical reference inside each property's portion
of the existing grid. Select the grid cell with the largest parcel overlap,
break ties by cell ID, intersect that portion with the full-purpose Austin
boundary, and take a point on its surface in EPSG:3083. This yields Caliza in
cell 3261 and Nexus in cell 6929. Use the same reference for housing units and
verified property-linked filings.

The reviewed parcels are each more than 99.9% within the April 29, 2026
full-purpose boundary. Their original reference points are inside the correct
parcels and Austin, but outside the fixed grid by approximately 19 and 31 metres.
The H3 cells containing those original points have centers outside the current
City boundary. Merely updating the boundary while retaining the center-based
grid selection would not include those cells.

This is a documented exception to decision 0015's treatment of boundary
properties. It does not change the grid, add nearest-cell fallback matching, or
establish a rule for substantially out-of-city developments. The new point is a
property reference, not a verified dwelling location. Full project counts are
assigned once; they are not multiplied or prorated by parcel overlap.

## Evidence and safeguards

The production review configuration pins the selected points, original Nexus
coordinate, parcel polygons, City boundary and supporting account evidence.
Caliza retains its original point-on-surface coordinate as provenance when
added as a named supplemental account. Rebuilds verify the destination cell,
parcel containment, City containment and minimum 99.9% City overlap. Changes to
the evidence hashes fail until reviewed.

Caliza contributes 270 units, corroborated by the
[April 19, 2023 JLL sale announcement](https://www.jll.com/en-us/newsroom/sale-of-northwest-austin-multihousing-community-closes)
and the active WCAD account R500219. Nexus retains its county-reported 294
units; only its analytical reference changes. Raw records remain unchanged.

Existing county/court coverage, original ambiguity exclusions and the 20-unit
rate threshold still apply. The paired and annual filing products share the
same crosswalk. Historical ownership is rebuilt from its own sources; the
present owner is not substituted for missing historical observations.

## Validation

Check point containment and drift rejection, exact project totals, no duplicate
accounts, unchanged unreviewed parcel counts/coordinates, preserved source-case
dispositions, paired/annual agreement, and coverage at the destination. Preserve
the pre-batch outputs and compare all dependent measurements and refitted
profiles. See the batch-2 audit for the separate Canyon Creek and Ocotillo
corrections and unresolved mobile-home evidence.
