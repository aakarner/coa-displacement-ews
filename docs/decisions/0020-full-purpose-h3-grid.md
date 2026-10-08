# 0020 — Expand the H3 surface to cover the adopted full-purpose boundary

Date: October 7, 2026. Authorized by the user during the residential repair
cluster rebuild, choosing the existing April 2026 boundary over a new vintage.

The Census-2021-derived 7,027-cell computational grid omits about 33.4345 square
kilometers of land within the April 29, 2026 Austin full-purpose boundary.
Filtering that grid to a newer boundary cannot restore the omitted cells.

Use the union of the legacy grid and every resolution-9 H3 cell with positive
area intersection with that adopted boundary. This adds 923 cells, producing
7,950 computational cells, including 7,215 that intersect the City. Retaining
legacy cells preserves historical audit coverage. Full H3 polygons remain
unclipped. Candidate discovery uses a padded boundary; exact intersection and
an independent uncovered-area check establish full coverage.

Preserve every original H3 index, numeric `hex_id`, geometry and reference
coordinate. Append new identifiers in a permanent `config/hex_id_registry.csv`;
never regenerate the numeric IDs from a newly sorted full H3 list. Existing
case and property reviews thus retain their meaning.

The change expands computational coverage. It does not change the adopted
center-based analytical scope: 6,196 cells have their projected point on surface
inside the City, versus 6,060 before. Boundary-only cells remain explicitly
outside that analytical scope. Unit support, source coverage and feature
availability still determine which City cells enter clustering. Any future
change to the boundary-cell eligibility rule requires a separate documented
measurement decision.

`output/hex_grid_manifest.json` records the version, rule, counts and hashes.
Replace scattered fixed-size assertions with checks against that contract,
backed by independent tests of boundary coverage and original-ID preservation.
Rebuild county and effective-dated JP assignments, parcel/event aggregation,
ACS allocations, all selected current/paired features, and cluster products.
Use the same boundary and grid for both comparison vintages. Preserve raw
sources and frozen historical evidence; preserve normalization bounds where
the existing paired measurement contract requires them.

The rebuild is authorized to supersede the derived grid, county/JP references,
and dependent products pinned in older run-preservation inventories. Their
archived copies remain historical evidence; they are not permanent locks on
an explicitly approved new geography. Raw source hashes remain protected.
