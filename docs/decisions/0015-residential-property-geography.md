# 0015: Recover omitted residential accounts and align verified filing geography

- **Status:** Accepted; user authorized implementation
- **Decision date:** October 2, 2026
- **Scope:** Residential inventory, Part 1 current measurements and clusters, Part 2 paired measurements and clusters, Part 3 eviction counts and labels

## Decision

Recover active Williamson C3/C5 accounts with positive living area when an
independent exact-parcel City apartment/condominium classification or documented
housing review corroborates residential use. The codes alone are insufficient.
Apply this rule across the available inventory, independently of filings.
Continue the established residential selection, project reconciliation, observed
count/model hierarchy and final nonresidential land-use exclusions.

Reference-only appraisal accounts provide geography, not extra housing units.
Use explicit certified NON-REF account links, plus the documented Lakeline
Station account review. Prefer a current account's own geometry; otherwise use
the legacy footprint with the largest certified living area, with a stable ID
tie-break. Retain every documented legacy footprint as an address-linking alias.
One active tax account contributes its units once.

After the existing case resolver accepts a case, link each reliable defendant
address to county parcel polygons containing its geocode. A polygon qualifies
when its occupied residential accounts identify one project and one operational
unit cell. Documented reference-account aliases use that account's unit cell.
Require PointAddress, Subaddress or APT precision and agreement across all of a
case's reliable addresses before changing its analytical cell. Retain original
coordinates, original cell, proposed property, reference distance and disposition.
The analytical destination must stay inside the fixed City study scope, county
and, for Williamson, the same effective covered court.

Unverified or conflicting property links retain the original accepted cell and
an explicit review flag. Never choose a nearest property merely to obtain a
denominator. Originally excluded ambiguous cases remain unassigned, even when
their candidate addresses can be linked to one development. Candidate cells
remain usable under [decision 0014](0014-eviction-ambiguity-keeps-cells.md).

Both the paired eviction snapshots used by Parts 1–2 and the annual Part 3 panel
consume the same versioned crosswalk. Its manifest rejects changes to its unit
surface, grid, parcel geometry, registries, account links or implementation until
the crosswalk is rebuilt. The initial rule is `verified_residential_parcel_reference_v1`.
[Decision 0016](0016-reviewed-case-property-locations.md) adds source-pinned,
case-specific manual reviews in version 2 while retaining these automatic
matching, original-exclusion and geographic safeguards.

## Consequences and limits

The 20-unit rate threshold, coverage rules, filing dates, accepted case universe,
index formulas and feature weights stay unchanged. Raw geocodes remain evidence;
the property reference determines the analytical cell only after verification.
Housing units can still be estimates. Verifying a parcel relationship does not
verify an estimated unit count, residential occupancy or actual displacement.
Properties spanning the study boundary and unresolved multi-account developments
remain review items; unit points outside the grid are not silently moved inside.

Rebuild dependent unit-weighted ACS/ownership/311 features and current and paired
eviction features, refit and review both cluster models, and rebuild annual
outcome labels. Preserve existing Part 2 reference component bounds where that
domain's contract freezes them. Refit current Part 1 bounds on the repaired
current inventory. This does not resume Part 3 model training.

The ownership index accepts a freshly generated, internally reconciled ownership
manifest and verifies all pinned inputs/outputs instead of hard-coding one run's
manifest hash and eligible-cell count. Source-classifier pins and the 95% common
coverage requirement remain enforced. Unit validation reuses saved source Census
observations, recalculates all allocations, and fails if observations are absent
instead of leaving a stale targeted parcel universe behind.

## Validation

Check reference-account deduplication, a known commercial negative control,
unit-total reconciliation, exact preservation of original case dispositions,
identical annual/paired case assignments, destination geography, count
conservation, ambiguity retention and the unchanged 20-unit threshold. Compare
the preserved pre-repair baseline, an inventory-only counterfactual and the
combined repair. Separate fixed-classifier input effects from refitted profile
changes. Detailed case evidence remains in ignored local output.
