# 0017: Promote Reviewed Housing Counts and Physical References

- **Status:** Accepted
- **Date:** October 7, 2026

**Subsequent decision:** [0018](0018-ben-white-provisional-denominator.md) adopts
170 provisional units at Ben White. The unresolved-denominator choice below
records the earlier stage of the review and is superseded for that property.

## Context

The property audit identified modeled apartment counts that conflict with direct
property evidence, four Domain accounts sharing a misplaced coordinate, and a
fifth Domain apartment account omitted because its coordinate was missing. The
user authorized completing these repairs after the case-location changes in
decision 0016, and investigating the Ben White facility's denominator.

## Decision

Apply pinned, reviewed project evidence at unit promotion, ahead of land-use
validation and all consumers of the canonical unit surface. Keep the original
estimator outputs and raw appraisal inputs intact. Select Bell Springs 400,
Monarch Bluffs 330, Asher 452, Domain south 412 and Domain Building P north 26.
The new review is a production hierarchy override, not an additional training
sample selected at random for the unit estimation model.

Existing project totals replace estimates once and preserve existing allocation
shares, including Monarch's zero-unit land account. Add only the explicitly
reviewed, previously absent account 774412. Reject changed project membership,
overlapping supplemental accounts, changed original Domain coordinates, invalid
geometry, or missing/changed evidence. A future upstream import of this account
must reconcile the supplement rather than silently counting it twice.

For each Domain footprint use the mapped home nearest the mean projected
location of its distinct homes. The mean itself can fall in a courtyard or
parcel hole; the selected reference must lie inside the verified county polygon
and the reviewed grid cell. The four existing southern accounts share cell 3485;
the separate northern account uses cell 3459. The operator map's 438 homes remain
separate from the neighboring 390-unit Residences at the Domain. The City's
436-unit inventory differs by two; retain that discrepancy with the source review.

The northern association is supported by the county's Building P DBA and owner,
the City's 2009 Building P report (26 apartments), and the operator's separately
mapped 26 northern homes. This does not claim that county link 774412 -> 866401
has been resolved into a direct link to polygon 737155. Preserve that original
link and describe the new association as independently reviewed geography.

Account 774412 was also absent from the pinned historical ownership target
extract. Its historical ownership stays unknown in both vintages. Do not use
the current owner to fabricate historical ownership, and retain the ordinary
ownership completeness gate for cluster eligibility.

Ben White's known transitional residential use is recorded on the shared filing
crosswalk and case ledgers. Its independent-housing-unit denominator is unverified.
Retain filings and geocodes, assign no invented unit count, and suppress no cells.
The existing minimum-unit eligibility rule continues to apply. No general
housing-equivalent conversion for beds, rooms, shelters or group quarters is
introduced by this review.

## Consequences

The current inventory is used retrospectively at both Part 2 cutoffs, consistent
with the existing measurement contract. Rebuild parcel-dependent ACS geography,
ownership support, 311 rates, eviction rates, current measurement, annual filing
outputs, and the dependent clusters. Preserve their source/scaling rules and
review semantic cluster labels after fitting; numeric cluster IDs can change.

The original ambiguous-case exclusions remain unchanged. Adding another verified
project inside a mixed-use county polygon may withdraw an overly broad former
single-project match; such cases keep their original accepted geocodes and must
be reported in the before/after audit.

## Implementation and Evidence

- `R/reviewed_unit_properties.R`
- `config/residential_unit_property_reviews.json`
- Local pinned evidence: `data/reviewed_unit_properties/batch1_20261007/`
- Generated project decisions: `output/residential_unit_reviewed_projects.csv`
- [Production audit](../audits/reviewed-unit-properties-2026-10.md)

Some older source PDFs could be read through the web index but could not be
downloaded from their current URLs. Their local evidence records explicitly
identify this limitation; they are review transcriptions, not archived PDFs.
