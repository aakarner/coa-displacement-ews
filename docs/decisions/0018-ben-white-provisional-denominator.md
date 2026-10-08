# 0018: Adopt 170 Provisional Units at Ben White

- **Status:** Accepted, provisional count
- **Date:** October 7, 2026

The user explicitly authorized adopting 170, documenting the assumption and its
sources, and proceeding without additional property research. Use **170
housing-unit equivalents** for South Austin Marketplace, 2101 E Ben White Blvd,
account 291453, cell 6260. Describe its use as probable SRO/rooming-house housing.
This supersedes the unresolved-denominator treatment in decision 0017.

The [July 1, 2019 property listing](https://www.loopnet.com/Listing/2101-E-Ben-White-Blvd-Austin-TX/16495941/)
identifies the account and reports approximately **178 rooms**, transitional
housing and shared kitchen facilities. The
[2018 Travis County reentry guide](https://www.austintexas.gov/sites/default/files/files/HR/TravisCountyReentryGuidebook2018.pdf)
describes weekly single/double room rentals and **100 beds**. These sources
support residential room rentals but do not verify 170 separate dwellings.
The adopted 170 is an explicitly authorized analytical assumption, not a claim
that either source reports that exact housing-unit count.

Shared kitchen or bathroom facilities do not automatically disqualify a room
as a housing unit. [HUD's SRO guidance, page 4](https://www.hud.gov/sites/dfiles/PIH/documents/Special_Housing_Types_Updated_November%202020.pdf)
recognizes shared facilities; the actual Ben White occupancy arrangement and
formal classification have not been established. Beds and rooms are not
automatically equivalent to independently rented units.

Add the omitted account once through the reviewed promotion layer. Preserve
its raw commercial appraisal classification and zero reported units; mark the
selected count `provisional_assumption`, the selection method
`reviewed_assumed_project_total`, and count confidence `low`. Keep the same
property reference for filings and units, with a provisional-denominator flag
on the filing ledgers. Future verified evidence can replace the assumption.

Both retrospective vintages use this fixed inventory. No historical ownership
is inferred from current owner information; ordinary ownership completeness
requirements remain in force. This decision changes the denominator, not the
source-case ambiguity rule or cell-suppression policy.

Sources and authorization are also recorded in
`config/ben_white_170_assumption.json`, whose checksum is pinned by
`config/residential_unit_property_reviews.json`.

The completed rebuild and its before/after checks are recorded in the
[production audit](../audits/ben-white-provisional-units-2026-10.md).
