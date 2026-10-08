# Remaining property review: first batch

**Production update, October 7, 2026:** the 54 Bell, six Asher and 11 Monarch
location links below have been applied to the shared annual/paired pipeline
under [decision 0016](../decisions/0016-reviewed-case-property-locations.md).
The reviewed Bell Springs, Asher and Monarch counts and the Domain account and
geography repairs have subsequently been applied under
[decision 0017](../decisions/0017-reviewed-unit-counts-and-geography.md). Ben
White subsequently receives the user-approved provisional 170-unit denominator
under [decision 0018](../decisions/0018-ben-white-provisional-denominator.md).
The evidence and proposed actions below record the review preceding those
implementations. See the [location audit](reviewed-property-locations-2026-10.md)
and [unit/geography audit](reviewed-unit-properties-2026-10.md) for results.

Initially reviewed October 2, 2026; twenty-three Bell property resolutions using
court records added October 7, 2026. One retains an unresolved apartment-number
discrepancy. This is an evidence review and proposed correction
plan; it does not change production filings, unit counts, or cluster results.
The cohort is **162 unique filings**, dated **April 2, 2025–April 1, 2026**, in
four cells from the original 41-cell audit. It covers five property situations.

## Findings and proposed actions

| Property | Filings | Finding | Proposed action |
|---|---:|---|---|
| Bell Southpark, 10500 S IH 35 | 54 | Thirty-one cases have a unique Springs candidate from apartment numbers; court evidence supports twenty-two further Springs locations and one Meadows location. One Springs case retains conflicting apartment numbers, but its case-specific plaintiff/DBA and shared street address support the property assignment. | Prepare the 54 supported case links: 53 Springs and one Meadows. Preserve the apartment discrepancy without correcting either source entry. Reconcile the Springs phase's 400 mapped units with its existing account. |
| Bridge at Asher, 10505 S IH 35 | 6 | Correct residential account 513751 already exists; geocode misses its polygon by 8.15 m. | Add an explicitly reviewed address-to-property link to the existing unit reference in cell 6973. Review 452 documented units against the current 449-unit estimate. |
| Bridge at Monarch Bluffs, 8515 S IH 35 | 11 | Nearest polygon is a restaurant. The apartment land and improvement accounts already exist farther away. | Link to the verified apartment project, whose units are in cell 6964. Use the documented 330-unit count to review the current 361.3-unit estimate. |
| Villages at the Domain | 31 | Four apartment accounts already contribute 405.3 estimated units, but their common coordinate is misplaced in cell 3486. The operator maps 412 homes in the southern footprint and 26 in a separate northern footprint. | Repair the existing accounts' geography and reconcile their count with the 412-home southern footprint. Review the separate northern account before adding its 26 homes. Count the 438-home development once. |
| South Austin Marketplace, 2101 E Ben White Blvd | 60 | Residential/transitional use is corroborated, but sources describe rooms and beds rather than a verified stock of independent housing units. | Retain and flag the filings. Establish a distinct residential-facility classification and denominator treatment before using them in an ordinary housing-unit rate. |

There are **71 filing-location proposals with a unique supported destination**
(54 Bell, six Asher, 11 Monarch). The Domain's 31 filings have a concrete
inventory/geography repair path. The other **60 filings** are Ben White cases
with an unresolved denominator. One supported Bell property assignment retains
unresolved apartment detail, as documented below. These are
review dispositions, not counts of repairs already applied. The other 55 filings
in the 217-case queue remain outside this first batch.

## Bell Southpark: distinguish the phases before moving filings

The operator announced in June 2021 that it combined Lenox Springs and Lenox
Meadows (619 units) with the existing Bell Southpark (330 units), for 949 units
across three phases. Its current website directs visitors to separate Parks,
Springs and Meadows maps. [Operator acquisition announcement](https://bellpartnersinc.com/2021/bell-partners-acquires-two-austin-properties-to-create-949-unit-multifamily-community/),
[operator floorplans and maps](https://www.bellsouthpark.com/floor-plans/).

The public georeferenced map embedded by that operator contains 949 distinct
unit IDs. Its embedded geometry creation date is September 25, 2025; it was
retrieved October 2, 2026. Intersecting its unit label locations with county
polygons gives:

| County parcel | Phase evidence | Mapped homes |
|---|---|---:|
| 878332 | Lenox Springs / Bell Southpark II | 400 |
| 879205 | Bell Southpark I / Parks | 330 |
| 887111 | Bell Southpark III / Meadows | 219 |

The current unit estimate for parcel 878332 is 330.86, whereas the map identifies
400 homes. Parcel 887111 already has 219 operational units. These source counts
describe the existing phases; they must not be added on top of those accounts.

Every audited 10500 address includes an apartment number. Matching those numbers
against all three phases exposes an ambiguity that the street geocoder loses:
for example, `1207` appears as `1207` in one phase and `01207` in another. Because
the filing addresses inconsistently retain leading zeros, choosing an exact
unpadded match would create false certainty. Thirty-one cases have a single
phase candidate even after allowing that normalization; 22 have more than one.
One additional case has both a matched address variant and a unit number absent
from the map, so it fails the automatic apartment-map matching requirement.
Its subsequent manual property review is documented below; the automatic
matching result and both source apartment numbers remain unchanged.

On October 7, a user-provided Register of Actions resolved one of the 22
multiple-candidate cases. Its case number and May 22, 2025 filing date match the
export, and its plaintiff is **BELL FUND VII SOUTHPARK SPRINGS LLC dba BELL
SOUTHPARK (SPRINGS)**. The named company matches the county owner of parcel
878332. The export's apartment 1207 matches Springs map label 01207 after
leading-zero normalization. Together, these establish Springs as this case's
property; its proposed destination is the existing unit reference in cell 6670.
The screenshot and a structured, source-hashed review record are saved under
the ignored `output/residential_property_batch1/reviewed_cases/` directory.
This is a case-specific resolution, not evidence that all other ambiguous
filings belong to Springs. The record's June 4 dismissal does not remove the
May 22 filing from a filing-count measure or establish a completed eviction.

Another October 7 screenshot, corroborated in the active Chrome case page,
resolves a second case filed April 10, 2025. Its plaintiff is **BELL FUND VII
SOUTHPARK SPRINGS LLC**, matching parcel 878332's owner. Its exported apartment
6205 matches Springs map label 06205; the same proposed unit reference in cell
6670 applies. The case number and filing date match the export. Its screenshot,
case-detail URL and source-hashed review record are saved in `reviewed_cases/`.

A third October 7 screenshot resolves another May 22, 2025 filing using the
portal's search-results row. The case number, filing date and Precinct Three
match the export, and the full case style identifies **BELL FUND VII SOUTHPARK
SPRINGS LLC dba BELL SOUTHPARK (SPRINGS)**. Both accepted address records agree
on apartment 2107, matching Springs map label 02107. This supports parcel
878332 and the same proposed unit reference in cell 6670. A full Register of
Actions is not required for this property link because the necessary case and
plaintiff evidence is visible in the search result. The screenshot and review
record are retained in `reviewed_cases/`; no disposition is inferred from it.

A fourth October 7 search-results screenshot resolves a July 21, 2025 filing.
Its case number, filing date and court match the export, and its plaintiff and
DBA again explicitly identify Springs. The accepted apartment 2303 matches
Springs map label 02303, supporting parcel 878332 and proposed cell 6670.
The displayed dismissed status does not remove this filing from the count.
The screenshot and case-specific review are retained in `reviewed_cases/`.

A fifth October 7 search-results screenshot resolves an April 11, 2025 filing.
Its case number, filing date and court match the export, and its plaintiff is
**BELL FUND VII SOUTHPARK SPRINGS LLC**. All three accepted defendant address
records agree on apartment 6301, matching Springs map label 06301. This
supports parcel 878332 and proposed cell 6670, counted as one filing. The
generic Final Status label does not establish a particular disposition. Its
screenshot and case-specific review are retained in `reviewed_cases/`.

A sixth October 7 search-results screenshot resolves another July 21, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. The accepted apartment 1304
matches Springs map label 01304, supporting parcel 878332 and proposed cell
6670. The displayed Appealed status is recorded without inferring an appeal
outcome or completed eviction. Its screenshot and case-specific review are
retained in `reviewed_cases/`.

A seventh October 7 search-results screenshot resolves another July 21, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. All three accepted defendant
address records agree on apartment 6105, matching Springs map label 06105.
This supports parcel 878332 and proposed cell 6670, counted as one filing
despite the displayed dismissed status. Its screenshot and case-specific
review are retained in `reviewed_cases/`.

A further October 7 search-results screenshot resolves an August 18, 2025
filing, bringing the court-verified count to eight. Its case number, filing
date and court match the export, and its plaintiff and DBA identify Springs.
The accepted apartment 7110 matches Springs map label 07110, supporting parcel
878332 and proposed cell 6670. Its generic Final Status label does not establish
a particular disposition. The screenshot and case-specific review are retained
in `reviewed_cases/`.

A ninth October 7 search-results screenshot resolves another August 18, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. Both accepted defendant address
records agree on apartment 6304, matching Springs map label 06304. This
supports parcel 878332 and proposed cell 6670, counted as one filing despite
the displayed dismissed status. Its screenshot and case-specific review are
retained in `reviewed_cases/`.

A tenth October 7 search-results screenshot resolves a September 22, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. The accepted apartment 4302
matches Springs map label 04302, supporting parcel 878332 and proposed cell
6670. Its generic Final Status label does not establish a particular
disposition. Its screenshot and case-specific review are retained in
`reviewed_cases/`.

A further October 7 search-results screenshot resolves another September 22,
2025 filing, bringing the court-verified count to eleven. Its case number,
filing date and court match the export, and its plaintiff and DBA explicitly
identify Springs. Apartment 2303 matches Springs map label 02303, supporting
parcel 878332 and proposed cell 6670. It remains a separate filing from the
July case at the same apartment. The generic Final Status label does not
establish a particular disposition. Its screenshot and case-specific review
are retained in `reviewed_cases/`.

A twelfth October 7 search-results screenshot resolves another September 22,
2025 filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. Apartment 2207 matches Springs
map label 02207, supporting parcel 878332 and proposed cell 6670. The generic
Final Status label does not establish a particular disposition. Its screenshot
and case-specific review are retained in `reviewed_cases/`.

A thirteenth October 7 search-results screenshot resolves an October 15, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff is **BELL FUND VII SOUTHPARK SPRINGS LLC**. Apartment 7104 matches
Springs map label 07104, supporting parcel 878332 and proposed cell 6670.
The displayed dismissed status does not remove this filing from the count.
Its screenshot and case-specific review are retained in `reviewed_cases/`.

A fourteenth October 7 search-results screenshot resolves another October 15,
2025 filing. Its case number, filing date and court match the export, and its
plaintiff is **BELL FUND VII SOUTHPARK SPRINGS LLC**. Apartment 6201 matches
Springs map label 06201, supporting parcel 878332 and proposed cell 6670.
The displayed dismissed status does not remove this filing from the count.
Its screenshot and case-specific review are retained in `reviewed_cases/`.

A fifteenth October 7 search-results screenshot resolves a November 17, 2025
filing. Its case number, filing date and court match the export, and its
plaintiff and DBA explicitly identify Springs. Apartment 1210 has candidates
in all three phases; the court evidence selects Springs map label 01210,
supporting parcel 878332 and proposed cell 6670. The generic Final Status
label does not establish a particular disposition. Its screenshot and
case-specific review are retained in `reviewed_cases/`.

A sixteenth October 7 search-results screenshot resolves a November 20, 2025
filing to **Meadows**. Its case number, filing date and court match the export.
The plaintiff is **BELL FUND VII SOUTH PARK MEADOWS LLC DBA BELL SOUTHPARK
(MEADOWS)**, agreeing with parcel 887111's county owner apart from spacing in
Southpark. Apartment 4305 matches Meadows map label 4305; the competing Springs
label is 04305. The supported destination is the existing Meadows unit reference
in cell 6974. This case demonstrates why the earlier Springs resolutions must
not be generalized to all remaining filings. The generic Final Status label
does not establish a particular disposition. Its screenshot and case-specific
review are retained in `reviewed_cases/`.

A seventeenth October 7 search-results screenshot initially supplied partial evidence
for a November 24, 2025 filing. Its case number, filing date and court match
the export, and its plaintiff and DBA explicitly identify Springs. However,
the export contains apartment variants 11103 and 11130. Only 11103 matches
the operator map; the screenshot displays no apartment address and cannot
establish which variant is correct. The initial review withheld the property
proposal pending apartment clarification. That assessment was subsequently
revised after distinguishing the property-level evidence from apartment-level
uncertainty, as documented below. The initial review is preserved under
`reviewed_cases/history/`; neither source apartment number is corrected.

A further October 7 search-results screenshot resolves a December 16, 2025
filing, bringing completed court-record resolutions to seventeen. Its case
number, filing date and court match the export, and its plaintiff and DBA
explicitly identify Springs. Both accepted defendant address records agree
on apartment 2205, matching Springs map label 02205, parcel 878332 and proposed
cell 6670. This remains one filing despite the two address records and the
displayed dismissed status. Its screenshot and case-specific review are
retained in `reviewed_cases/`.

A further October 7 search-results screenshot resolves another December 16,
2025 filing, bringing completed court-record resolutions to eighteen. Its case
number, filing date and court match the export, and its plaintiff and DBA
explicitly identify Springs. Apartment 4107 matches Springs map label 04107,
supporting parcel 878332 and proposed cell 6670. The generic Final Status label
does not establish a particular disposition. Its screenshot and case-specific
review are retained in `reviewed_cases/`.

A further October 7 search-results screenshot resolves another December 16,
2025 filing, bringing completed court-record resolutions to nineteen. Its case
number, filing date and court match the export, and its plaintiff and DBA
explicitly identify Springs. Apartment 6201 matches Springs map label 06201,
supporting parcel 878332 and proposed cell 6670. This is a separate filing from
the October case at the same apartment. The generic Final Status label does
not establish a particular disposition. Its screenshot and case-specific
review are retained in `reviewed_cases/`.

A further October 7 search-results screenshot resolves a January 23, 2026
filing, bringing completed court-record resolutions to twenty. Its case number,
filing date and court match the export, and its plaintiff and DBA explicitly
identify Springs. Apartment 4303 matches Springs map label 04303, supporting
parcel 878332 and proposed cell 6670. The displayed Appealed status is recorded
without inferring an appeal outcome or completed eviction. Its screenshot and
case-specific review are retained in `reviewed_cases/`.

A further October 7 search-results screenshot resolves a February 18, 2026
filing, bringing completed court-record resolutions to twenty-one. Its case
number, filing date and court match the export, and its plaintiff and DBA
explicitly identify Springs. Apartment 6309 matches Springs map label 06309,
supporting parcel 878332 and proposed cell 6670. The displayed dismissed status
does not remove this filing from the count. Its screenshot and case-specific
review are retained in `reviewed_cases/`.

A final October 7 search-results screenshot resolves another February 18, 2026
filing, bringing completed court-record resolutions to twenty-two. Its case
number, filing date and court match the export, and its plaintiff and DBA
explicitly identify Springs. Apartment 7104 matches Springs map label 07104,
supporting parcel 878332 and proposed cell 6670. It remains a separate filing
from the October case at the same apartment, despite its dismissed status.
Its screenshot and case-specific review are retained in `reviewed_cases/`.
All 22 cases with multiple phase candidates now have court-supported phase
resolutions: 21 Springs and one Meadows.

The user then supplied the full Register of Actions and its financial
continuation for the November 24 conflicting-address case. The docket confirms
the case number, filing date, court, and Springs plaintiff/DBA. It lists only
city/state/ZIP for the defendants, so it supplies no new street or apartment
evidence. Both exported address variants nevertheless share the same street
address; the case-specific plaintiff/DBA identifies Springs, matching that
parcel's county owner. Apartment 11103 corroborates Springs on the operator
map; 11130 is unmatched rather than a competing mapped property. Together,
this supports a manual case-level proposal for parcel 878332 and cell 6670.
Requiring an exact apartment correction before accepting this property proposal
was unnecessarily strict. The exact apartment remains unresolved, and neither
variant is silently corrected or declared verified. This is a case-specific
assessment using the combined evidence, not a general plaintiff-name matching
rule. The original search result, both new screenshots, source hashes and
review history are retained in `reviewed_cases/`.

**Proposed correction:** a review crosswalk for the 31 uniquely matched cases
and the twenty-three court-supported property cases, with their source coordinates retained,
plus a reviewed 400-unit observation for the existing Springs account. These
54 supported locations comprise 53 Springs cases and one Meadows case. Preserve
the unresolved apartment-number flag on the one manually verified property
case. Exact apartment evidence would be needed to correct its source address,
but is not a prerequisite for this property-level proposal. No further portal
lookups are needed for this Bell property review.
Do not infer the phase from the nearest parcel or add all 949 homes to the
filing cell. The map corroborates physical geography; it does not independently
verify each unit's occupancy at a filing date.

This new phase uncertainty does not automatically change the original raw
case-resolution policy. Unverified property links retain the accepted original
cell under decision 0015. Any proposal to exclude newly discovered ambiguous
cases would require an explicit extension of that policy; no cells are suppressed.

## Bridge at Asher: a supported boundary correction

The court address is 10505 South IH 35, and the geocode is 8.15 m outside parcel
513751. Both the county account's DBA and the operator identify Bridge at Asher.
Its existing unit reference is in cell 6973, with 449 operational units.
The operator confirms the address; HACA's April 2019 acquisition materials
describe 452 apartments at that address. [Operator](https://www.bridgeatasher.com/),
[HACA acquisition materials](https://www.hacanet.org/wp-content/uploads/2019/04/20190418_HACA-Packet.pdf).

**Proposed correction:** explicitly link this address to parcel 513751 and use
the existing property reference for the six filings. Reconcile the observed
452-unit source through the unit-count hierarchy separately. This is distinct
from Bell Southpark across I-35.

## Monarch Bluffs: nearest does not mean correct

The nearest polygon, parcel 576183, is 5.51 m away. County attributes identify
it as **Dario's Mexican Restaurant, 8625 S IH 35**, with commercial state code
F1. The second-nearest polygon is also not the target apartment parcel. The
correct land parcel, 533185, is 26.39 m away and has the exact 8515 address.
Residential improvement account 975264 shares that address, the historical
Griffis Southpark DBA, and the existing project `project:533185`. Its units are
already assigned to cell 6964.

The operator confirms 8515 as Monarch Bluffs. Austin Energy's December 28, 2023
completed-project fact sheet names the property and reports **330 rentable
units**, corroborating residential use before the analytical cutoff.
[Operator](https://www.bridgeatmonarchbluffs.com/),
[Austin Energy report, page 5](https://services.austintexas.gov/edims/document.cfm?id=421606).

**Proposed correction:** a reviewed address link to the existing apartment
project, preserving the land/improvement distinction, and a 330-unit observed
count for the project's housing. Its current 361.29-unit estimate is not an
additional stock of housing. The preliminary checklist's nearest-parcel lead
was not a valid property identification.

## Domain: move and reconcile existing housing; do not add a duplicate project

TCAD's four residential accounts 774341–774344 identify Villages buildings
S, T, V and Z. Explicit county account links connect all four to mapped parent
parcel 774333. They have B1 residential improvement codes and are already in
the unit inventory. All four inherited the same `address_point_nearest`
coordinate for **3314 W Braker Lane**. That coordinate lies **384.88 m outside
their mapped parent parcel**, in cell 3486, and carries 405.326 estimated units.

The 31 filing geocodes lie on parent parcel 774333 in cell 3485. The parent
has commercial land-use coding because this is a mixed-use development; its
apartment accounts still contain residential units. Thus, the original failure
is largely misplaced account geography and an absent parent-account link,
rather than omission of all of the housing.

The operator's georeferenced map contains **438 distinct homes**, agreeing with
the operator's published total and Simon's historical development announcement.
The map's embedded creation date is April 21, 2025, and retrieval date is October
2, 2026. Spatial reconciliation gives **412 homes on parcel 774333** and **26 on
northern parcel 737155**. The City's affordable inventory reports 436; that
two-unit difference remains documented rather than silently discarded.
[Operator's count](https://www.willowbridgepc.com/properties/villages-at-the-domain-austin-tx/area),
[operator map entry point](https://villagesdomain.com/floorplans/),
[Simon's historical announcement](https://investors.simon.com/node/8796/pdf).

A fifth county residential account, 774412, is named Villages Building P but has
no coordinate in the upstream extraction and is absent from the operational
inventory. Its county link points to 866401, not directly to 737155. The link
between this account and the 26 mapped northern homes is a strong candidate
that still needs confirmation. These northern homes must not be folded into
the southern footprint merely to match 438.

**Proposed correction:** use the four explicit parent-account links to correct
the southern project's shared residential reference and replace its modeled
total with the supported 412-home footprint count. A center of the mapped
southern homes lies in cell 3485; a center of the northern 26 lies in cell 3459.
These centers are diagnostic candidates, not yet production allocations. Verify
the fifth account independently before adding any missing northern homes. The
31 audited filings have apartment/building designations at the southern
addresses; their accepted cell already agrees with the southern candidate
reference. Repairing the denominator is the necessary first step here.

![Domain site-map and existing unit reference comparison](../../output/residential_property_batch1/domain_reconciliation.png)

## Ben White: residential use is supported; the denominator is unresolved

The county DBA is South Austin Marketplace, at parcel 291453. Its F1 improvement
code and commercial zoning explain why the upstream A/B-or-SF/MF residential
selection rule omitted it. That rule does not capture every form of housing.

The City's 2018 reentry guide identifies the exact address as South Austin
Marketplace and describes single/double rooms and **100 beds**. A 2019 property
listing describes transitional housing with approximately **178 rooms**. A
Travis County payment record identifies EBW Transitional Housing LLC at the
same address. These support residential use, but neither beds nor a historical
room count automatically establishes the April 2026 housing-unit denominator.
[City reentry guide](https://www.austintexas.gov/sites/default/files/files/HR/TravisCountyReentryGuidebook2018.pdf),
[property listing](https://www.loopnet.com/Listing/2101-E-Ben-White-Blvd-Austin-TX/16495941/),
[county payment record](https://tctransparency.traviscountytx.gov/Checks/OutstandingChecks?page=24).

**Recommendation:** retain the 60 filings, explicitly flag the facility, and
keep its ordinary housing-unit rate unavailable until a comparable denominator
is established. A separate residential-facility category would make this
pressure visible while documenting the stock measurement problem. Treating
room or bed counts as interchangeable with apartment units would change the
meaning of the rate. No exclusion or new denominator has been applied.

## Implementation and evidence handoff

Apply supported repairs through versioned, source-cited review records, not
by editing output counts. Preserve original geocodes and ambiguity dispositions;
require complete reliable case-address agreement; count each housing account
once. For the manually reviewed apartment-discrepancy case, carry the explicit
case-specific property evidence and unresolved apartment flag through a reviewed
implementation path. Do not falsely mark its automatic apartment match complete,
generalize its address link to other cases, or weaken the automatic matcher.
The current production crosswalk does not ingest these manual review records;
that integration remains to be implemented and validated. Rebuild the shared
unit/property geography before dependent measurements
and clusters. Preserve a before/after snapshot and verify filing conservation,
county/City coverage, source dates, unit reconciliation and annual/paired
assignment agreement. Do not interpret this review as an already measured
change in Part 1 results.

Local evidence is produced by `scripts/audits/residential_property_batch1.R`
under ignored `output/residential_property_batch1/`. It includes the 162-case
address cohort, seven geocode locations, county account attributes and links,
nearest-parcel distances, operator map snapshots, Bell phase matches, Domain
unit reconciliation, source hashes and review maps. Detailed case IDs and
apartment addresses remain in that ignored directory. The script asserts the
162-case total, 949 Bell map units, 438 Domain map units and the 31/22/1 Bell
automatic candidate split. That automatic output remains unchanged by manual
review; case-specific court evidence is recorded separately in `reviewed_cases/`.
Completed resolutions have review status `property_verified_court_record`;
the apartment-discrepancy case separately records
`apartment_review_status: unresolved_conflicting_unit_numbers`,
`apartment_number_verified: false`, and a case-specific court-evidence basis.
Its superseded partial review is preserved under `reviewed_cases/history/`
and must not be counted as a second case.
The current review total is 54 supported Bell property locations, including
one with unresolved apartment detail. No property lookups remain pending.
The workbook still reflects the October 2 snapshot. The subsequent production
location update applies all 71 supported case links while retaining the
original court evidence and apartment discrepancy. Unit counts are unchanged.
