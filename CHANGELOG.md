# Analytical changelog

Major design decisions and consequential corrections, newest first. This is a
short running record, not a copy of every model run. Current methods describe
what the code does; decision records explain why; current audit reports give
the results and validation. Git records code changes.

## 2026-10-07 — Complete residential repairs and expand the H3 surface

- Promote the accumulated residential, ownership and case-geography repairs.
- Cover the entire adopted April 29, 2026 full-purpose boundary: 7,950 H3 cells,
  preserving all original 7,027 IDs; 6,196 remain within center-based City scope.
- Rebuild selected current/paired measures and both cluster analyses. Part 1
  eligibility rises from 2,677 to 2,693; 97.4% of shared cells retain their
  substantive profile. Part 2 eligibility rises from 2,505 to 2,515.
- Recent low-unit filings fall from 215 in 77 cells to 106 in 55 cells. This
  residual is not a claim that every remaining case is mislocated.
- See [decision 0020](docs/decisions/0020-full-purpose-h3-grid.md) and the
  [completed rebuild report](docs/audits/residential-cluster-rebuild-2026-10.md)
  for validation, matched-profile changes, and limitations.

## 2026-10-07 — Wire the remaining high-value residential follow-ups

- Share a geocode precision gate across paired/current and annual eviction
  builders. Stage 76 coarse-only filings as unassigned and recover eight false
  conflicts; retain source evidence and avoid whole-cell suppression.
- Extend the reviewed Ocotillo property reference to a narrow base-address
  alias, moving nine additional recent filings to cell 596.
- Recover 2025 ownership evidence for R577451 and R605025 through pinned,
  reviewed reference accounts, preserving certified-only sensitivities.
- Integrate 793 Oak Ranch homes (+791 net units), 24 individual-home case links
  and independently classified source-year ownership. All 793 classify in
  2025; 208 remain unknown in 2024. Preserve the 39 deferred home records.
- Combined staging diagnostic: 161 low-unit filings in 65 cells, versus the
  current production 215 in 77. Focused checks pass; **production outputs,
  clusters and full-suite validation remain deferred**. See the
  [follow-up audit](docs/audits/residential-followup-2026-10.md) for evidence,
  recovery queues and exact accounting.

## 2026-10-07 — Reconcile residential geography repair status

- Publish the [consolidated status report](docs/audits/residential-geography-closeout-2026-10.md)
  covering review issues 3, 7 and the Oak Ranch portion of 26. Distinguish
  production-applied corrections, staged ownership/home fixes, provisional
  assumptions and explicitly deferred coverage gaps.
- Independently reconcile current outputs: 1,195 of the 1,219 original audited
  filings have at least 20 units in their assigned cells, including 60 using
  Ben White's provisional denominator. The remaining 24 have supported staged
  Oak Ranch links. The broader low-unit total is still 215 filings in 77 cells,
  down from 1,614; 191 of these are outside the original audit cohort.
- Document 941 additional accepted filings with adequate unit denominators but
  other cluster exclusions, and distinguish unresolved property links from
  unresolved case locations. Preserve the user-directed deferral of 39 Oak
  Ranch home records and the batched production rebuild/full-test plan.
- Add a read-only reporting script with source hashes and aggregate checks.
  **No production outputs or model fits changed.**

## 2026-10-07 — Review Oak Ranch's 44 address and identifier exceptions

- Resolve 29 location/count exceptions: 22 street suffix differences, one
  street-name typo, two missing serials with other home evidence, and four
  conflicting space fields with unique primary addresses. Preserve all raw
  fields and flags; the space conflicts and missing serials remain visible.
  Disputed spaces cannot establish a case-to-home match.
- Stage 793 accepted in-grid home locations: 791 additions plus the two
  already-counted A2 units. All 24 audited filing links remain unchanged.
  Cell 3767 has 194 staged homes and cell 3769 has 150; these are partial
  inventory diagnostics, not production denominators.
- Retain 15 unresolved home locations (ten shared addresses, five missing
  City matches) and 24 points outside the fixed grid. City subaddresses and
  county notes/links did not resolve the remaining home locations.
- Preserve the original baseline; save decisions, sources, updated staging
  files and hashes in
  `data/reviewed_manufactured_housing/oak_ranch_20261007/exception_review_01/`.
  The focused staging checks pass. **No production rebuild or full test
  suite run; both remain deferred for the larger repair batch.**

## 2026-10-07 — Stage Oak Ranch manufactured-home inventory and case links

- Read the original July 20, 2025 TCAD export, including all M00109 subdivision
  records. Find 832 active home accounts: 830 omitted M1 accounts and two A2
  accounts already carrying one unit each at park-level coordinates. The
  earlier 237-account diagnostic was only a street subset. Extracted serial
  evidence exists for 830 accounts, with no repeated serials in the inspected
  county records; the two missing identifiers remain flagged.
- Query the independent City of Austin address-point service and preserve its
  October 7, 2026 response. Find 826 unique point candidates inside the two park
  parent polygons. Stage 764 unambiguous in-grid home references (762 additions
  and two relocations), distributed across 11 cells. Preserve 68 exceptions:
  24 outside the fixed grid and 44 with address/identity issues. A unique
  number/street match with a conflicting street suffix remains a review item.
- All 24 remaining original-audit filings match 17 unambiguous home accounts
  and independent City points, including the exported space numbers where
  supplied. Their current hex assignments agree with the reviewed home points.
  The staged clean subset provides 188 units in cell 3767 and 148 in 3769;
  these are partial-inventory diagnostic counts, not adopted denominators.
- Preserve the parent accounts for geography; do not add the parent's 377
  improvement units to individual homes. The meaning of that parent quantity
  remains unverified. Do not transfer park-owner classifications to homes.
- Evidence, hashes, home/case crosswalks and the remaining review queue are in
  `data/reviewed_manufactured_housing/oak_ranch_20261007/manifest.json`.
  Reproducible review stages are `scripts/audits/oak_ranch_source_extract.py`,
  `oak_ranch_inventory.py`, `oak_ranch_geography.R` and `oak_ranch_cases.py`.
  Focused checks verify uniqueness, one unit/account, the two existing units,
  spatial containment, all 24 links and the explicit exception queue.
- **Staged only; no production unit, filing or cluster outputs changed.**
  Resolve the remaining records or explicitly document partial coverage before
  integration. Then apply accepted home and filing geography together and run
  the deferred measurement/clustering rebuild and full validation suite.

## 2026-10-07 — Recover cell 2326's complete source-year mailing address

- Williamson reconciliation v4 completes a GIS `MAIL_LINE1` at its observed
  60-character limit from the same record's `MAIL_ADDR`, only when the shorter
  field is a strict prefix, there is no second delivery line, and the full
  address's city/state/ZIP corroborate the separate locality fields. Preserve
  the raw source fields and record full-address recovery in reconciliation QA.
- R066465 (MLVI Marthas Vineyard Apartments LLC) has suite 1050 in both the
  certified report and full GIS address in 2024 and 2025. The shortened GIS
  field ends at 105. Completing that field, together with the preceding
  heading repair, supports its corporate-ownership classification in both
  years without a parcel-specific override.
- The diagnostic calculation raises cell 2326's 2025 usable ownership-unit
  coverage from 19.41% to 99.73%, exceeding the existing 95% threshold. Its
  roughly 294 modeled apartment units and filing locations are unchanged.
- The 36-test focused reconciliation suite passes. Across the 202 flagged
  mailing parcel-years, cumulative v3/v4 fixes resolve ten in each year;
  182 retain their reconciliation status. The full-field repair completes
  five GIS parcel records in 2024 and four in 2025; two still fail certified
  corroboration and remain unresolved. Replay evidence:
  `tmp/ownership_address_audit_20261007/cell2326_repair_status.json`.
- **Production rebuild remains deferred.** Keep batching data fixes; the
  ownership/features/clustering/maps rebuild and full test suite listed below
  remain pending. No new cluster assignment is claimed.

## 2026-10-07 — Fix clipped ownership mailing headings; rebuild deferred

- Williamson reconciliation v3 recognizes a clipped attention, care-of, or
  trust heading followed by an intact delivery address. Require a printed
  column-boundary truncation, matching street/PO box and locality, and an
  alphabetic-only heading extension. Preserve owner and situs corroboration;
  do not relax missing-address, changed-owner, or changed-unit/box checks.
- The focused reconciliation suite passes 34 tests. Replaying all 202 flagged
  mailing parcel-year records identifies seven repairs in 2025 and six in
  2024; the other 189 retain their reconciliation status. The seven current
  parcels represent 884.27 modeled units. A diagnostic calculation restores
  ownership eligibility for cells 2169, 2354, 3214 and 3261; this is not a new
  fitted clustering result.
- At the user's direction, batch data fixes before the next model run.
  **Pending:** rebuild ownership snapshots/index, Part 1 measures/features,
  Part 2 paired matrix, cluster fits and interpretations, maps/summaries, and
  run the full test suite. Existing production outputs still reflect v2;
  source/code checksum checks may flag them as stale until that rebuild.
- Local replay evidence and pending status:
  `tmp/ownership_address_audit_20261007/repair_status.json`.

## 2026-10-07 — Apply the second property-review batch

- Recover Caliza's 270 units and place Nexus's existing 294 units at explicitly
  reviewed references inside their parcels and the fixed grid. Verify City
  containment and retain the original reference coordinates.
- Apply 31 case reviews across Canyon Creek, Caliza, Nexus and Ocotillo. Use
  documented project totals of 332 at Canyon Creek and 308 at Ocotillo.
- Preserve original ambiguity exclusions. Keep the 24 Oak Ranch-area filings
  unresolved: omitted M1 accounts share parent-level coordinates, and park
  totals have not been reconciled with individual homes.
- See [decision 0019](docs/decisions/0019-reviewed-boundary-property-references.md)
  and the [batch-2 audit](docs/audits/residential-property-batch2-2026-10.md).

## 2026-10-07 — Adopt a provisional Ben White denominator

- Add omitted account 291453 with 170 housing-unit equivalents, following the
  user's explicit decision to adopt and document this assumption. Preserve raw
  commercial classification and zero reported units; mark the selected count
  provisional with low confidence.
- Cite the 2019 listing's approximately 178 rooms and the 2018 county guide's
  100 beds separately from the adopted 170. The sources support residential
  room rentals but do not verify 170 dwellings.
- The property's 60 recent filings yield 35.29 filings per 100 assumed units.
  Low-unit-cell filings fall from 306 to 246, with no case reassignment or
  change to source exclusions. The property remains outside clustering because
  historical ownership is unknown.
- Rebuild dependent measures and clusters. One of 2,676 Part 1 cells changes
  profile; neighborhood predominant profiles remain unchanged. Part 2 retains
  its 2,504-cell sample and baseline profiles, with one later-year change.
- See [decision 0018](docs/decisions/0018-ben-white-provisional-denominator.md)
  and the [incremental audit](docs/audits/ben-white-provisional-units-2026-10.md).

## 2026-10-07 — Repair reviewed housing counts and Domain geography

- Promote documented totals for Bell Springs (400), Monarch Bluffs (330), Asher
  (452) and the Villages at the Domain (412 southern plus 26 northern homes).
  Move the misplaced Domain accounts and recover the omitted Building P account;
  retain unknown historical ownership for the new account.
- Rebuild the dependent measures and clusters. Current low-unit filings fall
  from 334 to 306; all 11,784 covered recent filings remain. Part 1 gains one
  eligible cell, with eight of 2,675 previously eligible cells changing profile.
- Withdraw unsupported blanket links within the mixed northern Domain parcel:
  36 paired-ledger cases (37 in the longer annual ledger) keep their original
  geocodes. Preserve the earlier 71 manual case corrections and all exclusions.
- Flag Ben White's 60 recent filings for an unverified independent-housing-unit
  denominator; beds and rooms are not substituted for housing units.
- See [decision 0017](docs/decisions/0017-reviewed-unit-counts-and-geography.md)
  and the [production audit](docs/audits/reviewed-unit-properties-2026-10.md).

## 2026-10-07 — Apply reviewed property locations to 71 filings

- Apply the completed Bell Southpark, Asher and Monarch reviews through a
  case-specific, source-pinned property layer shared by annual and paired counts.
  Preserve original addresses, exclusions and unit counts; retain the unresolved
  apartment detail on one court-supported Springs property assignment.
- Move 71 filings out of low-unit cells: the recent low-unit total falls from
  405 to 334, while all covered-cell filings remain 11,784. The eligible Part 1
  sample remains 2,675 cells, now containing 10,579 recent filings.
- See [decision 0016](docs/decisions/0016-reviewed-case-property-locations.md) and
  the [production audit](docs/audits/reviewed-property-locations-2026-10.md).

## 2026-10-02 — Repair residential coverage and filing/property geography

- Recover 35 active Williamson residential accounts using independent housing
  evidence and explicit reference-account links; 34 contribute positive units
  after the existing final land-use review. Reference-only accounts do not add
  duplicate units.
- Use one verified residential-property crosswalk for paired and annual filing
  assignments. Preserve original geocodes, accepted case totals, ambiguity
  exclusions, court coverage and the 20-unit threshold.
- Resolve denominator/geography support for 1,002 of the 1,219 filings in the
  audited 41 cells. The remaining 217 filings stay flagged for review. Across
  all covered cells, low-unit-cell filings fall from 1,614 to 405.
- Rebuild affected measurements, clusters, local maps and annual labels. Part 1
  includes 2,675 cells and 10,508 recent filings in eligible cells, versus 2,660
  and 9,528 before. No Part 3 forecast training or deployment is performed.
- See [decision 0015](docs/decisions/0015-residential-property-geography.md) and
  the [repair audit](docs/audits/residential-property-repair-2026-10.md).

## 2026-09-11 — Keep candidate cells when eviction cases are ambiguous

- Keep ambiguous cases flagged and unassigned. Remove the whole-cell veto in
  current/paired eviction snapshots and the annual outcome panel; retain
  candidate-location audit counts and flags. Covered cells count accepted,
  uniquely mapped filings, and a zero describes that proxy.
- Restore 103 Part 1 cells (2,557→2,660) and 139 paired Part 2 cells
  (2,351→2,490), without losing any previously eligible cells. The restored
  current cells contain 2,619 recent and 2,217 previous-window valid filings.
- Keep geocodes, raw mapped counts, source coverage, units, rates, feature
  weights and component-completeness requirements unchanged. Refit eviction
  scales and affected clusters on the revised support, and review profile
  labels before updating maps. Rebuild annual labels; 2026 remains partial
  and Part 3 ML remains paused.
- See [decision 0014](docs/decisions/0014-eviction-ambiguity-keeps-cells.md) and
  the [before/after audit](docs/audits/eviction-ambiguity-policy-2026-09.md).

## 2026-09-10 — Harmonize Part 1 and Part 2 measurement

Follow-up correction: eviction eligibility now checks only the two annual
windows that enter each score, not all history since January 2022. The pair
requires April 2023–April 2025 and April 2024–April 2026 respectively. Retain
potentially in-window missing/conflicting dates and all other safeguards;
older-only ambiguity no longer removes a cell. Source filing counts and
geocodes are unchanged. Downstream results replace the preceding run in place.
The change restores 60/109 usable eviction scores in 2025/2026. Full Part 1
eligibility rises from 2,456 to 2,557 (+101); paired Part 2 rises from 2,300 to
2,351 (+51), with no eligibility losses. Current profiles remain recognizable
(eviction group 81→87); Part 2 movement is 35.0% and fixed/refitted agreement
95.5%. The compact audit is output/part1/eviction_window_change_summary.csv.

- Apply the corrected complete-component recipes to the current Part 1 model,
  not only the historical comparison. Retire available-component averaging and
  the old eviction percentage-change/expanding-history-share recipe.
- Part 1 uses all eligible cells at the April 1, 2026 cutoff. Rent requires a
  coherent reliable 2014/2019/2024 block-group history, otherwise a coherent
  tract history. Ownership uses jointly known parcels in 2025 only, with the
  same 20-unit and 95% parcel/unit evidence thresholds.
- Part 2 additionally holds rent source level and ownership parcel support
  fixed across the paired dates and requires both dates to pass. Cells may
  drop out; missing evidence is not zero pressure.
- Fit current Part 1 component bounds and cluster standardization on 2026
  evidence. Part 2 retains its 2025 reference scales for temporal comparison.
  Shared formulas do not imply identical samples, scores, or cluster IDs.
- Replace superseded generated outputs in place. Preserve raw source vintages,
  reviewed evidence, current artifacts and checksum provenance needed for
  reproduction; do not create another archive directory for each failed run.
- First harmonized refit, before the window relaxation above: 2,456 cells,
  including all 2,300 then-current Part 2 cells plus
  156 current-only eligible cells. Refresh labels, maps and neighborhood
  summaries; retire 24 obsolete generated selection/island-review artifacts.
  Seven clusters remain provisional; old spatial-holdout findings are not
  transferred to the new fit. See the [current audit](docs/audits/part1-harmonized-measurement-2026-09.md).
- ML remains paused. See [decision 0013](docs/decisions/0013-harmonized-measurement.md)
  and [current measurement methods](docs/methods/current-measurement.md).

## 2026-09-08 — Correct the historical measurement design

- Require complete fixed recipes at both dates; missing subcomponents no longer
  silently change weights. Retain rent level, growth and acceleration through
  an all-six-vintage reliable BG/tract hierarchy.
- Use recent event rates and signed rate differences for evictions; remove the
  expanding-history recent share. Replace 311 percentage change with signed
  rate change. Zero signed change scores 50, not a literal no-risk zero.
- Regenerate the 2025/2026 comparison in place: 2,300 eligible cells; 34.9%
  fixed-definition movement; 95.5% fixed/refitted 2026 agreement.
- Review qualitative concern labels against the corrected profiles, separately
  from nominal cluster IDs. See the [current Part 2 audit](docs/audits/part2-cluster-comparison-2026-09.md).

## Earlier decisions

The [decision index](docs/decisions/README.md) records the grid, unit surface,
ACS allocation, seven-domain architecture, seven-cluster choice, update design
and neighborhood summaries. Source investigation and import evidence are
indexed in [the documentation guide](docs/README.md). Those records predate
this single chronological log; their dates are not being reconstructed here.

## Maintenance rule

Add a dated entry when the measurement, analytical universe, denominator,
interpretation or update design changes materially. Link to current methods
and the relevant decision. Routine reruns and cosmetic edits need no entry.
An as-run preservation check is evidence that a run did not mutate its inputs;
it is not a permanent prohibition on an explicitly approved later rebuild.

### Staged residential residual repairs — 2026-10-07

- Recover 676 independently owned/classified manufactured-home accounts across
  five parks; retain three duplicate/conflicting accounts outside the supplement.
- Correct Olivine/Bennett unit geography while preserving their existing count
  methods; support explicit shared park-phase references and 18 additional
  case reviews (144 total).
- Full staged property crosswalk removes false Olivine links to Elm Ridge Lane.
  Below-20-unit recent filings fall from 161/65 cells to 108/56 cells, with no
  additional case exclusions. Hudson's 276-unit City-boundary allocation remains
  unresolved. Production rebuilds, clustering and full tests remain deferred.
- See `docs/audits/residential-residual-repairs-2026-10.md` for evidence, source
  links, limitations and reproducible staging commands.
