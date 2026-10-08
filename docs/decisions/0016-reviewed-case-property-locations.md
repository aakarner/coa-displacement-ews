# 0016: Apply reviewed case-specific property locations

- **Status:** Accepted; user authorized production implementation
- **Decision date:** October 7, 2026
- **Scope:** The 54 Bell Southpark, six Bridge at Asher and 11 Bridge at Monarch
  Bluffs filings reviewed in the first residual-location batch

## Decision

Extend the shared property crosswalk with explicitly reviewed links keyed to a
county/court-namespaced case. For this batch, each case's filing date, court,
complete set of reliable source addresses and geocode coordinates are pinned.
Court screenshots, operator-map evidence and property/account review evidence
are preserved with file hashes in local private inputs. The tracked batch
manifest pins those inputs; missing or changed evidence fails the build.

Locate each reviewed case using the existing operational unit reference of its
verified property project. Derive the cell from that reference, require positive
project units in one cell, and verify agreement with the reviewed destination.
Do not create global address aliases: apartment numbers recur across Bell's
phases and a later filing with the same address has not necessarily been reviewed.

One Springs case has two conflicting apartment numbers at the same street
address. Its case-specific court plaintiff and DBA, county owner and corroborating
operator-map evidence support Springs as the property. Preserve both original
apartment numbers and an explicit apartment-conflict flag. Property verification
does not establish the exact apartment. This manual assessment does not loosen
the automatic address matcher or silently fix a presumed transposition.

The correction is a second stage after the original case resolver. Existing
excluded/ambiguous cases remain excluded, candidate cells remain usable, and
accepted filing counts are conserved. The reviewed destination must satisfy the
existing City, county and effective court rules. Changed dates, courts, reliable
address sets, geocodes or property reference cells require a new review instead
of silently reusing the old link.

Both paired snapshots (Parts 1–2) and annual counts (Part 3) consume the same
versioned artifact, `verified_residential_parcel_reference_v2`. Their ledgers
retain original cells, proposed property/cell, source of verification, review ID,
reference distance and the apartment-conflict flag. The generated artifact and
its inputs are hash-checked before consumption.

## Scope limits and validation

This update changes locations only. Existing residential unit counts, source
addresses, case inclusion, filing dates, coverage and the 20-unit rule remain
unchanged. Proposed unit-count reconciliations and the unfinished Domain and
Ben White reviews are separate. Part 3 model training remains paused.

Validate exact annual/paired agreement for all 71 reviewed cases, unchanged
assignments for other cases, unchanged original exclusions and annual totals,
and unchanged source/unit-file hashes. Test that an unreviewed case sharing an
address cannot inherit a review, source changes are rejected, original excluded
cases are not rescued, and geographic boundaries remain enforced. Rebuild
measurements, both cluster products and annual labels; compare against a saved
pre-change snapshot. See the [production audit](../audits/reviewed-property-locations-2026-10.md).
