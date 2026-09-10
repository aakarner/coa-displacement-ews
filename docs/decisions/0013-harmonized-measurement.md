# 0013: Shared measurement recipes with snapshot-specific eligibility

- **Status:** Accepted
- **Decision date:** September 10, 2026
- **Context:** Measurement corrections adopted for the retrospective comparison
  had not reached the current Part 1 baseline. The user requested integration,
  current-cutoff coverage in Part 1, temporal consistency in Part 2, and a
  compact changelog rather than repeated archives of superseded model runs.

## Decision

Use complete fixed equal-weight composites in both parts: three rent, five
vulnerability, three demolition, two eviction, three selected-311, three
ownership and three amenity-category scores. Retain valid zero observations;
never treat unknown evidence as zero or average only the available terms.

Evictions combine recent filings per 100 fixed units with the signed difference
from the previous 12-month rate. Selected 311 combines recent rate, density
and signed rate change. The change scales are symmetric, with zero at 50.
The demolition and amenity positive-change terms are unchanged.

Part 1 is a single-cutoff reconstruction, not the two-date complete-case sample.
Rent chooses one reliable BG or tract level across its three required vintages.
Ownership uses this year's jointly known parcels and applies 20-observed-unit
and 95% unit/parcel coverage screens. Part 2 additionally fixes those choices
across dates: six rent vintages and a jointly observed parcel cohort. Keep all
exclusion reasons visible. This supplements decisions 0005–0009.

Part 1 calibrates component bounds and cluster means/SDs on its 2026 reference;
Part 2 retains its frozen 2025 reference. Formula consistency is required;
identical numerical scores or cluster IDs are not. Keep k=7 provisionally and
review names and concern tiers against the newly estimated profiles.

## Consequences and limits

The current sample can exceed the paired sample but still excludes unsupported
city cells. Rent fallback gains reliability at the cost of spatial detail;
historical ACS boundaries are not fully harmonized. Court, geocoding, permit
and amenity-coverage limitations remain. The reconstruction is not a claim of
past information availability or a calibrated displacement probability.

## Recordkeeping

Maintain the root changelog and current methods/results. Overwrite obsolete
generated runs; retain source vintages and provenance needed to reproduce the
current work. Existing source-evidence archives are not indiscriminately
deleted. Checksums document preservation during a run, not an immutable ban
on authorized future Part 1 rebuilds.

## Revisit

September 10 follow-up: restrict eviction source-coverage and localizable
ambiguity eligibility to each snapshot's two scored annual windows (2023-04-02
through 2025-04-01; 2024-04-02 through 2026-04-01). The historical screen back to
January 2022 was more conservative than the new score required. Retain all
source rows when resolving potentially relevant cases, including unknown dates
and conflicting dates across window boundaries. Keep reliable-geocode, court,
city, 20-unit, complete-component and zero-in-window-ambiguity safeguards.
Other indices and eligibility rules are unchanged. Older records remain
unscored audit evidence, not an eligibility condition.

Reconsider source coverage, rent reliability, unit thresholds, qualitative
labels and k when sources or evidence change. Document the choice here or in
a successor decision and add a changelog entry. ML remains paused.
