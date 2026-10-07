# Hays eviction property geography screening

Current JP4 source and refreshed results: [October canonical-report update](hays-eviction-jp4-update-2026-10.md). This document preserves the September snapshot.

Reviewed September 10, 2026. This follows the [PDF intake and portal pilot](hays-eviction-intake-2026-09.md).

This audit records the screening state before collection. The subsequent
[address collection audit](hays-eviction-address-collection-2026-09.md) documents
completed lookups for all 506 active cases and updated date/review counts.

## Result

The 1,157 supplied cases remain intact. Screening recurring property names
reduces the active address-lookup queue to **506 cases**, with **651 deferred**
in 16 property groups outside the project's Austin FULL-purpose boundary.
This is reversible property-level triage; it does not verify individual eviction
premises, establish an Austin case count, or demonstrate source completeness.

| Active queue component | Cases |
| --- | ---: |
| JP5 filings in April 2, 2024–April 1, 2026 | 170 |
| JP4 case-number years 2024–2026, filing date still unknown | 73 |
| Older JP4 case-number years, filing date still unknown | 67 |
| Known filing dates outside the current two-year window | 196 |
| Total | 506 |

The first two rows comprise priority 1: **243 cases**. A case-number year
orders work only; it never substitutes for a filing date. One retained case
already has a portal observation, leaving 505 unstarted active lookups.
The other pilot case is in a deferred property group. Neither pilot observation
is a verified premises address.

Twenty identified property groups account for 743 cases. Nineteen groups have
parcel geometry; all nineteen lie outside the selected city boundary. Three
of those groups have address inconsistencies and remain active: Harvest Meadows
(55 cases), Anthem Ledge Stone (27), and Belterra Springs (3). The View at
Belterra (7) remains active because its parcel identity is unresolved. The
remaining 414 active cases have no established property-group identity.
No case is deferred merely because its plaintiff or postal city says Buda,
Dripping Springs, or Austin. Individuals and landlords with scattered holdings
remain unresolved unless a specific property identity has been established.

## Evidence and method

- `config/hays_eviction_property_groups.json` records manually reviewed plaintiff
  patterns, property addresses, CAD IDs, primary-source links, evidence, and
  identity-review flags. Matching is restricted by court. Ambiguous matches
  stop execution. The generated alias table exposes every match for review.
- County property and ownership CSV exports dated April 28, 2026 supply parcel
  situs addresses, property labels, and ownership corroboration. Property names
  and official property pages support aliases; ownership alone does not establish
  the premises for a historical case.
- `data/raw_parcels/hays/hays_parcels.gpkg` supplies county parcel polygons.
  A fresh [county GIS query](https://services.arcgis.com/0L95CJ0VTaxqcmED/arcgis/rest/services/EXTERNAL_hcad_parcels/FeatureServer/0)
  confirmed that 11 of the 25 registered IDs were absent from that service.
- The official [Texas 2025 parcel resource listing](https://api.tnris.org/api/v1/resources?collection_id=0fa04328-872e-481c-b453-126a74777593)
  identifies the Hays archive. Its [public download](https://s3.amazonaws.com/data.tnris.org/0fa04328-872e-481c-b453-126a74777593/resources/stratmap25-landparcels_48209_lp.zip)
  supplies all 11 missing IDs. The archive's Hays dataset is dated March 2025.
  Numeric `Prop_ID` values match CAD `QuickRefID` after adding the `R` prefix;
  situs addresses and owners corroborate the matches. Archive size, integrity,
  download provenance, and SHA-256 are saved under `data/raw_parcels/hays/txgio_2025`.
- Geometry is compared with `data/BOUNDARIES_jurisdictions_20260429.geojson`,
  selecting `city_name == CITY OF AUSTIN` and `jurisdiction_type == FULL`.
  All selected boundary components are retained. Whole registered parcel
  geometries are unioned by property and compared in EPSG:26914 (meters).
  Missing geometry, intersection, identity conflicts, or a distance of 100 m
  or less prevent deferral. The closest mapped property is about 1.30 km away.

All 25 registered parcel IDs now have geometry, but the View at Belterra has
no registered parcel ID. The county geometry is preferred where available;
the state archive fills gaps. Older parcel geometry and a current boundary
are appropriate to preliminary screening, not proof of historical municipal
membership. No coordinates from a rejected Austin geocoder result were used:
the View at Belterra query matched a different street and ZIP.

## Files and use

- `output/hays_eviction_austin_lookup_queue.csv`: active cases, ordered by
  priority, with court/case keys, source fields, group evidence, portal status,
  candidate address, and current-window status. The filename denotes the
  Austin review queue, not confirmed Austin cases.
- `output/hays_eviction_deferred_outside_property_groups.csv`: all deferred
  cases with their property evidence; these can be restored to the queue.
- `output/hays_eviction_case_geography_screen.csv`: every source case and
  its triage decision.
- `output/hays_eviction_property_screen.csv`,
  `output/hays_eviction_property_parcel_evidence.csv`, and
  `output/hays_eviction_property_aliases.csv`: property, parcel, and raw-name
  audit tables. Distances are from whole polygons to the city boundary.
- `output/hays_eviction_screened_properties.geojson`: mapped property polygons
  for independent GIS inspection.
- `output/hays_eviction_geography_screen_summary.json` and
  `output/hays_eviction_geography_screen_counts.csv`: totals and input hashes.

Keep new address observations in `data/hays_eviction_address_review.csv`;
generated outputs are replaced when screening is rerun. Source PDFs, the
manual review log, pipeline scores, and coverage definitions are unchanged.
Raw data and generated CSVs follow the repository's existing local-data ignore
rules; the raw Hays directory is also excluded by `.gitignore`.

Reproduce from the repository root after PDF extraction:

```sh
python3 scripts/data/hays_eviction_fetch_property_geometry.py --state-archive
Rscript scripts/data/hays_evictions_screen_properties.R
```

The fetch step caches the county query and downloads/extracts the official
state archive if absent. R requires sf, dplyr, readr, jsonlite, and digest.
Checks verify unique case keys, exhaustive/nonoverlapping queue partition,
unambiguous aliases, and that only conflict-free outside groups are deferred.
An independent positive check confirms that known Austin Hays parcel R132593
intersects the selected boundary, guarding against an empty/wrong-city filter.
The source and manual-review hashes are checked across a repeated screening run.

## Next collection step

Start with the 243 priority-1 cases, recovering JP4 filing dates as well as
addresses. Record raw party addresses and identify the eviction premises;
conflicting addresses require case-specific evidence. Apply the city boundary
to reviewed premises and preserve unresolved cases. Audit case-level samples
of deferred groups before treating them as final geographic exclusions.

The queue is smaller but is not yet a small, confirmed set of Austin cases.
At an illustrative one to two minutes per lookup, priority 1 would require
roughly four to eight hours before difficult address verification; this is
not measured throughput. JP4's inactive-status selection remains a separate
coverage problem even after addresses are recovered. A filing-date-selected,
all-status report is still needed before claiming complete annual coverage.
