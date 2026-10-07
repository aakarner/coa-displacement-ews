# Hays portal address collection

Current JP4 source and refreshed results: [October canonical-report update](hays-eviction-jp4-update-2026-10.md). This document preserves the September snapshot.

Completed September 10, 2026, following the [property geography screen](hays-eviction-geography-screen-2026-09.md).

## Results

All **506 active cases** received a lookup. The 243 initial priority cases were
completed first, followed by the remaining active records. The prior active
pilot observation was preserved; this pass added 504 case-detail observations
and one unsuccessful case lookup.

| Result in the active queue | Cases |
| --- | ---: |
| Defendant address available | 504 |
| Single candidate address | 440 |
| Multiple distinct defendant addresses | 64 |
| No defendant address shown | 1 |
| Case not found | 1 |

The single/multiple rows partition the 504 cases with addresses. They are
**party addresses, not verified eviction premises or confirmed Austin cases**.
Even a single defendant address may have been updated since filing. No case
was assigned a verified premises address or an Austin boundary result.

Recovered filing dates place **231 active cases** in April 2, 2024–April 1, 2026:
170 JP5 and 61 JP4. This includes 209 single-address candidates, 21 cases with
multiple addresses, and one missing address. The other 275 active cases fall
outside the window. All active cases now have an effective filing date; the
unavailable JP5 case retains its source-PDF filing date.

The 651 cases deferred by property screening remain in the source dataset.
One had an earlier pilot observation; 650 were not looked up. Deferral remains
preliminary, and collecting the active cases does not establish complete Hays
filing coverage. JP4's supplied report selects inactive status dates rather
than a complete filing cohort.

## Exceptions

- **F24-089J5:** the public Party Information table contains only the plaintiff,
  Southwest Leasing Solutions. Its address was not substituted for the defendant's
  address. A second live check confirmed the rendered table has no defendant entry.
  The portal filing date is August 14, 2024.
- **F21-005J5:** exact case-number search returned zero records. An exact search
  for plaintiff Raquel Granados returned only F21-021J5, filed May 25, 2021,
  involving the same named parties. That is a different case, so its address
  was not borrowed. F21-005J5 retains the PDF filing date January 20, 2021.
- The 64 cases with multiple distinct defendant addresses have no automatically
  selected candidate premises. Their raw addresses and party associations remain
  available for review. Minor differences in case, abbreviations, or unit text
  are not silently collapsed; some conflicts may resolve during normalization.

## Collection method and provenance

Used the county's [public portal](https://portal-txhays.tylertech.cloud/PublicAccess/default.aspx),
Civil, Family & Probate search, by exact case number with all statuses selected.
Every successful search had one exact case-number result. Case-detail headers
were checked for the requested number, court, and filing date before saving.
Browser-assisted batches used visible search controls and read-only extraction
of the rendered Party Information table. Verification interruptions and delayed
page loads stopped the batch; completed records were already saved locally.

Raw structured observations are in
`data/raw_hays_evictions/portal_observations/2026-09-10.jsonl`.
Each includes case number, court, observation date, portal filing date, exact
case-detail URL, and party roles/names/address lines. Demographics, DOB, and
financial details were excluded. `search_attempts.jsonl` documents the unavailable
case and the fallback search. No emails or external messages were sent.

The persistent review log is `data/hays_eviction_address_review.csv`. Its import
keeps original filing dates separate from portal dates and preserves previous
manual observations. Unit text remains embedded in the raw/candidate address;
the separate unit column is not newly populated. The JSONL preserves the exact
party-to-address associations, including parties with no address.

## Outputs

- `output/hays_eviction_collected_active_cases.csv`: all 506 active cases with
  collected addresses, filing dates, provenance, and review flags.
- `output/hays_eviction_current_window_address_review.csv`: the 231 cases in
  the current analysis window, including unresolved addresses.
- `output/hays_eviction_single_candidate_addresses.csv`: 440 single-address
  candidates for normalization, premises review, and geocoding.
- `output/hays_eviction_address_review_exceptions.csv`: 66 cases requiring
  resolution of multiple addresses, a missing address, or an unavailable case.
- `output/hays_eviction_collection_completion.json`: reconciled counts and
  validation results. `output/hays_eviction_portal_collection_summary.json`
  describes imports across the full 1,157-case review log, including the two
  earlier pilot observations.

The geographic-screen outputs have been refreshed with portal dates and statuses.
They still preserve the same 506/651 active/deferred partition. Initial priority
counts in the earlier screening audit describe the state before collection.
Raw Hays records follow the repository's local-data ignore convention.

Rebuild the review outputs from already collected observations:

```sh
python3 scripts/data/hays_evictions_import_portal.py
Rscript scripts/data/hays_evictions_screen_properties.R
python3 scripts/data/hays_evictions_collection_report.py
```

`scripts/data/hays_evictions_browser_collector.mjs` preserves the collection
helper for use inside an authorized CUA browser session. It requires a selected
Case search form and stops at unavailable results or verification; it is not
an unattended network scraper. It takes browser handles, a filesystem handle,
repository root, and observation date as arguments.

## Validation and remaining work

Confirmed 504 unique newly observed case keys and 504 unique case-detail URLs;
every active case has an outcome. Portal filing dates agree with the source
dates for every successfully retrieved JP5 case. Source case fields, previous
manual observations, verified-address fields, and boundary-status fields were
preserved. Reimporting produced no change to the review-log hash. A subsequent address-normalization check found SID labels embedded in 85
portal address lines. These non-address identifiers were removed from the saved
observations and all regenerated address outputs; the collector and importer now
filter identifier labels. Removing them resolved two spurious address conflicts.
A scan of the cleaned address fields found no SID, DOB, demographic, or financial
labels. The counts above reflect this cleanup.

Collection is complete for the active queue. Next, normalize and geocode the
209 current-window single-address candidates, establish whether they identify
the eviction premises, and apply the exact Austin FULL-purpose boundary. Review
the 21 current-window address conflicts and request the missing defendant/premises
information for F24-089J5. The unavailable older case and the separate JP4 source
coverage problem remain documented; neither should become a zero count.
