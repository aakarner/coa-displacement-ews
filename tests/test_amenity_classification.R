# Run from repository root: Rscript tests/test_amenity_classification.R
# Synthetic classification/reconciliation tests; no source downloads or geocoding.
suppressPackageStartupMessages(library(dplyr))
source("R/amenity_classification.R")

expect_equal <- function(actual, expected) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
}
expect_error <- function(expr, pattern = NULL) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(result)))
}
taxonomy <- readr::read_csv(
  "config/amenity_categories.csv", show_col_types = FALSE,
  col_types = readr::cols(naics = readr::col_character(), category = readr::col_character(),
    evidence_tier = readr::col_character(), include_in_index = readr::col_logical(),
    name_filter_required = readr::col_logical(), name_pattern = readr::col_character(),
    notes = readr::col_character())
)
fixture <- function(n = 1L) {
  tibble(
    tp_number = sprintf("000%04d", seq_len(n)), loc_number = "00001",
    loc_name = "Fixture Coffee LLC", address_number = "100", address_text = "MAIN STREET STE 2",
    permit_date = "2023-01-01", juris_city = "AUSTIN", loc_city = "AUSTIN",
    loc_state = "TX", loc_zip = "78701-1234", loc_county = "227", naics = "722515",
    first_sale_date = "2024-01-10", out_of_business_date = NA_character_
  )
}
classify <- function(x, cutoff = as.Date("2025-04-01"), tax = taxonomy) {
  amenity_classify_sales(x, tax, cutoff)
}

# Cutoff dates are inclusive; the two windows share no days.
windows <- amenity_windows("2025-04-01")
stopifnot(identical(windows$previous_window_start, as.Date("2022-04-02")),
          identical(windows$recent_window_start, as.Date("2023-10-02")),
          windows$window_months == 18L)
later_windows <- amenity_windows("2026-04-01")
stopifnot(identical(later_windows$previous_window_start, as.Date("2023-04-02")),
          identical(later_windows$recent_window_start, as.Date("2024-10-02")))
boundaries <- fixture(7)
boundaries$first_sale_date <- c("2022-04-01", "2022-04-02", "2023-10-01",
                               "2023-10-02", "2025-04-01", "2025-04-02", NA)
classified <- classify(boundaries)
expect_equal(classified$event_window,
             c("outside", "previous", "previous", "recent", "recent", "outside", "outside"))
expect_equal(classified$core_index_eligible, c(FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE))
expect_error(amenity_windows(NA), "Invalid amenity date")
expect_error(amenity_windows(c("2025-04-01", "2026-04-01")), "Invalid amenity date")
expect_error(amenity_windows("2025-04-01", 0), "Invalid amenity date")
expect_error(amenity_windows("2025-04-01", 1.5), "Invalid amenity date")

# Permit dates remain an explicit eligibility rule, not a publication claim.
permits <- fixture(4)
permits$permit_date <- c("2025-04-01", "2025-04-02", NA, "2025-05-01")
classified <- classify(permits)
expect_equal(classified$core_index_eligible, c(TRUE, FALSE, TRUE, FALSE))
expect_equal(classified$permit_after_cutoff_flag, c(FALSE, TRUE, FALSE, TRUE))
stopifnot(all(classify(permits, as.Date("2026-04-01"))$core_index_eligible))

# Closures do not remove historical opening events from either window.
closures <- fixture(4)
closures$out_of_business_date <- c("2024-02-01", "2025-04-01", "2025-04-02", NA)
classified <- classify(closures)
stopifnot(all(classified$core_index_eligible))
expect_equal(classified$active_as_of, c(FALSE, FALSE, TRUE, TRUE))

# Preserve the established taxonomy and name, residence, and institution rules.
exclusions <- fixture(8)
exclusions$loc_name <- c("Fixture Coffee LLC", "Fixture Ice Cream", "University Dining",
                         "Neighborhood Restaurant", "Fixture Bar", "Fixture Bakery",
                         "Fixture Cafe", "Fixture Tea Shop")
exclusions$naics <- c("722515", "722515", "722511", "722511", "722410", "311811", "722515", "722515")
exclusions$address_text[7] <- "MAIN STREET APT 2"
classified <- classify(exclusions)
expect_equal(classified$core_index_eligible, c(TRUE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, TRUE))
stopifnot(classified$category_classified[2] == "other_snack_non_alcoholic",
          classified$institutional_flag[3], classified$home_business_flag[7])
county_fixture <- fixture(3)
county_fixture$loc_county <- c("105", "227", "246")
expect_equal(classify(county_fixture)$county, c("Hays", "Travis", "Williamson"))
bad <- fixture(); bad$loc_county <- "999"
expect_error(classify(bad), "unmapped county")
bad_taxonomy <- bind_rows(taxonomy, taxonomy[1, ])
expect_error(classify(fixture(), tax = bad_taxonomy), "duplicate NAICS")

# Stable IDs remain text (leading zeroes intact); address/name normalization
# supports geocoding and matching without replacing the stable event identity.
normalized <- fixture()
normalized$juris_city <- NA_character_
classified <- classify(normalized)
stopifnot(classified$event_id == "0000001:00001", classified$city == "AUSTIN",
          classified$zip == "78701", classified$street_key == "100 MAIN ST",
          classified$normalized_name == "FIXTURE COFFEE")

# Exact repeated source rows collapse; conflicting repeated IDs fail closed.
same <- fixture()
stopifnot(nrow(amenity_validate_sales(bind_rows(same, same))) == 1L,
          nrow(classify(bind_rows(same, same))) == 1L)
changed <- same; changed$first_sale_date <- "2024-02-10"
expect_error(amenity_validate_sales(bind_rows(same, changed)), "conflicting duplicate stable IDs")
bad <- same; bad$tp_number <- " "
expect_error(amenity_validate_sales(bad), "missing stable IDs")
bad <- same; bad$loc_number <- NA_character_
expect_error(amenity_validate_sales(bad), "missing stable IDs")
expect_error(amenity_validate_sales(select(same, -first_sale_date)), "schema is incomplete")

# Shared IDs retain the complete archived record; live-only IDs supplement it.
# A deletion from the live extract cannot erase an archived historical opening.
archive <- fixture(3)
archive$loc_name <- c("Unchanged Coffee", "Archived Cafe", "Archive Only Coffee")
archive$out_of_business_date[2] <- NA_character_
live <- fixture(4)[c(1, 2, 4), ]
live$loc_name <- c("Unchanged Coffee", "Corrected Cafe", "Live Only Coffee")
live$first_sale_date[2] <- "2024-02-20"
live$out_of_business_date[2] <- "2024-12-01"
live$first_sale_date[3] <- "2025-06-01"
live$permit_date[3] <- "2025-05-01"
reconciled <- amenity_reconcile_sources(archive, live, "archive-fixture", "live-fixture")
stopifnot(nrow(reconciled$ledger) == 4L, !anyDuplicated(reconciled$ledger$event_id),
          nrow(reconciled$field_changes) == 3L)
expect_equal(reconciled$field_changes$field,
             c("loc_name", "first_sale_date", "out_of_business_date"))
selected <- reconciled$ledger
expect_equal(selected$disposition, c("shared_archive_retained", "shared_archive_retained",
                                     "archive_only_retained", "live_only_supplement"))
expect_equal(selected$source_fields_changed, c(FALSE, TRUE, FALSE, FALSE))
expect_equal(selected$selected_source_id,
             c("archive-fixture", "archive-fixture", "archive-fixture", "live-fixture"))
stopifnot(selected$loc_name[2] == "Archived Cafe", selected$first_sale_date[2] == "2024-01-10",
          is.na(selected$out_of_business_date[2]))
old_events <- classify(selected)
new_events <- classify(selected, as.Date("2026-04-01"))
expect_equal(old_events$core_index_eligible, c(TRUE, TRUE, TRUE, FALSE))
stopifnot(all(new_events$core_index_eligible))
expect_equal(old_events$opening_date[1:3], new_events$opening_date[1:3])
expect_equal(old_events$event_window[1:3], rep("recent", 3))
expect_equal(new_events$event_window[1:3], rep("previous", 3))

# Input order and exact duplicate source rows do not alter the final ledger.
permuted <- amenity_reconcile_sources(archive[3:1, ], live[3:1, ], "archive-fixture", "live-fixture")
expect_equal(permuted$ledger, reconciled$ledger)
duplicates <- amenity_reconcile_sources(bind_rows(archive, archive[1, ]),
                                        bind_rows(live, live[1, ]), "archive-fixture", "live-fixture")
expect_equal(duplicates$ledger, reconciled$ledger)
conflicting <- archive[1, ]; conflicting$loc_name <- "Different Name"
expect_error(amenity_reconcile_sources(bind_rows(archive, conflicting), live, "a", "l"),
             "conflicting duplicate stable IDs")

# Empty source sides are legal; stable-ID reconciliation still preserves the
# nonempty source. This does not itself claim historical coverage completeness.
archive_only <- amenity_reconcile_sources(archive, live[FALSE, ], "a", "l")
live_only <- amenity_reconcile_sources(archive[FALSE, ], live, "a", "l")
stopifnot(nrow(archive_only$ledger) == nrow(archive),
          all(archive_only$ledger$disposition == "archive_only_retained"),
          nrow(live_only$ledger) == nrow(live),
          all(live_only$ledger$disposition == "live_only_supplement"))

# Exact opening aliases count once after eligibility, preferring an archived
# source, but unit, name, date, ZIP, or NAICS differences remain distinct events.
aliases <- fixture(9)
aliases$address_text[3] <- "MAIN STREET STE 3"
aliases$loc_name[4] <- "Another Coffee Company"
aliases$first_sale_date[5] <- "2024-01-11"
aliases$naics[6] <- "722511"
aliases$loc_zip[7] <- "78702"
aliases$loc_name[8] <- "  fixture coffee llc "
aliases$address_text[8] <- "  main street   ste 2 "
aliases$permit_date[9] <- "2025-05-01"
alias_candidates <- classify(aliases) %>% mutate(in_archive = row_number() %in% c(2L, 9L))
deduped <- amenity_deduplicate_events(alias_candidates)
expect_equal(deduped$categorized$duplicate_opening_alias,
             c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE))
expect_equal(deduped$categorized$core_index_eligible,
             c(FALSE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE))
stopifnot(nrow(deduped$alias_audit) == 3L,
          all(deduped$alias_audit$opening_id == alias_candidates$event_id[2]),
          all(deduped$alias_audit$alias_group_size == 3L),
          !deduped$categorized$core_index_eligible_before_alias_dedup[9])

# A preferred archived alias with a future permit cannot suppress an eligible
# live-only row; deduplication runs only after the date/permit rules.
future <- fixture(2)
future$permit_date[1] <- "2025-05-01"
future_candidates <- classify(future) %>% mutate(in_archive = c(TRUE, FALSE))
future_dedup <- amenity_deduplicate_events(future_candidates)
expect_equal(future_dedup$categorized$core_index_eligible, c(FALSE, TRUE))
stopifnot(nrow(future_dedup$alias_audit) == 0L,
          !any(future_dedup$categorized$duplicate_opening_alias))

# Within a source-preference tier, known/earlier permits and stable-ID order
# determine the retained row, not input order or a missing permit date.
ordered <- fixture(4)
ordered$permit_date <- c(NA, "2024-01-01", "2023-01-01", "2023-01-01")
ordered_candidates <- classify(ordered) %>% mutate(in_archive = TRUE)
ordered_dedup <- amenity_deduplicate_events(ordered_candidates)
expect_equal(ordered_dedup$categorized$core_index_eligible, c(FALSE, FALSE, TRUE, FALSE))
shuffled <- amenity_deduplicate_events(ordered_candidates[4:1, ])
expect_equal(arrange(shuffled$categorized, event_id),
             arrange(ordered_dedup$categorized, event_id))
expect_equal(ordered_dedup$categorized$core_index_eligible_before_alias_dedup, rep(TRUE, 4))

# Empty eligible sets are valid and do not accidentally mark excluded rows as aliases.
all_outside <- fixture(2)
all_outside$first_sale_date <- "2026-04-02"
outside_candidates <- classify(all_outside) %>% mutate(in_archive = FALSE)
outside_dedup <- amenity_deduplicate_events(outside_candidates)
stopifnot(!any(outside_dedup$categorized$core_index_eligible),
          !any(outside_dedup$categorized$duplicate_opening_alias),
          nrow(outside_dedup$alias_audit) == 0L)

# Missing identifiers are not evidence that two otherwise similar rows describe
# one opening. Keep incomplete fingerprints distinct by their stable event IDs.
for (field in c("location_name", "street", "zip", "naics", "opening_date")) {
  incomplete <- classify(fixture(2)) %>% mutate(in_archive = TRUE)
  incomplete[[field]][] <- NA
  incomplete_dedup <- amenity_deduplicate_events(incomplete)
  stopifnot(all(incomplete_dedup$categorized$core_index_eligible),
            !any(incomplete_dedup$categorized$duplicate_opening_alias),
            nrow(incomplete_dedup$alias_audit) == 0L)
}
for (field in c("location_name", "street", "zip", "naics")) {
  incomplete <- classify(fixture(2)) %>% mutate(in_archive = TRUE)
  incomplete[[field]][] <- " "
  stopifnot(all(amenity_deduplicate_events(incomplete)$categorized$core_index_eligible))
}
missing_number <- classify(fixture(2)) %>% mutate(in_archive = TRUE, street = "NA MAIN STREET")
stopifnot(all(amenity_deduplicate_events(missing_number)$categorized$core_index_eligible))

cat("Amenity date, taxonomy, eligibility, and source-reconciliation tests passed.\n")
