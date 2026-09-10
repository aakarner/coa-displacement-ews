# Offline integration checks against persisted evidence, not annual outcome labels.
suppressPackageStartupMessages({library(dplyr); library(readr); library(testthat)})
source("R/part2_index_scoring.R")
root <- "output/part2/evictions"
paired <- readRDS(file.path(root, "eviction_features_paired.rds"))
ledger <- readRDS(file.path(root, "eviction_case_ledger.rds"))
scale <- readRDS(file.path(root, "eviction_scaling.rds"))
grid <- readRDS("output/hex_grid.rds")
units <- sf::st_drop_geometry(readRDS("output/corporate_ownership_by_hex.rds"))
dates <- as.Date(c("2025-04-01", "2026-04-01"))
manifest <- jsonlite::fromJSON(file.path(root, "eviction_run_manifest.json"))
stopifnot(identical(manifest$status, "paired_eviction_features_complete_v2"))
stopifnot(identical(manifest$eligibility_rule,"rolling_scored_24_months_v1"))
summary <- read_csv(file.path(root, "eviction_snapshot_summary.csv"), show_col_types = FALSE)
stopifnot(all(c("usable_rate_change_hexes", "usable_legacy_percent_change_hexes") %in% names(summary)),
  !"usable_change_hexes" %in% names(summary))
baseline <- paired[paired$analysis_as_of_date == dates[1], ]
stopifnot(identical(scale, part2_fit_index_scaling(baseline, scale$components, scale$index_name,
  dates[1], preserved_scaling = scale)))
bound <- as.numeric(quantile(abs(baseline$eviction_latest_12mo_rate_change_per_100_units), .99, na.rm=TRUE, type=7))
stopifnot(identical(scale$bounds$upper_bound[2], bound), identical(scale$bounds$lower_bound[2], -bound))

test_that("all grid rows and fixed denominators persist at both dates", {
  expect_equal(nrow(paired), 14054L); expect_true(is.integer(paired$hex_id))
  expect_equal(nrow(distinct(paired, hex_id, analysis_as_of_date)), nrow(paired))
  expect_equal(sort(unique(paired$analysis_as_of_date)), dates)
  expect_equal(paired$residential_units, units$residential_units[match(paired$hex_id, units$hex_id)])
  expect_equal(paired$area_km2, grid$area_km2[match(paired$hex_id, grid$hex_id)])
  expect_false(any(paired$eviction_all_filing_locations_complete))
  expect_true(all(paired$eviction_history_start == as.Date("2022-01-01")))
})

for (i in seq_along(dates)) local({
  date <- dates[i]; f <- filter(paired, analysis_as_of_date == date)
  directory <- file.path(root, as.character(date))
  test_that(paste("counts and missingness are independently reproduced", date), {
    recent_start <- as.Date(paste0(as.integer(format(date, "%Y")) - 1L, "-04-02"))
    previous_start <- as.Date(paste0(as.integer(format(date, "%Y")) - 2L, "-04-02"))
    eligible <- ledger %>% filter(assignment_status == "assigned_unique_hex", file_date >= as.Date("2022-01-01"), file_date <= date) %>%
      mutate(hex_id = as.integer(assigned_hex_key)) %>% group_by(hex_id) %>% summarise(
        total = n(), recent = sum(file_date >= recent_start),
        previous = sum(file_date >= previous_start & file_date < recent_start), .groups = "drop")
    expected <- select(f, hex_id) %>% left_join(eligible, by = "hex_id") %>% mutate(across(c(total, recent, previous), ~coalesce(.x, 0L)))
    expect_equal(f$eviction_history_observed_cases, expected$total)
    expect_equal(f$eviction_recent_observed_cases, expected$recent)
    expect_equal(f$eviction_previous_observed_cases, expected$previous)
    expect_true(all(is.na(f$eviction_cases_latest_12mo[!f$eviction_count_observed])))
    expect_true(all(is.na(f$eviction_pressure_index[!f$eviction_count_observed])))
    expect_true(all(is.na(f$eviction_cases_latest_12mo_change_pct[f$eviction_previous_observed_cases == 0L])))
    expect_true(all(is.na(f$eviction_latest_12mo_per_100_units[f$residential_units < 20])))
    rate_difference <- ifelse(f$eviction_count_observed & is.finite(f$residential_units) & f$residential_units >= 20,
      100*(expected$recent - expected$previous)/f$residential_units, NA_real_)
    expect_equal(f$eviction_latest_12mo_rate_change_per_100_units, rate_difference)
    scores <- lapply(scale$components, function(component) {
      b <- scale$bounds[match(component, scale$bounds$component), ]; value <- f[[component]]
      if(b$degenerate_range) ifelse(is.na(value), NA_real_, if(grepl("rate_change_per_100_units$", component)) 50 else 0) else
        100*(pmax(pmin(value,b$upper_bound),b$lower_bound)-b$lower_bound)/(b$upper_bound-b$lower_bound)
    })
    expect_equal(f$eviction_pressure_index, rowMeans(as.data.frame(scores), na.rm=FALSE))
    expect_equal(f$eviction_pressure_index_components_complete, is.finite(f$eviction_pressure_index))
    expect_true(all(f$eviction_pressure_index_components_required == 2L))
    expect_equal(f$eviction_recent_share[f$eviction_cases_total > 0 & !is.na(f$eviction_cases_total)],
      (f$eviction_cases_latest_12mo / f$eviction_cases_total)[f$eviction_cases_total > 0 & !is.na(f$eviction_cases_total)])
    segments <- read_csv(file.path(directory, "eviction_source_coverage_segments.csv"), show_col_types = FALSE)
    summary <- segments %>% group_by(hex_id) %>% summarise(covered = all(segment_complete), .groups = "drop")
    expect_equal(f$eviction_scored_window_source_covered, summary$covered[match(f$hex_id, summary$hex_id)])
    expect_equal(min(segments$requested_start),previous_start)
    expect_equal(max(segments$requested_end),date)
    expect_true(all(f$eviction_eligibility_window_start==previous_start))
    expect_true(all(f$eviction_eligibility_window_end==date))
    expect_equal(sum(f$eviction_inside_current_city), 6060L)
    expect_equal(sum(f$eviction_source_covered), 5977L)
    uncertainty <- read_csv(file.path(directory, "eviction_localizable_uncertainty.csv"), show_col_types = FALSE)
    source <- read_csv(file.path(root,"eviction_source_case_dates.csv"),show_col_types=FALSE)
    candidates <- read_csv(file.path(root,"eviction_candidate_hexes.csv"),show_col_types=FALSE)
    possible <- unique(source$case_number[is.na(source$file_date) | (source$file_date>=previous_start & source$file_date<=date)])
    ambiguous <- ledger$case_number[ledger$assignment_status %in% c("excluded_inconsistent_source_county",
      "excluded_inconsistent_source_jp","excluded_missing_valid_case_identifier","excluded_missing_filing_date",
      "excluded_inconsistent_filing_dates","excluded_multiple_hexes","excluded_mixed_inside_outside_grid",
      "excluded_mixed_inside_outside_study_geography") & ledger$case_number %in% possible]
    expected_uncertainty <- distinct(filter(candidates,case_number %in% ambiguous),case_number,hex_id)
    expect_setequal(paste(uncertainty$case_number,uncertainty$hex_id),paste(expected_uncertainty$case_number,expected_uncertainty$hex_id))
    expected_n <- table(factor(expected_uncertainty$hex_id,levels=f$hex_id))
    expect_equal(f$eviction_unresolved_candidate_cases,as.integer(expected_n))
    expect_equal(f$eviction_count_observed,f$eviction_source_covered & as.integer(expected_n)==0L)
    expect_true(all(!f$eviction_count_observed[f$hex_id %in% uncertainty$hex_id]))
    expect_true(all(is.na(f$eviction_pressure_index[f$source_county == "Hays"])))
    rescored <- part2_apply_index_scaling(f, scale)$features
    expect_equal(f$eviction_pressure_index, rescored$eviction_pressure_index)
    expect_true(all(f$eviction_pressure_index_scaling_reference_as_of_date == dates[1]))
  })
})

test_that("original inputs and all generated artifacts match their manifest hashes", {
  expect_true(manifest$canonical_outputs_unchanged)
  for (kind in c("inputs", "outputs")) {
    paths <- manifest[[kind]]$path
    actual <- vapply(paths, function(p) digest::digest(file = p, algo = "sha256"), character(1))
    expect_equal(unname(actual), manifest[[kind]]$sha256)
  }
  before <- read_csv(file.path(root, "canonical_preservation_before.csv"), show_col_types = FALSE)
  after <- read_csv(file.path(root, "canonical_preservation_after.csv"), show_col_types = FALSE)
  expect_equal(before$path, after$path); expect_equal(before$sha256, after$sha256)
})
cat("Part2 eviction persisted-output tests complete.\n")
