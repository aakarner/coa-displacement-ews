# Read-only reconciliation of existing production and staged review artifacts.
# Writes this audit's summaries only; never rebuilds measurements or clusters.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
root <- "output/residential_repair_closeout_20261007"
dir.create(root, recursive = TRUE, showWarnings = FALSE)
paths <- c(
  measurement = "output/part1/measurement/current_measurement.rds",
  ledger = "output/part2/evictions/eviction_case_ledger.rds",
  original_cohort = "output/residential_geography_repair/audited_case_outcomes.csv",
  initial_stages = "output/residential_geography_repair/stage_summary.csv",
  last_rebuild = "output/residential_property_batch2/summary.json",
  oak_cases = "data/reviewed_manufactured_housing/oak_ranch_20261007/exception_review_01/case_home_crosswalk.csv",
  oak_manifest = "data/reviewed_manufactured_housing/oak_ranch_20261007/exception_review_01/manifest.json",
  ownership_status = "tmp/ownership_address_audit_20261007/cell2326_repair_status.json")
m <- st_drop_geometry(readRDS(paths[["measurement"]]))
l <- readRDS(paths[["ledger"]])
cohort <- read_csv(paths[["original_cohort"]], show_col_types = FALSE)
oak <- read_csv(paths[["oak_cases"]], show_col_types = FALSE)
recent <- l %>% filter(file_date >= as.Date("2025-04-02"), file_date <= as.Date("2026-04-01"))
assigned <- recent %>% filter(assignment_status == "assigned_unique_hex") %>%
  mutate(hex_id = as.integer(assigned_hex_key)) %>%
  left_join(m %>% select(hex_id, residential_units, primary_cluster_eligible), by = "hex_id")
original <- assigned %>% semi_join(cohort, by = "case_number")
low <- assigned %>% filter(residential_units < 20)
stopifnot(!anyDuplicated(l$case_number), nrow(assigned) == 11784L,
  nrow(original) == 1219L, nrow(low) == 215L,
  sum(original$residential_units >= 20) == 1195L,
  setequal(original$case_number[original$residential_units < 20], oak$case_number),
  all(oak$status == "ready_for_batched_integration"),
  all(oak$original_hex == oak$reviewed_hex))
assignment_status <- recent %>% count(assignment_status, name = "filings")
property_status <- assigned %>% count(property_assignment_status, name = "filings")
low_summary <- low %>% mutate(original_audit = case_number %in% cohort$case_number,
  denominator_band = if_else(residential_units == 0, "zero_units", "positive_below_20")) %>%
  group_by(original_audit, denominator_band) %>%
  summarise(filings = n(), cells = n_distinct(hex_id), .groups = "drop")
summary <- list(
  reported_on = "2026-10-07", recent_window = c("2025-04-02", "2026-04-01"),
  production_rebuilt_by_this_audit = FALSE,
  current_production = list(covered_recent_filings = nrow(assigned),
    grid_units = sum(m$residential_units), part1_eligible_cells = sum(m$primary_cluster_eligible),
    filings_in_part1_eligible_cells = sum(assigned$primary_cluster_eligible),
    low_unit_filings = nrow(low), low_unit_cells = n_distinct(low$hex_id),
    filings_at_least_20_units_but_not_clustered = sum(assigned$residential_units >= 20 & !assigned$primary_cluster_eligible),
    original_audit_filings = nrow(original), original_audit_at_least_20_units = sum(original$residential_units >= 20),
    original_audit_low_unit_filings = sum(original$residential_units < 20),
    other_low_unit_filings = sum(!low$case_number %in% cohort$case_number),
    original_audit_provisional_ben_white = sum(original$property_denominator_status == "provisional_assumed_170_units", na.rm = TRUE)),
  staged_oak_links = list(filings = nrow(oak), distinct_homes = n_distinct(oak$home_parcel_id),
    changed_hexes = sum(oak$original_hex != oak$reviewed_hex), production_applied = FALSE),
  assignment_status_all_source_recent_cases = assignment_status,
  property_status_assigned_recent_cases = property_status,
  low_unit_breakdown = low_summary,
  caveat = "Source-ledger exclusion totals include records outside the Austin study area. They are not all missing Austin filings. Denominator support does not establish complete-case cluster eligibility.",
  source_hashes = data.frame(path = unname(paths), sha256 = vapply(unname(paths),
    function(p) digest::digest(file = p, algo = "sha256"), character(1))))
stopifnot(summary$current_production$original_audit_provisional_ben_white == 60L,
  summary$current_production$filings_in_part1_eligible_cells == 10628L,
  summary$current_production$filings_at_least_20_units_but_not_clustered == 941L)
write_csv(assignment_status, file.path(root, "assignment_status.csv"))
write_csv(property_status, file.path(root, "property_status.csv"))
write_csv(low_summary, file.path(root, "low_unit_breakdown.csv"))
jsonlite::write_json(summary, file.path(root, "summary.json"), pretty = TRUE, auto_unbox = TRUE, digits = 10)
print(summary$current_production)
