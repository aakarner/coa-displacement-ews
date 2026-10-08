# Compare the completed reviewed-unit rebuild with the immutable pre-review run.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
root <- Sys.getenv("EWS_UNIT_REVIEW_AUDIT_ROOT", unset = "output/reviewed_units_20261007")
before <- file.path(root, "before")
read_before <- function(path) readRDS(file.path(before, path))
old <- st_drop_geometry(read_before("output/part1/measurement/current_measurement.rds"))
new <- st_drop_geometry(readRDS("output/part1/measurement/current_measurement.rds"))
summary_measurement <- function(x) data.frame(eligible_cells = sum(x$primary_cluster_eligible),
  total_units = sum(x$residential_units), total_recent_filings = sum(x$eviction_recent_observed_cases, na.rm = TRUE),
  eligible_recent_filings = sum(x$eviction_recent_observed_cases[x$primary_cluster_eligible], na.rm = TRUE),
  low_unit_cells = sum(x$residential_units < 20 & x$eviction_recent_observed_cases > 0, na.rm = TRUE),
  low_unit_filings = sum(x$eviction_recent_observed_cases[x$residential_units < 20], na.rm = TRUE))
measurements <- bind_rows(before = summary_measurement(old), after = summary_measurement(new), .id = "stage")
write_csv(measurements, file.path(root, "measurement_comparison.csv"))
fields <- c("hex_id", "residential_units", "eviction_recent_observed_cases", "primary_cluster_eligible")
cells <- inner_join(select(old, all_of(fields)), select(new, all_of(fields)), by = "hex_id", suffix = c("_before", "_after")) %>%
  filter(abs(residential_units_before - residential_units_after) > 1e-7 |
    eviction_recent_observed_cases_before != eviction_recent_observed_cases_after |
    primary_cluster_eligible_before != primary_cluster_eligible_after) %>%
  mutate(recent_rate_before = if_else(residential_units_before >= 20,
    100 * eviction_recent_observed_cases_before / residential_units_before, NA_real_),
    recent_rate_after = if_else(residential_units_after >= 20,
      100 * eviction_recent_observed_cases_after / residential_units_after, NA_real_))
write_csv(cells, file.path(root, "changed_cells.csv"))
cases <- readRDS("output/part2/evictions/eviction_case_ledger.rds")
old_cases <- read_before("output/part2/evictions/eviction_case_ledger.rds")
changed_cases <- cases %>% left_join(old_cases %>% select(case_number, before_hex_key = assigned_hex_key,
  before_property_id = property_id), by = "case_number") %>% filter(assigned_hex_key != before_hex_key)
write_csv(changed_cases, file.path(root, "final_case_assignment_changes.csv"))
annual_path <- "output/part3/eviction_property_assignment_ledger.csv"
annual_types <- cols(.default = col_guess(), case_number = "c", assigned_hex_key = "c",
  original_assigned_hex_key = "c", property_review_id = "c", property_review_basis = "c")
annual <- read_csv(annual_path, col_types = annual_types, show_col_types = FALSE)
annual_old <- read_csv(file.path(before, annual_path), col_types = annual_types, show_col_types = FALSE)
annual_changed <- annual %>% left_join(annual_old %>% select(case_number, before_hex_key = assigned_hex_key),
  by = "case_number") %>% filter(assigned_hex_key != before_hex_key)
write_csv(annual_changed, file.path(root, "annual_changed_case_assignments.csv"))
facility <- cases %>% filter(!is.na(property_facility_type), nzchar(property_facility_type))
write_csv(facility, file.path(root, "final_facility_flagged_cases.csv"))
audited <- read_csv("output/residential_geography_repair/audited_case_outcomes.csv", show_col_types = FALSE)
residual <- cases %>% semi_join(audited, by = "case_number") %>% mutate(hex_id = as.integer(assigned_hex_key)) %>%
  left_join(new %>% select(hex_id, residential_units), by = "hex_id") %>% filter(residential_units < 20)
write_csv(residual %>% count(hex_id, name = "filings"), file.path(root, "remaining_original_audit_cells.csv"))

a <- read_csv(file.path(before, "output/part1/baseline_cluster_assignments.csv"), show_col_types = FALSE)
b <- read_csv("output/part1/baseline_cluster_assignments.csv", show_col_types = FALSE)
j <- inner_join(a %>% select(hex_id, before = tentative_name), b %>% select(hex_id, after = tentative_name), by = "hex_id")
write_csv(count(j, before, after, name = "cells"), file.path(root, "part1_profile_transitions.csv"))
part1 <- data.frame(common_cells = nrow(j), same_profile = sum(j$before == j$after),
  changed_profile = sum(j$before != j$after), new_cells = sum(!b$hex_id %in% a$hex_id), lost_cells = sum(!a$hex_id %in% b$hex_id))
write_csv(part1, file.path(root, "part1_profile_comparison.csv"))
write_csv(full_join(count(a, tentative_name, name = "before"), count(b, tentative_name, name = "after"),
  by = "tentative_name"), file.path(root, "part1_profile_sizes.csv"))
neighborhoods <- inner_join(
  read_csv(file.path(before, "output/part1/neighborhood_cluster_summary.csv"), show_col_types = FALSE) %>%
    select(neighborhood_name, before = population_plurality_name),
  read_csv("output/part1/neighborhood_cluster_summary.csv", show_col_types = FALSE) %>%
    select(neighborhood_name, after = population_plurality_name), by = "neighborhood_name") %>% filter(before != after)
write_csv(neighborhoods, file.path(root, "neighborhood_profile_changes.csv"))
m <- read_before("output/part1/baseline_cluster_model.rds")
x <- st_drop_geometry(readRDS("output/hex_features.rds")); x <- x[x$primary_cluster_eligible, ]
z <- scale(as.matrix(x[m$features]), center = m$preprocessing$center, scale = m$preprocessing$scale)
d <- sapply(seq_len(m$k), function(k) rowSums(sweep(z, 2, m$centroids[k, ], "-")^2))
fixed <- data.frame(hex_id = x$hex_id, old_cluster = m$training_assignment[match(x$hex_id, m$training_hex_ids)],
  fixed_new_cluster = max.col(-d, ties.method = "first"))
write_csv(fixed, file.path(root, "fixed_classifier_comparison.csv"))

p0 <- read_before("output/part2/clusters/part2_cluster_assignments.rds")
p1 <- readRDS("output/part2/clusters/part2_cluster_assignments.rds")
l0 <- jsonlite::fromJSON(file.path(before, "config/part2_cluster_interpretation.json"))$clusters
l1 <- jsonlite::fromJSON("config/part2_cluster_interpretation.json")$clusters
for (field in c("cluster_2025", "cluster_2026_fixed")) {
  p0[[paste0(field, "_profile")]] <- l0$profile[match(p0[[field]], l0$cluster)]
  p1[[paste0(field, "_profile")]] <- l1$profile[match(p1[[field]], l1$cluster)]
}
q <- inner_join(filter(p0, common_comparison_ready), filter(p1, common_comparison_ready), by = "hex_id", suffix = c("_before", "_after"))
part2 <- data.frame(common_before = sum(p0$common_comparison_ready), common_after = sum(p1$common_comparison_ready),
  common_both = nrow(q), changed_2025_profile = sum(q$cluster_2025_profile_before != q$cluster_2025_profile_after),
  changed_2026_fixed_profile = sum(q$cluster_2026_fixed_profile_before != q$cluster_2026_fixed_profile_after),
  fixed_transitions_before = sum(p0$moved_fixed[p0$common_comparison_ready]),
  fixed_transitions_after = sum(p1$moved_fixed[p1$common_comparison_ready]))
write_csv(part2, file.path(root, "part2_comparison.csv"))
review_config <- jsonlite::read_json("config/residential_unit_property_reviews.json")
added_accounts <- nrow(readRDS("output/residential_parcels_unit_promoted.rds")) -
  nrow(read_before("output/residential_parcels_unit_promoted.rds"))
jsonlite::write_json(list(reviewed_projects = length(review_config$projects), added_accounts = added_accounts, changed_case_assignments = nrow(changed_cases),
  annual_changed_case_assignments = nrow(annual_changed),
  remaining_original_audit_filings = nrow(residual), facility_flagged_cases = nrow(facility),
  changed_neighborhood_population_pluralities = nrow(neighborhoods),
  changed_fixed_previous_classifier = sum(fixed$old_cluster != fixed$fixed_new_cluster, na.rm = TRUE),
  measurement = measurements, part1 = part1, part2 = part2), file.path(root, "summary.json"), pretty = TRUE, auto_unbox = TRUE)
print(measurements); print(cells); print(part1); print(part2)
