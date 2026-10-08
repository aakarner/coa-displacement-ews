# Compare the reviewed-location production rebuild with its preserved baseline.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/eviction_property_geography.R")
root <- "output/property_review_production"
before <- file.path(root, "before")
stopifnot(dir.exists(before))
read_before <- function(path) readRDS(file.path(before, path))
old <- st_drop_geometry(read_before("output/part1/measurement/current_measurement.rds"))
new <- st_drop_geometry(readRDS("output/part1/measurement/current_measurement.rds"))
summary <- function(x) data.frame(eligible_cells = sum(x$primary_cluster_eligible),
  total_units = sum(x$residential_units), total_recent_filings = sum(x$eviction_recent_observed_cases, na.rm = TRUE),
  eligible_recent_filings = sum(x$eviction_recent_observed_cases[x$primary_cluster_eligible], na.rm = TRUE),
  low_unit_cells = sum(x$residential_units < 20 & x$eviction_recent_observed_cases > 0, na.rm = TRUE),
  low_unit_filings = sum(x$eviction_recent_observed_cases[x$residential_units < 20], na.rm = TRUE))
measurements <- bind_rows(before = summary(old), after = summary(new), .id = "stage")
write_csv(measurements, file.path(root, "measurement_comparison.csv"))
cells <- inner_join(old %>% select(hex_id, before_filings = eviction_recent_observed_cases),
  new %>% select(hex_id, residential_units, primary_cluster_eligible, after_filings = eviction_recent_observed_cases), by = "hex_id") %>%
  filter(before_filings != after_filings)
write_csv(cells, file.path(root, "changed_cells.csv"))
cases <- readRDS("output/part2/evictions/eviction_case_ledger.rds")
old_cases <- read_before("output/part2/evictions/eviction_case_ledger.rds")
reviewed <- attr(read_eviction_property_geography(), "case_reviews") %>% distinct(case_number)
applied <- cases %>% semi_join(reviewed, by = "case_number") %>%
  select(case_number, file_date, property_id, property_review_id, property_review_basis,
    property_apartment_conflict, original_assigned_hex_key, assigned_hex_key, property_assignment_status) %>%
  left_join(old_cases %>% select(case_number, before_hex_key = assigned_hex_key), by = "case_number")
stopifnot(nrow(applied) == 71L, all(applied$assigned_hex_key != applied$before_hex_key))
write_csv(applied, file.path(root, "applied_cases.csv"))

audited <- read_csv("output/residential_geography_repair/audited_case_outcomes.csv", show_col_types = FALSE)
residual <- cases %>% semi_join(audited, by = "case_number") %>%
  mutate(hex_id = as.integer(assigned_hex_key)) %>%
  left_join(new %>% select(hex_id, residential_units), by = "hex_id") %>% filter(residential_units < 20)
write_csv(residual %>% count(hex_id, name = "filings"), file.path(root, "remaining_original_audit_cells.csv"))

a <- read_csv(file.path(before, "output/part1/baseline_cluster_assignments.csv"), show_col_types = FALSE)
b <- read_csv("output/part1/baseline_cluster_assignments.csv", show_col_types = FALSE)
j <- inner_join(a %>% select(hex_id, before = tentative_name), b %>% select(hex_id, after = tentative_name), by = "hex_id")
write_csv(count(j, before, after, name = "cells"), file.path(root, "part1_profile_transitions.csv"))
part1 <- data.frame(common_cells = nrow(j), same_profile = sum(j$before == j$after),
  changed_profile = sum(j$before != j$after), new_cells = sum(!b$hex_id %in% a$hex_id), lost_cells = sum(!a$hex_id %in% b$hex_id))
write_csv(part1, file.path(root, "part1_profile_comparison.csv"))
profiles <- full_join(count(a, tentative_name, name = "before"), count(b, tentative_name, name = "after"), by = "tentative_name")
write_csv(profiles, file.path(root, "part1_profile_sizes.csv"))
neighborhoods <- inner_join(
  read_csv(file.path(before, "output/part1/neighborhood_cluster_summary.csv"), show_col_types = FALSE) %>%
    select(neighborhood_name, before = population_plurality_name),
  read_csv("output/part1/neighborhood_cluster_summary.csv", show_col_types = FALSE) %>%
    select(neighborhood_name, after = population_plurality_name), by = "neighborhood_name") %>%
  filter(before != after)
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
p0 <- p0[match(p1$hex_id, p0$hex_id), ]
stopifnot(identical(p0$hex_id, p1$hex_id), identical(p0$common_comparison_ready, p1$common_comparison_ready))
ok <- p1$common_comparison_ready
part2 <- data.frame(common_cells = sum(ok), changed_2025 = sum(p0$cluster_2025[ok] != p1$cluster_2025[ok]),
  changed_2026_fixed = sum(p0$cluster_2026_fixed[ok] != p1$cluster_2026_fixed[ok]),
  fixed_transitions_before = sum(p0$moved_fixed[ok]), fixed_transitions_after = sum(p1$moved_fixed[ok]))
write_csv(part2, file.path(root, "part2_comparison.csv"))
jsonlite::write_json(list(reviewed_cases = nrow(applied), changed_cells = nrow(cells),
  apartment_conflicts_retained = sum(applied$property_apartment_conflict),
  remaining_original_audit_filings = nrow(residual),
  changed_neighborhood_population_pluralities = nrow(neighborhoods),
  changed_fixed_previous_classifier = sum(fixed$old_cluster != fixed$fixed_new_cluster),
  measurement = measurements, part1 = part1, part2 = part2), file.path(root, "summary.json"), pretty = TRUE, auto_unbox = TRUE)
print(measurements); print(part1); print(part2)
