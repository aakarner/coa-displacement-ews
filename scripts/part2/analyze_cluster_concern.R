# Qualitative interpretation of a reviewed baseline, not a learned risk model.
suppressPackageStartupMessages({library(dplyr); library(readr)})
source("R/pipeline.R")
config_path <- "config/part2_cluster_interpretation.json"
model_path <- "output/part2/clusters/part2_cluster_models.rds"
assignment_path <- "output/part2/clusters/part2_cluster_assignments.rds"
root <- "output/part2/interpretation"
config <- jsonlite::fromJSON(config_path)
model <- readRDS(model_path)
assignments <- readRDS(assignment_path)
mapping <- config$clusters
stopifnot(identical(config$baseline_centers_sha256,
  digest::digest(model$baseline$centers, algo = "sha256")),
  identical(model$measurement_version, "part2-fixed-components-v2"),
  !anyDuplicated(mapping$cluster), setequal(mapping$cluster, rownames(model$baseline$centers)),
  all(mapping$concern_rank %in% 1:4),
  all(mapping$concern == c("Low", "Moderate", "High", "Very high")[mapping$concern_rank]))
input_paths <- c(config_path, model_path, assignment_path,
  "output/part2/evictions/eviction_features_paired.rds", "R/pipeline.R",
  "scripts/part2/analyze_cluster_concern.R")
before <- build_file_manifest(input_paths, require_all = TRUE, hash_files = TRUE)
for (snapshot in c("2025", "2026_fixed", "2026_refit_aligned")) {
  position <- match(assignments[[paste0("cluster_", snapshot)]], mapping$cluster)
  assignments[[paste0("concern_", snapshot)]] <- mapping$concern[position]
  assignments[[paste0("concern_rank_", snapshot)]] <- mapping$concern_rank[position]
}
assignments$concern_rank_change <- assignments$concern_rank_2026_fixed - assignments$concern_rank_2025
assignments$concern_direction <- ifelse(is.na(assignments$concern_rank_change), NA_character_,
  ifelse(assignments$concern_rank_change > 0, "Higher", ifelse(assignments$concern_rank_change < 0, "Lower", "Same tier")))
assignments$transition_size <- ifelse(is.na(assignments$concern_rank_change), NA_character_,
  ifelse(!assignments$moved_fixed, "Same cluster", ifelse(assignments$concern_rank_change == 0,
    "Different cluster, same tier", ifelse(abs(assignments$concern_rank_change) == 1,
      "One tier", "Two or more tiers"))))
x <- filter(assignments, common_comparison_ready)
stopifnot(nrow(x) == length(model$training_hex_ids), !anyNA(x$concern_rank_change),
  all(is.na(assignments$concern_rank_change[!assignments$common_comparison_ready])))
transitions <- x %>% count(concern_2025, concern_2026_fixed, concern_rank_2025,
  concern_rank_2026_fixed, name = "hexes") %>% mutate(share_of_all = hexes / nrow(x))
directions <- x %>% group_by(concern_direction) %>% summarise(hexes = n(),
  fixed_residential_units = sum(residential_units), .groups = "drop") %>%
  mutate(share_of_all = hexes / nrow(x), fixed_units_share = fixed_residential_units / sum(x$residential_units))
sizes <- x %>% count(transition_size, name = "hexes") %>% mutate(share_of_all = hexes / nrow(x))
steps <- x %>% count(concern_rank_change, name = "hexes") %>% mutate(share_of_all = hexes / nrow(x))

# Descriptive small-count and unit-support diagnostics; these are not refitted
# alternative models and do not establish robustness to event-location errors.
events <- readRDS("output/part2/evictions/eviction_features_paired.rds")
events <- events %>% group_by(hex_id) %>% summarise(max_recent_filings = max(eviction_cases_latest_12mo),
  any_first_filing = any(eviction_cases_previous_12mo == 0 & eviction_cases_latest_12mo > 0), .groups = "drop")
x <- x %>% left_join(events, by = "hex_id") %>% mutate(
  filing_band = case_when(max_recent_filings == 0 ~ "0", max_recent_filings == 1 ~ "1",
    max_recent_filings < 5 ~ "2-4", TRUE ~ "5+"),
  unit_band = case_when(residential_units < 50 ~ "20-49", residential_units < 100 ~ "50-99", TRUE ~ "100+"))
small_counts <- x %>% group_by(filing_band, unit_band) %>% summarise(hexes = n(),
  moved = sum(moved_fixed), moved_share = mean(moved_fixed),
  higher = sum(concern_rank_change > 0), lower = sum(concern_rank_change < 0),
  first_filing_cells = sum(any_first_filing), .groups = "drop")
unit_sensitivity <- bind_rows(lapply(c(20, 50, 100), function(threshold) {
  y <- filter(x, residential_units >= threshold)
  data.frame(minimum_fixed_units = threshold, hexes = nrow(y),
    moved = sum(y$moved_fixed), moved_share = mean(y$moved_fixed),
    higher_share = mean(y$concern_rank_change > 0), lower_share = mean(y$concern_rank_change < 0),
    interpretation = "Subset of the same fitted model, not a refit with a different unit cutoff")
}))
dir.create(root, recursive = TRUE, showWarnings = FALSE)
saveRDS(assignments, file.path(root, "part2_concern_assignments.rds"))
write_csv(assignments, file.path(root, "part2_concern_assignments.csv"))
tables <- list(cluster_interpretation = mapping, concern_transitions = transitions,
  concern_directions = directions, concern_transition_sizes = sizes, concern_step_distribution = steps,
  small_count_movement = small_counts, unit_support_sensitivity = unit_sensitivity)
for (name in names(tables)) write_csv(tables[[name]], file.path(root, paste0("part2_", name, ".csv")))
after <- build_file_manifest(input_paths, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(before$sha256, after$sha256))
manifest <- list(schema_version = 1L, status = "qualitative_concern_interpretation_complete",
  common_hexes = nrow(x), baseline_centers_sha256 = config$baseline_centers_sha256,
  automatic_cluster_ids_are_ordinal = FALSE, qualitative_categories_are_ordered = TRUE,
  concern_is_probability = FALSE, labels_reviewed_against_corrected_profiles = TRUE,
  mapping_rationale = config$rationale,
  limitations = c("Qualitative cluster-level concern, not calibrated risk or individual displacement",
    "Tier differences are ordinal steps, not equally spaced amounts of risk",
    "Later refit labels use the baseline interpretation through coeval matching and need substantive review if profiles diverge",
    "Unit-threshold summaries condition on the same fit rather than refitting; small event counts remain sensitive"),
  inputs = after, outputs = build_file_manifest(list.files(root, full.names = TRUE,
    pattern = "[.](rds|csv)$"), require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null")
print(mapping); print(directions); print(sizes); print(unit_sensitivity)
