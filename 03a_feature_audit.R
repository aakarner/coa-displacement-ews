################################################################################
# 03a - Feature Coverage and Role Audit
################################################################################
#
# Verifies that the repaired Part 1 clustering inputs exist, reports their
# coverage, and keeps planned and profile-only fields out of the clustering
# matrix. Run after 03_feature_engineering.R and before any clustering script.

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))
source(project_path("R", "analysis_config.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
})

print_header("03a - FEATURE COVERAGE AND ROLE AUDIT")

OUTPUT_DIR <- project_path("output")
feature_spec_file <- project_path("config", "feature_dictionary.csv")
feature_file <- file.path(OUTPUT_DIR, "hex_features.rds")

feature_spec <- read_csv(feature_spec_file, show_col_types = FALSE)
hex_features <- load_output(feature_file, "engineered features") %>%
  st_drop_geometry()

eligibility_col <- if ("primary_cluster_eligible" %in% names(hex_features)) {
  "primary_cluster_eligible"
} else if ("sufficient_data" %in% names(hex_features)) {
  "sufficient_data"
} else {
  NA_character_
}

analysis_features <- if (is.na(eligibility_col)) {
  hex_features
} else {
  hex_features %>% filter(.data[[eligibility_col]])
}

coverage_rows <- lapply(feature_spec$feature, function(feature) {
  present <- feature %in% names(analysis_features)
  values <- if (present) analysis_features[[feature]] else rep(NA_real_, nrow(analysis_features))

  tibble(
    feature = feature,
    present = present,
    eligible_hexes = nrow(analysis_features),
    nonmissing_hexes = sum(!is.na(values)),
    coverage_pct = if (nrow(analysis_features) > 0) {
      100 * mean(!is.na(values))
    } else {
      NA_real_
    },
    nonzero_hexes = if (is.numeric(values)) sum(!is.na(values) & values != 0) else NA_integer_
  )
})

feature_audit <- feature_spec %>%
  left_join(bind_rows(coverage_rows), by = "feature") %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

write_csv(feature_audit, file.path(OUTPUT_DIR, "feature_coverage_audit.csv"))

required_inputs <- feature_audit %>%
  filter(role == "cluster_input")

missing_required <- required_inputs %>%
  filter(!present | nonmissing_hexes == 0)

print(required_inputs %>% select(feature, domain, coverage_pct, status))

if (nrow(missing_required) > 0) {
  stop(
    "Required Part 1 feature(s) are unavailable: ",
    paste(missing_required$feature, collapse = ", "),
    ". Run the corresponding processing scripts before clustering.",
    call. = FALSE
  )
}

duplicate_domains <- required_inputs %>%
  count(domain, name = "cluster_input_count") %>%
  filter(cluster_input_count > 1)

if (nrow(duplicate_domains) > 0) {
  stop(
    "More than one default cluster input is assigned to domain(s): ",
    paste(duplicate_domains$domain, collapse = ", "),
    call. = FALSE
  )
}

print_progress("Feature role and coverage audit passed.")
print_progress("Saved output/feature_coverage_audit.csv")
