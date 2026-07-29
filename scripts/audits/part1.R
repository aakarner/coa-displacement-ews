################################################################################
# Part 1 Baseline Validation and Lock Audit
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))
source(project_path("R", "analysis_config.R"))
source(project_path("R", "cluster_assignment.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
})

print_header("PART 1 - BASELINE VALIDATION AND LOCK AUDIT")

OUTPUT_DIR <- project_path("output")
PART1_DIR <- file.path(OUTPUT_DIR, "part1")
dir.create(PART1_DIR, recursive = TRUE, showWarnings = FALSE)

files <- c(
  features = file.path(OUTPUT_DIR, "hex_features.rds"),
  results = file.path(OUTPUT_DIR, "amenity_cluster_sensitivity.rds"),
  assignments = file.path(OUTPUT_DIR, "amenity_cluster_assignments.csv"),
  metrics = file.path(OUTPUT_DIR, "amenity_cluster_metrics.csv"),
  gap = file.path(OUTPUT_DIR, "amenity_cluster_gap_statistics.csv"),
  stability = file.path(OUTPUT_DIR, "amenity_cluster_stability.csv"),
  recommendations = file.path(OUTPUT_DIR, "amenity_cluster_recommendations.csv"),
  coverage = file.path(OUTPUT_DIR, "amenity_cluster_population_coverage.csv"),
  model = file.path(PART1_DIR, "baseline_cluster_model.rds"),
  labels = project_path("config", "amenity_cluster_labels_k6.csv"),
  dictionary = project_path("config", "feature_dictionary.csv"),
  requests_311_selection =
    file.path(OUTPUT_DIR, "311_service_request_selection.csv")
)
missing_files <- files[!file.exists(files)]
if (length(missing_files) > 0L) {
  stop(
    "Part 1 validation inputs are missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

features_sf <- readRDS(files[["features"]])
features <- st_drop_geometry(features_sf)
results <- readRDS(files[["results"]])
model <- readRDS(files[["model"]])
assignments <- read_csv(files[["assignments"]], show_col_types = FALSE)
metrics <- read_csv(files[["metrics"]], show_col_types = FALSE)
gap <- read_csv(files[["gap"]], show_col_types = FALSE)
stability <- read_csv(files[["stability"]], show_col_types = FALSE)
recommendations <- read_csv(
  files[["recommendations"]],
  show_col_types = FALSE
)
coverage <- read_csv(files[["coverage"]], show_col_types = FALSE)
labels <- read_csv(files[["labels"]], show_col_types = FALSE)
dictionary <- read_csv(files[["dictionary"]], show_col_types = FALSE)
requests_311_selection <- read_csv(
  files[["requests_311_selection"]],
  show_col_types = FALSE
)

checks <- tibble(
  check = character(),
  passed = logical(),
  detail = character()
)
record_check <- function(check, passed, detail) {
  checks <<- bind_rows(
    checks,
    tibble(
      check = check,
      passed = isTRUE(passed),
      detail = as.character(detail)
    )
  )
}

specification <- model$specification
k <- model$k
selected_assignments <- assignments %>%
  filter(specification == !!specification, k == !!k) %>%
  select(hex_id, cluster)
configured_features <- dictionary %>%
  filter(role == "cluster_input") %>%
  pull(feature)

record_check(
  "unique_feature_hex_ids",
  !anyDuplicated(features$hex_id),
  paste(nrow(features), "feature rows")
)
record_check(
  "unique_h3_indices",
  "h3_index" %in% names(features) && !anyDuplicated(features$h3_index),
  paste(n_distinct(features$h3_index), "H3 indices")
)

feature_cutoffs <- unique(as.Date(features$analysis_as_of_date))
record_check(
  "analysis_cutoff",
  length(feature_cutoffs) == 1L &&
    !is.na(feature_cutoffs) &&
    feature_cutoffs == EWS_CONFIG$analysis_as_of_date &&
    as.Date(model$analysis_as_of_date) == EWS_CONFIG$analysis_as_of_date,
  paste("feature/model cutoff", paste(feature_cutoffs, collapse = ", "))
)
record_check(
  "h3_resolution",
  identical(as.integer(model$h3_resolution), EWS_CONFIG$h3_resolution),
  paste("resolution", model$h3_resolution)
)
record_check(
  "configured_feature_schema",
  !anyDuplicated(configured_features) &&
    setequal(configured_features, model$features),
  paste(model$features, collapse = " | ")
)
record_check(
  "results_feature_schema",
  identical(
    model$features,
    if (identical(specification, "baseline")) {
      results$baseline_vars
    } else {
      c(results$baseline_vars, results$amenity_var)
    }
  ),
  paste("specification", specification)
)

feature_matrix <- as.matrix(features[, model$features, drop = FALSE])
storage.mode(feature_matrix) <- "double"
eligible <- !is.na(features[[model$eligibility_feature]]) &
  as.logical(features[[model$eligibility_feature]])
complete <- eligible &
  apply(feature_matrix, 1, function(row) all(is.finite(row)))
expected_hex_ids <- features$hex_id[complete]

record_check(
  "unique_selected_assignments",
  !anyDuplicated(selected_assignments$hex_id),
  paste(nrow(selected_assignments), "selected assignments")
)
record_check(
  "selected_sample_contract",
  setequal(selected_assignments$hex_id, expected_hex_ids),
  paste(length(expected_hex_ids), "eligible complete hexes")
)
record_check(
  "model_training_sample",
  setequal(model$training_hex_ids, expected_hex_ids),
  paste(length(model$training_hex_ids), "frozen training hexes")
)

training_matrix <- feature_matrix[
  match(model$training_hex_ids, features$hex_id),
  ,
  drop = FALSE
]
recomputed_center <- colMeans(training_matrix)
recomputed_scale <- apply(training_matrix, 2, stats::sd)
record_check(
  "frozen_scaling_parameters",
  isTRUE(all.equal(
    unname(model$preprocessing$center[model$features]),
    unname(recomputed_center),
    tolerance = 1e-12
  )) &&
    isTRUE(all.equal(
      unname(model$preprocessing$scale[model$features]),
      unname(recomputed_scale),
      tolerance = 1e-12
    )),
  "frozen means and standard deviations reproduce the training sample"
)

fit <- results$full_evaluations[[specification]]$models[[as.character(k)]]
record_check(
  "frozen_centroids",
  isTRUE(all.equal(model$centroids, fit$centers, tolerance = 1e-12)),
  paste(nrow(model$centroids), "centroids")
)

reassigned <- assign_fixed_clusters(features_sf, model) %>%
  filter(assignment_status == "assigned") %>%
  select(hex_id, reassigned_cluster = cluster)
assignment_comparison <- selected_assignments %>%
  left_join(reassigned, by = "hex_id", relationship = "one-to-one")
mismatches <- sum(
  assignment_comparison$cluster != assignment_comparison$reassigned_cluster,
  na.rm = TRUE
)
record_check(
  "exact_frozen_reassignment",
  nrow(reassigned) == nrow(selected_assignments) &&
    !anyNA(assignment_comparison$reassigned_cluster) &&
    mismatches == 0L,
  paste(mismatches, "assignment mismatches")
)

selected_labels <- labels %>% filter(solution_k == !!k)
record_check(
  "cluster_label_contract",
  nrow(selected_labels) == k &&
    !anyDuplicated(selected_labels$cluster) &&
    setequal(selected_labels$cluster, seq_len(k)),
  paste(nrow(selected_labels), "configured labels")
)
record_check(
  "cluster_membership",
  setequal(unique(selected_assignments$cluster), seq_len(k)),
  paste(sort(unique(selected_assignments$cluster)), collapse = ", ")
)

assigned_features <- features %>%
  filter(hex_id %in% selected_assignments$hex_id)
selected_values <- as.matrix(
  assigned_features[, model$features, drop = FALSE]
)
record_check(
  "selected_feature_values",
  all(is.finite(selected_values)) &&
    all(selected_values >= 0) &&
    all(selected_values <= 100),
  "all selected domain indices are finite and within 0-100"
)

profile_means <- features %>%
  select(hex_id, all_of(model$features)) %>%
  inner_join(selected_assignments, by = "hex_id", relationship = "one-to-one") %>%
  group_by(cluster) %>%
  summarise(across(all_of(model$features), mean), .groups = "drop")
anchor_labels <- selected_labels %>%
  filter(!is.na(profile_anchor), nzchar(profile_anchor))
anchor_matches <- vapply(
  seq_len(nrow(anchor_labels)),
  function(index) {
    feature <- anchor_labels$profile_anchor[[index]]
    expected_cluster <- anchor_labels$cluster[[index]]
    feature %in% names(profile_means) &&
      profile_means$cluster[[which.max(profile_means[[feature]])]] ==
        expected_cluster
  },
  logical(1)
)
record_check(
  "tentative_label_anchors",
  all(anchor_matches),
  paste(sum(anchor_matches), "of", length(anchor_matches), "anchors matched")
)

selected_recommendation <- recommendations %>%
  filter(
    specification == !!specification,
    diagnostic == "substantive_selected"
  )
selected_metrics <- metrics %>%
  filter(specification == !!specification, k == !!k)
selected_gap <- gap %>%
  filter(specification == !!specification, k == !!k)
selected_stability <- stability %>%
  filter(specification == !!specification, k == !!k)
record_check(
  "selected_diagnostics",
  nrow(selected_recommendation) == 1L &&
    selected_recommendation$recommended_k[[1]] == k &&
    nrow(selected_metrics) == 1L &&
    nrow(selected_gap) == 1L &&
    nrow(selected_stability) == 1L,
  paste("selected", specification, "k =", k)
)

classified_coverage <- coverage %>%
  filter(coverage_status == "classified")
record_check(
  "population_coverage",
  nrow(classified_coverage) == 1L &&
    sum(coverage$hexes) == nrow(features) &&
    abs(sum(coverage$total_population_share) - 1) < 1e-10,
  paste(
    if (nrow(classified_coverage) == 1L) {
      sprintf(
        "%.2f%% of allocated population classified",
        100 * classified_coverage$total_population_share[[1]]
      )
    } else {
      "classified row missing"
    }
  )
)

record_check(
  "311_selection_contract",
  nrow(requests_311_selection) > 0L &&
    all(requests_311_selection$request_count > 0) &&
    setequal(
      requests_311_selection$sr_type_desc,
      read_csv(
        project_path("config", "311_smoke_signal_types.csv"),
        show_col_types = FALSE
      )$sr_type_desc
    ) &&
    all(
      unique(features$sr_311_source_scope) ==
        "configured_displacement_smoke_signal_types"
    ),
  paste(
    nrow(requests_311_selection),
    "configured request descriptions;",
    sum(requests_311_selection$request_count),
    "assigned requests"
  )
)

validation_file <- file.path(
  PART1_DIR,
  "baseline_cluster_validation.csv"
)
write_csv(checks, validation_file)

if (any(!checks$passed)) {
  failed <- checks$check[!checks$passed]
  stop(
    "Part 1 validation failed: ",
    paste(failed, collapse = ", "),
    ". See ",
    validation_file,
    ".",
    call. = FALSE
  )
}

canonical_assignments <- features %>%
  select(
    hex_id,
    h3_index,
    residential_units,
    total_pop,
    all_of(model$features)
  ) %>%
  inner_join(selected_assignments, by = "hex_id", relationship = "one-to-one") %>%
  left_join(
    selected_labels,
    by = "cluster",
    relationship = "many-to-one"
  ) %>%
  arrange(h3_index)
assignment_file <- file.path(
  PART1_DIR,
  "baseline_cluster_assignments.csv"
)
write_csv(canonical_assignments, assignment_file)

assignment_hash <- digest::digest(
  canonical_assignments %>% select(h3_index, cluster),
  algo = "sha256",
  serialize = TRUE
)
summary <- tibble(
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
  specification = specification,
  k = k,
  domains = length(model$features),
  classified_hexes = nrow(canonical_assignments),
  classified_population_share =
    classified_coverage$total_population_share[[1]],
  classified_housing_unit_share =
    classified_coverage$housing_unit_share[[1]],
  average_silhouette = selected_metrics$avg_silhouette[[1]],
  gap_statistic = selected_gap$gap[[1]],
  mean_subsample_adjusted_rand =
    selected_stability$mean_adjusted_rand[[1]],
  smallest_cluster_hexes = selected_metrics$min_cluster_n[[1]],
  largest_cluster_hexes = selected_metrics$max_cluster_n[[1]],
  assignment_sha256 = assignment_hash
)
summary_file <- file.path(PART1_DIR, "baseline_cluster_summary.csv")
write_csv(summary, summary_file)

pipeline_code_files <- c(
  "00_requirements.R",
  "01_create_hex_grid.R",
  "run_analysis.R",
  "_targets.R",
  list.files("R", pattern = "[.]R$", recursive = TRUE, full.names = TRUE),
  list.files(
    "scripts/data",
    pattern = "[.]R$",
    recursive = TRUE,
    full.names = TRUE
  ),
  list.files(
    "scripts/features",
    pattern = "[.]R$",
    recursive = TRUE,
    full.names = TRUE
  ),
  list.files(
    "scripts/part1",
    pattern = "[.]R$",
    recursive = TRUE,
    full.names = TRUE
  ),
  list.files(
    "scripts/audits",
    pattern = "[.]R$",
    recursive = TRUE,
    full.names = TRUE
  )
)
lock_files <- c(
  pipeline_code_files,
  "config/311_smoke_signal_types.csv",
  "config/feature_dictionary.csv",
  "config/amenity_cluster_labels_k6.csv",
  files[["features"]],
  files[["results"]],
  files[["assignments"]],
  files[["model"]]
)
lock_files <- unique(lock_files[file.exists(lock_files)])
project_root <- paste0(
  normalizePath(project_path(), winslash = "/", mustWork = TRUE),
  "/"
)
lock_items <- vapply(
  lock_files,
  function(path) {
    absolute_path <- normalizePath(path, winslash = "/", mustWork = TRUE)
    if (startsWith(absolute_path, project_root)) {
      substring(absolute_path, nchar(project_root) + 1L)
    } else {
      absolute_path
    }
  },
  character(1)
)
lock_manifest <- tibble(
  item_type = "file_sha256",
  item = unname(lock_items),
  value = vapply(
    lock_files,
    digest::digest,
    character(1),
    algo = "sha256",
    file = TRUE
  )
)
runtime_packages <- c(
  "targets", "dplyr", "readr", "sf", "cluster", "digest"
)
runtime_manifest <- tibble(
  item_type = "runtime",
  item = c("R", runtime_packages),
  value = c(
    R.version.string,
    vapply(
      runtime_packages,
      function(package) as.character(utils::packageVersion(package)),
      character(1)
    )
  )
)
git_commit <- tryCatch(
  system2("git", c("rev-parse", "HEAD"), stdout = TRUE, stderr = FALSE),
  error = function(error) NA_character_
)
git_status <- tryCatch(
  system2("git", c("status", "--porcelain"), stdout = TRUE, stderr = FALSE),
  error = function(error) NA_character_
)
git_manifest <- tibble(
  item_type = "git",
  item = c("commit_at_validation", "worktree_at_validation"),
  value = c(
    if (length(git_commit) == 1L) git_commit else NA_character_,
    if (length(git_status) == 0L) {
      "clean"
    } else if (all(is.na(git_status))) {
      NA_character_
    } else {
      paste0("dirty (", length(git_status), " changed paths)")
    }
  )
)
lock_file <- file.path(PART1_DIR, "baseline_cluster_lock.csv")
write_csv(
  bind_rows(lock_manifest, runtime_manifest, git_manifest),
  lock_file
)

print_progress(paste0("Passed ", nrow(checks), " Part 1 checks."))
print_progress(paste0("Validation: ", validation_file))
print_progress(paste0("Summary: ", summary_file))
print_progress(paste0("Canonical assignments: ", assignment_file))
print_progress(paste0("Lock manifest: ", lock_file))
