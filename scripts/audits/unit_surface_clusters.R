################################################################################
# Audit Unit-Surface Cluster Effects
################################################################################
#
# Re-fits the six-cluster amenity-augmented solution on the shadow unit feature
# table, aligns its arbitrary labels to the current six-cluster assignments,
# and reports eligibility, population coverage, profile, and transition effects.
# Canonical cluster outputs are not overwritten.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(tibble)
  library(tidyr)
})

source(here::here("R", "utils.R"))
source(here::here("R", "analysis_config.R"))

print_header("03f - COMPARE UNIT-SHADOW CLUSTER EFFECTS")

OUTPUT_DIR <- here::here("output")
CURRENT_FEATURE_FILE <- file.path(OUTPUT_DIR, "hex_features.rds")
SHADOW_FEATURE_FILE <- file.path(
  OUTPUT_DIR,
  "hex_features_unit_shadow.rds"
)
CURRENT_ASSIGNMENT_FILE <- file.path(
  OUTPUT_DIR,
  "amenity_cluster_assignments.csv"
)
LABEL_FILE <- here::here("config", "amenity_cluster_labels_k6.csv")
SOLUTION_K <- EWS_CONFIG$amenity_cluster_k
RANDOM_SEED <- 42L

required_files <- c(
  CURRENT_FEATURE_FILE,
  SHADOW_FEATURE_FILE,
  CURRENT_ASSIGNMENT_FILE,
  LABEL_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Build shadow features before 03f. Missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

cluster_vars <- c(
  "rent_pressure_citywide_index",
  "demographic_vulnerability_index",
  "demolition_pressure_index",
  "eviction_pressure_index",
  "sr_311_pressure_index",
  "ownership_pressure_index",
  "amenity_change_index"
)

current_features <- readRDS(CURRENT_FEATURE_FILE)
shadow_features <- readRDS(SHADOW_FEATURE_FILE)
missing_shadow_vars <- setdiff(cluster_vars, names(shadow_features))
if (length(missing_shadow_vars) > 0L) {
  stop(
    "Shadow feature table is missing clustering domains: ",
    paste(missing_shadow_vars, collapse = ", "),
    call. = FALSE
  )
}

current_assignments <- read_csv(
  CURRENT_ASSIGNMENT_FILE,
  show_col_types = FALSE
) %>%
  filter(
    specification == "amenity_augmented",
    k == SOLUTION_K
  ) %>%
  transmute(
    hex_id = as.character(hex_id),
    current_cluster = as.integer(cluster)
  )
if (
  nrow(current_assignments) == 0L ||
    n_distinct(current_assignments$current_cluster) != SOLUTION_K ||
    anyDuplicated(current_assignments$hex_id)
) {
  stop("Current amenity k=6 assignments failed validation.", call. = FALSE)
}

shadow_analysis <- shadow_features %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id)) %>%
  filter(primary_cluster_eligible) %>%
  select(
    hex_id,
    residential_units,
    total_pop,
    all_of(cluster_vars)
  ) %>%
  filter(if_all(all_of(cluster_vars), ~is.finite(.x)))
if (nrow(shadow_analysis) < 100L) {
  stop("Too few complete shadow hexes for six-cluster comparison.", call. = FALSE)
}

shadow_matrix <- shadow_analysis %>%
  select(all_of(cluster_vars)) %>%
  scale() %>%
  as.matrix()
if (any(!is.finite(shadow_matrix))) {
  stop("Shadow clustering matrix contains non-finite values.", call. = FALSE)
}

set.seed(RANDOM_SEED + SOLUTION_K)
shadow_fit <- kmeans(
  shadow_matrix,
  centers = SOLUTION_K,
  nstart = 100,
  iter.max = 500,
  algorithm = "Lloyd"
)
shadow_assignments_raw <- shadow_analysis %>%
  transmute(
    hex_id,
    shadow_cluster_raw = as.integer(shadow_fit$cluster)
  )

choose_two <- function(x) x * (x - 1) / 2
adjusted_rand_index <- function(labels_a, labels_b) {
  contingency <- table(labels_a, labels_b)
  pair_total <- choose_two(length(labels_a))
  observed <- sum(choose_two(contingency))
  row_pairs <- sum(choose_two(rowSums(contingency)))
  column_pairs <- sum(choose_two(colSums(contingency)))
  expected <- row_pairs * column_pairs / pair_total
  maximum <- 0.5 * (row_pairs + column_pairs)
  denominator <- maximum - expected
  if (abs(denominator) < .Machine$double.eps) {
    return(if (all(labels_a == labels_b)) 1 else 0)
  }
  (observed - expected) / denominator
}

all_permutations <- function(values) {
  if (length(values) == 1L) {
    return(matrix(values, nrow = 1L))
  }
  do.call(
    rbind,
    lapply(
      seq_along(values),
      function(index) {
        cbind(
          values[[index]],
          all_permutations(values[-index])
        )
      }
    )
  )
}

common_raw <- current_assignments %>%
  inner_join(
    shadow_assignments_raw,
    by = "hex_id",
    relationship = "one-to-one"
  )
if (nrow(common_raw) == 0L) {
  stop("Current and shadow cluster samples do not overlap.", call. = FALSE)
}

overlap_table <- table(
  factor(common_raw$current_cluster, levels = seq_len(SOLUTION_K)),
  factor(common_raw$shadow_cluster_raw, levels = seq_len(SOLUTION_K))
)
permutations <- all_permutations(seq_len(SOLUTION_K))
permutation_scores <- apply(
  permutations,
  1,
  function(mapping) {
    sum(
      vapply(
        seq_len(SOLUTION_K),
        function(shadow_id) {
          overlap_table[mapping[[shadow_id]], shadow_id]
        },
        numeric(1)
      )
    )
  }
)
best_mapping <- permutations[which.max(permutation_scores), ]
cluster_mapping <- tibble(
  shadow_cluster_raw = seq_len(SOLUTION_K),
  shadow_cluster = as.integer(best_mapping)
)

shadow_assignments <- shadow_assignments_raw %>%
  left_join(
    cluster_mapping,
    by = "shadow_cluster_raw",
    relationship = "many-to-one"
  )

assignment_comparison <- shadow_features %>%
  st_drop_geometry() %>%
  transmute(
    hex_id = as.character(hex_id),
    shadow_residential_units = residential_units,
    total_pop
  ) %>%
  left_join(
    current_assignments,
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  left_join(
    shadow_assignments,
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    assignment_status = case_when(
      !is.na(current_cluster) & !is.na(shadow_cluster) ~ "assigned_both",
      is.na(current_cluster) & !is.na(shadow_cluster) ~
        "assigned_shadow_only",
      !is.na(current_cluster) & is.na(shadow_cluster) ~
        "assigned_current_only",
      TRUE ~ "unassigned_both"
    ),
    cluster_changed = assignment_status == "assigned_both" &
      current_cluster != shadow_cluster
  )

common_assignments <- assignment_comparison %>%
  filter(assignment_status == "assigned_both")
ari <- adjusted_rand_index(
  common_assignments$current_cluster,
  common_assignments$shadow_cluster
)
exact_agreement <- mean(
  common_assignments$current_cluster ==
    common_assignments$shadow_cluster
)

transition_summary <- assignment_comparison %>%
  filter(assignment_status != "unassigned_both") %>%
  count(
    assignment_status,
    current_cluster,
    shadow_cluster,
    name = "hexes"
  ) %>%
  left_join(
    assignment_comparison %>%
      filter(assignment_status != "unassigned_both") %>%
      group_by(
        assignment_status,
        current_cluster,
        shadow_cluster
      ) %>%
      summarise(
        total_population = sum(total_pop, na.rm = TRUE),
        shadow_residential_units = sum(
          shadow_residential_units,
          na.rm = TRUE
        ),
        .groups = "drop"
      ),
    by = c(
      "assignment_status",
      "current_cluster",
      "shadow_cluster"
    ),
    relationship = "one-to-one"
  ) %>%
  arrange(
    assignment_status,
    current_cluster,
    shadow_cluster
  )

population_coverage <- assignment_comparison %>%
  mutate(
    coverage_status = case_when(
      !is.na(current_cluster) & !is.na(shadow_cluster) ~ "classified_both",
      is.na(current_cluster) & !is.na(shadow_cluster) ~
        "classified_shadow_only",
      !is.na(current_cluster) & is.na(shadow_cluster) ~
        "classified_current_only",
      TRUE ~ "unclassified_both"
    )
  ) %>%
  group_by(coverage_status) %>%
  summarise(
    hexes = n(),
    total_population = sum(total_pop, na.rm = TRUE),
    shadow_residential_units = sum(
      shadow_residential_units,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) %>%
  mutate(
    total_population_share = total_population /
      sum(total_population)
  )

labels <- read_csv(LABEL_FILE, show_col_types = FALSE) %>%
  filter(solution_k == SOLUTION_K) %>%
  select(
    cluster,
    tentative_name,
    concern_level
  )

current_profiles <- current_features %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id)) %>%
  inner_join(
    current_assignments,
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  group_by(cluster = current_cluster) %>%
  summarise(
    n = n(),
    across(all_of(cluster_vars), ~mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(scenario = "current", .before = 1)

shadow_profiles <- shadow_features %>%
  st_drop_geometry() %>%
  mutate(hex_id = as.character(hex_id)) %>%
  inner_join(
    shadow_assignments %>% select(hex_id, shadow_cluster),
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  group_by(cluster = shadow_cluster) %>%
  summarise(
    n = n(),
    across(all_of(cluster_vars), ~mean(.x, na.rm = TRUE)),
    .groups = "drop"
  ) %>%
  mutate(scenario = "unit_shadow", .before = 1)

profiles <- bind_rows(current_profiles, shadow_profiles) %>%
  left_join(labels, by = "cluster", relationship = "many-to-one")

metrics <- tibble(
  metric = c(
    "current_assigned_hexes",
    "shadow_assigned_hexes",
    "common_assigned_hexes",
    "shadow_only_assigned_hexes",
    "current_only_assigned_hexes",
    "changed_cluster_on_common_sample",
    "aligned_exact_agreement",
    "adjusted_rand_index"
  ),
  value = c(
    nrow(current_assignments),
    nrow(shadow_assignments),
    nrow(common_assignments),
    sum(assignment_comparison$assignment_status == "assigned_shadow_only"),
    sum(assignment_comparison$assignment_status == "assigned_current_only"),
    sum(common_assignments$cluster_changed),
    exact_agreement,
    ari
  )
)

results <- list(
  shadow_model = shadow_fit,
  cluster_mapping = cluster_mapping,
  assignments = assignment_comparison,
  metrics = metrics,
  transitions = transition_summary,
  population_coverage = population_coverage,
  profiles = profiles,
  cluster_variables = cluster_vars,
  solution_k = SOLUTION_K,
  random_seed = RANDOM_SEED
)

save_output(
  results,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_comparison.rds"),
  "unit-shadow cluster comparison"
)
write_csv(
  assignment_comparison,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_assignments.csv")
)
write_csv(
  cluster_mapping,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_label_mapping.csv")
)
write_csv(
  metrics,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_metrics.csv")
)
write_csv(
  transition_summary,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_transitions.csv")
)
write_csv(
  population_coverage,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_population_coverage.csv")
)
write_csv(
  profiles,
  file.path(OUTPUT_DIR, "unit_shadow_cluster_profiles.csv")
)

print_progress("Six-cluster shadow comparison:")
print(metrics)
print_progress("Population coverage transitions:")
print(population_coverage)
print_progress(
  "Saved shadow cluster comparison; canonical assignments remain unchanged."
)
