################################################################################
# Part 1 - Fit Baseline Clusters
################################################################################
#
# Isolates the effect of amenity pressure on the balanced clustering model:
#   1. Baseline: six equally scaled conceptual-domain indices
#   2. Amenity augmented: the same six indices plus amenity_change_index
#
# Both specifications use the same complete-case hex sample. Cluster counts from
# 2 through CLUSTER_MAX_K are compared using silhouette width, the gap statistic,
# cluster-size balance, and repeated 80-percent subsample stability. Stability
# is measured with the adjusted Rand index (ARI) against the corresponding
# full-sample solution, so arbitrary cluster-label permutations do not matter.
#
# Optional environment variables:
#   CLUSTER_MAX_K                default: 12
#   CLUSTER_GAP_BOOTSTRAPS       default: 100
#   CLUSTER_STABILITY_REPLICATES default: 100
#   CLUSTER_STABILITY_SHARE      default: 0.80
#   EWS_AMENITY_CLUSTER_K        default: 6; shared substantive solution
#
# Outputs:
#   output/amenity_cluster_sensitivity.rds
#   output/amenity_cluster_metrics.csv
#   output/amenity_cluster_gap_statistics.csv
#   output/amenity_cluster_stability.csv
#   output/amenity_cluster_recommendations.csv
#   output/amenity_cluster_agreement.csv
#   output/amenity_cluster_assignments.csv
#   output/amenity_cluster_profiles.csv
#   output/amenity_cluster_crosswalk.csv
#   output/amenity_cluster_selected_crosswalk.csv
#   output/amenity_cluster_selected_label_mapping.csv
#   output/amenity_cluster_population_coverage.csv
#   figures/03d_amenity_cluster_diagnostics.png
#   figures/03d_amenity_cluster_selected_maps.png
#   figures/03d_amenity_cluster_selected_profiles.png
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

suppressPackageStartupMessages({
  library(cluster)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(tidyr)
})

print_header("03d - AMENITY CLUSTER SENSITIVITY ANALYSIS")

OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
seed <- 42L

max_k <- as.integer(Sys.getenv("CLUSTER_MAX_K", unset = "12"))
gap_bootstraps <- as.integer(
  Sys.getenv("CLUSTER_GAP_BOOTSTRAPS", unset = "100")
)
stability_replicates <- as.integer(
  Sys.getenv("CLUSTER_STABILITY_REPLICATES", unset = "100")
)
stability_share <- as.numeric(
  Sys.getenv("CLUSTER_STABILITY_SHARE", unset = "0.80")
)

if (is.na(max_k) || max_k < 3) {
  stop("CLUSTER_MAX_K must be an integer of at least 3.", call. = FALSE)
}
if (is.na(gap_bootstraps) || gap_bootstraps < 10) {
  stop("CLUSTER_GAP_BOOTSTRAPS must be at least 10.", call. = FALSE)
}
if (is.na(stability_replicates) || stability_replicates < 20) {
  stop("CLUSTER_STABILITY_REPLICATES must be at least 20.", call. = FALSE)
}
if (is.na(stability_share) || stability_share < 0.50 || stability_share >= 1) {
  stop("CLUSTER_STABILITY_SHARE must be at least 0.50 and below 1.", call. = FALSE)
}

k_values <- 2:max_k
selected_solution_k <- EWS_CONFIG$amenity_cluster_k
if (!selected_solution_k %in% k_values) {
  stop(
    "EWS_AMENITY_CLUSTER_K must be within the evaluated range 2:",
    max_k,
    ".",
    call. = FALSE
  )
}

################################################################################
# Step 1: Build the paired analysis matrices
################################################################################

print_progress("Loading engineered features...")
hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  "engineered features"
)

baseline_vars <- c(
  "rent_pressure_citywide_index",
  "demographic_vulnerability_index",
  "demolition_pressure_index",
  "eviction_pressure_index",
  "sr_311_pressure_index",
  "ownership_pressure_index"
)
amenity_var <- "amenity_change_index"
all_vars <- c(baseline_vars, amenity_var)

missing_vars <- setdiff(all_vars, names(hex_features))
if (length(missing_vars) > 0) {
  stop(
    "Missing required clustering feature(s): ",
    paste(missing_vars, collapse = ", "),
    call. = FALSE
  )
}

eligibility_col <- if ("primary_cluster_eligible" %in% names(hex_features)) {
  "primary_cluster_eligible"
} else {
  "sufficient_data"
}

analysis_data <- hex_features %>%
  st_drop_geometry() %>%
  filter(.data[[eligibility_col]]) %>%
  select(hex_id, all_of(all_vars)) %>%
  filter(if_all(all_of(all_vars), ~is.finite(.x)))

if (nrow(analysis_data) < 20) {
  stop("Too few complete observations for amenity sensitivity.", call. = FALSE)
}

scale_matrix <- function(data, vars) {
  matrix <- data %>%
    select(all_of(vars)) %>%
    scale() %>%
    as.matrix()

  if (any(!is.finite(matrix))) {
    stop("Clustering matrix contains non-finite values.", call. = FALSE)
  }
  matrix
}

baseline_matrix <- scale_matrix(analysis_data, baseline_vars)
amenity_matrix <- scale_matrix(analysis_data, all_vars)
scaling_parameters <- list(
  baseline = list(
    center = attr(baseline_matrix, "scaled:center"),
    scale = attr(baseline_matrix, "scaled:scale")
  ),
  amenity_augmented = list(
    center = attr(amenity_matrix, "scaled:center"),
    scale = attr(amenity_matrix, "scaled:scale")
  )
)
specification_matrices <- list(
  baseline = baseline_matrix,
  amenity_augmented = amenity_matrix
)

print_progress(
  paste0(
    "Shared sample: ", format(nrow(analysis_data), big.mark = ","),
    " hexes; baseline has ", length(baseline_vars),
    " domains and augmented has ", length(all_vars), "."
  )
)

################################################################################
# Step 2: Fit full-sample solutions and calculate internal diagnostics
################################################################################

summarize_cluster_sizes <- function(cluster_ids) {
  sizes <- as.integer(table(cluster_ids))
  tibble(
    min_cluster_n = min(sizes),
    max_cluster_n = max(sizes),
    min_cluster_share = min(sizes) / length(cluster_ids),
    max_cluster_share = max(sizes) / length(cluster_ids)
  )
}

evaluate_full_sample <- function(matrix, specification) {
  pairwise_distance <- dist(matrix)
  models <- list()
  metrics <- list()

  for (k in k_values) {
    set.seed(seed + k)
    fit <- kmeans(
      matrix,
      centers = k,
      nstart = 100,
      iter.max = 500,
      algorithm = "Lloyd"
    )
    silhouette_result <- silhouette(fit$cluster, pairwise_distance)
    size_metrics <- summarize_cluster_sizes(fit$cluster)

    models[[as.character(k)]] <- fit
    metrics[[as.character(k)]] <- tibble(
      specification = specification,
      k = k,
      avg_silhouette = mean(silhouette_result[, "sil_width"]),
      between_ss_share = fit$betweenss / fit$totss,
      min_cluster_n = size_metrics$min_cluster_n,
      max_cluster_n = size_metrics$max_cluster_n,
      min_cluster_share = size_metrics$min_cluster_share,
      max_cluster_share = size_metrics$max_cluster_share
    )
  }

  list(models = models, metrics = bind_rows(metrics))
}

print_progress("Fitting full-sample baseline solutions...")
baseline_evaluation <- evaluate_full_sample(baseline_matrix, "baseline")

print_progress("Fitting full-sample amenity-augmented solutions...")
amenity_evaluation <- evaluate_full_sample(
  amenity_matrix,
  "amenity_augmented"
)

full_evaluations <- list(
  baseline = baseline_evaluation,
  amenity_augmented = amenity_evaluation
)
method_metrics <- bind_rows(
  baseline_evaluation$metrics,
  amenity_evaluation$metrics
)

################################################################################
# Step 3: Calculate gap statistics
################################################################################

gap_kmeans <- function(x, k) {
  list(
    cluster = kmeans(
      x,
      centers = k,
      nstart = 25,
      iter.max = 500,
      algorithm = "Lloyd"
    )$cluster
  )
}

calculate_gap <- function(matrix, specification) {
  print_progress(
    paste0(
      "Calculating gap statistic for ", specification,
      " with B = ", gap_bootstraps, "..."
    )
  )
  set.seed(seed)
  result <- clusGap(
    matrix,
    FUNcluster = gap_kmeans,
    K.max = max_k,
    B = gap_bootstraps,
    d.power = 2,
    spaceH0 = "scaledPCA",
    verbose = FALSE
  )

  table <- as.data.frame(result$Tab) %>%
    mutate(
      specification = specification,
      k = seq_len(n()),
      .before = 1
    ) %>%
    rename(
      log_within = logW,
      expected_log_within = E.logW,
      gap = gap,
      gap_se = SE.sim
    )

  list(
    result = result,
    table = table,
    tibs_k = maxSE(
      f = result$Tab[, "gap"],
      SE.f = result$Tab[, "SE.sim"],
      method = "Tibs2001SEmax"
    ),
    global_k = which.max(result$Tab[, "gap"])
  )
}

baseline_gap <- calculate_gap(baseline_matrix, "baseline")
amenity_gap <- calculate_gap(amenity_matrix, "amenity_augmented")
gap_results <- list(
  baseline = baseline_gap,
  amenity_augmented = amenity_gap
)
gap_statistics <- bind_rows(baseline_gap$table, amenity_gap$table)

################################################################################
# Step 4: Measure repeated-subsample stability
################################################################################

choose_two <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(labels_a, labels_b) {
  if (length(labels_a) != length(labels_b)) {
    stop("ARI label vectors must have equal length.", call. = FALSE)
  }

  contingency <- table(labels_a, labels_b)
  pair_total <- choose_two(length(labels_a))
  if (pair_total == 0) {
    return(NA_real_)
  }

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

sample_n <- floor(nrow(analysis_data) * stability_share)
stability_raw <- list()
stability_index <- 1L

print_progress(
  paste0(
    "Running ", stability_replicates,
    " paired stability subsamples of ", format(sample_n, big.mark = ","),
    " hexes..."
  )
)

set.seed(seed + 1000L)
subsamples <- replicate(
  stability_replicates,
  sample.int(nrow(analysis_data), size = sample_n, replace = FALSE),
  simplify = FALSE
)

for (specification in names(specification_matrices)) {
  matrix <- specification_matrices[[specification]]
  full_models <- full_evaluations[[specification]]$models

  for (replicate_id in seq_len(stability_replicates)) {
    sampled_rows <- subsamples[[replicate_id]]

    for (k in k_values) {
      set.seed(seed + replicate_id * 100L + k)
      subsample_fit <- kmeans(
        matrix[sampled_rows, , drop = FALSE],
        centers = k,
        nstart = 10,
        iter.max = 300,
        algorithm = "Lloyd"
      )

      stability_raw[[stability_index]] <- tibble(
        specification = specification,
        replicate = replicate_id,
        k = k,
        adjusted_rand = adjusted_rand_index(
          full_models[[as.character(k)]]$cluster[sampled_rows],
          subsample_fit$cluster
        )
      )
      stability_index <- stability_index + 1L
    }
  }
}

stability_raw <- bind_rows(stability_raw)
stability_summary <- stability_raw %>%
  group_by(specification, k) %>%
  summarise(
    mean_adjusted_rand = mean(adjusted_rand),
    median_adjusted_rand = median(adjusted_rand),
    p10_adjusted_rand = quantile(adjusted_rand, 0.10),
    p90_adjusted_rand = quantile(adjusted_rand, 0.90),
    .groups = "drop"
  )

################################################################################
# Step 5: Compare recommendations, assignments, and profiles
################################################################################

silhouette_recommendations <- method_metrics %>%
  group_by(specification) %>%
  slice_max(avg_silhouette, n = 1, with_ties = FALSE) %>%
  transmute(
    specification,
    diagnostic = "silhouette_global_max",
    recommended_k = k,
    diagnostic_value = avg_silhouette
  ) %>%
  ungroup()

stability_recommendations <- stability_summary %>%
  group_by(specification) %>%
  slice_max(mean_adjusted_rand, n = 1, with_ties = FALSE) %>%
  transmute(
    specification,
    diagnostic = "stability_global_max",
    recommended_k = k,
    diagnostic_value = mean_adjusted_rand
  ) %>%
  ungroup()

gap_recommendations <- bind_rows(
  tibble(
    specification = "baseline",
    diagnostic = c("gap_Tibs2001SEmax", "gap_global_max"),
    recommended_k = c(baseline_gap$tibs_k, baseline_gap$global_k),
    diagnostic_value = c(
      baseline_gap$table$gap[baseline_gap$tibs_k],
      baseline_gap$table$gap[baseline_gap$global_k]
    )
  ),
  tibble(
    specification = "amenity_augmented",
    diagnostic = c("gap_Tibs2001SEmax", "gap_global_max"),
    recommended_k = c(amenity_gap$tibs_k, amenity_gap$global_k),
    diagnostic_value = c(
      amenity_gap$table$gap[amenity_gap$tibs_k],
      amenity_gap$table$gap[amenity_gap$global_k]
    )
  )
)

substantive_selection <- tibble(
  specification = c("baseline", "amenity_augmented"),
  diagnostic = "substantive_selected",
  recommended_k = selected_solution_k,
  diagnostic_value = NA_real_
)

recommendations <- bind_rows(
  silhouette_recommendations,
  stability_recommendations,
  gap_recommendations,
  substantive_selection
) %>%
  mutate(
    at_search_boundary = recommended_k == max_k,
    supports_multiple_clusters = recommended_k >= 2
  )

assignment_rows <- list()
profile_rows <- list()
assignment_index <- 1L

for (specification in names(full_evaluations)) {
  for (k in k_values) {
    cluster_ids <- full_evaluations[[specification]]$models[[
      as.character(k)
    ]]$cluster

    assignment_rows[[assignment_index]] <- tibble(
      hex_id = analysis_data$hex_id,
      specification = specification,
      k = k,
      cluster = cluster_ids
    )

    profile_rows[[assignment_index]] <- analysis_data %>%
      mutate(cluster = cluster_ids) %>%
      group_by(cluster) %>%
      summarise(
        n = n(),
        across(all_of(all_vars), ~mean(.x, na.rm = TRUE)),
        .groups = "drop"
      ) %>%
      mutate(specification = specification, k = k, .before = 1)

    assignment_index <- assignment_index + 1L
  }
}

assignments <- bind_rows(assignment_rows)
profiles <- bind_rows(profile_rows)

agreement <- lapply(k_values, function(k) {
  baseline_cluster <- baseline_evaluation$models[[as.character(k)]]$cluster
  amenity_cluster <- amenity_evaluation$models[[as.character(k)]]$cluster
  tibble(
    k = k,
    adjusted_rand = adjusted_rand_index(baseline_cluster, amenity_cluster)
  )
}) %>%
  bind_rows()

crosswalk <- lapply(k_values, function(k) {
  tibble(
    baseline_cluster = baseline_evaluation$models[[as.character(k)]]$cluster,
    amenity_cluster = amenity_evaluation$models[[as.character(k)]]$cluster
  ) %>%
    count(baseline_cluster, amenity_cluster, name = "n") %>%
    group_by(baseline_cluster) %>%
    mutate(baseline_cluster_share = n / sum(n)) %>%
    ungroup() %>%
    group_by(amenity_cluster) %>%
    mutate(amenity_cluster_share = n / sum(n)) %>%
    ungroup() %>%
    mutate(k = k, .before = 1)
}) %>%
  bind_rows()

selected_k <- substantive_selection %>%
  select(specification, recommended_k) %>%
  tibble::deframe()

selected_assignments <- assignments %>%
  filter(
    (specification == "baseline" & k == selected_k[["baseline"]]) |
      (specification == "amenity_augmented" &
        k == selected_k[["amenity_augmented"]])
  )

selected_crosswalk <- selected_assignments %>%
  select(hex_id, specification, cluster) %>%
  pivot_wider(names_from = specification, values_from = cluster) %>%
  count(baseline, amenity_augmented, name = "n") %>%
  group_by(baseline) %>%
  mutate(baseline_cluster_share = n / sum(n)) %>%
  ungroup() %>%
  group_by(amenity_augmented) %>%
  mutate(amenity_cluster_share = n / sum(n)) %>%
  ungroup() %>%
  arrange(baseline, desc(n))

overlap_order <- selected_crosswalk %>% arrange(desc(n))
matched_baseline <- integer()
matched_augmented <- integer()
matched_rows <- list()

for (row_index in seq_len(nrow(overlap_order))) {
  baseline_id <- overlap_order$baseline[[row_index]]
  augmented_id <- overlap_order$amenity_augmented[[row_index]]

  if (!baseline_id %in% matched_baseline &&
      !augmented_id %in% matched_augmented) {
    matched_baseline <- c(matched_baseline, baseline_id)
    matched_augmented <- c(matched_augmented, augmented_id)
    matched_rows[[length(matched_rows) + 1L]] <- tibble(
      baseline = baseline_id,
      amenity_augmented = augmented_id
    )
  }
}

matched_rows <- bind_rows(matched_rows)
baseline_ids <- sort(unique(selected_crosswalk$baseline))
augmented_ids <- sort(unique(selected_crosswalk$amenity_augmented))
additional_augmented <- setdiff(augmented_ids, matched_rows$amenity_augmented)

cluster_label_mapping <- bind_rows(
  tibble(
    specification = "baseline",
    cluster = baseline_ids,
    display_pattern = paste0("B", baseline_ids)
  ),
  matched_rows %>%
    transmute(
      specification = "amenity_augmented",
      cluster = amenity_augmented,
      display_pattern = paste0("B", baseline)
    ),
  tibble(
    specification = "amenity_augmented",
    cluster = additional_augmented,
    display_pattern = paste0("Additional ", seq_along(additional_augmented))
  )
)

display_pattern_levels <- c(
  paste0("B", baseline_ids),
  paste0("Additional ", seq_along(additional_augmented))
)

selected_profiles <- profiles %>%
  filter(
    (specification == "baseline" & k == selected_k[["baseline"]]) |
      (specification == "amenity_augmented" &
        k == selected_k[["amenity_augmented"]])
  )

coverage_columns <- c(
  "residential_units",
  "total_pop",
  "population_in_occupied_housing",
  "total_housing_units"
)
missing_coverage_columns <- setdiff(coverage_columns, names(hex_features))

if (length(missing_coverage_columns) == 0) {
  population_coverage_data <- hex_features %>%
    st_drop_geometry() %>%
    select(hex_id, all_of(coverage_columns)) %>%
    mutate(
      coverage_status = case_when(
        hex_id %in% analysis_data$hex_id ~ "classified",
        residential_units < EWS_CONFIG$minimum_residential_units_for_rates ~
          "below_minimum_parcel_units",
        TRUE ~ "eligible_but_missing_cluster_feature"
      )
    )

  population_coverage <- population_coverage_data %>%
    group_by(coverage_status) %>%
    summarise(
      hexes = n(),
      parcel_units = sum(residential_units, na.rm = TRUE),
      total_population = sum(total_pop, na.rm = TRUE),
      household_population = sum(
        population_in_occupied_housing,
        na.rm = TRUE
      ),
      housing_units = sum(total_housing_units, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      total_population_share = total_population /
        sum(population_coverage_data$total_pop, na.rm = TRUE),
      household_population_share = household_population /
        sum(
          population_coverage_data$population_in_occupied_housing,
          na.rm = TRUE
        ),
      housing_unit_share = housing_units /
        sum(population_coverage_data$total_housing_units, na.rm = TRUE)
    )
} else {
  population_coverage <- tibble(
    coverage_status = "unavailable",
    note = paste(
      "Missing coverage columns:",
      paste(missing_coverage_columns, collapse = ", ")
    )
  )
}

################################################################################
# Step 6: Save tables, model objects, and diagnostic plot
################################################################################

write_csv(method_metrics, file.path(OUTPUT_DIR, "amenity_cluster_metrics.csv"))
write_csv(
  gap_statistics,
  file.path(OUTPUT_DIR, "amenity_cluster_gap_statistics.csv")
)
write_csv(
  stability_summary,
  file.path(OUTPUT_DIR, "amenity_cluster_stability.csv")
)
write_csv(
  recommendations,
  file.path(OUTPUT_DIR, "amenity_cluster_recommendations.csv")
)
write_csv(agreement, file.path(OUTPUT_DIR, "amenity_cluster_agreement.csv"))
write_csv(
  assignments,
  file.path(OUTPUT_DIR, "amenity_cluster_assignments.csv")
)
write_csv(profiles, file.path(OUTPUT_DIR, "amenity_cluster_profiles.csv"))
write_csv(crosswalk, file.path(OUTPUT_DIR, "amenity_cluster_crosswalk.csv"))
write_csv(
  selected_crosswalk,
  file.path(OUTPUT_DIR, "amenity_cluster_selected_crosswalk.csv")
)
write_csv(
  cluster_label_mapping,
  file.path(OUTPUT_DIR, "amenity_cluster_selected_label_mapping.csv")
)
write_csv(
  population_coverage,
  file.path(OUTPUT_DIR, "amenity_cluster_population_coverage.csv")
)

results <- list(
  created_at = Sys.time(),
  seed = seed,
  n_observations = nrow(analysis_data),
  baseline_vars = baseline_vars,
  amenity_var = amenity_var,
  scaling_parameters = scaling_parameters,
  max_k = max_k,
  gap_bootstraps = gap_bootstraps,
  stability_replicates = stability_replicates,
  stability_share = stability_share,
  full_evaluations = full_evaluations,
  gap_results = gap_results,
  stability_raw = stability_raw,
  stability_summary = stability_summary,
  recommendations = recommendations,
  agreement = agreement,
  assignments = assignments,
  profiles = profiles,
  crosswalk = crosswalk,
  selected_k = selected_k,
  selected_crosswalk = selected_crosswalk,
  cluster_label_mapping = cluster_label_mapping,
  population_coverage = population_coverage
)

save_output(
  results,
  file.path(OUTPUT_DIR, "amenity_cluster_sensitivity.rds"),
  "amenity cluster sensitivity analysis"
)

diagnostic_plot_data <- bind_rows(
  method_metrics %>%
    transmute(
      specification,
      k,
      diagnostic = "Average silhouette",
      value = avg_silhouette,
      lower = NA_real_,
      upper = NA_real_
    ),
  gap_statistics %>%
    filter(k >= 2) %>%
    transmute(
      specification,
      k,
      diagnostic = "Gap statistic",
      value = gap,
      lower = gap - gap_se,
      upper = gap + gap_se
    ),
  stability_summary %>%
    transmute(
      specification,
      k,
      diagnostic = "Subsample stability (ARI)",
      value = mean_adjusted_rand,
      lower = p10_adjusted_rand,
      upper = p90_adjusted_rand
    ),
  agreement %>%
    transmute(
      specification = "Baseline vs amenity",
      k,
      diagnostic = "Cross-specification agreement (ARI)",
      value = adjusted_rand,
      lower = NA_real_,
      upper = NA_real_
    )
)

p_diagnostics <- ggplot(
  diagnostic_plot_data,
  aes(x = k, y = value, color = specification, group = specification)
) +
  geom_ribbon(
    aes(ymin = lower, ymax = upper, fill = specification),
    alpha = 0.10,
    color = NA,
    na.rm = TRUE
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.8) +
  facet_wrap(~diagnostic, scales = "free_y", ncol = 2) +
  scale_x_continuous(breaks = k_values) +
  labs(
    title = "Amenity Domain Cluster Sensitivity",
    subtitle = paste0(
      format(nrow(analysis_data), big.mark = ","),
      " shared hexes; gap B = ", gap_bootstraps,
      "; stability replicates = ", stability_replicates
    ),
    x = "Number of clusters (k)",
    y = NULL,
    color = "Specification",
    fill = "Specification"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

ggsave(
  file.path(FIGURES_DIR, "03d_amenity_cluster_diagnostics.png"),
  p_diagnostics,
  width = 11,
  height = 8,
  dpi = 300,
  bg = "white"
)

specification_labels <- c(
  baseline = paste0("Baseline (k = ", selected_k[["baseline"]], ")"),
  amenity_augmented = paste0(
    "Amenity augmented (k = ", selected_k[["amenity_augmented"]], ")"
  )
)

selected_map_data <- hex_features %>%
  select(hex_id) %>%
  inner_join(selected_assignments, by = "hex_id") %>%
  left_join(
    cluster_label_mapping,
    by = c("specification", "cluster")
  ) %>%
  mutate(
    specification_label = factor(
      specification_labels[specification],
      levels = unname(specification_labels)
    ),
    display_pattern = factor(
      display_pattern,
      levels = display_pattern_levels
    )
  )

pattern_palette <- setNames(
  grDevices::hcl.colors(length(display_pattern_levels), palette = "Dark 3"),
  display_pattern_levels
)

p_maps <- ggplot(selected_map_data) +
  geom_sf(aes(fill = display_pattern), color = NA) +
  facet_wrap(~specification_label) +
  scale_fill_manual(values = pattern_palette, name = "Matched pattern") +
  coord_sf(datum = NA) +
  labs(title = "Selected Baseline and Amenity-Augmented Cluster Solutions") +
  theme_void(base_size = 11) +
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  file.path(FIGURES_DIR, "03d_amenity_cluster_selected_maps.png"),
  p_maps,
  width = 12,
  height = 6,
  dpi = 300,
  bg = "white"
)

feature_labels <- c(
  rent_pressure_citywide_index = "Rent pressure",
  demographic_vulnerability_index = "Demographic vulnerability",
  demolition_pressure_index = "Demolition pressure",
  eviction_pressure_index = "Eviction pressure",
  sr_311_pressure_index = "311 pressure",
  ownership_pressure_index = "Ownership pressure",
  amenity_change_index = "Amenity pressure"
)
feature_means <- vapply(analysis_data[all_vars], mean, numeric(1))
feature_sds <- vapply(analysis_data[all_vars], sd, numeric(1))

selected_profile_long <- selected_profiles %>%
  left_join(
    cluster_label_mapping,
    by = c("specification", "cluster")
  ) %>%
  pivot_longer(
    cols = all_of(all_vars),
    names_to = "feature",
    values_to = "feature_mean"
  ) %>%
  mutate(
    standardized_mean =
      (feature_mean - feature_means[feature]) / feature_sds[feature],
    feature_label = factor(
      feature_labels[feature],
      levels = rev(unname(feature_labels))
    ),
    specification_label = factor(
      specification_labels[specification],
      levels = unname(specification_labels)
    ),
    display_pattern = factor(
      display_pattern,
      levels = display_pattern_levels
    )
  )

p_profiles <- ggplot(
  selected_profile_long,
  aes(x = display_pattern, y = feature_label, fill = standardized_mean)
) +
  geom_tile(color = "white", linewidth = 0.4) +
  facet_wrap(~specification_label, scales = "free_x") +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    name = "Mean z-score"
  ) +
  labs(
    title = "Selected Cluster Profiles",
    subtitle = "Domain means standardized across the shared analysis sample",
    x = "Matched pattern",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    legend.position = "bottom",
    strip.text = element_text(face = "bold")
  )

ggsave(
  file.path(FIGURES_DIR, "03d_amenity_cluster_selected_profiles.png"),
  p_profiles,
  width = 12,
  height = 7,
  dpi = 300,
  bg = "white"
)

print_header("AMENITY CLUSTER SENSITIVITY RESULTS")
print(recommendations)

cat("\nDiagnostics at the substantively selected k:\n")
print(
  method_metrics %>%
    inner_join(
      substantive_selection %>% select(specification, recommended_k),
      by = "specification"
    ) %>%
    filter(k == recommended_k) %>%
    left_join(stability_summary, by = c("specification", "k")) %>%
    select(
      specification,
      k,
      avg_silhouette,
      mean_adjusted_rand,
      min_cluster_n,
      max_cluster_share
    )
)

cat("\n03d analysis complete. Canonical 03b outputs were not changed.\n")
