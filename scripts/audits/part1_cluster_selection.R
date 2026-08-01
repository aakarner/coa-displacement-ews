################################################################################
# Part 1 Cluster-Count Selection and Spatial Stability Audit
################################################################################
#
# Compares candidate amenity-augmented k-means solutions using matched random
# and spatially blocked holdouts. Scaling parameters and centroids are learned
# from training areas only. The audit also reports whether clusters separate
# zero-event from positive-event hexes for sparse displacement signals.
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
  library(dplyr)
  library(ggplot2)
  library(h3jsr)
  library(readr)
  library(sf)
  library(tidyr)
})

print_header("PART 1 CLUSTER SELECTION AND SPATIAL STABILITY")

OUTPUT_DIR <- project_path("output")
PART1_DIR <- file.path(OUTPUT_DIR, "part1")
FIGURES_DIR <- project_path("figures")
dir.create(PART1_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

FEATURE_FILE <- file.path(OUTPUT_DIR, "hex_features.rds")
RESULT_FILE <- file.path(OUTPUT_DIR, "amenity_cluster_sensitivity.rds")
ASSIGNMENT_FILE <- file.path(OUTPUT_DIR, "amenity_cluster_assignments.csv")
METRIC_FILE <- file.path(OUTPUT_DIR, "amenity_cluster_metrics.csv")
GAP_FILE <- file.path(OUTPUT_DIR, "amenity_cluster_gap_statistics.csv")
RANDOM_STABILITY_FILE <- file.path(OUTPUT_DIR, "amenity_cluster_stability.csv")
SELECTION_CONFIG_FILE <- project_path("config", "part1_cluster_selection.csv")

required_files <- c(
  FEATURE_FILE,
  RESULT_FILE,
  ASSIGNMENT_FILE,
  METRIC_FILE,
  GAP_FILE,
  RANDOM_STABILITY_FILE,
  SELECTION_CONFIG_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Cluster-selection inputs are missing: ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

spatial_replicates <- as.integer(
  Sys.getenv("CLUSTER_SPATIAL_REPLICATES", unset = "100")
)
holdout_share <- as.numeric(
  Sys.getenv("CLUSTER_HOLDOUT_SHARE", unset = "0.20")
)
spatial_parent_resolutions <- strsplit(
  Sys.getenv("CLUSTER_SPATIAL_PARENT_RESOLUTIONS", unset = "8,7"),
  ",",
  fixed = TRUE
)[[1]] %>%
  as.integer() %>%
  unique() %>%
  sort(decreasing = TRUE)
holdout_nstart <- as.integer(
  Sys.getenv("CLUSTER_HOLDOUT_NSTART", unset = "25")
)

if (is.na(spatial_replicates) || spatial_replicates < 20L) {
  stop("CLUSTER_SPATIAL_REPLICATES must be at least 20.", call. = FALSE)
}
if (is.na(holdout_share) || holdout_share <= 0 || holdout_share >= 0.5) {
  stop("CLUSTER_HOLDOUT_SHARE must be above 0 and below 0.5.", call. = FALSE)
}
if (
  length(spatial_parent_resolutions) < 2L ||
    anyNA(spatial_parent_resolutions) ||
    any(spatial_parent_resolutions >= EWS_CONFIG$h3_resolution) ||
    any(spatial_parent_resolutions < 0L)
) {
  stop(
    "Specify at least two valid H3 parent resolutions below the analysis resolution.",
    call. = FALSE
  )
}
if (is.na(holdout_nstart) || holdout_nstart < 10L) {
  stop("CLUSTER_HOLDOUT_NSTART must be at least 10.", call. = FALSE)
}

results <- readRDS(RESULT_FILE)
features_sf <- readRDS(FEATURE_FILE)
features <- st_drop_geometry(features_sf)
assignments <- read_csv(ASSIGNMENT_FILE, show_col_types = FALSE)
method_metrics <- read_csv(METRIC_FILE, show_col_types = FALSE)
gap_statistics <- read_csv(GAP_FILE, show_col_types = FALSE)
random_subsample_stability <- read_csv(
  RANDOM_STABILITY_FILE,
  show_col_types = FALSE
)
selection_config <- read_csv(SELECTION_CONFIG_FILE, show_col_types = FALSE)

specification <- "amenity_augmented"
cluster_features <- c(results$baseline_vars, results$amenity_var)
k_values <- seq.int(2L, results$max_k)
focus_k <- intersect(5:8, k_values)
seed <- as.integer(results$seed) + 7000L

if (
  nrow(selection_config) != 1L ||
    selection_config$selected_specification[[1]] != specification ||
    selection_config$selected_k[[1]] != EWS_CONFIG$amenity_cluster_k
) {
  stop(
    "Part 1 cluster-selection configuration must contain one row matching ",
    "the configured specification and selected k.",
    call. = FALSE
  )
}

required_feature_columns <- c(
  "hex_id",
  "h3_index",
  "total_pop",
  "residential_units",
  cluster_features,
  "eviction_cases_latest_12mo",
  "demo_total_demolition_count",
  "sr_311_smoke_signal_latest_12mo",
  "ownership_change_recent_count",
  "amenity_recent_opening_events"
)
missing_feature_columns <- setdiff(required_feature_columns, names(features))
if (length(missing_feature_columns) > 0L) {
  stop(
    "Cluster-selection features are missing: ",
    paste(missing_feature_columns, collapse = ", "),
    call. = FALSE
  )
}
if (anyDuplicated(features$hex_id) || anyDuplicated(features$h3_index)) {
  stop("Feature hex IDs and H3 indexes must be unique.", call. = FALSE)
}

amenity_assignments <- assignments %>%
  filter(specification == !!specification, k %in% k_values)
assignment_counts <- amenity_assignments %>% count(k, name = "n")
if (
  nrow(assignment_counts) != length(k_values) ||
    n_distinct(assignment_counts$n) != 1L
) {
  stop("Candidate solutions do not share one analysis sample.", call. = FALSE)
}

analysis_hex_ids <- amenity_assignments %>%
  filter(k == min(k_values)) %>%
  arrange(hex_id) %>%
  pull(hex_id)
analysis_data <- features %>%
  filter(hex_id %in% analysis_hex_ids) %>%
  arrange(match(hex_id, analysis_hex_ids))
if (
  nrow(analysis_data) != length(analysis_hex_ids) ||
    any(!is.finite(as.matrix(analysis_data[, cluster_features])))
) {
  stop("The cluster-selection sample is incomplete or non-finite.", call. = FALSE)
}

raw_matrix <- as.matrix(analysis_data[, cluster_features, drop = FALSE])
storage.mode(raw_matrix) <- "double"
full_center <- results$scaling_parameters[[specification]]$center[
  cluster_features
]
full_scale <- results$scaling_parameters[[specification]]$scale[
  cluster_features
]
if (
  any(!is.finite(full_center)) ||
    any(!is.finite(full_scale)) ||
    any(full_scale <= 0)
) {
  stop("Full-sample cluster scaling parameters are invalid.", call. = FALSE)
}
full_matrix <- sweep(raw_matrix, 2, full_center, FUN = "-")
full_matrix <- sweep(full_matrix, 2, full_scale, FUN = "/")

full_labels <- lapply(k_values, function(k) {
  candidate <- amenity_assignments %>%
    filter(k == !!k) %>%
    select(hex_id, cluster)
  labels <- candidate$cluster[match(analysis_data$hex_id, candidate$hex_id)]
  if (anyNA(labels) || !setequal(labels, seq_len(k))) {
    stop("Full-sample assignments are invalid for k = ", k, call. = FALSE)
  }
  as.integer(labels)
})
names(full_labels) <- as.character(k_values)

full_models <- results$full_evaluations[[specification]]$models
for (k in k_values) {
  fit <- full_models[[as.character(k)]]
  nearest <- max.col(
    -vapply(
      seq_len(k),
      function(center_index) {
        centered <- sweep(
          full_matrix,
          2,
          fit$centers[center_index, ],
          FUN = "-"
        )
        sqrt(rowSums(centered^2))
      },
      numeric(nrow(full_matrix))
    ),
    ties.method = "first"
  )
  if (!identical(as.integer(nearest), full_labels[[as.character(k)]])) {
    stop("Stored centroids do not reproduce assignments for k = ", k, call. = FALSE)
  }
}

choose_two <- function(x) x * (x - 1) / 2

adjusted_rand_index <- function(labels_a, labels_b) {
  if (length(labels_a) != length(labels_b)) {
    stop("ARI label vectors must have equal length.", call. = FALSE)
  }
  contingency <- table(labels_a, labels_b)
  pair_total <- choose_two(length(labels_a))
  if (pair_total == 0) return(NA_real_)
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

distance_matrix <- function(x, centers) {
  distances <- vapply(
    seq_len(nrow(centers)),
    function(center_index) {
      centered <- sweep(x, 2, centers[center_index, ], FUN = "-")
      sqrt(rowSums(centered^2))
    },
    numeric(nrow(x))
  )
  if (is.null(dim(distances))) {
    distances <- matrix(distances, ncol = nrow(centers))
  }
  distances
}

assignment_margin <- function(distances) {
  nearest <- max.col(-distances, ties.method = "first")
  minimum <- distances[cbind(seq_len(nrow(distances)), nearest)]
  second <- apply(distances, 1, function(row) sort(row, partial = 2)[[2]])
  tibble(
    nearest = nearest,
    minimum_distance = minimum,
    second_distance = second,
    margin_confidence = pmax(0, pmin(1, 1 - minimum / second))
  )
}

# O(k^3) Hungarian assignment for the small square cluster-overlap matrices.
hungarian_min_assignment <- function(cost) {
  cost <- as.matrix(cost)
  n <- nrow(cost)
  m <- ncol(cost)
  if (n == 0L || n != m || any(!is.finite(cost))) {
    stop("Hungarian assignment requires a finite square matrix.", call. = FALSE)
  }

  u <- numeric(n)
  v <- numeric(m + 1L)
  p <- integer(m + 1L)
  way <- integer(m + 1L)

  for (i in seq_len(n)) {
    p[[1L]] <- i
    j0 <- 0L
    minv <- rep(Inf, m + 1L)
    used <- rep(FALSE, m + 1L)

    repeat {
      used[[j0 + 1L]] <- TRUE
      i0 <- p[[j0 + 1L]]
      delta <- Inf
      j1 <- 0L

      for (j in seq_len(m)) {
        if (!used[[j + 1L]]) {
          current <- cost[i0, j] - u[[i0]] - v[[j + 1L]]
          if (current < minv[[j + 1L]]) {
            minv[[j + 1L]] <- current
            way[[j + 1L]] <- j0
          }
          if (minv[[j + 1L]] < delta) {
            delta <- minv[[j + 1L]]
            j1 <- j
          }
        }
      }

      for (j in 0:m) {
        if (used[[j + 1L]]) {
          assigned_row <- p[[j + 1L]]
          if (assigned_row > 0L) u[[assigned_row]] <- u[[assigned_row]] + delta
          v[[j + 1L]] <- v[[j + 1L]] - delta
        } else {
          minv[[j + 1L]] <- minv[[j + 1L]] - delta
        }
      }

      j0 <- j1
      if (p[[j0 + 1L]] == 0L) break
    }

    repeat {
      j1 <- way[[j0 + 1L]]
      p[[j0 + 1L]] <- p[[j1 + 1L]]
      j0 <- j1
      if (j0 == 0L) break
    }
  }

  assignment <- integer(n)
  for (j in seq_len(m)) {
    if (p[[j + 1L]] > 0L) assignment[[p[[j + 1L]]]] <- j
  }
  if (!setequal(assignment, seq_len(n))) {
    stop("Hungarian assignment did not return a permutation.", call. = FALSE)
  }
  assignment
}

hungarian_test <- matrix(c(4, 1, 3, 2, 0, 5, 3, 2, 2), nrow = 3, byrow = TRUE)
hungarian_test_assignment <- hungarian_min_assignment(hungarian_test)
if (sum(hungarian_test[cbind(1:3, hungarian_test_assignment)]) != 5) {
  stop("Hungarian assignment self-test failed.", call. = FALSE)
}

select_block_holdout <- function(block_ids, target_n, selection_seed) {
  block_sizes <- table(block_ids)
  set.seed(selection_seed)
  order_index <- sample.int(length(block_sizes))
  cumulative <- cumsum(as.integer(block_sizes[order_index]))
  take <- which.min(abs(cumulative - target_n))
  take <- max(1L, min(take, length(block_sizes) - 1L))
  held_blocks <- names(block_sizes)[order_index[seq_len(take)]]
  which(block_ids %in% held_blocks)
}

################################################################################
# Spatial and random held-out stability
################################################################################

scheme_blocks <- list(random_hex = paste0("hex_", seq_len(nrow(analysis_data))))
for (parent_resolution in spatial_parent_resolutions) {
  scheme_blocks[[paste0("h3_parent_r", parent_resolution)]] <- get_parent(
    analysis_data$h3_index,
    res = parent_resolution
  )
}

scheme_summary <- lapply(names(scheme_blocks), function(scheme) {
  sizes <- as.integer(table(scheme_blocks[[scheme]]))
  tibble(
    scheme = scheme,
    block_count = length(sizes),
    median_hexes_per_block = median(sizes),
    p90_hexes_per_block = quantile(sizes, 0.90),
    maximum_hexes_per_block = max(sizes)
  )
}) %>% bind_rows()

target_holdout_n <- round(nrow(analysis_data) * holdout_share)
replicate_rows <- list()
cluster_replicate_rows <- list()
replicate_index <- 1L
cluster_replicate_index <- 1L

for (scheme_index in seq_along(scheme_blocks)) {
  scheme <- names(scheme_blocks)[[scheme_index]]
  block_ids <- scheme_blocks[[scheme]]
  print_progress(
    paste0(
      "Running ", spatial_replicates, " ", scheme,
      " holdouts across k = ", min(k_values), ":", max(k_values), "..."
    )
  )

  for (replicate_id in seq_len(spatial_replicates)) {
    if (replicate_id %% 20L == 0L) {
      print_progress(paste0(scheme, " replicate ", replicate_id))
    }

    holdout_rows <- select_block_holdout(
      block_ids,
      target_n = target_holdout_n,
      selection_seed = seed + scheme_index * 100000L + replicate_id
    )
    training_rows <- setdiff(seq_len(nrow(analysis_data)), holdout_rows)
    training_center <- colMeans(raw_matrix[training_rows, , drop = FALSE])
    training_scale <- apply(
      raw_matrix[training_rows, , drop = FALSE],
      2,
      sd
    )
    if (
      any(!is.finite(training_center)) ||
        any(!is.finite(training_scale)) ||
        any(training_scale <= 0)
    ) {
      stop(
        "Invalid training scaling in ", scheme,
        " replicate ", replicate_id, ".",
        call. = FALSE
      )
    }

    training_matrix <- sweep(
      raw_matrix[training_rows, , drop = FALSE],
      2,
      training_center,
      FUN = "-"
    )
    training_matrix <- sweep(training_matrix, 2, training_scale, FUN = "/")
    holdout_matrix <- sweep(
      raw_matrix[holdout_rows, , drop = FALSE],
      2,
      training_center,
      FUN = "-"
    )
    holdout_matrix <- sweep(holdout_matrix, 2, training_scale, FUN = "/")

    for (k in k_values) {
      set.seed(seed + scheme_index * 1000000L + replicate_id * 100L + k)
      fit <- kmeans(
        training_matrix,
        centers = k,
        nstart = holdout_nstart,
        iter.max = 500,
        algorithm = "Lloyd"
      )

      full_cluster <- full_labels[[as.character(k)]]
      overlap <- table(
        factor(fit$cluster, levels = seq_len(k)),
        factor(full_cluster[training_rows], levels = seq_len(k))
      )
      label_mapping <- hungarian_min_assignment(max(overlap) - overlap)

      holdout_distances <- distance_matrix(holdout_matrix, fit$centers)
      holdout_confidence <- assignment_margin(holdout_distances)
      predicted_cluster <- label_mapping[holdout_confidence$nearest]
      observed_cluster <- full_cluster[holdout_rows]

      original_centers <- sweep(fit$centers, 2, training_scale, FUN = "*")
      original_centers <- sweep(original_centers, 2, training_center, FUN = "+")
      common_centers <- sweep(original_centers, 2, full_center, FUN = "-")
      common_centers <- sweep(common_centers, 2, full_scale, FUN = "/")
      full_fit_centers <- full_models[[as.character(k)]]$centers
      centroid_drift <- vapply(
        seq_len(k),
        function(center_index) {
          sqrt(sum(
            (
              common_centers[center_index, ] -
                full_fit_centers[label_mapping[[center_index]], ]
            )^2
          ))
        },
        numeric(1)
      )

      replicate_rows[[replicate_index]] <- tibble(
        scheme = scheme,
        replicate = replicate_id,
        k = k,
        holdout_hexes = length(holdout_rows),
        training_hexes = length(training_rows),
        holdout_blocks = n_distinct(block_ids[holdout_rows]),
        adjusted_rand = adjusted_rand_index(
          observed_cluster,
          predicted_cluster
        ),
        matched_agreement = mean(observed_cluster == predicted_cluster),
        mean_centroid_drift = mean(centroid_drift),
        maximum_centroid_drift = max(centroid_drift),
        occupied_holdout_clusters = n_distinct(predicted_cluster),
        mean_holdout_margin = mean(holdout_confidence$margin_confidence),
        p10_holdout_margin = quantile(
          holdout_confidence$margin_confidence,
          0.10
        )
      )
      replicate_index <- replicate_index + 1L

      for (cluster_id in seq_len(k)) {
        true_member <- observed_cluster == cluster_id
        predicted_member <- predicted_cluster == cluster_id
        intersection_n <- sum(true_member & predicted_member)
        union_n <- sum(true_member | predicted_member)
        cluster_replicate_rows[[cluster_replicate_index]] <- tibble(
          scheme = scheme,
          replicate = replicate_id,
          k = k,
          cluster = cluster_id,
          true_n = sum(true_member),
          predicted_n = sum(predicted_member),
          intersection_n = intersection_n,
          precision = if_else(
            sum(predicted_member) > 0,
            intersection_n / sum(predicted_member),
            NA_real_
          ),
          recall = if_else(
            sum(true_member) > 0,
            intersection_n / sum(true_member),
            NA_real_
          ),
          jaccard = if_else(union_n > 0, intersection_n / union_n, NA_real_),
          true_cluster_absent = sum(true_member) == 0L,
          predicted_cluster_absent = sum(predicted_member) == 0L
        )
        cluster_replicate_index <- cluster_replicate_index + 1L
      }
    }
  }
}

stability_replicates <- bind_rows(replicate_rows)
stability_cluster_replicates <- bind_rows(cluster_replicate_rows)

stability_summary <- stability_replicates %>%
  group_by(scheme, k) %>%
  summarise(
    replicates = n(),
    mean_adjusted_rand = mean(adjusted_rand),
    median_adjusted_rand = median(adjusted_rand),
    p10_adjusted_rand = quantile(adjusted_rand, 0.10),
    worst_adjusted_rand = min(adjusted_rand),
    mean_matched_agreement = mean(matched_agreement),
    median_matched_agreement = median(matched_agreement),
    p10_matched_agreement = quantile(matched_agreement, 0.10),
    mean_centroid_drift = mean(mean_centroid_drift),
    p90_maximum_centroid_drift = quantile(maximum_centroid_drift, 0.90),
    holdout_cluster_collapse_rate = mean(occupied_holdout_clusters < k),
    mean_holdout_margin = mean(mean_holdout_margin),
    p10_holdout_margin = mean(p10_holdout_margin),
    mean_holdout_hexes = mean(holdout_hexes),
    .groups = "drop"
  )

stability_cluster_summary <- stability_cluster_replicates %>%
  group_by(scheme, k, cluster) %>%
  summarise(
    mean_precision = mean(precision, na.rm = TRUE),
    p10_precision = quantile(precision, 0.10, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    p10_recall = quantile(recall, 0.10, na.rm = TRUE),
    mean_jaccard = mean(jaccard, na.rm = TRUE),
    p10_jaccard = quantile(jaccard, 0.10, na.rm = TRUE),
    true_cluster_absence_rate = mean(true_cluster_absent),
    predicted_cluster_absence_rate = mean(predicted_cluster_absent),
    .groups = "drop"
  )

################################################################################
# Full-sample assignment confidence and sparse-signal separation
################################################################################

confidence_rows <- list()
profile_rows <- list()
confidence_index <- 1L

for (k in k_values) {
  candidate_k <- k
  fit <- full_models[[as.character(k)]]
  confidence <- assignment_margin(distance_matrix(full_matrix, fit$centers))
  confidence_rows[[confidence_index]] <- tibble(
    hex_id = analysis_data$hex_id,
    k = candidate_k,
    cluster = full_labels[[as.character(candidate_k)]],
    minimum_distance = confidence$minimum_distance,
    second_distance = confidence$second_distance,
    margin_confidence = confidence$margin_confidence,
    low_margin = margin_confidence < 0.10
  )

  profile_rows[[confidence_index]] <- analysis_data %>%
    mutate(cluster = full_labels[[as.character(k)]]) %>%
    group_by(cluster) %>%
    summarise(
      n = n(),
      total_population = sum(total_pop, na.rm = TRUE),
      total_residential_units = sum(residential_units, na.rm = TRUE),
      across(
        all_of(cluster_features),
        list(
          mean = ~mean(.x),
          median = ~median(.x),
          zero_share = ~mean(.x == 0)
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    ) %>%
    mutate(
      population_share = total_population / sum(total_population),
      residential_unit_share =
        total_residential_units / sum(total_residential_units),
      k = k,
      .before = 1
    )
  confidence_index <- confidence_index + 1L
}

assignment_confidence <- bind_rows(confidence_rows)
cluster_profiles <- bind_rows(profile_rows)
confidence_summary <- assignment_confidence %>%
  group_by(k) %>%
  summarise(
    median_margin = median(margin_confidence),
    p10_margin = quantile(margin_confidence, 0.10),
    low_margin_share = mean(low_margin),
    mean_minimum_distance = mean(minimum_distance),
    p95_minimum_distance = quantile(minimum_distance, 0.95),
    .groups = "drop"
  )
confidence_cluster_summary <- assignment_confidence %>%
  group_by(k, cluster) %>%
  summarise(
    n = n(),
    median_margin = median(margin_confidence),
    p10_margin = quantile(margin_confidence, 0.10),
    low_margin_share = mean(low_margin),
    mean_minimum_distance = mean(minimum_distance),
    .groups = "drop"
  )

signal_specification <- tribble(
  ~signal, ~raw_feature, ~pressure_feature, ~signal_role,
  "eviction", "eviction_cases_latest_12mo", "eviction_pressure_index", "direct_proxy",
  "demolition", "demo_total_demolition_count", "demolition_pressure_index", "direct_proxy",
  "selected_311", "sr_311_smoke_signal_latest_12mo", "sr_311_pressure_index", "smoke_signal",
  "ownership_change", "ownership_change_recent_count", "ownership_pressure_index", "smoke_signal",
  "amenity_opening", "amenity_recent_opening_events", "amenity_change_index", "smoke_signal"
)

binary_separation <- function(value, cluster) {
  observed <- is.finite(value)
  value <- value[observed]
  cluster <- cluster[observed]
  if (length(value) == 0L || var(value) == 0) return(NA_real_)
  overall <- mean(value)
  grouped <- tibble(value = value, cluster = cluster) %>%
    group_by(cluster) %>%
    summarise(n = n(), mean = mean(value), .groups = "drop")
  sum(grouped$n * (grouped$mean - overall)^2) /
    sum((value - overall)^2)
}

binary_entropy <- function(probability) {
  ifelse(
    probability <= 0 | probability >= 1,
    0,
    -(probability * log2(probability) +
        (1 - probability) * log2(1 - probability))
  )
}

signal_cluster_rows <- list()
signal_summary_rows <- list()
signal_index <- 1L

for (k in k_values) {
  cluster <- full_labels[[as.character(k)]]
  for (signal_row in seq_len(nrow(signal_specification))) {
    signal_name <- signal_specification$signal[[signal_row]]
    raw_feature <- signal_specification$raw_feature[[signal_row]]
    pressure_feature <- signal_specification$pressure_feature[[signal_row]]
    raw_value <- as.numeric(analysis_data[[raw_feature]])
    pressure_value <- as.numeric(analysis_data[[pressure_feature]])
    observed <- is.finite(raw_value)

    by_cluster <- tibble(
      cluster = cluster,
      raw_value = raw_value,
      pressure_value = pressure_value,
      observed = observed
    ) %>%
      group_by(cluster) %>%
      summarise(
        n = n(),
        observed_n = sum(observed),
        positive_n = sum(raw_value > 0, na.rm = TRUE),
        zero_n = sum(raw_value == 0, na.rm = TRUE),
        positive_share = if_else(observed_n > 0, positive_n / observed_n, NA_real_),
        zero_share = if_else(observed_n > 0, zero_n / observed_n, NA_real_),
        pressure_mean = mean(pressure_value),
        pressure_median = median(pressure_value),
        pressure_zero_share = mean(pressure_value == 0),
        .groups = "drop"
      ) %>%
      mutate(
        k = k,
        signal = signal_name,
        signal_role = signal_specification$signal_role[[signal_row]],
        .before = 1
      )
    signal_cluster_rows[[signal_index]] <- by_cluster

    high_pressure_cluster <- by_cluster$cluster[[
      which.max(by_cluster$pressure_mean)
    ]]
    high_pressure_row <- by_cluster %>%
      filter(cluster == high_pressure_cluster)
    weighted_entropy <- with(
      by_cluster,
      weighted.mean(binary_entropy(positive_share), observed_n, na.rm = TRUE)
    )

    signal_summary_rows[[signal_index]] <- tibble(
      k = k,
      signal = signal_name,
      signal_role = signal_specification$signal_role[[signal_row]],
      observed_n = sum(observed),
      overall_positive_share = mean(raw_value[observed] > 0),
      presence_separation_r2 = binary_separation(
        as.numeric(raw_value > 0),
        cluster
      ),
      weighted_presence_entropy = weighted_entropy,
      mixed_cluster_count = sum(
        by_cluster$positive_share >= 0.25 &
          by_cluster$positive_share <= 0.75,
        na.rm = TRUE
      ),
      high_pressure_cluster = high_pressure_cluster,
      high_pressure_cluster_n = high_pressure_row$n,
      high_pressure_cluster_positive_share = high_pressure_row$positive_share,
      high_pressure_cluster_zero_share = high_pressure_row$zero_share,
      high_pressure_cluster_pressure_zero_share =
        high_pressure_row$pressure_zero_share,
      high_pressure_cluster_mean = high_pressure_row$pressure_mean,
      high_pressure_cluster_median = high_pressure_row$pressure_median
    )
    signal_index <- signal_index + 1L
  }
}

signal_cluster_prevalence <- bind_rows(signal_cluster_rows)
signal_separation_summary <- bind_rows(signal_summary_rows)

################################################################################
# Spatial contiguity of the full-sample solutions
################################################################################

analysis_index <- setNames(seq_len(nrow(analysis_data)), analysis_data$h3_index)
neighbor_disks <- get_disk(analysis_data$h3_index, ring_size = 1L, simple = TRUE)
edge_rows <- lapply(seq_along(neighbor_disks), function(from_index) {
  to_index <- unname(analysis_index[neighbor_disks[[from_index]]])
  to_index <- to_index[!is.na(to_index) & to_index > from_index]
  if (length(to_index) == 0L) return(NULL)
  tibble(from = from_index, to = as.integer(to_index))
})
edges <- bind_rows(edge_rows)
adjacency <- vector("list", nrow(analysis_data))
for (edge_index in seq_len(nrow(edges))) {
  from <- edges$from[[edge_index]]
  to <- edges$to[[edge_index]]
  adjacency[[from]] <- c(adjacency[[from]], to)
  adjacency[[to]] <- c(adjacency[[to]], from)
}

cluster_component_sizes <- function(member, adjacency) {
  member_indices <- which(member)
  visited <- rep(FALSE, length(member))
  sizes <- integer()
  for (start in member_indices) {
    if (visited[[start]]) next
    queue <- start
    visited[[start]] <- TRUE
    size <- 0L
    while (length(queue) > 0L) {
      current <- queue[[1L]]
      queue <- queue[-1L]
      size <- size + 1L
      candidates <- adjacency[[current]]
      candidates <- candidates[
        member[candidates] & !visited[candidates]
      ]
      if (length(candidates) > 0L) {
        visited[candidates] <- TRUE
        queue <- c(queue, candidates)
      }
    }
    sizes <- c(sizes, size)
  }
  sizes
}

spatial_behavior_rows <- list()
spatial_behavior_index <- 1L
for (k in k_values) {
  cluster <- full_labels[[as.character(k)]]
  overall_same <- cluster[edges$from] == cluster[edges$to]
  for (cluster_id in seq_len(k)) {
    member <- cluster == cluster_id
    internal_edges <- sum(member[edges$from] & member[edges$to])
    boundary_edges <- sum(xor(member[edges$from], member[edges$to]))
    component_sizes <- cluster_component_sizes(member, adjacency)
    spatial_behavior_rows[[spatial_behavior_index]] <- tibble(
      k = k,
      cluster = cluster_id,
      n = sum(member),
      internal_neighbor_edges = internal_edges,
      boundary_neighbor_edges = boundary_edges,
      same_cluster_neighbor_share = if_else(
        2 * internal_edges + boundary_edges > 0,
        2 * internal_edges / (2 * internal_edges + boundary_edges),
        NA_real_
      ),
      connected_components = length(component_sizes),
      largest_component_n = max(component_sizes),
      largest_component_share = max(component_sizes) / sum(member),
      overall_same_cluster_edge_share = mean(overall_same)
    )
    spatial_behavior_index <- spatial_behavior_index + 1L
  }
}
spatial_behavior <- bind_rows(spatial_behavior_rows)
spatial_behavior_summary <- spatial_behavior %>%
  group_by(k) %>%
  summarise(
    overall_same_cluster_edge_share = first(overall_same_cluster_edge_share),
    weighted_largest_component_share = weighted.mean(
      largest_component_share,
      n
    ),
    minimum_largest_component_share = min(largest_component_share),
    total_connected_components = sum(connected_components),
    .groups = "drop"
  )

################################################################################
# Candidate scorecard and k = 6 to k = 7 split/merge audit
################################################################################

stability_wide <- stability_summary %>%
  select(
    scheme,
    k,
    median_adjusted_rand,
    p10_adjusted_rand,
    worst_adjusted_rand,
    median_matched_agreement,
    holdout_cluster_collapse_rate
  ) %>%
  pivot_wider(
    names_from = scheme,
    values_from = c(
      median_adjusted_rand,
      p10_adjusted_rand,
      worst_adjusted_rand,
      median_matched_agreement,
      holdout_cluster_collapse_rate
    ),
    names_glue = "{scheme}_{.value}"
  )

eviction_selection <- signal_separation_summary %>%
  filter(signal == "eviction") %>%
  select(
    k,
    eviction_presence_separation_r2 = presence_separation_r2,
    eviction_weighted_presence_entropy = weighted_presence_entropy,
    eviction_mixed_cluster_count = mixed_cluster_count,
    eviction_high_pressure_cluster = high_pressure_cluster,
    eviction_high_pressure_zero_share = high_pressure_cluster_zero_share,
    eviction_high_pressure_pressure_zero_share =
      high_pressure_cluster_pressure_zero_share,
    eviction_high_pressure_mean = high_pressure_cluster_mean
  )

selection_scorecard <- method_metrics %>%
  filter(specification == !!specification, k %in% k_values) %>%
  select(-specification) %>%
  left_join(
    gap_statistics %>%
      filter(specification == !!specification) %>%
      select(k, gap, gap_se),
    by = "k"
  ) %>%
  left_join(
    random_subsample_stability %>%
      filter(specification == !!specification) %>%
      select(
        k,
        existing_random_subsample_mean_ari = mean_adjusted_rand,
        existing_random_subsample_p10_ari = p10_adjusted_rand
      ),
    by = "k"
  ) %>%
  left_join(stability_wide, by = "k") %>%
  left_join(confidence_summary, by = "k") %>%
  left_join(eviction_selection, by = "k") %>%
  left_join(spatial_behavior_summary, by = "k") %>%
  mutate(
    focused_candidate = k %in% focus_k,
    configured_selected = k == EWS_CONFIG$amenity_cluster_k
  )

selection_decision <- selection_config %>%
  mutate(decision_date = as.Date(decision_date)) %>%
  left_join(
    selection_scorecard %>%
      filter(configured_selected) %>%
      select(
        selected_k = k,
        avg_silhouette,
        min_cluster_n,
        max_cluster_n,
        existing_random_subsample_mean_ari,
        random_hex_median_adjusted_rand,
        random_hex_p10_adjusted_rand,
        h3_parent_r8_median_adjusted_rand,
        h3_parent_r8_p10_adjusted_rand,
        h3_parent_r7_median_adjusted_rand,
        h3_parent_r7_p10_adjusted_rand,
        median_margin,
        low_margin_share,
        eviction_presence_separation_r2,
        eviction_high_pressure_zero_share
      ),
    by = "selected_k"
  )

if (all(c(6L, 7L) %in% k_values)) {
  k6_assignment <- tibble(
    hex_id = analysis_data$hex_id,
    k6_cluster = full_labels[["6"]]
  )
  k7_assignment <- tibble(
    hex_id = analysis_data$hex_id,
    k7_cluster = full_labels[["7"]]
  )
  k6_k7_crosswalk <- k6_assignment %>%
    inner_join(k7_assignment, by = "hex_id") %>%
    count(k6_cluster, k7_cluster, name = "hexes") %>%
    group_by(k6_cluster) %>%
    mutate(k6_share = hexes / sum(hexes)) %>%
    ungroup() %>%
    group_by(k7_cluster) %>%
    mutate(k7_share = hexes / sum(hexes)) %>%
    ungroup()
} else {
  k6_k7_crosswalk <- tibble(
    k6_cluster = integer(),
    k7_cluster = integer(),
    hexes = integer(),
    k6_share = double(),
    k7_share = double()
  )
}

################################################################################
# Write machine-readable outputs
################################################################################

write_csv(
  scheme_summary,
  file.path(PART1_DIR, "cluster_selection_block_schemes.csv")
)
write_csv(
  stability_replicates,
  file.path(PART1_DIR, "cluster_selection_stability_replicates.csv")
)
write_csv(
  stability_summary,
  file.path(PART1_DIR, "cluster_selection_stability_summary.csv")
)
write_csv(
  stability_cluster_summary,
  file.path(PART1_DIR, "cluster_selection_stability_by_cluster.csv")
)
write_csv(
  assignment_confidence,
  file.path(PART1_DIR, "cluster_selection_assignment_confidence.csv")
)
write_csv(
  confidence_cluster_summary,
  file.path(PART1_DIR, "cluster_selection_confidence_by_cluster.csv")
)
write_csv(
  cluster_profiles,
  file.path(PART1_DIR, "cluster_selection_profiles.csv")
)
write_csv(
  signal_cluster_prevalence,
  file.path(PART1_DIR, "cluster_selection_signal_prevalence.csv")
)
write_csv(
  signal_separation_summary,
  file.path(PART1_DIR, "cluster_selection_signal_separation.csv")
)
write_csv(
  spatial_behavior,
  file.path(PART1_DIR, "cluster_selection_spatial_behavior.csv")
)
write_csv(
  selection_scorecard,
  file.path(PART1_DIR, "cluster_selection_scorecard.csv")
)
write_csv(
  k6_k7_crosswalk,
  file.path(PART1_DIR, "cluster_selection_k6_k7_crosswalk.csv")
)
write_csv(
  selection_decision,
  file.path(PART1_DIR, "cluster_selection_decision.csv")
)

audit_results <- list(
  created_at = Sys.time(),
  seed = seed,
  specification = specification,
  cluster_features = cluster_features,
  candidate_k = k_values,
  focus_k = focus_k,
  replicates = spatial_replicates,
  holdout_share = holdout_share,
  holdout_nstart = holdout_nstart,
  spatial_parent_resolutions = spatial_parent_resolutions,
  scheme_summary = scheme_summary,
  stability_replicates = stability_replicates,
  stability_summary = stability_summary,
  stability_cluster_summary = stability_cluster_summary,
  assignment_confidence = assignment_confidence,
  confidence_summary = confidence_summary,
  confidence_cluster_summary = confidence_cluster_summary,
  cluster_profiles = cluster_profiles,
  signal_cluster_prevalence = signal_cluster_prevalence,
  signal_separation_summary = signal_separation_summary,
  spatial_behavior = spatial_behavior,
  selection_scorecard = selection_scorecard,
  selection_decision = selection_decision,
  k6_k7_crosswalk = k6_k7_crosswalk
)
saveRDS(
  audit_results,
  file.path(PART1_DIR, "cluster_selection_audit.rds")
)

################################################################################
# Figures
################################################################################

scheme_labels <- c(
  random_hex = "Random hex holdout",
  h3_parent_r8 = "H3 resolution 8 blocks",
  h3_parent_r7 = "H3 resolution 7 blocks"
)
stability_plot_data <- stability_summary %>%
  filter(k %in% k_values) %>%
  mutate(
    scheme_label = factor(
      coalesce(scheme_labels[scheme], scheme),
      levels = unname(scheme_labels[names(scheme_labels) %in% scheme])
    )
  )

p_stability <- ggplot(
  stability_plot_data,
  aes(x = k, y = median_adjusted_rand, color = scheme_label)
) +
  geom_ribbon(
    aes(
      ymin = p10_adjusted_rand,
      ymax = median_adjusted_rand,
      fill = scheme_label
    ),
    alpha = 0.12,
    color = NA
  ) +
  geom_line(linewidth = 0.85) +
  geom_point(size = 1.8) +
  geom_vline(xintercept = c(6, 7), linetype = "dashed", color = "#555555") +
  scale_x_continuous(breaks = k_values) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    title = "Random and Spatially Blocked Cluster Stability",
    subtitle = paste0(
      spatial_replicates,
      " repeated 20% holdouts; ribbon spans the 10th percentile to median ARI"
    ),
    x = "Number of clusters (k)",
    y = "Held-out adjusted Rand index",
    color = NULL,
    fill = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )
ggsave(
  file.path(FIGURES_DIR, "03f_cluster_selection_stability.png"),
  p_stability,
  width = 10.5,
  height = 6.5,
  dpi = 300,
  bg = "white"
)

if (all(c(6L, 7L) %in% k_values)) {
  map_assignments <- bind_rows(
    tibble(hex_id = analysis_data$hex_id, k = 6L, cluster = full_labels[["6"]]),
    tibble(hex_id = analysis_data$hex_id, k = 7L, cluster = full_labels[["7"]])
  ) %>%
    mutate(
      cluster_label = factor(
        paste0("Cluster ", cluster),
        levels = paste0("Cluster ", seq_len(7))
      ),
      solution = factor(
        paste0("k = ", k),
        levels = c("k = 6", "k = 7")
      )
    )
  map_data <- features_sf %>%
    select(hex_id) %>%
    inner_join(map_assignments, by = "hex_id")
  map_palette <- setNames(
    grDevices::hcl.colors(7, palette = "Dark 3"),
    levels(map_assignments$cluster_label)
  )
  p_maps <- ggplot(map_data) +
    geom_sf(aes(fill = cluster_label), color = "white", linewidth = 0.03) +
    facet_wrap(~solution) +
    scale_fill_manual(values = map_palette, drop = FALSE, name = NULL) +
    coord_sf(datum = NA) +
    labs(
      title = "Candidate k = 6 and k = 7 Spatial Assignments",
      subtitle = "Numeric labels are solution-specific; use the crosswalk to interpret splits"
    ) +
    theme_void(base_size = 11) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
  ggsave(
    file.path(FIGURES_DIR, "03f_cluster_selection_k6_k7_maps.png"),
    p_maps,
    width = 12,
    height = 6.5,
    dpi = 300,
    bg = "white"
  )

  confidence_map_data <- features_sf %>%
    select(hex_id) %>%
    inner_join(
      assignment_confidence %>%
        filter(k %in% c(6L, 7L)) %>%
        mutate(
          solution = factor(
            paste0("k = ", k),
            levels = c("k = 6", "k = 7")
          )
        ),
      by = "hex_id"
    )
  p_confidence_maps <- ggplot(confidence_map_data) +
    geom_sf(aes(fill = margin_confidence), color = NA) +
    facet_wrap(~solution) +
    scale_fill_viridis_c(
      option = "C",
      limits = c(0, 1),
      name = "Assignment\nmargin"
    ) +
    coord_sf(datum = NA) +
    labs(
      title = "Geography of Assignment Confidence",
      subtitle = "Larger margins indicate clearer separation from the second-nearest centroid"
    ) +
    theme_void(base_size = 11) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
  ggsave(
    file.path(FIGURES_DIR, "03f_cluster_selection_k6_k7_confidence.png"),
    p_confidence_maps,
    width = 12,
    height = 6.5,
    dpi = 300,
    bg = "white"
  )

  feature_labels <- c(
    rent_pressure_citywide_index = "Rent pressure",
    demographic_vulnerability_index = "Demographic vulnerability",
    demolition_pressure_index = "Demolition pressure",
    eviction_pressure_index = "Eviction pressure",
    sr_311_pressure_index = "311 pressure",
    ownership_pressure_index = "Corporate ownership pressure",
    amenity_change_index = "Amenity pressure"
  )
  feature_means <- colMeans(raw_matrix)
  feature_sds <- apply(raw_matrix, 2, sd)
  profile_plot_data <- cluster_profiles %>%
    filter(k %in% c(6L, 7L)) %>%
    select(k, cluster, ends_with("_mean")) %>%
    pivot_longer(
      cols = -c(k, cluster),
      names_to = "feature",
      values_to = "feature_mean"
    ) %>%
    mutate(
      feature = sub("_mean$", "", feature),
      standardized_mean =
        (feature_mean - feature_means[feature]) / feature_sds[feature],
      feature_label = factor(
        feature_labels[feature],
        levels = rev(unname(feature_labels))
      ),
      cluster_label = factor(
        paste0("C", cluster),
        levels = paste0("C", seq_len(7))
      ),
      solution = factor(paste0("k = ", k), levels = c("k = 6", "k = 7"))
    )
  p_profiles <- ggplot(
    profile_plot_data,
    aes(x = cluster_label, y = feature_label, fill = standardized_mean)
  ) +
    geom_tile(color = "white", linewidth = 0.4) +
    facet_grid(. ~ solution, scales = "free_x", space = "free_x") +
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      name = "Mean z-score"
    ) +
    labs(
      title = "Candidate Cluster Profiles",
      subtitle = "Domain means standardized across the common analysis sample",
      x = "Cluster",
      y = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )
  ggsave(
    file.path(FIGURES_DIR, "03f_cluster_selection_k6_k7_profiles.png"),
    p_profiles,
    width = 11.5,
    height = 6.5,
    dpi = 300,
    bg = "white"
  )

  p_crosswalk <- ggplot(
    k6_k7_crosswalk,
    aes(x = factor(k7_cluster), y = factor(k6_cluster), fill = k6_share)
  ) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(
      aes(label = paste0(hexes, "\n", scales::percent(k6_share, accuracy = 1))),
      size = 3
    ) +
    scale_fill_gradient(
      low = "white",
      high = "#2166AC",
      labels = scales::percent,
      name = "Share of k = 6 cluster"
    ) +
    labs(
      title = "How k = 6 Clusters Split at k = 7",
      x = "k = 7 cluster",
      y = "k = 6 cluster"
    ) +
    theme_minimal(base_size = 11) +
    theme(panel.grid = element_blank(), legend.position = "bottom")
  ggsave(
    file.path(FIGURES_DIR, "03f_cluster_selection_k6_k7_crosswalk.png"),
    p_crosswalk,
    width = 8.5,
    height = 6.5,
    dpi = 300,
    bg = "white"
  )

  signal_labels <- c(
    eviction = "Eviction filing",
    demolition = "Residential demolition",
    selected_311 = "Selected 311 request",
    ownership_change = "Ownership change",
    amenity_opening = "Amenity opening"
  )
  prevalence_plot_data <- signal_cluster_prevalence %>%
    filter(k %in% c(6L, 7L)) %>%
    mutate(
      signal_label = factor(
        signal_labels[signal],
        levels = unname(signal_labels)
      ),
      cluster_label = factor(
        paste0("C", cluster),
        levels = paste0("C", seq_len(7))
      ),
      solution = factor(paste0("k = ", k), levels = c("k = 6", "k = 7"))
    )
  p_prevalence <- ggplot(
    prevalence_plot_data,
    aes(x = cluster_label, y = positive_share, fill = signal_role)
  ) +
    geom_col(width = 0.78) +
    facet_grid(signal_label ~ solution, scales = "free_x", space = "free_x") +
    scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    scale_fill_manual(
      values = c(direct_proxy = "#B2182B", smoke_signal = "#2166AC"),
      guide = "none"
    ) +
    labs(
      title = "Observed Signal Prevalence Within Candidate Clusters",
      subtitle = "A zero means no observed event in the feature's current analysis window",
      x = "Cluster",
      y = "Hexes with a positive event count"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      panel.grid.minor = element_blank(),
      strip.text = element_text(face = "bold")
    )
  ggsave(
    file.path(FIGURES_DIR, "03f_cluster_selection_signal_prevalence.png"),
    p_prevalence,
    width = 11.5,
    height = 9,
    dpi = 300,
    bg = "white"
  )
}

print_header("CLUSTER SELECTION AUDIT COMPLETE")
print(
  selection_scorecard %>%
    filter(k %in% focus_k) %>%
    select(
      k,
      avg_silhouette,
      min_cluster_n,
      random_hex_median_adjusted_rand,
      h3_parent_r8_median_adjusted_rand,
      h3_parent_r7_median_adjusted_rand,
      eviction_high_pressure_zero_share,
      median_margin,
      low_margin_share
    )
)
