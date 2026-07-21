################################################################################
# 03c - Balanced Cluster Method Sensitivity Analysis
################################################################################
#
# Compares three clustering specifications without replacing the canonical
# output from 03b_cluster_analysis.R:
#   1. Balanced k-means with one measure per conceptual domain
#   2. K-means with equal total weight for proxies, vulnerability, and smoke
#   3. K-means using one principal component per conceptual domain
#   4. Gower-distance PAM using the balanced domain inputs
#
# Cluster counts from 2 through CLUSTER_MAX_K are evaluated with silhouette
# scores and cluster-size balance. The Tibshirani gap statistic is also computed
# for the two Euclidean k-means specifications.
#
# Optional environment variables:
#   CLUSTER_MAX_K           default: 12
#   CLUSTER_GAP_BOOTSTRAPS  default: 100
#
# Outputs:
#   output/cluster_method_comparison.rds
#   output/cluster_method_metrics.csv
#   output/cluster_number_recommendations.csv
#   output/cluster_gap_statistics.csv
#   output/cluster_method_assignments.csv
#   output/cluster_selected_method_profiles.csv
#   output/cluster_domain_pca_loadings.csv
#   output/cluster_domain_pca_summary.csv
#   output/cluster_component_imputation_summary.csv
#   figures/03c_cluster_selection_diagnostics.png
#   figures/03c_cluster_balance_diagnostics.png
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))

suppressPackageStartupMessages({
  library(cluster)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(sf)
  library(tidyr)
})

print_header("03c - BALANCED CLUSTER METHOD SENSITIVITY ANALYSIS")

OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
set.seed(42)

max_k <- as.integer(Sys.getenv("CLUSTER_MAX_K", unset = "12"))
gap_bootstraps <- as.integer(Sys.getenv("CLUSTER_GAP_BOOTSTRAPS", unset = "100"))

if (is.na(max_k) || max_k < 3) {
  stop("CLUSTER_MAX_K must be an integer of at least 3.", call. = FALSE)
}
if (is.na(gap_bootstraps) || gap_bootstraps < 10) {
  stop("CLUSTER_GAP_BOOTSTRAPS must be an integer of at least 10.", call. = FALSE)
}

k_values <- 2:max_k

################################################################################
# Step 1: Create a shared, balanced analysis sample
################################################################################

print_progress("Loading engineered features...")
hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  "engineered features"
)

balanced_vars <- c(
  "rent_pressure_citywide_index",
  "demographic_vulnerability_index",
  "demolition_pressure_index",
  "eviction_pressure_index",
  "sr_311_pressure_index",
  "ownership_pressure_index"
)

profile_vars <- c(
  balanced_vars,
  "costar_present",
  "costar_rent_pressure_index",
  "residential_units_per_km2",
  "pct_poc",
  "demographic_vulnerability_equity_index"
)

domain_family <- c(
  rent_pressure_citywide_index = "displacement_proxy",
  demolition_pressure_index = "displacement_proxy",
  eviction_pressure_index = "displacement_proxy",
  demographic_vulnerability_index = "vulnerability",
  sr_311_pressure_index = "smoke_signal",
  ownership_pressure_index = "smoke_signal"
)

domain_components <- list(
  rent = c(
    "acs_rent_current_real",
    "acs_rent_growth_recent_for_clustering",
    "acs_rent_acceleration_for_clustering"
  ),
  demographic = c(
    "rent_burden_proxy",
    "pct_rent_burden_30plus",
    "median_income",
    "poverty_rate",
    "pct_renter",
    "pct_college"
  ),
  demolition = c(
    "demo_density",
    "demo_recent_density"
  ),
  eviction = c(
    "eviction_cases_per_100_units",
    "eviction_latest_12mo_per_100_units",
    "eviction_cases_total_density",
    "eviction_cases_latest_12mo_density"
  ),
  sr_311 = c(
    "sr_311_latest_12mo_per_100_units",
    "sr_311_smoke_signal_latest_12mo_per_100_units",
    "sr_311_latest_12mo_density",
    "sr_311_smoke_signal_latest_12mo_density",
    "sr_311_smoke_signal_share"
  ),
  ownership = c(
    "pct_corporate_units",
    "pct_corporate_parcels",
    "pct_financialized_owner_parcels",
    "corporate_owned_units_per_km2"
  )
)

domain_anchors <- c(
  rent = "rent_pressure_citywide_index",
  demographic = "demographic_vulnerability_index",
  demolition = "demolition_pressure_index",
  eviction = "eviction_pressure_index",
  sr_311 = "sr_311_pressure_index",
  ownership = "ownership_pressure_index"
)

required_vars <- unique(c(profile_vars, unlist(domain_components, use.names = FALSE)))
missing_vars <- setdiff(required_vars, names(hex_features))
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
  select(hex_id, all_of(required_vars)) %>%
  filter(if_all(all_of(balanced_vars), ~!is.na(.x)))

if (nrow(analysis_data) < 20) {
  stop("Too few complete balanced observations for clustering.", call. = FALSE)
}

print_progress(
  paste0(
    "Shared analysis sample: ", nrow(analysis_data),
    " hexes with ", length(balanced_vars), " balanced inputs"
  )
)

################################################################################
# Step 2: Build domain principal components
################################################################################

median_impute <- function(x) {
  replacement <- median(x, na.rm = TRUE)
  if (!is.finite(replacement)) {
    stop("Cannot median-impute a component with no finite observations.", call. = FALSE)
  }
  replace(x, is.na(x), replacement)
}

component_vars <- unique(unlist(domain_components, use.names = FALSE))
imputation_summary <- tibble(
  feature = component_vars,
  missing_before = vapply(
    analysis_data[component_vars],
    function(x) sum(is.na(x)),
    integer(1)
  ),
  median_used = vapply(
    analysis_data[component_vars],
    function(x) median(x, na.rm = TRUE),
    numeric(1)
  )
)

analysis_data <- analysis_data %>%
  mutate(across(all_of(component_vars), median_impute))

domain_scores <- tibble(hex_id = analysis_data$hex_id)
pca_loadings <- list()
pca_summary <- list()
pca_models <- list()

for (domain in names(domain_components)) {
  vars <- domain_components[[domain]]
  component_matrix <- analysis_data %>%
    select(all_of(vars)) %>%
    scale() %>%
    as.matrix()

  if (any(!is.finite(component_matrix))) {
    stop("Non-finite scaled values in the ", domain, " PCA domain.", call. = FALSE)
  }

  pca_fit <- prcomp(component_matrix, center = FALSE, scale. = FALSE)
  pc1 <- pca_fit$x[, 1]
  anchor <- analysis_data[[domain_anchors[[domain]]]]
  anchor_correlation <- cor(pc1, anchor)

  if (is.finite(anchor_correlation) && anchor_correlation < 0) {
    pc1 <- -pc1
    pca_fit$x[, 1] <- -pca_fit$x[, 1]
    pca_fit$rotation[, 1] <- -pca_fit$rotation[, 1]
    anchor_correlation <- -anchor_correlation
  }

  score_name <- paste0(domain, "_pc1")
  domain_scores[[score_name]] <- pc1
  explained <- pca_fit$sdev[1]^2 / sum(pca_fit$sdev^2)

  pca_loadings[[domain]] <- tibble(
    domain = domain,
    feature = rownames(pca_fit$rotation),
    pc1_loading = pca_fit$rotation[, 1]
  )
  pca_summary[[domain]] <- tibble(
    domain = domain,
    n_components = length(vars),
    pc1_variance_explained = explained,
    anchor_feature = domain_anchors[[domain]],
    anchor_correlation = anchor_correlation
  )
  pca_models[[domain]] <- pca_fit
}

pca_loadings <- bind_rows(pca_loadings)
pca_summary <- bind_rows(pca_summary)

pca_feature_data <- analysis_data %>%
  select(hex_id) %>%
  left_join(domain_scores, by = "hex_id")

################################################################################
# Step 3: Evaluate k-means and Gower/PAM across cluster counts
################################################################################

scale_numeric_matrix <- function(data, vars) {
  matrix <- data %>%
    select(all_of(vars)) %>%
    scale() %>%
    as.matrix()

  if (any(!is.finite(matrix))) {
    stop("Clustering matrix contains non-finite scaled values.", call. = FALSE)
  }
  matrix
}

summarize_cluster_sizes <- function(cluster_ids) {
  sizes <- table(cluster_ids)
  tibble(
    min_cluster_n = min(as.integer(sizes)),
    max_cluster_n = max(as.integer(sizes)),
    max_cluster_share = max(as.integer(sizes)) / length(cluster_ids)
  )
}

evaluate_kmeans <- function(matrix, method_name) {
  distance <- dist(matrix)
  models <- list()
  metrics <- list()

  for (k in k_values) {
    set.seed(42)
    fit <- kmeans(matrix, centers = k, nstart = 50, iter.max = 100)
    silhouette_result <- silhouette(fit$cluster, distance)
    size_metrics <- summarize_cluster_sizes(fit$cluster)

    models[[as.character(k)]] <- fit
    metrics[[as.character(k)]] <- tibble(
      method = method_name,
      k = k,
      avg_silhouette = mean(silhouette_result[, "sil_width"]),
      between_ss_share = fit$betweenss / fit$totss,
      min_cluster_n = size_metrics$min_cluster_n,
      max_cluster_n = size_metrics$max_cluster_n,
      max_cluster_share = size_metrics$max_cluster_share
    )
  }

  list(metrics = bind_rows(metrics), models = models)
}

balanced_matrix <- scale_numeric_matrix(analysis_data, balanced_vars)
family_counts <- table(domain_family[balanced_vars])
family_weights <- vapply(
  balanced_vars,
  function(feature) {
    family <- domain_family[[feature]]
    (1 / length(family_counts)) / unname(family_counts[[family]])
  },
  numeric(1)
)
names(family_weights) <- balanced_vars
equal_family_matrix <- sweep(
  balanced_matrix,
  2,
  sqrt(family_weights[colnames(balanced_matrix)]),
  `*`
)
pca_vars <- setdiff(names(pca_feature_data), "hex_id")
pca_matrix <- scale_numeric_matrix(pca_feature_data, pca_vars)

print_progress("Evaluating balanced k-means for k = 2 through k = max...")
balanced_kmeans_eval <- evaluate_kmeans(balanced_matrix, "balanced_kmeans")

print_progress("Evaluating equal-family k-means for k = 2 through k = max...")
equal_family_kmeans_eval <- evaluate_kmeans(
  equal_family_matrix,
  "equal_family_kmeans"
)

print_progress("Evaluating domain-PCA k-means for k = 2 through k = max...")
pca_kmeans_eval <- evaluate_kmeans(pca_matrix, "domain_pca_kmeans")

print_progress("Computing Gower distance for balanced domain inputs...")
gower_data <- analysis_data %>%
  select(all_of(balanced_vars))

gower_distance <- daisy(
  gower_data,
  metric = "gower"
)

print_progress("Evaluating Gower/PAM for k = 2 through k = max...")
pam_models <- list()
pam_metrics <- list()

for (k in k_values) {
  pam_fit <- pam(
    gower_distance,
    k = k,
    diss = TRUE,
    pamonce = 5,
    keep.diss = FALSE,
    keep.data = FALSE
  )
  silhouette_result <- silhouette(pam_fit$clustering, gower_distance)
  size_metrics <- summarize_cluster_sizes(pam_fit$clustering)

  pam_models[[as.character(k)]] <- pam_fit
  pam_metrics[[as.character(k)]] <- tibble(
    method = "balanced_gower_pam",
    k = k,
    avg_silhouette = mean(silhouette_result[, "sil_width"]),
    between_ss_share = NA_real_,
    min_cluster_n = size_metrics$min_cluster_n,
    max_cluster_n = size_metrics$max_cluster_n,
    max_cluster_share = size_metrics$max_cluster_share
  )
}

method_metrics <- bind_rows(
  balanced_kmeans_eval$metrics,
  equal_family_kmeans_eval$metrics,
  pca_kmeans_eval$metrics,
  bind_rows(pam_metrics)
)

################################################################################
# Step 4: Calculate the gap statistic for Euclidean specifications
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

calculate_gap <- function(matrix, method_name) {
  print_progress(
    paste0(
      "Calculating gap statistic for ", method_name,
      " with B = ", gap_bootstraps, "..."
    )
  )
  set.seed(42)
  result <- clusGap(
    matrix,
    FUNcluster = gap_kmeans,
    K.max = max_k,
    B = gap_bootstraps,
    d.power = 2,
    spaceH0 = "scaledPCA",
    verbose = TRUE
  )

  table <- as.data.frame(result$Tab) %>%
    mutate(
      method = method_name,
      k = seq_len(n()),
      .before = 1
    ) %>%
    rename(
      log_within = logW,
      expected_log_within = E.logW,
      gap = gap,
      gap_se = SE.sim
    )

  tibs_k <- maxSE(
    f = result$Tab[, "gap"],
    SE.f = result$Tab[, "SE.sim"],
    method = "Tibs2001SEmax"
  )

  list(
    result = result,
    table = table,
    tibs_k = tibs_k,
    global_k = which.max(result$Tab[, "gap"])
  )
}

balanced_gap <- calculate_gap(balanced_matrix, "balanced_kmeans")
equal_family_gap <- calculate_gap(equal_family_matrix, "equal_family_kmeans")
pca_gap <- calculate_gap(pca_matrix, "domain_pca_kmeans")
gap_statistics <- bind_rows(
  balanced_gap$table,
  equal_family_gap$table,
  pca_gap$table
)

################################################################################
# Step 5: Compare recommendations and save selected assignments
################################################################################

silhouette_recommendations <- method_metrics %>%
  group_by(method) %>%
  slice_max(avg_silhouette, n = 1, with_ties = FALSE) %>%
  transmute(
    method,
    diagnostic = "silhouette_global_max",
    recommended_k = k,
    diagnostic_value = avg_silhouette
  ) %>%
  ungroup()

gap_recommendations <- bind_rows(
  tibble(
    method = "balanced_kmeans",
    diagnostic = c("gap_Tibs2001SEmax", "gap_global_max"),
    recommended_k = c(balanced_gap$tibs_k, balanced_gap$global_k),
    diagnostic_value = c(
      balanced_gap$table$gap[balanced_gap$tibs_k],
      balanced_gap$table$gap[balanced_gap$global_k]
    )
  ),
  tibble(
    method = "equal_family_kmeans",
    diagnostic = c("gap_Tibs2001SEmax", "gap_global_max"),
    recommended_k = c(equal_family_gap$tibs_k, equal_family_gap$global_k),
    diagnostic_value = c(
      equal_family_gap$table$gap[equal_family_gap$tibs_k],
      equal_family_gap$table$gap[equal_family_gap$global_k]
    )
  ),
  tibble(
    method = "domain_pca_kmeans",
    diagnostic = c("gap_Tibs2001SEmax", "gap_global_max"),
    recommended_k = c(pca_gap$tibs_k, pca_gap$global_k),
    diagnostic_value = c(
      pca_gap$table$gap[pca_gap$tibs_k],
      pca_gap$table$gap[pca_gap$global_k]
    )
  )
)

recommendations <- bind_rows(silhouette_recommendations, gap_recommendations) %>%
  mutate(at_search_boundary = recommended_k == max_k)

best_k <- silhouette_recommendations %>%
  select(method, recommended_k) %>%
  tibble::deframe()

cluster_assignments <- tibble(
  hex_id = analysis_data$hex_id,
  balanced_kmeans = balanced_kmeans_eval$models[[
    as.character(best_k[["balanced_kmeans"]])
  ]]$cluster,
  equal_family_kmeans = equal_family_kmeans_eval$models[[
    as.character(best_k[["equal_family_kmeans"]])
  ]]$cluster,
  domain_pca_kmeans = pca_kmeans_eval$models[[
    as.character(best_k[["domain_pca_kmeans"]])
  ]]$cluster,
  balanced_gower_pam = pam_models[[
    as.character(best_k[["balanced_gower_pam"]])
  ]]$clustering,
  balanced_kmeans_gap = balanced_kmeans_eval$models[[
    as.character(balanced_gap$tibs_k)
  ]]$cluster,
  equal_family_kmeans_gap = equal_family_kmeans_eval$models[[
    as.character(equal_family_gap$tibs_k)
  ]]$cluster,
  domain_pca_kmeans_gap = pca_kmeans_eval$models[[
    as.character(pca_gap$tibs_k)
  ]]$cluster
)

selected_method_profiles <- bind_rows(
  tibble(
    hex_id = analysis_data$hex_id,
    method = "equal_family_kmeans",
    selected_k = best_k[["equal_family_kmeans"]],
    cluster = cluster_assignments$equal_family_kmeans
  ),
  tibble(
    hex_id = analysis_data$hex_id,
    method = "balanced_kmeans",
    selected_k = best_k[["balanced_kmeans"]],
    cluster = cluster_assignments$balanced_kmeans
  ),
  tibble(
    hex_id = analysis_data$hex_id,
    method = "domain_pca_kmeans",
    selected_k = best_k[["domain_pca_kmeans"]],
    cluster = cluster_assignments$domain_pca_kmeans
  ),
  tibble(
    hex_id = analysis_data$hex_id,
    method = "balanced_gower_pam",
    selected_k = best_k[["balanced_gower_pam"]],
    cluster = cluster_assignments$balanced_gower_pam
  )
) %>%
  left_join(
    analysis_data %>% select(hex_id, all_of(profile_vars)),
    by = "hex_id"
  ) %>%
  group_by(method, selected_k, cluster) %>%
  summarise(
    n = n(),
    across(all_of(profile_vars), ~mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

################################################################################
# Step 6: Save diagnostics and plots
################################################################################

write_csv(method_metrics, file.path(OUTPUT_DIR, "cluster_method_metrics.csv"))
write_csv(recommendations, file.path(OUTPUT_DIR, "cluster_number_recommendations.csv"))
write_csv(gap_statistics, file.path(OUTPUT_DIR, "cluster_gap_statistics.csv"))
write_csv(cluster_assignments, file.path(OUTPUT_DIR, "cluster_method_assignments.csv"))
write_csv(
  selected_method_profiles,
  file.path(OUTPUT_DIR, "cluster_selected_method_profiles.csv")
)
write_csv(pca_loadings, file.path(OUTPUT_DIR, "cluster_domain_pca_loadings.csv"))
write_csv(pca_summary, file.path(OUTPUT_DIR, "cluster_domain_pca_summary.csv"))
write_csv(
  imputation_summary,
  file.path(OUTPUT_DIR, "cluster_component_imputation_summary.csv")
)

comparison_results <- list(
  created_at = Sys.time(),
  seed = 42,
  n_observations = nrow(analysis_data),
  max_k = max_k,
  gap_bootstraps = gap_bootstraps,
  balanced_vars = balanced_vars,
  domain_family = domain_family,
  family_weights = family_weights,
  domain_components = domain_components,
  domain_pca_models = pca_models,
  domain_pca_summary = pca_summary,
  method_metrics = method_metrics,
  gap_statistics = gap_statistics,
  recommendations = recommendations,
  cluster_assignments = cluster_assignments,
  selected_method_profiles = selected_method_profiles,
  selected_models = list(
    balanced_kmeans = balanced_kmeans_eval$models[[
      as.character(best_k[["balanced_kmeans"]])
    ]],
    equal_family_kmeans = equal_family_kmeans_eval$models[[
      as.character(best_k[["equal_family_kmeans"]])
    ]],
    domain_pca_kmeans = pca_kmeans_eval$models[[
      as.character(best_k[["domain_pca_kmeans"]])
    ]],
    balanced_gower_pam = pam_models[[
      as.character(best_k[["balanced_gower_pam"]])
    ]]
  ),
  gap_results = list(
    balanced_kmeans = balanced_gap$result,
    equal_family_kmeans = equal_family_gap$result,
    domain_pca_kmeans = pca_gap$result
  )
)

save_output(
  comparison_results,
  file.path(OUTPUT_DIR, "cluster_method_comparison.rds"),
  "cluster method comparison"
)

selection_plot_data <- bind_rows(
  method_metrics %>%
    transmute(
      method,
      k,
      diagnostic = "Average silhouette",
      value = avg_silhouette,
      standard_error = NA_real_
    ),
  gap_statistics %>%
    filter(k >= 2) %>%
    transmute(
      method,
      k,
      diagnostic = "Gap statistic",
      value = gap,
      standard_error = gap_se
    )
)

p_selection <- ggplot(
  selection_plot_data,
  aes(x = k, y = value, color = method, group = method)
) +
  geom_errorbar(
    aes(ymin = value - standard_error, ymax = value + standard_error),
    width = 0.12,
    na.rm = TRUE,
    alpha = 0.55
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  facet_wrap(~diagnostic, scales = "free_y", ncol = 1) +
  scale_x_continuous(breaks = k_values) +
  labs(
    title = "Cluster Count Diagnostics Across Balanced Methods",
    subtitle = paste0(
      nrow(analysis_data), " eligible hexes; gap statistic B = ", gap_bootstraps
    ),
    x = "Number of clusters (k)",
    y = NULL,
    color = "Method"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom")

ggsave(
  file.path(FIGURES_DIR, "03c_cluster_selection_diagnostics.png"),
  p_selection,
  width = 10,
  height = 8,
  dpi = 300,
  bg = "white"
)

p_balance <- ggplot(
  method_metrics,
  aes(x = k, y = max_cluster_share, color = method, group = method)
) +
  geom_hline(yintercept = 0.50, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = k_values) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Largest Cluster Share Across Methods",
    subtitle = "Lower values indicate more balanced cluster-size solutions",
    x = "Number of clusters (k)",
    y = "Share in largest cluster",
    color = "Method"
  ) +
  theme_minimal(base_size = 11) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom")

ggsave(
  file.path(FIGURES_DIR, "03c_cluster_balance_diagnostics.png"),
  p_balance,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)

print_header("CLUSTER METHOD COMPARISON RESULTS")
print(recommendations)

cat("\nDomain PCA summaries:\n")
print(pca_summary)

cat("\nSilhouette metrics at each method's recommended k:\n")
print(
  method_metrics %>%
    inner_join(
      silhouette_recommendations %>% select(method, recommended_k),
      by = "method"
    ) %>%
    filter(k == recommended_k) %>%
    select(method, k, avg_silhouette, min_cluster_n, max_cluster_share)
)

cat("\n03c sensitivity analysis complete. Canonical 03b outputs were not changed.\n")
