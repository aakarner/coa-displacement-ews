################################################################################
# 03b - Cluster Analysis for Displacement Risk Pattern Discovery
################################################################################
#
# This script performs unsupervised clustering to identify distinct patterns
# of displacement risk across Austin. Unlike the synthetic composite outcome
# in the original approach, clustering allows empirical patterns to emerge
# from the data.
#
# METHODOLOGY:
# Instead of creating a synthetic "displacement_risk" variable from the same
# features used for prediction (circular reasoning), we:
# 1. Use unsupervised clustering to identify natural groupings in the data
# 2. Characterize and label these clusters based on their profiles
# 3. Use cluster assignments as the outcome variable for supervised learning
#
# This creates a defensible, non-circular methodology where displacement
# "types" or "patterns" emerge from data rather than assumptions.
#
# ALGORITHMS TESTED:
# - K-means: Partitional clustering, works well with compact, spherical clusters
# - Hierarchical: Builds cluster hierarchy, good for understanding nested patterns
# - DBSCAN: Density-based, can identify irregular shapes and outliers
#
# INPUTS:
#   - output/engineered_features.rds: Feature matrix from script 03
#
# OUTPUTS:
#   - output/clustered_features.rds: Features with cluster assignments added
#   - output/cluster_assignments.rds: Cluster membership for each hexagon
#   - figures/cluster_*.png: Visualizations of clustering results
#   - figures/cluster_map.png: Spatial map of cluster patterns
#
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(project_path("R", "utils.R"))

print_header("03b - CLUSTER ANALYSIS FOR DISPLACEMENT PATTERNS")

# Load required packages for clustering
suppressPackageStartupMessages({
  library(sf)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(viridis)
  library(cluster)      # For clustering algorithms
})

# Configuration
OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
# Set seed for reproducibility (enables standalone execution; harmless when run via run_analysis.R)
set.seed(42)

################################################################################
# Step 1: Load engineered features
################################################################################

print_progress("Loading engineered features...")
hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  "engineered features"
)

################################################################################
# Step 2: Select and prepare clustering variables
################################################################################

print_header("PREPARING CLUSTERING VARIABLES")

# Use one default input per currently available substantive domain. Detailed
# components and data-coverage fields are retained for profiling, not repeated
# in the clustering matrix.
clustering_vars <- c(
  "rent_pressure_citywide_index",
  "demolition_pressure_index",
  "eviction_pressure_index",
  "sr_311_pressure_index",
  "ownership_pressure_index",
  "demographic_vulnerability_index"
)

profile_vars <- c(
  clustering_vars,
  "costar_present",
  "costar_rent_pressure_index",
  "acs_rent_current_real",
  "acs_rent_growth_recent_annualized_pct",
  "acs_rent_acceleration_pp",
  "demo_density",
  "demo_recent_density",
  "demo_total_recent_density",
  "eviction_latest_12mo_per_100_units",
  "sr_311_smoke_signal_latest_12mo_per_100_units",
  "pct_corporate_units",
  "pct_corporate_parcels",
  "pct_financialized_owner_parcels",
  "ownership_change_index",
  "ownership_history_coverage_pct",
  "corporate_acquisition_recent_per_100_parcels",
  "corporate_acquisition_recent_unit_exposure_pct",
  "transaction_pressure_index",
  "transaction_window_coverage_pct",
  "transaction_recent_per_100_parcels",
  "transaction_recent_unit_exposure_pct",
  "transaction_rate_change_per_100_parcels",
  "amenity_change_index",
  "amenity_recent_weighted_openings",
  "amenity_weighted_opening_change",
  "residential_units_per_km2",
  "median_income",
  "poverty_rate",
  "pct_renter",
  "pct_rent_burden_30plus",
  "pct_college",
  "pct_poc",
  "demographic_vulnerability_equity_index"
)

clustering_vars <- intersect(clustering_vars, names(hex_features))
profile_vars <- intersect(profile_vars, names(hex_features))

print_progress(paste0("Selected ", length(clustering_vars), " variables for clustering"))
cat("Variables:\n")
for(var in clustering_vars) {
  cat(paste0("  - ", var, "\n"))
}

# Prepare clustering dataset. The first-stage citywide cluster should not require
# CoStar rent data, so use the residential eligibility flag when available.
eligibility_col <- if ("primary_cluster_eligible" %in% names(hex_features)) {
  "primary_cluster_eligible"
} else {
  "sufficient_data"
}

cluster_data <- hex_features %>%
  st_drop_geometry() %>%
  filter(.data[[eligibility_col]]) %>%
  select(hex_id, all_of(clustering_vars))

if (length(clustering_vars) < 2) {
  stop("Fewer than two clustering variables are available. Run 03_feature_engineering.R first and check output/feature_list.csv.")
}

# Check missing data in clustering variables
missing_by_var <- cluster_data %>%
  select(-hex_id) %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  mutate(missing_pct = (missing_count / nrow(cluster_data)) * 100) %>%
  arrange(desc(missing_pct))

cat("\nMissing data in clustering variables:\n")
print(head(missing_by_var, 10))

# Filter to only include variables with <50% missing data
vars_to_keep <- missing_by_var %>%
  filter(missing_pct < 50) %>%
  pull(variable)

print_progress(paste0("Keeping ", length(vars_to_keep), " variables with <50% missing data"))

# Update clustering variables
clustering_vars <- vars_to_keep

# Remove rows with any missing values in selected variables
cluster_data_complete <- cluster_data %>%
  select(hex_id, all_of(clustering_vars)) %>%
  drop_na()

if (nrow(cluster_data_complete) < 20) {
  stop("Too few complete observations for clustering after missing-data filtering.")
}

print_progress(paste0("Clustering dataset: ", nrow(cluster_data_complete), " observations"))
print_progress(paste0("Removed ", nrow(cluster_data) - nrow(cluster_data_complete), 
                     " observations with missing values"))

# Scale the data for clustering
# Clustering algorithms are sensitive to scale
cluster_matrix <- cluster_data_complete %>%
  select(-hex_id) %>%
  scale() %>%
  as.matrix()

# Store scaling parameters for later use
scaling_params <- list(
  center = attr(cluster_matrix, "scaled:center"),
  scale = attr(cluster_matrix, "scaled:scale")
)

################################################################################
# Step 3: Determine optimal number of clusters
################################################################################

print_header("DETERMINING OPTIMAL NUMBER OF CLUSTERS")

print_progress("Computing elbow plot (Within-cluster sum of squares)...")

# Calculate WCSS for different k values
set.seed(42)
wcss_values <- sapply(2:10, function(k) {
  kmeans(cluster_matrix, centers = k, nstart = 25, iter.max = 100)$tot.withinss
})

# Create elbow plot
elbow_data <- data.frame(
  k = 2:10,
  wcss = wcss_values
)

p_elbow <- ggplot(elbow_data, aes(x = k, y = wcss)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 3) +
  scale_x_continuous(breaks = 2:10) +
  labs(
    title = "Elbow Plot for K-means Clustering",
    subtitle = "Determining optimal number of clusters",
    x = "Number of Clusters (k)",
    y = "Within-Cluster Sum of Squares (WCSS)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(FIGURES_DIR, "03b_elbow_plot.png"),
  plot = p_elbow,
  width = 10,
  height = 6,
  dpi = 300
)

print_progress("Saved elbow plot")

# Calculate silhouette scores for different k values
print_progress("Computing silhouette scores...")

set.seed(42)
silhouette_scores <- sapply(2:10, function(k) {
  km <- kmeans(cluster_matrix, centers = k, nstart = 25, iter.max = 100)
  sil <- silhouette(km$cluster, dist(cluster_matrix))
  mean(sil[, 3])
})

silhouette_data <- data.frame(
  k = 2:10,
  avg_silhouette = silhouette_scores
)

p_silhouette <- ggplot(silhouette_data, aes(x = k, y = avg_silhouette)) +
  geom_line(color = "darkgreen", size = 1) +
  geom_point(color = "darkgreen", size = 3) +
  scale_x_continuous(breaks = 2:10) +
  labs(
    title = "Average Silhouette Score by Number of Clusters",
    subtitle = "Higher scores indicate better-defined clusters",
    x = "Number of Clusters (k)",
    y = "Average Silhouette Score"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(FIGURES_DIR, "03b_silhouette_plot.png"),
  plot = p_silhouette,
  width = 10,
  height = 6,
  dpi = 300
)

print_progress("Saved silhouette plot")

# Determine optimal k
optimal_k_silhouette <- which.max(silhouette_scores) + 1
cat("\nOptimal number of clusters:\n")
cat(paste0("  - Based on silhouette score: k = ", optimal_k_silhouette, "\n"))
cat(paste0("  - Silhouette score: ", round(silhouette_scores[optimal_k_silhouette - 1], 3), "\n"))

# Use the optimal k (or default to 4 if silhouette suggests too few)
optimal_k <- max(optimal_k_silhouette, 4)
cat(paste0("  - Using k = ", optimal_k, " for final clustering\n"))

################################################################################
# Step 4: Perform K-means clustering
################################################################################

print_header("K-MEANS CLUSTERING")

print_progress(paste0("Running K-means with k = ", optimal_k, "..."))

set.seed(42)
kmeans_result <- kmeans(
  cluster_matrix,
  centers = optimal_k,
  nstart = 50,        # Try 50 different random starts
  iter.max = 100      # Maximum iterations
)

cat("\nK-means clustering results:\n")
cat(paste0("  - Total SSE: ", round(kmeans_result$tot.withinss, 2), "\n"))
cat(paste0("  - Between SS / Total SS: ", 
          round(kmeans_result$betweenss / kmeans_result$totss * 100, 1), "%\n"))

# Cluster sizes
cat("\nCluster sizes:\n")
print(table(kmeans_result$cluster))

################################################################################
# Step 5: Perform hierarchical clustering
################################################################################

print_header("HIERARCHICAL CLUSTERING")

print_progress("Running hierarchical clustering...")

# Use Ward's method (minimizes within-cluster variance)
dist_matrix <- dist(cluster_matrix, method = "euclidean")
hc_result <- hclust(dist_matrix, method = "ward.D2")

# Cut tree to get same number of clusters as k-means
hc_clusters <- cutree(hc_result, k = optimal_k)

cat("\nHierarchical clustering results:\n")
cat("Cluster sizes:\n")
print(table(hc_clusters))

png(file.path(FIGURES_DIR, "03b_dendrogram.png"), width = 1800, height = 1200, res = 150)
plot(
  hc_result,
  labels = FALSE,
  main = paste0("Hierarchical Clustering (", nrow(cluster_data_complete), " observations)"),
  xlab = "Hexagons",
  ylab = "Height"
)
rect.hclust(hc_result, k = optimal_k, border = 2:(optimal_k + 1))
dev.off()

print_progress("Saved dendrogram")

################################################################################
# Step 6: Perform DBSCAN clustering
################################################################################

print_header("DBSCAN CLUSTERING")

if (requireNamespace("dbscan", quietly = TRUE)) {
  print_progress("Running DBSCAN (density-based clustering)...")

  knn_dist <- dbscan::kNNdist(cluster_matrix, k = 5)
  epsilon <- quantile(knn_dist, 0.90)

  print_progress(paste0("Using epsilon = ", round(epsilon, 3), " and minPts = 5"))

  dbscan_result <- dbscan::dbscan(cluster_matrix, eps = epsilon, minPts = 5)

  cat("\nDBSCAN clustering results:\n")
  cat(paste0("  - Number of clusters: ", max(dbscan_result$cluster), "\n"))
  cat(paste0("  - Number of noise points: ", sum(dbscan_result$cluster == 0), "\n"))

  if(max(dbscan_result$cluster) > 0) {
    cat("\nCluster sizes (excluding noise):\n")
    print(table(dbscan_result$cluster[dbscan_result$cluster > 0]))
  }
} else {
  print_progress("Package 'dbscan' not installed; skipping DBSCAN comparison.")
  dbscan_result <- list(cluster = rep(NA_integer_, nrow(cluster_data_complete)), skipped = TRUE)
}

################################################################################
# Step 7: Compare clustering algorithms
################################################################################

print_header("COMPARING CLUSTERING ALGORITHMS")

# Use K-means as primary clustering (most stable and interpretable)
# But save all three for comparison
cluster_assignments <- data.frame(
  hex_id = cluster_data_complete$hex_id,
  cluster_kmeans = kmeans_result$cluster,
  cluster_hierarchical = hc_clusters,
  cluster_dbscan = dbscan_result$cluster
)

# Calculate agreement between methods
agreement_kmeans_hc <- sum(cluster_assignments$cluster_kmeans == 
                           cluster_assignments$cluster_hierarchical) / nrow(cluster_assignments)

cat(paste0("\nAgreement between K-means and Hierarchical: ", 
          round(agreement_kmeans_hc * 100, 1), "%\n"))

# Use K-means clusters as primary
cluster_assignments$cluster <- cluster_assignments$cluster_kmeans

################################################################################
# Step 8: Cluster profiling and characterization
################################################################################

print_header("CLUSTER PROFILING")

print_progress("Calculating cluster profiles...")

# Join back with original (unscaled) data for interpretation
cluster_profiles <- cluster_data_complete %>%
  select(hex_id) %>%
  left_join(cluster_assignments, by = "hex_id") %>%
  left_join(
    hex_features %>%
      st_drop_geometry() %>%
      select(hex_id, all_of(profile_vars)),
    by = "hex_id"
  ) %>%
  select(hex_id, cluster, all_of(profile_vars))

# Calculate summary statistics for each cluster. Keep a dynamic full profile and
# a few legacy aliases used by downstream risk-score code.
profile_summary <- cluster_profiles %>%
  group_by(cluster) %>%
  summarise(
    n = n(),
    across(all_of(profile_vars), ~mean(.x, na.rm = TRUE), .names = "mean_{.col}"),
    .groups = "drop"
  ) %>%
  arrange(cluster)

if (!"mean_rent_change_total" %in% names(profile_summary)) profile_summary$mean_rent_change_total <- NA_real_
if (!"mean_rent_change_recent" %in% names(profile_summary)) profile_summary$mean_rent_change_recent <- NA_real_
if (!"mean_demo_density" %in% names(profile_summary)) profile_summary$mean_demo_density <- NA_real_
if (!"mean_demo_recent" %in% names(profile_summary)) profile_summary$mean_demo_recent <- NA_real_
if (!"mean_vulnerability" %in% names(profile_summary)) {
  profile_summary$mean_vulnerability <- if ("mean_eviction_pressure_index" %in% names(profile_summary)) {
    profile_summary$mean_eviction_pressure_index
  } else {
    NA_real_
  }
}
if (!"mean_median_income" %in% names(profile_summary)) profile_summary$mean_median_income <- NA_real_
if (!"mean_poverty_rate" %in% names(profile_summary)) profile_summary$mean_poverty_rate <- NA_real_
if (!"mean_pct_renter" %in% names(profile_summary)) profile_summary$mean_pct_renter <- NA_real_

cat("\nCluster Profile Summary:\n")
print(profile_summary)

# Save profile summary as CSV
write_csv(profile_summary, file.path(OUTPUT_DIR, "cluster_profiles.csv"))
print_progress("Saved cluster profiles to: output/cluster_profiles.csv")

# Create detailed cluster characterization
print_progress("Creating detailed cluster characterizations...")

characterize_cluster <- function(cluster_num, data) {
  cluster_data <- data %>% filter(cluster == cluster_num)

  mean_or_na <- function(candidates) {
    metric <- candidates[candidates %in% names(cluster_data)][1]
    if (is.na(metric)) return(NA_real_)
    mean(cluster_data[[metric]], na.rm = TRUE)
  }

  threshold_label <- function(value, low_cut, high_cut) {
    if (is.na(value) || is.nan(value)) return("Unmeasured")
    if (value < low_cut) return("Low")
    if (value < high_cut) return("Moderate")
    "High"
  }

  rent_label <- threshold_label(mean_or_na(c("rent_pressure_citywide_index")), 25, 50)
  demo_label <- threshold_label(mean_or_na(c("demolition_pressure_index")), 25, 50)
  eviction_label <- threshold_label(mean_or_na(c("eviction_pressure_index")), 25, 50)
  calls_311_label <- threshold_label(mean_or_na(c("sr_311_pressure_index")), 25, 50)
  ownership_label <- threshold_label(mean_or_na(c("ownership_pressure_index")), 25, 50)
  vulnerability_label <- threshold_label(mean_or_na(c("demographic_vulnerability_index", "vulnerability_index")), 25, 50)

  label <- paste0(
    "Cluster ", cluster_num, ": ",
    rent_label, " Rent, ",
    demo_label, " Demolition, ",
    eviction_label, " Eviction, ",
    calls_311_label, " 311 Smoke Signals, ",
    ownership_label, " Ownership, ",
    vulnerability_label, " Demographic Vulnerability"
  )
  
  return(label)
}

cluster_labels <- sapply(1:optimal_k, function(k) {
  characterize_cluster(k, cluster_profiles)
})

cat("\nCluster Characterizations:\n")
for(i in 1:length(cluster_labels)) {
  cat(paste0("  ", cluster_labels[i], "\n"))
}

################################################################################
# Step 9: Create cluster visualizations
################################################################################

print_header("CREATING CLUSTER VISUALIZATIONS")

# 9.1 PCA visualization
print_progress("Creating PCA visualization...")

set.seed(42)
pca_result <- prcomp(cluster_matrix)

# Get explained variance
var_explained <- summary(pca_result)$importance[2, 1:2] * 100

pca_data <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  cluster = factor(kmeans_result$cluster)
)

p_pca <- ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.6, size = 2) +
  scale_color_viridis_d(option = "turbo", end = 0.9) +
  labs(
    title = "PCA Projection of Displacement Clusters",
    subtitle = paste0("First two principal components (", 
                     round(sum(var_explained), 1), "% variance explained)"),
    x = paste0("PC1 (", round(var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(var_explained[2], 1), "%)"),
    color = "Cluster"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "right"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "03b_pca_clusters.png"),
  plot = p_pca,
  width = 10,
  height = 7,
  dpi = 300
)

print_progress("Saved PCA cluster plot")

# 9.2 Cluster profiles radar chart
print_progress("Creating cluster profile charts...")

# Normalize each variable to 0-100 scale for radar chart
profile_normalized <- cluster_profiles %>%
  group_by(cluster) %>%
  summarise(across(all_of(clustering_vars), ~mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  mutate(across(-cluster, ~normalize_to_100(.x)))

# Create a bar plot for each cluster showing normalized profiles
profile_long <- profile_normalized %>%
  pivot_longer(cols = -cluster, names_to = "variable", values_to = "value") %>%
  mutate(
    cluster = factor(cluster),
    variable = factor(variable, levels = clustering_vars)
  )

p_profiles <- ggplot(profile_long, aes(x = variable, y = value, fill = cluster)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(option = "turbo", end = 0.9) +
  labs(
    title = "Cluster Profiles Across Key Displacement Variables",
    subtitle = "Normalized to 0-100 scale for comparison",
    x = "Variable",
    y = "Normalized Value (0-100)",
    fill = "Cluster"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "03b_cluster_profiles.png"),
  plot = p_profiles,
  width = 14,
  height = 8,
  dpi = 300
)

print_progress("Saved cluster profile plot")

################################################################################
# Step 10: Create geographic cluster map
################################################################################

print_header("CREATING GEOGRAPHIC CLUSTER MAP")

print_progress("Creating spatial cluster visualization...")

# Join clusters with spatial features
hex_features_with_clusters <- hex_features %>%
  left_join(
    cluster_assignments %>% select(hex_id, cluster),
    by = "hex_id"
  )

# Create map of clusters
cluster_map_data <- hex_features_with_clusters %>%
  filter(!is.na(cluster)) %>%
  mutate(cluster = factor(cluster))

p_cluster_map <- ggplot() +
  geom_sf(data = cluster_map_data, aes(fill = cluster), color = NA) +
  scale_fill_viridis_d(option = "turbo", end = 0.9, 
                       labels = paste("Cluster", 1:optimal_k)) +
  labs(
    title = "Geographic Distribution of Displacement Clusters",
    subtitle = paste0("Austin, TX - ", nrow(cluster_map_data), " hexagons clustered"),
    fill = "Cluster"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    panel.grid = element_blank()
  )

ggsave(
  filename = file.path(FIGURES_DIR, "03b_cluster_map.png"),
  plot = p_cluster_map,
  width = 10,
  height = 10,
  dpi = 300
)

print_progress("Saved cluster map")

################################################################################
# Step 11: Save clustering results
################################################################################

print_header("SAVING CLUSTERING RESULTS")

# Comprehensive clustering results object
clustering_results <- list(
  # Cluster assignments
  cluster_assignments = cluster_assignments,
  
  # Algorithm results
  kmeans_result = kmeans_result,
  hierarchical_result = hc_result,
  dbscan_result = dbscan_result,
  
  # Cluster profiles
  cluster_profiles = profile_summary,
  cluster_labels = cluster_labels,
  
  # Optimal parameters
  optimal_k = optimal_k,
  optimal_k_silhouette = optimal_k_silhouette,
  
  # Evaluation metrics
  silhouette_scores = silhouette_data,
  wcss_values = elbow_data,
  
  # Clustering variables used
  clustering_vars = clustering_vars,
  
  # Scaling parameters (for applying to new data)
  scaling_params = scaling_params,
  
  # Metadata
  n_observations = nrow(cluster_data_complete),
  n_variables = length(clustering_vars),
  date_created = Sys.time()
)

save_output(
  clustering_results,
  file.path(OUTPUT_DIR, "cluster_analysis_results.rds"),
  "clustering analysis results"
)

# Save features with cluster assignments
save_output(
  hex_features_with_clusters,
  file.path(OUTPUT_DIR, "hex_features_with_clusters.rds"),
  "features with cluster assignments"
)

################################################################################
# Step 12: Generate cluster interpretation guide
################################################################################

print_header("CLUSTER INTERPRETATION GUIDE")

cat("\n")
cat("=============================================================================\n")
cat("CLUSTER-BASED DISPLACEMENT RISK FRAMEWORK\n")
cat("=============================================================================\n\n")

cat("METHODOLOGY:\n")
cat("Instead of creating a synthetic 'displacement_risk' variable from the same\n")
cat("features used for prediction (circular reasoning), we used unsupervised\n")
cat("clustering to identify natural patterns in the data.\n\n")

cat("INTERPRETATION:\n")
for(i in 1:optimal_k) {
  cat(paste0("\n", cluster_labels[i], "\n"))
  profile <- profile_summary %>% filter(cluster == i)
  cat(paste0("  - Hexagons: ", profile$n, "\n"))
  if ("mean_costar_present" %in% names(profile)) {
    cat(paste0("  - CoStar-covered share: ", round(profile$mean_costar_present * 100, 1), "%\n"))
  }
  cat(paste0("  - Avg demo density: ", round(profile$mean_demo_density, 2), "/km2\n"))
  if ("mean_eviction_latest_12mo_per_100_units" %in% names(profile)) {
    cat(paste0("  - Avg latest-12mo evictions: ",
               round(profile$mean_eviction_latest_12mo_per_100_units, 2),
               " cases per 100 units\n"))
  }
  if ("mean_pct_corporate_units" %in% names(profile)) {
    cat(paste0("  - Avg corporate unit share: ", round(profile$mean_pct_corporate_units, 1), "%\n"))
  }
  if ("mean_sr_311_pressure_index" %in% names(profile)) {
    cat(paste0("  - Avg 311 smoke-signal pressure: ",
               round(profile$mean_sr_311_pressure_index, 1), "/100\n"))
  }
  if ("mean_demographic_vulnerability_index" %in% names(profile)) {
    cat(paste0("  - Avg demographic vulnerability: ",
               round(profile$mean_demographic_vulnerability_index, 1), "/100\n"))
  } else if ("mean_vulnerability_index" %in% names(profile)) {
    cat(paste0("  - Avg demographic vulnerability: ",
               round(profile$mean_vulnerability_index, 1), "/100\n"))
  }
}

cat("\n")
cat("NEXT STEPS:\n")
cat("1. Review cluster profiles and geographic distribution\n")
cat("2. Label clusters based on displacement risk (High/Moderate/Low)\n")
cat("3. Use cluster assignments as outcome variable for supervised learning\n")
cat("4. Models will predict which cluster a hexagon belongs to\n")
cat("5. This provides interpretable results: 'This area resembles high-risk clusters'\n")
cat("\n")

################################################################################
# Summary
################################################################################

print_header("STEP 03b COMPLETE")
cat("✓ Clustering performed on displacement indicators\n")
cat(paste0("✓ Optimal number of clusters determined: k = ", optimal_k, "\n"))
cat("✓ Three clustering algorithms tested:\n")
cat("  - K-means (primary)\n")
cat("  - Hierarchical\n")
cat("  - DBSCAN (if dbscan package is installed)\n")
cat("✓ Cluster profiles and characterizations created\n")
cat("✓ Visualizations generated:\n")
cat("  - Elbow plot\n")
cat("  - Silhouette plot\n")
cat("  - PCA cluster visualization\n")
cat("  - Cluster profile charts\n")
cat("  - Geographic cluster map\n")
cat(paste0("✓ Features with clusters saved to: ", 
          file.path(OUTPUT_DIR, "hex_features_with_clusters.rds"), "\n"))
cat(paste0("✓ Cluster analysis results saved to: ",
          file.path(OUTPUT_DIR, "cluster_analysis_results.rds"), "\n"))
cat(paste0("✓ Cluster profiles saved to: ",
          file.path(OUTPUT_DIR, "cluster_profiles.csv"), "\n"))
