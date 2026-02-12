################################################################################
# 06 - Generate Displacement Risk Scores and Classifications
################################################################################
#
# This script uses the trained models to generate displacement risk scores
# for all hexagonal cells in Austin. It produces:
# - Risk scores (0-100 scale)
# - Risk categories (Low, Moderate, High, Very High)
# - Contributing factors for each cell
# - Ensemble predictions combining multiple models
#
################################################################################

print_header("06 - GENERATING DISPLACEMENT RISK SCORES")

# Source utilities
source(here::here("R/utils.R"))

# Configuration
OUTPUT_DIR <- here::here("output")
set.seed(42)

################################################################################
# Step 1: Load models and data
################################################################################

print_progress("Loading trained models and feature data...")

models_list <- load_output(
  file.path(OUTPUT_DIR, "trained_models.rds"),
  "trained models"
)

hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features_with_clusters.rds"),
  "engineered features with cluster assignments"
)

# Load clustering results to understand cluster labels
clustering_results <- load_output(
  file.path(OUTPUT_DIR, "cluster_analysis_results.rds"),
  "cluster analysis results"
)

# Extract components
model_elastic <- models_list$elastic_net
model_rf <- models_list$random_forest
model_xgb <- models_list$xgboost
predictor_vars <- models_list$predictor_vars

################################################################################
# Step 2: Prepare data for prediction
################################################################################

print_progress("Preparing data for prediction...")

# Create prediction dataset
# For cells with missing values, we'll use model-specific approaches
prediction_data <- hex_features %>%
  st_drop_geometry() %>%
  select(hex_id, h3_index, longitude, latitude, all_of(predictor_vars))

# Count how many cells have complete data
complete_cases <- sum(complete.cases(prediction_data[, predictor_vars]))
total_cases <- nrow(prediction_data)

cat(paste0("\nData completeness:\n"))
cat(paste0("  - Cells with complete data: ", complete_cases, 
          " (", round(complete_cases/total_cases*100, 1), "%)\n"))
cat(paste0("  - Cells with missing values: ", total_cases - complete_cases,
          " (", round((total_cases - complete_cases)/total_cases*100, 1), "%)\n"))

################################################################################
# Step 3: Generate cluster predictions from each model
################################################################################

print_header("GENERATING CLUSTER PREDICTIONS")

print_progress("Generating cluster predictions from Elastic Net model...")
# Elastic Net requires complete cases (no missing values)
pred_elastic_class <- rep(NA, nrow(prediction_data))
complete_idx <- complete.cases(prediction_data[, predictor_vars])
pred_elastic_class[complete_idx] <- as.character(predict(
  model_elastic,
  newdata = prediction_data[complete_idx, predictor_vars]
))

# Also get class probabilities for ensemble
pred_elastic_prob <- matrix(NA, nrow = nrow(prediction_data), 
                            ncol = clustering_results$optimal_k)
if(sum(complete_idx) > 0) {
  pred_elastic_prob[complete_idx, ] <- predict(
    model_elastic,
    newdata = prediction_data[complete_idx, predictor_vars],
    type = "prob"
  )
}

print_progress("Generating cluster predictions from Random Forest model...")
# Random Forest can handle some missing values
pred_rf_class <- as.character(predict(
  model_rf,
  newdata = prediction_data[, predictor_vars]
))

pred_rf_prob <- predict(
  model_rf,
  newdata = prediction_data[, predictor_vars],
  type = "prob"
)

print_progress("Generating cluster predictions from XGBoost model...")
# XGBoost can handle missing values
pred_xgb_class <- as.character(predict(
  model_xgb,
  newdata = prediction_data[, predictor_vars]
))

pred_xgb_prob <- predict(
  model_xgb,
  newdata = prediction_data[, predictor_vars],
  type = "prob"
)

cat("\nPrediction coverage:\n")
cat(paste0("  - Elastic Net: ", sum(!is.na(pred_elastic_class)), " cells\n"))
cat(paste0("  - Random Forest: ", sum(!is.na(pred_rf_class)), " cells\n"))
cat(paste0("  - XGBoost: ", sum(!is.na(pred_xgb_class)), " cells\n"))

################################################################################
# Step 4: Convert cluster predictions to risk scores
################################################################################

print_header("CONVERTING CLUSTER PREDICTIONS TO RISK SCORES")

# METHODOLOGY:
# For cluster-based predictions, we need to convert cluster assignments
# into continuous risk scores. We use two approaches:
# 1. Direct cluster mapping: Assign risk scores based on cluster profiles
# 2. Probability-based: Use probability of high-risk cluster membership

print_progress("Mapping clusters to risk scores based on cluster profiles...")

# Get cluster profiles to determine which clusters are high-risk
cluster_profiles <- clustering_results$cluster_profiles

# Simple heuristic: Rank clusters by average displacement indicators
# Users can customize this based on domain knowledge
cluster_risk_mapping <- cluster_profiles %>%
  mutate(
    # Composite risk score for each cluster
    cluster_risk_score = (
      normalize_to_100(mean_rent_change_total) * 0.4 +
      normalize_to_100(mean_demo_density) * 0.3 +
      normalize_to_100(mean_vulnerability) * 0.3
    )
  ) %>%
  select(cluster, cluster_risk_score) %>%
  arrange(cluster)

cat("\nCluster Risk Score Mapping:\n")
print(cluster_risk_mapping)

# Function to convert cluster prediction to risk score
cluster_to_risk <- function(cluster_pred) {
  cluster_num <- as.numeric(cluster_pred)
  ifelse(!is.na(cluster_num),
         cluster_risk_mapping$cluster_risk_score[cluster_num],
         NA_real_)
}

# Convert class predictions to risk scores
pred_elastic_risk <- cluster_to_risk(pred_elastic_class)
pred_rf_risk <- cluster_to_risk(pred_rf_class)
pred_xgb_risk <- cluster_to_risk(pred_xgb_class)

# Alternative: Use probability-weighted risk scores
# Weight each cluster's risk score by the probability of membership
pred_elastic_risk_prob <- rowSums(
  sweep(pred_elastic_prob, 2, cluster_risk_mapping$cluster_risk_score, "*"),
  na.rm = TRUE
)
pred_elastic_risk_prob[!complete_idx] <- NA_real_

pred_rf_risk_prob <- rowSums(
  sweep(as.matrix(pred_rf_prob), 2, cluster_risk_mapping$cluster_risk_score, "*")
)

pred_xgb_risk_prob <- rowSums(
  sweep(as.matrix(pred_xgb_prob), 2, cluster_risk_mapping$cluster_risk_score, "*")
)

################################################################################
# Step 5: Create ensemble prediction
################################################################################

print_header("CREATING ENSEMBLE PREDICTIONS")

# CONCEPT: Ensemble = combining multiple models
# For cluster-based predictions, we combine using probability-weighted risk scores
# This provides more nuanced predictions than simple cluster assignment

print_progress("Combining models into ensemble prediction...")

# Get model performance weights from validation (use accuracy instead of RMSE)
performance <- models_list$performance
weights <- performance$Accuracy  # Higher accuracy = higher weight
weights <- weights / sum(weights)  # Normalize to sum to 1

cat("\nEnsemble weights (based on test accuracy):\n")
cat(paste0("  - Elastic Net: ", round(weights[1], 3), "\n"))
cat(paste0("  - Random Forest: ", round(weights[2], 3), "\n"))
cat(paste0("  - XGBoost: ", round(weights[3], 3), "\n"))

# Create weighted ensemble using probability-based risk scores
risk_scores <- prediction_data %>%
  mutate(
    # Individual model predictions (cluster-based risk)
    pred_elastic_net_cluster = pred_elastic_class,
    pred_random_forest_cluster = pred_rf_class,
    pred_xgboost_cluster = pred_xgb_class,
    
    # Individual model risk scores (probability-weighted)
    pred_elastic_net = pred_elastic_risk_prob,
    pred_random_forest = pred_rf_risk_prob,
    pred_xgboost = pred_xgb_risk_prob,
    
    # Weighted ensemble (when all models available)
    risk_score_ensemble = if_else(
      !is.na(pred_elastic_net),
      weights[1] * pred_elastic_net + weights[2] * pred_random_forest + weights[3] * pred_xgboost,
      (pred_random_forest + pred_xgboost) / 2  # Simple average if elastic net missing
    ),
    
    # Constrain to 0-100 range
    risk_score_ensemble = pmax(0, pmin(100, risk_score_ensemble)),
    
    # Also keep individual model scores
    risk_score_rf = pmax(0, pmin(100, pred_random_forest)),
    risk_score_xgb = pmax(0, pmin(100, pred_xgboost)),
    
    # Most likely cluster for each model (for interpretation)
    cluster_pred_ensemble = {
      # Use Random Forest's prediction as default (best coverage)
      pred_rf_class
    }
  )

################################################################################
# Step 6: Classify into risk categories
################################################################################

print_progress("Classifying cells into risk categories...")

risk_scores <- risk_scores %>%
  mutate(
    # Primary classification (using ensemble)
    risk_category = categorize_risk(risk_score_ensemble),
    
    # Also classify by individual models for comparison
    risk_category_rf = categorize_risk(risk_score_rf),
    risk_category_xgb = categorize_risk(risk_score_xgb)
  )

# Summary of risk categories
cat("\nRisk category distribution (Ensemble):\n")
print(table(risk_scores$risk_category))

cat("\nRisk category percentages:\n")
print(round(prop.table(table(risk_scores$risk_category)) * 100, 1))

################################################################################
# Step 7: Identify key contributing factors
################################################################################

print_header("IDENTIFYING KEY CONTRIBUTING FACTORS")

print_progress("Determining top contributing factors for each cell...")

# Get feature importance (average across models)
validation_results <- readRDS(file.path(OUTPUT_DIR, "validation_results.rds"))
feature_importance <- validation_results$feature_importance %>%
  arrange(desc(importance_avg))

# For each cell, identify which risk factors are elevated
# We'll flag features in the top 75th percentile for that feature

identify_contributing_factors <- function(data, top_n = 3) {
  factors <- character(nrow(data))
  
  # Get top important features to check
  top_features <- head(feature_importance$feature, 10)
  
  for(i in 1:nrow(data)) {
    cell_factors <- c()
    
    # Check each top feature
    for(feat in top_features) {
      if(feat %in% names(data)) {
        value <- data[[feat]][i]
        if(!is.na(value)) {
          # Get percentile of this value
          percentile <- ecdf(data[[feat]])(value)
          
          # If in top 25% for this risk factor, add it
          if(percentile >= 0.75) {
            cell_factors <- c(cell_factors, feat)
          }
        }
      }
    }
    
    # Take top N factors
    if(length(cell_factors) > 0) {
      factors[i] <- paste(head(cell_factors, top_n), collapse = "; ")
    } else {
      factors[i] <- "Multiple factors"
    }
  }
  
  return(factors)
}

# Get original feature data for contributing factor analysis
original_features <- hex_features %>%
  st_drop_geometry() %>%
  select(hex_id, any_of(predictor_vars))

risk_scores_with_factors <- risk_scores %>%
  left_join(original_features, by = "hex_id") %>%
  mutate(
    contributing_factors = identify_contributing_factors(., top_n = 3)
  )

# Clean up by removing the extra feature columns
risk_scores <- risk_scores_with_factors %>%
  select(hex_id, h3_index, longitude, latitude,
         pred_elastic_net_cluster, pred_random_forest_cluster, pred_xgboost_cluster,
         cluster_pred_ensemble,
         pred_elastic_net, pred_random_forest, pred_xgboost,
         risk_score_ensemble, risk_score_rf, risk_score_xgb,
         risk_category, risk_category_rf, risk_category_xgb,
         contributing_factors)

################################################################################
# Step 8: Add spatial geometry back
################################################################################

print_progress("Adding spatial geometry to risk scores...")

# Join back to hex grid
hex_grid <- readRDS(file.path(OUTPUT_DIR, "hex_grid.rds"))

risk_scores_spatial <- hex_grid %>%
  select(hex_id, h3_index, geometry) %>%
  left_join(
    st_drop_geometry(risk_scores),
    by = c("hex_id", "h3_index")
  )

################################################################################
# Step 9: Generate summary statistics
################################################################################

print_header("RISK SCORE SUMMARY STATISTICS")

cat("\nEnsemble Risk Score Statistics:\n")
print(summary(risk_scores$risk_score_ensemble))

cat("\nPredicted Cluster Distribution:\n")
print(table(risk_scores$cluster_pred_ensemble))

cat("\nRisk Categories:\n")
risk_category_summary <- risk_scores %>%
  count(risk_category) %>%
  mutate(percentage = round(n / sum(n) * 100, 1))
print(risk_category_summary)

cat("\nGeographic Distribution:\n")
cat(paste0("  - Mean risk score: ", round(mean(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))
cat(paste0("  - Median risk score: ", round(median(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))
cat(paste0("  - Standard deviation: ", round(sd(risk_scores$risk_score_ensemble, na.rm = TRUE), 2), "\n"))

# Top 10 highest risk cells
cat("\nTop 10 Highest Risk Cells:\n")
top_risk <- risk_scores %>%
  arrange(desc(risk_score_ensemble)) %>%
  head(10) %>%
  select(hex_id, longitude, latitude, risk_score_ensemble, risk_category, 
         cluster_pred_ensemble, contributing_factors)
print(top_risk)

################################################################################
# Step 10: Save risk scores
################################################################################

print_progress("Saving risk scores...")

# Save as RDS (with spatial geometry)
save_output(
  risk_scores_spatial,
  file.path(OUTPUT_DIR, "displacement_risk_scores.rds"),
  "displacement risk scores (spatial)"
)

# Save as CSV (without geometry for easy inspection)
risk_scores_csv <- risk_scores_spatial %>%
  st_drop_geometry() %>%
  select(hex_id, h3_index, longitude, latitude,
         risk_score = risk_score_ensemble,
         risk_category,
         predicted_cluster = cluster_pred_ensemble,
         contributing_factors,
         risk_score_rf, risk_score_xgb)

write_csv(
  risk_scores_csv,
  file.path(OUTPUT_DIR, "displacement_risk_scores.csv")
)

print_progress("Also saved CSV version to: output/displacement_risk_scores.csv")

# Save summary statistics
summary_stats <- list(
  risk_category_counts = risk_category_summary,
  summary_statistics = data.frame(
    metric = c("Mean", "Median", "SD", "Min", "Max"),
    value = c(
      mean(risk_scores$risk_score_ensemble, na.rm = TRUE),
      median(risk_scores$risk_score_ensemble, na.rm = TRUE),
      sd(risk_scores$risk_score_ensemble, na.rm = TRUE),
      min(risk_scores$risk_score_ensemble, na.rm = TRUE),
      max(risk_scores$risk_score_ensemble, na.rm = TRUE)
    )
  ),
  top_10_highest_risk = top_risk
)

save_output(
  summary_stats,
  file.path(OUTPUT_DIR, "risk_score_summary.rds"),
  "risk score summary statistics"
)

################################################################################
# Summary
################################################################################

print_header("STEP 06 COMPLETE")
cat("✓ Cluster predictions generated from all three models\n")
cat("✓ Cluster assignments converted to risk scores\n")
cat("✓ Ensemble predictions created using weighted probabilities\n")
cat("✓ Risk scores calculated for all cells (0-100 scale)\n")
cat("✓ Cells classified into risk categories\n")
cat("✓ Contributing factors identified for each cell\n")
cat("\nInterpretation:\n")
cat("  - Risk scores are based on probability-weighted cluster membership\n")
cat("  - Higher scores indicate similarity to high-risk displacement clusters\n")
cat("  - Cluster predictions show which displacement pattern each area resembles\n")
cat(paste0("\n✓ Spatial risk scores saved to: ", 
          file.path(OUTPUT_DIR, "displacement_risk_scores.rds"), "\n"))
cat(paste0("✓ CSV export saved to: ",
          file.path(OUTPUT_DIR, "displacement_risk_scores.csv"), "\n"))
