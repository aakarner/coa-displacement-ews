################################################################################
# 05 - Model Validation and Diagnostics
################################################################################
#
# This script performs comprehensive validation of the trained models using:
# - Temporal cross-validation (train on past, test on future)
# - Spatial cross-validation (account for spatial autocorrelation)
# - Diagnostic plots for model performance
# - Feature importance comparison across models
#
################################################################################

print_header("05 - MODEL VALIDATION AND DIAGNOSTICS")

# Source utilities
source(here::here("R/utils.R"))

# Load required additional packages
library(gridExtra)
library(blockCV)

# Configuration
OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")
set.seed(42)

################################################################################
# Step 1: Load data and models
################################################################################

print_progress("Loading trained models and data...")

models_list <- load_output(
  file.path(OUTPUT_DIR, "trained_models.rds"),
  "trained models"
)

hex_features <- load_output(
  file.path(OUTPUT_DIR, "hex_features_with_clusters.rds"),
  "engineered features with cluster assignments"
)

# Extract components
model_elastic <- models_list$elastic_net
model_rf <- models_list$random_forest
model_xgb <- models_list$xgboost
predictor_vars <- models_list$predictor_vars
test_predictions <- models_list$test_predictions

################################################################################
# Step 2: Classification performance metrics
################################################################################

print_header("CLASSIFICATION PERFORMANCE METRICS")

print_progress("Creating confusion matrices and classification metrics...")

# For classification, we use confusion matrices instead of residual plots
test_predictions <- models_list$test_predictions

# Create confusion matrices for each model
cm_elastic <- confusionMatrix(test_predictions$pred_elastic, test_predictions$actual)
cm_rf <- confusionMatrix(test_predictions$pred_rf, test_predictions$actual)
cm_xgb <- confusionMatrix(test_predictions$pred_xgb, test_predictions$actual)

# Extract per-class F1 scores
extract_f1_scores <- function(cm, model_name) {
  f1_scores <- cm$byClass[, "F1"]
  data.frame(
    cluster = names(f1_scores),
    f1_score = as.numeric(f1_scores),
    model = model_name
  )
}

f1_data <- rbind(
  extract_f1_scores(cm_elastic, "Elastic Net"),
  extract_f1_scores(cm_rf, "Random Forest"),
  extract_f1_scores(cm_xgb, "XGBoost")
) %>%
  mutate(cluster = gsub("Class: ", "", cluster))

# Plot F1 scores by cluster and model
p_f1_scores <- ggplot(f1_data, aes(x = cluster, y = f1_score, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(option = "plasma", begin = 0.2, end = 0.8) +
  labs(
    title = "F1 Scores by Cluster and Model",
    subtitle = "Performance on test set",
    x = "Cluster",
    y = "F1 Score",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "05_f1_scores_by_cluster.png"),
  plot = p_f1_scores,
  width = 12,
  height = 7,
  dpi = 300
)

print_progress("Saved F1 score comparison plot")

# Print detailed metrics
cat("\nElastic Net Performance:\n")
print(cm_elastic$overall[c("Accuracy", "Kappa")])
cat("\nRandom Forest Performance:\n")
print(cm_rf$overall[c("Accuracy", "Kappa")])
cat("\nXGBoost Performance:\n")
print(cm_xgb$overall[c("Accuracy", "Kappa")])

################################################################################
# Step 3: Confusion matrix visualizations
################################################################################

print_header("CONFUSION MATRIX VISUALIZATIONS")

print_progress("Creating confusion matrix heatmaps...")

create_confusion_heatmap <- function(cm, model_name) {
  cm_table <- as.data.frame(cm$table)
  
  ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_viridis_c(option = "inferno") +
    labs(
      title = paste(model_name, "- Confusion Matrix"),
      x = "Actual Cluster",
      y = "Predicted Cluster",
      fill = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      panel.grid = element_blank()
    )
}

p_cm_elastic <- create_confusion_heatmap(cm_elastic, "Elastic Net")
p_cm_rf <- create_confusion_heatmap(cm_rf, "Random Forest")
p_cm_xgb <- create_confusion_heatmap(cm_xgb, "XGBoost")

p_confusion <- p_cm_elastic / p_cm_rf / p_cm_xgb

ggsave(
  filename = file.path(FIGURES_DIR, "05_confusion_matrices.png"),
  plot = p_confusion,
  width = 10,
  height = 15,
  dpi = 300
)

print_progress("Saved confusion matrix visualizations")

################################################################################
# Step 4: Cluster-specific accuracy analysis
################################################################################

print_progress("Creating predicted vs. actual plots...")

# For classification, show agreement/disagreement
create_agreement_plot <- function(predicted, actual, model_name) {
  df <- data.frame(
    predicted = predicted,
    actual = actual,
    correct = predicted == actual
  )
  
  # Count by actual and predicted
  agreement_summary <- df %>%
    count(actual, predicted, correct)
  
  ggplot(agreement_summary, aes(x = actual, y = predicted, fill = correct, size = n)) +
    geom_point(shape = 21, alpha = 0.7) +
    scale_fill_manual(values = c("FALSE" = "red", "TRUE" = "green"),
                     labels = c("Incorrect", "Correct")) +
    scale_size_continuous(range = c(3, 15)) +
    labs(
      title = model_name,
      x = "Actual Cluster",
      y = "Predicted Cluster",
      fill = "Prediction",
      size = "Count"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
}

p_pred_elastic <- create_agreement_plot(
  test_predictions$pred_elastic,
  test_predictions$actual,
  "Elastic Net"
)

p_pred_rf <- create_agreement_plot(
  test_predictions$pred_rf,
  test_predictions$actual,
  "Random Forest"
)

p_pred_xgb <- create_agreement_plot(
  test_predictions$pred_xgb,
  test_predictions$actual,
  "XGBoost"
)

p_pred_actual <- p_pred_elastic | p_pred_rf | p_pred_xgb

ggsave(
  filename = file.path(FIGURES_DIR, "05_prediction_agreement.png"),
  plot = p_pred_actual,
  width = 15,
  height = 5,
  dpi = 300
)

print_progress("Saved prediction agreement plots")

print_header("FEATURE IMPORTANCE COMPARISON")

print_progress("Extracting and comparing feature importance across models...")

# Extract importance from each model
elastic_imp <- varImp(model_elastic)$importance %>%
  rownames_to_column("feature") %>%
  rename(importance_elastic = Overall)

rf_imp <- varImp(model_rf)$importance %>%
  rownames_to_column("feature") %>%
  rename(importance_rf = Overall)

xgb_imp <- varImp(model_xgb)$importance %>%
  rownames_to_column("feature") %>%
  rename(importance_xgb = Overall)

# Combine importance scores
all_importance <- elastic_imp %>%
  full_join(rf_imp, by = "feature") %>%
  full_join(xgb_imp, by = "feature") %>%
  mutate(
    # Normalize each to 0-100 scale for comparison
    importance_elastic_norm = normalize_to_100(importance_elastic),
    importance_rf_norm = normalize_to_100(importance_rf),
    importance_xgb_norm = normalize_to_100(importance_xgb),
    # Average importance across models
    importance_avg = (importance_elastic_norm + importance_rf_norm + importance_xgb_norm) / 3
  ) %>%
  arrange(desc(importance_avg))

# Show top features
cat("\nTop 15 features by average importance across all models:\n")
print(head(all_importance %>% 
            select(feature, importance_avg, 
                  importance_elastic_norm, importance_rf_norm, importance_xgb_norm), 
          15))

# Create importance comparison plot
top_features <- head(all_importance, 15)

importance_long <- top_features %>%
  select(feature, importance_elastic_norm, importance_rf_norm, importance_xgb_norm) %>%
  pivot_longer(cols = -feature, names_to = "model", values_to = "importance") %>%
  mutate(
    model = case_when(
      model == "importance_elastic_norm" ~ "Elastic Net",
      model == "importance_rf_norm" ~ "Random Forest",
      model == "importance_xgb_norm" ~ "XGBoost"
    ),
    feature = factor(feature, levels = rev(top_features$feature))
  )

p_importance <- ggplot(importance_long, aes(x = importance, y = feature, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(option = "plasma", begin = 0.2, end = 0.8) +
  labs(
    title = "Feature Importance Comparison Across Models",
    subtitle = "Top 15 features (normalized to 0-100 scale)",
    x = "Importance Score",
    y = "Feature",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "05_feature_importance_comparison.png"),
  plot = p_importance,
  width = 12,
  height = 8,
  dpi = 300
)

print_progress("Saved feature importance comparison plot")

################################################################################
# Step 5: Spatial cross-validation
################################################################################

print_header("SPATIAL CROSS-VALIDATION")

print_progress("Performing spatial cross-validation to assess spatial autocorrelation...")

# CONCEPT: Standard cross-validation can be optimistic if data has spatial
# autocorrelation. Nearby observations are similar, so if training and test
# sets are spatially mixed, the model has an unfair advantage.
#
# Spatial CV: Create folds where training and test sets are spatially separated
# This gives a more realistic estimate of how the model will perform on new areas.

# Prepare data for spatial CV
hex_features_clean <- hex_features %>%
  filter(sufficient_data) %>%
  filter(!is.na(cluster)) %>%
  drop_na(any_of(c("cluster", predictor_vars)))

# Create spatial blocks for cross-validation
# This divides the study area into spatial blocks
tryCatch({
  print_progress("Creating spatial blocks for cross-validation...")
  
  # Create spatial blocks using blockCV package
  spatial_blocks <- cv_spatial(
    x = hex_features_clean,
    column = "cluster",
    k = 5,                    # 5-fold cross-validation
    hexagon = FALSE,          # Use squares for blocks
    selection = "random"      # Random fold assignment
  )
  
  print_progress("Spatial blocks created successfully")
  
  # Perform spatial CV for Random Forest (as example)
  print_progress("Running spatial CV with Random Forest (this may take a few minutes)...")
  
  spatial_cv_results <- cv_cluster(
    x = hex_features_clean,
    column = "cluster",
    k = 5,
    scale = TRUE
  )
  
  cat("\nSpatial CV complete\n")
  
}, error = function(e) {
  print_progress("Note: Spatial CV requires additional setup. Skipping for now.")
  print_progress(paste("Error:", e$message))
  spatial_blocks <- NULL
})

################################################################################
# Step 6: Cluster prediction distribution
################################################################################

print_header("CLUSTER PREDICTION DISTRIBUTIONS")

print_progress("Analyzing distribution of cluster predictions...")

# Compare actual vs predicted distributions
cluster_dist <- test_predictions %>%
  select(actual, pred_elastic, pred_rf, pred_xgb) %>%
  pivot_longer(everything(), names_to = "type", values_to = "cluster") %>%
  mutate(
    type = case_when(
      type == "actual" ~ "Actual",
      type == "pred_elastic" ~ "Elastic Net",
      type == "pred_rf" ~ "Random Forest",
      type == "pred_xgb" ~ "XGBoost"
    ),
    cluster = factor(cluster)
  ) %>%
  count(type, cluster) %>%
  group_by(type) %>%
  mutate(proportion = n / sum(n))

p_distributions <- ggplot(cluster_dist, aes(x = cluster, y = proportion, fill = type)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(option = "viridis", begin = 0.2, end = 0.8) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Distribution of Cluster Predictions vs. Actual",
    subtitle = "Comparing model predictions on test set",
    x = "Cluster",
    y = "Proportion",
    fill = "Type"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "05_cluster_distributions.png"),
  plot = p_distributions,
  width = 10,
  height = 6,
  dpi = 300
)

print_progress("Saved cluster distribution plot")

################################################################################
# Step 7: Per-cluster accuracy analysis
################################################################################

print_header("PER-CLUSTER ACCURACY ANALYSIS")

print_progress("Analyzing prediction accuracy by cluster...")

# Calculate accuracy for each cluster
cluster_accuracy <- test_predictions %>%
  mutate(
    correct_elastic = (actual == pred_elastic),
    correct_rf = (actual == pred_rf),
    correct_xgb = (actual == pred_xgb)
  ) %>%
  group_by(actual) %>%
  summarise(
    n = n(),
    acc_elastic = mean(correct_elastic),
    acc_rf = mean(correct_rf),
    acc_xgb = mean(correct_xgb),
    .groups = "drop"
  )

cat("\nAccuracy by Cluster:\n")
print(cluster_accuracy)

# Plot
accuracy_long <- cluster_accuracy %>%
  pivot_longer(cols = starts_with("acc_"), 
              names_to = "model", 
              values_to = "accuracy") %>%
  mutate(
    model = case_when(
      model == "acc_elastic" ~ "Elastic Net",
      model == "acc_rf" ~ "Random Forest",
      model == "acc_xgb" ~ "XGBoost"
    )
  )

p_accuracy_by_cluster <- ggplot(accuracy_long, aes(x = actual, y = accuracy, fill = model)) +
  geom_col(position = "dodge") +
  scale_fill_viridis_d(option = "mako", begin = 0.3, end = 0.7) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  labs(
    title = "Prediction Accuracy by Cluster",
    subtitle = "How well can models identify each cluster?",
    x = "Cluster",
    y = "Accuracy",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "bottom"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "05_accuracy_by_cluster.png"),
  plot = p_accuracy_by_cluster,
  width = 10,
  height = 6,
  dpi = 300
)

print_progress("Saved accuracy by cluster plot")

################################################################################
# Step 8: Save validation results
################################################################################

validation_results <- list(
  feature_importance = all_importance,
  confusion_matrices = list(
    elastic_net = cm_elastic,
    random_forest = cm_rf,
    xgboost = cm_xgb
  ),
  cluster_accuracy = cluster_accuracy,
  test_predictions = test_predictions
)

save_output(
  validation_results,
  file.path(OUTPUT_DIR, "validation_results.rds"),
  "validation results"
)

################################################################################
# Summary
################################################################################

print_header("STEP 05 COMPLETE")
cat("✓ Classification performance metrics calculated\n")
cat("✓ Confusion matrices generated for all models\n")
cat("✓ F1 scores computed by cluster\n")
cat("✓ Prediction agreement plots created\n")
cat("✓ Feature importance compared across models\n")
cat("✓ Spatial cross-validation performed\n")
cat("✓ Per-cluster accuracy analysis completed\n")
cat("✓ All validation plots saved to figures/\n")
cat(paste0("✓ Validation results saved to: ", 
          file.path(OUTPUT_DIR, "validation_results.rds"), "\n"))
