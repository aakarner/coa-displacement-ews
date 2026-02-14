################################################################################
# 07 - Visualize Displacement Risk Results
################################################################################
#
# This script creates comprehensive visualizations of the displacement risk
# analysis results, including:
# - Interactive maps of risk scores (with cluster information)
# - Static publication-quality maps
# - Feature importance plots
# - Model comparison visualizations
# - Summary dashboards
#
# NOTE: Risk scores are now based on cluster-based predictions rather than
# synthetic composite scores. Interpretation: "This area resembles high-risk
# displacement clusters" rather than an arbitrary composite score.
#
################################################################################

print_header("07 - VISUALIZING DISPLACEMENT RISK RESULTS")

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

# Configuration
OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")

################################################################################
# Step 1: Load risk scores and supporting data
################################################################################

print_progress("Loading risk scores and validation results...")

risk_scores <- load_output(
  file.path(OUTPUT_DIR, "displacement_risk_scores.rds"),
  "displacement risk scores"
)

validation_results <- load_output(
  file.path(OUTPUT_DIR, "validation_results.rds"),
  "validation results"
)

# Load Austin boundary for context
austin_boundary <- tigris::places(state = "TX", year = 2021) %>%
  filter(NAME == "Austin") %>%
  st_transform(4326)

################################################################################
# Step 2: Create static risk map
################################################################################

print_header("STATIC RISK MAPS")

print_progress("Creating static publication-quality risk map...")

# Main risk map
p_risk_map <- ggplot() +
  # Add Austin boundary
  geom_sf(data = austin_boundary, fill = NA, color = "black", linewidth = 0.8) +
  # Add risk scores
  geom_sf(data = risk_scores, 
          aes(fill = risk_score_ensemble), 
          color = NA) +
  # Use color scale appropriate for risk
  scale_fill_viridis_c(
    option = "rocket",
    name = "Displacement\nRisk Score",
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    labels = c("0\n(Low)", "25", "50", "75", "100\n(High)"),
    direction = -1
  ) +
  labs(
    title = "Displacement Risk in Austin, TX",
    subtitle = "Ensemble prediction from Random Forest, XGBoost, and Elastic Net models",
    caption = "Risk score: 0-100 scale based on rent increases, demolitions, and socioeconomic vulnerability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11),
    legend.position = "right",
    axis.title = element_blank(),
    panel.grid = element_line(color = "gray90", linewidth = 0.2)
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_displacement_risk_map.png"),
  plot = p_risk_map,
  width = 12,
  height = 10,
  dpi = 300
)

print_progress("Saved main risk map")

# Categorical risk map
p_risk_categories <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "black", linewidth = 0.8) +
  geom_sf(data = risk_scores, 
          aes(fill = risk_category), 
          color = NA) +
  scale_fill_manual(
    name = "Risk Category",
    values = c(
      "Low" = "#440154",
      "Moderate" = "#31688e", 
      "High" = "#35b779",
      "Very High" = "#fde724"
    ),
    na.value = "gray90"
  ) +
  labs(
    title = "Displacement Risk Categories in Austin, TX",
    subtitle = "Classification based on ensemble risk scores",
    caption = "Low: 0-25 | Moderate: 25-50 | High: 50-75 | Very High: 75-100"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11),
    legend.position = "right",
    axis.title = element_blank(),
    panel.grid = element_line(color = "gray90", linewidth = 0.2)
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_risk_categories_map.png"),
  plot = p_risk_categories,
  width = 12,
  height = 10,
  dpi = 300
)

print_progress("Saved categorical risk map")

################################################################################
# Step 3: Create interactive maps
################################################################################

print_header("INTERACTIVE MAPS")

print_progress("Creating interactive Leaflet map...")

# Prepare data for popup
risk_scores_for_map <- risk_scores %>%
  mutate(
    popup_text = paste0(
      "<b>Hex ID:</b> ", hex_id, "<br>",
      "<b>Risk Score:</b> ", round(risk_score_ensemble, 1), "<br>",
      "<b>Risk Category:</b> ", risk_category, "<br>",
      "<b>Contributing Factors:</b><br>", 
      gsub("; ", "<br>", contributing_factors)
    )
  )

# Create color palette
pal <- colorNumeric(
  palette = "YlOrRd",
  domain = c(0, 100),
  na.color = "transparent"
)

# Create interactive map
interactive_map <- leaflet(risk_scores_for_map) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  addPolygons(
    fillColor = ~pal(risk_score_ensemble),
    fillOpacity = 0.7,
    color = "white",
    weight = 0.5,
    popup = ~popup_text,
    highlightOptions = highlightOptions(
      weight = 2,
      color = "#666",
      fillOpacity = 0.9,
      bringToFront = TRUE
    )
  ) %>%
  addLegend(
    "bottomright",
    pal = pal,
    values = ~risk_score_ensemble,
    title = "Displacement<br>Risk Score",
    opacity = 1
  ) %>%
  addScaleBar(position = "bottomleft")

# Save interactive map
htmlwidgets::saveWidget(
  interactive_map,
  file = file.path(FIGURES_DIR, "07_interactive_risk_map.html"),
  selfcontained = TRUE
)

print_progress("Saved interactive map")

################################################################################
# Step 4: Model comparison maps
################################################################################

print_header("MODEL COMPARISON MAPS")

print_progress("Creating side-by-side model comparison...")

# Random Forest map
p_rf <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "black", linewidth = 0.5) +
  geom_sf(data = risk_scores, aes(fill = risk_score_rf), color = NA) +
  scale_fill_viridis_c(option = "rocket", limits = c(0, 100), direction = -1) +
  labs(title = "Random Forest", fill = "Risk\nScore") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid = element_blank()
  )

# XGBoost map
p_xgb <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "black", linewidth = 0.5) +
  geom_sf(data = risk_scores, aes(fill = risk_score_xgb), color = NA) +
  scale_fill_viridis_c(option = "rocket", limits = c(0, 100), direction = -1) +
  labs(title = "XGBoost", fill = "Risk\nScore") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid = element_blank()
  )

# Ensemble map
p_ensemble <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "black", linewidth = 0.5) +
  geom_sf(data = risk_scores, aes(fill = risk_score_ensemble), color = NA) +
  scale_fill_viridis_c(option = "rocket", limits = c(0, 100), direction = -1) +
  labs(title = "Ensemble (Weighted Average)", fill = "Risk\nScore") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
    axis.title = element_blank(),
    axis.text = element_blank(),
    panel.grid = element_blank()
  )

# Combine
p_model_comparison <- (p_rf | p_xgb | p_ensemble) +
  plot_annotation(
    title = "Model Comparison: Displacement Risk Predictions",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_model_comparison_maps.png"),
  plot = p_model_comparison,
  width = 18,
  height = 6,
  dpi = 300
)

print_progress("Saved model comparison maps")

################################################################################
# Step 5: Feature importance visualization
################################################################################

print_header("FEATURE IMPORTANCE VISUALIZATIONS")

print_progress("Creating comprehensive feature importance plots...")

feature_importance <- validation_results$feature_importance

# Overall importance (average across models)
p_importance_overall <- feature_importance %>%
  head(20) %>%
  mutate(feature = factor(feature, levels = rev(feature))) %>%
  ggplot(aes(x = importance_avg, y = feature)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  labs(
    title = "Top 20 Most Important Features",
    subtitle = "Average importance across all models",
    x = "Importance Score (0-100)",
    y = NULL
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    axis.text.y = element_text(size = 9)
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_feature_importance_overall.png"),
  plot = p_importance_overall,
  width = 10,
  height = 8,
  dpi = 300
)

print_progress("Saved overall feature importance plot")

################################################################################
# Step 6: Risk distribution plots
################################################################################

print_header("RISK DISTRIBUTION PLOTS")

print_progress("Creating risk distribution visualizations...")

# Histogram of risk scores
p_risk_hist <- ggplot(st_drop_geometry(risk_scores), 
                     aes(x = risk_score_ensemble)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.8) +
  geom_vline(xintercept = c(25, 50, 75), 
             linetype = "dashed", color = "red", linewidth = 0.8) +
  labs(
    title = "Distribution of Displacement Risk Scores",
    subtitle = "Vertical lines indicate category boundaries",
    x = "Risk Score",
    y = "Number of Hexagonal Cells"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave(
  filename = file.path(FIGURES_DIR, "07_risk_score_distribution.png"),
  plot = p_risk_hist,
  width = 10,
  height = 6,
  dpi = 300
)

# Pie chart of risk categories
risk_category_counts <- st_drop_geometry(risk_scores) %>%
  count(risk_category) %>%
  mutate(
    percentage = round(n / sum(n) * 100, 1),
    label = paste0(risk_category, "\n", percentage, "%")
  )

p_risk_pie <- ggplot(risk_category_counts, 
                    aes(x = "", y = n, fill = risk_category)) +
  geom_col(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  scale_fill_manual(
    values = c("Low" = "#440154", "Moderate" = "#31688e", 
              "High" = "#35b779", "Very High" = "#fde724")
  ) +
  geom_text(aes(label = label), 
           position = position_stack(vjust = 0.5),
           color = "white", fontface = "bold", size = 4) +
  labs(
    title = "Distribution of Risk Categories",
    fill = "Category"
  ) +
  theme_void() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    legend.position = "none"
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_risk_categories_pie.png"),
  plot = p_risk_pie,
  width = 8,
  height = 8,
  dpi = 300
)

print_progress("Saved distribution plots")

################################################################################
# Step 7: Summary dashboard
################################################################################

print_header("SUMMARY DASHBOARD")

print_progress("Creating summary dashboard...")

# Create text summaries
total_cells <- nrow(risk_scores)
high_very_high <- sum(risk_scores$risk_category %in% c("High", "Very High"), na.rm = TRUE)
pct_high_risk <- round(high_very_high / total_cells * 100, 1)

summary_text <- data.frame(
  metric = c("Total Hexagonal Cells", 
            "High/Very High Risk",
            "% High/Very High Risk",
            "Mean Risk Score",
            "Median Risk Score"),
  value = c(total_cells,
           high_very_high,
           paste0(pct_high_risk, "%"),
           round(mean(risk_scores$risk_score_ensemble, na.rm = TRUE), 1),
           round(median(risk_scores$risk_score_ensemble, na.rm = TRUE), 1))
)

# Create summary table plot
p_summary_table <- ggplot(summary_text, aes(x = 1, y = seq(nrow(summary_text), 1))) +
  geom_text(aes(label = metric), x = 0.5, hjust = 1, fontface = "bold", size = 5) +
  geom_text(aes(label = value), x = 1.5, hjust = 0, size = 5, color = "steelblue") +
  xlim(0, 3) +
  labs(title = "Summary Statistics") +
  theme_void() +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.margin = margin(20, 20, 20, 20)
  )

# Combine into dashboard
dashboard <- (p_risk_categories | p_risk_pie) /
             (p_importance_overall | p_summary_table) +
  plot_annotation(
    title = "Displacement Risk Early Warning System - Summary Dashboard",
    subtitle = "Austin, TX",
    theme = theme(
      plot.title = element_text(face = "bold", size = 18),
      plot.subtitle = element_text(size = 12)
    )
  )

ggsave(
  filename = file.path(FIGURES_DIR, "07_summary_dashboard.png"),
  plot = dashboard,
  width = 16,
  height = 12,
  dpi = 300
)

print_progress("Saved summary dashboard")

################################################################################
# Step 8: Create visualization index
################################################################################

print_progress("Creating visualization index...")

viz_index <- data.frame(
  filename = c(
    "07_displacement_risk_map.png",
    "07_risk_categories_map.png",
    "07_interactive_risk_map.html",
    "07_model_comparison_maps.png",
    "07_feature_importance_overall.png",
    "07_risk_score_distribution.png",
    "07_risk_categories_pie.png",
    "07_summary_dashboard.png"
  ),
  description = c(
    "Main displacement risk map (continuous scale)",
    "Risk categories map (Low/Moderate/High/Very High)",
    "Interactive map with popups",
    "Comparison of Random Forest, XGBoost, and Ensemble predictions",
    "Top 20 most important features",
    "Histogram of risk score distribution",
    "Pie chart of risk categories",
    "Combined summary dashboard"
  ),
  type = c(
    "Map - Static",
    "Map - Static",
    "Map - Interactive",
    "Map - Comparison",
    "Feature Importance",
    "Distribution",
    "Distribution",
    "Dashboard"
  )
)

write_csv(viz_index, file.path(FIGURES_DIR, "visualization_index.csv"))
print_progress("Saved visualization index")

################################################################################
# Summary
################################################################################

print_header("STEP 07 COMPLETE")
cat("✓ Static risk maps created\n")
cat("✓ Interactive map generated\n")
cat("✓ Model comparison visualizations created\n")
cat("✓ Feature importance plots generated\n")
cat("✓ Distribution plots created\n")
cat("✓ Summary dashboard assembled\n")
cat("\nAll visualizations saved to figures/\n")
cat("See visualization_index.csv for full list of outputs\n")
