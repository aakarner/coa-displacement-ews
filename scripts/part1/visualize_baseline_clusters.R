################################################################################
# Part 1 - Visualize the Selected Baseline Clusters
################################################################################
#
# Creates static and interactive maps for the substantively selected amenity
# sensitivity solution from scripts/part1/fit_baseline_clusters.R. Tentative names,
# concern levels, colors, and interpretations are maintained separately in
# config/amenity_cluster_labels_k6.csv so they can be revised transparently.
#
# Outputs:
#   figures/03e_amenity_clusters_tentative.png
#   figures/03e_amenity_clusters_interactive.html
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
  library(htmltools)
  library(htmlwidgets)
  library(leaflet)
  library(readr)
  library(sf)
})

print_header("03e - VISUALIZE AMENITY-AUGMENTED CLUSTERS")

OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
LABEL_FILE <- project_path("config", "amenity_cluster_labels_k6.csv")

required_files <- c(
  file.path(OUTPUT_DIR, "hex_features.rds"),
  file.path(OUTPUT_DIR, "amenity_cluster_assignments.csv"),
  file.path(OUTPUT_DIR, "amenity_cluster_recommendations.csv"),
  LABEL_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(
    "Missing required map input(s): ",
    paste(basename(missing_files), collapse = ", "),
    ". Run scripts/part1/fit_baseline_clusters.R first.",
    call. = FALSE
  )
}

print_progress("Loading selected amenity cluster solution...")
hex_features <- readRDS(file.path(OUTPUT_DIR, "hex_features.rds"))
assignments <- read_csv(
  file.path(OUTPUT_DIR, "amenity_cluster_assignments.csv"),
  show_col_types = FALSE
)
recommendations <- read_csv(
  file.path(OUTPUT_DIR, "amenity_cluster_recommendations.csv"),
  show_col_types = FALSE
)
cluster_labels <- read_csv(LABEL_FILE, show_col_types = FALSE)

selected_k <- recommendations %>%
  filter(
    specification == "amenity_augmented",
    diagnostic == "substantive_selected"
  ) %>%
  pull(recommended_k)

if (length(selected_k) != 1 || !is.finite(selected_k)) {
  stop("Could not identify one selected amenity cluster count.", call. = FALSE)
}

configured_k <- unique(cluster_labels$solution_k)
if (
  length(configured_k) != 1 ||
    selected_k != configured_k ||
    selected_k != EWS_CONFIG$amenity_cluster_k
) {
  stop(
    "The selected amenity solution uses k = ", selected_k,
    ", but the tentative labels are configured for k = ",
    paste(configured_k, collapse = ", "),
    " and shared analysis configuration uses k = ",
    EWS_CONFIG$amenity_cluster_k,
    ". Review and update the configuration before mapping.",
    call. = FALSE
  )
}

required_label_columns <- c(
  "solution_k", "cluster", "tentative_name", "concern_level",
  "map_color", "interpretation", "profile_anchor"
)
missing_label_columns <- setdiff(required_label_columns, names(cluster_labels))
if (length(missing_label_columns) > 0) {
  stop(
    "Cluster label configuration is missing: ",
    paste(missing_label_columns, collapse = ", "),
    call. = FALSE
  )
}

selected_assignments <- assignments %>%
  filter(specification == "amenity_augmented", k == selected_k) %>%
  select(hex_id, cluster)

if (nrow(selected_assignments) == 0) {
  stop("No assignments found for the selected amenity solution.", call. = FALSE)
}

assigned_clusters <- sort(unique(selected_assignments$cluster))
configured_clusters <- sort(unique(cluster_labels$cluster))
if (!identical(assigned_clusters, configured_clusters)) {
  stop(
    "Assigned and configured cluster IDs do not match.",
    call. = FALSE
  )
}

profile_vars <- c(
  "rent_pressure_citywide_index",
  "demographic_vulnerability_index",
  "demolition_pressure_index",
  "eviction_pressure_index",
  "sr_311_pressure_index",
  "ownership_pressure_index",
  "amenity_change_index"
)
missing_profile_vars <- setdiff(profile_vars, names(hex_features))
if (length(missing_profile_vars) > 0) {
  stop(
    "Engineered features are missing map popup values: ",
    paste(missing_profile_vars, collapse = ", "),
    call. = FALSE
  )
}

cluster_profile_means <- hex_features %>%
  st_drop_geometry() %>%
  select(hex_id, all_of(profile_vars)) %>%
  inner_join(selected_assignments, by = "hex_id") %>%
  group_by(cluster) %>%
  summarise(
    across(all_of(profile_vars), ~mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )
anchor_labels <- cluster_labels %>%
  filter(!is.na(profile_anchor), nzchar(profile_anchor))
invalid_anchor_features <- setdiff(
  anchor_labels$profile_anchor,
  profile_vars
)
if (length(invalid_anchor_features) > 0L) {
  stop(
    "Unknown profile anchor(s) in cluster labels: ",
    paste(invalid_anchor_features, collapse = ", "),
    call. = FALSE
  )
}
for (anchor_index in seq_len(nrow(anchor_labels))) {
  anchor_feature <- anchor_labels$profile_anchor[[anchor_index]]
  expected_cluster <- anchor_labels$cluster[[anchor_index]]
  observed_cluster <- cluster_profile_means$cluster[[
    which.max(cluster_profile_means[[anchor_feature]])
  ]]
  if (observed_cluster != expected_cluster) {
    stop(
      "Cluster label anchor mismatch for ",
      anchor_feature,
      ": configured cluster ",
      expected_cluster,
      " but the current profile maximum is cluster ",
      observed_cluster,
      ". Reconcile numeric cluster IDs before mapping.",
      call. = FALSE
    )
  }
}

map_data <- hex_features %>%
  select(hex_id, all_of(profile_vars)) %>%
  inner_join(selected_assignments, by = "hex_id") %>%
  left_join(cluster_labels, by = "cluster") %>%
  mutate(
    cluster_label = paste0(
      "Cluster ", cluster, ": ", tentative_name, "\n", concern_level
    ),
    cluster_label = factor(
      cluster_label,
      levels = paste0(
        "Cluster ", cluster_labels$cluster, ": ",
        cluster_labels$tentative_name, "\n", cluster_labels$concern_level
      )
    )
  )

if (nrow(map_data) != nrow(selected_assignments) ||
    any(is.na(map_data$tentative_name))) {
  stop("Cluster map join did not preserve every selected hex.", call. = FALSE)
}

cluster_levels <- levels(map_data$cluster_label)
cluster_colors <- setNames(cluster_labels$map_color, cluster_levels)

################################################################################
# Static map
################################################################################

print_progress("Creating static cluster map...")
map_background <- hex_features %>% select(hex_id)

p_static <- ggplot() +
  geom_sf(
    data = map_background,
    fill = "#F1F3F5",
    color = "#FFFFFF",
    linewidth = 0.03
  ) +
  geom_sf(
    data = map_data,
    aes(fill = cluster_label),
    color = "#FFFFFF",
    linewidth = 0.05
  ) +
  scale_fill_manual(
    values = cluster_colors,
    name = "Tentative cluster"
  ) +
  coord_sf(datum = NA) +
  labs(
    title = "Amenity-Augmented Displacement Pressure Patterns",
    subtitle = paste0(
      "Substantively selected k = ", selected_k,
      "; ", format(nrow(map_data), big.mark = ","), " eligible hexes"
    ),
    caption = paste0(
      "Names and concern levels are interpretive labels, not an ordinal risk score. ",
      "Grey hexes were outside the shared clustering sample."
    )
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "right",
    legend.key.height = grid::unit(0.72, "cm"),
    legend.text = element_text(size = 8.5),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 10.5),
    plot.caption = element_text(size = 8, hjust = 0),
    plot.margin = margin(10, 10, 10, 10)
  )

static_path <- file.path(FIGURES_DIR, "03e_amenity_clusters_tentative.png")
ggsave(
  static_path,
  p_static,
  width = 12,
  height = 9,
  dpi = 300,
  bg = "white"
)

################################################################################
# Interactive map
################################################################################

print_progress("Creating interactive cluster map...")
interactive_data <- map_data %>%
  st_transform(4326) %>%
  mutate(
    popup_html = paste0(
      "<div style='min-width:260px;line-height:1.35'>",
      "<strong style='font-size:14px'>Cluster ", cluster, ": ",
      tentative_name, "</strong><br>",
      "<strong>Concern:</strong> ", concern_level, "<br>",
      "<span>", interpretation, "</span><hr style='margin:7px 0'>",
      "<strong>Hex ID:</strong> ", hex_id, "<br>",
      "Rent pressure: ", round(rent_pressure_citywide_index, 1), "<br>",
      "Demographic vulnerability: ",
      round(demographic_vulnerability_index, 1), "<br>",
      "Demolition pressure: ", round(demolition_pressure_index, 1), "<br>",
      "Eviction pressure: ", round(eviction_pressure_index, 1), "<br>",
      "311 pressure: ", round(sr_311_pressure_index, 1), "<br>",
      "Ownership pressure: ", round(ownership_pressure_index, 1), "<br>",
      "Amenity pressure: ", round(amenity_change_index, 1),
      "</div>"
    )
  )

interactive_map <- leaflet(
  options = leafletOptions(preferCanvas = TRUE, minZoom = 8)
) %>%
  addProviderTiles(
    providers$CartoDB.Positron,
    options = providerTileOptions(maxZoom = 19)
  )

for (cluster_id in configured_clusters) {
  label_row <- cluster_labels %>% filter(cluster == cluster_id)
  cluster_data <- interactive_data %>% filter(cluster == cluster_id)
  layer_name <- paste0(
    "Cluster ", cluster_id, ": ", label_row$tentative_name
  )

  interactive_map <- interactive_map %>%
    addPolygons(
      data = cluster_data,
      group = layer_name,
      fillColor = label_row$map_color,
      fillOpacity = 0.78,
      color = "#FFFFFF",
      weight = 0.35,
      opacity = 0.8,
      smoothFactor = 0.4,
      popup = ~popup_html,
      highlightOptions = highlightOptions(
        weight = 2,
        color = "#222222",
        fillOpacity = 0.92,
        bringToFront = TRUE
      )
    )
}

layer_names <- paste0(
  "Cluster ", cluster_labels$cluster, ": ", cluster_labels$tentative_name
)
legend_labels <- paste0(
  "<strong>Cluster ", cluster_labels$cluster, "</strong>: ",
  cluster_labels$tentative_name, "<br><span style='font-size:11px'>",
  cluster_labels$concern_level, "</span>"
)

interactive_map <- interactive_map %>%
  addLayersControl(
    overlayGroups = layer_names,
    options = layersControlOptions(collapsed = TRUE)
  ) %>%
  addLegend(
    position = "bottomright",
    colors = cluster_labels$map_color,
    labels = lapply(legend_labels, HTML),
    title = "Tentative displacement patterns",
    opacity = 0.9
  ) %>%
  addScaleBar(
    position = "bottomleft",
    options = scaleBarOptions(metric = TRUE, imperial = FALSE)
  ) %>%
  addControl(
    html = HTML(
      paste0(
        "<div style='background:rgba(255,255,255,.94);padding:8px 10px;",
        "border:1px solid #bbb;max-width:300px'>",
        "<strong>Amenity-Augmented Clusters</strong><br>",
        "<span style='font-size:11px'>Tentative k = ", selected_k,
        " pattern names and concern levels</span></div>"
      )
    ),
    position = "topright"
  )

interactive_path <- file.path(
  FIGURES_DIR,
  "03e_amenity_clusters_interactive.html"
)
saveWidget(
  interactive_map,
  file = interactive_path,
  selfcontained = TRUE,
  title = "Amenity-Augmented Displacement Pressure Clusters"
)

cat("\nStatic map: ", static_path, "\n", sep = "")
cat("Interactive map: ", interactive_path, "\n", sep = "")
cat("03e visualization complete.\n")
