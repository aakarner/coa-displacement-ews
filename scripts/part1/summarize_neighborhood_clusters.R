################################################################################
# Part 1 - Summarize Baseline Clusters by Neighborhood Reporting Area
################################################################################
#
# Assigns each H3 cell to the City Neighborhood Reporting Area containing its
# center, then summarizes cluster composition with allocated population and
# promoted residential-unit weights. The mapped neighborhood category is the
# cluster containing the plurality of classified population. A separate flag
# records whether that plurality exceeds 50 percent.
#
# Outputs:
#   output/part1/neighborhood_cluster_composition.csv
#   output/part1/neighborhood_cluster_summary.csv
#   output/part1/neighborhood_cluster_coverage.csv
#   output/part1/neighborhood_cluster_summary.rds
#   figures/03g_neighborhood_cluster_plurality.png
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
  library(jsonlite)
  library(readr)
  library(sf)
  library(tidyr)
})

print_header("PART 1 - NEIGHBORHOOD CLUSTER SUMMARY")

DATA_DIR <- project_path("data")
OUTPUT_DIR <- project_path("output")
PART1_DIR <- file.path(OUTPUT_DIR, "part1")
FIGURES_DIR <- project_path("figures")

NEIGHBORHOOD_FILE <- file.path(
  DATA_DIR,
  "neighborhood_reporting_areas.geojson"
)
NEIGHBORHOOD_METADATA_FILE <- file.path(
  DATA_DIR,
  "neighborhood_reporting_areas_metadata.json"
)
JURISDICTION_FILE <- file.path(
  DATA_DIR,
  "BOUNDARIES_jurisdictions_20260429.geojson"
)
FEATURE_FILE <- file.path(OUTPUT_DIR, "hex_features.rds")
ASSIGNMENT_FILE <- file.path(
  PART1_DIR,
  "baseline_cluster_assignments.csv"
)
LABEL_FILE <- project_path("config", "amenity_cluster_labels.csv")
ORIENTATION_FILE <- file.path(OUTPUT_DIR, "map_orientation_reference.rds")

required_files <- c(
  NEIGHBORHOOD_FILE,
  NEIGHBORHOOD_METADATA_FILE,
  JURISDICTION_FILE,
  FEATURE_FILE,
  ASSIGNMENT_FILE,
  LABEL_FILE,
  ORIENTATION_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing neighborhood-summary input(s): ",
    paste(basename(missing_files), collapse = ", "),
    ". Refresh the reporting areas with ",
    "scripts/data/download_neighborhood_reporting_areas.R and run Part 1.",
    call. = FALSE
  )
}

dir.create(PART1_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

require_columns <- function(data, columns, description) {
  missing <- setdiff(columns, names(data))
  if (length(missing) > 0L) {
    stop(
      description, " is missing: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
}

safe_share <- function(numerator, denominator) {
  ifelse(denominator > 0, numerator / denominator, NA_real_)
}

print_progress("Loading reporting areas and baseline assignments...")
reporting_areas_raw <- st_read(NEIGHBORHOOD_FILE, quiet = TRUE)
invalid_reporting_geometries <- sum(!st_is_valid(reporting_areas_raw))
reporting_areas <- reporting_areas_raw %>%
  transmute(
    neighborhood_name = trimws(as.character(neighname))
  ) %>%
  st_make_valid()

if (
  any(is.na(reporting_areas$neighborhood_name)) ||
    any(!nzchar(reporting_areas$neighborhood_name)) ||
    anyDuplicated(reporting_areas$neighborhood_name)
) {
  stop("Reporting-area names must be unique and nonmissing.", call. = FALSE)
}
if (any(!st_is_valid(reporting_areas)) || any(st_is_empty(reporting_areas))) {
  stop("Reporting-area geometry repair failed.", call. = FALSE)
}

metadata <- fromJSON(NEIGHBORHOOD_METADATA_FILE)
if (!identical(metadata$id, "a7ap-j2yt")) {
  stop("Unexpected Neighborhood Reporting Area dataset ID.", call. = FALSE)
}
source_updated_at <- format(
  as.POSIXct(metadata$rowsUpdatedAt, origin = "1970-01-01", tz = "UTC"),
  "%Y-%m-%dT%H:%M:%SZ",
  tz = "UTC"
)

jurisdictions <- st_read(JURISDICTION_FILE, quiet = TRUE)
require_columns(
  jurisdictions,
  c("jurisdiction_label"),
  "Jurisdiction boundary"
)
full_purpose_parts <- jurisdictions %>%
  filter(jurisdiction_label == "AUSTIN FULL PURPOSE") %>%
  st_make_valid()
if (nrow(full_purpose_parts) == 0L) {
  stop("Could not identify Austin full-purpose jurisdiction geometry.", call. = FALSE)
}
full_purpose <- st_union(full_purpose_parts)

hex_features <- readRDS(FEATURE_FILE)
require_columns(
  hex_features,
  c(
    "hex_id", "longitude", "latitude", "total_pop", "residential_units",
    "analysis_as_of_date"
  ),
  "Hex feature surface"
)
assignments <- read_csv(ASSIGNMENT_FILE, show_col_types = FALSE)
cluster_labels <- read_csv(LABEL_FILE, show_col_types = FALSE) %>%
  arrange(display_cluster)
require_columns(assignments, c("hex_id", "cluster"), "Baseline assignments")
require_columns(
  cluster_labels,
  c(
    "solution_k", "cluster", "display_cluster", "tentative_name",
    "concern_level", "map_color"
  ),
  "Cluster labels"
)
if (anyDuplicated(assignments$hex_id) || anyDuplicated(cluster_labels$cluster)) {
  stop("Assignments and cluster labels must have unique IDs.", call. = FALSE)
}
if (
  length(unique(cluster_labels$solution_k)) != 1L ||
    unique(cluster_labels$solution_k) != EWS_CONFIG$amenity_cluster_k
) {
  stop("Neighborhood labels do not match the configured cluster count.", call. = FALSE)
}

assignment_labels <- assignments %>%
  select(hex_id, cluster) %>%
  inner_join(
    cluster_labels %>%
      select(
        cluster,
        display_cluster,
        tentative_name,
        concern_level,
        map_color
      ),
    by = "cluster",
    relationship = "many-to-one"
  )
if (nrow(assignment_labels) != nrow(assignments)) {
  stop("A baseline assignment lacks a configured cluster label.", call. = FALSE)
}

hex_data <- hex_features %>%
  st_drop_geometry() %>%
  transmute(
    hex_id,
    longitude,
    latitude,
    total_population = coalesce(as.numeric(total_pop), 0),
    housing_units = coalesce(as.numeric(residential_units), 0)
  ) %>%
  left_join(assignment_labels, by = "hex_id", relationship = "one-to-one")
if (
  any(!is.finite(hex_data$total_population)) ||
    any(!is.finite(hex_data$housing_units)) ||
    any(hex_data$total_population < 0) ||
    any(hex_data$housing_units < 0)
) {
  stop("Neighborhood weights must be finite and nonnegative.", call. = FALSE)
}

hex_points <- st_as_sf(
  hex_data,
  coords = c("longitude", "latitude"),
  crs = 4326,
  remove = FALSE
)
inside_full_purpose <- lengths(st_within(hex_points, full_purpose)) > 0L
analysis_points <- hex_points[inside_full_purpose, ]

reporting_hits <- st_within(analysis_points, reporting_areas)
if (any(lengths(reporting_hits) > 1L)) {
  stop("A hex center falls within multiple reporting areas.", call. = FALSE)
}
reporting_index <- vapply(
  reporting_hits,
  function(index) if (length(index) == 1L) index[[1]] else NA_integer_,
  integer(1)
)
analysis_points$neighborhood_name <- reporting_areas$neighborhood_name[
  reporting_index
]
analysis_points$inside_reporting_area <- !is.na(
  analysis_points$neighborhood_name
)
analysis_points$neighborhood_name <- coalesce(
  analysis_points$neighborhood_name,
  "Outside reporting areas"
)
analysis_points$category_id <- if_else(
  is.na(analysis_points$display_cluster),
  "unclassified",
  paste0("cluster_", analysis_points$display_cluster)
)

map_crs <- 2277
full_purpose_projected <- st_transform(full_purpose, map_crs)
reporting_areas_projected <- st_transform(reporting_areas, map_crs)
reporting_map <- suppressWarnings(
  st_intersection(reporting_areas_projected, full_purpose_projected)
) %>%
  filter(!st_is_empty(geometry)) %>%
  st_make_valid()

category_lookup <- bind_rows(
  cluster_labels %>%
    transmute(
      category_id = paste0("cluster_", display_cluster),
      cluster,
      display_cluster,
      tentative_name,
      concern_level,
      map_color,
      category_label = paste0("Cluster ", display_cluster, ": ", tentative_name),
      category_order = display_cluster
    ),
  tibble(
    category_id = "unclassified",
    cluster = NA_integer_,
    display_cluster = NA_integer_,
    tentative_name = "Unclassified",
    concern_level = NA_character_,
    map_color = "#D5D9DC",
    category_label = "Unclassified",
    category_order = nrow(cluster_labels) + 1L
  )
)

neighborhood_names <- c(
  sort(unique(reporting_map$neighborhood_name)),
  "Outside reporting areas"
)
composition_grid <- crossing(
  neighborhood_name = neighborhood_names,
  category_id = category_lookup$category_id
)
composition_counts <- analysis_points %>%
  st_drop_geometry() %>%
  group_by(neighborhood_name, category_id) %>%
  summarise(
    hex_count = n(),
    population = sum(total_population),
    housing_units = sum(housing_units),
    .groups = "drop"
  )
neighborhood_totals <- analysis_points %>%
  st_drop_geometry() %>%
  group_by(neighborhood_name) %>%
  summarise(
    total_hexes = n(),
    classified_population = sum(total_population[!is.na(display_cluster)]),
    classified_housing_units = sum(housing_units[!is.na(display_cluster)]),
    total_population = sum(total_population),
    total_housing_units = sum(housing_units),
    .groups = "drop"
  )

composition <- composition_grid %>%
  left_join(
    composition_counts,
    by = c("neighborhood_name", "category_id")
  ) %>%
  mutate(
    across(c(hex_count, population, housing_units), ~coalesce(.x, 0))
  ) %>%
  left_join(category_lookup, by = "category_id", relationship = "many-to-one") %>%
  left_join(neighborhood_totals, by = "neighborhood_name") %>%
  mutate(
    across(
      c(
        total_hexes,
        total_population,
        total_housing_units,
        classified_population,
        classified_housing_units
      ),
      ~coalesce(.x, 0)
    ),
    population_share_total = safe_share(population, total_population),
    housing_unit_share_total = safe_share(housing_units, total_housing_units),
    population_share_classified = if_else(
      !is.na(display_cluster),
      safe_share(population, classified_population),
      NA_real_
    ),
    housing_unit_share_classified = if_else(
      !is.na(display_cluster),
      safe_share(housing_units, classified_housing_units),
      NA_real_
    ),
    classified_population_share = safe_share(
      classified_population,
      total_population
    ),
    classified_housing_unit_share = safe_share(
      classified_housing_units,
      total_housing_units
    ),
    inside_reporting_area = neighborhood_name != "Outside reporting areas"
  ) %>%
  arrange(inside_reporting_area, neighborhood_name, category_order)

cluster_composition <- composition %>% filter(!is.na(display_cluster))
population_top <- cluster_composition %>%
  group_by(neighborhood_name) %>%
  arrange(desc(population), display_cluster, .by_group = TRUE) %>%
  mutate(
    population_rank = row_number(),
    population_tie_count = sum(abs(population - max(population)) < 1e-9)
  ) %>%
  filter(population_rank == 1L) %>%
  ungroup() %>%
  transmute(
    neighborhood_name,
    population_plurality_cluster = cluster,
    population_plurality_display_cluster = display_cluster,
    population_plurality_name = tentative_name,
    population_plurality_concern_level = concern_level,
    population_plurality_color = map_color,
    population_plurality = population,
    population_plurality_share_total = population_share_total,
    population_plurality_share_classified = population_share_classified,
    population_plurality_tie = population_tie_count > 1L
  )
housing_top <- cluster_composition %>%
  group_by(neighborhood_name) %>%
  arrange(desc(housing_units), display_cluster, .by_group = TRUE) %>%
  mutate(
    housing_rank = row_number(),
    housing_tie_count = sum(abs(housing_units - max(housing_units)) < 1e-9)
  ) %>%
  filter(housing_rank == 1L) %>%
  ungroup() %>%
  transmute(
    neighborhood_name,
    housing_plurality_cluster = cluster,
    housing_plurality_display_cluster = display_cluster,
    housing_plurality_name = tentative_name,
    housing_plurality_concern_level = concern_level,
    housing_plurality = housing_units,
    housing_plurality_share_total = housing_unit_share_total,
    housing_plurality_share_classified = housing_unit_share_classified,
    housing_plurality_tie = housing_tie_count > 1L
  )

cluster_wide <- cluster_composition %>%
  select(
    neighborhood_name,
    display_cluster,
    population,
    housing_units,
    population_share_classified,
    housing_unit_share_classified
  ) %>%
  pivot_wider(
    names_from = display_cluster,
    values_from = c(
      population,
      housing_units,
      population_share_classified,
      housing_unit_share_classified
    ),
    names_glue = "cluster_{display_cluster}_{.value}"
  )

neighborhood_summary <- neighborhood_totals %>%
  right_join(
    tibble(neighborhood_name = neighborhood_names),
    by = "neighborhood_name"
  ) %>%
  mutate(
    across(
      c(
        total_hexes,
        total_population,
        total_housing_units,
        classified_population,
        classified_housing_units
      ),
      ~coalesce(.x, 0)
    ),
    classified_population_share = safe_share(
      classified_population,
      total_population
    ),
    classified_housing_unit_share = safe_share(
      classified_housing_units,
      total_housing_units
    )
  ) %>%
  left_join(population_top, by = "neighborhood_name") %>%
  left_join(housing_top, by = "neighborhood_name") %>%
  left_join(cluster_wide, by = "neighborhood_name") %>%
  mutate(
    population_plurality_cluster = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_cluster,
      NA_integer_
    ),
    population_plurality_display_cluster = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_display_cluster,
      NA_integer_
    ),
    population_plurality_name = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_name,
      NA_character_
    ),
    population_plurality_concern_level = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_concern_level,
      NA_character_
    ),
    population_plurality_color = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_color,
      NA_character_
    ),
    population_plurality = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality,
      NA_real_
    ),
    population_plurality_share_total = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_share_total,
      NA_real_
    ),
    population_plurality_share_classified = if_else(
      classified_population > 0 & !population_plurality_tie,
      population_plurality_share_classified,
      NA_real_
    ),
    population_plurality_type = case_when(
      classified_population <= 0 ~ "No classified population",
      population_plurality_tie ~ "Tied plurality",
      population_plurality_share_classified > 0.5 ~ "Majority",
      TRUE ~ "Plurality"
    ),
    housing_plurality_cluster = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_cluster,
      NA_integer_
    ),
    housing_plurality_display_cluster = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_display_cluster,
      NA_integer_
    ),
    housing_plurality_name = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_name,
      NA_character_
    ),
    housing_plurality_concern_level = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_concern_level,
      NA_character_
    ),
    housing_plurality = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality,
      NA_real_
    ),
    housing_plurality_share_total = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_share_total,
      NA_real_
    ),
    housing_plurality_share_classified = if_else(
      classified_housing_units > 0 & !housing_plurality_tie,
      housing_plurality_share_classified,
      NA_real_
    ),
    housing_plurality_type = case_when(
      classified_housing_units <= 0 ~ "No classified housing units",
      housing_plurality_tie ~ "Tied plurality",
      housing_plurality_share_classified > 0.5 ~ "Majority",
      TRUE ~ "Plurality"
    ),
    population_housing_plurality_agree = if_else(
      !is.na(population_plurality_display_cluster) &
        !is.na(housing_plurality_display_cluster),
      population_plurality_display_cluster == housing_plurality_display_cluster,
      NA
    ),
    inside_reporting_area = neighborhood_name != "Outside reporting areas"
  ) %>%
  arrange(inside_reporting_area, neighborhood_name)

population_share_check <- cluster_composition %>%
  filter(classified_population > 0) %>%
  group_by(neighborhood_name) %>%
  summarise(share = sum(population_share_classified), .groups = "drop")
unit_share_check <- cluster_composition %>%
  filter(classified_housing_units > 0) %>%
  group_by(neighborhood_name) %>%
  summarise(share = sum(housing_unit_share_classified), .groups = "drop")
if (
  any(abs(population_share_check$share - 1) > 1e-8) ||
    any(abs(unit_share_check$share - 1) > 1e-8)
) {
  stop("Neighborhood cluster shares do not sum to one.", call. = FALSE)
}

analysis_as_of <- unique(as.Date(hex_features$analysis_as_of_date))
if (length(analysis_as_of) != 1L || is.na(analysis_as_of)) {
  stop("Hex features must contain one analysis cutoff.", call. = FALSE)
}
matched <- analysis_points$inside_reporting_area
classified <- !is.na(analysis_points$display_cluster)
coverage <- tibble(
  source_dataset_id = metadata$id,
  source_rows_updated_at = source_updated_at,
  source_feature_count = nrow(reporting_areas_raw),
  source_invalid_geometry_count = invalid_reporting_geometries,
  source_md5 = unname(tools::md5sum(NEIGHBORHOOD_FILE)),
  analysis_as_of_date = analysis_as_of,
  assignment_method = "H3 center within reporting area",
  full_purpose_hexes = nrow(analysis_points),
  reporting_area_hexes = sum(matched),
  reporting_area_hex_share = mean(matched),
  full_purpose_population = sum(analysis_points$total_population),
  reporting_area_population = sum(analysis_points$total_population[matched]),
  reporting_area_population_share = safe_share(
    reporting_area_population,
    full_purpose_population
  ),
  full_purpose_housing_units = sum(analysis_points$housing_units),
  reporting_area_housing_units = sum(analysis_points$housing_units[matched]),
  reporting_area_housing_unit_share = safe_share(
    reporting_area_housing_units,
    full_purpose_housing_units
  ),
  classified_hexes = sum(classified),
  reporting_area_classified_hexes = sum(matched & classified),
  classified_population = sum(analysis_points$total_population[classified]),
  reporting_area_classified_population = sum(
    analysis_points$total_population[matched & classified]
  ),
  reporting_area_classified_population_share = safe_share(
    reporting_area_classified_population,
    classified_population
  ),
  classified_housing_units = sum(analysis_points$housing_units[classified]),
  reporting_area_classified_housing_units = sum(
    analysis_points$housing_units[matched & classified]
  ),
  reporting_area_classified_housing_unit_share = safe_share(
    reporting_area_classified_housing_units,
    classified_housing_units
  )
)
if (
  coverage$reporting_area_classified_population_share < 0.9 ||
    coverage$reporting_area_classified_housing_unit_share < 0.9
) {
  stop("Neighborhood Reporting Areas cover less than 90% of Part 1 support.")
}

write_csv(
  composition,
  file.path(PART1_DIR, "neighborhood_cluster_composition.csv")
)
write_csv(
  neighborhood_summary,
  file.path(PART1_DIR, "neighborhood_cluster_summary.csv")
)
write_csv(
  coverage,
  file.path(PART1_DIR, "neighborhood_cluster_coverage.csv")
)

map_data <- reporting_map %>%
  left_join(neighborhood_summary, by = "neighborhood_name") %>%
  mutate(
    map_cluster_label = if_else(
      is.na(population_plurality_display_cluster),
      "No classified population",
      paste0(
        "Cluster ", population_plurality_display_cluster, ": ",
        population_plurality_name
      )
    )
  )
saveRDS(
  map_data,
  file.path(PART1_DIR, "neighborhood_cluster_summary.rds")
)

print_progress("Creating neighborhood plurality map...")
cluster_map_labels <- paste0(
  "Cluster ", cluster_labels$display_cluster, ": ",
  cluster_labels$tentative_name
)
map_levels <- c(cluster_map_labels, "No classified population")
map_colors <- setNames(
  c(cluster_labels$map_color, "#D5D9DC"),
  map_levels
)
map_data$map_cluster_label <- factor(
  map_data$map_cluster_label,
  levels = map_levels
)

orientation_reference <- readRDS(ORIENTATION_FILE)
if (
  !inherits(orientation_reference$roads, "sf") ||
    !inherits(orientation_reference$water, "sf")
) {
  stop("Map orientation reference is incomplete.", call. = FALSE)
}
map_bbox <- st_bbox(map_data)
water <- suppressWarnings(
  st_crop(st_transform(orientation_reference$water, map_crs), map_bbox)
)
roads <- suppressWarnings(
  st_crop(st_transform(orientation_reference$roads, map_crs), map_bbox)
)

majority_count <- sum(
  map_data$population_plurality_type == "Majority",
  na.rm = TRUE
)
classified_map_count <- sum(
  !is.na(map_data$population_plurality_display_cluster)
)
analysis_cutoff_label <- format(analysis_as_of, "%B %d, %Y")

p_map <- ggplot() +
  geom_sf(
    data = map_data,
    aes(fill = map_cluster_label),
    color = "#FFFFFF",
    linewidth = 0.28
  ) +
  geom_sf(
    data = map_data %>% filter(population_plurality_type == "Majority"),
    fill = NA,
    color = "#20262B",
    linewidth = 0.62
  ) +
  geom_sf(
    data = water,
    fill = "#8FC7E3",
    color = "#4F86A0",
    linewidth = 0.16,
    alpha = 0.82
  ) +
  geom_sf(
    data = roads,
    color = "#FFFFFF",
    linewidth = 0.52,
    alpha = 0.82
  ) +
  geom_sf(
    data = roads,
    color = "#48545D",
    linewidth = 0.18,
    alpha = 0.68
  ) +
  scale_fill_manual(
    values = map_colors,
    drop = FALSE,
    name = "Population-plurality cluster"
  ) +
  coord_sf(
    xlim = c(map_bbox[["xmin"]], map_bbox[["xmax"]]),
    ylim = c(map_bbox[["ymin"]], map_bbox[["ymax"]]),
    datum = NA,
    expand = FALSE
  ) +
  labs(
    title = "Baseline Cluster Composition by Neighborhood",
    subtitle = paste0(
      "Color shows the cluster containing the largest share of classified ",
      "allocated population | ", majority_count, " of ",
      classified_map_count, " mapped neighborhoods have a majority | cutoff ",
      analysis_cutoff_label
    ),
    caption = paste0(
      "Neighborhoods are City of Austin Reporting Areas clipped to the ",
      "full-purpose boundary. Dark outlines indicate a cluster majority ",
      "(more than 50% of classified population).\n",
      "Unclassified population is excluded from plurality selection and ",
      "reported separately in the companion tables."
    )
  ) +
  theme_void(base_size = 11) +
  theme(
    legend.position = "right",
    legend.key.height = grid::unit(0.7, "cm"),
    legend.text = element_text(size = 8.2),
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 9.8),
    plot.caption = element_text(size = 8, hjust = 0),
    plot.margin = margin(10, 10, 10, 10)
  )

map_path <- file.path(
  FIGURES_DIR,
  "03g_neighborhood_cluster_plurality.png"
)
ggsave(
  map_path,
  p_map,
  width = 13,
  height = 9,
  dpi = 300,
  bg = "white"
)

print_progress(
  paste0(
    "Reporting areas contain ",
    scales::percent(
      coverage$reporting_area_classified_population_share,
      accuracy = 0.1
    ),
    " of classified population and ",
    scales::percent(
      coverage$reporting_area_classified_housing_unit_share,
      accuracy = 0.1
    ),
    " of classified housing units."
  )
)
cat("Neighborhood composition: output/part1/neighborhood_cluster_composition.csv\n")
cat("Neighborhood summary: output/part1/neighborhood_cluster_summary.csv\n")
cat("Neighborhood map: figures/03g_neighborhood_cluster_plurality.png\n")
