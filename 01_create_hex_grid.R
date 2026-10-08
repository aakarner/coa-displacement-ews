################################################################################
# 01 - Create Hexagonal Grid for Austin, TX
################################################################################
#
# This script creates a hexagonal grid covering Austin, TX city boundaries
# using the H3 spatial indexing system. The grid serves as the spatial unit
# of analysis for the displacement early warning system.
#
# H3 Resolution Guide:
# - Resolution 8: ~0.74 km² per cell (~461,354 cells globally)
# - Resolution 9: ~0.10 km² per cell (~3,279,871 cells globally)
# - Resolution 10: ~0.015 km² per cell (~23,000,000 cells globally)
# 
# We use Resolution 9 as it provides good spatial detail while maintaining
# computational efficiency.
#
# INPUTS:
#   - Adopted April 29, 2026 Austin full-purpose boundary and permanent H3 ID registry
#
# OUTPUTS:
#   - output/hex_grid.rds: H3 hexagonal grid as sf object
#     Columns: hex_id (character), geometry (polygon)
#
# DEPENDENCIES:
#   - sf, h3jsr, tigris packages
#   - R/utils.R for helper functions
#
################################################################################

# Source utilities before using the shared console helpers.
source(here::here("R/utils.R"))
source(here::here("R/analysis_config.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggthemes)
  library(h3jsr)
  library(htmlwidgets)
  library(mapview)
  library(scales)
  library(sf)
})

print_header("01 - CREATING HEXAGONAL GRID")

# Configuration
H3_RESOLUTION <- EWS_CONFIG$h3_resolution
OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")

# Create output directories if they don't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

################################################################################
# Step 1: Get Austin city boundary
################################################################################

boundary_path <- "data/BOUNDARIES_jurisdictions_20260429.geojson"
registry_path <- "config/hex_id_registry.csv"
austin_boundary <- st_read(boundary_path, quiet = TRUE) %>%
  filter(toupper(trimws(city_name)) == "CITY OF AUSTIN", toupper(trimws(jurisdiction_type)) == "FULL") %>%
  st_transform(3083) %>% st_make_valid() %>% summarise()
# Pad for candidate discovery, then retain every positive-area intersecting cell.
# Keep legacy audit cells too, so their identifiers/review links remain stable.
candidates <- polygon_to_cells(st_transform(st_buffer(austin_boundary,500),4326),res=H3_RESOLUTION)[[1]]
candidate_polys <- st_as_sf(cell_to_polygon(unique(as.character(candidates)),simple=FALSE)) %>% st_transform(3083)
intersection <- suppressWarnings(st_intersection(candidate_polys,austin_boundary))
required_ids <- sort(unique(intersection$h3_address[as.numeric(st_area(intersection)) > 1e-4]))
if (!file.exists(registry_path)) stop("Missing permanent H3 ID registry; do not renumber the existing grid.")
registry <- readr::read_csv(registry_path,show_col_types=FALSE,col_types=readr::cols(hex_id='i',h3_index='c'))
stopifnot(!anyDuplicated(registry$hex_id),!anyDuplicated(registry$h3_index))
missing_ids <- setdiff(required_ids,registry$h3_index)
if(length(missing_ids)) stop("H3 registry does not cover the adopted boundary; review and append missing IDs.")
hex_grid <- st_as_sf(cell_to_polygon(registry$h3_index,simple=FALSE)) %>%
  rename(h3_index=h3_address) %>% left_join(registry,by='h3_index') %>% arrange(hex_id)
centers <- suppressWarnings(st_centroid(st_geometry(hex_grid)))
hex_grid$longitude <- st_coordinates(centers)[,1]
hex_grid$latitude <- st_coordinates(centers)[,2]
hex_grid$area_km2 <- as.numeric(st_area(hex_grid))/1e6
hex_grid <- select(hex_grid,hex_id,h3_index,longitude,latitude,area_km2,geometry)
# Preserve the exact existing polygons and metadata when extending a live grid.
old_path <- file.path(OUTPUT_DIR,'hex_grid.rds')
if(file.exists(old_path)) {
 old <- readRDS(old_path);j <- match(old$h3_index,hex_grid$h3_index)
 stopifnot(!anyNA(j),identical(old$hex_id,hex_grid$hex_id[j]))
 hex_grid <- bind_rows(old,hex_grid[!hex_grid$h3_index %in% old$h3_index,]) %>% arrange(hex_id)
}
projected <- st_transform(hex_grid,3083)
uncovered <- suppressWarnings(st_difference(st_geometry(austin_boundary),st_union(projected)))
uncovered_m2 <- sum(as.numeric(st_area(uncovered)))
stopifnot(uncovered_m2 < 1)
center_count <- sum(lengths(st_covered_by(suppressWarnings(st_point_on_surface(projected)),austin_boundary))>0L)
print_progress(paste(nrow(hex_grid),'cells;',center_count,'center-selected City cells; uncovered square meters:',uncovered_m2))

################################################################################
# Step 4: Save the grid
################################################################################

output_file <- file.path(OUTPUT_DIR, "hex_grid.rds")
save_output(hex_grid, output_file, "hexagonal grid")
jsonlite::write_json(list(schema_version=1L,version='austin_full_20260429_stable_h3_v1',
 resolution=H3_RESOLUTION,grid_cells=nrow(hex_grid),city_center_cells=center_count,
 city_intersecting_cells=length(required_ids),uncovered_city_m2=uncovered_m2,
 computational_rule='legacy_cells_union_positive_area_full_city_intersection',
 analytical_rule='hex_point_on_surface_within_current_city_full',
 boundary_path=boundary_path,boundary_sha256=digest::digest(file=boundary_path,algo='sha256'),
 registry_path=registry_path,registry_sha256=digest::digest(file=registry_path,algo='sha256'),
 grid_sha256=digest::digest(file=output_file,algo='sha256')),
 file.path(OUTPUT_DIR,'hex_grid_manifest.json'),pretty=TRUE,auto_unbox=TRUE,digits=NA)


################################################################################
# Step 5: Create visualization
################################################################################

print_progress("Creating visualization...")

# Static plot using ggplot2
p1 <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "red", linewidth = 1) +
  geom_sf(data = hex_grid, fill = alpha("steelblue", 0.3), 
          color = "steelblue", linewidth = 0.3) +
  ggthemes::theme_map() +
  labs(
    title = "Hexagonal Grid for Austin, TX",
    subtitle = paste0("H3 Resolution ", H3_RESOLUTION, " (", 
                     nrow(hex_grid), " hexagons)")
  ) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11),
    axis.title = element_blank()
  )

# Save static plot
ggsave(
  filename = file.path(FIGURES_DIR, "01_hex_grid_static.png"),
  plot = p1,
  width = 10,
  height = 8,
  dpi = 300
)

print_progress("Saved static visualization")

# Interactive map using mapview
print_progress("Creating interactive map...")

# Sample a subset for faster interactive viewing if grid is large
hex_sample <- if(nrow(hex_grid) > 1000) {
  hex_grid %>% slice_sample(n = 1000)
} else {
  hex_grid
}

# Remove the problematic h3_index list column
hex_sample <- hex_sample %>%
  select(hex_id, longitude, latitude, area_km2, geometry)


map <- mapview(
  austin_boundary,
  color = "red",
  col.regions = "transparent",
  alpha.regions = 0,
  legend = FALSE,
  layer.name = "Austin Boundary"
) +
  mapview(
    hex_sample,
    zcol = "hex_id",
    alpha.regions = 0.4,
    legend = FALSE,
    layer.name = "Hexagonal Grid"
  )

# Save interactive map
htmlwidgets::saveWidget(
  map@map,
  file = file.path(normalizePath(FIGURES_DIR), "01_hex_grid_interactive.html"),
  selfcontained = TRUE
)

print_progress("Saved interactive map")

################################################################################
# Summary
################################################################################

print_header("STEP 01 COMPLETE")
cat("✓ Hexagonal grid created and saved\n")
cat("✓ Visualizations generated\n")
cat(paste0("✓ Grid file: ", output_file, "\n"))
cat(paste0("✓ Static map: ", file.path(FIGURES_DIR, "01_hex_grid_static.png"), "\n"))
cat(paste0("✓ Interactive map: ", file.path(FIGURES_DIR, "01_hex_grid_interactive.html"), "\n"))
