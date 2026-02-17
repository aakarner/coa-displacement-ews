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
#   - Austin city boundary from tigris package (downloaded automatically)
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

print_header("01 - CREATING HEXAGONAL GRID")

# Source utilities (enables standalone execution; also sourced by run_analysis.R)
source(here::here("R/utils.R"))

# Configuration
H3_RESOLUTION <- 9  # Hexagon resolution (~0.1km² cells)
OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")

# Create output directories if they don't exist
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

################################################################################
# Step 1: Get Austin city boundary
################################################################################

print_progress("Fetching Austin, TX city boundary from Census...")

# Get Texas places (cities) from Census TIGER/Line
austin_boundary <- tigris::places(state = "TX", year = 2021) %>%
  filter(NAME == "Austin") %>%
  st_transform(4326)  # WGS84 coordinate system

print_progress(paste0("Austin boundary loaded. Area: ", 
                     round(st_area(austin_boundary) / 1e6, 2), " km²"))

################################################################################
# Step 2: Create hexagonal grid using H3
################################################################################

print_progress(paste0("Creating H3 hexagonal grid at resolution ", H3_RESOLUTION, "..."))

# Use polygon_to_cells to get all H3 hexagons that cover Austin
# This approach captures all hexagons that overlap with the boundary,
# including those partially outside, ensuring no internal gaps
h3_indices_sf <- polygon_to_cells(austin_boundary, res = H3_RESOLUTION, simple = FALSE)

# Extract H3 indices as a character vector
h3_indices <- h3_indices_sf$h3_address

print_progress(paste0("Generated ", length(h3_indices), " H3 hexagons covering Austin"))

# Convert H3 indices to polygon geometries
print_progress("Converting H3 indices to polygon geometries...")
hex_grid <- cell_to_polygon(h3_indices, simple = FALSE) |>
  st_as_sf()

# Add H3 index as a column
hex_grid$h3_index <- h3_indices

print_progress(paste0("Final grid contains ", nrow(hex_grid), " hexagons"))

################################################################################
# Step 3: Add grid metadata
################################################################################

print_progress("Adding metadata to hexagonal grid...")

hex_grid <- hex_grid %>%
  mutate(
    # Calculate centroid coordinates
    centroid = st_centroid(geometry),
    longitude = st_coordinates(centroid)[, 1],
    latitude = st_coordinates(centroid)[, 2],
  ) %>%
  select(-centroid) %>%
  mutate(
    # Calculate area in km²
    area_km2 = as.numeric(st_area(geometry)) / 1e6,
    
    # Create sequential ID
    hex_id = row_number()
  ) %>%
  select(hex_id, h3_index, longitude, latitude, area_km2, geometry)
  

# Summary statistics
print_progress("Grid summary:")
cat(paste0("  - Number of hexagons: ", nrow(hex_grid), "\n"))
cat(paste0("  - Average area: ", round(mean(hex_grid$area_km2), 3), " km²\n"))
cat(paste0("  - Total area covered: ", round(sum(hex_grid$area_km2), 2), " km²\n"))

################################################################################
# Step 4: Save the grid
################################################################################

output_file <- file.path(OUTPUT_DIR, "hex_grid.rds")
save_output(hex_grid, output_file, "hexagonal grid")

################################################################################
# Step 5: Create visualization
################################################################################

print_progress("Creating visualization...")

# Static plot using ggplot2
p1 <- ggplot() +
  geom_sf(data = austin_boundary, fill = NA, color = "red", linewidth = 1) +
  geom_sf(data = hex_grid, fill = alpha("steelblue", 0.3), 
          color = "steelblue", linewidth = 0.3) +
  theme_minimal() +
  labs(
    title = "Hexagonal Grid for Austin, TX",
    subtitle = paste0("H3 Resolution ", H3_RESOLUTION, " (", 
                     nrow(hex_grid), " hexagons)"),
    caption = "Red boundary = Austin city limits"
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