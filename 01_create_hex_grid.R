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
# 
# We use Resolution 8 as it provides good balance between spatial detail
# and computational efficiency.
#
################################################################################

print_header("01 - CREATING HEXAGONAL GRID")

# Source utilities
source(here::here("R/utils.R"))

# Configuration
H3_RESOLUTION <- 8  # Hexagon resolution (~0.5km² cells)
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

# Get the bounding box
bbox <- st_bbox(austin_boundary)

# Create a grid of points covering the bounding box
# We'll use these to generate H3 hexagons
grid_points <- st_make_grid(
  austin_boundary,
  cellsize = 0.01,  # About 1km spacing
  what = "centers"
) %>%
  st_sf() %>%
  st_transform(4326)

# Convert points to H3 indices
h3_indices <- point_to_cell(grid_points, res = H3_RESOLUTION)

# Get unique H3 indices
unique_h3 <- unique(h3_indices)

print_progress(paste0("Generated ", length(unique_h3), " unique H3 hexagons"))

# Convert H3 indices to polygon geometries
print_progress("Converting H3 indices to polygon geometries...")
hex_grid <- cell_to_polygon(unique_h3, simple = FALSE) %>%
  st_as_sf()

# Add H3 index as a column
hex_grid$h3_index <- unique_h3

# Filter to hexagons that intersect Austin boundary
print_progress("Filtering hexagons to Austin boundary...")
hex_grid <- hex_grid %>%
  st_filter(austin_boundary, .predicate = st_intersects)

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
    
    # Calculate area in km²
    area_km2 = as.numeric(st_area(geometry)) / 1e6,
    
    # Create sequential ID
    hex_id = row_number()
  ) %>%
  select(hex_id, h3_index, longitude, latitude, area_km2, geometry)
  # select(-centroid)

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
mapshot(
  map,
  file = file.path(FIGURES_DIR, "01_hex_grid_interactive.html"),
  remove_controls = c("zoomControl", "layersControl")
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
