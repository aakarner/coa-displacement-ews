################################################################################
# Audit Austin Boundary, County Boundaries, and Hex Counts
################################################################################
#
# This script maps the City of Austin full-purpose boundary against Travis,
# Hays, and Williamson county boundaries, then counts H3 hex cells by county.
#
# Hex counts use each hex cell's point-on-surface location so every hex is
# assigned to exactly one county. A separate intersection count is also saved
# for audit purposes; those counts can double-count hexes that touch more than
# one county.
#
# INPUTS:
#   - output/hex_grid.rds
#   - data/BOUNDARIES_jurisdictions_20260429.geojson
#
# OUTPUTS:
#   - output/hex_counts_by_county.csv
#   - output/hex_counts_by_county_intersections.csv
#   - figures/audit_austin_county_boundaries_hex_counts.png
#
################################################################################

source(here::here("R/utils.R"))

suppressPackageStartupMessages({
  library(sf)
  library(tidyverse)
  library(tigris)
  library(scales)
  library(ggthemes)
})

print_header("AUSTIN COUNTY BOUNDARIES AND HEX COUNTS")

OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
FIGURES_DIR <- here::here("figures")
ANALYSIS_CRS <- 3083  # NAD83 / Texas Centric Albers Equal Area

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

options(tigris_use_cache = TRUE)
sf_use_s2(FALSE)

################################################################################
# Step 1: Load spatial inputs
################################################################################

print_progress("Loading hex grid...")

hex_grid <- load_output(
  file.path(OUTPUT_DIR, "hex_grid.rds"),
  "hexagonal grid"
)

hex_grid_analysis <- st_transform(hex_grid, ANALYSIS_CRS)

print_progress("Loading City of Austin full-purpose boundary...")

jurisdictions_file <- file.path(DATA_DIR, "BOUNDARIES_jurisdictions_20260429.geojson")

if (!file.exists(jurisdictions_file)) {
  stop(paste0("File not found: ", jurisdictions_file))
}

austin_full_purpose <- st_read(jurisdictions_file, quiet = TRUE) %>%
  st_make_valid() %>%
  filter(jurisdiction_type == "FULL") %>%
  st_transform(ANALYSIS_CRS) %>%
  summarise(city_name = "City of Austin Full Purpose", .groups = "drop")

print_progress("Loading Travis, Hays, and Williamson county boundaries...")

target_counties <- c("Travis", "Hays", "Williamson")

county_boundaries <- tigris::counties(state = "TX", year = 2024, cb = TRUE, class = "sf") %>%
  filter(NAME %in% target_counties) %>%
  st_transform(ANALYSIS_CRS) %>%
  transmute(
    county = paste0(NAME, " County"),
    geoid = GEOID,
    geometry
  )

if (nrow(county_boundaries) != length(target_counties)) {
  stop("Did not retrieve all target counties from tigris.")
}

################################################################################
# Step 2: Assign hex cells to counties
################################################################################

print_progress("Assigning hex cells to counties using point-on-surface locations...")

hex_points <- suppressWarnings(
  hex_grid_analysis %>%
    st_point_on_surface()
)

hex_county_assignment <- hex_points %>%
  st_join(county_boundaries, join = st_within, left = TRUE)

hex_counts_by_county <- hex_county_assignment %>%
  st_drop_geometry() %>%
  mutate(county = replace_na(county, "Outside target counties")) %>%
  count(county, name = "hex_count") %>%
  mutate(
    pct_hexes = hex_count / sum(hex_count) * 100,
    assignment_method = "hex point-on-surface within county"
  ) %>%
  arrange(match(county, paste0(target_counties, " County")))

print_progress("Calculating county intersection counts for audit table...")

hex_counts_by_county_intersections <- hex_grid_analysis %>%
  st_join(county_boundaries, join = st_intersects, left = FALSE) %>%
  st_drop_geometry() %>%
  count(county, name = "intersecting_hex_count") %>%
  mutate(assignment_method = "hex polygon intersects county") %>%
  arrange(match(county, paste0(target_counties, " County")))

write_csv(hex_counts_by_county, file.path(OUTPUT_DIR, "hex_counts_by_county.csv"))
write_csv(
  hex_counts_by_county_intersections,
  file.path(OUTPUT_DIR, "hex_counts_by_county_intersections.csv")
)

print_progress("Hex counts by county:")
print(hex_counts_by_county)

################################################################################
# Step 3: Create map
################################################################################

print_progress("Creating county and city boundary map...")

county_labels <- county_boundaries %>%
  left_join(hex_counts_by_county, by = "county") %>%
  mutate(
    label = paste0(county, "\n", comma(hex_count), " hexes")
  ) %>%
  { suppressWarnings(st_point_on_surface(.)) }

p_county_city <- ggplot() +
  geom_sf(data = county_boundaries, aes(fill = county), color = "grey35", linewidth = 0.5, alpha = 0.18) +
  geom_sf(data = hex_grid_analysis, fill = NA, color = "grey75", linewidth = 0.08, alpha = 0.7) +
  geom_sf(data = austin_full_purpose, fill = NA, color = "#d73027", linewidth = 0.9) +
  geom_sf_text(data = county_labels, aes(label = label), size = 3.3, color = "grey15", lineheight = 0.95) +
  scale_fill_brewer(type = "qual", palette = "Set2", name = NULL) +
  coord_sf(datum = NA) +
  ggthemes::theme_map() +
  labs(
    title = "City of Austin Boundary with County Boundaries",
    subtitle = "Hex counts use each H3 cell's point-on-surface county assignment",
    caption = "Red outline: City of Austin full-purpose jurisdiction"
  ) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10),
    plot.caption = element_text(size = 8, color = "grey35")
  )

ggsave(
  filename = file.path(FIGURES_DIR, "audit_austin_county_boundaries_hex_counts.png"),
  plot = p_county_city,
  width = 10,
  height = 8,
  dpi = 300,
  bg = "white"
)

print_header("CITY/COUNTY HEX AUDIT COMPLETE")
cat("✓ County and city boundary map generated\n")
cat("✓ Hex counts by county saved\n")
cat(paste0("✓ Map: ", file.path(FIGURES_DIR, "audit_austin_county_boundaries_hex_counts.png"), "\n"))
cat(paste0("✓ Counts: ", file.path(OUTPUT_DIR, "hex_counts_by_county.csv"), "\n"))
