################################################################################
# Map Orientation Reference Layers
################################################################################
#
# Downloads 2025 TIGER/Line primary and secondary roads for Texas and area
# water for Travis County, then retains a small set of Austin-area reference
# features used only to orient readers on cluster maps. These geometries are
# cartographic context and are not inputs to the clustering analysis.
#
# Output:
#   output/map_orientation_reference.rds
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(sf)
  library(tigris)
})

print_header("MAP ORIENTATION REFERENCE LAYERS")

TIGER_YEAR <- 2025L
OUTPUT_PATH <- project_path("output", "map_orientation_reference.rds")
HEX_PATH <- project_path("output", "hex_grid.rds")

if (!file.exists(HEX_PATH)) {
  stop("Missing map-orientation input: hex_grid.rds", call. = FALSE)
}

hex_grid <- readRDS(HEX_PATH)
if (!inherits(hex_grid, "sf") || is.na(st_crs(hex_grid))) {
  stop("The hex grid must be an sf object with a coordinate system.", call. = FALSE)
}
# Use the map extent rather than the union of individual hexes. Clipping to the
# hex union creates artificial breaks wherever the municipal grid has holes or
# narrow discontinuities, which makes otherwise continuous roads and water look
# dashed on the finished map.
study_bbox <- st_as_sfc(st_bbox(hex_grid))

options(tigris_use_cache = TRUE)

print_progress("Downloading and filtering 2025 TIGER/Line roads...")
roads_all <- primary_secondary_roads(
  state = "TX",
  year = TIGER_YEAR,
  progress_bar = FALSE
)
study_bbox_roads <- st_transform(study_bbox, st_crs(roads_all))

roads_in_extent <- suppressWarnings(
  st_crop(roads_all, st_bbox(study_bbox_roads))
)
roads <- roads_in_extent %>%
  mutate(
    feature_id = case_when(
      FULLNAME %in% c("I- 35", "N I-35") ~ "i35",
      FULLNAME %in% c(
        "N Mopac Expy", "S Mopac Expy", "State Loop 1"
      ) ~ "mopac",
      FULLNAME %in% c("US Hwy 290", "W US Hwy 290") ~ "us290",
      FULLNAME %in% c(
        "S Hwy 183", "US Hwy 183", "US Hwy 183 S"
      ) ~ "us183",
      FULLNAME %in% c(
        "Hwy 71 E", "State Hwy 71", "State Hwy 71 E", "W Hwy 71"
      ) ~ "sh71",
      FULLNAME == "State Loop 360" ~ "loop360",
      FULLNAME == "State Hwy 130" ~ "sh130",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(feature_id)) %>%
  select(feature_id) %>%
  group_by(feature_id) %>%
  summarise(do_union = TRUE, .groups = "drop") %>%
  st_transform(4326)

expected_road_ids <- c(
  "mopac", "i35", "us290", "us183", "sh71", "loop360", "sh130"
)
missing_road_ids <- setdiff(expected_road_ids, roads$feature_id)
if (length(missing_road_ids) > 0L) {
  stop(
    "TIGER/Line roads did not contain configured corridor(s): ",
    paste(missing_road_ids, collapse = ", "),
    call. = FALSE
  )
}

print_progress("Downloading and filtering 2025 TIGER/Line area water...")
water_all <- area_water(
  state = "TX",
  county = "Travis",
  year = TIGER_YEAR,
  progress_bar = FALSE
)
study_bbox_water <- st_transform(study_bbox, st_crs(water_all))
water_in_extent <- suppressWarnings(
  st_crop(
    water_all %>% filter(FULLNAME == "Colorado Riv"),
    st_bbox(study_bbox_water)
  )
)
water <- water_in_extent %>%
  summarise(feature_id = "colorado_river", do_union = TRUE) %>%
  st_transform(4326)
if (nrow(water) != 1L || any(st_is_empty(water))) {
  stop("Could not isolate the Colorado River area-water feature.", call. = FALSE)
}

orientation_reference <- list(
  tiger_year = TIGER_YEAR,
  source = paste0(
    "U.S. Census Bureau ", TIGER_YEAR,
    " TIGER/Line primary and secondary roads and area water"
  ),
  roads = roads,
  water = water
)

dir.create(dirname(OUTPUT_PATH), recursive = TRUE, showWarnings = FALSE)
saveRDS(orientation_reference, OUTPUT_PATH)

cat("\nMap orientation reference: ", OUTPUT_PATH, "\n", sep = "")
cat("Map orientation reference layer complete.\n")
