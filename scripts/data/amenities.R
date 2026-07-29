################################################################################
# Process Amenity Change
################################################################################
#
# Builds a single, equal-category amenity reorientation score from recent
# openings of cafes, full-service restaurants, and drinking places. The score
# combines each category's recent 800-meter exposure and positive change from
# an equal prior window. Mixed-beverage and inspection records are retained as
# corroboration flags, not duplicated as opening events.
#
# Run scripts/audits/amenity_sources.R first.
#
# Outputs:
#   output/amenity_events_geocoded.rds
#   output/amenity_change_features_by_hex.rds/.csv
#   output/amenity_geocoding_qa.csv
#   output/amenity_geocoding_method_qa.csv
#   output/amenity_hex_distribution_qa.csv
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
  library(data.table)
  library(dplyr)
  library(readr)
  library(sf)
  library(tibble)
  library(tidygeocoder)
})

print_header("02n - AMENITY CHANGE PROCESSING")

OUTPUT_DIR <- project_path("output")
RAW_DIR <- project_path("data", "raw_amenities")
CANDIDATES_FILE <- file.path(OUTPUT_DIR, "amenity_source_candidates.rds")
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")
GEOCODE_CACHE <- file.path(RAW_DIR, "amenity_census_geocodes.csv")

required_files <- c(CANDIDATES_FILE, HEX_GRID_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Amenity processing is missing required input(s):\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}

dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

source_candidates <- load_output(
  CANDIDATES_FILE,
  "amenity source candidates"
)
hex_grid <- load_output(HEX_GRID_FILE, "hexagonal grid")

if (!inherits(hex_grid, "sf")) {
  stop("Amenity processing requires an sf hex grid.", call. = FALSE)
}
if (!identical(
  as.Date(source_candidates$analysis_as_of_date),
  EWS_CONFIG$analysis_as_of_date
)) {
  stop(
    "Amenity source audit cutoff does not match the pipeline cutoff. ",
    "Rerun scripts/audits/amenity_sources.R.",
    call. = FALSE
  )
}

analysis_as_of <- as.Date(source_candidates$analysis_as_of_date)
previous_start <- as.Date(source_candidates$previous_window_start)
recent_start <- as.Date(source_candidates$recent_window_start)
window_months <- as.integer(source_candidates$window_months)
access_radius_m <- as.numeric(EWS_CONFIG$amenity_access_radius_m)
core_categories <- c("cafe", "full_service_restaurant", "drinking_place")

if (!is.finite(access_radius_m) || access_radius_m <= 0) {
  stop("Amenity access radius must be positive.", call. = FALSE)
}

events <- source_candidates$sales_tax_locations %>%
  filter(
    core_index_eligible,
    category_classified %in% core_categories,
    event_window %in% c("previous", "recent")
  ) %>%
  mutate(
    address_key = as.character(address_key),
    street = as.character(street),
    city = as.character(city),
    state = as.character(state),
    zip = as.character(zip)
  )

if (nrow(events) == 0L) {
  stop("No core amenity opening events survived the source audit.", call. = FALSE)
}
if (anyDuplicated(events$event_id)) {
  stop("Amenity opening events contain duplicate stable IDs.", call. = FALSE)
}

################################################################################
# Batch geocode unique business addresses
################################################################################

address_table <- events %>%
  distinct(address_key, street, city, state, zip)

if (file.exists(GEOCODE_CACHE)) {
  geocode_cache <- read_csv(
    GEOCODE_CACHE,
    col_types = cols(
      address_key = col_character(),
      street = col_character(),
      city = col_character(),
      state = col_character(),
      zip = col_character(),
      lat = col_double(),
      long = col_double(),
      match_indicator = col_character(),
      match_type = col_character(),
      matched_address = col_character(),
      geocode_date = col_date()
    ),
    show_col_types = FALSE
  ) %>%
    distinct(address_key, .keep_all = TRUE)
} else {
  geocode_cache <- tibble(
    address_key = character(),
    street = character(),
    city = character(),
    state = character(),
    zip = character(),
    lat = double(),
    long = double(),
    match_indicator = character(),
    match_type = character(),
    matched_address = character(),
    geocode_method = character(),
    geocode_score = double(),
    geocode_date = as.Date(character())
  )
}

if (!"geocode_method" %in% names(geocode_cache)) {
  geocode_cache$geocode_method <- if_else(
    is.finite(geocode_cache$lat) & is.finite(geocode_cache$long),
    "census",
    NA_character_
  )
}
if (!"geocode_score" %in% names(geocode_cache)) {
  geocode_cache$geocode_score <- NA_real_
}

addresses_to_geocode <- address_table %>%
  anti_join(geocode_cache %>% select(address_key), by = "address_key")

if (nrow(addresses_to_geocode) > 0L) {
  print_progress(paste0(
    "Batch geocoding ", nrow(addresses_to_geocode),
    " unique amenity address(es) with the US Census geocoder..."
  ))

  geocoded_new <- addresses_to_geocode %>%
    tidygeocoder::geocode(
      street = street,
      city = city,
      state = state,
      postalcode = zip,
      method = "census",
      mode = "batch",
      batch_limit = 10000,
      full_results = TRUE,
      quiet = FALSE
    ) %>%
    transmute(
      address_key,
      street,
      city,
      state,
      zip,
      lat = as.numeric(lat),
      long = as.numeric(long),
      match_indicator = as.character(match_indicator),
      match_type = as.character(match_type),
      matched_address = as.character(matched_address),
      geocode_method = "census",
      geocode_score = NA_real_,
      geocode_date = Sys.Date()
    )

  geocode_cache <- bind_rows(geocode_cache, geocoded_new) %>%
    distinct(address_key, .keep_all = TRUE)
  write_csv(geocode_cache, GEOCODE_CACHE)
  print_progress(paste0("Updated geocode cache: ", GEOCODE_CACHE))
}

fallback_addresses <- address_table %>%
  left_join(
    geocode_cache %>% select(address_key, lat, long, geocode_method),
    by = "address_key"
  ) %>%
  filter(
    (!is.finite(lat) | !is.finite(long)) &
      (is.na(geocode_method) | geocode_method != "arcgis")
  ) %>%
  select(-lat, -long, -geocode_method)

if (nrow(fallback_addresses) > 0L) {
  print_progress(paste0(
    "Trying ArcGIS fallback geocoding for ", nrow(fallback_addresses),
    " Census-unmatched address(es)..."
  ))

  fallback_results <- fallback_addresses %>%
    tidygeocoder::geocode(
      street = street,
      city = city,
      state = state,
      postalcode = zip,
      method = "arcgis",
      full_results = TRUE,
      quiet = FALSE
    )
  if (!"address" %in% names(fallback_results)) {
    fallback_results$address <- NA_character_
  }

  fallback_results <- fallback_results %>%
    transmute(
      address_key,
      street,
      city,
      state,
      zip,
      lat = if_else(score >= 90, as.numeric(lat), NA_real_),
      long = if_else(score >= 90, as.numeric(long), NA_real_),
      match_indicator = if_else(score >= 90, "Match", "No_Match"),
      match_type = if_else(score >= 90, "ArcGIS score >= 90", NA_character_),
      matched_address = as.character(address),
      geocode_method = "arcgis",
      geocode_score = as.numeric(score),
      geocode_date = Sys.Date()
    )

  geocode_cache <- geocode_cache %>%
    filter(!address_key %in% fallback_results$address_key) %>%
    bind_rows(fallback_results) %>%
    distinct(address_key, .keep_all = TRUE)
  write_csv(geocode_cache, GEOCODE_CACHE)
  print_progress(paste0("Updated geocode cache: ", GEOCODE_CACHE))
}

events_geocoded <- events %>%
  left_join(
    geocode_cache %>%
      select(
        address_key, lat, long, match_indicator, match_type,
        matched_address, geocode_method, geocode_score, geocode_date
      ),
    by = "address_key"
  ) %>%
  mutate(
    geocode_matched = is.finite(lat) & is.finite(long) &
      match_indicator == "Match"
  )

geocode_qa <- events_geocoded %>%
  group_by(county, category_classified, event_window) %>%
  summarise(
    opening_events = n(),
    unique_addresses = n_distinct(address_key),
    geocoded_events = sum(geocode_matched),
    geocode_match_pct = 100 * mean(geocode_matched),
    mixed_beverage_address_matches = sum(
      mixed_beverage_address_match,
      na.rm = TRUE
    ),
    austin_food_address_matches = sum(austin_food_address_match, na.rm = TRUE),
    .groups = "drop"
  )

overall_geocode_pct <- 100 * mean(events_geocoded$geocode_matched)
category_geocode_pct <- events_geocoded %>%
  group_by(category_classified) %>%
  summarise(match_pct = 100 * mean(geocode_matched), .groups = "drop")

if (!is.finite(overall_geocode_pct) || overall_geocode_pct < 85) {
  stop(
    "Amenity event geocoding coverage is ",
    round(overall_geocode_pct, 1),
    "%; at least 85% is required before constructing spatial exposure.",
    call. = FALSE
  )
}
if (any(category_geocode_pct$match_pct < 75)) {
  low_categories <- category_geocode_pct %>%
    filter(match_pct < 75) %>%
    transmute(label = paste0(category_classified, " (", round(match_pct, 1), "%)")) %>%
    pull(label)
  stop(
    "Amenity geocoding coverage is below 75% for: ",
    paste(low_categories, collapse = "; "),
    call. = FALSE
  )
}

################################################################################
# Spatial exposure within an 800-meter neighborhood
################################################################################

events_sf <- events_geocoded %>%
  filter(geocode_matched) %>%
  st_as_sf(coords = c("long", "lat"), crs = 4326, remove = FALSE)

projection_crs <- 26914
hex_projected <- st_transform(hex_grid, projection_crs)
event_projected <- st_transform(events_sf, projection_crs)
hex_centroids <- st_centroid(st_geometry(hex_projected))

study_buffer <- st_buffer(st_union(hex_projected), dist = access_radius_m)
events_sf$within_study_buffer <- lengths(st_intersects(
  event_projected,
  study_buffer
)) > 0L

direct_hex <- st_join(
  events_sf,
  hex_grid %>% select(hex_id),
  join = st_within,
  left = TRUE
) %>%
  st_drop_geometry() %>%
  select(event_id, direct_hex_id = hex_id) %>%
  group_by(event_id) %>%
  summarise(
    direct_hex_id = first(
      direct_hex_id[!is.na(direct_hex_id)],
      default = NA_integer_
    ),
    .groups = "drop"
  )
events_sf <- events_sf %>% left_join(direct_hex, by = "event_id")

spatial_event_qa <- st_drop_geometry(events_sf) %>%
  group_by(county, category_classified, event_window) %>%
  summarise(
    events_within_study_buffer = sum(within_study_buffer),
    events_inside_hex_grid = sum(!is.na(direct_hex_id)),
    .groups = "drop"
  )

nearby_events <- st_is_within_distance(
  hex_centroids,
  event_projected,
  dist = access_radius_m
)

links <- rbindlist(lapply(seq_along(nearby_events), function(hex_row) {
  event_rows <- nearby_events[[hex_row]]
  if (length(event_rows) == 0L) return(NULL)
  data.table(hex_row = hex_row, event_row = event_rows)
}))

if (nrow(links) == 0L) {
  stop("No amenity opening events fall within the study-area radius.", call. = FALSE)
}

links[, distance_m := as.numeric(st_distance(
  hex_centroids[hex_row, ],
  event_projected[event_row, ],
  by_element = TRUE
))]
links[, exposure_weight := pmax(0, 1 - distance_m / access_radius_m)]

event_attributes <- as.data.table(st_drop_geometry(events_sf))[, .(
  event_id,
  category = category_classified,
  event_window,
  active_as_of,
  mixed_beverage_address_match,
  austin_food_address_match
)]
event_attributes[, event_row := .I]
links <- merge(links, event_attributes, by = "event_row", all.x = TRUE)

weighted_exposure <- links[, .(
  weighted_openings = sum(exposure_weight),
  opening_events = uniqueN(event_id),
  active_opening_events = uniqueN(event_id[active_as_of]),
  mixed_beverage_matches = uniqueN(event_id[mixed_beverage_address_match]),
  austin_food_matches = uniqueN(event_id[austin_food_address_match])
), by = .(hex_row, category, event_window)]

weighted_wide <- dcast(
  weighted_exposure,
  hex_row ~ category + event_window,
  value.var = "weighted_openings",
  fill = 0
)
count_wide <- dcast(
  weighted_exposure,
  hex_row ~ category + event_window,
  value.var = "opening_events",
  fill = 0
)
setnames(
  count_wide,
  setdiff(names(count_wide), "hex_row"),
  paste0("count_", setdiff(names(count_wide), "hex_row"))
)

hex_base <- as.data.table(st_drop_geometry(hex_grid))[, .(
  hex_row = .I,
  hex_id = as.character(hex_id)
)]
hex_features <- merge(hex_base, weighted_wide, by = "hex_row", all.x = TRUE)
hex_features <- merge(hex_features, count_wide, by = "hex_row", all.x = TRUE)

expected_weighted <- as.vector(outer(
  core_categories,
  c("previous", "recent"),
  paste,
  sep = "_"
))
expected_counts <- paste0("count_", expected_weighted)
for (column in c(expected_weighted, expected_counts)) {
  if (!column %in% names(hex_features)) hex_features[, (column) := 0]
  set(hex_features, which(is.na(hex_features[[column]])), column, 0)
}

normalize_robust <- function(x) {
  normalize_robust_to_100(as.numeric(x))
}

for (category in core_categories) {
  recent_col <- paste0(category, "_recent")
  previous_col <- paste0(category, "_previous")
  change_col <- paste0("amenity_", category, "_weighted_change")
  score_col <- paste0("amenity_", category, "_score")

  hex_features[, (change_col) := get(recent_col) - get(previous_col)]
  hex_features[, (score_col) := rowMeans(cbind(
    normalize_robust(get(recent_col)),
    normalize_robust(pmax(get(change_col), 0))
  ), na.rm = FALSE)]
}

category_score_cols <- paste0("amenity_", core_categories, "_score")
hex_features[, amenity_change_index := rowMeans(
  as.matrix(.SD),
  na.rm = FALSE
), .SDcols = category_score_cols]
hex_features[, `:=`(
  amenity_recent_weighted_openings = cafe_recent +
    full_service_restaurant_recent + drinking_place_recent,
  amenity_previous_weighted_openings = cafe_previous +
    full_service_restaurant_previous + drinking_place_previous,
  amenity_weighted_opening_change = (
    cafe_recent + full_service_restaurant_recent + drinking_place_recent
  ) - (
    cafe_previous + full_service_restaurant_previous +
      drinking_place_previous
  ),
  amenity_recent_opening_events = count_cafe_recent +
    count_full_service_restaurant_recent + count_drinking_place_recent,
  amenity_previous_opening_events = count_cafe_previous +
    count_full_service_restaurant_previous + count_drinking_place_previous,
  amenity_window_complete = TRUE,
  amenity_geocode_match_pct = overall_geocode_pct,
  amenity_analysis_as_of_date = analysis_as_of,
  amenity_previous_window_start = previous_start,
  amenity_recent_window_start = recent_start,
  amenity_window_months = window_months,
  amenity_access_radius_m = access_radius_m
)]

setorder(hex_features, hex_row)
hex_features[, hex_row := NULL]

if (anyDuplicated(hex_features$hex_id)) {
  stop("Amenity feature output contains duplicate hex IDs.", call. = FALSE)
}
if (nrow(hex_features) != nrow(hex_grid)) {
  stop("Amenity feature output does not cover the complete hex grid.", call. = FALSE)
}
if (any(!is.finite(hex_features$amenity_change_index))) {
  stop("Amenity change index contains non-finite values.", call. = FALSE)
}
if (any(hex_features$amenity_change_index < 0 |
        hex_features$amenity_change_index > 100)) {
  stop("Amenity change index falls outside the expected 0-100 range.", call. = FALSE)
}

geocoding_qa <- geocode_qa %>%
  left_join(
    spatial_event_qa,
    by = c("county", "category_classified", "event_window")
  ) %>%
  mutate(
    analysis_as_of_date = analysis_as_of,
    previous_window_start = previous_start,
    recent_window_start = recent_start,
    access_radius_m = access_radius_m
  )

geocoding_method_qa <- events_geocoded %>%
  count(geocode_method, geocode_matched, name = "opening_events") %>%
  mutate(
    opening_event_pct = 100 * opening_events / sum(opening_events),
    analysis_as_of_date = analysis_as_of
  )

hex_distribution_qa <- as_tibble(hex_features) %>%
  summarise(
    hexes = n(),
    nonzero_recent_exposure_hexes = sum(amenity_recent_weighted_openings > 0),
    nonzero_previous_exposure_hexes = sum(
      amenity_previous_weighted_openings > 0
    ),
    positive_change_hexes = sum(amenity_weighted_opening_change > 0),
    amenity_change_index_min = min(amenity_change_index),
    amenity_change_index_median = median(amenity_change_index),
    amenity_change_index_mean = mean(amenity_change_index),
    amenity_change_index_max = max(amenity_change_index),
    geocode_match_pct = overall_geocode_pct,
    geocoded_events = sum(events_geocoded$geocode_matched),
    events_within_study_buffer = sum(events_sf$within_study_buffer),
    events_inside_hex_grid = sum(!is.na(events_sf$direct_hex_id)),
    analysis_as_of_date = analysis_as_of,
    previous_window_start = previous_start,
    recent_window_start = recent_start,
    window_months = window_months,
    access_radius_m = access_radius_m
  )

save_output(
  events_sf,
  file.path(OUTPUT_DIR, "amenity_events_geocoded.rds"),
  "geocoded amenity opening events"
)
save_output(
  as_tibble(hex_features),
  file.path(OUTPUT_DIR, "amenity_change_features_by_hex.rds"),
  "amenity change hex features"
)
write_csv(
  as_tibble(hex_features),
  file.path(OUTPUT_DIR, "amenity_change_features_by_hex.csv")
)
write_csv(geocoding_qa, file.path(OUTPUT_DIR, "amenity_geocoding_qa.csv"))
write_csv(
  geocoding_method_qa,
  file.path(OUTPUT_DIR, "amenity_geocoding_method_qa.csv")
)
write_csv(
  hex_distribution_qa,
  file.path(OUTPUT_DIR, "amenity_hex_distribution_qa.csv")
)

cat("\nAmenity geocoding QA:\n")
print(geocoding_qa)
cat("\nAmenity hex distribution QA:\n")
print(hex_distribution_qa)
cat("\nAmenity change processing complete.\n")
