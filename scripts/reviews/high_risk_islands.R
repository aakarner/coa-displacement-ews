################################################################################
# Part 1 High-Risk Island and Property-Driver Review
################################################################################
#
# Screens high- and very-high-concern Part 1 hexes for low-risk immediate
# neighbors, decomposes their cluster features, and attributes property-based
# signals where the source evidence supports that link. Rent and demographic
# measures remain explicitly area-level. A leading-property counterfactual
# removes observed events or corporate status while preserving the residential
# denominator, then reassigns the hex with the frozen Part 1 model.
#
# This is an optional diagnostic review. It does not alter production features,
# the frozen model, or canonical assignments.
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
source(project_path("R", "cluster_assignment.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(h3jsr)
  library(htmltools)
  library(htmlwidgets)
  library(leaflet)
  library(lubridate)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

print_header("PART 1 HIGH-RISK ISLAND AND PROPERTY-DRIVER REVIEW")

OUTPUT_DIR <- project_path("output")
PART1_DIR <- file.path(OUTPUT_DIR, "part1")
FIGURES_DIR <- project_path("figures")
PARAMETER_FILE <- project_path("config", "high_risk_island_parameters.csv")
FEATURE_FILE <- file.path(OUTPUT_DIR, "hex_features.rds")
ASSIGNMENT_FILE <- file.path(PART1_DIR, "baseline_cluster_assignments.csv")
MODEL_FILE <- file.path(PART1_DIR, "baseline_cluster_model.rds")
PARCEL_FILE <- file.path(OUTPUT_DIR, "residential_parcels_unit_promoted.rds")
PROJECT_MEMBERSHIP_FILE <- file.path(
  OUTPUT_DIR,
  "residential_unit_project_membership.rds"
)
EVICTION_FILE <- file.path(OUTPUT_DIR, "eviction_filings_full_geocoded_hex.rds")
CODE_CASE_FILE <- file.path(OUTPUT_DIR, "311_code_complaint_case_audit.rds")
DEMOLITION_FILE <- project_path("data", "Issued_Construction_Permits_20260401.csv")
LAND_USE_FILE <- project_path("data", "austin_land_use_inventory_202607.csv")
NEIGHBORHOOD_FILE <- project_path("data", "neighborhood_reporting_areas.geojson")
JURISDICTION_FILE <- project_path(
  "data",
  "BOUNDARIES_jurisdictions_20260429.geojson"
)

required_files <- c(
  PARAMETER_FILE,
  FEATURE_FILE,
  ASSIGNMENT_FILE,
  MODEL_FILE,
  PARCEL_FILE,
  PROJECT_MEMBERSHIP_FILE,
  EVICTION_FILE,
  CODE_CASE_FILE,
  DEMOLITION_FILE,
  LAND_USE_FILE,
  NEIGHBORHOOD_FILE,
  JURISDICTION_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "High-risk-island review is missing input(s):\n- ",
    paste(missing_files, collapse = "\n- "),
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
  ifelse(is.finite(denominator) & denominator > 0, numerator / denominator, NA_real_)
}

safe_weighted_share <- function(member, weight) {
  weight <- coalesce(as.numeric(weight), 0)
  if (sum(weight) <= 0) return(NA_real_)
  sum(weight[member], na.rm = TRUE) / sum(weight, na.rm = TRUE)
}

first_nonmissing <- function(x) {
  x <- x[!is.na(x) & nzchar(as.character(x))]
  if (length(x) == 0L) NA_character_ else as.character(x[[1L]])
}

paste_unique <- function(x) {
  x <- sort(unique(as.character(x[!is.na(x) & nzchar(as.character(x))])))
  if (length(x) == 0L) NA_character_ else paste(x, collapse = "; ")
}

normalize_identifier <- function(x) {
  x <- str_to_upper(str_squish(as.character(x)))
  x[x == "" | x == "NA"] <- NA_character_
  str_replace_all(x, "[^A-Z0-9]", "")
}

normalize_base_address <- function(x) {
  x <- str_to_upper(str_squish(as.character(x)))
  x[x == "" | x == "NA"] <- NA_character_
  x <- str_replace_all(x, "<BR>", ",")
  x <- str_replace(x, ",.*$", "")
  x <- str_replace(
    x,
    "\\s+(UNIT|APT|APARTMENT|SUITE|STE|BLDG|BUILDING|ROOM|RM|LOT|#)\\s*[-A-Z0-9].*$",
    ""
  )
  x <- str_replace(x, "\\s+[A-Z]?[0-9]{1,4}$", "")
  replacements <- c(
    "\\bSTREET\\b" = "ST",
    "\\bROAD\\b" = "RD",
    "\\bAVENUE\\b" = "AVE",
    "\\bBOULEVARD\\b" = "BLVD",
    "\\bDRIVE\\b" = "DR",
    "\\bLANE\\b" = "LN",
    "\\bCOURT\\b" = "CT",
    "\\bPLACE\\b" = "PL",
    "\\bPARKWAY\\b" = "PKWY",
    "\\bHIGHWAY\\b" = "HWY",
    "\\bTERRACE\\b" = "TER",
    "\\bTRAIL\\b" = "TRL",
    "\\bNORTH\\b" = "N",
    "\\bSOUTH\\b" = "S",
    "\\bEAST\\b" = "E",
    "\\bWEST\\b" = "W"
  )
  for (pattern in names(replacements)) {
    x <- str_replace_all(x, pattern, replacements[[pattern]])
  }
  x <- str_replace_all(x, "[^A-Z0-9]+", " ")
  x <- str_squish(x)
  x[x == ""] <- NA_character_
  x
}

normalize_parcel_address <- function(address, city) {
  address_normalized <- normalize_base_address(address)
  city_normalized <- normalize_base_address(city)

  vapply(
    seq_along(address_normalized),
    function(index) {
      normalized <- address_normalized[[index]]
      locality <- city_normalized[[index]]
      if (is.na(normalized) || is.na(locality)) return(normalized)

      locality <- str_replace_all(locality, "[^A-Z0-9 ]", "")
      normalized <- str_remove(
        normalized,
        paste0(
          "\\s+", locality,
          "(\\s+TX)?(\\s+[0-9]{5}(\\s+[0-9]{4})?)?$"
        )
      )
      str_squish(normalized)
    },
    character(1)
  )
}

extract_house_number <- function(x) {
  str_extract(normalize_base_address(x), "^[0-9]+")
}

robust_fixed_score <- function(value, reference, lower = 0.01, upper = 0.99) {
  bounds <- as.numeric(
    quantile(reference, probs = c(lower, upper), na.rm = TRUE, names = FALSE)
  )
  if (is.na(value)) return(NA_real_)
  if (!all(is.finite(bounds)) || diff(bounds) == 0) return(0)
  clipped <- min(max(value, bounds[[1L]]), bounds[[2L]])
  (clipped - bounds[[1L]]) / diff(bounds) * 100
}

mean_available <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

top_n_share <- function(x, n) {
  x <- sort(coalesce(as.numeric(x), 0), decreasing = TRUE)
  sum(head(x, n), na.rm = TRUE)
}

concentration_hhi <- function(x) {
  x <- coalesce(as.numeric(x), 0)
  if (sum(x) <= 0) return(NA_real_)
  shares <- x / sum(x)
  sum(shares^2)
}

parameters <- read_csv(
  PARAMETER_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
)
if (anyDuplicated(parameters$parameter) || anyNA(parameters$value)) {
  stop("High-risk-island parameters must be unique and nonmissing.", call. = FALSE)
}
parameter <- setNames(parameters$value, parameters$parameter)
parse_integer_set <- function(name) {
  as.integer(strsplit(parameter[[name]], ";", fixed = TRUE)[[1L]])
}
high_display_clusters <- parse_integer_set("high_display_clusters")
low_display_clusters <- parse_integer_set("low_display_clusters")
minimum_classified_neighbors <- as.integer(
  parameter[["minimum_classified_ring1_neighbors"]]
)
minimum_low_share <- as.numeric(parameter[["minimum_low_ring1_share"]])
top_properties_per_hex <- as.integer(parameter[["top_properties_per_hex"]])
property_match_max_distance_m <- as.numeric(
  parameter[["property_match_max_distance_m"]]
)
dominant_property_share <- as.numeric(parameter[["dominant_property_share"]])

if (
  anyNA(c(high_display_clusters, low_display_clusters)) ||
    minimum_classified_neighbors < 1L ||
    minimum_classified_neighbors > 6L ||
    !between(minimum_low_share, 0, 1) ||
    top_properties_per_hex < 1L ||
    property_match_max_distance_m <= 0 ||
    !between(dominant_property_share, 0, 1)
) {
  stop("High-risk-island parameter values are invalid.", call. = FALSE)
}

################################################################################
# Identify spatial islands
################################################################################

print_progress("Loading frozen Part 1 assignments and building H3 neighbors...")
features_sf <- readRDS(FEATURE_FILE)
features <- st_drop_geometry(features_sf)
assignments <- read_csv(ASSIGNMENT_FILE, show_col_types = FALSE)
model <- readRDS(MODEL_FILE)

domain_features <- model$features
required_feature_columns <- c(
  "hex_id", "h3_index", "longitude", "latitude", "area_km2", "total_pop",
  "residential_units", "residential_parcels", "primary_cluster_eligible",
  domain_features,
  "eviction_rate_units_denominator", "eviction_cases_total",
  "eviction_cases_latest_12mo", "eviction_cases_previous_12mo",
  "eviction_cases_latest_12mo_change_pct", "eviction_recent_share",
  "demo_latest_24mo", "demo_previous_24mo", "demo_total_latest_24mo",
  "demo_total_previous_24mo", "demo_recent_density", "demo_trend_positive",
  "demo_total_recent_density", "sr_311_smoke_signal_latest_12mo",
  "sr_311_smoke_signal_previous_12mo",
  "sr_311_smoke_signal_latest_12mo_density",
  "sr_311_smoke_signal_latest_12mo_change_pct", "corporate_owned_units",
  "financialized_owner_parcels", "pct_corporate_units",
  "corporate_owned_units_per_km2", "pct_financialized_owner_parcels"
)
require_columns(features, required_feature_columns, "Hex feature surface")
require_columns(
  assignments,
  c(
    "hex_id", "h3_index", "cluster", "display_cluster", "tentative_name",
    "concern_level", "map_color"
  ),
  "Baseline assignments"
)
if (
  model$k != 7L ||
    nrow(assignments) != length(model$training_hex_ids) ||
    anyDuplicated(assignments$hex_id) ||
    anyDuplicated(assignments$h3_index) ||
    !setequal(assignments$hex_id, model$training_hex_ids)
) {
  stop("Frozen model and baseline assignments do not align.", call. = FALSE)
}

model_metrics <- tibble(
  hex_id = model$training_hex_ids,
  model_cluster = model$training_assignment,
  distance_to_centroid = model$training_minimum_distance,
  margin_confidence = model$training_margin_confidence
)

cluster_hex <- assignments %>%
  select(
    hex_id,
    h3_index,
    cluster,
    display_cluster,
    tentative_name,
    concern_level,
    map_color
  ) %>%
  inner_join(
    features %>% select(all_of(required_feature_columns)),
    by = c("hex_id", "h3_index"),
    relationship = "one-to-one"
  ) %>%
  left_join(model_metrics, by = "hex_id", relationship = "one-to-one")
if (
  nrow(cluster_hex) != nrow(assignments) ||
    any(cluster_hex$cluster != cluster_hex$model_cluster)
) {
  stop("Frozen-model assignments were not reproduced in the review data.", call. = FALSE)
}

concern_ordinal <- c(
  "Low" = 1,
  "Moderate" = 2,
  "High" = 3,
  "Very high" = 4
)
if (any(!cluster_hex$concern_level %in% names(concern_ordinal))) {
  stop("Unexpected Part 1 concern category.", call. = FALSE)
}

scaled_matrix <- as.matrix(cluster_hex[, domain_features, drop = FALSE])
storage.mode(scaled_matrix) <- "double"
scaled_matrix <- sweep(
  scaled_matrix,
  2,
  model$preprocessing$center[domain_features],
  FUN = "-"
)
scaled_matrix <- sweep(
  scaled_matrix,
  2,
  model$preprocessing$scale[domain_features],
  FUN = "/"
)
rownames(scaled_matrix) <- cluster_hex$h3_index

h3_row <- setNames(seq_len(nrow(cluster_hex)), cluster_hex$h3_index)
ring1 <- get_ring(cluster_hex$h3_index, ring_size = 1L, simple = TRUE)
disk2 <- get_disk(cluster_hex$h3_index, ring_size = 2L, simple = TRUE)

build_neighbor_rows <- function(neighbor_lists, ring_label) {
  bind_rows(lapply(seq_along(neighbor_lists), function(index) {
    neighbors <- neighbor_lists[[index]]
    neighbors <- setdiff(neighbors, cluster_hex$h3_index[[index]])
    neighbor_index <- unname(h3_row[neighbors])
    neighbor_index <- neighbor_index[!is.na(neighbor_index)]
    if (length(neighbor_index) == 0L) return(NULL)
    tibble(
      focal_hex_id = cluster_hex$hex_id[[index]],
      focal_h3_index = cluster_hex$h3_index[[index]],
      neighbor_hex_id = cluster_hex$hex_id[neighbor_index],
      neighbor_h3_index = cluster_hex$h3_index[neighbor_index],
      ring = ring_label,
      neighbor_display_cluster = cluster_hex$display_cluster[neighbor_index],
      neighbor_concern_level = cluster_hex$concern_level[neighbor_index],
      neighbor_population = cluster_hex$total_pop[neighbor_index],
      neighbor_residential_units = cluster_hex$residential_units[neighbor_index]
    )
  }))
}

ring1_context <- build_neighbor_rows(ring1, "ring1")
disk2_context <- build_neighbor_rows(disk2, "disk2")

ring1_domain_means <- ring1_context %>%
  left_join(
    cluster_hex %>%
      select(hex_id, all_of(domain_features)) %>%
      rename(neighbor_hex_id = hex_id),
    by = "neighbor_hex_id",
    relationship = "many-to-one"
  ) %>%
  group_by(focal_hex_id) %>%
  summarise(
    across(all_of(domain_features), ~ mean(.x, na.rm = TRUE), .names = "neighbor_mean_{.col}"),
    .groups = "drop"
  )

ring1_summary <- ring1_context %>%
  group_by(focal_hex_id) %>%
  summarise(
    classified_ring1_neighbors = n(),
    low_ring1_neighbors = sum(neighbor_display_cluster %in% low_display_clusters),
    low_ring1_share = mean(neighbor_display_cluster %in% low_display_clusters),
    low_ring1_population_share = safe_weighted_share(
      neighbor_display_cluster %in% low_display_clusters,
      neighbor_population
    ),
    low_ring1_unit_share = safe_weighted_share(
      neighbor_display_cluster %in% low_display_clusters,
      neighbor_residential_units
    ),
    neighbor_mean_concern_ordinal = mean(
      unname(concern_ordinal[neighbor_concern_level])
    ),
    neighbor_display_clusters = paste_unique(neighbor_display_cluster),
    .groups = "drop"
  )

disk2_summary <- disk2_context %>%
  group_by(focal_hex_id) %>%
  summarise(
    classified_disk2_neighbors = n(),
    low_disk2_share = mean(neighbor_display_cluster %in% low_display_clusters),
    low_disk2_population_share = safe_weighted_share(
      neighbor_display_cluster %in% low_display_clusters,
      neighbor_population
    ),
    low_disk2_unit_share = safe_weighted_share(
      neighbor_display_cluster %in% low_display_clusters,
      neighbor_residential_units
    ),
    .groups = "drop"
  )

profile_contrast <- vapply(seq_len(nrow(cluster_hex)), function(index) {
  neighbor_rows <- unname(h3_row[ring1[[index]]])
  neighbor_rows <- neighbor_rows[!is.na(neighbor_rows) & neighbor_rows != index]
  if (length(neighbor_rows) == 0L) return(NA_real_)
  sqrt(sum((scaled_matrix[index, ] - colMeans(scaled_matrix[neighbor_rows, , drop = FALSE]))^2))
}, numeric(1))

cluster_hex <- cluster_hex %>%
  mutate(
    focal_concern_ordinal = unname(concern_ordinal[concern_level]),
    local_profile_contrast = profile_contrast
  ) %>%
  left_join(ring1_summary, by = c("hex_id" = "focal_hex_id")) %>%
  left_join(disk2_summary, by = c("hex_id" = "focal_hex_id")) %>%
  left_join(ring1_domain_means, by = c("hex_id" = "focal_hex_id")) %>%
  mutate(
    concern_ordinal_contrast =
      focal_concern_ordinal - neighbor_mean_concern_ordinal,
    high_risk_island_candidate =
      display_cluster %in% high_display_clusters &
      classified_ring1_neighbors >= minimum_classified_neighbors &
      low_ring1_share >= minimum_low_share - 1e-6
  )

domain_labels <- c(
  rent_pressure_citywide_index = "Rent pressure",
  demographic_vulnerability_index = "Demographic vulnerability",
  demolition_pressure_index = "Demolition pressure",
  eviction_pressure_index = "Eviction pressure",
  sr_311_pressure_index = "311 pressure",
  ownership_pressure_index = "Corporate ownership pressure",
  amenity_change_index = "Amenity change"
)

candidate_domains <- cluster_hex %>%
  filter(high_risk_island_candidate) %>%
  select(hex_id, all_of(domain_features)) %>%
  pivot_longer(
    all_of(domain_features),
    names_to = "domain_variable",
    values_to = "domain_score"
  ) %>%
  mutate(domain_label = unname(domain_labels[domain_variable])) %>%
  arrange(hex_id, desc(domain_score), domain_label) %>%
  group_by(hex_id) %>%
  summarise(
    leading_domain = first(domain_label),
    leading_domain_score = first(domain_score),
    second_domain = nth(domain_label, 2L),
    second_domain_score = nth(domain_score, 2L),
    .groups = "drop"
  )

reporting_areas <- st_read(NEIGHBORHOOD_FILE, quiet = TRUE) %>%
  transmute(neighborhood_name = trimws(as.character(neighname))) %>%
  st_make_valid() %>%
  st_transform(4326)
candidate_points <- cluster_hex %>%
  filter(high_risk_island_candidate) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE)
reporting_hits <- st_within(candidate_points, reporting_areas)
if (any(lengths(reporting_hits) > 1L)) {
  stop("A candidate hex center falls in multiple reporting areas.", call. = FALSE)
}
reporting_index <- vapply(
  reporting_hits,
  function(index) if (length(index) == 1L) index[[1L]] else NA_integer_,
  integer(1)
)
candidate_neighborhoods <- tibble(
  hex_id = candidate_points$hex_id,
  neighborhood_name = reporting_areas$neighborhood_name[reporting_index]
)

candidate_hexes <- cluster_hex %>%
  filter(high_risk_island_candidate) %>%
  left_join(candidate_domains, by = "hex_id", relationship = "one-to-one") %>%
  left_join(candidate_neighborhoods, by = "hex_id", relationship = "one-to-one") %>%
  arrange(display_cluster, hex_id)

if (nrow(candidate_hexes) == 0L) {
  stop("No high-risk-island candidates met the configured screen.", call. = FALSE)
}

################################################################################
# Build project and parcel linkage surface
################################################################################

print_progress("Building parcel and physical-project attribution surface...")
parcels_raw <- readRDS(PARCEL_FILE)
project_membership <- readRDS(PROJECT_MEMBERSHIP_FILE)
require_columns(
  parcels_raw,
  c(
    "parcel_id", "situs_address", "situs_city", "lat", "lon", "source_county",
    "parcel_count", "units_calibrated_targeted", "promoted_units",
    "is_corporate_owned", "has_financialized_owner", "owner_names",
    "unit_estimation_method", "unit_estimation_confidence",
    "unit_land_use_validation_excluded"
  ),
  "Promoted residential parcel surface"
)
require_columns(
  project_membership,
  c("parcel_id", "project_id", "project_grouping_methods"),
  "Residential project membership"
)
if (
  anyDuplicated(parcels_raw$parcel_id) ||
    anyDuplicated(project_membership$parcel_id)
) {
  stop("Parcel and project membership IDs must be unique.", call. = FALSE)
}

hex_geometry <- features_sf %>% select(hex_id, h3_index)
parcels_sf <- parcels_raw %>%
  left_join(
    project_membership %>%
      select(parcel_id, project_id, project_grouping_methods),
    by = "parcel_id",
    relationship = "one-to-one"
  ) %>%
  filter(!coalesce(as.logical(unit_land_use_validation_excluded), FALSE)) %>%
  mutate(
    parcel_id = as.character(parcel_id),
    project_id = as.character(project_id),
    parcel_address_normalized = normalize_parcel_address(situs_address, situs_city),
    parcel_house_number = extract_house_number(situs_address),
    feature_residential_units = coalesce(as.numeric(units_calibrated_targeted), 0),
    promoted_residential_units = coalesce(as.numeric(promoted_units), 0),
    feature_corporate_units = if_else(
      coalesce(as.logical(is_corporate_owned), FALSE),
      feature_residential_units,
      0
    ),
    feature_residential_parcels = coalesce(as.numeric(parcel_count), 0),
    feature_financialized_parcels = if_else(
      coalesce(as.logical(has_financialized_owner), FALSE),
      feature_residential_parcels,
      0
    )
  ) %>%
  filter(is.finite(lat), is.finite(lon), !is.na(project_id)) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_join(hex_geometry, join = st_within, left = TRUE)

parcel_flat <- parcels_sf %>% st_drop_geometry()
parcel_project_lookup <- parcel_flat %>%
  select(
    parcel_id,
    project_id,
    property_hex_id = hex_id,
    parcel_address_normalized,
    parcel_house_number
  )

address_project_lookup <- parcel_flat %>%
  filter(!is.na(parcel_address_normalized)) %>%
  group_by(parcel_address_normalized) %>%
  summarise(
    address_project_count = n_distinct(project_id),
    address_property_hex_count = n_distinct(hex_id, na.rm = TRUE),
    address_project_id = if_else(
      address_project_count == 1L,
      first(project_id),
      NA_character_
    ),
    address_property_hex_id = if_else(
      address_property_hex_count == 1L,
      suppressWarnings(as.numeric(first_nonmissing(hex_id))),
      NA_real_
    ),
    .groups = "drop"
  )

project_representatives <- parcel_flat %>%
  arrange(project_id, desc(promoted_residential_units), desc(feature_residential_units)) %>%
  group_by(project_id) %>%
  slice(1L) %>%
  ungroup() %>%
  transmute(
    project_id,
    representative_parcel_id = parcel_id,
    representative_address = situs_address,
    representative_owner = owner_names,
    representative_latitude = lat,
    representative_longitude = lon,
    unit_estimation_method,
    unit_estimation_confidence
  )

project_global <- parcel_flat %>%
  group_by(project_id) %>%
  summarise(
    project_parcel_count = n_distinct(parcel_id),
    project_hex_count = n_distinct(hex_id, na.rm = TRUE),
    project_feature_units = sum(feature_residential_units),
    project_promoted_units = sum(promoted_residential_units),
    project_feature_corporate_units = sum(feature_corporate_units),
    project_is_corporate_owned = any(coalesce(as.logical(is_corporate_owned), FALSE)),
    project_has_financialized_owner = any(
      coalesce(as.logical(has_financialized_owner), FALSE)
    ),
    source_counties = paste_unique(source_county),
    project_grouping_methods = paste_unique(project_grouping_methods),
    .groups = "drop"
  ) %>%
  mutate(project_crosses_hex_boundary = project_hex_count > 1L) %>%
  left_join(project_representatives, by = "project_id", relationship = "one-to-one")

project_hex <- parcel_flat %>%
  filter(!is.na(hex_id)) %>%
  group_by(hex_id, project_id) %>%
  summarise(
    parcels_in_hex = n_distinct(parcel_id),
    feature_units_in_hex = sum(feature_residential_units),
    promoted_units_in_hex = sum(promoted_residential_units),
    feature_corporate_units_in_hex = sum(feature_corporate_units),
    feature_financialized_parcels_in_hex = sum(feature_financialized_parcels),
    project_latitude_in_hex = weighted.mean(
      lat,
      pmax(promoted_residential_units, 1),
      na.rm = TRUE
    ),
    project_longitude_in_hex = weighted.mean(
      lon,
      pmax(promoted_residential_units, 1),
      na.rm = TRUE
    ),
    .groups = "drop"
  )

parcel_points_utm <- st_transform(parcels_sf, 26914)

match_events_to_projects <- function(
  events,
  event_id_column,
  longitude_column,
  latitude_column,
  address_column,
  existing_parcel_column = NULL
) {
  event_id <- events[[event_id_column]]
  if (anyDuplicated(event_id)) {
    stop("Event IDs must be unique before property matching.", call. = FALSE)
  }
  events <- events %>%
    mutate(
      event_address_normalized = normalize_base_address(.data[[address_column]]),
      event_house_number = extract_house_number(.data[[address_column]])
    ) %>%
    left_join(
      address_project_lookup,
      by = c("event_address_normalized" = "parcel_address_normalized"),
      relationship = "many-to-one"
    )

  if (!is.null(existing_parcel_column)) {
    events$source_exact_parcel_id <- as.character(
      events[[existing_parcel_column]]
    )
    exact_parcel <- parcel_project_lookup %>%
      select(
        lookup_parcel_id = parcel_id,
        exact_project_id = project_id,
        exact_property_hex_id = property_hex_id
      )
    events <- events %>%
      left_join(
        exact_parcel,
        by = c("source_exact_parcel_id" = "lookup_parcel_id"),
        relationship = "many-to-one"
      )
  } else {
    events$exact_project_id <- NA_character_
    events$source_exact_parcel_id <- NA_character_
    events$exact_property_hex_id <- NA_real_
  }

  events <- events %>%
    mutate(
      matched_project_id = coalesce(exact_project_id, address_project_id),
      matched_parcel_id = source_exact_parcel_id,
      property_match_method = case_when(
        !is.na(exact_project_id) ~ "exact_source_parcel_id",
        !is.na(address_project_id) ~ "unique_normalized_address",
        TRUE ~ "unmatched"
      ),
      matched_property_hex_id = case_when(
        !is.na(exact_property_hex_id) ~ exact_property_hex_id,
        !is.na(address_project_id) ~ address_property_hex_id,
        TRUE ~ NA_real_
      ),
      nearest_parcel_distance_m = NA_real_,
      nearest_parcel_house_number = NA_character_,
      nearest_property_hex_id = NA_real_
    )

  nearest_candidates <- events %>%
    filter(
      is.na(matched_project_id),
      is.finite(.data[[longitude_column]]),
      is.finite(.data[[latitude_column]])
    )
  if (nrow(nearest_candidates) > 0L) {
    event_points <- nearest_candidates %>%
      st_as_sf(
        coords = c(longitude_column, latitude_column),
        crs = 4326,
        remove = FALSE
      ) %>%
      st_transform(26914)
    nearest_index <- st_nearest_feature(event_points, parcel_points_utm)
    nearest_distance <- as.numeric(st_distance(
      event_points,
      parcel_points_utm[nearest_index, ],
      by_element = TRUE
    ))
    nearest_lookup <- tibble(
      match_event_id = nearest_candidates[[event_id_column]],
      nearest_project_id = parcel_points_utm$project_id[nearest_index],
      nearest_parcel_id = parcel_points_utm$parcel_id[nearest_index],
      nearest_parcel_distance_m = nearest_distance,
      nearest_parcel_house_number =
        parcel_points_utm$parcel_house_number[nearest_index],
      nearest_property_hex_id = parcel_points_utm$hex_id[nearest_index]
    )
    names(nearest_lookup)[names(nearest_lookup) == "match_event_id"] <- event_id_column
    events <- events %>%
      select(
        -nearest_parcel_distance_m,
        -nearest_parcel_house_number,
        -nearest_property_hex_id
      ) %>%
      left_join(nearest_lookup, by = event_id_column, relationship = "one-to-one") %>%
      mutate(
        conservative_nearest_match =
          is.na(matched_project_id) &
          is.finite(nearest_parcel_distance_m) &
          nearest_parcel_distance_m <= property_match_max_distance_m &
          !is.na(event_house_number) &
          event_house_number == nearest_parcel_house_number,
        matched_project_id = if_else(
          conservative_nearest_match,
          nearest_project_id,
          matched_project_id
        ),
        matched_parcel_id = if_else(
          conservative_nearest_match,
          nearest_parcel_id,
          matched_parcel_id
        ),
        property_match_method = if_else(
          conservative_nearest_match,
          "nearest_parcel_same_house_number",
          property_match_method
        ),
        matched_property_hex_id = case_when(
          !is.na(matched_property_hex_id) ~ matched_property_hex_id,
          conservative_nearest_match ~ nearest_property_hex_id,
          TRUE ~ NA_real_
        )
      )
  } else {
    events <- events %>%
      mutate(
        nearest_project_id = NA_character_,
        nearest_parcel_id = NA_character_,
        conservative_nearest_match = FALSE
      )
  }

  events %>%
    select(-address_project_id, -address_property_hex_id, -exact_project_id)
}

################################################################################
# Attribute eviction, 311, and demolition events
################################################################################

latest_12mo_start <- EWS_CONFIG$analysis_as_of_date %m-% years(1) + days(1)
previous_12mo_start <- latest_12mo_start %m-% years(1)
latest_24mo_start <- EWS_CONFIG$analysis_as_of_date %m-% years(2) + days(1)
previous_24mo_start <- latest_24mo_start %m-% years(2)
candidate_ids <- candidate_hexes$hex_id

print_progress("Matching eviction filings to residential projects...")
eviction_events <- readRDS(EVICTION_FILE) %>%
  st_drop_geometry() %>%
  filter(hex_id %in% candidate_ids) %>%
  arrange(hex_id, case_number, file_date) %>%
  distinct(hex_id, case_number, .keep_all = TRUE) %>%
  mutate(eviction_event_id = paste(hex_id, case_number, sep = ":"))
eviction_events <- match_events_to_projects(
  eviction_events,
  event_id_column = "eviction_event_id",
  longitude_column = "longitude",
  latitude_column = "latitude",
  address_column = "address_for_geocoding"
) %>%
  mutate(
    latest_window = file_date >= latest_12mo_start &
      file_date <= EWS_CONFIG$analysis_as_of_date,
    previous_window = file_date >= previous_12mo_start &
      file_date < latest_12mo_start,
    event_property_hex_disagreement =
      !is.na(matched_property_hex_id) & hex_id != matched_property_hex_id
  )

eviction_by_project <- eviction_events %>%
  filter(!is.na(matched_project_id)) %>%
  group_by(hex_id, project_id = matched_project_id) %>%
  summarise(
    eviction_cases_total_attributed = n_distinct(case_number),
    eviction_cases_latest_12mo_attributed = n_distinct(case_number[latest_window]),
    eviction_cases_previous_12mo_attributed = n_distinct(case_number[previous_window]),
    eviction_property_hex_disagreements = sum(event_property_hex_disagreement),
    eviction_match_methods = paste_unique(property_match_method),
    .groups = "drop"
  )

print_progress("Matching linked Code Officer requests to residential projects...")
code_cases <- readRDS(CODE_CASE_FILE)
code_requests <- code_cases %>%
  filter(
    linked_to_ews_311,
    !is.na(service_request_id),
    is.finite(ews_311_latitude),
    is.finite(ews_311_longitude)
  ) %>%
  group_by(service_request_id) %>%
  summarise(
    request_date = first(ews_311_created_date),
    request_latitude = first(ews_311_latitude),
    request_longitude = first(ews_311_longitude),
    exact_residential_parcel_count = n_distinct(
      residential_parcel_id[exact_promoted_residential_match],
      na.rm = TRUE
    ),
    residential_parcel_id = if_else(
      exact_residential_parcel_count == 1L,
      first(na.omit(residential_parcel_id[exact_promoted_residential_match])),
      NA_character_
    ),
    is_structure_condition = any(complaint_category == "structure_condition"),
    complaint_categories = paste_unique(complaint_category),
    .groups = "drop"
  )
code_requests_sf <- code_requests %>%
  st_as_sf(
    coords = c("request_longitude", "request_latitude"),
    crs = 4326,
    remove = FALSE
  ) %>%
  st_join(hex_geometry %>% select(hex_id), join = st_within, left = FALSE) %>%
  st_drop_geometry() %>%
  filter(hex_id %in% candidate_ids) %>%
  left_join(
    parcel_project_lookup,
    by = c("residential_parcel_id" = "parcel_id"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    latest_window = request_date >= latest_12mo_start &
      request_date <= EWS_CONFIG$analysis_as_of_date,
    previous_window = request_date >= previous_12mo_start &
      request_date < latest_12mo_start,
    event_property_hex_disagreement =
      !is.na(property_hex_id) & hex_id != property_hex_id
  )

requests_311_by_project <- code_requests_sf %>%
  filter(!is.na(project_id)) %>%
  group_by(hex_id, project_id) %>%
  summarise(
    requests_311_total_attributed = n_distinct(service_request_id),
    requests_311_latest_12mo_attributed = n_distinct(
      service_request_id[latest_window]
    ),
    requests_311_previous_12mo_attributed = n_distinct(
      service_request_id[previous_window]
    ),
    structure_311_total_attributed = n_distinct(
      service_request_id[is_structure_condition]
    ),
    structure_311_latest_12mo_attributed = n_distinct(
      service_request_id[is_structure_condition & latest_window]
    ),
    structure_311_previous_12mo_attributed = n_distinct(
      service_request_id[is_structure_condition & previous_window]
    ),
    request_property_hex_disagreements = sum(event_property_hex_disagreement),
    .groups = "drop"
  )

print_progress("Matching demolition permits to residential projects...")
land_use_crosswalk <- read_csv(
  LAND_USE_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
) %>%
  transmute(
    permit_tcad_id = normalize_identifier(parcel_id_10),
    travis_property_id = normalize_identifier(property_id)
  ) %>%
  filter(!is.na(permit_tcad_id), !is.na(travis_property_id)) %>%
  distinct() %>%
  group_by(permit_tcad_id) %>%
  summarise(
    property_id_count = n_distinct(travis_property_id),
    exact_residential_parcel_id = if_else(
      property_id_count == 1L,
      first(travis_property_id),
      NA_character_
    ),
    .groups = "drop"
  )

demolition_events <- read_csv(
  DEMOLITION_FILE,
  show_col_types = FALSE,
  guess_max = 20000,
  col_select = all_of(c(
    "Permit Num", "Project Name", "Description", "TCAD ID",
    "Original Address 1", "Permit Class Mapped", "Work Class",
    "Issued Date", "Latitude", "Longitude"
  ))
) %>%
  mutate(
    issue_date = ymd(.data[["Issued Date"]]),
    is_demolition_work_class = str_detect(
      .data[["Work Class"]],
      regex("^demolition$", ignore_case = TRUE)
    ),
    is_residential_demo = str_detect(
      .data[["Permit Class Mapped"]],
      regex("residential", ignore_case = TRUE)
    ),
    is_total_demolition = str_detect(
      Description,
      regex("total\\s+demo", ignore_case = TRUE)
    )
  ) %>%
  filter(
    is_demolition_work_class,
    is_residential_demo,
    !is.na(issue_date),
    issue_date <= EWS_CONFIG$analysis_as_of_date,
    is.finite(Latitude),
    is.finite(Longitude)
  ) %>%
  mutate(
    demolition_event_id = paste0("demo_", row_number()),
    permit_tcad_id = normalize_identifier(.data[["TCAD ID"]])
  ) %>%
  left_join(land_use_crosswalk, by = "permit_tcad_id", relationship = "many-to-one") %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326, remove = FALSE) %>%
  st_join(hex_geometry %>% select(hex_id), join = st_within, left = FALSE) %>%
  st_drop_geometry() %>%
  filter(hex_id %in% candidate_ids)

demolition_events <- match_events_to_projects(
  demolition_events,
  event_id_column = "demolition_event_id",
  longitude_column = "Longitude",
  latitude_column = "Latitude",
  address_column = "Original Address 1",
  existing_parcel_column = "exact_residential_parcel_id"
) %>%
  mutate(
    latest_window = issue_date >= latest_24mo_start &
      issue_date <= EWS_CONFIG$analysis_as_of_date,
    previous_window = issue_date >= previous_24mo_start &
      issue_date < latest_24mo_start,
    event_property_hex_disagreement =
      !is.na(matched_property_hex_id) & hex_id != matched_property_hex_id
  )

demolition_by_project <- demolition_events %>%
  filter(!is.na(matched_project_id)) %>%
  group_by(hex_id, project_id = matched_project_id) %>%
  summarise(
    demolition_permits_total_attributed = n(),
    demolition_permits_latest_24mo_attributed = sum(latest_window),
    demolition_permits_previous_24mo_attributed = sum(previous_window),
    total_demolition_latest_24mo_attributed = sum(
      latest_window & is_total_demolition
    ),
    total_demolition_previous_24mo_attributed = sum(
      previous_window & is_total_demolition
    ),
    demolition_property_hex_disagreements = sum(event_property_hex_disagreement),
    demolition_match_methods = paste_unique(property_match_method),
    .groups = "drop"
  )

################################################################################
# Property-driver table and concentration
################################################################################

print_progress("Calculating property concentration and leading drivers...")
candidate_project_keys <- bind_rows(
  project_hex %>%
    filter(hex_id %in% candidate_ids) %>%
    select(hex_id, project_id),
  eviction_by_project %>% select(hex_id, project_id),
  requests_311_by_project %>% select(hex_id, project_id),
  demolition_by_project %>% select(hex_id, project_id)
) %>%
  distinct()

event_columns <- c(
  "eviction_cases_total_attributed",
  "eviction_cases_latest_12mo_attributed",
  "eviction_cases_previous_12mo_attributed",
  "eviction_property_hex_disagreements",
  "requests_311_total_attributed",
  "requests_311_latest_12mo_attributed",
  "requests_311_previous_12mo_attributed",
  "structure_311_total_attributed",
  "structure_311_latest_12mo_attributed",
  "structure_311_previous_12mo_attributed",
  "request_property_hex_disagreements",
  "demolition_permits_total_attributed",
  "demolition_permits_latest_24mo_attributed",
  "demolition_permits_previous_24mo_attributed",
  "total_demolition_latest_24mo_attributed",
  "total_demolition_previous_24mo_attributed",
  "demolition_property_hex_disagreements"
)

property_drivers <- candidate_project_keys %>%
  left_join(project_hex, by = c("hex_id", "project_id"), relationship = "one-to-one") %>%
  left_join(project_global, by = "project_id", relationship = "many-to-one") %>%
  left_join(eviction_by_project, by = c("hex_id", "project_id"), relationship = "one-to-one") %>%
  left_join(requests_311_by_project, by = c("hex_id", "project_id"), relationship = "one-to-one") %>%
  left_join(demolition_by_project, by = c("hex_id", "project_id"), relationship = "one-to-one") %>%
  mutate(
    across(all_of(event_columns), ~ coalesce(as.numeric(.x), 0)),
    across(
      c(
        parcels_in_hex, feature_units_in_hex, promoted_units_in_hex,
        feature_corporate_units_in_hex,
        feature_financialized_parcels_in_hex
      ),
      ~ coalesce(as.numeric(.x), 0)
    ),
    project_has_parcel_in_event_hex = parcels_in_hex > 0
  ) %>%
  left_join(
    candidate_hexes %>%
      select(
        hex_id,
        display_cluster,
        tentative_name,
        concern_level,
        residential_units,
        residential_parcels,
        eviction_cases_total,
        eviction_cases_latest_12mo,
        eviction_cases_previous_12mo,
        sr_311_smoke_signal_latest_12mo,
        sr_311_smoke_signal_previous_12mo,
        demo_latest_24mo,
        demo_previous_24mo,
        demo_total_latest_24mo,
        corporate_owned_units,
        financialized_owner_parcels,
        eviction_pressure_index,
        demolition_pressure_index,
        sr_311_pressure_index,
        ownership_pressure_index
      ),
    by = "hex_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    eviction_latest_share = safe_share(
      eviction_cases_latest_12mo_attributed,
      eviction_cases_latest_12mo
    ),
    eviction_total_share = safe_share(
      eviction_cases_total_attributed,
      eviction_cases_total
    ),
    requests_311_latest_share = safe_share(
      requests_311_latest_12mo_attributed,
      sr_311_smoke_signal_latest_12mo
    ),
    demolition_latest_share = safe_share(
      demolition_permits_latest_24mo_attributed,
      demo_latest_24mo
    ),
    total_demolition_latest_share = safe_share(
      total_demolition_latest_24mo_attributed,
      demo_total_latest_24mo
    ),
    corporate_unit_share = safe_share(
      feature_corporate_units_in_hex,
      corporate_owned_units
    ),
    financialized_parcel_share = safe_share(
      feature_financialized_parcels_in_hex,
      financialized_owner_parcels
    ),
    ownership_driver_share = pmax(
      coalesce(corporate_unit_share, 0),
      coalesce(financialized_parcel_share, 0)
    ),
    demolition_driver_share = pmax(
      coalesce(demolition_latest_share, 0),
      coalesce(total_demolition_latest_share, 0)
    ),
    attributable_domain_weight =
      pmax(eviction_pressure_index, 0) +
      pmax(demolition_pressure_index, 0) +
      pmax(sr_311_pressure_index, 0) +
      pmax(ownership_pressure_index, 0),
    combined_driver_score = if_else(
      attributable_domain_weight > 0,
      (
        pmax(eviction_pressure_index, 0) * coalesce(eviction_latest_share, 0) +
        pmax(demolition_pressure_index, 0) * demolition_driver_share +
        pmax(sr_311_pressure_index, 0) * coalesce(requests_311_latest_share, 0) +
        pmax(ownership_pressure_index, 0) * ownership_driver_share
      ) / attributable_domain_weight,
      0
    ),
    maximum_single_domain_share = pmax(
      coalesce(eviction_latest_share, 0),
      coalesce(requests_311_latest_share, 0),
      demolition_driver_share,
      ownership_driver_share
    ),
    has_attributed_pressure_signal =
      eviction_cases_total_attributed > 0 |
      requests_311_total_attributed > 0 |
      demolition_permits_total_attributed > 0 |
      feature_corporate_units_in_hex > 0 |
      feature_financialized_parcels_in_hex > 0
  ) %>%
  group_by(hex_id) %>%
  arrange(
    desc(combined_driver_score),
    desc(maximum_single_domain_share),
    desc(promoted_units_in_hex),
    project_id,
    .by_group = TRUE
  ) %>%
  mutate(property_driver_rank = row_number()) %>%
  ungroup()

attribution_coverage <- property_drivers %>%
  group_by(hex_id) %>%
  summarise(
    eviction_latest_observed = first(eviction_cases_latest_12mo),
    eviction_latest_attributed = sum(eviction_cases_latest_12mo_attributed),
    requests_311_latest_observed = first(sr_311_smoke_signal_latest_12mo),
    code_requests_latest_attributed = sum(requests_311_latest_12mo_attributed),
    demolition_latest_observed = first(demo_latest_24mo),
    demolition_latest_attributed = sum(demolition_permits_latest_24mo_attributed),
    total_demolition_latest_observed = first(demo_total_latest_24mo),
    total_demolition_latest_attributed = sum(
      total_demolition_latest_24mo_attributed
    ),
    corporate_units_observed = first(corporate_owned_units),
    corporate_units_attributed = sum(feature_corporate_units_in_hex),
    financialized_parcels_observed = first(financialized_owner_parcels),
    financialized_parcels_attributed = sum(
      feature_financialized_parcels_in_hex
    ),
    .groups = "drop"
  ) %>%
  mutate(
    eviction_latest_attribution_share = safe_share(
      eviction_latest_attributed,
      eviction_latest_observed
    ),
    code_requests_latest_attribution_share = safe_share(
      code_requests_latest_attributed,
      requests_311_latest_observed
    ),
    demolition_latest_attribution_share = safe_share(
      demolition_latest_attributed,
      demolition_latest_observed
    ),
    total_demolition_latest_attribution_share = safe_share(
      total_demolition_latest_attributed,
      total_demolition_latest_observed
    ),
    corporate_units_reconciliation_difference =
      corporate_units_attributed - corporate_units_observed,
    financialized_parcels_reconciliation_difference =
      financialized_parcels_attributed - financialized_parcels_observed
  )

attribution_overages <- attribution_coverage %>%
  filter(
    eviction_latest_attributed > eviction_latest_observed + 1e-8 |
      code_requests_latest_attributed > requests_311_latest_observed + 1e-8 |
      demolition_latest_attributed > demolition_latest_observed + 1e-8 |
      total_demolition_latest_attributed >
        total_demolition_latest_observed + 1e-8 |
      abs(corporate_units_reconciliation_difference) > 1e-8 |
      abs(financialized_parcels_reconciliation_difference) > 1e-8
  )
if (nrow(attribution_overages) > 0L) {
  stop(
    "Property attribution does not reconcile with the clustered hex features.",
    call. = FALSE
  )
}

property_concentration <- property_drivers %>%
  group_by(hex_id) %>%
  summarise(
    attributed_projects = sum(has_attributed_pressure_signal),
    leading_project_id = project_id[which.max(combined_driver_score)],
    leading_property_address = representative_address[which.max(combined_driver_score)],
    leading_property_owner = representative_owner[which.max(combined_driver_score)],
    leading_combined_driver_score = max(combined_driver_score),
    leading_maximum_domain_share = max(maximum_single_domain_share),
    eviction_top1_share = top_n_share(eviction_latest_share, 1L),
    eviction_top3_share = top_n_share(eviction_latest_share, 3L),
    eviction_hhi_matched = concentration_hhi(
      eviction_cases_latest_12mo_attributed
    ),
    requests_311_top1_share = top_n_share(requests_311_latest_share, 1L),
    requests_311_top3_share = top_n_share(requests_311_latest_share, 3L),
    requests_311_hhi_matched = concentration_hhi(
      requests_311_latest_12mo_attributed
    ),
    demolition_top1_share = top_n_share(demolition_driver_share, 1L),
    demolition_top3_share = top_n_share(demolition_driver_share, 3L),
    demolition_hhi_matched = concentration_hhi(
      demolition_permits_latest_24mo_attributed
    ),
    ownership_top1_share = top_n_share(ownership_driver_share, 1L),
    ownership_top3_share = top_n_share(ownership_driver_share, 3L),
    ownership_hhi_matched = concentration_hhi(feature_corporate_units_in_hex),
    one_property_signal_dominance_flag = any(
      maximum_single_domain_share >= dominant_property_share
    ),
    one_property_dominance_flag =
      max(combined_driver_score) >= dominant_property_share,
    project_crosses_hex_boundary_flag = any(
      project_crosses_hex_boundary,
      na.rm = TRUE
    ),
    event_property_hex_disagreement_flag = any(
      eviction_property_hex_disagreements > 0 |
        request_property_hex_disagreements > 0 |
        demolition_property_hex_disagreements > 0,
      na.rm = TRUE
    ),
    spatial_attribution_flag =
      project_crosses_hex_boundary_flag |
      event_property_hex_disagreement_flag,
    .groups = "drop"
  )

top_properties <- property_drivers %>%
  filter(property_driver_rank <= top_properties_per_hex) %>%
  select(
    hex_id,
    property_driver_rank,
    project_id,
    representative_parcel_id,
    representative_address,
    representative_owner,
    project_grouping_methods,
    project_parcel_count,
    project_hex_count,
    project_crosses_hex_boundary,
    project_has_parcel_in_event_hex,
    parcels_in_hex,
    feature_units_in_hex,
    promoted_units_in_hex,
    unit_estimation_method,
    unit_estimation_confidence,
    project_is_corporate_owned,
    project_has_financialized_owner,
    feature_corporate_units_in_hex,
    eviction_cases_total_attributed,
    eviction_cases_latest_12mo_attributed,
    eviction_latest_share,
    requests_311_latest_12mo_attributed,
    structure_311_latest_12mo_attributed,
    requests_311_latest_share,
    demolition_permits_latest_24mo_attributed,
    total_demolition_latest_24mo_attributed,
    demolition_driver_share,
    ownership_driver_share,
    combined_driver_score,
    maximum_single_domain_share,
    eviction_match_methods,
    demolition_match_methods,
    project_latitude_in_hex,
    project_longitude_in_hex,
    representative_latitude,
    representative_longitude
  )

################################################################################
# Leading-property counterfactual
################################################################################

print_progress("Reassigning candidates after leading-property influence tests...")
reference_eviction_rate <- if_else(
  !is.na(features$eviction_rate_units_denominator),
  100 * features$eviction_cases_latest_12mo /
    features$eviction_rate_units_denominator,
  NA_real_
)
reference_311_rate <- if_else(
  !is.na(features$eviction_rate_units_denominator),
  100 * features$sr_311_smoke_signal_latest_12mo /
    features$eviction_rate_units_denominator,
  NA_real_
)

leading_properties <- property_drivers %>%
  filter(property_driver_rank == 1L) %>%
  select(
    hex_id,
    leading_project_id = project_id,
    leading_property_address = representative_address,
    leading_combined_driver_score = combined_driver_score,
    eviction_cases_total_attributed,
    eviction_cases_latest_12mo_attributed,
    eviction_cases_previous_12mo_attributed,
    requests_311_latest_12mo_attributed,
    requests_311_previous_12mo_attributed,
    demolition_permits_latest_24mo_attributed,
    demolition_permits_previous_24mo_attributed,
    total_demolition_latest_24mo_attributed,
    feature_corporate_units_in_hex,
    feature_financialized_parcels_in_hex
  )

counterfactual_rows <- vector("list", nrow(leading_properties))
for (index in seq_len(nrow(leading_properties))) {
  driver <- leading_properties[index, ]
  original <- features %>% filter(hex_id == driver$hex_id)
  if (nrow(original) != 1L) {
    stop("Counterfactual hex was not unique in the feature surface.", call. = FALSE)
  }

  eviction_total_cf <- pmax(
    original$eviction_cases_total - driver$eviction_cases_total_attributed,
    0
  )
  eviction_latest_cf <- pmax(
    original$eviction_cases_latest_12mo -
      driver$eviction_cases_latest_12mo_attributed,
    0
  )
  eviction_previous_cf <- pmax(
    original$eviction_cases_previous_12mo -
      driver$eviction_cases_previous_12mo_attributed,
    0
  )
  eviction_rate_cf <- if_else(
    !is.na(original$eviction_rate_units_denominator),
    100 * eviction_latest_cf / original$eviction_rate_units_denominator,
    NA_real_
  )
  eviction_change_cf <- if_else(
    eviction_previous_cf > 0,
    100 * (eviction_latest_cf / eviction_previous_cf - 1),
    NA_real_
  )
  eviction_share_cf <- if_else(
    eviction_total_cf > 0,
    eviction_latest_cf / eviction_total_cf,
    NA_real_
  )
  eviction_index_cf <- mean_available(c(
    robust_fixed_score(eviction_rate_cf, reference_eviction_rate),
    robust_fixed_score(
      eviction_change_cf,
      features$eviction_cases_latest_12mo_change_pct
    ),
    robust_fixed_score(eviction_share_cf, features$eviction_recent_share)
  ))

  requests_latest_cf <- pmax(
    original$sr_311_smoke_signal_latest_12mo -
      driver$requests_311_latest_12mo_attributed,
    0
  )
  requests_previous_cf <- pmax(
    original$sr_311_smoke_signal_previous_12mo -
      driver$requests_311_previous_12mo_attributed,
    0
  )
  requests_rate_cf <- if_else(
    !is.na(original$eviction_rate_units_denominator),
    100 * requests_latest_cf / original$eviction_rate_units_denominator,
    NA_real_
  )
  requests_density_cf <- requests_latest_cf / original$area_km2
  requests_change_cf <- if_else(
    requests_previous_cf > 0,
    100 * (requests_latest_cf / requests_previous_cf - 1),
    NA_real_
  )
  requests_index_cf <- mean_available(c(
    robust_fixed_score(requests_rate_cf, reference_311_rate),
    robust_fixed_score(
      requests_density_cf,
      features$sr_311_smoke_signal_latest_12mo_density
    ),
    robust_fixed_score(
      requests_change_cf,
      features$sr_311_smoke_signal_latest_12mo_change_pct
    )
  ))

  demo_latest_cf <- pmax(
    original$demo_latest_24mo - driver$demolition_permits_latest_24mo_attributed,
    0
  )
  demo_previous_cf <- pmax(
    original$demo_previous_24mo -
      driver$demolition_permits_previous_24mo_attributed,
    0
  )
  total_demo_latest_cf <- pmax(
    original$demo_total_latest_24mo -
      driver$total_demolition_latest_24mo_attributed,
    0
  )
  demo_recent_density_cf <- demo_latest_cf / original$area_km2
  demo_trend_positive_cf <- pmax(
    log1p(demo_latest_cf) - log1p(demo_previous_cf),
    0
  )
  total_demo_density_cf <- total_demo_latest_cf / original$area_km2
  demolition_index_cf <- mean_available(c(
    robust_fixed_score(demo_recent_density_cf, features$demo_recent_density),
    robust_fixed_score(demo_trend_positive_cf, features$demo_trend_positive),
    robust_fixed_score(
      total_demo_density_cf,
      features$demo_total_recent_density
    )
  ))

  corporate_units_cf <- pmax(
    original$corporate_owned_units - driver$feature_corporate_units_in_hex,
    0
  )
  financialized_parcels_cf <- pmax(
    original$financialized_owner_parcels -
      driver$feature_financialized_parcels_in_hex,
    0
  )
  pct_corporate_units_cf <- if_else(
    original$residential_units > 0,
    100 * corporate_units_cf / original$residential_units,
    NA_real_
  )
  corporate_density_cf <- corporate_units_cf / original$area_km2
  pct_financialized_cf <- if_else(
    original$residential_parcels > 0,
    100 * financialized_parcels_cf / original$residential_parcels,
    NA_real_
  )
  ownership_index_cf <- mean_available(c(
    robust_fixed_score(pct_corporate_units_cf, features$pct_corporate_units),
    robust_fixed_score(
      corporate_density_cf,
      features$corporate_owned_units_per_km2
    ),
    robust_fixed_score(
      pct_financialized_cf,
      features$pct_financialized_owner_parcels
    )
  ))

  counterfactual_rows[[index]] <- original %>%
    mutate(
      eviction_pressure_index = eviction_index_cf,
      sr_311_pressure_index = requests_index_cf,
      demolition_pressure_index = demolition_index_cf,
      ownership_pressure_index = ownership_index_cf,
      leading_project_id = driver$leading_project_id,
      leading_property_address = driver$leading_property_address,
      leading_combined_driver_score = driver$leading_combined_driver_score,
      eviction_pressure_before = original$eviction_pressure_index,
      demolition_pressure_before = original$demolition_pressure_index,
      requests_311_pressure_before = original$sr_311_pressure_index,
      ownership_pressure_before = original$ownership_pressure_index
    )
}

counterfactual_features <- bind_rows(counterfactual_rows)
counterfactual_assignments <- assign_fixed_clusters(counterfactual_features, model) %>%
  filter(assignment_status == "assigned") %>%
  select(
    hex_id,
    counterfactual_cluster = cluster,
    counterfactual_display_cluster = display_cluster,
    counterfactual_name = tentative_name,
    counterfactual_concern_level = concern_level,
    counterfactual_distance_to_centroid = distance_to_centroid,
    counterfactual_margin_confidence = margin_confidence,
    counterfactual_boundary_flag = boundary_flag
  )

counterfactuals <- counterfactual_features %>%
  st_drop_geometry() %>%
  select(
    hex_id,
    leading_project_id,
    leading_property_address,
    leading_combined_driver_score,
    eviction_pressure_before,
    eviction_pressure_after = eviction_pressure_index,
    demolition_pressure_before,
    demolition_pressure_after = demolition_pressure_index,
    requests_311_pressure_before,
    requests_311_pressure_after = sr_311_pressure_index,
    ownership_pressure_before,
    ownership_pressure_after = ownership_pressure_index
  ) %>%
  left_join(
    candidate_hexes %>%
      select(
        hex_id,
        original_cluster = cluster,
        original_display_cluster = display_cluster,
        original_name = tentative_name,
        original_concern_level = concern_level
      ),
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  left_join(counterfactual_assignments, by = "hex_id", relationship = "one-to-one") %>%
  mutate(
    cluster_changed = original_cluster != counterfactual_cluster,
    concern_level_changed =
      original_concern_level != counterfactual_concern_level,
    concern_level_reduced =
      unname(concern_ordinal[counterfactual_concern_level]) <
      unname(concern_ordinal[original_concern_level])
  )

################################################################################
# Summaries, QA, and maps
################################################################################

hex_summary <- candidate_hexes %>%
  left_join(property_concentration, by = "hex_id", relationship = "one-to-one") %>%
  left_join(
    counterfactuals %>%
      select(
        hex_id,
        counterfactual_display_cluster,
        counterfactual_name,
        counterfactual_concern_level,
        cluster_changed,
        concern_level_reduced
      ),
    by = "hex_id",
    relationship = "one-to-one"
  ) %>%
  mutate(
    area_level_only_domains = "Rent pressure; Demographic vulnerability",
    leading_property_interpretation = case_when(
      concern_level_reduced ~
        "removing the leading project's attributable signals lowers concern category",
      cluster_changed ~
        "removing the leading project's attributable signals changes typology",
      one_property_dominance_flag ~
        "one property dominates weighted attributable pressures",
      one_property_signal_dominance_flag ~
        "one property dominates one observed signal but not weighted pressures",
      TRUE ~ "no single-property classification change"
    )
  ) %>%
  select(
    hex_id,
    h3_index,
    neighborhood_name,
    display_cluster,
    tentative_name,
    concern_level,
    total_pop,
    residential_units,
    classified_ring1_neighbors,
    low_ring1_neighbors,
    low_ring1_share,
    low_ring1_population_share,
    low_ring1_unit_share,
    low_disk2_share,
    concern_ordinal_contrast,
    local_profile_contrast,
    distance_to_centroid,
    margin_confidence,
    leading_domain,
    leading_domain_score,
    second_domain,
    second_domain_score,
    all_of(domain_features),
    leading_project_id,
    leading_property_address,
    leading_property_owner,
    leading_combined_driver_score,
    leading_maximum_domain_share,
    one_property_signal_dominance_flag,
    one_property_dominance_flag,
    project_crosses_hex_boundary_flag,
    event_property_hex_disagreement_flag,
    spatial_attribution_flag,
    counterfactual_display_cluster,
    counterfactual_name,
    counterfactual_concern_level,
    cluster_changed,
    concern_level_reduced,
    leading_property_interpretation,
    area_level_only_domains
  )

unmatched_events <- bind_rows(
  eviction_events %>%
    filter(is.na(matched_project_id)) %>%
    transmute(
      source = "eviction",
      hex_id,
      event_id = eviction_event_id,
      event_date = file_date,
      address = address_for_geocoding,
      property_match_method,
      nearest_parcel_distance_m
    ),
  code_requests_sf %>%
    filter(is.na(project_id)) %>%
    transmute(
      source = "311_code_case",
      hex_id,
      event_id = service_request_id,
      event_date = request_date,
      address = NA_character_,
      property_match_method = if_else(
        exact_residential_parcel_count == 0L,
        "no_exact_residential_parcel",
        "ambiguous_residential_parcel"
      ),
      nearest_parcel_distance_m = NA_real_
    ),
  demolition_events %>%
    filter(is.na(matched_project_id)) %>%
    transmute(
      source = "demolition",
      hex_id,
      event_id = demolition_event_id,
      event_date = issue_date,
      address = .data[["Original Address 1"]],
      property_match_method,
      nearest_parcel_distance_m
    )
)

attribution_qa <- bind_rows(
  eviction_events %>%
    summarise(
      source = "eviction",
      candidate_events = n(),
      matched_events = sum(!is.na(matched_project_id)),
      exact_or_address_events = sum(
        property_match_method %in% c(
          "exact_source_parcel_id",
          "unique_normalized_address"
        )
      ),
      nearest_events = sum(
        property_match_method == "nearest_parcel_same_house_number"
      ),
      unmatched_events = sum(is.na(matched_project_id)),
      property_hex_disagreements = sum(event_property_hex_disagreement, na.rm = TRUE)
    ),
  code_requests_sf %>%
    summarise(
      source = "311_code_case",
      candidate_events = n(),
      matched_events = sum(!is.na(project_id)),
      exact_or_address_events = sum(!is.na(project_id)),
      nearest_events = 0L,
      unmatched_events = sum(is.na(project_id)),
      property_hex_disagreements = sum(event_property_hex_disagreement, na.rm = TRUE)
    ),
  demolition_events %>%
    summarise(
      source = "demolition",
      candidate_events = n(),
      matched_events = sum(!is.na(matched_project_id)),
      exact_or_address_events = sum(
        property_match_method %in% c(
          "exact_source_parcel_id",
          "unique_normalized_address"
        )
      ),
      nearest_events = sum(
        property_match_method == "nearest_parcel_same_house_number"
      ),
      unmatched_events = sum(is.na(matched_project_id)),
      property_hex_disagreements = sum(event_property_hex_disagreement, na.rm = TRUE)
    )
) %>%
  mutate(match_pct = 100 * safe_share(matched_events, candidate_events))

review_metrics <- tibble(
  metric = c(
    "eligible_part1_hexes",
    "high_or_very_high_hexes",
    "island_candidates",
    "demolition_led_islands",
    "eviction_led_islands",
    "one_property_signal_dominance_hexes",
    "one_property_weighted_dominance_hexes",
    "leading_property_cluster_changes",
    "leading_property_concern_reductions",
    "projects_crossing_hex_boundary_hexes",
    "event_parcel_hex_disagreement_hexes",
    "spatial_attribution_flag_hexes"
  ),
  value = c(
    nrow(cluster_hex),
    sum(cluster_hex$display_cluster %in% high_display_clusters),
    nrow(hex_summary),
    sum(hex_summary$display_cluster == 6L),
    sum(hex_summary$display_cluster == 7L),
    sum(hex_summary$one_property_signal_dominance_flag, na.rm = TRUE),
    sum(hex_summary$one_property_dominance_flag, na.rm = TRUE),
    sum(hex_summary$cluster_changed, na.rm = TRUE),
    sum(hex_summary$concern_level_reduced, na.rm = TRUE),
    sum(hex_summary$project_crosses_hex_boundary_flag, na.rm = TRUE),
    sum(hex_summary$event_property_hex_disagreement_flag, na.rm = TRUE),
    sum(hex_summary$spatial_attribution_flag, na.rm = TRUE)
  )
)

print_progress("Creating high-risk-island review maps...")
map_data <- features_sf %>%
  select(hex_id) %>%
  inner_join(
    cluster_hex %>%
      select(hex_id, display_cluster, tentative_name, concern_level, map_color),
    by = "hex_id",
    relationship = "one-to-one"
  )
candidate_map <- map_data %>%
  filter(hex_id %in% hex_summary$hex_id) %>%
  left_join(
    hex_summary %>%
      select(
        hex_id,
        neighborhood_name,
        low_ring1_share,
        leading_domain,
        leading_property_address,
        leading_property_interpretation
      ),
    by = "hex_id",
    relationship = "one-to-one"
  )

jurisdictions <- st_read(JURISDICTION_FILE, quiet = TRUE)
full_purpose <- jurisdictions %>%
  filter(jurisdiction_label == "AUSTIN FULL PURPOSE") %>%
  st_make_valid() %>%
  st_union()

static_map <- ggplot() +
  geom_sf(data = map_data, fill = "#E7E8E6", color = "#FFFFFF", linewidth = 0.08) +
  geom_sf(
    data = candidate_map,
    aes(fill = factor(display_cluster)),
    color = "#202020",
    linewidth = 0.35
  ) +
  geom_sf(data = full_purpose, fill = NA, color = "#222222", linewidth = 0.55) +
  scale_fill_manual(
    values = c("6" = "#FB6A4A", "7" = "#CB181D"),
    labels = c("6" = "Cluster 6: Demolition-led", "7" = "Cluster 7: Eviction-led"),
    name = "Island candidate"
  ) +
  coord_sf(crs = 2277, datum = NA) +
  labs(
    title = "High-risk hexes surrounded by low-risk neighbors",
    subtitle = "Each outlined hex meets the same configured island screen",
    caption = paste0(
      "Part 1 baseline through ", EWS_CONFIG$analysis_as_of_date,
      "; screening diagnostic, not a new risk category"
    )
  ) +
  theme_void(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 10),
    plot.caption = element_text(size = 8, color = "#555555"),
    legend.position = "bottom"
  )

static_path <- file.path(FIGURES_DIR, "03h_high_risk_islands.png")
ggsave(static_path, static_map, width = 11, height = 9, dpi = 220, bg = "white")

interactive_candidates <- candidate_map %>%
  st_transform(4326) %>%
  mutate(
    popup_html = paste0(
      "<div style='min-width:280px;line-height:1.35'>",
      "<strong>High-risk island candidate</strong><br>",
      "Cluster ", display_cluster, ": ", tentative_name, "<br>",
      "<strong>Neighborhood:</strong> ", coalesce(neighborhood_name, "Not assigned"), "<br>",
      "<strong>Low-risk immediate neighbors:</strong> ",
      round(100 * low_ring1_share, 1), "%<br>",
      "<strong>Leading domain:</strong> ", leading_domain, "<br>",
      "<strong>Leading property:</strong> ",
      coalesce(leading_property_address, "No attributed property"), "<br>",
      "<strong>Influence result:</strong> ", leading_property_interpretation,
      "</div>"
    )
  )

interactive_map <- leaflet(
  options = leafletOptions(preferCanvas = TRUE, minZoom = 8)
) %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  addPolygons(
    data = st_transform(map_data, 4326),
    fillColor = "#C9CCCA",
    fillOpacity = 0.12,
    color = "#FFFFFF",
    weight = 0.25,
    opacity = 0.35,
    options = pathOptions(interactive = FALSE)
  ) %>%
  addPolygons(
    data = interactive_candidates,
    fillColor = ~ ifelse(display_cluster == 6L, "#FB6A4A", "#CB181D"),
    fillOpacity = 0.68,
    color = "#202020",
    weight = 1.2,
    popup = ~ popup_html,
    highlightOptions = highlightOptions(
      weight = 3,
      color = "#111111",
      fillOpacity = 0.88,
      bringToFront = TRUE
    )
  ) %>%
  addLegend(
    position = "bottomright",
    colors = c("#FB6A4A", "#CB181D"),
    labels = c("Cluster 6: Demolition-led", "Cluster 7: Eviction-led"),
    title = "High-risk island candidates",
    opacity = 0.9
  ) %>%
  addScaleBar(
    position = "bottomleft",
    options = scaleBarOptions(metric = TRUE, imperial = FALSE)
  )

interactive_path <- file.path(FIGURES_DIR, "03h_high_risk_islands_interactive.html")
saveWidget(
  interactive_map,
  file = interactive_path,
  selfcontained = TRUE,
  title = "Part 1 High-Risk Island Review"
)

print_progress("Writing high-risk-island review outputs...")
write_csv(hex_summary, file.path(PART1_DIR, "high_risk_island_hex_summary.csv"))
write_csv(
  ring1_context %>% filter(focal_hex_id %in% candidate_ids),
  file.path(PART1_DIR, "high_risk_island_neighbor_context.csv")
)
write_csv(
  property_drivers,
  file.path(PART1_DIR, "high_risk_island_property_drivers.csv")
)
write_csv(
  top_properties,
  file.path(PART1_DIR, "high_risk_island_top_properties.csv")
)
write_csv(
  counterfactuals,
  file.path(PART1_DIR, "high_risk_island_counterfactuals.csv")
)
write_csv(
  unmatched_events,
  file.path(PART1_DIR, "high_risk_island_unmatched_events.csv")
)
write_csv(
  attribution_qa,
  file.path(PART1_DIR, "high_risk_island_attribution_qa.csv")
)
write_csv(
  attribution_coverage,
  file.path(PART1_DIR, "high_risk_island_attribution_coverage.csv")
)
write_csv(
  review_metrics,
  file.path(PART1_DIR, "high_risk_island_review_metrics.csv")
)
saveRDS(
  list(
    parameters = parameters,
    hex_summary = hex_summary,
    neighbor_context = ring1_context %>% filter(focal_hex_id %in% candidate_ids),
    property_drivers = property_drivers,
    top_properties = top_properties,
    counterfactuals = counterfactuals,
    unmatched_events = unmatched_events,
    attribution_qa = attribution_qa,
    attribution_coverage = attribution_coverage,
    review_metrics = review_metrics
  ),
  file.path(PART1_DIR, "high_risk_island_review.rds")
)

print_progress(
  paste0(
    "Identified ", nrow(hex_summary), " island candidates; ",
    sum(hex_summary$one_property_dominance_flag),
    " have a property contributing at least ",
    scales::percent(dominant_property_share, accuracy = 1),
    " of weighted property-attributable pressure."
  )
)
cat("Hex summary: output/part1/high_risk_island_hex_summary.csv\n")
cat("Top properties: output/part1/high_risk_island_top_properties.csv\n")
cat("Interactive map: figures/03h_high_risk_islands_interactive.html\n")
