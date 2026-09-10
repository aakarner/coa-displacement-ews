################################################################################
# Build Complete Part 3 Eviction Outcome Panel
################################################################################
#
# Creates one row for every analysis hex and calendar year represented by the
# Travis County JP1-JP5 and Williamson County JP1/JP2 exports. Complete-year
# labels use unique filed cases. Williamson zeroes are available only within a
# supplied court's effective-dated official precinct; JP3/JP4 and Hays remain
# missing. Filing points and zero-eligible hex centers must also fall inside the
# April 2026 City of Austin full-purpose boundary, which defines the fixed pilot
# study geography. Calendar year 2026 is retained as an explicitly partial
# observation but is never exposed as a completed outcome label.
#
# INPUTS:
#   - output/hex_grid.rds
#   - output/eviction_filings_prepared_for_geocoding.csv
#   - output/eviction_addresses_geocoded.csv
#   - output/williamson_eviction_filings_prepared_for_geocoding.csv
#   - output/williamson_eviction_addresses_geocoded_with_arcgis.csv
#   - config/hex_county_assignment_2024.csv
#   - config/eviction_sources.csv
#   - config/williamson_jp_hex_assignment.csv
#
# OUTPUTS:
#   - output/eviction_filings_complete_by_hex_year.csv
#   - output/part3/eviction_case_assignment_issues.csv
#   - output/part3/eviction_case_assignment_summary.csv
#   - output/part3/eviction_complete_panel_qa.csv
#   - output/part3/eviction_source_coverage_qa.csv
#   - output/part3/eviction_source_geography_qa.csv
#   - output/part3/eviction_panel_source_manifest.csv
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(tidyr)
})

source(here::here("R/utils.R"))
source(here::here("R/analysis_config.R"))
source(here::here("R/eviction_panel.R"))
source(here::here("R/eviction_coverage.R"))

print_header("PART 3 - BUILD COMPLETE EVICTION OUTCOME PANEL")

OUTPUT_DIR <- here::here("output")
PART3_DIR <- file.path(OUTPUT_DIR, "part3")
REQUESTED_OBSERVED_THROUGH_DATE <- EWS_CONFIG$analysis_as_of_date

dir.create(PART3_DIR, recursive = TRUE, showWarnings = FALSE)

input_files <- c(
  hex_grid = file.path(OUTPUT_DIR, "hex_grid.rds"),
  source_filings = file.path(
    OUTPUT_DIR,
    "eviction_filings_prepared_for_geocoding.csv"
  ),
  travis_geocoded_addresses = file.path(
    OUTPUT_DIR,
    "eviction_addresses_geocoded.csv"
  ),
  williamson_source_filings = file.path(
    OUTPUT_DIR,
    "williamson_eviction_filings_prepared_for_geocoding.csv"
  ),
  williamson_geocoded_addresses = file.path(
    OUTPUT_DIR,
    "williamson_eviction_addresses_geocoded_with_arcgis.csv"
  ),
  williamson_census_geocode_qa = file.path(
    OUTPUT_DIR,
    "williamson_eviction_geocode_qa.csv"
  ),
  williamson_arcgis_geocode_qa = file.path(
    OUTPUT_DIR,
    "williamson_eviction_geocode_arcgis_qa.csv"
  ),
  williamson_address_reference_qa = file.path(
    OUTPUT_DIR,
    "williamson_address_reference_qa.csv"
  ),
  williamson_local_geocode_qa = file.path(
    OUTPUT_DIR,
    "williamson_eviction_geocode_local_qa.csv"
  ),
  williamson_coa_geocode_qa = file.path(
    OUTPUT_DIR,
    "williamson_eviction_geocode_coa_qa.csv"
  ),
  current_city_jurisdictions = here::here(
    "data",
    "BOUNDARIES_jurisdictions_20260429.geojson"
  ),
  source_config = here::here("config", "eviction_sources.csv"),
  hex_counties = here::here("config", "hex_county_assignment_2024.csv"),
  hex_county_metadata = here::here(
    "config",
    "hex_county_assignment_2024_metadata.csv"
  ),
  williamson_jp_hexes = here::here(
    "config",
    "williamson_jp_hex_assignment.csv"
  ),
  williamson_jp_metadata = here::here(
    "config",
    "williamson_jp_hex_assignment_metadata.csv"
  )
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing eviction panel input(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

print_progress("Loading the analysis grid and county/court source references...")
hex_grid <- readRDS(input_files[["hex_grid"]])
if (
  !inherits(hex_grid, "sf") ||
    is.na(st_crs(hex_grid)) ||
    !all(c("hex_id", "h3_index") %in% names(hex_grid)) ||
    anyNA(hex_grid$hex_id) ||
    anyDuplicated(hex_grid$hex_id)
) {
  stop(
    "The analysis grid must be an sf object with a CRS and unique hex IDs.",
    call. = FALSE
  )
}

CITY_BOUNDARY_SNAPSHOT_DATE <- as.Date("2026-04-29")
CITY_STUDY_GEOGRAPHY <- "current_austin_full_purpose_fixed"
CITY_HEX_ASSIGNMENT_METHOD <- "hex_point_on_surface_within_current_city_full"
SPATIAL_ANALYSIS_CRS <- 3083

current_jurisdictions <- st_read(
  input_files[["current_city_jurisdictions"]],
  quiet = TRUE
)
if (
  !inherits(current_jurisdictions, "sf") ||
    is.na(st_crs(current_jurisdictions)) ||
    !all(c("city_name", "jurisdiction_type") %in%
      names(current_jurisdictions))
) {
  stop(
    "The current jurisdiction snapshot lacks the required City fields or CRS.",
    call. = FALSE
  )
}
current_austin_full_rows <- current_jurisdictions |>
  filter(
    toupper(trimws(as.character(.data$city_name))) == "CITY OF AUSTIN",
    toupper(trimws(as.character(.data$jurisdiction_type))) == "FULL"
  ) |>
  st_make_valid() |>
  st_transform(SPATIAL_ANALYSIS_CRS)
if (nrow(current_austin_full_rows) == 0L) {
  stop(
    "The jurisdiction snapshot contains no City of Austin FULL polygon.",
    call. = FALSE
  )
}
current_austin_full <- st_sf(
  city_study_geography = CITY_STUDY_GEOGRAPHY,
  geometry = st_union(st_geometry(current_austin_full_rows))
)

hex_grid_projected <- st_transform(hex_grid, SPATIAL_ANALYSIS_CRS)
hex_centers_projected <- suppressWarnings(
  st_point_on_surface(hex_grid_projected)
)
hex_center_inside_city <- lengths(
  st_covered_by(hex_centers_projected, current_austin_full)
) > 0L
hex_intersects_city <- lengths(
  st_intersects(hex_grid_projected, current_austin_full)
) > 0L
city_hex_reference <- hex_grid |>
  st_drop_geometry() |>
  transmute(
    .data$hex_id,
    .data$h3_index,
    city_study_geography = CITY_STUDY_GEOGRAPHY,
    city_boundary_snapshot_date = CITY_BOUNDARY_SNAPSHOT_DATE,
    city_hex_assignment_method = CITY_HEX_ASSIGNMENT_METHOD,
    hex_center_inside_current_austin_full = hex_center_inside_city,
    hex_intersects_current_austin_full = hex_intersects_city
  )
if (
  nrow(city_hex_reference) != nrow(hex_grid) ||
    anyDuplicated(city_hex_reference$hex_id) ||
    !any(city_hex_reference$hex_center_inside_current_austin_full) ||
    all(city_hex_reference$hex_center_inside_current_austin_full) ||
    any(
      city_hex_reference$hex_center_inside_current_austin_full &
        !city_hex_reference$hex_intersects_current_austin_full
    )
) {
  stop("The fixed City study-hex reference is structurally invalid.", call. = FALSE)
}

source_config <- read_csv(
  input_files[["source_config"]],
  col_types = cols(
    source_id = col_character(),
    source_county = col_character(),
    jp_district = col_character(),
    path = col_character(),
    sheet = col_character(),
    parser = col_character(),
    source_period_start = col_date(),
    source_period_end = col_date(),
    coverage_geography = col_character(),
    notes = col_character()
  ),
  show_col_types = FALSE
)
SOURCE_START_DATE <- min(source_config$source_period_start)
OBSERVED_THROUGH_DATE <- REQUESTED_OBSERVED_THROUGH_DATE

print_progress("Loading and namespacing Travis and Williamson filing records...")
travis_source_filings <- read_csv(
  input_files[["source_filings"]],
  show_col_types = FALSE
) |>
  mutate(
    file_date = as.Date(.data$file_date),
    source_county = "Travis",
    case_number = eviction_panel_normalize_case_number(.data$case_number),
    case_uid = if_else(
      !is.na(.data$case_number) & !is.na(.data$jp_district),
      paste("Travis", .data$jp_district, .data$case_number, sep = ":"),
      NA_character_
    ),
    geocode_registry = "travis_reviewed"
  ) |>
  select(
    "case_uid", "case_number", "file_date", "source_county",
    "jp_district", "address_for_geocoding", "geocoding_candidate",
    "geocode_registry"
  )
williamson_source_filings <- read_csv(
  input_files[["williamson_source_filings"]],
  show_col_types = FALSE
) |>
  mutate(
    file_date = as.Date(.data$file_date),
    source_county = "Williamson",
    geocode_registry = "williamson_geocode_cascade"
  ) |>
  select(
    "case_uid", "case_number", "file_date", "source_county",
    "jp_district", "address_for_geocoding", "geocoding_candidate",
    "geocode_registry"
  )

required_prepared_columns <- c(
  "case_uid", "file_date", "source_county", "jp_district",
  "address_for_geocoding", "geocoding_candidate", "geocode_registry"
)
if (!all(required_prepared_columns %in% names(travis_source_filings)) ||
    !all(required_prepared_columns %in% names(williamson_source_filings))) {
  stop(
    "Prepared eviction filings are missing the county-aware geocoding contract.",
    call. = FALSE
  )
}

source_filings_all <- bind_rows(
  travis_source_filings,
  williamson_source_filings
) |>
  mutate(
    case_number_raw = .data$case_number,
    case_number = .data$case_uid
  )
source_filings <- source_filings_all |>
  filter(
    !is.na(.data$file_date),
    .data$file_date >= SOURCE_START_DATE,
    .data$file_date <= OBSERVED_THROUGH_DATE
  )

read_geocode_registry <- function(path, registry) {
  read_csv(
    path,
    col_types = cols(
      address_for_geocoding = col_character(),
      status = col_character(),
      score = col_double(),
      longitude = col_double(),
      latitude = col_double(),
      .default = col_skip()
    ),
    show_col_types = FALSE
  ) |>
    mutate(geocode_registry = registry, .before = 1)
}
travis_geocodes <- read_geocode_registry(
  input_files[["travis_geocoded_addresses"]],
  "travis_reviewed"
)
williamson_geocodes <- read_geocode_registry(
  input_files[["williamson_geocoded_addresses"]],
  "williamson_geocode_cascade"
)

validate_geocode_registry <- function(filings, geocodes, registry_label) {
  candidate_addresses <- filings |>
    filter(
      .data$geocoding_candidate %in% TRUE,
      !is.na(.data$address_for_geocoding),
      .data$address_for_geocoding != ""
    ) |>
    distinct(.data$address_for_geocoding) |>
    pull(.data$address_for_geocoding)
  geocode_addresses <- geocodes$address_for_geocoding
  if (
    anyNA(geocode_addresses) ||
      any(geocode_addresses == "") ||
      anyDuplicated(geocode_addresses) ||
      !setequal(candidate_addresses, geocode_addresses)
  ) {
    stop(
      registry_label,
      " geocodes must contain exactly one row for every prepared candidate ",
      "address. Expected ",
      length(candidate_addresses),
      "; found ",
      length(unique(stats::na.omit(geocode_addresses))),
      ".",
      call. = FALSE
    )
  }
  invisible(TRUE)
}
validate_geocode_registry(
  travis_source_filings,
  travis_geocodes,
  "Travis reviewed"
)
validate_geocode_registry(
  williamson_source_filings,
  williamson_geocodes,
  "Williamson local/City/Census/ArcGIS cascade"
)
geocoded_addresses <- bind_rows(travis_geocodes, williamson_geocodes)

reliable_geocoded_rows <- source_filings |>
  inner_join(
    geocoded_addresses,
    by = c("geocode_registry", "address_for_geocoding"),
    na_matches = "never"
  ) |>
  filter(
    .data$status %in% c("M", "T"),
    .data$score >= 90,
    is.finite(.data$longitude),
    is.finite(.data$latitude),
    dplyr::between(.data$longitude, -180, 180),
    dplyr::between(.data$latitude, -90, 90)
  )
reliably_geocoded_case_numbers <- reliable_geocoded_rows |>
  distinct(.data$case_number) |>
  pull(.data$case_number)

print_progress("Loading the versioned 2024 hex-to-county assignment...")
hex_county_reference <- read_csv(
  input_files[["hex_counties"]],
  col_types = cols(
    hex_id = col_integer(),
    h3_index = col_character(),
    source_county = col_character()
  ),
  show_col_types = FALSE
)
hex_county_metadata <- read_csv(
  input_files[["hex_county_metadata"]],
  show_col_types = FALSE
)
grid_reference <- hex_grid |>
  st_drop_geometry() |>
  select("hex_id", "h3_index")
if (
  nrow(hex_county_metadata) != 1L ||
    hex_county_metadata$grid_rows[[1]] != nrow(grid_reference) ||
  nrow(hex_county_reference) != nrow(grid_reference) ||
    anyNA(hex_county_reference[c("hex_id", "h3_index", "source_county")]) ||
    anyDuplicated(hex_county_reference$hex_id) ||
    !identical(
      as.integer(arrange(hex_county_reference, .data$hex_id)$hex_id),
      as.integer(arrange(grid_reference, .data$hex_id)$hex_id)
    ) ||
    !identical(
      as.character(arrange(hex_county_reference, .data$hex_id)$h3_index),
      as.character(arrange(grid_reference, .data$hex_id)$h3_index)
    )
) {
  stop(
    "The versioned hex-to-county reference does not match the analysis grid.",
    call. = FALSE
  )
}
hex_counties <- hex_county_reference |>
  select("hex_id", "source_county")

print_progress(
  "Assigning reliable geocodes within the City, county, and grid references..."
)
grid_with_county <- hex_grid_projected |>
  left_join(
    hex_counties |>
      rename(target_source_county = "source_county"),
    by = "hex_id"
  ) |>
  left_join(
    city_hex_reference |>
      select(
        "hex_id",
        "hex_center_inside_current_austin_full",
        "hex_intersects_current_austin_full"
      ),
    by = "hex_id"
  ) |>
  select(
    "hex_id",
    "target_source_county",
    "hex_center_inside_current_austin_full",
    "hex_intersects_current_austin_full"
  )
reliable_geocode_points <- reliable_geocoded_rows |>
  st_as_sf(
    coords = c("longitude", "latitude"),
    crs = 4326,
    remove = FALSE
  ) |>
  st_transform(SPATIAL_ANALYSIS_CRS)
point_inside_current_austin_full <- lengths(
  st_covered_by(reliable_geocode_points, current_austin_full)
) > 0L
reliable_geocode_points <- reliable_geocode_points |>
  mutate(
    point_inside_current_austin_full = point_inside_current_austin_full
  )
reliable_geocode_grid_evidence <- reliable_geocode_points |>
  st_join(
    grid_with_county,
    join = st_within,
    left = TRUE
  ) |>
  mutate(
    inside_source_county = !is.na(.data$hex_id) &
      .data$source_county == .data$target_source_county,
    inside_city_study_hex = !is.na(.data$hex_id) &
      coalesce(.data$hex_center_inside_current_austin_full, FALSE)
  )
if (nrow(reliable_geocode_grid_evidence) != nrow(reliable_geocoded_rows)) {
  stop(
    "At least one reliable eviction point joined to multiple analysis hexes.",
    call. = FALSE
  )
}

print_progress("Building effective-dated Travis/Williamson source coverage...")
williamson_jp_reference <- read_csv(
  input_files[["williamson_jp_hexes"]],
  col_types = cols(
    hex_id = col_integer(),
    h3_index = col_character(),
    effective_start_date = col_date(),
    effective_end_date = col_date(),
    jp_district = col_character(),
    boundary_vintage = col_character(),
    assignment_method = col_character(),
    assignment_status = col_character(),
    boundary_source_url = col_character()
  ),
  show_col_types = FALSE
)
williamson_jp_metadata <- read_csv(
  input_files[["williamson_jp_metadata"]],
  show_col_types = FALSE
)
williamson_hex_count <- sum(hex_counties$source_county == "Williamson")
if (nrow(williamson_jp_metadata) != 2L ||
    nrow(williamson_jp_reference) != 2L * williamson_hex_count ||
    anyDuplicated(williamson_jp_reference[c(
      "hex_id", "effective_start_date"
    )]) ||
    !setequal(
      williamson_jp_reference$hex_id,
      hex_counties$hex_id[hex_counties$source_county == "Williamson"]
    )) {
  stop(
    "The versioned Williamson JP reference does not match the analysis grid.",
    call. = FALSE
  )
}
outcome_years <- seq.int(
  as.integer(format(SOURCE_START_DATE, "%Y")),
  as.integer(format(OBSERVED_THROUGH_DATE, "%Y"))
)
hex_year_coverage <- build_eviction_hex_year_coverage(
  hex_counties = hex_counties,
  outcome_years = outcome_years,
  source_config = source_config,
  williamson_jp_reference = williamson_jp_reference,
  analysis_as_of_date = REQUESTED_OBSERVED_THROUGH_DATE
) |>
  left_join(
    city_hex_reference |>
      select(
        "hex_id",
        "hex_center_inside_current_austin_full",
        "hex_intersects_current_austin_full"
      ),
    by = "hex_id",
    relationship = "many-to-one"
  ) |>
  mutate(
    source_covered_before_city_filter = .data$source_covered,
    source_covered = .data$source_covered &
      .data$hex_center_inside_current_austin_full,
    coverage_start_date = if_else(
      .data$source_covered,
      .data$coverage_start_date,
      as.Date(NA)
    ),
    coverage_end_date = if_else(
      .data$source_covered,
      .data$coverage_end_date,
      as.Date(NA)
    ),
    uncovered_reason = case_when(
      !.data$hex_center_inside_current_austin_full ~
        "outside_current_austin_full_purpose_center_selected_grid",
      .data$source_covered ~ NA_character_,
      TRUE ~ .data$uncovered_reason
    )
  )
if (
  anyNA(hex_year_coverage$hex_center_inside_current_austin_full) ||
    any(
      hex_year_coverage$source_covered &
        !hex_year_coverage$hex_center_inside_current_austin_full
    )
) {
  stop("City study-geography masking failed for eviction coverage.", call. = FALSE)
}

# A court-specific filing can only contribute inside the court geography that
# applies in its filing year. Travis is covered by all five supplied courts;
# Williamson is constrained to the effective-dated JP assignment.
reliable_geocode_grid_evidence <- reliable_geocode_grid_evidence |>
  mutate(outcome_year = as.integer(format(.data$file_date, "%Y"))) |>
  left_join(
    hex_year_coverage |>
      select(
        "hex_id", "outcome_year", "source_covered",
        "source_covered_before_city_filter",
        "coverage_jp_district"
      ),
    by = c("hex_id", "outcome_year")
  ) |>
  mutate(
    inside_source_jp = case_when(
      !.data$inside_source_county ~ FALSE,
      .data$source_county == "Travis" ~ TRUE,
      .data$source_county == "Williamson" ~
        coalesce(.data$source_covered_before_city_filter, FALSE) &
          .data$jp_district == .data$coverage_jp_district,
      TRUE ~ FALSE
    ),
    inside_source_geography = .data$point_inside_current_austin_full &
      .data$inside_city_study_hex &
      .data$inside_source_county &
      .data$inside_source_jp,
    source_geography_status = case_when(
      is.na(.data$hex_id) ~ "outside_analysis_grid",
      !.data$point_inside_current_austin_full ~
        "outside_current_austin_full_purpose",
      !.data$inside_city_study_hex ~
        "inside_city_point_outside_center_selected_city_hexes",
      !.data$inside_source_county ~ "cross_county_hex",
      .data$source_county == "Williamson" & !.data$inside_source_jp ~
        "outside_effective_source_jp",
      .data$inside_source_geography ~ "inside_source_geography",
      TRUE ~ "outside_source_geography"
    )
  )

hex_filings <- reliable_geocode_grid_evidence |>
  filter(.data$inside_source_geography)
if (any(
  is.na(hex_filings$hex_id) |
    !hex_filings$point_inside_current_austin_full |
    !hex_filings$inside_city_study_hex |
    hex_filings$source_county != hex_filings$target_source_county |
    (
      hex_filings$source_county == "Williamson" &
        hex_filings$jp_district != hex_filings$coverage_jp_district
    )
)) {
  stop(
    "At least one accepted eviction location violates its source geography.",
    call. = FALSE
  )
}
reliably_geocoded_outside_grid_case_numbers <-
  reliable_geocode_grid_evidence |>
    filter(is.na(.data$hex_id)) |>
    st_drop_geometry() |>
    distinct(.data$case_number) |>
    filter(!is.na(.data$case_number)) |>
    pull(.data$case_number)
reliably_geocoded_outside_study_case_numbers <-
  reliable_geocode_grid_evidence |>
    filter(
      !is.na(.data$hex_id),
      !coalesce(.data$inside_source_geography, FALSE)
    ) |>
    st_drop_geometry() |>
    distinct(.data$case_number) |>
    filter(!is.na(.data$case_number)) |>
    pull(.data$case_number)
source_geography_qa <- reliable_geocode_grid_evidence |>
  st_drop_geometry() |>
  filter(!is.na(.data$case_number)) |>
  distinct(
    .data$case_number,
    .data$outcome_year,
    .data$source_county,
    .data$jp_district,
    .data$point_inside_current_austin_full,
    .data$inside_city_study_hex,
    .data$inside_source_county,
    .data$inside_source_jp,
    .data$source_geography_status
  ) |>
  count(
    .data$outcome_year,
    .data$source_county,
    source_jp_district = .data$jp_district,
    .data$point_inside_current_austin_full,
    .data$inside_city_study_hex,
    .data$inside_source_county,
    .data$inside_source_jp,
    .data$source_geography_status,
    name = "unique_cases"
  ) |>
  arrange(
    .data$outcome_year,
    .data$source_county,
    .data$source_jp_district,
    .data$source_geography_status
  )

print_progress("Resolving one analysis hex per unique filed case...")
resolved_cases <- resolve_eviction_case_hexes(
  source_filings,
  hex_filings,
  reliably_geocoded_case_numbers = reliably_geocoded_case_numbers,
  reliably_geocoded_outside_grid_case_numbers =
    reliably_geocoded_outside_grid_case_numbers,
  reliably_geocoded_outside_study_case_numbers =
    reliably_geocoded_outside_study_case_numbers
)

print_progress("Expanding to the complete covered and uncovered hex-year grid...")
eviction_panel <- build_complete_eviction_panel(
  hex_counties = hex_counties,
  assigned_cases = resolved_cases$assigned_cases,
  source_start_date = SOURCE_START_DATE,
  observed_through_date = OBSERVED_THROUGH_DATE,
  analysis_as_of_date = REQUESTED_OBSERVED_THROUGH_DATE,
  uncertain_hex_years = resolved_cases$uncertain_hex_years,
  hex_year_coverage = hex_year_coverage
) |>
  left_join(
    city_hex_reference,
    by = "hex_id",
    relationship = "many-to-one"
  ) |>
  relocate(
    "city_study_geography",
    "city_boundary_snapshot_date",
    "city_hex_assignment_method",
    "hex_center_inside_current_austin_full",
    "hex_intersects_current_austin_full",
    .after = "county_assignment_method"
  )

panel_qa <- summarize_complete_eviction_panel(
  eviction_panel,
  resolved_cases
)

if (any(
  eviction_panel$source_covered &
    !eviction_panel$hex_center_inside_current_austin_full
)) {
  stop("A hex outside the fixed City study geography was covered.", call. = FALSE)
}
if (any(!panel_qa$assigned_count_reconciles)) {
  stop("Eviction case assignments do not reconcile to panel totals.", call. = FALSE)
}

assignment_summary <- resolved_cases$cases |>
  count(
    .data$source_county,
    .data$source_jp_district,
    .data$assignment_status,
    name = "unique_cases"
  ) |>
  arrange(.data$source_county, .data$source_jp_district, desc(.data$unique_cases))

coverage_qa <- eviction_panel |>
  group_by(
    .data$outcome_year,
    .data$city_study_geography,
    .data$city_boundary_snapshot_date,
    .data$city_hex_assignment_method,
    .data$hex_center_inside_current_austin_full,
    .data$hex_intersects_current_austin_full,
    .data$source_county,
    .data$coverage_jp_district,
    .data$coverage_reason
  ) |>
  summarize(
    hex_year_rows = n(),
    source_covered_hexes = sum(.data$source_covered),
    count_observed_hexes = sum(.data$count_observed),
    eviction_cases = sum(.data$eviction_cases, na.rm = TRUE),
    eviction_cases_observed_to_date = sum(
      .data$eviction_cases_observed_to_date,
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |>
  arrange(
    .data$outcome_year,
    desc(.data$hex_center_inside_current_austin_full),
    .data$source_county,
    .data$coverage_jp_district,
    .data$coverage_reason
  )

panel_file <- file.path(
  OUTPUT_DIR,
  "eviction_filings_complete_by_hex_year.csv"
)
issues_file <- file.path(PART3_DIR, "eviction_case_assignment_issues.csv")
assignment_summary_file <- file.path(
  PART3_DIR,
  "eviction_case_assignment_summary.csv"
)
panel_qa_file <- file.path(PART3_DIR, "eviction_complete_panel_qa.csv")
coverage_qa_file <- file.path(PART3_DIR, "eviction_source_coverage_qa.csv")
source_geography_qa_file <- file.path(
  PART3_DIR,
  "eviction_source_geography_qa.csv"
)
source_manifest_file <- file.path(
  PART3_DIR,
  "eviction_panel_source_manifest.csv"
)

write_csv(eviction_panel, panel_file)
write_csv(resolved_cases$issues, issues_file)
write_csv(assignment_summary, assignment_summary_file)
write_csv(panel_qa, panel_qa_file)
write_csv(coverage_qa, coverage_qa_file)
write_csv(source_geography_qa, source_geography_qa_file)
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Package 'digest' is required for the eviction source manifest.", call. = FALSE)
}
configured_raw_paths <- sort(unique(here::here(source_config$path)))
names(configured_raw_paths) <- sprintf(
  "configured_raw_eviction_%02d",
  seq_along(configured_raw_paths)
)
manifest_inputs <- c(input_files, configured_raw_paths)
missing_manifest_inputs <- manifest_inputs[!file.exists(manifest_inputs)]
if (length(missing_manifest_inputs) > 0L) {
  stop(
    "Eviction source-manifest input(s) are missing: ",
    paste(missing_manifest_inputs, collapse = ", "),
    call. = FALSE
  )
}
source_manifest <- tibble::tibble(
  artifact_role = names(manifest_inputs),
  path = unname(manifest_inputs),
  sha256 = vapply(
    unname(manifest_inputs),
    digest::digest,
    character(1),
    file = TRUE,
    algo = "sha256"
  ),
  bytes = as.numeric(file.info(unname(manifest_inputs))$size),
  source_start_date = SOURCE_START_DATE,
  source_observed_through_date = OBSERVED_THROUGH_DATE,
  requested_analysis_as_of_date = REQUESTED_OBSERVED_THROUGH_DATE
)
write_csv(source_manifest, source_manifest_file)

print_progress(
  paste0(
    "Wrote ",
    scales::comma(nrow(eviction_panel)),
    " complete hex-year rows to ",
    panel_file,
    "."
  )
)
print_progress("Case-to-hex assignment status:")
print(assignment_summary)
print_progress("Annual panel reconciliation:")
print(panel_qa)

print_header("PART 3 EVICTION OUTCOME PANEL COMPLETE")
