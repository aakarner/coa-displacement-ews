################################################################################
# Build the Part 3 Residential-Demolition Outcome Panel
################################################################################
#
# Creates one row for every EWS hex and calendar year from 2009 through 2026.
# Calendar years 2010-2025 are complete.  The partial 2009 and 2026 counts are
# retained separately but are never exposed as complete annual outcome labels.

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))
source(project_path("R", "analysis_config.R"))
source(project_path("R", "demolition_coverage_history.R"))
source(project_path("R", "demolition_panel.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
  library(sf)
  library(stringr)
  library(tidyr)
})

print_header("PART 3 - RESIDENTIAL DEMOLITION OUTCOME PANEL")

SOURCE_START_DATE <- as.Date("2009-10-02")
SOURCE_EXTRACT_THROUGH_DATE <- as.Date("2026-04-01")
if (EWS_CONFIG$analysis_as_of_date > SOURCE_EXTRACT_THROUGH_DATE) {
  stop(
    "The analysis cutoff exceeds the demolition extract vintage; refresh ",
    "the source before treating later periods as observed.",
    call. = FALSE
  )
}
OBSERVED_THROUGH_DATE <- min(
  EWS_CONFIG$analysis_as_of_date,
  SOURCE_EXTRACT_THROUGH_DATE
)
if (OBSERVED_THROUGH_DATE < SOURCE_START_DATE) {
  stop(
    "The analysis cutoff precedes the demolition source start date.",
    call. = FALSE
  )
}
first_panel_year <- lubridate::year(SOURCE_START_DATE)
last_panel_year <- lubridate::year(OBSERVED_THROUGH_DATE)
PANEL_YEARS <- seq.int(first_panel_year, last_panel_year)
first_complete_year <- if (SOURCE_START_DATE == as.Date(
  paste0(first_panel_year, "-01-01")
)) {
  first_panel_year
} else {
  first_panel_year + 1L
}
last_complete_year <- if (OBSERVED_THROUGH_DATE == as.Date(
  paste0(last_panel_year, "-12-31")
)) {
  last_panel_year
} else {
  last_panel_year - 1L
}
COMPLETE_YEARS <- if (first_complete_year <= last_complete_year) {
  seq.int(first_complete_year, last_complete_year)
} else {
  integer()
}

DEMOLITION_SOURCE_FILE <- project_path(
  "data",
  "Issued_Construction_Permits_20260401.csv"
)
HEX_GRID_FILE <- project_path("output", "hex_grid.rds")
JURISDICTIONS_FILE <- project_path(
  "data",
  "BOUNDARIES_jurisdictions_20260429.geojson"
)
JURISDICTION_SOURCE_CONFIG_FILE <- project_path(
  "config",
  "demolition_jurisdiction_sources.csv"
)
PANEL_FILE <- project_path("output", "demolition_permits_by_hex_year.csv")
SOURCE_QA_FILE <- project_path("output", "demolition_permits_source_qa.csv")
ANNUAL_QA_FILE <- project_path("output", "demolition_permits_annual_qa.csv")
UNMATCHED_QA_FILE <- project_path("output", "demolition_permits_unmatched_qa.csv")
COVERAGE_PANEL_FILE <- project_path(
  "output",
  "part3",
  "demolition_historical_coverage_by_hex_year.csv"
)
COVERAGE_SNAPSHOT_QA_FILE <- project_path(
  "output",
  "part3",
  "demolition_coverage_current_snapshot_qa.csv"
)
COVERAGE_SNAPSHOT_SUMMARY_FILE <- project_path(
  "output",
  "part3",
  "demolition_coverage_current_snapshot_summary.csv"
)
SOURCE_MANIFEST_FILE <- project_path(
  "output",
  "part3",
  "demolition_panel_source_manifest.csv"
)

# Part 3 uses one fixed study geography for every outcome: the H3 cells whose
# centers fall within the City of Austin FULL-purpose boundary in the dated
# jurisdiction snapshot.  The full 7,027-cell grid remains in the output so
# exclusion is explicit rather than silently changing the panel's shape.
CITY_BOUNDARY_SNAPSHOT_DATE <- as.Date("2026-04-29")
CITY_STUDY_GEOGRAPHY <- "current_austin_full_purpose_fixed"
CITY_HEX_ASSIGNMENT_METHOD <-
  "hex_point_on_surface_within_current_city_full"
EXPECTED_CITY_STUDY_HEXES <- 6060L
SPATIAL_ANALYSIS_CRS <- 3083

required_files <- c(
  DEMOLITION_SOURCE_FILE,
  HEX_GRID_FILE,
  JURISDICTIONS_FILE,
  JURISDICTION_SOURCE_CONFIG_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing demolition-panel input(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

jurisdiction_source_config <- read_csv(
  JURISDICTION_SOURCE_CONFIG_FILE,
  show_col_types = FALSE
) |>
  filter(.data$status == "active")
required_source_config_columns <- c(
  "coverage_role", "local_path", "source_url", "retrieved_date", "sha256"
)
if (
  !all(required_source_config_columns %in% names(jurisdiction_source_config)) ||
    !setequal(
      jurisdiction_source_config$coverage_role,
      c("historical_baseline", "dated_actions")
    )
) {
  stop(
    "The demolition jurisdiction source config must define one active ",
    "historical_baseline and dated_actions source.",
    call. = FALSE
  )
}
jurisdiction_source_config <- jurisdiction_source_config |>
  mutate(resolved_path = vapply(.data$local_path, project_path, character(1)))
missing_coverage_files <- jurisdiction_source_config$resolved_path[
  !file.exists(jurisdiction_source_config$resolved_path)
]
if (length(missing_coverage_files) > 0L) {
  stop(
    "Missing historical jurisdiction source(s): ",
    paste(missing_coverage_files, collapse = ", "),
    call. = FALSE
  )
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Package 'digest' is required for source verification.", call. = FALSE)
}
jurisdiction_source_config <- jurisdiction_source_config |>
  mutate(
    actual_sha256 = vapply(
      .data$resolved_path,
      digest::digest,
      character(1),
      file = TRUE,
      algo = "sha256"
    )
  )
if (any(
  jurisdiction_source_config$actual_sha256 !=
    jurisdiction_source_config$sha256
)) {
  stop(
    "A historical jurisdiction source checksum differs from its config.",
    call. = FALSE
  )
}

dir.create(dirname(PANEL_FILE), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(COVERAGE_PANEL_FILE), recursive = TRUE, showWarnings = FALSE)

print_progress(
  "Loading issued demolition permits, H3 grid, and City jurisdiction history..."
)
raw_permits <- read_csv(
  DEMOLITION_SOURCE_FILE,
  col_select = all_of(demolition_required_columns),
  col_types = cols(
    .default = col_character(),
    Latitude = col_double(),
    Longitude = col_double()
  ),
  show_col_types = FALSE
)
hex_grid <- readRDS(HEX_GRID_FILE)
jurisdictions <- st_read(JURISDICTIONS_FILE, quiet = TRUE)
historical_baselines <- st_read(
  jurisdiction_source_config$resolved_path[
    jurisdiction_source_config$coverage_role == "historical_baseline"
  ][[1]],
  quiet = TRUE
)
dated_actions <- st_read(
  jurisdiction_source_config$resolved_path[
    jurisdiction_source_config$coverage_role == "dated_actions"
  ][[1]],
  quiet = TRUE
)

if (
  !inherits(jurisdictions, "sf") ||
    is.na(st_crs(jurisdictions)) ||
    !all(c("city_name", "jurisdiction_type") %in% names(jurisdictions))
) {
  stop(
    "The current jurisdiction snapshot lacks the required City fields or CRS.",
    call. = FALSE
  )
}
current_austin_full_rows <- jurisdictions |>
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

hex_grid_projected <- hex_grid |>
  st_make_valid() |>
  st_transform(SPATIAL_ANALYSIS_CRS)
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
    city_study_geography = CITY_STUDY_GEOGRAPHY,
    city_boundary_snapshot_date = CITY_BOUNDARY_SNAPSHOT_DATE,
    city_hex_assignment_method = CITY_HEX_ASSIGNMENT_METHOD,
    hex_center_inside_current_austin_full = hex_center_inside_city,
    hex_intersects_current_austin_full = hex_intersects_city
  )
if (
  nrow(city_hex_reference) != nrow(hex_grid) ||
    anyDuplicated(city_hex_reference$hex_id) ||
    anyNA(city_hex_reference) ||
    sum(city_hex_reference$hex_center_inside_current_austin_full) !=
      EXPECTED_CITY_STUDY_HEXES ||
    any(
      city_hex_reference$hex_center_inside_current_austin_full &
        !city_hex_reference$hex_intersects_current_austin_full
    )
) {
  stop(
    "The fixed City study-hex reference is structurally invalid or no longer ",
    "contains exactly ", EXPECTED_CITY_STUDY_HEXES, " cells.",
    call. = FALSE
  )
}

print_progress("Replaying effective-dated City jurisdiction coverage by year...")
historical_coverage_raw <- build_demolition_historical_hex_coverage(
  hex_grid = hex_grid,
  historical_baselines = historical_baselines,
  dated_actions = dated_actions,
  years = PANEL_YEARS,
  source_start_date = SOURCE_START_DATE,
  observed_through_date = OBSERVED_THROUGH_DATE,
  supported_jurisdiction_types = c("FULL", "LTD", "2MILE")
)
historical_coverage <- historical_coverage_raw |>
  left_join(
    city_hex_reference,
    by = "hex_id",
    relationship = "many-to-one"
  ) |>
  mutate(
    coverage_resolved = .data$coverage_period_resolved,
    source_covered_before_city_filter = coalesce(
      .data$source_covered,
      FALSE
    ),
    source_covered = .data$source_covered_before_city_filter &
      .data$hex_center_inside_current_austin_full,
    geographic_coverage_reason = case_when(
      !.data$hex_center_inside_current_austin_full ~
        "outside_current_austin_full_purpose_center_selected_grid",
      !.data$coverage_resolved ~ paste0(
        "unresolved_historical_jurisdiction_during_observed_period;",
        .data$coverage_status
      ),
      .data$source_covered ~ paste0(
        "covered_supported_historical_jurisdiction_continuously;states_",
        coalesce(.data$coverage_candidate_types, "missing")
      ),
      TRUE ~ paste0(
        "not_covered_supported_historical_jurisdiction_continuously;",
        .data$coverage_status,
        ";states_",
        coalesce(.data$coverage_candidate_types, "missing")
      )
    )
  )

coverage_snapshot_comparison <- compare_demolition_coverage_to_current_snapshot(
  coverage = historical_coverage_raw,
  hex_grid = hex_grid,
  current_jurisdictions = jurisdictions,
  comparison_year = max(PANEL_YEARS),
  current_snapshot_as_of_date = as.Date("2026-04-29"),
  supported_jurisdiction_types = c("FULL", "LTD", "2MILE")
)
coverage_snapshot_summary <- summarize_demolition_coverage_snapshot_qa(
  coverage_snapshot_comparison
)

print_progress("Filtering unique issued residential-demolition permits...")
artifacts <- build_demolition_panel_artifacts(
  raw_permits = raw_permits,
  hex_grid = hex_grid,
  hex_coverage = historical_coverage,
  source_start_date = SOURCE_START_DATE,
  observed_through_date = OBSERVED_THROUGH_DATE,
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
  panel_years = PANEL_YEARS,
  complete_years = COMPLETE_YEARS,
  fixed_study_boundary = current_austin_full,
  analysis_crs = SPATIAL_ANALYSIS_CRS
)

print_progress("Writing demolition panel and aggregate QA artifacts...")
write_csv(artifacts$panel, PANEL_FILE, na = "")
write_csv(artifacts$source_qa, SOURCE_QA_FILE, na = "")
write_csv(artifacts$annual_qa, ANNUAL_QA_FILE, na = "")
write_csv(artifacts$unmatched_qa, UNMATCHED_QA_FILE, na = "")
write_csv(historical_coverage, COVERAGE_PANEL_FILE, na = "")
write_csv(
  coverage_snapshot_comparison,
  COVERAGE_SNAPSHOT_QA_FILE,
  na = ""
)
write_csv(
  coverage_snapshot_summary,
  COVERAGE_SNAPSHOT_SUMMARY_FILE,
  na = ""
)
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Package 'digest' is required for the demolition source manifest.", call. = FALSE)
}
manifest_inputs <- c(
  raw_permits = DEMOLITION_SOURCE_FILE,
  hex_grid = HEX_GRID_FILE,
  current_jurisdictions = JURISDICTIONS_FILE
)
manifest_inputs <- c(
  manifest_inputs,
  historical_baseline = jurisdiction_source_config$resolved_path[
    jurisdiction_source_config$coverage_role == "historical_baseline"
  ][[1]],
  dated_actions = jurisdiction_source_config$resolved_path[
    jurisdiction_source_config$coverage_role == "dated_actions"
  ][[1]],
  jurisdiction_source_config = JURISDICTION_SOURCE_CONFIG_FILE
)
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
  requested_analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
)
dir.create(dirname(SOURCE_MANIFEST_FILE), recursive = TRUE, showWarnings = FALSE)
write_csv(source_manifest, SOURCE_MANIFEST_FILE, na = "")

annual_total <- sum(
  artifacts$panel$residential_demolition_permits,
  na.rm = TRUE
)
partial_total <- artifacts$panel %>%
  filter(!.data$period_complete, .data$source_covered) %>%
  summarise(
    total = sum(
      .data$residential_demolition_permits_observed_to_date,
      na.rm = TRUE
    )
  ) %>%
  pull("total")

print_progress(
  paste0(
    "Saved ",
    format(nrow(artifacts$panel), big.mark = ","),
    " hex-year rows with ",
    format(annual_total, big.mark = ","),
    " mapped permits in complete years."
  )
)
print_progress(
  paste0(
    "Retained ",
    format(partial_total, big.mark = ","),
    " mapped permits as observed-to-date counts in partial years."
  )
)
print_progress(paste0("Panel: ", PANEL_FILE))
print_progress(paste0("QA: ", SOURCE_QA_FILE))
print_progress(paste0("QA: ", ANNUAL_QA_FILE))
print_progress(paste0("QA: ", UNMATCHED_QA_FILE))
