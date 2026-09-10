################################################################################
# Build a Versioned Williamson JP-to-Hex Reference
################################################################################
#
# This manual refresh utility intersects analysis-hex points on surface with
# official Williamson County precinct polygons. The historical layer is used
# for 2020-2021 and the post-redistricting layer for 2022 onward. Generated CSV
# references are tracked so routine analysis runs do not depend on a live GIS.
################################################################################

required_packages <- c("digest", "dplyr", "readr", "sf")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s): ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
})

source(here::here("R", "utils.R"))

print_header("BUILD WILLIAMSON JP HEX REFERENCE")

HISTORICAL_URL <- paste0(
  "https://gis.wilco.org/arcgis/rest/services/public/",
  "county_historical_administrative_boundaries/MapServer/0/query?",
  "where=1%3D1&outFields=PCT_NUMBER&returnGeometry=true&",
  "outSR=4326&f=geojson"
)
CURRENT_URL <- paste0(
  "https://gis.wilco.org/arcgis/rest/services/public/",
  "adopted_county_precincts/MapServer/0/query?",
  "where=1%3D1&outFields=PCT_NUMBER&returnGeometry=true&",
  "outSR=4326&f=geojson"
)
HISTORICAL_INPUT <- Sys.getenv(
  "WILLIAMSON_HISTORICAL_PRECINCT_INPUT",
  HISTORICAL_URL
)
CURRENT_INPUT <- Sys.getenv(
  "WILLIAMSON_CURRENT_PRECINCT_INPUT",
  CURRENT_URL
)
RETRIEVED_DATE <- as.Date(Sys.getenv(
  "WILLIAMSON_PRECINCT_RETRIEVED_DATE",
  as.character(Sys.Date())
))
if (is.na(RETRIEVED_DATE)) {
  stop("WILLIAMSON_PRECINCT_RETRIEVED_DATE must be YYYY-MM-DD.", call. = FALSE)
}

input_files <- c(
  grid = here::here("output", "hex_grid.rds"),
  counties = here::here("config", "hex_county_assignment_2024.csv")
)
missing_files <- input_files[!file.exists(input_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing reference input(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

read_precincts <- function(input, label) {
  print_progress(paste0("Reading ", label, " precinct polygons..."))
  polygons <- sf::read_sf(input, quiet = TRUE) |>
    sf::st_make_valid()
  precinct_field <- names(polygons)[
    toupper(names(polygons)) == "PCT_NUMBER"
  ]
  if (length(precinct_field) != 1L) {
    stop(label, " polygons do not contain one PCT_NUMBER field.", call. = FALSE)
  }
  polygons <- polygons |>
    dplyr::transmute(
      jp_district = paste0("JP", as.integer(.data[[precinct_field]]))
    )
  if (!setequal(polygons$jp_district, paste0("JP", 1:4))) {
    stop(label, " polygons do not contain JP1 through JP4.", call. = FALSE)
  }
  polygons
}

grid <- readRDS(input_files[["grid"]])
county_reference <- read_csv(
  input_files[["counties"]],
  col_types = cols(
    hex_id = col_integer(),
    h3_index = col_character(),
    source_county = col_character()
  ),
  show_col_types = FALSE
)
williamson_hexes <- grid |>
  inner_join(
    county_reference |>
      filter(.data$source_county == "Williamson") |>
      select("hex_id", "h3_index"),
    by = c("hex_id", "h3_index")
  )
if (nrow(williamson_hexes) == 0L || anyDuplicated(williamson_hexes$hex_id)) {
  stop("Williamson analysis hex inventory is empty or duplicated.", call. = FALSE)
}

analysis_crs <- 3083
hex_points <- suppressWarnings(
  williamson_hexes |>
    st_transform(analysis_crs) |>
    st_point_on_surface()
)

assign_period <- function(
  polygons,
  effective_start_date,
  effective_end_date,
  boundary_vintage,
  source_url
) {
  assigned <- hex_points |>
    select("hex_id", "h3_index") |>
    st_join(
      polygons |>
        st_transform(analysis_crs) |>
        select("jp_district"),
      join = st_within,
      left = TRUE
    )
  if (nrow(assigned) != nrow(hex_points) || anyDuplicated(assigned$hex_id)) {
    stop(
      boundary_vintage,
      " precinct polygons assigned a hex point more than once.",
      call. = FALSE
    )
  }
  assigned |>
    st_drop_geometry() |>
    mutate(
      effective_start_date = as.Date(effective_start_date),
      effective_end_date = as.Date(effective_end_date),
      boundary_vintage = boundary_vintage,
      assignment_method = "hex_point_on_surface",
      assignment_status = if_else(
        is.na(.data$jp_district),
        "outside_official_precinct_geometry",
        "assigned"
      ),
      boundary_source_url = source_url
    ) |>
    select(
      "hex_id", "h3_index", "effective_start_date", "effective_end_date",
      "jp_district", "boundary_vintage", "assignment_method",
      "assignment_status", "boundary_source_url"
    )
}

historical_polygons <- read_precincts(
  HISTORICAL_INPUT,
  "2012-2021 historical"
)
current_polygons <- read_precincts(
  CURRENT_INPUT,
  "2022+ adopted"
)

reference <- bind_rows(
  assign_period(
    historical_polygons,
    "2012-01-01",
    "2021-12-31",
    "2012-2021",
    HISTORICAL_URL
  ),
  assign_period(
    current_polygons,
    "2022-01-01",
    NA_character_,
    "2022-present",
    CURRENT_URL
  )
) |>
  arrange(.data$effective_start_date, .data$hex_id)

expected_rows <- 2L * nrow(williamson_hexes)
if (nrow(reference) != expected_rows ||
    anyDuplicated(reference[c("hex_id", "effective_start_date")])) {
  stop("Generated Williamson JP reference is incomplete or duplicated.", call. = FALSE)
}

input_sha256 <- function(input) {
  if (file.exists(input)) {
    digest::digest(input, file = TRUE, algo = "sha256")
  } else {
    NA_character_
  }
}
metadata <- reference |>
  group_by(
    .data$boundary_vintage,
    .data$effective_start_date,
    .data$effective_end_date,
    .data$boundary_source_url
  ) |>
  summarize(
    retrieved_date = RETRIEVED_DATE,
    analysis_grid_rows = nrow(grid),
    williamson_hex_rows = n(),
    assigned_hex_rows = sum(.data$assignment_status == "assigned"),
    unassigned_hex_rows = sum(.data$assignment_status != "assigned"),
    jp1_hex_rows = sum(.data$jp_district == "JP1", na.rm = TRUE),
    jp2_hex_rows = sum(.data$jp_district == "JP2", na.rm = TRUE),
    jp3_hex_rows = sum(.data$jp_district == "JP3", na.rm = TRUE),
    jp4_hex_rows = sum(.data$jp_district == "JP4", na.rm = TRUE),
    assignment_method = "hex_point_on_surface",
    grid_sha256 = digest::digest(
      input_files[["grid"]],
      file = TRUE,
      algo = "sha256"
    ),
    county_reference_sha256 = digest::digest(
      input_files[["counties"]],
      file = TRUE,
      algo = "sha256"
    ),
    precinct_download_sha256 = if (
      dplyr::first(.data$boundary_vintage) == "2012-2021"
    ) {
      input_sha256(HISTORICAL_INPUT)
    } else {
      input_sha256(CURRENT_INPUT)
    },
    .groups = "drop"
  )

reference_file <- here::here("config", "williamson_jp_hex_assignment.csv")
metadata_file <- here::here(
  "config",
  "williamson_jp_hex_assignment_metadata.csv"
)
write_csv(reference, reference_file, na = "")
write_csv(metadata, metadata_file, na = "")

print_progress(paste0("Wrote ", nrow(reference), " period-specific hex rows."))
print(metadata)
print_header("WILLIAMSON JP HEX REFERENCE COMPLETE")
