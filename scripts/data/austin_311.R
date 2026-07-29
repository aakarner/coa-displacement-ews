################################################################################
# Process Austin 311 Requests to Hexagonal Grid
################################################################################
#
# Pulls a versioned set of displacement-related Austin 311 service-request
# types from the Socrata API v3 endpoint, assigns geocoded requests to the
# project hex grid, and builds hex-level smoke-signal summaries.
#
# Credentials:
#   Public anonymous access is supported. These optional variables can provide
#   authenticated access when API limits require it:
#     AUSTIN_DATA_API_KEY
#     AUSTIN_DATA_API_SECRET
#
# Optional runtime controls:
#     AUSTIN_311_START_DATE   default: 2020-01-01
#     AUSTIN_311_PAGE_SIZE    default: 50000
#     AUSTIN_311_MAX_PAGES    default: Inf
#     EWS_REFRESH_311         default: false
#
# Outputs:
#   - output/311_requests_by_hex_summary.rds
#   - output/311_requests_by_hex_summary.csv
#   - output/311_requests_by_hex_year.csv
#   - output/311_service_request_counts.csv
#   - output/311_service_request_selection.csv
#
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
  library(sf)
  library(dplyr)
  library(tidyr)
  library(readr)
  library(lubridate)
  library(stringr)
  library(httr)
  library(jsonlite)
})

print_header("02g - AUSTIN 311 REQUESTS TO HEX GRID")

OUTPUT_DIR <- project_path("output")
FIGURES_DIR <- project_path("figures")
RAW_DIR <- project_path("data", "raw_311")
TYPE_CONFIG_FILE <- project_path("config", "311_smoke_signal_types.csv")
API_ENDPOINT <- "https://data.austintexas.gov/api/v3/views/xwdj-i9he/query.json"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)

api_key <- Sys.getenv("AUSTIN_DATA_API_KEY")
api_secret <- Sys.getenv("AUSTIN_DATA_API_SECRET")

if (xor(nzchar(api_key), nzchar(api_secret))) {
  stop(
    "Set both AUSTIN_DATA_API_KEY and AUSTIN_DATA_API_SECRET, or neither.",
    call. = FALSE
  )
}

start_date <- Sys.getenv("AUSTIN_311_START_DATE", unset = "2020-01-01")
page_size <- as.integer(Sys.getenv("AUSTIN_311_PAGE_SIZE", unset = "50000"))
max_pages_env <- Sys.getenv("AUSTIN_311_MAX_PAGES", unset = "")
max_pages <- if (max_pages_env == "") Inf else as.integer(max_pages_env)
refresh_311 <- tolower(
  Sys.getenv("EWS_REFRESH_311", unset = "false")
) %in% c("true", "t", "1", "yes", "y")

if (is.na(as.Date(start_date))) {
  stop("AUSTIN_311_START_DATE must be an ISO date.", call. = FALSE)
}
if (is.na(page_size) || page_size < 1L || page_size > 50000L) {
  stop("AUSTIN_311_PAGE_SIZE must be between 1 and 50000.", call. = FALSE)
}
if (!is.infinite(max_pages) && (is.na(max_pages) || max_pages < 1L)) {
  stop("AUSTIN_311_MAX_PAGES must be a positive integer.", call. = FALSE)
}
if (!file.exists(TYPE_CONFIG_FILE)) {
  stop("Missing 311 smoke-signal type configuration: ", TYPE_CONFIG_FILE)
}

smoke_signal_types <- read_csv(TYPE_CONFIG_FILE, show_col_types = FALSE)
required_type_columns <- c("sr_type_desc", "signal_category", "rationale")
missing_type_columns <- setdiff(required_type_columns, names(smoke_signal_types))
if (length(missing_type_columns) > 0L) {
  stop(
    "311 type configuration is missing: ",
    paste(missing_type_columns, collapse = ", "),
    call. = FALSE
  )
}
if (
  nrow(smoke_signal_types) == 0L ||
    any(is.na(smoke_signal_types$sr_type_desc)) ||
    anyDuplicated(smoke_signal_types$sr_type_desc)
) {
  stop(
    "311 type configuration must contain unique, nonmissing descriptions.",
    call. = FALSE
  )
}
selected_type_descriptions <- sort(smoke_signal_types$sr_type_desc)
sql_string <- function(x) {
  paste0("'", gsub("'", "''", x, fixed = TRUE), "'")
}
selected_type_sql <- paste(
  vapply(selected_type_descriptions, sql_string, character(1)),
  collapse = ", "
)

cache_slug <- paste0(
  gsub("[^0-9]", "", start_date),
  "_",
  gsub("[^0-9]", "", as.character(EWS_CONFIG$analysis_as_of_date))
)
raw_cache_file <- file.path(
  RAW_DIR,
  paste0("austin_311_selected_", cache_slug, ".rds")
)

selected_fields <- c(
  "sr_number",
  "sr_type_desc",
  "sr_department_desc",
  "sr_method_received_desc",
  "sr_status_desc",
  "sr_created_date",
  "sr_closed_date",
  "sr_location_lat",
  "sr_location_long",
  "sr_location_council_district"
)

fetch_311_page <- function(offset, limit) {
  end_date_exclusive <- EWS_CONFIG$analysis_as_of_date + days(1)
  query <- paste0(
    "SELECT ", paste(selected_fields, collapse = ", "), " ",
    "WHERE sr_created_date >= '", start_date, "T00:00:00' ",
    "AND sr_created_date < '", end_date_exclusive, "T00:00:00' ",
    "AND sr_type_desc IN (", selected_type_sql, ") ",
    "AND sr_location_lat IS NOT NULL ",
    "AND sr_location_long IS NOT NULL ",
    "ORDER BY sr_created_date, sr_number ",
    "LIMIT ", limit, " OFFSET ", offset
  )

  request_body <- jsonlite::toJSON(list(query = query), auto_unbox = TRUE)
  response <- if (nzchar(api_key) && nzchar(api_secret)) {
    httr::POST(
      API_ENDPOINT,
      httr::authenticate(api_key, api_secret),
      httr::content_type_json(),
      body = request_body,
      encode = "raw"
    )
  } else {
    httr::POST(
      API_ENDPOINT,
      httr::content_type_json(),
      body = request_body,
      encode = "raw"
    )
  }

  if (httr::http_error(response)) {
    stop(
      "Austin 311 API request failed: ",
      httr::status_code(response),
      "\n",
      httr::content(response, as = "text", encoding = "UTF-8")
    )
  }

  body <- httr::content(response, as = "text", encoding = "UTF-8")
  parsed <- jsonlite::fromJSON(body, flatten = TRUE)

  if (length(parsed) == 0) {
    return(tibble())
  }

  as_tibble(parsed)
}

requests_raw <- NULL
if (file.exists(raw_cache_file) && !refresh_311) {
  cached <- readRDS(raw_cache_file)
  cache_matches <- is.list(cached) &&
    identical(cached$schema_version, 1L) &&
    isTRUE(cached$complete) &&
    identical(as.character(cached$start_date), start_date) &&
    identical(
      as.Date(cached$analysis_as_of_date),
      EWS_CONFIG$analysis_as_of_date
    ) &&
    identical(sort(cached$selected_type_descriptions), selected_type_descriptions)
  if (cache_matches) {
    requests_raw <- cached$data
    print_progress(paste0("Loaded selected 311 request cache: ", raw_cache_file))
  } else {
    print_progress("Existing 311 cache does not match the current query contract.")
  }
}

if (is.null(requests_raw)) {
  print_progress(
    paste0(
      "Fetching selected Austin 311 requests from ", start_date,
      " onward in pages of ", page_size, "..."
    )
  )

  pages <- list()
  offset <- 0L
  page_index <- 1L
  fetch_complete <- FALSE

  repeat {
    print_progress(
      paste0("Fetching 311 page ", page_index, " at offset ", offset, "...")
    )
    page <- fetch_311_page(offset = offset, limit = page_size)

    if (nrow(page) == 0) {
      print_progress("No more 311 records returned.")
      fetch_complete <- TRUE
      break
    }

    pages[[page_index]] <- page
    print_progress(paste0("Fetched ", nrow(page), " records."))

    if (nrow(page) < page_size) {
      fetch_complete <- TRUE
      break
    }
    if (page_index >= max_pages) break

    offset <- offset + page_size
    page_index <- page_index + 1L
  }

  requests_raw <- bind_rows(pages)
  if (!fetch_complete) {
    stop(
      "AUSTIN_311_MAX_PAGES truncated the source extract. ",
      "Increase or unset it before producing canonical outputs.",
      call. = FALSE
    )
  }
  saveRDS(
    list(
      schema_version = 1L,
      complete = TRUE,
      fetched_at = Sys.time(),
      start_date = as.Date(start_date),
      analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
      selected_type_descriptions = selected_type_descriptions,
      data = requests_raw
    ),
    raw_cache_file
  )
  print_progress(paste0("Saved selected 311 request cache: ", raw_cache_file))
}

if (nrow(requests_raw) == 0) {
  stop("No Austin 311 records were returned for the requested date range.")
}

print_progress(paste0("Fetched ", nrow(requests_raw), " total 311 records."))

requests_clean <- requests_raw %>%
  transmute(
    sr_number = as.character(sr_number),
    sr_type_desc = as.character(sr_type_desc),
    sr_department_desc = as.character(sr_department_desc),
    sr_method_received_desc = as.character(sr_method_received_desc),
    sr_status_desc = as.character(sr_status_desc),
    sr_created_datetime = ymd_hms(sr_created_date, quiet = TRUE),
    sr_closed_datetime = ymd_hms(sr_closed_date, quiet = TRUE),
    sr_location_lat = as.numeric(sr_location_lat),
    sr_location_long = as.numeric(sr_location_long),
    sr_location_council_district = as.integer(sr_location_council_district)
  ) %>%
  mutate(
    sr_created_date = as.Date(sr_created_datetime),
    sr_closed_date = as.Date(sr_closed_datetime)
  ) %>%
  filter(
    !is.na(sr_created_date),
    sr_created_date <= EWS_CONFIG$analysis_as_of_date,
    !is.na(sr_location_lat),
    !is.na(sr_location_long),
    between(sr_location_lat, 29.8, 30.7),
    between(sr_location_long, -98.3, -97.2)
  ) %>%
  left_join(
    smoke_signal_types %>%
      select(sr_type_desc, signal_category, rationale),
    by = "sr_type_desc",
    relationship = "many-to-one"
  ) %>%
  mutate(
    sr_year = year(sr_created_date),
    sr_month = floor_date(sr_created_date, "month"),
    is_code_related = signal_category == "code_enforcement",
    is_housing_condition = signal_category == "housing_condition",
    is_tenant_distress = signal_category == "tenant_distress",
    is_nuisance_or_disorder = FALSE,
    is_311_smoke_signal = TRUE
  )

if (any(is.na(requests_clean$signal_category))) {
  unexpected_types <- sort(unique(
    requests_clean$sr_type_desc[is.na(requests_clean$signal_category)]
  ))
  stop(
    "Fetched 311 records include unconfigured request type(s): ",
    paste(unexpected_types, collapse = ", "),
    call. = FALSE
  )
}
if (anyDuplicated(requests_clean$sr_number)) {
  stop(
    "Selected Austin 311 extract contains duplicate service-request numbers.",
    call. = FALSE
  )
}

print_progress(
  paste0(
    "Retained ", nrow(requests_clean),
    " geocoded Austin-area 311 records after coordinate/date cleaning."
  )
)

hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid") %>%
  st_transform(4326)

requests_sf <- requests_clean %>%
  st_as_sf(coords = c("sr_location_long", "sr_location_lat"), crs = 4326, remove = FALSE) %>%
  st_transform(st_crs(hex_grid))

requests_hex <- requests_sf %>%
  st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE)

print_progress(paste0("Assigned ", nrow(requests_hex), " 311 records to project hexagons."))

type_counts <- requests_hex %>%
  st_drop_geometry() %>%
  count(
    sr_type_desc,
    signal_category,
    rationale,
    sr_department_desc,
    sort = TRUE,
    name = "request_count"
  )

write_csv(type_counts, file.path(OUTPUT_DIR, "311_service_request_counts.csv"))
smoke_signal_types %>%
  left_join(
    type_counts %>%
      group_by(sr_type_desc, signal_category, rationale) %>%
      summarise(request_count = sum(request_count), .groups = "drop"),
    by = c("sr_type_desc", "signal_category", "rationale"),
    relationship = "one-to-one"
  ) %>%
  mutate(
    request_count = replace_na(request_count, 0),
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
    start_date = as.Date(start_date)
  ) %>%
  write_csv(file.path(OUTPUT_DIR, "311_service_request_selection.csv"))

annual_by_hex <- requests_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id, sr_year) %>%
  summarise(
    sr_311_total = n(),
    sr_311_code_related = sum(is_code_related, na.rm = TRUE),
    sr_311_housing_condition = sum(is_housing_condition, na.rm = TRUE),
    sr_311_tenant_distress = sum(is_tenant_distress, na.rm = TRUE),
    sr_311_smoke_signal = sum(is_311_smoke_signal, na.rm = TRUE),
    sr_311_nuisance_or_disorder = sum(is_nuisance_or_disorder, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(annual_by_hex, file.path(OUTPUT_DIR, "311_requests_by_hex_year.csv"))

max_request_date <- max(requests_hex$sr_created_date, na.rm = TRUE)
latest_12mo_start <- EWS_CONFIG$analysis_as_of_date %m-% years(1) + days(1)
previous_12mo_start <- latest_12mo_start %m-% years(1)

summary_by_hex <- requests_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarise(
    sr_311_total = n(),
    sr_311_code_related_total = sum(is_code_related, na.rm = TRUE),
    sr_311_housing_condition_total = sum(is_housing_condition, na.rm = TRUE),
    sr_311_tenant_distress_total = sum(is_tenant_distress, na.rm = TRUE),
    sr_311_smoke_signal_total = sum(is_311_smoke_signal, na.rm = TRUE),
    sr_311_nuisance_or_disorder_total = sum(is_nuisance_or_disorder, na.rm = TRUE),
    sr_311_latest_12mo = sum(
      sr_created_date >= latest_12mo_start &
        sr_created_date <= EWS_CONFIG$analysis_as_of_date,
      na.rm = TRUE
    ),
    sr_311_previous_12mo = sum(
      sr_created_date >= previous_12mo_start & sr_created_date < latest_12mo_start,
      na.rm = TRUE
    ),
    sr_311_smoke_signal_latest_12mo = sum(
      is_311_smoke_signal &
        sr_created_date >= latest_12mo_start &
        sr_created_date <= EWS_CONFIG$analysis_as_of_date,
      na.rm = TRUE
    ),
    sr_311_smoke_signal_previous_12mo = sum(
      is_311_smoke_signal &
        sr_created_date >= previous_12mo_start &
        sr_created_date < latest_12mo_start,
      na.rm = TRUE
    ),
    sr_311_first_date = min(sr_created_date, na.rm = TRUE),
    sr_311_last_date = max(sr_created_date, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    sr_311_latest_12mo_change_pct = if_else(
      sr_311_previous_12mo > 0,
      100 * (sr_311_latest_12mo / sr_311_previous_12mo - 1),
      NA_real_
    ),
    sr_311_smoke_signal_latest_12mo_change_pct = if_else(
      sr_311_smoke_signal_previous_12mo > 0,
      100 * (sr_311_smoke_signal_latest_12mo / sr_311_smoke_signal_previous_12mo - 1),
      NA_real_
    )
  )

summary_full <- hex_grid %>%
  left_join(summary_by_hex, by = "hex_id") %>%
  mutate(
    across(
      c(
        sr_311_total,
        sr_311_code_related_total,
        sr_311_housing_condition_total,
        sr_311_tenant_distress_total,
        sr_311_smoke_signal_total,
        sr_311_nuisance_or_disorder_total,
        sr_311_latest_12mo,
        sr_311_previous_12mo,
        sr_311_smoke_signal_latest_12mo,
        sr_311_smoke_signal_previous_12mo
      ),
      ~replace_na(.x, 0)
    ),
    sr_311_per_km2 = if_else(area_km2 > 0, sr_311_total / area_km2, NA_real_),
    sr_311_latest_12mo_per_km2 = if_else(area_km2 > 0, sr_311_latest_12mo / area_km2, NA_real_),
    sr_311_smoke_signal_per_km2 = if_else(area_km2 > 0, sr_311_smoke_signal_total / area_km2, NA_real_),
    sr_311_smoke_signal_latest_12mo_per_km2 = if_else(
      area_km2 > 0,
      sr_311_smoke_signal_latest_12mo / area_km2,
      NA_real_
    ),
    sr_311_data_start = min(requests_hex$sr_created_date, na.rm = TRUE),
    sr_311_data_end = max_request_date,
    sr_311_analysis_as_of = EWS_CONFIG$analysis_as_of_date,
    sr_311_latest_12mo_start = latest_12mo_start,
    sr_311_previous_12mo_start = previous_12mo_start,
    sr_311_source_scope = "configured_displacement_smoke_signal_types",
    sr_311_selected_type_count = nrow(smoke_signal_types),
    sr_311_type_config_md5 = unname(tools::md5sum(TYPE_CONFIG_FILE))
  )

save_output(
  summary_full,
  file.path(OUTPUT_DIR, "311_requests_by_hex_summary.rds"),
  "311 request hex summary"
)

summary_full %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "311_requests_by_hex_summary.csv"))

print_header("STEP 02g COMPLETE")
cat(paste0("311 records fetched: ", nrow(requests_raw), "\n"))
cat(paste0("311 records assigned to hexes: ", nrow(requests_hex), "\n"))
cat(paste0("Latest request date: ", max_request_date, "\n"))
cat("Outputs:\n")
cat("  - output/311_requests_by_hex_summary.rds\n")
cat("  - output/311_requests_by_hex_summary.csv\n")
cat("  - output/311_requests_by_hex_year.csv\n")
cat("  - output/311_service_request_counts.csv\n")
cat("  - output/311_service_request_selection.csv\n")
