################################################################################
# Audit Austin Code Complaint Cases Against 311 and Residential Parcels
################################################################################
#
# Downloads a cutoff-limited snapshot of Austin Code complaint cases, classifies
# the four published complaint descriptions, and measures linkage to:
#   1. the existing EWS Code-enforcement 311 intake universe; and
#   2. the promoted EWS residential parcel universe.
#
# Exact service-request and City parcel identifiers are evaluated first.
# Normalized address and nearest-residential-point evidence are reported as
# separate fallback tiers and never relabeled as exact matches.
#
# Optional environment variables:
#   EWS_REFRESH_CODE_COMPLAINTS=true  Redownload the cutoff-limited snapshot
#   SOCRATA_APP_TOKEN=...             Optional Socrata application token
#
# Outputs:
#   output/311_code_complaint_source_audit.csv
#   output/311_code_complaint_linkage_methods.csv
#   output/311_code_complaint_year_qa.csv
#   output/311_code_complaint_month_qa.csv
#   output/311_code_complaint_overlap_period_qa.csv
#   output/311_code_complaint_hex_sparsity_audit.csv
#   output/311_code_complaint_unfiltered_311_types.csv
#   output/311_code_complaint_cardinality_audit.csv
#   output/311_code_complaint_case_audit.rds
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
  library(dplyr)
  library(httr)
  library(lubridate)
  library(readr)
  library(sf)
  library(stringr)
  library(tibble)
  library(tidyr)
})

print_header("AUDIT AUSTIN CODE COMPLAINT CASES")

OUTPUT_DIR <- project_path("output")
RAW_DIR <- project_path("data", "raw_311")
CATEGORY_FILE <- project_path("config", "311_code_complaint_categories.csv")
EWS_311_CACHE_FILE <- file.path(
  RAW_DIR,
  paste0(
    "austin_311_selected_20200101_",
    format(EWS_CONFIG$analysis_as_of_date, "%Y%m%d"),
    ".rds"
  )
)
PROMOTED_PARCEL_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_promoted.rds"
)
LAND_USE_FILE <- project_path("data", "austin_land_use_inventory_202607.csv")
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")
PART1_ASSIGNMENTS_FILE <- file.path(
  OUTPUT_DIR,
  "part1",
  "baseline_cluster_assignments.csv"
)
EWS_311_HEX_FILE <- file.path(OUTPUT_DIR, "311_requests_by_hex_summary.rds")
UNFILTERED_311_API_ENDPOINT <- paste0(
  "https://data.austintexas.gov/api/v3/views/xwdj-i9he/query.json"
)

DATASET_ID <- "6wtj-zbtb"
API_ENDPOINT <- paste0(
  "https://data.austintexas.gov/resource/", DATASET_ID, ".json"
)
START_DATE <- as.Date("2020-01-01")
END_DATE <- EWS_CONFIG$analysis_as_of_date
END_DATE_EXCLUSIVE <- END_DATE + days(1)
PAGE_SIZE <- 50000L
CACHE_FILE <- file.path(
  RAW_DIR,
  paste0(
    "austin_code_complaints_",
    format(START_DATE, "%Y%m%d"),
    "_",
    format(END_DATE, "%Y%m%d"),
    ".rds"
  )
)
UNFILTERED_311_CACHE_FILE <- file.path(
  RAW_DIR,
  paste0(
    "austin_311_code_case_identifier_audit_",
    format(START_DATE, "%Y%m%d"),
    "_",
    format(END_DATE, "%Y%m%d"),
    ".rds"
  )
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

required_files <- c(
  CATEGORY_FILE,
  EWS_311_CACHE_FILE,
  PROMOTED_PARCEL_FILE,
  LAND_USE_FILE,
  HEX_GRID_FILE,
  PART1_ASSIGNMENTS_FILE,
  EWS_311_HEX_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Code complaint audit is missing required input(s):\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}

refresh <- tolower(
  Sys.getenv("EWS_REFRESH_CODE_COMPLAINTS", unset = "false")
) %in% c("1", "true", "yes", "y")
app_token <- Sys.getenv("SOCRATA_APP_TOKEN", unset = "")
api_key <- Sys.getenv("AUSTIN_DATA_API_KEY", unset = "")
api_secret <- Sys.getenv("AUSTIN_DATA_API_SECRET", unset = "")
if (xor(nzchar(api_key), nzchar(api_secret))) {
  stop(
    "Set both AUSTIN_DATA_API_KEY and AUSTIN_DATA_API_SECRET, or neither.",
    call. = FALSE
  )
}

blank_to_na <- function(x) {
  x <- str_squish(as.character(x))
  x[x == "" | toupper(x) == "NA"] <- NA_character_
  x
}

parse_api_date <- function(x) {
  as.Date(substr(as.character(x), 1L, 10L))
}

normalize_service_request <- function(x) {
  x <- str_to_upper(blank_to_na(x))
  str_extract(x, "[0-9]{2}-[0-9]{8}")
}

normalize_parcel_id <- function(x) {
  x <- str_to_upper(blank_to_na(x))
  str_replace_all(x, "\\s+", "")
}

normalize_base_address <- function(x) {
  x <- str_to_upper(blank_to_na(x))
  x <- str_replace(x, ",.*$", "")
  x <- str_replace(x, "\\s+#\\s*[A-Z0-9-]+.*$", "")
  x <- str_replace(
    x,
    paste0(
      "\\b(UNIT|APT|APARTMENT|SUITE|STE|BLDG|BUILDING|ROOM|RM|#)\\b.*$"
    ),
    ""
  )
  replacements <- c(
    " STREET\\b" = " ST",
    " ROAD\\b" = " RD",
    " AVENUE\\b" = " AVE",
    " BOULEVARD\\b" = " BLVD",
    " DRIVE\\b" = " DR",
    " LANE\\b" = " LN",
    " COURT\\b" = " CT",
    " PLACE\\b" = " PL",
    " PARKWAY\\b" = " PKWY",
    " HIGHWAY\\b" = " HWY",
    " NORTH\\b" = " N",
    " SOUTH\\b" = " S",
    " EAST\\b" = " E",
    " WEST\\b" = " W"
  )
  for (pattern in names(replacements)) {
    x <- str_replace_all(x, pattern, replacements[[pattern]])
  }
  x <- str_replace_all(x, "[^A-Z0-9]+", " ")
  x <- str_squish(x)
  x[x == ""] <- NA_character_
  x
}

safe_pct <- function(numerator, denominator) {
  ifelse(denominator > 0, 100 * numerator / denominator, NA_real_)
}

safe_min_date <- function(x) {
  x <- as.Date(x)
  if (all(is.na(x))) as.Date(NA) else min(x, na.rm = TRUE)
}

safe_max_date <- function(x) {
  x <- as.Date(x)
  if (all(is.na(x))) as.Date(NA) else max(x, na.rm = TRUE)
}

download_code_cases <- function() {
  selected_fields <- c(
    "case_id",
    "priority",
    "status",
    "address",
    "opened_date",
    "closed_date",
    "department",
    "case_type",
    "description",
    "date_updated",
    "last_update",
    "violationcasenumber",
    "parcelid",
    "latitude",
    "longitude",
    "repeatoffenderrelated",
    "shorttermrentalrelated",
    "servicerequestnumber"
  )
  where <- paste0(
    "opened_date >= '", format(START_DATE, "%Y-%m-%d"), "T00:00:00' ",
    "AND opened_date < '",
    format(END_DATE_EXCLUSIVE, "%Y-%m-%d"),
    "T00:00:00'"
  )
  headers <- if (nzchar(app_token)) {
    add_headers(`X-App-Token` = app_token)
  } else {
    add_headers(Accept = "application/json")
  }

  pages <- list()
  offset <- 0L
  repeat {
    print_progress(paste0("Downloading Code cases at offset ", offset, "..."))
    response <- GET(
      API_ENDPOINT,
      headers,
      query = list(
        `$select` = paste(selected_fields, collapse = ","),
        `$where` = where,
        `$order` = "opened_date,case_id",
        `$limit` = PAGE_SIZE,
        `$offset` = offset
      )
    )
    if (http_error(response)) {
      stop(
        "Austin Code complaint API request failed: ",
        status_code(response), " ",
        content(response, as = "text", encoding = "UTF-8"),
        call. = FALSE
      )
    }
    page <- jsonlite::fromJSON(
      content(response, as = "text", encoding = "UTF-8"),
      flatten = TRUE
    )
    page <- as_tibble(page)
    if (nrow(page) == 0L) break
    pages[[length(pages) + 1L]] <- page
    if (nrow(page) < PAGE_SIZE) break
    offset <- offset + PAGE_SIZE
  }

  data <- bind_rows(pages)
  if (nrow(data) == 0L) {
    stop("No Austin Code complaint cases were returned.", call. = FALSE)
  }
  cache <- list(
    schema_version = 1L,
    complete = TRUE,
    fetched_at = Sys.time(),
    dataset_id = DATASET_ID,
    start_date = START_DATE,
    analysis_as_of_date = END_DATE,
    data = data
  )
  saveRDS(cache, CACHE_FILE)
  data
}

code_cases_raw <- NULL
if (file.exists(CACHE_FILE) && !refresh) {
  cached <- readRDS(CACHE_FILE)
  cache_matches <- is.list(cached) &&
    identical(cached$schema_version, 1L) &&
    isTRUE(cached$complete) &&
    identical(cached$dataset_id, DATASET_ID) &&
    identical(as.Date(cached$start_date), START_DATE) &&
    identical(as.Date(cached$analysis_as_of_date), END_DATE)
  if (cache_matches) {
    code_cases_raw <- cached$data
    print_progress(paste0("Using cached Code cases: ", CACHE_FILE))
  }
}
if (is.null(code_cases_raw)) {
  code_cases_raw <- download_code_cases()
}

categories <- read_csv(
  CATEGORY_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
)
if (
  nrow(categories) == 0L ||
    any(is.na(categories$description_pattern)) ||
    anyDuplicated(categories$complaint_category)
) {
  stop("Code complaint category configuration is invalid.", call. = FALSE)
}

print_progress("Classifying complaint cases and linking service requests...")
code_cases <- code_cases_raw %>%
  mutate(
    across(everything(), blank_to_na),
    audit_row_id = row_number(),
    case_id = blank_to_na(case_id),
    service_request_id = normalize_service_request(servicerequestnumber),
    code_parcel_id = normalize_parcel_id(parcelid),
    opened_date = parse_api_date(opened_date),
    closed_date = parse_api_date(closed_date),
    date_updated = parse_api_date(date_updated),
    latitude = suppressWarnings(as.numeric(latitude)),
    longitude = suppressWarnings(as.numeric(longitude)),
    complaint_category = NA_character_
  )

for (index in seq_len(nrow(categories))) {
  matches <- str_detect(
    coalesce(code_cases$description, ""),
    regex(categories$description_pattern[[index]], ignore_case = TRUE)
  )
  already_classified <- !is.na(code_cases$complaint_category)
  if (any(matches & already_classified)) {
    stop("A Code complaint description matches multiple categories.", call. = FALSE)
  }
  code_cases$complaint_category[matches] <-
    categories$complaint_category[[index]]
}
code_cases$complaint_category <- coalesce(
  code_cases$complaint_category,
  "unmapped"
)
code_cases <- code_cases %>%
  mutate(
    case_address_key = normalize_base_address(address),
    has_violation_case = !is.na(violationcasenumber),
    repeat_offender_related = str_to_upper(repeatoffenderrelated) == "YES",
    short_term_rental_related = str_to_upper(shorttermrentalrelated) == "YES",
    has_valid_coordinates =
      is.finite(latitude) & is.finite(longitude) &
      between(latitude, 29.8, 30.7) &
      between(longitude, -98.3, -97.2)
  )

if (
  any(is.na(code_cases$case_id)) ||
    anyDuplicated(code_cases$case_id) ||
    any(code_cases$opened_date < START_DATE, na.rm = TRUE) ||
    any(code_cases$opened_date > END_DATE, na.rm = TRUE) ||
    any(code_cases$complaint_category == "unmapped") ||
    !setequal(
      unique(code_cases$complaint_category),
      categories$complaint_category
    )
) {
  stop(
    "Downloaded Code cases failed ID, category, or cutoff validation.",
    call. = FALSE
  )
}

ews_cache <- readRDS(EWS_311_CACHE_FILE)
if (
  !is.list(ews_cache) ||
    !isTRUE(ews_cache$complete) ||
    !identical(as.Date(ews_cache$analysis_as_of_date), END_DATE)
) {
  stop("Existing EWS 311 cache is incomplete or uses another cutoff.", call. = FALSE)
}
ews_311 <- ews_cache$data %>%
  transmute(
    service_request_id = normalize_service_request(sr_number),
    ews_311_created_date = parse_api_date(sr_created_date),
    ews_311_latitude = suppressWarnings(as.numeric(sr_location_lat)),
    ews_311_longitude = suppressWarnings(as.numeric(sr_location_long)),
    ews_311_type = as.character(sr_type_desc)
  )
if (
  any(is.na(ews_311$service_request_id)) ||
    anyDuplicated(ews_311$service_request_id)
) {
  stop("Existing EWS 311 cache does not have unique normalized IDs.", call. = FALSE)
}

code_cases <- code_cases %>%
  left_join(
    ews_311,
    by = "service_request_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    linked_to_ews_311 = !is.na(ews_311_created_date),
    case_open_lag_days = as.integer(opened_date - ews_311_created_date)
  )

missing_selected_311_ids <- sort(unique(na.omit(
  code_cases$service_request_id[!code_cases$linked_to_ews_311]
)))

fetch_unfiltered_311_ids <- function(request_ids) {
  if (length(request_ids) == 0L) return(tibble())

  quote_sql <- function(x) {
    paste0("'", str_replace_all(x, "'", "''"), "'")
  }
  selected_fields <- c(
    "sr_number",
    "sr_type_desc",
    "sr_department_desc",
    "sr_method_received_desc",
    "sr_status_desc",
    "sr_created_date",
    "sr_closed_date",
    "sr_location_lat",
    "sr_location_long"
  )
  chunks <- split(request_ids, ceiling(seq_along(request_ids) / 250L))
  pages <- vector("list", length(chunks))

  for (index in seq_along(chunks)) {
    print_progress(
      paste0(
        "Checking missing service-request IDs in unfiltered 311 source: ",
        index, "/", length(chunks), "..."
      )
    )
    query <- paste0(
      "SELECT ", paste(selected_fields, collapse = ", "), " ",
      "WHERE sr_number IN (",
      paste(vapply(chunks[[index]], quote_sql, character(1)), collapse = ","),
      ")"
    )
    request_body <- jsonlite::toJSON(list(query = query), auto_unbox = TRUE)
    response <- if (nzchar(api_key) && nzchar(api_secret)) {
      POST(
        UNFILTERED_311_API_ENDPOINT,
        authenticate(api_key, api_secret),
        content_type_json(),
        body = request_body,
        encode = "raw"
      )
    } else {
      POST(
        UNFILTERED_311_API_ENDPOINT,
        content_type_json(),
        body = request_body,
        encode = "raw"
      )
    }
    if (http_error(response)) {
      stop(
        "Unfiltered Austin 311 identifier query failed: ",
        status_code(response), " ",
        content(response, as = "text", encoding = "UTF-8"),
        call. = FALSE
      )
    }
    parsed <- jsonlite::fromJSON(
      content(response, as = "text", encoding = "UTF-8"),
      flatten = TRUE
    )
    pages[[index]] <- if (length(parsed) == 0L) tibble() else as_tibble(parsed)
  }

  bind_rows(pages)
}

unfiltered_311_raw <- NULL
if (file.exists(UNFILTERED_311_CACHE_FILE) && !refresh) {
  cached <- readRDS(UNFILTERED_311_CACHE_FILE)
  cache_matches <- is.list(cached) &&
    identical(cached$schema_version, 1L) &&
    isTRUE(cached$complete) &&
    identical(as.Date(cached$analysis_as_of_date), END_DATE) &&
    identical(sort(cached$requested_ids), missing_selected_311_ids)
  if (cache_matches) {
    unfiltered_311_raw <- cached$data
    print_progress(
      paste0("Using cached unfiltered 311 ID audit: ", UNFILTERED_311_CACHE_FILE)
    )
  }
}
if (is.null(unfiltered_311_raw)) {
  unfiltered_311_raw <- fetch_unfiltered_311_ids(missing_selected_311_ids)
  saveRDS(
    list(
      schema_version = 1L,
      complete = TRUE,
      fetched_at = Sys.time(),
      analysis_as_of_date = END_DATE,
      requested_ids = missing_selected_311_ids,
      data = unfiltered_311_raw
    ),
    UNFILTERED_311_CACHE_FILE
  )
}

unfiltered_311 <- unfiltered_311_raw %>%
  transmute(
    service_request_id = normalize_service_request(sr_number),
    unfiltered_311_found = TRUE,
    unfiltered_311_type = blank_to_na(sr_type_desc),
    unfiltered_311_department = blank_to_na(sr_department_desc),
    unfiltered_311_method_received = blank_to_na(sr_method_received_desc),
    unfiltered_311_status = blank_to_na(sr_status_desc),
    unfiltered_311_created_date = parse_api_date(sr_created_date),
    unfiltered_311_closed_date = parse_api_date(sr_closed_date),
    unfiltered_311_latitude = suppressWarnings(as.numeric(sr_location_lat)),
    unfiltered_311_longitude = suppressWarnings(as.numeric(sr_location_long))
  )
if (
  any(is.na(unfiltered_311$service_request_id)) ||
    anyDuplicated(unfiltered_311$service_request_id) ||
    any(!unfiltered_311$service_request_id %in% missing_selected_311_ids)
) {
  stop("Unfiltered 311 identifier results failed validation.", call. = FALSE)
}

code_cases <- code_cases %>%
  left_join(
    unfiltered_311,
    by = "service_request_id",
    relationship = "many-to-one"
  ) %>%
  mutate(
    unfiltered_311_found = coalesce(unfiltered_311_found, FALSE),
    linked_to_full_311 = linked_to_ews_311 | unfiltered_311_found,
    full_311_link_source = case_when(
      linked_to_ews_311 ~ "existing_ews_selected_extract",
      unfiltered_311_found ~ "supplemental_identifier_query",
      is.na(service_request_id) ~ "no_service_request_id_reported",
      TRUE ~ "service_request_id_not_found"
    ),
    unfiltered_311_has_valid_coordinates =
      is.finite(unfiltered_311_latitude) &
      is.finite(unfiltered_311_longitude) &
      between(unfiltered_311_latitude, 29.8, 30.7) &
      between(unfiltered_311_longitude, -98.3, -97.2)
  )

distance_candidates <- code_cases %>%
  filter(
    linked_to_ews_311,
    has_valid_coordinates,
    is.finite(ews_311_latitude),
    is.finite(ews_311_longitude)
  )
if (nrow(distance_candidates) > 0L) {
  code_points <- st_as_sf(
    distance_candidates,
    coords = c("longitude", "latitude"),
    crs = 4326,
    remove = FALSE
  ) %>%
    st_transform(26914)
  intake_points <- st_as_sf(
    distance_candidates,
    coords = c("ews_311_longitude", "ews_311_latitude"),
    crs = 4326,
    remove = FALSE
  ) %>%
    st_transform(26914)
  distance_lookup <- tibble(
    audit_row_id = distance_candidates$audit_row_id,
    code_311_location_distance_m = as.numeric(
      st_distance(code_points, intake_points, by_element = TRUE)
    )
  )

  code_cases <- code_cases %>%
    left_join(
      distance_lookup,
      by = "audit_row_id",
      relationship = "one-to-one"
    )
} else {
  code_cases$code_311_location_distance_m <- NA_real_
}

print_progress("Building exact City-to-appraisal parcel crosswalk...")
land_use <- read_csv(
  LAND_USE_FILE,
  show_col_types = FALSE,
  col_types = cols(.default = col_character())
) %>%
  transmute(
    city_parcel_id = normalize_parcel_id(parcel_id_10),
    city_property_id = normalize_parcel_id(property_id),
    land_use_code = suppressWarnings(as.integer(land_use)),
    city_record_county = case_when(
      str_detect(city_parcel_id, "^[0-9]{10}$") ~ "Travis",
      str_detect(city_parcel_id, "^R[0-9]+$") &
        str_remove(city_parcel_id, "^R") == city_property_id ~ "Hays",
      str_detect(city_parcel_id, "^R[0-9]+$") ~ "Williamson",
      TRUE ~ NA_character_
    ),
    city_match_key = case_when(
      city_record_county == "Travis" ~ city_property_id,
      city_record_county %in% c("Hays", "Williamson") ~ city_parcel_id,
      TRUE ~ NA_character_
    )
  ) %>%
  filter(
    !is.na(city_parcel_id),
    !is.na(city_record_county),
    !is.na(city_match_key)
  )

city_parcel_crosswalk <- land_use %>%
  group_by(city_parcel_id) %>%
  summarise(
    city_crosswalk_county_count = n_distinct(city_record_county, na.rm = TRUE),
    city_crosswalk_key_count = n_distinct(city_match_key, na.rm = TRUE),
    city_record_county = if_else(
      city_crosswalk_county_count == 1L,
      first(na.omit(city_record_county)),
      NA_character_
    ),
    city_match_key = if_else(
      city_crosswalk_key_count == 1L,
      first(na.omit(city_match_key)),
      NA_character_
    ),
    city_land_use_codes = paste(
      sort(unique(na.omit(land_use_code))),
      collapse = ";"
    ),
    .groups = "drop"
  ) %>%
  mutate(
    city_parcel_crosswalk_unambiguous =
      city_crosswalk_county_count == 1L & city_crosswalk_key_count == 1L
  )

promoted <- readRDS(PROMOTED_PARCEL_FILE) %>%
  transmute(
    residential_parcel_id = as.character(parcel_id),
    source_county = as.character(source_county),
    parcel_match_key = case_when(
      source_county == "Travis" ~ as.character(parcel_id),
      source_county %in% c("Hays", "Williamson") ~
        str_remove(as.character(parcel_id), "^(HAYS|WILLIAMSON):"),
      TRUE ~ NA_character_
    ),
    residential_address_key = normalize_base_address(parcel_address_key),
    residential_latitude = suppressWarnings(as.numeric(lat)),
    residential_longitude = suppressWarnings(as.numeric(lon)),
    promoted_units = coalesce(suppressWarnings(as.numeric(promoted_units)), 0),
    is_multifamily_like = coalesce(as.logical(is_multifamily_like), FALSE),
    city_any_multiunit = coalesce(as.logical(city_any_multiunit), FALSE),
    in_austin_full_purpose = coalesce(
      as.logical(in_austin_full_purpose),
      FALSE
    )
  ) %>%
  mutate(
    is_multiunit_context =
      is_multifamily_like | city_any_multiunit | promoted_units > 1
  )
if (anyDuplicated(promoted$residential_parcel_id)) {
  stop("Promoted residential parcel IDs are not unique.", call. = FALSE)
}

promoted_crosswalk <- promoted %>%
  select(
    source_county,
    parcel_match_key,
    residential_parcel_id,
    promoted_units,
    is_multifamily_like,
    city_any_multiunit,
    is_multiunit_context,
    in_austin_full_purpose
  )
if (anyDuplicated(promoted_crosswalk[c("source_county", "parcel_match_key")])) {
  stop("Promoted parcel match keys are not unique.", call. = FALSE)
}

code_cases <- code_cases %>%
  left_join(
    city_parcel_crosswalk,
    by = c("code_parcel_id" = "city_parcel_id"),
    relationship = "many-to-one"
  ) %>%
  left_join(
    promoted_crosswalk,
    by = c(
      "city_record_county" = "source_county",
      "city_match_key" = "parcel_match_key"
    ),
    relationship = "many-to-one"
  ) %>%
  mutate(
    exact_city_parcel_match = !is.na(city_match_key),
    exact_promoted_residential_match = !is.na(residential_parcel_id)
  )

print_progress("Evaluating address and nearest-point fallback evidence...")
address_lookup <- promoted %>%
  filter(!is.na(residential_address_key)) %>%
  group_by(residential_address_key) %>%
  summarise(
    address_residential_parcel_count = n_distinct(residential_parcel_id),
    address_residential_units = sum(promoted_units),
    address_has_multiunit_context = any(is_multiunit_context),
    address_in_full_purpose = any(in_austin_full_purpose),
    .groups = "drop"
  )
code_cases <- code_cases %>%
  left_join(
    address_lookup,
    by = c("case_address_key" = "residential_address_key"),
    relationship = "many-to-one"
  ) %>%
  mutate(
    residential_address_match = !is.na(address_residential_parcel_count)
  )

unmatched_points <- code_cases %>%
  filter(!exact_promoted_residential_match, has_valid_coordinates)
residential_points <- promoted %>%
  filter(
    is.finite(residential_latitude),
    is.finite(residential_longitude)
  ) %>%
  st_as_sf(
    coords = c("residential_longitude", "residential_latitude"),
    crs = 4326,
    remove = FALSE
  ) %>%
  st_transform(26914)
if (nrow(unmatched_points) > 0L && nrow(residential_points) > 0L) {
  unmatched_sf <- unmatched_points %>%
    st_as_sf(
      coords = c("longitude", "latitude"),
      crs = 4326,
      remove = FALSE
    ) %>%
    st_transform(26914)
  nearest_index <- st_nearest_feature(unmatched_sf, residential_points)
  nearest_distance <- as.numeric(st_distance(
    unmatched_sf,
    residential_points[nearest_index, ],
    by_element = TRUE
  ))
  nearest_lookup <- tibble(
    audit_row_id = unmatched_points$audit_row_id,
    nearest_residential_parcel_id =
      residential_points$residential_parcel_id[nearest_index],
    nearest_residential_distance_m = nearest_distance,
    nearest_residential_is_multiunit =
      residential_points$is_multiunit_context[nearest_index]
  )
  code_cases <- code_cases %>%
    left_join(
      nearest_lookup,
      by = "audit_row_id",
      relationship = "one-to-one"
    )
} else {
  code_cases$nearest_residential_parcel_id <- NA_character_
  code_cases$nearest_residential_distance_m <- NA_real_
  code_cases$nearest_residential_is_multiunit <- NA
}

code_cases <- code_cases %>%
  mutate(
    conservative_residential_fallback =
      !exact_promoted_residential_match &
      !exact_city_parcel_match &
      (
        residential_address_match |
          coalesce(nearest_residential_distance_m <= 25, FALSE)
      ),
    any_residential_evidence =
      exact_promoted_residential_match | conservative_residential_fallback,
    any_multiunit_evidence = case_when(
      exact_promoted_residential_match ~ is_multiunit_context,
      conservative_residential_fallback & residential_address_match ~
        address_has_multiunit_context,
      conservative_residential_fallback &
        nearest_residential_distance_m <= 25 ~
        nearest_residential_is_multiunit,
      TRUE ~ FALSE
    ),
    residential_match_method = case_when(
      exact_promoted_residential_match ~ "exact_city_parcel_to_promoted_parcel",
      exact_city_parcel_match ~ "exact_city_parcel_not_in_promoted_residential",
      conservative_residential_fallback & residential_address_match ~
        "normalized_address_fallback",
      conservative_residential_fallback & nearest_residential_distance_m <= 25 ~
        "nearest_parcel_within_25m",
      nearest_residential_distance_m <= 50 ~ "nearest_parcel_25_to_50m",
      nearest_residential_distance_m <= 100 ~ "nearest_parcel_50_to_100m",
      is.finite(nearest_residential_distance_m) ~ "nearest_parcel_over_100m",
      TRUE ~ "no_residential_location_evidence"
    )
  )

if (
  any(code_cases$exact_promoted_residential_match &
    !code_cases$exact_city_parcel_match) ||
    any(code_cases$conservative_residential_fallback &
      code_cases$exact_city_parcel_match) ||
    !identical(
      code_cases$any_residential_evidence,
      code_cases$exact_promoted_residential_match |
        code_cases$conservative_residential_fallback
    )
) {
  stop("Residential linkage hierarchy failed validation.", call. = FALSE)
}

print_progress("Writing Code complaint linkage audit outputs...")
source_audit <- code_cases %>%
  group_by(complaint_category) %>%
  summarise(
    source_cases = n(),
    unique_case_ids = n_distinct(case_id),
    cases_with_service_request_id = sum(!is.na(service_request_id)),
    unique_reported_service_requests = n_distinct(
      service_request_id,
      na.rm = TRUE
    ),
    cases_linked_to_full_311 = sum(linked_to_full_311),
    unique_full_311_requests_linked = n_distinct(
      service_request_id[linked_to_full_311],
      na.rm = TRUE
    ),
    cases_linked_to_ews_311 = sum(linked_to_ews_311),
    unique_ews_311_requests_linked = n_distinct(
      service_request_id[linked_to_ews_311],
      na.rm = TRUE
    ),
    cases_linked_to_ews_311_and_exact_residential = sum(
      linked_to_ews_311 & exact_promoted_residential_match
    ),
    unique_ews_311_requests_linked_to_exact_residential = n_distinct(
      service_request_id[
        linked_to_ews_311 & exact_promoted_residential_match
      ],
      na.rm = TRUE
    ),
    cases_linked_to_full_311_and_exact_residential = sum(
      linked_to_full_311 & exact_promoted_residential_match
    ),
    unique_full_311_requests_linked_to_exact_residential = n_distinct(
      service_request_id[
        linked_to_full_311 & exact_promoted_residential_match
      ],
      na.rm = TRUE
    ),
    cases_with_code_parcel_id = sum(!is.na(code_parcel_id)),
    cases_with_unambiguous_city_parcel_crosswalk = sum(
      exact_city_parcel_match
    ),
    cases_exactly_linked_to_promoted_residential_parcel = sum(
      exact_promoted_residential_match
    ),
    cases_with_conservative_residential_fallback = sum(
      conservative_residential_fallback
    ),
    cases_with_any_residential_evidence = sum(any_residential_evidence),
    cases_exactly_linked_to_multiunit_context = sum(
      exact_promoted_residential_match & is_multiunit_context
    ),
    cases_with_any_multiunit_evidence = sum(any_multiunit_evidence),
    cases_with_associated_violation_case = sum(has_violation_case),
    repeat_offender_related_cases = sum(repeat_offender_related, na.rm = TRUE),
    first_opened_date = safe_min_date(opened_date),
    last_opened_date = safe_max_date(opened_date),
    .groups = "drop"
  ) %>%
  mutate(
    pct_cases_with_service_request_id = safe_pct(
      cases_with_service_request_id,
      source_cases
    ),
    pct_cases_linked_to_full_311 = safe_pct(
      cases_linked_to_full_311,
      source_cases
    ),
    pct_reported_ids_linked_to_full_311 = safe_pct(
      cases_linked_to_full_311,
      cases_with_service_request_id
    ),
    pct_cases_linked_to_ews_311 = safe_pct(
      cases_linked_to_ews_311,
      source_cases
    ),
    pct_cases_exactly_linked_to_promoted_residential_parcel = safe_pct(
      cases_exactly_linked_to_promoted_residential_parcel,
      source_cases
    ),
    pct_cases_with_any_residential_evidence = safe_pct(
      cases_with_any_residential_evidence,
      source_cases
    ),
    pct_cases_linked_to_ews_311_and_exact_residential = safe_pct(
      cases_linked_to_ews_311_and_exact_residential,
      source_cases
    ),
    pct_cases_linked_to_full_311_and_exact_residential = safe_pct(
      cases_linked_to_full_311_and_exact_residential,
      source_cases
    ),
    pct_exact_residential_cases_with_multiunit_context = safe_pct(
      cases_exactly_linked_to_multiunit_context,
      cases_exactly_linked_to_promoted_residential_parcel
    ),
    pct_cases_with_any_multiunit_evidence = safe_pct(
      cases_with_any_multiunit_evidence,
      source_cases
    ),
    analysis_start_date = START_DATE,
    analysis_as_of_date = END_DATE,
    source_dataset_id = DATASET_ID
  )

overall_audit <- code_cases %>%
  summarise(
    complaint_category = "all_categories",
    source_cases = n(),
    unique_case_ids = n_distinct(case_id),
    cases_with_service_request_id = sum(!is.na(service_request_id)),
    unique_reported_service_requests = n_distinct(
      service_request_id,
      na.rm = TRUE
    ),
    cases_linked_to_full_311 = sum(linked_to_full_311),
    unique_full_311_requests_linked = n_distinct(
      service_request_id[linked_to_full_311],
      na.rm = TRUE
    ),
    cases_linked_to_ews_311 = sum(linked_to_ews_311),
    unique_ews_311_requests_linked = n_distinct(
      service_request_id[linked_to_ews_311],
      na.rm = TRUE
    ),
    cases_linked_to_ews_311_and_exact_residential = sum(
      linked_to_ews_311 & exact_promoted_residential_match
    ),
    unique_ews_311_requests_linked_to_exact_residential = n_distinct(
      service_request_id[
        linked_to_ews_311 & exact_promoted_residential_match
      ],
      na.rm = TRUE
    ),
    cases_linked_to_full_311_and_exact_residential = sum(
      linked_to_full_311 & exact_promoted_residential_match
    ),
    unique_full_311_requests_linked_to_exact_residential = n_distinct(
      service_request_id[
        linked_to_full_311 & exact_promoted_residential_match
      ],
      na.rm = TRUE
    ),
    cases_with_code_parcel_id = sum(!is.na(code_parcel_id)),
    cases_with_unambiguous_city_parcel_crosswalk = sum(
      exact_city_parcel_match
    ),
    cases_exactly_linked_to_promoted_residential_parcel = sum(
      exact_promoted_residential_match
    ),
    cases_with_conservative_residential_fallback = sum(
      conservative_residential_fallback
    ),
    cases_with_any_residential_evidence = sum(any_residential_evidence),
    cases_exactly_linked_to_multiunit_context = sum(
      exact_promoted_residential_match & is_multiunit_context
    ),
    cases_with_any_multiunit_evidence = sum(any_multiunit_evidence),
    cases_with_associated_violation_case = sum(has_violation_case),
    repeat_offender_related_cases = sum(repeat_offender_related, na.rm = TRUE),
    first_opened_date = safe_min_date(opened_date),
    last_opened_date = safe_max_date(opened_date)
  ) %>%
  mutate(
    pct_cases_with_service_request_id = safe_pct(
      cases_with_service_request_id,
      source_cases
    ),
    pct_cases_linked_to_full_311 = safe_pct(
      cases_linked_to_full_311,
      source_cases
    ),
    pct_reported_ids_linked_to_full_311 = safe_pct(
      cases_linked_to_full_311,
      cases_with_service_request_id
    ),
    pct_cases_linked_to_ews_311 = safe_pct(
      cases_linked_to_ews_311,
      source_cases
    ),
    pct_cases_exactly_linked_to_promoted_residential_parcel = safe_pct(
      cases_exactly_linked_to_promoted_residential_parcel,
      source_cases
    ),
    pct_cases_with_any_residential_evidence = safe_pct(
      cases_with_any_residential_evidence,
      source_cases
    ),
    pct_cases_linked_to_ews_311_and_exact_residential = safe_pct(
      cases_linked_to_ews_311_and_exact_residential,
      source_cases
    ),
    pct_cases_linked_to_full_311_and_exact_residential = safe_pct(
      cases_linked_to_full_311_and_exact_residential,
      source_cases
    ),
    pct_exact_residential_cases_with_multiunit_context = safe_pct(
      cases_exactly_linked_to_multiunit_context,
      cases_exactly_linked_to_promoted_residential_parcel
    ),
    pct_cases_with_any_multiunit_evidence = safe_pct(
      cases_with_any_multiunit_evidence,
      source_cases
    ),
    analysis_start_date = START_DATE,
    analysis_as_of_date = END_DATE,
    source_dataset_id = DATASET_ID
  )

source_audit <- bind_rows(overall_audit, source_audit)

linkage_methods <- code_cases %>%
  count(
    complaint_category,
    linked_to_ews_311,
    residential_match_method,
    name = "case_count"
  ) %>%
  group_by(complaint_category) %>%
  mutate(
    category_case_count = sum(case_count),
    category_case_pct = safe_pct(case_count, category_case_count)
  ) %>%
  ungroup() %>%
  arrange(complaint_category, desc(case_count))

year_qa <- code_cases %>%
  mutate(opened_year = year(opened_date)) %>%
  group_by(opened_year, complaint_category) %>%
  summarise(
    source_cases = n(),
    cases_linked_to_ews_311 = sum(linked_to_ews_311),
    cases_exactly_linked_to_promoted_residential_parcel = sum(
      exact_promoted_residential_match
    ),
    cases_with_any_residential_evidence = sum(any_residential_evidence),
    cases_with_any_multiunit_evidence = sum(any_multiunit_evidence),
    .groups = "drop"
  ) %>%
  mutate(
    pct_linked_to_ews_311 = safe_pct(cases_linked_to_ews_311, source_cases),
    pct_with_any_residential_evidence = safe_pct(
      cases_with_any_residential_evidence,
      source_cases
    )
  )

month_qa <- code_cases %>%
  mutate(opened_month = floor_date(opened_date, unit = "month")) %>%
  group_by(opened_month, complaint_category) %>%
  summarise(
    source_cases = n(),
    cases_with_service_request_id = sum(!is.na(service_request_id)),
    cases_linked_to_ews_311 = sum(linked_to_ews_311),
    cases_exactly_linked_to_promoted_residential_parcel = sum(
      exact_promoted_residential_match
    ),
    .groups = "drop"
  ) %>%
  complete(
    opened_month = seq(
      floor_date(START_DATE, unit = "month"),
      floor_date(END_DATE, unit = "month"),
      by = "month"
    ),
    complaint_category = categories$complaint_category,
    fill = list(
      source_cases = 0L,
      cases_with_service_request_id = 0L,
      cases_linked_to_ews_311 = 0L,
      cases_exactly_linked_to_promoted_residential_parcel = 0L
    )
  ) %>%
  mutate(
    pct_cases_linked_to_ews_311 = safe_pct(
      cases_linked_to_ews_311,
      source_cases
    )
  ) %>%
  arrange(opened_month, complaint_category)

# August 2023 is the first month in the downloaded public series with sustained
# case volume. Keep the full period visible, but report overlap separately so
# the sparse earlier publication history is not mistaken for low activity.
dense_public_series_start <- as.Date("2023-08-01")
latest_12mo_start <- END_DATE %m-% years(1) + days(1)
previous_12mo_start <- latest_12mo_start %m-% years(1)
overlap_periods <- tribble(
  ~period, ~period_start, ~period_end,
  "full_requested_period", START_DATE, END_DATE,
  "dense_public_case_series", dense_public_series_start, END_DATE,
  "current_two_12_month_windows", previous_12mo_start, END_DATE,
  "latest_12_month_window", latest_12mo_start, END_DATE
)

overlap_period_qa <- overlap_periods %>%
  rowwise() %>%
  mutate(
    ews_311_requests = sum(
      ews_311$ews_311_created_date >= period_start &
        ews_311$ews_311_created_date <= period_end,
      na.rm = TRUE
    ),
    ews_311_requests_linked_to_any_code_case = n_distinct(
      code_cases$service_request_id[
        code_cases$linked_to_ews_311 &
          code_cases$ews_311_created_date >= period_start &
          code_cases$ews_311_created_date <= period_end
      ],
      na.rm = TRUE
    ),
    ews_311_requests_linked_to_structure_condition_case = n_distinct(
      code_cases$service_request_id[
        code_cases$linked_to_ews_311 &
          code_cases$complaint_category == "structure_condition" &
          code_cases$ews_311_created_date >= period_start &
          code_cases$ews_311_created_date <= period_end
      ],
      na.rm = TRUE
    ),
    pct_ews_311_linked_to_any_code_case = safe_pct(
      ews_311_requests_linked_to_any_code_case,
      ews_311_requests
    ),
    pct_ews_311_linked_to_structure_condition_case = safe_pct(
      ews_311_requests_linked_to_structure_condition_case,
      ews_311_requests
    )
  ) %>%
  ungroup()

print_progress("Measuring strict structure-condition coverage by Part 1 hex...")
hex_grid <- readRDS(HEX_GRID_FILE) %>%
  st_transform(4326) %>%
  select(hex_id)
eligible_hexes <- read_csv(
  PART1_ASSIGNMENTS_FILE,
  show_col_types = FALSE,
  col_types = cols(hex_id = col_double(), .default = col_skip())
) %>%
  distinct(hex_id)
if (nrow(eligible_hexes) == 0L || anyDuplicated(eligible_hexes$hex_id)) {
  stop("Part 1 eligible hex IDs failed validation.", call. = FALSE)
}

strict_structure_calls <- code_cases %>%
  filter(
    complaint_category == "structure_condition",
    linked_to_ews_311,
    exact_promoted_residential_match,
    !is.na(service_request_id),
    is.finite(ews_311_latitude),
    is.finite(ews_311_longitude)
  ) %>%
  arrange(audit_row_id) %>%
  distinct(service_request_id, .keep_all = TRUE) %>%
  mutate(
    window = case_when(
      ews_311_created_date >= latest_12mo_start &
        ews_311_created_date <= END_DATE ~ "latest_12m",
      ews_311_created_date >= previous_12mo_start &
        ews_311_created_date < latest_12mo_start ~ "previous_12m",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(window)) %>%
  st_as_sf(
    coords = c("ews_311_longitude", "ews_311_latitude"),
    crs = 4326,
    remove = FALSE
  ) %>%
  st_join(hex_grid, join = st_within, left = FALSE) %>%
  st_drop_geometry() %>%
  inner_join(eligible_hexes, by = "hex_id")

if (anyDuplicated(strict_structure_calls$service_request_id)) {
  stop("A strict structure-condition request joined to multiple hexes.", call. = FALSE)
}

strict_hex_counts <- strict_structure_calls %>%
  count(hex_id, window, name = "request_count")
strict_hex_panel <- crossing(
  hex_id = eligible_hexes$hex_id,
  window = c("previous_12m", "latest_12m")
) %>%
  left_join(strict_hex_counts, by = c("hex_id", "window")) %>%
  mutate(request_count = coalesce(request_count, 0L))

current_311_hex <- readRDS(EWS_311_HEX_FILE) %>%
  st_drop_geometry() %>%
  select(
    hex_id,
    sr_311_smoke_signal_previous_12mo,
    sr_311_smoke_signal_latest_12mo
  ) %>%
  right_join(eligible_hexes, by = "hex_id") %>%
  mutate(
    across(
      starts_with("sr_311_smoke_signal_"),
      ~ coalesce(as.numeric(.x), 0)
    )
  ) %>%
  pivot_longer(
    starts_with("sr_311_smoke_signal_"),
    names_to = "window",
    values_to = "request_count"
  ) %>%
  mutate(
    window = recode(
      window,
      sr_311_smoke_signal_previous_12mo = "previous_12m",
      sr_311_smoke_signal_latest_12mo = "latest_12m"
    )
  )

summarize_hex_sparsity <- function(data, measure_scope) {
  data %>%
    group_by(window) %>%
    summarise(
      eligible_hexes = n(),
      nonzero_hexes = sum(request_count > 0),
      pct_nonzero_hexes = safe_pct(nonzero_hexes, eligible_hexes),
      total_requests = sum(request_count),
      median_requests_among_nonzero_hexes = if_else(
        nonzero_hexes > 0,
        median(request_count[request_count > 0]),
        NA_real_
      ),
      p95_requests_among_nonzero_hexes = if_else(
        nonzero_hexes > 0,
        unname(quantile(request_count[request_count > 0], 0.95)),
        NA_real_
      ),
      maximum_requests_in_one_hex = max(request_count),
      .groups = "drop"
    ) %>%
    mutate(
      measure_scope = measure_scope,
      window_start = if_else(
        window == "latest_12m",
        latest_12mo_start,
        previous_12mo_start
      ),
      window_end = if_else(
        window == "latest_12m",
        END_DATE,
        latest_12mo_start - days(1)
      ),
      .before = 1
    )
}

hex_sparsity_audit <- bind_rows(
  summarize_hex_sparsity(
    strict_hex_panel,
    "structure_condition_linked_to_exact_residential_parcel"
  ),
  summarize_hex_sparsity(
    current_311_hex,
    "current_configured_code_officer_intake"
  )
)

unfiltered_311_types <- code_cases %>%
  filter(full_311_link_source == "supplemental_identifier_query") %>%
  count(
    complaint_category,
    unfiltered_311_type,
    unfiltered_311_department,
    unfiltered_311_has_valid_coordinates,
    name = "case_count"
  ) %>%
  arrange(complaint_category, desc(case_count), unfiltered_311_type)

service_request_category_count <- code_cases %>%
  filter(!is.na(service_request_id)) %>%
  distinct(service_request_id, complaint_category) %>%
  count(service_request_id, name = "category_count")
cardinality_audit <- tibble(
  metric = c(
    "source_rows",
    "duplicate_case_id_rows",
    "cases_without_service_request_id",
    "service_request_ids_not_found_in_full_311",
    "service_requests_linked_to_multiple_cases",
    "service_requests_linked_to_multiple_categories",
    "ews_311_intake_requests",
    "full_311_requests_linked_to_any_code_case",
    "full_311_requests_linked_to_structure_condition_case",
    "ews_311_requests_linked_to_any_code_case",
    "ews_311_requests_linked_to_structure_condition_case",
    "linked_cases_with_open_lag_over_30_days",
    "linked_cases_with_location_distance_over_100m"
  ),
  value = c(
    nrow(code_cases),
    sum(duplicated(code_cases$case_id)),
    sum(is.na(code_cases$service_request_id)),
    n_distinct(
      code_cases$service_request_id[
        !is.na(code_cases$service_request_id) & !code_cases$linked_to_full_311
      ],
      na.rm = TRUE
    ),
    code_cases %>%
      filter(!is.na(service_request_id)) %>%
      count(service_request_id) %>%
      summarise(value = sum(n > 1L)) %>%
      pull(value),
    sum(service_request_category_count$category_count > 1L),
    nrow(ews_311),
    n_distinct(
      code_cases$service_request_id[code_cases$linked_to_full_311],
      na.rm = TRUE
    ),
    n_distinct(
      code_cases$service_request_id[
        code_cases$linked_to_full_311 &
          code_cases$complaint_category == "structure_condition"
      ],
      na.rm = TRUE
    ),
    n_distinct(code_cases$service_request_id[code_cases$linked_to_ews_311]),
    n_distinct(code_cases$service_request_id[
      code_cases$linked_to_ews_311 &
        code_cases$complaint_category == "structure_condition"
    ]),
    sum(
      code_cases$linked_to_ews_311 &
        abs(code_cases$case_open_lag_days) > 30,
      na.rm = TRUE
    ),
    sum(
      code_cases$linked_to_ews_311 &
        code_cases$code_311_location_distance_m > 100,
      na.rm = TRUE
    )
  )
)

write_csv(
  source_audit,
  file.path(OUTPUT_DIR, "311_code_complaint_source_audit.csv")
)
write_csv(
  linkage_methods,
  file.path(OUTPUT_DIR, "311_code_complaint_linkage_methods.csv")
)
write_csv(
  year_qa,
  file.path(OUTPUT_DIR, "311_code_complaint_year_qa.csv")
)
write_csv(
  month_qa,
  file.path(OUTPUT_DIR, "311_code_complaint_month_qa.csv")
)
write_csv(
  overlap_period_qa,
  file.path(OUTPUT_DIR, "311_code_complaint_overlap_period_qa.csv")
)
write_csv(
  hex_sparsity_audit,
  file.path(OUTPUT_DIR, "311_code_complaint_hex_sparsity_audit.csv")
)
write_csv(
  unfiltered_311_types,
  file.path(OUTPUT_DIR, "311_code_complaint_unfiltered_311_types.csv")
)
write_csv(
  cardinality_audit,
  file.path(OUTPUT_DIR, "311_code_complaint_cardinality_audit.csv")
)
saveRDS(
  code_cases,
  file.path(OUTPUT_DIR, "311_code_complaint_case_audit.rds")
)

structure_summary <- source_audit %>%
  filter(complaint_category == "structure_condition")
if (nrow(structure_summary) != 1L) {
  stop("Structure-condition audit row is missing.", call. = FALSE)
}

print_progress(
  paste0(
    "Structure-condition cases: ",
    scales::comma(structure_summary$source_cases),
    "; linked to EWS 311: ",
    scales::comma(structure_summary$cases_linked_to_ews_311),
    " (", scales::percent(
      structure_summary$pct_cases_linked_to_ews_311 / 100,
      accuracy = 0.1
    ), ")"
  )
)
print_progress(
  paste0(
    "Exact promoted residential parcel matches: ",
    scales::comma(
      structure_summary$cases_exactly_linked_to_promoted_residential_parcel
    ),
    "; any conservative residential evidence: ",
    scales::comma(structure_summary$cases_with_any_residential_evidence),
    " (", scales::percent(
      structure_summary$pct_cases_with_any_residential_evidence / 100,
      accuracy = 0.1
    ), ")"
  )
)
cat("Audit summary: output/311_code_complaint_source_audit.csv\n")
cat("Hex coverage: output/311_code_complaint_hex_sparsity_audit.csv\n")
cat("Case-level audit: output/311_code_complaint_case_audit.rds\n")
