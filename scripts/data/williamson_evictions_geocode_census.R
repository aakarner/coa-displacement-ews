################################################################################
# Add a Census Geocoder Fallback for Williamson Eviction Addresses
################################################################################
#
# The local Williamson address-point matcher remains the first-choice source.
# This stage submits the same candidate address strings to the official U.S.
# Census Bureau batch geocoder, uses those results only for locally unresolved
# addresses, and writes one deterministic cascade registry with the same wide
# contract as the reviewed Travis registry.
#
# Defendant address transmission is enabled only for the initial cache fill:
#
#   WILLIAMSON_EVICTION_CENSUS_NETWORK=true \
#     Rscript scripts/data/williamson_evictions_geocode_census.R
#
# Later runs validate and reuse the ignored local cache without a network call.
# The address-bearing request, raw response, merged registry, and review detail
# all live under output/, which is ignored by git. The tracked pipeline uses the
# non-sensitive QA counts and hashes to document the frozen geocoding result.
################################################################################

required_packages <- c(
  "curl", "digest", "dplyr", "here", "httr2", "readr", "sf", "stringr",
  "tibble"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before Census geocoding: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(stringr)
})

UNIQUE_ADDRESS_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_UNIQUE_ADDRESS_FILE",
  here::here(
    "output",
    "williamson_eviction_unique_addresses_for_geocoding.csv"
  )
)
PREPARED_FILING_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_PREPARED_FILING_FILE",
  here::here(
    "output",
    "williamson_eviction_filings_prepared_for_geocoding.csv"
  )
)
LOCAL_GEOCODE_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_LOCAL_GEOCODE_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded_local.csv")
)
COA_GEOCODE_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_GEOCODE_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded_coa.csv")
)
COA_REVIEW_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_REVIEW_FILE",
  here::here("output", "williamson_eviction_geocode_coa_review.csv")
)
GRID_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_GRID_FILE",
  here::here("output", "hex_grid.rds")
)
JURISDICTIONS_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_JURISDICTIONS_FILE",
  here::here("data", "BOUNDARIES_jurisdictions_20260429.geojson")
)
OUTPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_GEOCODE_OUTPUT_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded.csv")
)
QA_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_GEOCODE_QA_FILE",
  here::here("output", "williamson_eviction_geocode_qa.csv")
)
REVIEW_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_GEOCODE_REVIEW_FILE",
  here::here("output", "williamson_eviction_geocode_review.csv")
)
CACHE_ROOT <- Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_CACHE_DIR",
  here::here("output", "williamson_eviction_census_geocode_cache")
)

CENSUS_BATCH_URL <- Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_BATCH_URL",
  paste0(
    "https://geocoding.geo.census.gov/geocoder/locations/",
    "addressbatch"
  )
)
CENSUS_BENCHMARK <- Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_BENCHMARK",
  "Public_AR_Current"
)
CACHE_SCHEMA_VERSION <- 1L
NETWORK_ENABLED <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_NETWORK",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
FORCE_REFRESH <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_FORCE_REFRESH",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
BATCH_SIZE <- suppressWarnings(as.integer(Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_BATCH_SIZE",
  "1000"
)))
REQUEST_TIMEOUT <- suppressWarnings(as.numeric(Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_TIMEOUT",
  "300"
)))
REQUEST_RETRIES <- suppressWarnings(as.integer(Sys.getenv(
  "WILLIAMSON_EVICTION_CENSUS_RETRIES",
  "3"
)))
ACCEPTED_MATCH_TYPES <- trimws(strsplit(
  Sys.getenv(
    "WILLIAMSON_EVICTION_CENSUS_ACCEPT_MATCH_TYPES",
    "Exact|Non_Exact"
  ),
  "|",
  fixed = TRUE
)[[1]])
ACCEPTED_MATCH_TYPES <- ACCEPTED_MATCH_TYPES[
  ACCEPTED_MATCH_TYPES %in% c("Exact", "Non_Exact")
]

if (is.na(BATCH_SIZE) || BATCH_SIZE < 1L || BATCH_SIZE > 10000L) {
  stop(
    "WILLIAMSON_EVICTION_CENSUS_BATCH_SIZE must be between 1 and 10000.",
    call. = FALSE
  )
}
if (is.na(REQUEST_TIMEOUT) || REQUEST_TIMEOUT <= 0) {
  stop(
    "WILLIAMSON_EVICTION_CENSUS_TIMEOUT must be positive.",
    call. = FALSE
  )
}
if (is.na(REQUEST_RETRIES) || REQUEST_RETRIES < 1L) {
  stop(
    "WILLIAMSON_EVICTION_CENSUS_RETRIES must be a positive integer.",
    call. = FALSE
  )
}
if (length(ACCEPTED_MATCH_TYPES) == 0L) {
  stop(
    "At least one supported Census match type must be accepted.",
    call. = FALSE
  )
}

required_files <- c(
  UNIQUE_ADDRESS_FILE,
  PREPARED_FILING_FILE,
  LOCAL_GEOCODE_FILE,
  COA_GEOCODE_FILE,
  COA_REVIEW_FILE,
  GRID_FILE,
  JURISDICTIONS_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing Census-fallback input(s): ",
    paste(basename(missing_files), collapse = ", "),
    call. = FALSE
  )
}

as_flag <- function(x) {
  tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
}

hash_text <- function(x) {
  digest::digest(enc2utf8(x), algo = "sha256", serialize = FALSE)
}

hash_file <- function(path) {
  digest::digest(path, algo = "sha256", file = TRUE)
}

hash_object <- function(x) {
  digest::digest(x, algo = "sha256", serialize = TRUE)
}

atomic_save_rds <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(
    pattern = paste0(".", basename(path), "-"),
    tmpdir = dirname(path)
  )
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(object, temporary, version = 3)
  if (!file.rename(temporary, path)) {
    stop("Could not atomically replace a Census cache file.", call. = FALSE)
  }
  invisible(path)
}

atomic_write_csv <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- tempfile(
    pattern = paste0(".", basename(path), "-"),
    tmpdir = dirname(path),
    fileext = ".csv"
  )
  on.exit(unlink(temporary), add = TRUE)
  readr::write_csv(data, temporary, na = "")
  if (!file.rename(temporary, path)) {
    stop("Could not atomically replace a geocoding output.", call. = FALSE)
  }
  invisible(path)
}

assert_columns <- function(data_names, required, label) {
  missing <- setdiff(required, data_names)
  if (length(missing) > 0L) {
    stop(
      label,
      " is missing required column(s): ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
}

valid_coordinates <- function(longitude, latitude) {
  is.finite(longitude) &
    is.finite(latitude) &
    dplyr::between(longitude, -180, 180) &
    dplyr::between(latitude, -90, 90)
}

extract_first_match <- function(value, pattern) {
  result <- stringr::str_match(as.character(value), pattern)[, 2]
  result[result == ""] <- NA_character_
  result
}

extract_input_zip <- function(value) {
  extract_first_match(
    toupper(as.character(value)),
    "\\bTX[[:space:]]+([0-9]{5})(?:-[0-9]{4})?\\b"
  )
}

extract_leading_house_number <- function(value) {
  toupper(extract_first_match(
    trimws(as.character(value)),
    "^([0-9]+[A-Za-z]?)\\b"
  ))
}

clean_external_address <- function(value) {
  value <- trimws(as.character(value))
  # JP1 cells retain a spreadsheet wrapper quote at the end of the address.
  # Removing only terminal wrappers preserves meaningful interior characters.
  value <- gsub('^["“”]+|["“”]+$', "", value, perl = TRUE)
  value <- gsub("[\r\n\t]+", " ", value, perl = TRUE)
  trimws(gsub("[[:space:]]+", " ", value))
}

parse_census_batch <- function(response_text) {
  if (!is.character(response_text) || length(response_text) != 1L) {
    stop("A Census response must be one text value.", call. = FALSE)
  }
  response_columns <- c(
    "address_id", "input_address", "match_indicator", "match_type",
    "matched_address", "coordinates", "tigerline_id", "side"
  )
  # No_Match rows contain only their first three fields; matched rows contain
  # all eight. Base read.csv(fill = TRUE) preserves this documented ragged
  # layout, whereas a width guessed from an early No_Match row truncates later
  # matches.
  result <- tryCatch(
    utils::read.csv(
      text = response_text,
      header = FALSE,
      col.names = response_columns,
      colClasses = "character",
      fill = TRUE,
      na.strings = character(),
      strip.white = FALSE,
      check.names = FALSE,
      stringsAsFactors = FALSE
    ) |>
      tibble::as_tibble(),
    error = function(error) NULL
  )
  if (is.null(result) || ncol(result) != 8L) {
    stop(
      "The Census batch response did not have the expected eight columns.",
      call. = FALSE
    )
  }
  coordinate_parts <- stringr::str_split_fixed(
    dplyr::coalesce(result$coordinates, ""),
    ",",
    2L
  )
  result |>
    dplyr::mutate(
      across(everything(), ~ trimws(as.character(.x))),
      across(everything(), ~ dplyr::na_if(.x, "")),
      longitude = suppressWarnings(as.numeric(coordinate_parts[, 1L])),
      latitude = suppressWarnings(as.numeric(coordinate_parts[, 2L])),
      input_zip = extract_input_zip(.data$input_address),
      matched_zip = extract_first_match(
        .data$matched_address,
        ",?[[:space:]]*([0-9]{5})(?:-[0-9]{4})?[[:space:]]*$"
      ),
      input_house_number = extract_leading_house_number(.data$input_address),
      matched_house_number = extract_leading_house_number(
        .data$matched_address
      ),
      matched_state_tx = stringr::str_detect(
        dplyr::coalesce(.data$matched_address, ""),
        ",?[[:space:]]TX,?[[:space:]]+[0-9]{5}(?:-[0-9]{4})?[[:space:]]*$"
      ),
      zip_agrees = !is.na(.data$input_zip) &
        .data$input_zip == .data$matched_zip,
      house_number_agrees = !is.na(.data$input_house_number) &
        .data$input_house_number == .data$matched_house_number,
      valid_coordinate = valid_coordinates(.data$longitude, .data$latitude),
      passes_quality_gate = .data$match_indicator == "Match" &
        .data$match_type %in% ACCEPTED_MATCH_TYPES &
        .data$matched_state_tx &
        .data$zip_agrees &
        .data$house_number_agrees &
        .data$valid_coordinate
    )
}

validate_batch_result <- function(result, expected_ids) {
  expected_ids <- as.character(expected_ids)
  if (
    nrow(result) != length(expected_ids) ||
      anyNA(result$address_id) ||
      anyDuplicated(result$address_id) ||
      !setequal(result$address_id, expected_ids)
  ) {
    stop(
      "The Census batch response did not return each request ID exactly once.",
      call. = FALSE
    )
  }
  result |>
    dplyr::mutate(
      request_order = match(.data$address_id, expected_ids),
      .before = 1
    ) |>
    dplyr::arrange(.data$request_order) |>
    dplyr::select(-"request_order")
}

validate_cache <- function(payload, signature, expected_ids) {
  if (
    !is.list(payload) ||
      !identical(payload$cache_schema_version, CACHE_SCHEMA_VERSION) ||
      !identical(payload$request_signature_sha256, signature) ||
      !is.character(payload$response_text) ||
      length(payload$response_text) != 1L ||
      !identical(hash_text(payload$response_text), payload$response_sha256)
  ) {
    return(NULL)
  }
  result <- tryCatch(
    parse_census_batch(payload$response_text),
    error = function(error) NULL
  )
  if (is.null(result)) return(NULL)
  tryCatch(
    validate_batch_result(result, expected_ids),
    error = function(error) NULL
  )
}

perform_census_batch <- function(request_file) {
  if (!NETWORK_ENABLED) {
    stop(
      "A Census geocoding cache page is missing or invalid. Run with ",
      "WILLIAMSON_EVICTION_CENSUS_NETWORK=true to transmit and geocode the ",
      "authorized addresses.",
      call. = FALSE
    )
  }

  last_error <- NULL
  for (attempt in seq_len(REQUEST_RETRIES)) {
    response <- tryCatch(
      {
        request <- httr2::request(CENSUS_BATCH_URL) |>
          httr2::req_body_multipart(
            addressFile = curl::form_file(request_file, type = "text/csv"),
            benchmark = CENSUS_BENCHMARK
          ) |>
          httr2::req_timeout(REQUEST_TIMEOUT) |>
          httr2::req_user_agent("coa-displacement-ews/census-geocoder")
        httr2::req_perform(request)
      },
      error = function(error) {
        last_error <<- error
        NULL
      }
    )
    if (!is.null(response) && httr2::resp_status(response) == 200L) {
      return(httr2::resp_body_string(response))
    }
    if (attempt < REQUEST_RETRIES) {
      Sys.sleep(min(2^(attempt - 1L), 4))
    }
  }
  stop(
    "The Census batch request failed after ",
    REQUEST_RETRIES,
    " attempt(s). No address or response content is shown.",
    call. = FALSE
  )
}

message("Reading the local Williamson results and candidate registry...")
unique_addresses <- readr::read_csv(
  UNIQUE_ADDRESS_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
) |>
  dplyr::filter(as_flag(.data$geocoding_candidate)) |>
  dplyr::mutate(
    address_id_numeric = suppressWarnings(as.integer(.data$address_id))
  ) |>
  dplyr::arrange(.data$address_id_numeric)
local_geocodes <- readr::read_csv(
  LOCAL_GEOCODE_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
coa_geocodes <- readr::read_csv(
  COA_GEOCODE_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
coa_review <- readr::read_csv(
  COA_REVIEW_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
contract_columns <- names(local_geocodes)
required_contract_columns <- c(
  "address_id", "address_for_geocoding", "result_id", "loc_name",
  "status", "score", "match_addr", "addr_type", "longitude", "latitude"
)
assert_columns(
  names(unique_addresses),
  c("address_id", "address_for_geocoding", "geocoding_candidate"),
  basename(UNIQUE_ADDRESS_FILE)
)
assert_columns(
  contract_columns,
  required_contract_columns,
  basename(LOCAL_GEOCODE_FILE)
)
assert_columns(
  names(coa_review),
  c(
    "address_id", "jurisdiction_label", "longitude", "latitude",
    "status", "score", "zip_agrees", "house_number_agrees"
  ),
  basename(COA_REVIEW_FILE)
)
if (
  nrow(unique_addresses) == 0L ||
    anyNA(unique_addresses[c("address_id", "address_for_geocoding")]) ||
    anyNA(unique_addresses$address_id_numeric) ||
    anyDuplicated(unique_addresses$address_id) ||
    anyDuplicated(unique_addresses$address_for_geocoding) ||
    nrow(local_geocodes) != nrow(unique_addresses) ||
    nrow(coa_geocodes) != nrow(unique_addresses) ||
    anyNA(local_geocodes[c("address_id", "address_for_geocoding")]) ||
    anyNA(coa_geocodes[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(local_geocodes$address_id) ||
    anyDuplicated(local_geocodes$address_for_geocoding) ||
    anyDuplicated(coa_geocodes$address_id) ||
    anyDuplicated(coa_geocodes$address_for_geocoding) ||
    nrow(coa_review) != nrow(unique_addresses) ||
    anyNA(coa_review$address_id) ||
    anyDuplicated(coa_review$address_id) ||
    !setequal(local_geocodes$address_id, unique_addresses$address_id) ||
    !setequal(coa_geocodes$address_id, unique_addresses$address_id) ||
    !setequal(coa_review$address_id, unique_addresses$address_id) ||
    !setequal(
      local_geocodes$address_for_geocoding,
      unique_addresses$address_for_geocoding
    ) ||
    !setequal(
      coa_geocodes$address_for_geocoding,
      unique_addresses$address_for_geocoding
    ) ||
    !identical(names(coa_geocodes), contract_columns)
) {
  stop(
    "The local registry and candidate table must contain the same unique ",
    "Williamson addresses and IDs.",
    call. = FALSE
  )
}
local_geocodes <- unique_addresses |>
  dplyr::select("address_id", "address_for_geocoding") |>
  dplyr::left_join(
    local_geocodes,
    by = c("address_id", "address_for_geocoding"),
    relationship = "one-to-one"
  )
coa_geocodes <- unique_addresses |>
  dplyr::select("address_id", "address_for_geocoding") |>
  dplyr::left_join(
    coa_geocodes,
    by = c("address_id", "address_for_geocoding"),
    relationship = "one-to-one"
  )
coa_review <- unique_addresses |>
  dplyr::select("address_id") |>
  dplyr::left_join(
    coa_review,
    by = "address_id",
    relationship = "one-to-one"
  )

message(
  "Loading or requesting Census results in ",
  ceiling(nrow(unique_addresses) / BATCH_SIZE),
  " resumable batch(es)..."
)
dir.create(CACHE_ROOT, recursive = TRUE, showWarnings = FALSE)
request_rows <- unique_addresses |>
  dplyr::transmute(
    address_id = as.character(.data$address_id),
    street = clean_external_address(.data$address_for_geocoding),
    city = "",
    state = "",
    zip = ""
  )
request_batches <- split(
  request_rows,
  ceiling(seq_len(nrow(request_rows)) / BATCH_SIZE)
)
batch_results <- vector("list", length(request_batches))
batch_response_hashes <- character(length(request_batches))
downloaded_batches <- 0L
reused_batches <- 0L

for (batch_index in seq_along(request_batches)) {
  batch <- request_batches[[batch_index]]
  request_file <- tempfile(fileext = ".csv")
  on.exit(unlink(request_file), add = TRUE)
  readr::write_csv(
    batch,
    request_file,
    col_names = FALSE,
    na = "",
    quote = "needed"
  )
  request_file_hash <- hash_file(request_file)
  request_signature <- hash_object(list(
    cache_schema_version = CACHE_SCHEMA_VERSION,
    batch_url = CENSUS_BATCH_URL,
    benchmark = CENSUS_BENCHMARK,
    request_file_sha256 = request_file_hash,
    address_ids = batch$address_id
  ))
  cache_file <- file.path(
    CACHE_ROOT,
    sprintf(
      "batch_%05d_%s.rds",
      batch_index,
      substr(request_signature, 1L, 16L)
    )
  )

  result <- NULL
  payload <- NULL
  if (file.exists(cache_file) && !FORCE_REFRESH) {
    payload <- tryCatch(readRDS(cache_file), error = function(error) NULL)
    result <- validate_cache(payload, request_signature, batch$address_id)
  }

  if (is.null(result)) {
    response_text <- perform_census_batch(request_file)
    result <- validate_batch_result(
      parse_census_batch(response_text),
      batch$address_id
    )
    payload <- list(
      cache_schema_version = CACHE_SCHEMA_VERSION,
      request_signature_sha256 = request_signature,
      request_file_sha256 = request_file_hash,
      response_sha256 = hash_text(response_text),
      retrieved_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      response_text = response_text
    )
    atomic_save_rds(payload, cache_file)
    downloaded_batches <- downloaded_batches + 1L
  } else {
    reused_batches <- reused_batches + 1L
  }

  batch_results[[batch_index]] <- result
  batch_response_hashes[[batch_index]] <- payload$response_sha256
  unlink(request_file)
}

census <- dplyr::bind_rows(batch_results)
if (
  nrow(census) != nrow(unique_addresses) ||
    anyNA(census$address_id) ||
    anyDuplicated(census$address_id) ||
    !setequal(census$address_id, unique_addresses$address_id)
) {
  stop("The assembled Census result registry is incomplete.", call. = FALSE)
}
census <- unique_addresses |>
  dplyr::select("address_id", "address_for_geocoding") |>
  dplyr::left_join(
    census |>
      dplyr::select(-"input_address"),
    by = "address_id",
    relationship = "one-to-one"
  )

local_longitude <- suppressWarnings(as.numeric(local_geocodes$longitude))
local_latitude <- suppressWarnings(as.numeric(local_geocodes$latitude))
local_score <- suppressWarnings(as.numeric(local_geocodes$score))
local_accepted <- local_geocodes$status %in% c("M", "T") &
  local_score >= 90 &
  valid_coordinates(local_longitude, local_latitude)
coa_longitude <- suppressWarnings(as.numeric(coa_geocodes$longitude))
coa_latitude <- suppressWarnings(as.numeric(coa_geocodes$latitude))
coa_score <- suppressWarnings(as.numeric(coa_geocodes$score))
coa_accepted <- coa_geocodes$status %in% c("M", "T") &
  coa_score >= 95 &
  valid_coordinates(coa_longitude, coa_latitude)
coa_selected <- !local_accepted & coa_accepted
census_accepted <- census$passes_quality_gate %in% TRUE
census_selected <- !local_accepted & !coa_accepted & census_accepted

message("Merging local-first and Census-fallback geocodes...")
wide_output <- local_geocodes
wide_output$loc_name[!local_accepted] <- "williamson_geocode_cascade_unresolved"
wide_output$status[!local_accepted] <- "U"
wide_output$score[!local_accepted] <- "0"
if (any(coa_selected)) {
  for (column in contract_columns) {
    if (column %in% c("address_id", "address_for_geocoding")) next
    wide_output[[column]][coa_selected] <-
      coa_geocodes[[column]][coa_selected]
  }
}

matched_parts <- strsplit(
  dplyr::coalesce(census$matched_address, ""),
  ",",
  fixed = TRUE
)
matched_part <- function(parts, position_from_end) {
  vapply(
    parts,
    function(value) {
      value <- trimws(value)
      index <- length(value) - position_from_end + 1L
      if (index < 1L || index > length(value) || value[[index]] == "") {
        NA_character_
      } else {
        value[[index]]
      }
    },
    character(1)
  )
}
census_city <- matched_part(matched_parts, 3L)
census_region <- matched_part(matched_parts, 2L)
census_postal <- matched_part(matched_parts, 1L)

set_selected <- function(column, value) {
  if (column %in% names(wide_output)) {
    wide_output[[column]][census_selected] <<-
      as.character(value[census_selected])
  }
}

set_selected(
  "loc_name",
  paste0("census_public_ar_current_", tolower(census$match_type))
)
set_selected("status", rep("M", nrow(census)))
set_selected(
  "score",
  ifelse(census$match_type == "Exact", "95", "90")
)
set_selected("match_addr", census$matched_address)
set_selected("long_label", census$matched_address)
set_selected("short_label", census$matched_address)
set_selected("place_addr", census$matched_address)
set_selected("st_addr", census$matched_address)
set_selected("addr_type", rep("StreetAddress", nrow(census)))
set_selected("type_field", rep("Census_TIGER_address_range", nrow(census)))
set_selected("side", census$side)
set_selected("city", census_city)
set_selected("subregion", rep("Williamson", nrow(census)))
set_selected("region", rep("Texas", nrow(census)))
set_selected("region_abbr", census_region)
set_selected("postal", census_postal)
set_selected("country", rep("USA", nrow(census)))
set_selected("cntry_name", rep("United States", nrow(census)))
set_selected("x", census$longitude)
set_selected("y", census$latitude)
set_selected("display_x", census$longitude)
set_selected("display_y", census$latitude)
set_selected("xmin", census$longitude)
set_selected("xmax", census$longitude)
set_selected("ymin", census$latitude)
set_selected("ymax", census$latitude)
set_selected("longitude", census$longitude)
set_selected("latitude", census$latitude)

final_longitude <- suppressWarnings(as.numeric(wide_output$longitude))
final_latitude <- suppressWarnings(as.numeric(wide_output$latitude))
final_score <- suppressWarnings(as.numeric(wide_output$score))
final_accepted <- wide_output$status %in% c("M", "T") &
  final_score >= 90 &
  valid_coordinates(final_longitude, final_latitude)
if (
  nrow(wide_output) != nrow(unique_addresses) ||
    !identical(names(wide_output), contract_columns) ||
    anyNA(wide_output[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(wide_output$address_id) ||
    anyDuplicated(wide_output$address_for_geocoding) ||
    any(!wide_output$status %in% c("M", "T", "U")) ||
    any(final_accepted & is.na(final_score)) ||
    any(final_accepted & !valid_coordinates(final_longitude, final_latitude)) ||
    any(!final_accepted & (
      is.finite(final_longitude) | is.finite(final_latitude)
    ))
) {
  stop("The Census cascade failed its wide-contract checks.", call. = FALSE)
}

message("Classifying results against the exact current full-purpose boundary...")
grid <- readRDS(GRID_FILE)
if (!inherits(grid, "sf") || is.na(sf::st_crs(grid)) ||
    !"hex_id" %in% names(grid)) {
  stop("The analysis grid must be an sf object with hex_id and a CRS.", call. = FALSE)
}
jurisdictions <- sf::st_read(JURISDICTIONS_FILE, quiet = TRUE)
assert_columns(
  names(jurisdictions),
  c("city_name", "jurisdiction_type"),
  basename(JURISDICTIONS_FILE)
)
city_name <- toupper(trimws(as.character(jurisdictions$city_name)))
jurisdiction_type <- toupper(trimws(as.character(
  jurisdictions$jurisdiction_type
)))
austin_full <- jurisdictions[
  city_name == "CITY OF AUSTIN" & jurisdiction_type == "FULL",
] |>
  sf::st_make_valid() |>
  sf::st_transform(4326)
if (nrow(austin_full) == 0L) {
  stop("The jurisdiction snapshot contains no Austin FULL polygons.", call. = FALSE)
}
austin_full <- sf::st_union(austin_full)

classify_inside_city <- function(longitude, latitude) {
  result <- rep(FALSE, length(longitude))
  valid <- valid_coordinates(longitude, latitude)
  if (any(valid)) {
    points <- sf::st_as_sf(
      tibble::tibble(
        longitude = longitude[valid],
        latitude = latitude[valid]
      ),
      coords = c("longitude", "latitude"),
      crs = 4326,
      remove = FALSE
    )
    result[valid] <- lengths(sf::st_intersects(points, austin_full)) > 0L
  }
  result
}

assign_hex <- function(longitude, latitude) {
  result <- rep(NA_integer_, length(longitude))
  valid <- valid_coordinates(longitude, latitude)
  if (any(valid)) {
    points <- sf::st_as_sf(
      tibble::tibble(
        row_id = which(valid),
        longitude = longitude[valid],
        latitude = latitude[valid]
      ),
      coords = c("longitude", "latitude"),
      crs = 4326,
      remove = FALSE
    ) |>
      sf::st_transform(sf::st_crs(grid))
    intersections <- sf::st_intersects(points, grid)
    unique_match <- lengths(intersections) == 1L
    if (any(unique_match)) {
      grid_rows <- vapply(
        intersections[unique_match],
        function(index) index[[1L]],
        integer(1)
      )
      result[points$row_id[unique_match]] <- grid$hex_id[grid_rows]
    }
  }
  result
}

local_inside_city <- classify_inside_city(local_longitude, local_latitude)
coa_inside_city <- classify_inside_city(coa_longitude, coa_latitude)
census_inside_city <- classify_inside_city(census$longitude, census$latitude)
final_inside_city <- classify_inside_city(final_longitude, final_latitude)
coa_review_longitude <- suppressWarnings(as.numeric(coa_review$longitude))
coa_review_latitude <- suppressWarnings(as.numeric(coa_review$latitude))
coa_review_score <- suppressWarnings(as.numeric(coa_review$score))
coa_review_inside_city <- classify_inside_city(
  coa_review_longitude,
  coa_review_latitude
)
coa_review_city_label <- !is.na(coa_review$jurisdiction_label) &
  toupper(trimws(coa_review$jurisdiction_label)) == "AUSTIN FULL PURPOSE"
coa_locator_city_candidate <- coa_review_city_label &
  coa_review$status %in% c("M", "T") &
  valid_coordinates(coa_review_longitude, coa_review_latitude) &
  coa_review_inside_city
coa_locator_high_confidence_city_candidate <-
  coa_locator_city_candidate &
    coa_review_score >= 95 &
    as_flag(coa_review$zip_agrees) &
    as_flag(coa_review$house_number_agrees)

prepared_filings <- readr::read_csv(
  PREPARED_FILING_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
assert_columns(
  names(prepared_filings),
  c("source_id", "case_uid", "case_number", "address_for_geocoding"),
  basename(PREPARED_FILING_FILE)
)
if (
  nrow(prepared_filings) == 0L ||
    anyNA(prepared_filings[c("source_id", "case_number")])
) {
  stop(
    "The prepared filing input must contain source and case identifiers.",
    call. = FALSE
  )
}
address_city_linkage <- tibble::tibble(
  address_for_geocoding = unique_addresses$address_for_geocoding,
  coa_locator_current_full_candidate = coa_locator_city_candidate,
  coa_locator_high_confidence_current_full_candidate =
    coa_locator_high_confidence_city_candidate,
  cascade_confirmed_inside_current_austin_full =
    final_accepted & final_inside_city
)
filing_city_linkage <- prepared_filings |>
  dplyr::mutate(
    qa_case_key = dplyr::if_else(
      !is.na(.data$case_uid) & nzchar(.data$case_uid),
      .data$case_uid,
      paste(.data$source_id, .data$case_number, sep = "::")
    )
  ) |>
  dplyr::filter(!is.na(.data$address_for_geocoding)) |>
  dplyr::left_join(
    address_city_linkage,
    by = "address_for_geocoding",
    relationship = "many-to-one"
  )
city_candidate_filing_rows <-
  filing_city_linkage$coa_locator_current_full_candidate %in% TRUE
city_candidate_filing_rows_confirmed_inside <-
  city_candidate_filing_rows &
    filing_city_linkage$cascade_confirmed_inside_current_austin_full %in% TRUE
high_confidence_city_candidate_filing_rows <-
  filing_city_linkage$coa_locator_high_confidence_current_full_candidate %in%
    TRUE
high_confidence_city_candidate_filing_rows_confirmed_inside <-
  high_confidence_city_candidate_filing_rows &
    filing_city_linkage$cascade_confirmed_inside_current_austin_full %in% TRUE
case_city_linkage <- filing_city_linkage |>
  dplyr::group_by(.data$qa_case_key) |>
  dplyr::summarise(
    coa_locator_current_full_candidate = any(
      .data$coa_locator_current_full_candidate %in% TRUE
    ),
    coa_locator_high_confidence_current_full_candidate = any(
      .data$coa_locator_high_confidence_current_full_candidate %in% TRUE
    ),
    cascade_confirmed_inside_current_austin_full = any(
      .data$cascade_confirmed_inside_current_austin_full %in% TRUE
    ),
    .groups = "drop"
  )
city_candidate_cases <-
  case_city_linkage$coa_locator_current_full_candidate %in% TRUE
city_candidate_cases_confirmed_inside <-
  city_candidate_cases &
    case_city_linkage$cascade_confirmed_inside_current_austin_full %in% TRUE
high_confidence_city_candidate_cases <-
  case_city_linkage$coa_locator_high_confidence_current_full_candidate %in%
    TRUE
high_confidence_city_candidate_cases_confirmed_inside <-
  high_confidence_city_candidate_cases &
    case_city_linkage$cascade_confirmed_inside_current_austin_full %in% TRUE
local_hex_id <- assign_hex(local_longitude, local_latitude)
census_hex_id <- assign_hex(census$longitude, census$latitude)
final_hex_id <- assign_hex(final_longitude, final_latitude)

validation_available <- local_accepted & census_accepted
validation_distance_m <- rep(NA_real_, nrow(unique_addresses))
if (any(validation_available)) {
  local_validation_points <- sf::st_as_sf(
    tibble::tibble(
      longitude = local_longitude[validation_available],
      latitude = local_latitude[validation_available]
    ),
    coords = c("longitude", "latitude"),
    crs = 4326
  ) |>
    sf::st_transform(3083)
  census_validation_points <- sf::st_as_sf(
    tibble::tibble(
      longitude = census$longitude[validation_available],
      latitude = census$latitude[validation_available]
    ),
    coords = c("longitude", "latitude"),
    crs = 4326
  ) |>
    sf::st_transform(3083)
  validation_distance_m[validation_available] <- as.numeric(sf::st_distance(
    local_validation_points,
    census_validation_points,
    by_element = TRUE
  ))
}

explicit_austin <- stringr::str_detect(
  toupper(unique_addresses$address_for_geocoding),
  "(?:^|,|[[:space:]])AUSTIN(?:,|[[:space:]])+TX[[:space:]]+[0-9]{5}"
)

review <- tibble::tibble(
  address_id = unique_addresses$address_id,
  address_for_geocoding = unique_addresses$address_for_geocoding,
  local_accepted = local_accepted,
  local_method = local_geocodes$loc_name,
  local_longitude = local_longitude,
  local_latitude = local_latitude,
  local_inside_current_austin_full = local_inside_city,
  coa_accepted = coa_accepted,
  coa_method = coa_geocodes$loc_name,
  coa_longitude = coa_longitude,
  coa_latitude = coa_latitude,
  coa_inside_current_austin_full = coa_inside_city,
  census_match_indicator = census$match_indicator,
  census_match_type = census$match_type,
  census_matched_address = census$matched_address,
  census_longitude = census$longitude,
  census_latitude = census$latitude,
  census_zip_agrees = census$zip_agrees,
  census_house_number_agrees = census$house_number_agrees,
  census_passes_quality_gate = census_accepted,
  census_inside_current_austin_full = census_inside_city,
  selected_provider = dplyr::case_when(
    local_accepted ~ "williamson_public_address_points_local",
    coa_selected ~ "city_of_austin_public_coa_locator",
    census_selected ~ "census_public_ar_current",
    TRUE ~ NA_character_
  ),
  selected_status = wide_output$status,
  selected_score = final_score,
  selected_longitude = final_longitude,
  selected_latitude = final_latitude,
  selected_hex_id = final_hex_id,
  selected_inside_current_austin_full = final_inside_city,
  local_census_same_hex = dplyr::if_else(
    validation_available & !is.na(local_hex_id) & !is.na(census_hex_id),
    local_hex_id == census_hex_id,
    NA
  ),
  local_census_distance_m = validation_distance_m
)

metric_row <- function(metric, value) {
  tibble::tibble(metric = metric, value = as.character(value))
}

safe_rate <- function(numerator, denominator) {
  if (denominator == 0L) NA_real_ else numerator / denominator
}

safe_quantile <- function(value, probability) {
  value <- value[is.finite(value)]
  if (length(value) == 0L) {
    NA_real_
  } else {
    as.numeric(stats::quantile(value, probability, names = FALSE))
  }
}

qa <- dplyr::bind_rows(
  metric_row("generated_at_utc", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  metric_row("cache_schema_version", CACHE_SCHEMA_VERSION),
  metric_row("census_batch_url", CENSUS_BATCH_URL),
  metric_row("census_benchmark", CENSUS_BENCHMARK),
  metric_row("accepted_census_match_types", paste(ACCEPTED_MATCH_TYPES, collapse = "|")),
  metric_row("candidate_input_sha256", hash_file(UNIQUE_ADDRESS_FILE)),
  metric_row("prepared_filing_input_sha256", hash_file(PREPARED_FILING_FILE)),
  metric_row("local_geocode_input_sha256", hash_file(LOCAL_GEOCODE_FILE)),
  metric_row("coa_geocode_input_sha256", hash_file(COA_GEOCODE_FILE)),
  metric_row("coa_review_input_sha256", hash_file(COA_REVIEW_FILE)),
  metric_row("analysis_grid_sha256", hash_file(GRID_FILE)),
  metric_row("jurisdiction_snapshot_sha256", hash_file(JURISDICTIONS_FILE)),
  metric_row("census_response_bundle_sha256", hash_object(batch_response_hashes)),
  metric_row("network_enabled", NETWORK_ENABLED),
  metric_row("force_refresh", FORCE_REFRESH),
  metric_row("batch_size", BATCH_SIZE),
  metric_row("batches", length(request_batches)),
  metric_row("batches_downloaded", downloaded_batches),
  metric_row("batches_reused", reused_batches),
  metric_row("candidate_addresses", nrow(unique_addresses)),
  metric_row("candidate_addresses_explicit_austin", sum(explicit_austin)),
  metric_row("local_accepted_addresses", sum(local_accepted)),
  metric_row("local_accepted_inside_current_austin_full", sum(local_accepted & local_inside_city)),
  metric_row("coa_quality_gate_passes", sum(coa_accepted)),
  metric_row("coa_fallback_additions", sum(coa_selected)),
  metric_row("coa_fallback_additions_inside_current_austin_full", sum(coa_selected & coa_inside_city)),
  metric_row("census_match_exact", sum(census$match_indicator == "Match" & census$match_type == "Exact", na.rm = TRUE)),
  metric_row("census_match_non_exact", sum(census$match_indicator == "Match" & census$match_type == "Non_Exact", na.rm = TRUE)),
  metric_row("census_ties", sum(census$match_indicator == "Tie", na.rm = TRUE)),
  metric_row("census_no_match", sum(census$match_indicator == "No_Match", na.rm = TRUE)),
  metric_row("census_quality_gate_passes", sum(census_accepted)),
  metric_row("census_quality_gate_inside_current_austin_full", sum(census_accepted & census_inside_city)),
  metric_row("census_fallback_additions", sum(census_selected)),
  metric_row("census_fallback_additions_inside_current_austin_full", sum(census_selected & census_inside_city)),
  metric_row("cascade_accepted_addresses", sum(final_accepted)),
  metric_row("cascade_unresolved_addresses", sum(!final_accepted)),
  metric_row("cascade_match_rate", format(safe_rate(sum(final_accepted), nrow(unique_addresses)), digits = 10, trim = TRUE)),
  metric_row("cascade_accepted_inside_current_austin_full", sum(final_accepted & final_inside_city)),
  metric_row("explicit_austin_cascade_accepted", sum(explicit_austin & final_accepted)),
  metric_row("explicit_austin_cascade_match_rate", format(safe_rate(sum(explicit_austin & final_accepted), sum(explicit_austin)), digits = 10, trim = TRUE)),
  metric_row("explicit_austin_confirmed_inside_current_austin_full", sum(explicit_austin & final_accepted & final_inside_city)),
  metric_row("explicit_austin_confirmed_city_rate", format(safe_rate(sum(explicit_austin & final_accepted & final_inside_city), sum(explicit_austin)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_addresses", sum(coa_locator_city_candidate)),
  metric_row("coa_locator_current_full_candidates_cascade_accepted", sum(coa_locator_city_candidate & final_accepted)),
  metric_row("coa_locator_current_full_candidates_cascade_confirmed_inside", sum(coa_locator_city_candidate & final_accepted & final_inside_city)),
  metric_row("coa_locator_current_full_candidate_linkage_rate", format(safe_rate(sum(coa_locator_city_candidate & final_accepted & final_inside_city), sum(coa_locator_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_filing_rows", sum(city_candidate_filing_rows)),
  metric_row("coa_locator_current_full_candidate_filing_rows_cascade_confirmed_inside", sum(city_candidate_filing_rows_confirmed_inside)),
  metric_row("coa_locator_current_full_candidate_filing_row_linkage_rate", format(safe_rate(sum(city_candidate_filing_rows_confirmed_inside), sum(city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_cases", sum(city_candidate_cases)),
  metric_row("coa_locator_current_full_candidate_cases_cascade_confirmed_inside", sum(city_candidate_cases_confirmed_inside)),
  metric_row("coa_locator_current_full_candidate_case_linkage_rate", format(safe_rate(sum(city_candidate_cases_confirmed_inside), sum(city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidates", sum(coa_locator_high_confidence_city_candidate)),
  metric_row("coa_locator_high_confidence_current_full_candidates_cascade_confirmed_inside", sum(coa_locator_high_confidence_city_candidate & final_accepted & final_inside_city)),
  metric_row("coa_locator_high_confidence_current_full_candidate_linkage_rate", format(safe_rate(sum(coa_locator_high_confidence_city_candidate & final_accepted & final_inside_city), sum(coa_locator_high_confidence_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_rows", sum(high_confidence_city_candidate_filing_rows)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_rows_cascade_confirmed_inside", sum(high_confidence_city_candidate_filing_rows_confirmed_inside)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_row_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_filing_rows_confirmed_inside), sum(high_confidence_city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_cases", sum(high_confidence_city_candidate_cases)),
  metric_row("coa_locator_high_confidence_current_full_candidate_cases_cascade_confirmed_inside", sum(high_confidence_city_candidate_cases_confirmed_inside)),
  metric_row("coa_locator_high_confidence_current_full_candidate_case_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_cases_confirmed_inside), sum(high_confidence_city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("local_validation_census_quality_matches", sum(validation_available)),
  metric_row("local_validation_same_hex", sum(validation_available & !is.na(review$local_census_same_hex) & review$local_census_same_hex)),
  metric_row("local_validation_comparable_hexes", sum(validation_available & !is.na(review$local_census_same_hex))),
  metric_row("local_validation_same_hex_rate", format(safe_rate(sum(validation_available & !is.na(review$local_census_same_hex) & review$local_census_same_hex), sum(validation_available & !is.na(review$local_census_same_hex))), digits = 10, trim = TRUE)),
  metric_row("local_validation_distance_median", format(safe_quantile(validation_distance_m[validation_available], 0.5), digits = 10, trim = TRUE)),
  metric_row("local_validation_distance_p90", format(safe_quantile(validation_distance_m[validation_available], 0.9), digits = 10, trim = TRUE))
)

for (match_type in c("Exact", "Non_Exact")) {
  subset <- validation_available & census$match_type == match_type
  comparable <- subset & !is.na(review$local_census_same_hex)
  prefix <- paste0("local_validation_", tolower(match_type))
  qa <- dplyr::bind_rows(
    qa,
    metric_row(paste0(prefix, "_addresses"), sum(subset)),
    metric_row(paste0(prefix, "_comparable_hexes"), sum(comparable)),
    metric_row(paste0(prefix, "_same_hex"), sum(comparable & review$local_census_same_hex)),
    metric_row(
      paste0(prefix, "_same_hex_rate"),
      format(
        safe_rate(
          sum(comparable & review$local_census_same_hex),
          sum(comparable)
        ),
        digits = 10,
        trim = TRUE
      )
    ),
    metric_row(
      paste0(prefix, "_distance_median"),
      format(
        safe_quantile(validation_distance_m[subset], 0.5),
        digits = 10,
        trim = TRUE
      )
    ),
    metric_row(
      paste0(prefix, "_distance_p90"),
      format(
        safe_quantile(validation_distance_m[subset], 0.9),
        digits = 10,
        trim = TRUE
      )
    )
  )
}

atomic_write_csv(wide_output, OUTPUT_FILE)
atomic_write_csv(review, REVIEW_FILE)
atomic_write_csv(qa, QA_FILE)

message(
  "Wrote ",
  nrow(wide_output),
  " cascade records: ",
  sum(local_accepted),
  " local, ",
  sum(coa_selected),
  " COA, ",
  sum(census_selected),
  " Census fallback, and ",
  sum(!final_accepted),
  " unresolved."
)
