################################################################################
# Add a Cached ArcGIS World Refinement for Williamson Eviction Addresses
################################################################################
#
# This final stage follows the local -> COA -> Census cascade. Its network cache
# fill is explicit and authenticated. It sends only City-relevant records that
# still need a precise point: unresolved addresses placed in Austin full-purpose
# limits by the City locator, unresolved explicit-Austin addresses for which
# that locator returned no jurisdiction, and lower-precision Census matches
# placed in Austin full-purpose limits by the City locator. It applies a
# conservative point-address quality gate and writes a separate augmented
# registry consumed by the Part 3 eviction panel. It never overwrites the
# local/COA/Census registry.
#
# Network access is off by default. With an authorized ArcGIS account or key,
# explicitly fill missing caches with:
#
#   WILLIAMSON_EVICTION_ARCGIS_NETWORK=true \
#     Rscript scripts/data/williamson_evictions_geocode_arcgis.R
#
# Authentication reuses the conventions in evictions_prepare.R:
#   - ARCGIS_API_KEY
#   - ARCGIS_CLIENT and ARCGIS_SECRET
#   - ARCGIS_USER and ARCGIS_PASSWORD
#   - ARCGIS_CLIENT for interactive OAuth via auth_code()
#
# Set WILLIAMSON_EVICTION_ARCGIS_AUTH_METHOD to auto, key, client, user, code,
# or existing (to reuse a token already installed in the current R session).
# Batch geocoding is requested for storage because the results are written to
# disk. Depending on the credential and account, this operation can consume
# ArcGIS Online credits or ArcGIS Location Platform pay-as-you-go usage.
#
# Privacy and resumability safeguards:
#   - The provider receives an address plus a batch-local integer ResultID only;
#     case IDs, defendant names, and local address IDs are never submitted.
#   - Each response is cached by a SHA-256 hash of the normalized address and a
#     hash of the provider request profile. Unchanged addresses are not resent.
#   - Address-bearing caches, review data, and registries stay under output/.
#   - The QA file records aggregate counts and content hashes without
#     exposing addresses.
#
# Before enabling the network flag, confirm that transmitting residential
# addresses is consistent with applicable institutional policy and Esri terms.
################################################################################

required_packages <- c(
  "arcgisgeocode", "arcgisutils", "digest", "dplyr", "here", "readr",
  "sf", "stringr", "tibble"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before ArcGIS geocoding: ",
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

CASCADE_INPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_CASCADE_INPUT_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded.csv")
)
PREPARED_FILING_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_PREPARED_FILING_FILE",
  here::here(
    "output",
    "williamson_eviction_filings_prepared_for_geocoding.csv"
  )
)
COA_REVIEW_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_COA_REVIEW_FILE",
  here::here("output", "williamson_eviction_geocode_coa_review.csv")
)
GRID_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_GRID_FILE",
  here::here("output", "hex_grid.rds")
)
JURISDICTIONS_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_JURISDICTIONS_FILE",
  here::here("data", "BOUNDARIES_jurisdictions_20260429.geojson")
)
PROVIDER_OUTPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_PROVIDER_OUTPUT_FILE",
  here::here(
    "output",
    "williamson_eviction_addresses_geocoded_arcgis.csv"
  )
)
CASCADE_OUTPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_CASCADE_OUTPUT_FILE",
  here::here(
    "output",
    "williamson_eviction_addresses_geocoded_with_arcgis.csv"
  )
)
QA_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_QA_FILE",
  here::here("output", "williamson_eviction_geocode_arcgis_qa.csv")
)
REVIEW_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_REVIEW_FILE",
  here::here("output", "williamson_eviction_geocode_arcgis_review.csv")
)
CACHE_ROOT <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_CACHE_DIR",
  here::here("output", "williamson_eviction_arcgis_geocode_cache")
)

ARCGIS_WORLD_URL <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_WORLD_URL",
  paste0(
    "https://geocode.arcgis.com/arcgis/rest/services/World/",
    "GeocodeServer"
  )
)
AUTH_METHOD <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_AUTH_METHOD",
  Sys.getenv("ARCGIS_AUTH_METHOD", "auto")
))
SOURCE_COUNTRY <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_SOURCE_COUNTRY",
  "USA"
)
CATEGORY <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_CATEGORY",
  "Point Address,Subaddress"
)
LOCATION_TYPE <- Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_LOCATION_TYPE",
  "rooftop"
)
ACCEPTED_ADDRESS_TYPES <- trimws(strsplit(
  Sys.getenv(
    "WILLIAMSON_EVICTION_ARCGIS_ACCEPT_ADDRESS_TYPES",
    "PointAddress|Subaddress"
  ),
  "|",
  fixed = TRUE
)[[1]])
ACCEPTED_ADDRESS_TYPES <- ACCEPTED_ADDRESS_TYPES[
  ACCEPTED_ADDRESS_TYPES != ""
]
MINIMUM_SCORE <- suppressWarnings(as.numeric(Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_MIN_SCORE",
  "95"
)))
BATCH_SIZE <- suppressWarnings(as.integer(Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_BATCH_SIZE",
  "150"
)))
NETWORK_ENABLED <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_NETWORK",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
FORCE_REFRESH <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_ARCGIS_FORCE_REFRESH",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
CACHE_SCHEMA_VERSION <- 1L
# arcgisutils 0.4.x accepts a double WKID but currently rejects an integer
# scalar while formatting the validation error, so keep this as numeric.
OUTPUT_CRS <- 4326
FOR_STORAGE <- TRUE
MATCH_OUT_OF_RANGE <- FALSE

if (!AUTH_METHOD %in% c(
  "auto", "key", "client", "user", "code", "existing"
)) {
  stop(
    "WILLIAMSON_EVICTION_ARCGIS_AUTH_METHOD must be auto, key, client, ",
    "user, code, or existing.",
    call. = FALSE
  )
}
if (!identical(toupper(SOURCE_COUNTRY), "USA")) {
  stop(
    "The Williamson fallback is intentionally restricted to source country USA.",
    call. = FALSE
  )
}
if (length(ACCEPTED_ADDRESS_TYPES) == 0L) {
  stop("At least one ArcGIS address type must be accepted.", call. = FALSE)
}
if (is.na(MINIMUM_SCORE) || MINIMUM_SCORE < 0 || MINIMUM_SCORE > 100) {
  stop(
    "WILLIAMSON_EVICTION_ARCGIS_MIN_SCORE must be between 0 and 100.",
    call. = FALSE
  )
}
if (is.na(BATCH_SIZE) || BATCH_SIZE < 1L || BATCH_SIZE > 1000L) {
  stop(
    "WILLIAMSON_EVICTION_ARCGIS_BATCH_SIZE must be between 1 and 1000.",
    call. = FALSE
  )
}
if (FORCE_REFRESH && !NETWORK_ENABLED) {
  stop(
    "Force refresh requires WILLIAMSON_EVICTION_ARCGIS_NETWORK=true.",
    call. = FALSE
  )
}

required_files <- c(
  CASCADE_INPUT_FILE,
  PREPARED_FILING_FILE,
  COA_REVIEW_FILE,
  GRID_FILE,
  JURISDICTIONS_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing ArcGIS-fallback input(s): ",
    paste(basename(missing_files), collapse = ", "),
    call. = FALSE
  )
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
    stop("Could not atomically replace an ArcGIS cache file.", call. = FALSE)
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
    stop("Could not atomically replace an ArcGIS output.", call. = FALSE)
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

as_flag <- function(value) {
  tolower(trimws(as.character(value))) %in% c("true", "t", "1", "yes", "y")
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
  value <- gsub('^["“”]+|["“”]+$', "", value, perl = TRUE)
  value <- gsub("[\r\n\t]+", " ", value, perl = TRUE)
  value <- trimws(gsub("[[:space:]]+", " ", value))
  toupper(value)
}

new_opaque_id <- function() {
  seed <- list(
    time = as.numeric(Sys.time()),
    process = Sys.getpid(),
    random = stats::runif(4L)
  )
  paste0("arcgis_", substr(hash_object(seed), 1L, 24L))
}

get_arcgis_token <- function(auth_method = "auto") {
  auth_method <- match.arg(
    auth_method,
    c("auto", "key", "client", "user", "code", "existing")
  )
  if (auth_method == "auto") {
    if (nzchar(Sys.getenv("ARCGIS_API_KEY"))) {
      auth_method <- "key"
    } else if (
      nzchar(Sys.getenv("ARCGIS_CLIENT")) &&
        nzchar(Sys.getenv("ARCGIS_SECRET"))
    ) {
      auth_method <- "client"
    } else if (
      nzchar(Sys.getenv("ARCGIS_USER")) &&
        nzchar(Sys.getenv("ARCGIS_PASSWORD"))
    ) {
      auth_method <- "user"
    } else if (nzchar(Sys.getenv("ARCGIS_CLIENT")) && interactive()) {
      auth_method <- "code"
    } else {
      stop(
        "No ArcGIS credentials found. Set ARCGIS_API_KEY, ",
        "ARCGIS_CLIENT/ARCGIS_SECRET, ARCGIS_USER/ARCGIS_PASSWORD, or ",
        "ARCGIS_CLIENT for interactive OAuth.",
        call. = FALSE
      )
    }
  }

  message("Authenticating with ArcGIS using method: ", auth_method)
  token <- switch(
    auth_method,
    key = arcgisutils::auth_key(),
    client = arcgisutils::auth_client(),
    user = arcgisutils::auth_user(),
    code = arcgisutils::auth_code(),
    existing = arcgisutils::arc_token()
  )
  if (
    !inherits(token, "httr2_token") ||
      is.null(token$access_token) ||
      !nzchar(token$access_token) ||
      is.null(token$expires_at) ||
      !is.finite(as.numeric(token$expires_at)) ||
      as.numeric(token$expires_at) <= as.numeric(Sys.time()) + 30
  ) {
    stop(
      "No unexpired ArcGIS token is installed in the current R session.",
      call. = FALSE
    )
  }
  arcgisutils::set_arc_token(token)
  list(token = token, method = auth_method)
}

as_geocode_sf <- function(value) {
  if (inherits(value, "sf")) return(value)
  if ("geometry" %in% names(value) && inherits(value$geometry, "sfc")) {
    return(sf::st_as_sf(value, sf_column_name = "geometry"))
  }
  stop("ArcGIS returned no sf geometry column.", call. = FALSE)
}

geocode_addresses_for_storage <- function(...) {
  # Esri classifies /geocodeAddresses as a stored-geocode operation. In
  # arcgisgeocode 0.4.0, for_storage = TRUE correctly performs the token and
  # storage-intent check, but the resulting informational notice mistakenly
  # says the flag is FALSE. Suppress only that inverted notice while retaining
  # the package's authenticated stored-result check.
  previous_notice_frequency <- getOption("arcgisgeocode.storage")
  on.exit(
    if (is.null(previous_notice_frequency)) {
      options(arcgisgeocode.storage = NULL)
    } else {
      options(arcgisgeocode.storage = previous_notice_frequency)
    },
    add = TRUE
  )
  options(arcgisgeocode.storage = "never")
  arcgisgeocode::geocode_addresses(..., for_storage = TRUE)
}

extract_geometry_coordinates <- function(value) {
  value <- as_geocode_sf(value)
  result <- matrix(
    NA_real_,
    nrow = nrow(value),
    ncol = 2L,
    dimnames = list(NULL, c("longitude", "latitude"))
  )
  for (row in seq_len(nrow(value))) {
    geometry <- sf::st_geometry(value)[[row]]
    if (is.null(geometry) || sf::st_is_empty(geometry)) next
    coordinates <- sf::st_coordinates(geometry)
    if (nrow(coordinates) > 0L) {
      result[row, ] <- coordinates[1L, c("X", "Y")]
    }
  }
  result
}

normalize_provider_response <- function(value, expected_count) {
  value <- as_geocode_sf(value)
  assert_columns(
    names(value),
    c(
      "result_id", "loc_name", "status", "score", "match_addr",
      "addr_type", "add_num", "postal", "region_abbr"
    ),
    "ArcGIS World response"
  )
  provider_ids <- suppressWarnings(as.integer(value$result_id))
  if (
    nrow(value) != expected_count ||
      anyNA(provider_ids) ||
      !identical(provider_ids, seq_len(expected_count))
  ) {
    stop(
      "ArcGIS did not return one ordered batch-local ResultID per address.",
      call. = FALSE
    )
  }
  coordinates <- extract_geometry_coordinates(value)
  value |>
    sf::st_drop_geometry() |>
    tibble::as_tibble() |>
    dplyr::mutate(
      longitude = coordinates[, "longitude"],
      latitude = coordinates[, "latitude"]
    )
}

cache_path <- function(address_sha256, request_profile_sha256) {
  file.path(
    CACHE_ROOT,
    substr(request_profile_sha256, 1L, 16L),
    substr(address_sha256, 1L, 2L),
    paste0(address_sha256, ".rds")
  )
}

validate_cache <- function(
    payload,
    address_sha256,
    request_profile_sha256
) {
  if (
    !is.list(payload) ||
      !identical(payload$cache_schema_version, CACHE_SCHEMA_VERSION) ||
      !identical(payload$address_sha256, address_sha256) ||
      !identical(
        payload$request_profile_sha256,
        request_profile_sha256
      ) ||
      !is.character(payload$opaque_request_id) ||
      length(payload$opaque_request_id) != 1L ||
      !grepl("^arcgis_[0-9a-f]{24}$", payload$opaque_request_id) ||
      !is.data.frame(payload$response) ||
      nrow(payload$response) != 1L ||
      !identical(hash_object(payload$response), payload$response_sha256)
  ) {
    return(NULL)
  }
  required <- c(
    "result_id", "loc_name", "status", "score", "match_addr",
    "addr_type", "add_num", "postal", "region_abbr", "longitude",
    "latitude"
  )
  if (!all(required %in% names(payload$response))) return(NULL)
  payload
}

message("Reading the frozen Williamson geocoding cascade...")
cascade <- readr::read_csv(
  CASCADE_INPUT_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
contract_columns <- names(cascade)
required_contract_columns <- c(
  "address_id", "address_for_geocoding", "result_id", "loc_name",
  "status", "score", "match_addr", "addr_type", "longitude", "latitude"
)
assert_columns(
  contract_columns,
  required_contract_columns,
  basename(CASCADE_INPUT_FILE)
)
if (
  nrow(cascade) == 0L ||
    anyNA(cascade[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(cascade$address_id) ||
    anyDuplicated(cascade$address_for_geocoding)
) {
  stop(
    "The input cascade must contain one row per unique Williamson address.",
    call. = FALSE
  )
}
coa_review <- readr::read_csv(
  COA_REVIEW_FILE,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
assert_columns(
  names(coa_review),
  c("address_id", "jurisdiction_label"),
  basename(COA_REVIEW_FILE)
)
if (
  nrow(coa_review) != nrow(cascade) ||
    anyNA(coa_review$address_id) ||
    anyDuplicated(coa_review$address_id) ||
    !setequal(coa_review$address_id, cascade$address_id)
) {
  stop(
    "The COA review registry must contain every cascade address ID once.",
    call. = FALSE
  )
}
coa_scope <- cascade |>
  dplyr::select("address_id") |>
  dplyr::left_join(
    coa_review |>
      dplyr::select("address_id", "jurisdiction_label"),
    by = "address_id",
    relationship = "one-to-one"
  )

prior_longitude <- suppressWarnings(as.numeric(cascade$longitude))
prior_latitude <- suppressWarnings(as.numeric(cascade$latitude))
prior_score <- suppressWarnings(as.numeric(cascade$score))
prior_accepted <- cascade$status %in% c("M", "T") &
  prior_score >= 90 &
  valid_coordinates(prior_longitude, prior_latitude)
prior_unresolved <- !prior_accepted
prior_census <- stringr::str_detect(
  dplyr::coalesce(cascade$loc_name, ""),
  "^census_public_ar_current_"
)
explicit_austin <- stringr::str_detect(
  toupper(cascade$address_for_geocoding),
  "(?:^|,|[[:space:]])AUSTIN(?:,|[[:space:]])+TX[[:space:]]+[0-9]{5}"
)
coa_jurisdiction <- toupper(trimws(as.character(coa_scope$jurisdiction_label)))
coa_jurisdiction[coa_jurisdiction == ""] <- NA_character_
coa_places_in_austin_full <- !is.na(coa_jurisdiction) &
  coa_jurisdiction == "AUSTIN FULL PURPOSE"
arcgis_target_reason <- dplyr::case_when(
  coa_places_in_austin_full & prior_unresolved ~
    "coa_places_in_city_prior_unresolved",
  coa_places_in_austin_full & prior_census ~
    "coa_places_in_city_replace_census",
  prior_unresolved & explicit_austin & is.na(coa_jurisdiction) ~
    "explicit_austin_prior_unresolved_coa_jurisdiction_missing",
  TRUE ~ NA_character_
)
arcgis_target <- !is.na(arcgis_target_reason)

residual_rows <- cascade[arcgis_target, , drop = FALSE] |>
  dplyr::transmute(
    address_id = .data$address_id,
    address_for_geocoding = .data$address_for_geocoding,
    prior_loc_name = .data$loc_name,
    prior_status = .data$status,
    prior_score = suppressWarnings(as.numeric(.data$score)),
    arcgis_target_reason = arcgis_target_reason[arcgis_target],
    coa_jurisdiction_label = coa_jurisdiction[arcgis_target],
    normalized_address = clean_external_address(
      .data$address_for_geocoding
    )
  ) |>
  dplyr::mutate(
    requestable = !is.na(.data$normalized_address) &
      nchar(.data$normalized_address) > 0L &
      nchar(.data$normalized_address) <= 200L,
    address_sha256 = dplyr::if_else(
      .data$requestable,
      vapply(.data$normalized_address, hash_text, character(1)),
      NA_character_
    )
  )

request_unique <- residual_rows |>
  dplyr::filter(.data$requestable) |>
  dplyr::distinct(.data$address_sha256, .keep_all = TRUE) |>
  dplyr::arrange(.data$address_sha256) |>
  dplyr::select("address_sha256", "normalized_address")

request_profile_sha256 <- hash_object(list(
  cache_schema_version = CACHE_SCHEMA_VERSION,
  service_url = ARCGIS_WORLD_URL,
  source_country = SOURCE_COUNTRY,
  category = CATEGORY,
  output_crs = OUTPUT_CRS,
  for_storage = FOR_STORAGE,
  match_out_of_range = MATCH_OUT_OF_RANGE,
  location_type = LOCATION_TYPE,
  output_fields = "*"
))

dir.create(CACHE_ROOT, recursive = TRUE, showWarnings = FALSE)
cache_payloads <- stats::setNames(
  vector("list", nrow(request_unique)),
  request_unique$address_sha256
)
cache_reused <- 0L
cache_downloaded <- 0L

if (nrow(request_unique) > 0L) {
  for (index in seq_len(nrow(request_unique))) {
    address_sha256 <- request_unique$address_sha256[[index]]
    path <- cache_path(address_sha256, request_profile_sha256)
    payload <- NULL
    if (file.exists(path) && !FORCE_REFRESH) {
      payload <- tryCatch(readRDS(path), error = function(error) NULL)
      payload <- validate_cache(
        payload,
        address_sha256,
        request_profile_sha256
      )
    }
    if (!is.null(payload)) {
      cache_payloads[[address_sha256]] <- payload
      cache_reused <- cache_reused + 1L
    }
  }
}

missing_hashes <- names(cache_payloads)[vapply(
  cache_payloads,
  is.null,
  logical(1)
)]
if (length(missing_hashes) > 0L && !NETWORK_ENABLED) {
  stop(
    length(missing_hashes),
    " ArcGIS address cache file(s) are missing or invalid. No network ",
    "request was made. After confirming authorization and account usage, ",
    "rerun with WILLIAMSON_EVICTION_ARCGIS_NETWORK=true.",
    call. = FALSE
  )
}

authentication_method_used <- NA_character_
if (length(missing_hashes) > 0L) {
  authentication <- get_arcgis_token(AUTH_METHOD)
  authentication_method_used <- authentication$method
  world_geocoder <- arcgisgeocode::geocode_server(
    ARCGIS_WORLD_URL,
    token = authentication$token
  )
  missing_requests <- request_unique |>
    dplyr::filter(.data$address_sha256 %in% missing_hashes) |>
    dplyr::arrange(.data$address_sha256)
  request_batches <- split(
    missing_requests,
    ceiling(seq_len(nrow(missing_requests)) / BATCH_SIZE)
  )

  message(
    "Requesting ",
    nrow(missing_requests),
    " unique City-relevant address(es) in ",
    length(request_batches),
    " ArcGIS batch(es)..."
  )
  for (batch_index in seq_along(request_batches)) {
    batch <- request_batches[[batch_index]]
    # arcgisgeocode creates ResultID = 1..n inside each call. Those integers
    # are deliberately unrelated to local address or court identifiers.
    geocoded <- geocode_addresses_for_storage(
      single_line = batch$normalized_address,
      category = CATEGORY,
      crs = OUTPUT_CRS,
      match_out_of_range = MATCH_OUT_OF_RANGE,
      location_type = LOCATION_TYPE,
      source_country = SOURCE_COUNTRY,
      batch_size = nrow(batch),
      geocoder = world_geocoder,
      token = authentication$token,
      .progress = TRUE
    )
    response <- normalize_provider_response(geocoded, nrow(batch))

    opaque_ids <- vapply(seq_len(nrow(batch)), function(unused) {
      new_opaque_id()
    }, character(1))
    if (anyDuplicated(opaque_ids)) {
      stop("Could not generate unique local opaque request IDs.", call. = FALSE)
    }

    for (row in seq_len(nrow(batch))) {
      address_sha256 <- batch$address_sha256[[row]]
      response_row <- response[row, , drop = FALSE]
      payload <- list(
        cache_schema_version = CACHE_SCHEMA_VERSION,
        address_sha256 = address_sha256,
        request_profile_sha256 = request_profile_sha256,
        opaque_request_id = opaque_ids[[row]],
        provider_result_id = as.character(response_row$result_id[[1L]]),
        retrieved_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
        response_sha256 = hash_object(response_row),
        response = response_row
      )
      atomic_save_rds(
        payload,
        cache_path(address_sha256, request_profile_sha256)
      )
      cache_payloads[[address_sha256]] <- payload
      cache_downloaded <- cache_downloaded + 1L
    }
  }
}

empty_provider <- tibble::tibble(
  address_sha256 = character(),
  opaque_request_id = character(),
  result_id = character(),
  loc_name = character(),
  status = character(),
  score = numeric(),
  match_addr = character(),
  addr_type = character(),
  add_num = character(),
  postal = character(),
  region_abbr = character(),
  longitude = numeric(),
  latitude = numeric()
)
if (length(cache_payloads) == 0L) {
  provider_unique <- empty_provider
} else {
  provider_unique <- dplyr::bind_rows(lapply(
    sort(names(cache_payloads)),
    function(address_sha256) {
      payload <- cache_payloads[[address_sha256]]
      response <- tibble::as_tibble(payload$response)
      response$address_sha256 <- address_sha256
      response$opaque_request_id <- payload$opaque_request_id
      response |>
        dplyr::relocate("address_sha256", "opaque_request_id")
    }
  ))
}

provider <- residual_rows |>
  dplyr::left_join(
    provider_unique,
    by = "address_sha256",
    relationship = "many-to-one"
  ) |>
  dplyr::mutate(
    input_zip = extract_input_zip(.data$normalized_address),
    input_house_number = extract_leading_house_number(
      .data$normalized_address
    ),
    output_zip = stringr::str_extract(as.character(.data$postal), "[0-9]{5}"),
    output_house_number = toupper(trimws(as.character(.data$add_num))),
    zip_agrees = !is.na(.data$input_zip) &
      .data$input_zip == .data$output_zip,
    house_number_agrees = !is.na(.data$input_house_number) &
      .data$input_house_number == .data$output_house_number,
    texas_agrees = toupper(trimws(as.character(.data$region_abbr))) == "TX",
    valid_coordinate = valid_coordinates(
      suppressWarnings(as.numeric(.data$longitude)),
      suppressWarnings(as.numeric(.data$latitude))
    ),
    passes_quality_gate = .data$requestable &
      .data$status == "M" &
      suppressWarnings(as.numeric(.data$score)) >= MINIMUM_SCORE &
      .data$addr_type %in% ACCEPTED_ADDRESS_TYPES &
      .data$zip_agrees &
      .data$house_number_agrees &
      .data$texas_agrees &
      .data$valid_coordinate
  )
provider_accepted <- provider$passes_quality_gate %in% TRUE

provider_output <- tibble::as_tibble(stats::setNames(
  rep(list(rep(NA_character_, nrow(cascade))), length(contract_columns)),
  contract_columns
))
provider_output$address_id <- cascade$address_id
provider_output$address_for_geocoding <- cascade$address_for_geocoding
provider_output$loc_name <- dplyr::case_when(
  arcgis_target ~ "arcgis_world_city_target_unresolved",
  prior_accepted ~ "arcgis_world_not_requested_prior_cascade_match",
  TRUE ~ "arcgis_world_not_requested_outside_city_scope"
)
provider_output$status <- "U"
provider_output$score <- "0"

provider_indices <- match(provider$address_id, provider_output$address_id)
provider_output$result_id[provider_indices] <- provider$opaque_request_id
provider_output$loc_name[provider_indices[!provider$requestable]] <-
  "arcgis_world_not_requestable"
selected_provider_indices <- provider_indices[provider_accepted]
provider_output$loc_name[selected_provider_indices] <-
  "arcgis_world_point_fallback"
provider_output$status[selected_provider_indices] <- "M"
provider_output$score[selected_provider_indices] <- as.character(
  provider$score[provider_accepted]
)

for (contract_column in setdiff(
  contract_columns,
  c(
    "address_id", "address_for_geocoding", "result_id", "loc_name",
    "status", "score", "longitude", "latitude"
  )
)) {
  if (contract_column %in% names(provider)) {
    provider_output[[contract_column]][selected_provider_indices] <-
      as.character(provider[[contract_column]][provider_accepted])
  }
}
provider_output$longitude[selected_provider_indices] <- as.character(
  provider$longitude[provider_accepted]
)
provider_output$latitude[selected_provider_indices] <- as.character(
  provider$latitude[provider_accepted]
)

provider_output_longitude <- suppressWarnings(as.numeric(
  provider_output$longitude
))
provider_output_latitude <- suppressWarnings(as.numeric(
  provider_output$latitude
))
if (
  nrow(provider_output) != nrow(cascade) ||
    !identical(names(provider_output), contract_columns) ||
    anyNA(provider_output[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(provider_output$address_id) ||
    any(!provider_output$status %in% c("M", "U")) ||
    any(provider_output$status == "M" & !valid_coordinates(
      provider_output_longitude,
      provider_output_latitude
    )) ||
    any(provider_output$status == "U" & (
      is.finite(provider_output_longitude) |
        is.finite(provider_output_latitude)
    ))
) {
  stop("The ArcGIS provider registry failed its contract checks.", call. = FALSE)
}

wide_output <- cascade
for (contract_column in setdiff(
  contract_columns,
  c("address_id", "address_for_geocoding")
)) {
  wide_output[[contract_column]][selected_provider_indices] <-
    provider_output[[contract_column]][selected_provider_indices]
}

final_longitude <- suppressWarnings(as.numeric(wide_output$longitude))
final_latitude <- suppressWarnings(as.numeric(wide_output$latitude))
final_score <- suppressWarnings(as.numeric(wide_output$score))
final_accepted <- wide_output$status %in% c("M", "T") &
  final_score >= 90 &
  valid_coordinates(final_longitude, final_latitude)
if (
  nrow(wide_output) != nrow(cascade) ||
    !identical(names(wide_output), contract_columns) ||
    anyNA(wide_output[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(wide_output$address_id) ||
    anyDuplicated(wide_output$address_for_geocoding) ||
    any(!wide_output$status %in% c("M", "T", "U")) ||
    any(final_accepted & !valid_coordinates(final_longitude, final_latitude)) ||
    any(!final_accepted & (
      is.finite(final_longitude) | is.finite(final_latitude)
    ))
) {
  stop("The augmented ArcGIS cascade failed its contract checks.", call. = FALSE)
}

grid <- readRDS(GRID_FILE)
if (!inherits(grid, "sf") || is.na(sf::st_crs(grid)) ||
    !"hex_id" %in% names(grid)) {
  stop(
    "The analysis grid must be an sf object with hex_id and a CRS.",
    call. = FALSE
  )
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

provider_longitude <- suppressWarnings(as.numeric(provider$longitude))
provider_latitude <- suppressWarnings(as.numeric(provider$latitude))
provider_inside_city <- classify_inside_city(
  provider_longitude,
  provider_latitude
)
provider_hex_id <- assign_hex(provider_longitude, provider_latitude)
prior_inside_city <- classify_inside_city(prior_longitude, prior_latitude)
final_inside_city <- classify_inside_city(final_longitude, final_latitude)

# Retain the same operational City-only denominator used by the preceding
# cascade QA. The City locator identifies candidate addresses, and the prior or
# augmented cascade must independently place a reliable point inside the exact
# current full-purpose boundary. This is a linkage rate conditional on that
# candidate set, not an estimate of true geocoding recall.
coa_review_longitude <- suppressWarnings(as.numeric(coa_review$longitude))
coa_review_latitude <- suppressWarnings(as.numeric(coa_review$latitude))
coa_review_score <- suppressWarnings(as.numeric(coa_review$score))
assert_columns(
  names(coa_review),
  c(
    "status", "score", "longitude", "latitude", "zip_agrees",
    "house_number_agrees"
  ),
  basename(COA_REVIEW_FILE)
)
coa_review_inside_city <- classify_inside_city(
  coa_review_longitude,
  coa_review_latitude
)
coa_locator_city_candidate <- !is.na(coa_jurisdiction) &
  coa_jurisdiction == "AUSTIN FULL PURPOSE" &
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
  address_for_geocoding = cascade$address_for_geocoding,
  coa_locator_current_full_candidate = coa_locator_city_candidate,
  coa_locator_high_confidence_current_full_candidate =
    coa_locator_high_confidence_city_candidate,
  prior_cascade_confirmed_inside_current_austin_full =
    prior_accepted & prior_inside_city,
  augmented_cascade_confirmed_inside_current_austin_full =
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
  dplyr::filter(
    !is.na(.data$address_for_geocoding),
    .data$address_for_geocoding != ""
  ) |>
  dplyr::left_join(
    address_city_linkage,
    by = "address_for_geocoding",
    relationship = "many-to-one"
  )
case_city_linkage <- filing_city_linkage |>
  dplyr::group_by(.data$qa_case_key) |>
  dplyr::summarize(
    coa_locator_current_full_candidate = any(
      .data$coa_locator_current_full_candidate %in% TRUE
    ),
    coa_locator_high_confidence_current_full_candidate = any(
      .data$coa_locator_high_confidence_current_full_candidate %in% TRUE
    ),
    prior_cascade_confirmed_inside_current_austin_full = any(
      .data$prior_cascade_confirmed_inside_current_austin_full %in% TRUE
    ),
    augmented_cascade_confirmed_inside_current_austin_full = any(
      .data$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE
    ),
    .groups = "drop"
  )

city_candidate_filing_rows <-
  filing_city_linkage$coa_locator_current_full_candidate %in% TRUE
high_confidence_city_candidate_filing_rows <-
  filing_city_linkage$coa_locator_high_confidence_current_full_candidate %in%
    TRUE
city_candidate_cases <-
  case_city_linkage$coa_locator_current_full_candidate %in% TRUE
high_confidence_city_candidate_cases <-
  case_city_linkage$coa_locator_high_confidence_current_full_candidate %in%
    TRUE

review <- provider |>
  dplyr::transmute(
    address_id = .data$address_id,
    address_for_geocoding = .data$address_for_geocoding,
    address_sha256 = .data$address_sha256,
    opaque_request_id = .data$opaque_request_id,
    provider_batch_result_id = .data$result_id,
    prior_loc_name = .data$prior_loc_name,
    prior_status = .data$prior_status,
    prior_score = .data$prior_score,
    arcgis_target_reason = .data$arcgis_target_reason,
    coa_jurisdiction_label = .data$coa_jurisdiction_label,
    requestable = .data$requestable,
    provider_status = .data$status,
    provider_score = suppressWarnings(as.numeric(.data$score)),
    provider_match_address = .data$match_addr,
    provider_address_type = .data$addr_type,
    provider_house_number = .data$output_house_number,
    provider_zip = .data$output_zip,
    provider_region_abbr = .data$region_abbr,
    longitude = provider_longitude,
    latitude = provider_latitude,
    zip_agrees = .data$zip_agrees,
    house_number_agrees = .data$house_number_agrees,
    texas_agrees = .data$texas_agrees,
    valid_coordinate = .data$valid_coordinate,
    passes_quality_gate = .data$passes_quality_gate,
    inside_current_austin_full = provider_inside_city,
    analysis_hex_id = provider_hex_id
  )

metric_row <- function(metric, value) {
  tibble::tibble(metric = metric, value = as.character(value))
}

safe_rate <- function(numerator, denominator) {
  if (denominator == 0L) NA_real_ else numerator / denominator
}

result_bundle_sha256 <- if (length(cache_payloads) == 0L) {
  hash_object(character())
} else {
  hash_object(vapply(
    cache_payloads[sort(names(cache_payloads))],
    function(payload) payload$response_sha256,
    character(1)
  ))
}

qa <- dplyr::bind_rows(
  metric_row("generated_at_utc", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  metric_row("cache_schema_version", CACHE_SCHEMA_VERSION),
  metric_row("arcgis_world_url", ARCGIS_WORLD_URL),
  metric_row("arcgisgeocode_version", as.character(utils::packageVersion("arcgisgeocode"))),
  metric_row("request_profile_sha256", request_profile_sha256),
  metric_row("source_country", SOURCE_COUNTRY),
  metric_row("category", CATEGORY),
  metric_row("location_type", LOCATION_TYPE),
  metric_row("for_storage", FOR_STORAGE),
  metric_row("match_out_of_range", MATCH_OUT_OF_RANGE),
  metric_row("minimum_score", MINIMUM_SCORE),
  metric_row("accepted_address_types", paste(ACCEPTED_ADDRESS_TYPES, collapse = "|")),
  metric_row("cascade_input_sha256", hash_file(CASCADE_INPUT_FILE)),
  metric_row("prepared_filing_input_sha256", hash_file(PREPARED_FILING_FILE)),
  metric_row("coa_review_input_sha256", hash_file(COA_REVIEW_FILE)),
  metric_row("analysis_grid_sha256", hash_file(GRID_FILE)),
  metric_row("jurisdiction_snapshot_sha256", hash_file(JURISDICTIONS_FILE)),
  metric_row("arcgis_result_bundle_sha256", result_bundle_sha256),
  metric_row("network_enabled", NETWORK_ENABLED),
  metric_row("force_refresh", FORCE_REFRESH),
  metric_row("authentication_method_used", authentication_method_used),
  metric_row("batch_size", BATCH_SIZE),
  metric_row("candidate_addresses", nrow(cascade)),
  metric_row("prior_cascade_accepted", sum(prior_accepted)),
  metric_row("prior_cascade_unresolved", sum(prior_unresolved)),
  metric_row("arcgis_city_target_addresses", sum(arcgis_target)),
  metric_row("arcgis_target_coa_city_prior_unresolved", sum(arcgis_target_reason == "coa_places_in_city_prior_unresolved", na.rm = TRUE)),
  metric_row("arcgis_target_coa_city_replace_census", sum(arcgis_target_reason == "coa_places_in_city_replace_census", na.rm = TRUE)),
  metric_row("arcgis_target_explicit_austin_coa_jurisdiction_missing", sum(arcgis_target_reason == "explicit_austin_prior_unresolved_coa_jurisdiction_missing", na.rm = TRUE)),
  metric_row("target_requestable_addresses", sum(residual_rows$requestable)),
  metric_row("target_unrequestable_addresses", sum(!residual_rows$requestable)),
  metric_row("unique_normalized_addresses", nrow(request_unique)),
  metric_row("unique_address_caches_reused", cache_reused),
  metric_row("unique_addresses_sent_to_arcgis", cache_downloaded),
  metric_row("provider_status_matched", sum(provider$status == "M", na.rm = TRUE)),
  metric_row("provider_status_tied", sum(provider$status == "T", na.rm = TRUE)),
  metric_row("provider_status_unmatched", sum(provider$status == "U", na.rm = TRUE)),
  metric_row("provider_point_address", sum(provider$addr_type == "PointAddress", na.rm = TRUE)),
  metric_row("provider_subaddress", sum(provider$addr_type == "Subaddress", na.rm = TRUE)),
  metric_row("provider_street_address", sum(provider$addr_type == "StreetAddress", na.rm = TRUE)),
  metric_row("provider_score_at_least_threshold", sum(suppressWarnings(as.numeric(provider$score)) >= MINIMUM_SCORE, na.rm = TRUE)),
  metric_row("provider_zip_agreement", sum(provider$zip_agrees, na.rm = TRUE)),
  metric_row("provider_house_number_agreement", sum(provider$house_number_agrees, na.rm = TRUE)),
  metric_row("provider_texas_agreement", sum(provider$texas_agrees, na.rm = TRUE)),
  metric_row("provider_quality_gate_passes", sum(provider_accepted)),
  metric_row("provider_quality_gate_rate_of_target", format(safe_rate(sum(provider_accepted), sum(arcgis_target)), digits = 10, trim = TRUE)),
  metric_row("arcgis_accepted_inside_current_austin_full", sum(provider_accepted & provider_inside_city)),
  metric_row("arcgis_accepted_with_analysis_hex", sum(provider_accepted & !is.na(provider_hex_id))),
  metric_row("augmented_cascade_accepted", sum(final_accepted)),
  metric_row("augmented_cascade_unresolved", sum(!final_accepted)),
  metric_row("augmented_cascade_match_rate", format(safe_rate(sum(final_accepted), nrow(cascade)), digits = 10, trim = TRUE)),
  metric_row("prior_cascade_accepted_inside_current_austin_full", sum(prior_accepted & prior_inside_city)),
  metric_row("augmented_cascade_accepted_inside_current_austin_full", sum(final_accepted & final_inside_city)),
  metric_row("coa_locator_current_full_candidate_addresses", sum(coa_locator_city_candidate)),
  metric_row("coa_locator_current_full_candidates_prior_cascade_confirmed_inside", sum(coa_locator_city_candidate & prior_accepted & prior_inside_city)),
  metric_row("coa_locator_current_full_candidate_prior_linkage_rate", format(safe_rate(sum(coa_locator_city_candidate & prior_accepted & prior_inside_city), sum(coa_locator_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidates_augmented_cascade_confirmed_inside", sum(coa_locator_city_candidate & final_accepted & final_inside_city)),
  metric_row("coa_locator_current_full_candidate_augmented_linkage_rate", format(safe_rate(sum(coa_locator_city_candidate & final_accepted & final_inside_city), sum(coa_locator_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_filing_rows", sum(city_candidate_filing_rows)),
  metric_row("coa_locator_current_full_candidate_filing_rows_prior_cascade_confirmed_inside", sum(city_candidate_filing_rows & filing_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_current_full_candidate_filing_row_prior_linkage_rate", format(safe_rate(sum(city_candidate_filing_rows & filing_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_filing_rows_augmented_cascade_confirmed_inside", sum(city_candidate_filing_rows & filing_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_current_full_candidate_filing_row_augmented_linkage_rate", format(safe_rate(sum(city_candidate_filing_rows & filing_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_cases", sum(city_candidate_cases)),
  metric_row("coa_locator_current_full_candidate_cases_prior_cascade_confirmed_inside", sum(city_candidate_cases & case_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_current_full_candidate_case_prior_linkage_rate", format(safe_rate(sum(city_candidate_cases & case_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_current_full_candidate_cases_augmented_cascade_confirmed_inside", sum(city_candidate_cases & case_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_current_full_candidate_case_augmented_linkage_rate", format(safe_rate(sum(city_candidate_cases & case_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_addresses", sum(coa_locator_high_confidence_city_candidate)),
  metric_row("coa_locator_high_confidence_current_full_candidates_prior_cascade_confirmed_inside", sum(coa_locator_high_confidence_city_candidate & prior_accepted & prior_inside_city)),
  metric_row("coa_locator_high_confidence_current_full_candidate_prior_linkage_rate", format(safe_rate(sum(coa_locator_high_confidence_city_candidate & prior_accepted & prior_inside_city), sum(coa_locator_high_confidence_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidates_augmented_cascade_confirmed_inside", sum(coa_locator_high_confidence_city_candidate & final_accepted & final_inside_city)),
  metric_row("coa_locator_high_confidence_current_full_candidate_augmented_linkage_rate", format(safe_rate(sum(coa_locator_high_confidence_city_candidate & final_accepted & final_inside_city), sum(coa_locator_high_confidence_city_candidate)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_rows", sum(high_confidence_city_candidate_filing_rows)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_rows_prior_cascade_confirmed_inside", sum(high_confidence_city_candidate_filing_rows & filing_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_row_prior_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_filing_rows & filing_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(high_confidence_city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_rows_augmented_cascade_confirmed_inside", sum(high_confidence_city_candidate_filing_rows & filing_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_filing_row_augmented_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_filing_rows & filing_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(high_confidence_city_candidate_filing_rows)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_cases", sum(high_confidence_city_candidate_cases)),
  metric_row("coa_locator_high_confidence_current_full_candidate_cases_prior_cascade_confirmed_inside", sum(high_confidence_city_candidate_cases & case_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_case_prior_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_cases & case_city_linkage$prior_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(high_confidence_city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_cases_augmented_cascade_confirmed_inside", sum(high_confidence_city_candidate_cases & case_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE)),
  metric_row("coa_locator_high_confidence_current_full_candidate_case_augmented_linkage_rate", format(safe_rate(sum(high_confidence_city_candidate_cases & case_city_linkage$augmented_cascade_confirmed_inside_current_austin_full %in% TRUE), sum(high_confidence_city_candidate_cases)), digits = 10, trim = TRUE)),
  metric_row("provider_received_case_or_defendant_ids", FALSE),
  metric_row("provider_result_ids_are_batch_local", TRUE)
)

atomic_write_csv(provider_output, PROVIDER_OUTPUT_FILE)
atomic_write_csv(wide_output, CASCADE_OUTPUT_FILE)
atomic_write_csv(review, REVIEW_FILE)
atomic_write_csv(qa, QA_FILE)

message(
  "Wrote ",
  nrow(wide_output),
  " augmented cascade records: ",
  sum(prior_accepted),
  " prior matches, ",
  sum(provider_accepted),
  " accepted ArcGIS City-target results, and ",
  sum(!final_accepted),
  " unresolved."
)
