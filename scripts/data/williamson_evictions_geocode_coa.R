################################################################################
# Geocode Williamson Eviction Addresses with Austin's Public COA Locator
################################################################################
#
# This is the second stage of the Williamson cascade, after conservative local
# matching and before the lower-precision Census fallback. It submits only an
# opaque row ID and address string to the City of Austin's public locator. No
# defendant names or case numbers are used. Address-bearing caches and outputs
# remain under ignored output/ paths.
#
# Network access is needed only to fill a missing cache:
#
#   WILLIAMSON_EVICTION_COA_NETWORK=true \
#     Rscript scripts/data/williamson_evictions_geocode_coa.R
################################################################################

required_packages <- c(
  "arcgisgeocode", "digest", "dplyr", "here", "readr", "sf", "stringr",
  "tibble"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before COA geocoding: ",
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
LOCAL_GEOCODE_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_LOCAL_GEOCODE_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded_local.csv")
)
GRID_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_GRID_FILE",
  here::here("output", "hex_grid.rds")
)
OUTPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_GEOCODE_OUTPUT_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded_coa.csv")
)
QA_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_GEOCODE_QA_FILE",
  here::here("output", "williamson_eviction_geocode_coa_qa.csv")
)
REVIEW_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_GEOCODE_REVIEW_FILE",
  here::here("output", "williamson_eviction_geocode_coa_review.csv")
)
CACHE_ROOT <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_CACHE_DIR",
  here::here("output", "williamson_eviction_coa_geocode_cache")
)
COA_LOCATOR_URL <- Sys.getenv(
  "WILLIAMSON_EVICTION_COA_LOCATOR_URL",
  paste0(
    "https://maps.austintexas.gov/arcgis/rest/services/Geocode/",
    "COA_Locator/GeocodeServer"
  )
)
CACHE_SCHEMA_VERSION <- 1L
NETWORK_ENABLED <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_COA_NETWORK",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
FORCE_REFRESH <- tolower(Sys.getenv(
  "WILLIAMSON_EVICTION_COA_FORCE_REFRESH",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
CHUNK_SIZE <- suppressWarnings(as.integer(Sys.getenv(
  "WILLIAMSON_EVICTION_COA_CHUNK_SIZE",
  "500"
)))
MINIMUM_SCORE <- suppressWarnings(as.numeric(Sys.getenv(
  "WILLIAMSON_EVICTION_COA_MINIMUM_SCORE",
  "95"
)))
ACCEPTED_ADDRESS_TYPES <- trimws(strsplit(
  Sys.getenv(
    "WILLIAMSON_EVICTION_COA_ACCEPT_ADDRESS_TYPES",
    "SubAddress|PointAddress"
  ),
  "|",
  fixed = TRUE
)[[1]])
ACCEPTED_ADDRESS_TYPES <- ACCEPTED_ADDRESS_TYPES[
  ACCEPTED_ADDRESS_TYPES != ""
]

if (is.na(CHUNK_SIZE) || CHUNK_SIZE < 1L || CHUNK_SIZE > 1000L) {
  stop(
    "WILLIAMSON_EVICTION_COA_CHUNK_SIZE must be between 1 and 1000.",
    call. = FALSE
  )
}
if (is.na(MINIMUM_SCORE) || MINIMUM_SCORE < 0 || MINIMUM_SCORE > 100) {
  stop(
    "WILLIAMSON_EVICTION_COA_MINIMUM_SCORE must be between 0 and 100.",
    call. = FALSE
  )
}
if (length(ACCEPTED_ADDRESS_TYPES) == 0L) {
  stop("At least one COA address type must be accepted.", call. = FALSE)
}

required_files <- c(UNIQUE_ADDRESS_FILE, LOCAL_GEOCODE_FILE, GRID_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing COA-geocoding input(s): ",
    paste(basename(missing_files), collapse = ", "),
    call. = FALSE
  )
}

as_flag <- function(x) {
  tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
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
    stop("Could not atomically replace a COA cache file.", call. = FALSE)
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
    stop("Could not atomically replace a COA geocoding output.", call. = FALSE)
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

clean_external_address <- function(value) {
  value <- trimws(as.character(value))
  value <- gsub('^["“”]+|["“”]+$', "", value, perl = TRUE)
  value <- gsub("[\r\n\t]+", " ", value, perl = TRUE)
  trimws(gsub("[[:space:]]+", " ", value))
}

extract_first_match <- function(value, pattern) {
  result <- stringr::str_match(as.character(value), pattern)[, 2L]
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
    clean_external_address(value),
    "^([0-9]+[A-Za-z]?)\\b"
  ))
}

valid_coordinates <- function(longitude, latitude) {
  is.finite(longitude) &
    is.finite(latitude) &
    dplyr::between(longitude, -180, 180) &
    dplyr::between(latitude, -90, 90)
}

extract_geometry_coordinates <- function(result) {
  coordinates <- matrix(
    NA_real_,
    nrow = nrow(result),
    ncol = 2L,
    dimnames = list(NULL, c("longitude", "latitude"))
  )
  for (index in seq_len(nrow(result))) {
    geometry <- sf::st_geometry(result)[[index]]
    if (is.null(geometry) || sf::st_is_empty(geometry)) next
    point <- suppressWarnings(sf::st_coordinates(geometry))
    if (nrow(point) == 1L && ncol(point) >= 2L) {
      coordinates[index, ] <- as.numeric(point[1L, 1:2])
    }
  }
  coordinates
}

validate_provider_result <- function(result, expected_rows) {
  required <- c(
    "ResultID", "Status", "Score", "Match_addr", "Addr_type", "AddNum",
    "City", "RegionAbbr", "Postal", "JURISDICTION_LABEL", "geometry"
  )
  if (!inherits(result, "sf")) return(NULL)
  if (!all(required %in% names(result)) || nrow(result) != expected_rows) {
    return(NULL)
  }
  result_ids <- suppressWarnings(as.integer(result$ResultID))
  if (
    anyNA(result_ids) ||
      anyDuplicated(result_ids) ||
      !identical(result_ids, seq_len(expected_rows))
  ) {
    return(NULL)
  }
  if (is.na(sf::st_crs(result))) return(NULL)
  result
}

validate_cache <- function(payload, signature, expected_rows) {
  if (
    !is.list(payload) ||
      !identical(payload$cache_schema_version, CACHE_SCHEMA_VERSION) ||
      !identical(payload$request_signature_sha256, signature) ||
      !identical(
        hash_object(payload$result),
        payload$result_object_sha256
      )
  ) {
    return(NULL)
  }
  validate_provider_result(payload$result, expected_rows)
}

message("Reading Williamson candidates and local geocodes...")
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
contract_columns <- names(local_geocodes)
assert_columns(
  names(unique_addresses),
  c("address_id", "address_for_geocoding", "geocoding_candidate"),
  basename(UNIQUE_ADDRESS_FILE)
)
assert_columns(
  contract_columns,
  c(
    "address_id", "address_for_geocoding", "result_id", "loc_name",
    "status", "score", "match_addr", "addr_type", "longitude", "latitude"
  ),
  basename(LOCAL_GEOCODE_FILE)
)
if (
  nrow(unique_addresses) == 0L ||
    anyNA(unique_addresses[c("address_id", "address_for_geocoding")]) ||
    anyNA(unique_addresses$address_id_numeric) ||
    anyDuplicated(unique_addresses$address_id) ||
    anyDuplicated(unique_addresses$address_for_geocoding) ||
    nrow(local_geocodes) != nrow(unique_addresses) ||
    anyDuplicated(local_geocodes$address_id) ||
    anyDuplicated(local_geocodes$address_for_geocoding) ||
    !setequal(local_geocodes$address_id, unique_addresses$address_id) ||
    !setequal(
      local_geocodes$address_for_geocoding,
      unique_addresses$address_for_geocoding
    )
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

request_rows <- unique_addresses |>
  dplyr::transmute(
    address_id = as.character(.data$address_id),
    address_for_geocoding = clean_external_address(
      .data$address_for_geocoding
    )
  )
request_chunks <- split(
  request_rows,
  ceiling(seq_len(nrow(request_rows)) / CHUNK_SIZE)
)
dir.create(CACHE_ROOT, recursive = TRUE, showWarnings = FALSE)
chunk_results <- vector("list", length(request_chunks))
chunk_result_hashes <- character(length(request_chunks))
downloaded_chunks <- 0L
reused_chunks <- 0L
coa_geocoder <- NULL

message(
  "Loading or requesting COA results in ",
  length(request_chunks),
  " resumable chunk(s)..."
)
for (chunk_index in seq_along(request_chunks)) {
  chunk <- request_chunks[[chunk_index]]
  signature <- hash_object(list(
    cache_schema_version = CACHE_SCHEMA_VERSION,
    locator_url = COA_LOCATOR_URL,
    output_crs = 4326L,
    address_ids = chunk$address_id,
    address_values = chunk$address_for_geocoding
  ))
  cache_file <- file.path(
    CACHE_ROOT,
    sprintf(
      "chunk_%05d_%s.rds",
      chunk_index,
      substr(signature, 1L, 16L)
    )
  )
  result <- NULL
  payload <- NULL
  if (file.exists(cache_file) && !FORCE_REFRESH) {
    payload <- tryCatch(readRDS(cache_file), error = function(error) NULL)
    result <- validate_cache(payload, signature, nrow(chunk))
  }

  if (is.null(result)) {
    if (!NETWORK_ENABLED) {
      stop(
        "A COA locator cache chunk is missing or invalid. Run with ",
        "WILLIAMSON_EVICTION_COA_NETWORK=true to transmit and geocode the ",
        "authorized addresses.",
        call. = FALSE
      )
    }
    if (is.null(coa_geocoder)) {
      coa_geocoder <- arcgisgeocode::geocode_server(
        COA_LOCATOR_URL,
        token = NULL
      )
    }
    result <- tryCatch(
      arcgisgeocode::geocode_addresses(
        single_line = chunk$address_for_geocoding,
        crs = 4326,
        batch_size = nrow(chunk),
        geocoder = coa_geocoder,
        token = NULL,
        .progress = FALSE
      ),
      error = function(error) NULL
    )
    result <- validate_provider_result(result, nrow(chunk))
    if (is.null(result)) {
      stop(
        "A COA locator chunk failed its exact row/ID/schema validation. ",
        "No address or response content is shown.",
        call. = FALSE
      )
    }
    payload <- list(
      cache_schema_version = CACHE_SCHEMA_VERSION,
      request_signature_sha256 = signature,
      result_object_sha256 = hash_object(result),
      retrieved_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      result = result
    )
    atomic_save_rds(payload, cache_file)
    downloaded_chunks <- downloaded_chunks + 1L
  } else {
    reused_chunks <- reused_chunks + 1L
  }
  result$request_chunk <- chunk_index
  result$address_id <- chunk$address_id
  chunk_results[[chunk_index]] <- result
  chunk_result_hashes[[chunk_index]] <- payload$result_object_sha256
}

provider_sf <- do.call(rbind, chunk_results)
if (
  nrow(provider_sf) != nrow(unique_addresses) ||
    anyNA(provider_sf$address_id) ||
    anyDuplicated(provider_sf$address_id) ||
    !identical(as.character(provider_sf$address_id), unique_addresses$address_id)
) {
  stop("The assembled COA locator registry is incomplete.", call. = FALSE)
}
provider_coordinates <- extract_geometry_coordinates(provider_sf)
provider <- provider_sf |>
  sf::st_drop_geometry() |>
  dplyr::mutate(
    longitude = provider_coordinates[, "longitude"],
    latitude = provider_coordinates[, "latitude"],
    input_zip = extract_input_zip(request_rows$address_for_geocoding),
    input_house_number = extract_leading_house_number(
      request_rows$address_for_geocoding
    ),
    output_zip = stringr::str_extract(as.character(.data$Postal), "[0-9]{5}"),
    output_house_number = toupper(trimws(as.character(.data$AddNum))),
    zip_agrees = !is.na(.data$input_zip) &
      .data$input_zip == .data$output_zip,
    house_number_agrees = !is.na(.data$input_house_number) &
      .data$input_house_number == .data$output_house_number,
    valid_coordinate = valid_coordinates(.data$longitude, .data$latitude),
    passes_quality_gate = .data$Status == "M" &
      suppressWarnings(as.numeric(.data$Score)) >= MINIMUM_SCORE &
      .data$Addr_type %in% ACCEPTED_ADDRESS_TYPES &
      .data$zip_agrees &
      .data$house_number_agrees &
      .data$valid_coordinate
  )
provider_accepted <- provider$passes_quality_gate %in% TRUE

provider_output <- tibble::as_tibble(stats::setNames(
  rep(
    list(rep(NA_character_, nrow(unique_addresses))),
    length(contract_columns)
  ),
  contract_columns
))
provider_output$address_id <- unique_addresses$address_id
provider_output$address_for_geocoding <-
  unique_addresses$address_for_geocoding
provider_output$result_id <- unique_addresses$address_id
provider_output$loc_name <- ifelse(
  provider_accepted,
  "city_of_austin_public_coa_locator",
  "city_of_austin_public_coa_locator_unresolved"
)
provider_output$status <- ifelse(provider_accepted, "M", "U")
provider_output$score <- ifelse(
  provider_accepted,
  as.character(provider$Score),
  "0"
)

assign_if_present <- function(contract_column, provider_column) {
  if (contract_column %in% names(provider_output) &&
      provider_column %in% names(provider)) {
    provider_output[[contract_column]] <<- ifelse(
      provider_accepted,
      as.character(provider[[provider_column]]),
      NA_character_
    )
  }
}

provider_map <- c(
  match_addr = "Match_addr",
  long_label = "LongLabel",
  short_label = "ShortLabel",
  addr_type = "Addr_type",
  type_field = "Type",
  place_name = "PlaceName",
  place_addr = "Place_addr",
  rank = "Rank",
  add_bldg = "AddBldg",
  add_num = "AddNum",
  add_num_from = "AddNumFrom",
  add_num_to = "AddNumTo",
  add_range = "AddRange",
  side = "Side",
  st_pre_dir = "StPreDir",
  st_pre_type = "StPreType",
  st_name = "StName",
  st_type = "StType",
  st_dir = "StDir",
  sub_addr = "SubAddr",
  st_addr = "StAddr",
  block = "Block",
  sector = "Sector",
  nbrhd = "Nbrhd",
  district = "District",
  city = "City",
  metro_area = "MetroArea",
  subregion = "Subregion",
  region = "Region",
  region_abbr = "RegionAbbr",
  territory = "Territory",
  zone = "Zone",
  postal = "Postal",
  postal_ext = "PostalExt",
  country = "Country",
  cntry_name = "CntryName",
  lang_code = "LangCode",
  distance = "Distance",
  x = "X",
  y = "Y",
  display_x = "DisplayX",
  display_y = "DisplayY",
  xmin = "Xmin",
  xmax = "Xmax",
  ymin = "Ymin",
  ymax = "Ymax",
  ex_info = "ExInfo"
)
for (contract_column in names(provider_map)) {
  assign_if_present(contract_column, provider_map[[contract_column]])
}
provider_output$longitude <- ifelse(
  provider_accepted,
  as.character(provider$longitude),
  NA_character_
)
provider_output$latitude <- ifelse(
  provider_accepted,
  as.character(provider$latitude),
  NA_character_
)

if (
  nrow(provider_output) != nrow(unique_addresses) ||
    !identical(names(provider_output), contract_columns) ||
    anyNA(provider_output[c("address_id", "address_for_geocoding")]) ||
    anyDuplicated(provider_output$address_id) ||
    anyDuplicated(provider_output$address_for_geocoding) ||
    any(!provider_output$status %in% c("M", "U")) ||
    any(provider_accepted & !valid_coordinates(
      suppressWarnings(as.numeric(provider_output$longitude)),
      suppressWarnings(as.numeric(provider_output$latitude))
    ))
) {
  stop("The COA provider registry failed its wide-contract checks.", call. = FALSE)
}

grid <- readRDS(GRID_FILE)
if (!inherits(grid, "sf") || is.na(sf::st_crs(grid)) ||
    !"hex_id" %in% names(grid)) {
  stop("The analysis grid must be an sf object with hex_id and a CRS.", call. = FALSE)
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
      rows <- vapply(
        intersections[unique_match],
        function(index) index[[1L]],
        integer(1)
      )
      result[points$row_id[unique_match]] <- grid$hex_id[rows]
    }
  }
  result
}

local_longitude <- suppressWarnings(as.numeric(local_geocodes$longitude))
local_latitude <- suppressWarnings(as.numeric(local_geocodes$latitude))
local_score <- suppressWarnings(as.numeric(local_geocodes$score))
local_accepted <- local_geocodes$status %in% c("M", "T") &
  local_score >= 90 &
  valid_coordinates(local_longitude, local_latitude)
local_hex_id <- assign_hex(local_longitude, local_latitude)
provider_hex_id <- assign_hex(provider$longitude, provider$latitude)
validation_available <- local_accepted & provider_accepted
validation_distance_m <- rep(NA_real_, nrow(unique_addresses))
if (any(validation_available)) {
  local_points <- sf::st_as_sf(
    tibble::tibble(
      longitude = local_longitude[validation_available],
      latitude = local_latitude[validation_available]
    ),
    coords = c("longitude", "latitude"),
    crs = 4326
  ) |>
    sf::st_transform(3083)
  provider_points <- sf::st_as_sf(
    tibble::tibble(
      longitude = provider$longitude[validation_available],
      latitude = provider$latitude[validation_available]
    ),
    coords = c("longitude", "latitude"),
    crs = 4326
  ) |>
    sf::st_transform(3083)
  validation_distance_m[validation_available] <- as.numeric(sf::st_distance(
    local_points,
    provider_points,
    by_element = TRUE
  ))
}
same_hex <- dplyr::if_else(
  validation_available & !is.na(local_hex_id) & !is.na(provider_hex_id),
  local_hex_id == provider_hex_id,
  NA
)

review <- tibble::tibble(
  address_id = unique_addresses$address_id,
  address_for_geocoding = unique_addresses$address_for_geocoding,
  status = provider$Status,
  score = suppressWarnings(as.numeric(provider$Score)),
  match_address = provider$Match_addr,
  address_type = provider$Addr_type,
  output_house_number = provider$output_house_number,
  output_zip = provider$output_zip,
  jurisdiction_label = provider$JURISDICTION_LABEL,
  longitude = provider$longitude,
  latitude = provider$latitude,
  zip_agrees = provider$zip_agrees,
  house_number_agrees = provider$house_number_agrees,
  passes_quality_gate = provider_accepted,
  analysis_hex_id = provider_hex_id,
  local_accepted = local_accepted,
  local_analysis_hex_id = local_hex_id,
  local_coa_same_hex = same_hex,
  local_coa_distance_m = validation_distance_m
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
comparable_validation <- validation_available & !is.na(same_hex)
qa <- dplyr::bind_rows(
  metric_row("generated_at_utc", format(Sys.time(), tz = "UTC", usetz = TRUE)),
  metric_row("cache_schema_version", CACHE_SCHEMA_VERSION),
  metric_row("coa_locator_url", COA_LOCATOR_URL),
  metric_row("minimum_score", MINIMUM_SCORE),
  metric_row("accepted_address_types", paste(ACCEPTED_ADDRESS_TYPES, collapse = "|")),
  metric_row("candidate_input_sha256", hash_file(UNIQUE_ADDRESS_FILE)),
  metric_row("local_geocode_input_sha256", hash_file(LOCAL_GEOCODE_FILE)),
  metric_row("analysis_grid_sha256", hash_file(GRID_FILE)),
  metric_row("coa_result_bundle_sha256", hash_object(chunk_result_hashes)),
  metric_row("network_enabled", NETWORK_ENABLED),
  metric_row("force_refresh", FORCE_REFRESH),
  metric_row("chunk_size", CHUNK_SIZE),
  metric_row("chunks", length(request_chunks)),
  metric_row("chunks_downloaded", downloaded_chunks),
  metric_row("chunks_reused", reused_chunks),
  metric_row("candidate_addresses", nrow(unique_addresses)),
  metric_row("provider_status_matched", sum(provider$Status == "M", na.rm = TRUE)),
  metric_row("provider_status_tied", sum(provider$Status == "T", na.rm = TRUE)),
  metric_row("provider_status_unmatched", sum(provider$Status == "U", na.rm = TRUE)),
  metric_row("provider_point_address", sum(provider$Addr_type == "PointAddress", na.rm = TRUE)),
  metric_row("provider_subaddress", sum(provider$Addr_type == "SubAddress", na.rm = TRUE)),
  metric_row("provider_street_address", sum(provider$Addr_type == "StreetAddress", na.rm = TRUE)),
  metric_row("provider_score_at_least_threshold", sum(suppressWarnings(as.numeric(provider$Score)) >= MINIMUM_SCORE, na.rm = TRUE)),
  metric_row("provider_zip_agreement", sum(provider$zip_agrees, na.rm = TRUE)),
  metric_row("provider_house_number_agreement", sum(provider$house_number_agrees, na.rm = TRUE)),
  metric_row("provider_quality_gate_passes", sum(provider_accepted)),
  metric_row("provider_quality_gate_rate", format(safe_rate(sum(provider_accepted), nrow(unique_addresses)), digits = 10, trim = TRUE)),
  metric_row("local_validation_provider_matches", sum(validation_available)),
  metric_row("local_validation_comparable_hexes", sum(comparable_validation)),
  metric_row("local_validation_same_hex", sum(comparable_validation & same_hex)),
  metric_row("local_validation_same_hex_rate", format(safe_rate(sum(comparable_validation & same_hex), sum(comparable_validation)), digits = 10, trim = TRUE)),
  metric_row("local_validation_distance_median", format(safe_quantile(validation_distance_m[validation_available], 0.5), digits = 10, trim = TRUE)),
  metric_row("local_validation_distance_p90", format(safe_quantile(validation_distance_m[validation_available], 0.9), digits = 10, trim = TRUE))
)

atomic_write_csv(provider_output, OUTPUT_FILE)
atomic_write_csv(review, REVIEW_FILE)
atomic_write_csv(qa, QA_FILE)

message(
  "Wrote ",
  nrow(provider_output),
  " COA provider records; ",
  sum(provider_accepted),
  " passed the point-address quality gate."
)
