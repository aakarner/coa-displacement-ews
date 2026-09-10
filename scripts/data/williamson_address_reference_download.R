################################################################################
# Download a Local Williamson County Address-Point Reference
################################################################################
#
# This script downloads only public address points intersecting the bounding
# box of analysis hexes assigned to Williamson County. Court-file addresses are
# never read or transmitted. The public layer is queried for object IDs first,
# then fetched in independently validated and hashed pages of no more than
# 2,000 records.
#
# Network access is disabled unless explicitly enabled:
#
#   WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true \
#     Rscript scripts/data/williamson_address_reference_download.R
#
# Without that flag, a run can only assemble outputs from a complete, valid
# local cache. All cache and address-bearing outputs live under output/, which
# is ignored by git. The QA file contains counts, hashes, and source metadata;
# it never contains an address.
################################################################################

required_packages <- c(
  "digest", "dplyr", "here", "httr2", "jsonlite", "readr", "sf", "tibble"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before downloading the address reference: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
})

GRID_FILE <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_GRID_FILE",
  here::here("output", "hex_grid.rds")
)
COUNTY_FILE <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_COUNTY_FILE",
  here::here("config", "hex_county_assignment_2024.csv")
)
OUTPUT_RDS <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_RDS",
  here::here("output", "williamson_address_reference.rds")
)
OUTPUT_CSV <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_CSV",
  here::here("output", "williamson_address_reference.csv")
)
QA_FILE <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_QA_FILE",
  here::here("output", "williamson_address_reference_qa.csv")
)
CACHE_ROOT <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_CACHE_DIR",
  here::here("output", "williamson_address_reference_cache")
)

LAYER_URL <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_LAYER_URL",
  paste0(
    "https://gis.wilco.org/arcgis/rest/services/public/",
    "county_web_address_information/FeatureServer/0"
  )
)
QUERY_URL <- paste0(sub("/+$", "", LAYER_URL), "/query")
OUT_FIELDS <- c(
  "OBJECTID", "FullAddress", "AddressNumber", "AddressType",
  "AddressUnit", "County", "City", "PCT_NUMBER"
)
CACHE_SCHEMA_VERSION <- 1L
SERVER_MAX_RECORD_COUNT <- 2000L

NETWORK_ENABLED <- tolower(Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_NETWORK",
  "false"
)) %in% c("true", "t", "1", "yes", "y")
PAGE_SIZE <- suppressWarnings(as.integer(Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_PAGE_SIZE",
  as.character(SERVER_MAX_RECORD_COUNT)
)))
REQUEST_TIMEOUT <- suppressWarnings(as.numeric(Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_TIMEOUT",
  "120"
)))

if (is.na(PAGE_SIZE) || PAGE_SIZE < 1L ||
    PAGE_SIZE > SERVER_MAX_RECORD_COUNT) {
  stop(
    "WILLIAMSON_ADDRESS_REFERENCE_PAGE_SIZE must be between 1 and 2000.",
    call. = FALSE
  )
}
if (is.na(REQUEST_TIMEOUT) || REQUEST_TIMEOUT <= 0) {
  stop(
    "WILLIAMSON_ADDRESS_REFERENCE_TIMEOUT must be positive.",
    call. = FALSE
  )
}

required_files <- c(GRID_FILE, COUNTY_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing address-reference input(s): ",
    paste(basename(missing_files), collapse = ", "),
    call. = FALSE
  )
}

hash_text <- function(x) {
  digest::digest(enc2utf8(x), algo = "sha256", serialize = FALSE)
}

canonical_json <- function(x) {
  jsonlite::toJSON(
    x,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = 16
  )
}

hash_object <- function(x) {
  hash_text(canonical_json(x))
}

hash_file <- function(path) {
  digest::digest(path, algo = "sha256", file = TRUE)
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
    stop("Could not atomically replace cache/output RDS.", call. = FALSE)
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
    stop("Could not atomically replace cache/output CSV.", call. = FALSE)
  }
  invisible(path)
}

scalar_value <- function(x) {
  if (is.null(x) || length(x) == 0L || is.list(x)) {
    return(NA_character_)
  }
  value <- as.character(x[[1L]])
  if (length(value) == 0L || is.na(value) || !nzchar(value)) {
    NA_character_
  } else {
    value
  }
}

parse_arcgis_json <- function(response_text, request_label) {
  parsed <- tryCatch(
    jsonlite::fromJSON(response_text, simplifyVector = FALSE),
    error = function(error) NULL
  )
  if (is.null(parsed) || !is.list(parsed)) {
    stop(
      "The public address service returned invalid JSON for ",
      request_label,
      ". No response contents are shown.",
      call. = FALSE
    )
  }
  if (!is.null(parsed$error)) {
    code <- suppressWarnings(as.integer(scalar_value(parsed$error$code)))
    code_label <- if (is.na(code)) "unknown" else as.character(code)
    stop(
      "The public address service returned ArcGIS error code ",
      code_label,
      " for ",
      request_label,
      ". No response contents are shown.",
      call. = FALSE
    )
  }
  parsed
}

perform_public_query <- function(parameters, request_label) {
  if (!NETWORK_ENABLED) {
    stop(
      "A public-data download is required, but network access is disabled. ",
      "Set WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true to allow it.",
      call. = FALSE
    )
  }
  request <- httr2::request(QUERY_URL) |>
    httr2::req_method("POST") |>
    httr2::req_timeout(REQUEST_TIMEOUT) |>
    httr2::req_user_agent("coa-displacement-ews/address-reference")
  request <- do.call(
    httr2::req_body_form,
    c(list(.req = request), parameters)
  )
  response <- tryCatch(
    httr2::req_perform(request),
    error = function(error) {
      stop(
        "The public Williamson address-point request failed for ",
        request_label,
        ". No request or response values are shown.",
        call. = FALSE
      )
    }
  )
  response_text <- httr2::resp_body_string(response)
  list(
    text = response_text,
    parsed = parse_arcgis_json(response_text, request_label)
  )
}

read_counties <- function(path) {
  county_header <- readr::read_csv(
    path,
    n_max = 0L,
    show_col_types = FALSE,
    progress = FALSE
  )
  required <- c("hex_id", "h3_index", "source_county")
  missing <- setdiff(required, names(county_header))
  if (length(missing) > 0L) {
    stop(
      "County reference is missing column(s): ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  readr::read_csv(
    path,
    col_types = cols(
      hex_id = col_integer(),
      h3_index = col_character(),
      source_county = col_character(),
      .default = col_skip()
    ),
    show_col_types = FALSE,
    progress = FALSE
  )
}

message("Defining the public-data query from the Williamson analysis grid...")
grid <- readRDS(GRID_FILE)
if (!inherits(grid, "sf") || is.na(sf::st_crs(grid))) {
  stop("The analysis grid must be an sf object with a known CRS.", call. = FALSE)
}
county_reference <- read_counties(COUNTY_FILE)
grid_keys <- grid |>
  sf::st_drop_geometry() |>
  dplyr::select("hex_id", "h3_index")
if (
  anyNA(grid_keys) ||
    anyDuplicated(grid_keys$hex_id) ||
    anyDuplicated(county_reference$hex_id) ||
    !setequal(grid_keys$hex_id, county_reference$hex_id)
) {
  stop(
    "The grid and county reference must contain the same unique hex IDs.",
    call. = FALSE
  )
}

williamson_hexes <- grid |>
  dplyr::inner_join(
    county_reference |>
      dplyr::filter(.data$source_county == "Williamson") |>
      dplyr::select("hex_id", "h3_index"),
    by = c("hex_id", "h3_index")
  ) |>
  sf::st_make_valid() |>
  sf::st_transform(4326)
if (nrow(williamson_hexes) == 0L ||
    anyDuplicated(williamson_hexes$hex_id)) {
  stop(
    "The Williamson analysis-hex inventory is empty or duplicated.",
    call. = FALSE
  )
}

query_bbox <- sf::st_bbox(williamson_hexes)
bbox_values <- unname(as.numeric(query_bbox[c("xmin", "ymin", "xmax", "ymax")]))
if (any(!is.finite(bbox_values)) ||
    bbox_values[[1L]] >= bbox_values[[3L]] ||
    bbox_values[[2L]] >= bbox_values[[4L]]) {
  stop("The Williamson analysis-grid bounding box is invalid.", call. = FALSE)
}
bbox_parameters <- list(
  geometry = paste(format(bbox_values, digits = 16, trim = TRUE), collapse = ","),
  geometryType = "esriGeometryEnvelope",
  inSR = "4326",
  spatialRel = "esriSpatialRelIntersects"
)

request_signature <- list(
  cache_schema_version = CACHE_SCHEMA_VERSION,
  layer_url = LAYER_URL,
  grid_sha256 = hash_file(GRID_FILE),
  county_reference_sha256 = hash_file(COUNTY_FILE),
  bbox_wgs84 = bbox_values,
  object_id_field = "OBJECTID",
  out_fields = OUT_FIELDS,
  output_spatial_reference = 4326L,
  page_size = PAGE_SIZE
)
request_signature_hash <- hash_object(request_signature)
cache_directory <- file.path(
  CACHE_ROOT,
  paste0("request_", request_signature_hash)
)
dir.create(cache_directory, recursive = TRUE, showWarnings = FALSE)
ids_cache_file <- file.path(cache_directory, "object_ids.rds")

parse_object_ids <- function(parsed) {
  ids <- suppressWarnings(as.numeric(unlist(parsed$objectIds, use.names = FALSE)))
  if (length(ids) == 0L || any(!is.finite(ids)) ||
      any(ids != floor(ids)) || anyDuplicated(ids)) {
    stop(
      "The public address service returned an invalid object-ID inventory.",
      call. = FALSE
    )
  }
  sort(ids)
}

validate_ids_cache <- function(payload) {
  if (!is.list(payload) ||
      !identical(payload$cache_schema_version, CACHE_SCHEMA_VERSION) ||
      !identical(payload$request_signature_hash, request_signature_hash) ||
      !is.character(payload$response_text) ||
      length(payload$response_text) != 1L ||
      !identical(hash_text(payload$response_text), payload$response_sha256)) {
    return(NULL)
  }
  parsed <- tryCatch(
    parse_arcgis_json(payload$response_text, "cached object-ID inventory"),
    error = function(error) NULL
  )
  if (is.null(parsed)) return(NULL)
  ids <- tryCatch(parse_object_ids(parsed), error = function(error) NULL)
  if (is.null(ids) || !identical(hash_object(ids), payload$object_ids_sha256)) {
    return(NULL)
  }
  list(payload = payload, ids = ids)
}

if (NETWORK_ENABLED) {
  message("Downloading the public address-point object-ID inventory...")
  ids_response <- perform_public_query(
    c(
      list(
        f = "json",
        where = "1=1",
        returnIdsOnly = "true",
        returnGeometry = "false"
      ),
      bbox_parameters
    ),
    "the object-ID inventory"
  )
  object_ids <- parse_object_ids(ids_response$parsed)
  ids_payload <- list(
    cache_schema_version = CACHE_SCHEMA_VERSION,
    request_signature_hash = request_signature_hash,
    retrieved_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    response_sha256 = hash_text(ids_response$text),
    object_ids_sha256 = hash_object(object_ids),
    response_text = ids_response$text
  )
  atomic_save_rds(ids_payload, ids_cache_file)
} else {
  if (!file.exists(ids_cache_file)) {
    stop(
      "No valid local object-ID cache is available. Set ",
      "WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true for the initial public-data ",
      "download.",
      call. = FALSE
    )
  }
  cached_ids <- tryCatch(readRDS(ids_cache_file), error = function(error) NULL)
  validated_ids <- validate_ids_cache(cached_ids)
  if (is.null(validated_ids)) {
    stop(
      "The local object-ID cache failed validation. Re-download it with ",
      "WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true.",
      call. = FALSE
    )
  }
  ids_payload <- validated_ids$payload
  object_ids <- validated_ids$ids
}

id_pages <- split(
  object_ids,
  ceiling(seq_along(object_ids) / PAGE_SIZE)
)

parse_feature_rows <- function(parsed) {
  features <- parsed$features
  if (is.null(features)) features <- list()
  if (!is.list(features)) {
    stop("A public address-point page has an invalid feature array.", call. = FALSE)
  }

  rows <- lapply(features, function(feature) {
    attributes <- feature$attributes
    geometry <- feature$geometry
    if (is.null(attributes)) attributes <- list()
    if (is.null(geometry)) geometry <- list()
    tibble::tibble(
      object_id = scalar_value(attributes$OBJECTID),
      full_address = scalar_value(attributes$FullAddress),
      address_number = scalar_value(attributes$AddressNumber),
      address_type = scalar_value(attributes$AddressType),
      address_unit = scalar_value(attributes$AddressUnit),
      county = scalar_value(attributes$County),
      city = scalar_value(attributes$City),
      pct_number = scalar_value(attributes$PCT_NUMBER),
      longitude = suppressWarnings(as.numeric(scalar_value(geometry$x))),
      latitude = suppressWarnings(as.numeric(scalar_value(geometry$y)))
    )
  })
  dplyr::bind_rows(rows)
}

validate_page_payload <- function(payload, expected_ids) {
  expected_ids <- sort(as.numeric(expected_ids))
  if (!is.list(payload) ||
      !identical(payload$cache_schema_version, CACHE_SCHEMA_VERSION) ||
      !identical(payload$request_signature_hash, request_signature_hash) ||
      !identical(payload$requested_ids_sha256, hash_object(expected_ids)) ||
      !is.character(payload$response_text) ||
      length(payload$response_text) != 1L ||
      !identical(hash_text(payload$response_text), payload$response_sha256)) {
    return(NULL)
  }
  parsed <- tryCatch(
    parse_arcgis_json(payload$response_text, "a cached feature page"),
    error = function(error) NULL
  )
  if (is.null(parsed)) return(NULL)
  rows <- tryCatch(parse_feature_rows(parsed), error = function(error) NULL)
  if (is.null(rows) || nrow(rows) != length(expected_ids)) return(NULL)
  returned_ids <- suppressWarnings(as.numeric(rows$object_id))
  if (any(!is.finite(returned_ids)) || anyDuplicated(returned_ids) ||
      !identical(sort(returned_ids), expected_ids)) {
    return(NULL)
  }
  list(payload = payload, rows = rows)
}

message(
  "Loading ",
  length(object_ids),
  " public address points in ",
  length(id_pages),
  " validated page(s)..."
)
page_rows <- vector("list", length(id_pages))
page_response_hashes <- character(length(id_pages))
downloaded_pages <- 0L
reused_pages <- 0L

for (page_index in seq_along(id_pages)) {
  page_ids <- sort(as.numeric(id_pages[[page_index]]))
  page_ids_hash <- hash_object(page_ids)
  page_file <- file.path(
    cache_directory,
    sprintf(
      "page_%05d_%s.rds",
      page_index,
      substr(page_ids_hash, 1L, 16L)
    )
  )

  validated_page <- NULL
  if (file.exists(page_file)) {
    cached_page <- tryCatch(readRDS(page_file), error = function(error) NULL)
    validated_page <- validate_page_payload(cached_page, page_ids)
  }

  if (is.null(validated_page)) {
    if (!NETWORK_ENABLED) {
      stop(
        "A required public address-point cache page is missing or invalid. ",
        "Set WILLIAMSON_ADDRESS_REFERENCE_NETWORK=true to download it.",
        call. = FALSE
      )
    }
    response <- perform_public_query(
      list(
        f = "json",
        where = "1=1",
        objectIds = paste(page_ids, collapse = ","),
        outFields = paste(OUT_FIELDS, collapse = ","),
        returnGeometry = "true",
        outSR = "4326",
        orderByFields = "OBJECTID ASC",
        returnZ = "false",
        returnM = "false"
      ),
      paste0("feature page ", page_index)
    )
    page_payload <- list(
      cache_schema_version = CACHE_SCHEMA_VERSION,
      request_signature_hash = request_signature_hash,
      requested_ids_sha256 = page_ids_hash,
      response_sha256 = hash_text(response$text),
      retrieved_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
      response_text = response$text
    )
    validated_page <- validate_page_payload(page_payload, page_ids)
    if (is.null(validated_page)) {
      stop(
        "A downloaded public address-point page failed exact ID validation. ",
        "No response contents are shown.",
        call. = FALSE
      )
    }
    atomic_save_rds(page_payload, page_file)
    downloaded_pages <- downloaded_pages + 1L
  } else {
    reused_pages <- reused_pages + 1L
  }

  page_rows[[page_index]] <- validated_page$rows
  page_response_hashes[[page_index]] <-
    validated_page$payload$response_sha256
}

all_points <- dplyr::bind_rows(page_rows)
returned_ids <- suppressWarnings(as.numeric(all_points$object_id))
if (
  nrow(all_points) != length(object_ids) ||
    any(!is.finite(returned_ids)) ||
    anyDuplicated(returned_ids) ||
    !identical(sort(returned_ids), object_ids)
) {
  stop(
    "The assembled public address reference failed exact object-ID validation.",
    call. = FALSE
  )
}

message("Filtering public points locally to Williamson analysis hexes...")
valid_coordinate <- is.finite(all_points$longitude) &
  is.finite(all_points$latitude) &
  dplyr::between(all_points$longitude, -180, 180) &
  dplyr::between(all_points$latitude, -90, 90)
valid_county <- toupper(trimws(dplyr::coalesce(all_points$county, ""))) %in%
  c("WILLIAMSON", "WILLIAMSON COUNTY")

join_candidates <- all_points |>
  dplyr::mutate(
    valid_coordinate = valid_coordinate,
    valid_county = valid_county
  ) |>
  dplyr::filter(.data$valid_coordinate, .data$valid_county) |>
  sf::st_as_sf(
    coords = c("longitude", "latitude"),
    crs = 4326,
    remove = FALSE
  )

intersections <- sf::st_intersects(join_candidates, williamson_hexes)
intersection_count <- lengths(intersections)
unique_hex <- intersection_count == 1L
join_candidates$hex_intersection_count <- intersection_count
join_candidates$hex_id <- NA_integer_
join_candidates$h3_index <- NA_character_
if (any(unique_hex)) {
  matched_hex_rows <- vapply(
    intersections[unique_hex],
    function(index) index[[1L]],
    integer(1)
  )
  join_candidates$hex_id[unique_hex] <-
    williamson_hexes$hex_id[matched_hex_rows]
  join_candidates$h3_index[unique_hex] <-
    williamson_hexes$h3_index[matched_hex_rows]
}

reference <- join_candidates |>
  dplyr::filter(.data$hex_intersection_count == 1L) |>
  dplyr::select(
    "object_id", "full_address", "address_number", "address_type",
    "address_unit", "county", "city", "pct_number", "hex_id",
    "h3_index", "longitude", "latitude", "geometry"
  ) |>
  dplyr::arrange(as.numeric(.data$object_id))

if (nrow(reference) == 0L || anyDuplicated(reference$object_id)) {
  stop(
    "Local filtering produced an empty or duplicated address reference.",
    call. = FALSE
  )
}
if (anyNA(reference[c("hex_id", "h3_index", "longitude", "latitude")])) {
  stop("Retained address points have missing spatial keys.", call. = FALSE)
}

source_bundle_hash <- hash_object(c(
  ids_payload$response_sha256,
  page_response_hashes
))
qa <- tibble::tibble(
  metric = c(
    "cache_schema_version",
    "service_layer_url",
    "request_signature_sha256",
    "grid_sha256",
    "county_reference_sha256",
    "source_response_bundle_sha256",
    "source_retrieved_at_utc",
    "assembled_at_utc",
    "network_enabled",
    "page_size",
    "bbox_xmin_wgs84",
    "bbox_ymin_wgs84",
    "bbox_xmax_wgs84",
    "bbox_ymax_wgs84",
    "object_ids_in_bbox",
    "feature_pages",
    "feature_pages_downloaded",
    "feature_pages_reused",
    "features_received",
    "invalid_coordinate_features",
    "non_williamson_or_blank_county_features",
    "valid_williamson_features_outside_analysis_hexes",
    "valid_williamson_features_on_multiple_hexes",
    "retained_reference_points",
    "retained_points_with_full_address"
  ),
  value = c(
    as.character(CACHE_SCHEMA_VERSION),
    LAYER_URL,
    request_signature_hash,
    request_signature$grid_sha256,
    request_signature$county_reference_sha256,
    source_bundle_hash,
    ids_payload$retrieved_at_utc,
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    as.character(NETWORK_ENABLED),
    as.character(PAGE_SIZE),
    format(bbox_values[[1L]], digits = 16, trim = TRUE),
    format(bbox_values[[2L]], digits = 16, trim = TRUE),
    format(bbox_values[[3L]], digits = 16, trim = TRUE),
    format(bbox_values[[4L]], digits = 16, trim = TRUE),
    as.character(length(object_ids)),
    as.character(length(id_pages)),
    as.character(downloaded_pages),
    as.character(reused_pages),
    as.character(nrow(all_points)),
    as.character(sum(!valid_coordinate)),
    as.character(sum(valid_coordinate & !valid_county)),
    as.character(sum(join_candidates$hex_intersection_count == 0L)),
    as.character(sum(join_candidates$hex_intersection_count > 1L)),
    as.character(nrow(reference)),
    as.character(sum(
      !is.na(reference$full_address) & trimws(reference$full_address) != ""
    ))
  )
)

reference_csv <- reference |>
  sf::st_drop_geometry()
atomic_save_rds(reference, OUTPUT_RDS)
atomic_write_csv(reference_csv, OUTPUT_CSV)
atomic_write_csv(qa, QA_FILE)

message(
  "Wrote ",
  nrow(reference),
  " locally filtered public address points and non-sensitive QA."
)
