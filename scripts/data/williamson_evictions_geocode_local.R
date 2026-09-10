################################################################################
# Match Williamson Eviction Addresses to a Local Public Address Reference
################################################################################
#
# This script performs no network requests. It matches prepared Williamson
# eviction-address candidates to the public address points downloaded by
# williamson_address_reference_download.R. Candidate addresses never leave the
# machine and are never printed.
#
# Matching is deliberately conservative:
#   1. normalized punctuation/case exact primary-address match;
#   2. standardized direction, street-suffix, and unit-marker match;
#   3. unit-ignored match.
#
# A match is accepted at every stage only when every compatible public record
# sharing the key resolves to the same analysis hex. The first record in stable
# public-object-ID order supplies a representative coordinate. A confidently
# parsed candidate city rejects conflicting concrete public-city records;
# unknown, blank, and unincorporated public city values remain indeterminate.
# Other candidates receive status U and no coordinates. This local result is
# the first stage of the Williamson cascade; the Census fallback stage retains
# these matches and attempts the unresolved records. The output has exactly the
# same wide column contract as the reviewed Travis geocode registry.
################################################################################

required_packages <- c(
  "digest", "dplyr", "here", "readr", "stringi", "stringr", "tibble"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before local address matching: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

PREPARED_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_PREPARED_FILE",
  here::here(
    "output",
    "williamson_eviction_filings_prepared_for_geocoding.csv"
  )
)
UNIQUE_ADDRESS_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_UNIQUE_ADDRESS_FILE",
  here::here(
    "output",
    "williamson_eviction_unique_addresses_for_geocoding.csv"
  )
)
REFERENCE_FILE <- Sys.getenv(
  "WILLIAMSON_ADDRESS_REFERENCE_CSV",
  here::here("output", "williamson_address_reference.csv")
)
TRAVIS_CONTRACT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_GEOCODE_CONTRACT_FILE",
  here::here("output", "eviction_addresses_geocoded.csv")
)
OUTPUT_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_LOCAL_GEOCODE_OUTPUT_FILE",
  here::here("output", "williamson_eviction_addresses_geocoded_local.csv")
)
QA_FILE <- Sys.getenv(
  "WILLIAMSON_EVICTION_LOCAL_GEOCODE_QA_FILE",
  here::here("output", "williamson_eviction_geocode_local_qa.csv")
)

required_files <- c(
  PREPARED_FILE,
  UNIQUE_ADDRESS_FILE,
  REFERENCE_FILE,
  TRAVIS_CONTRACT_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing local geocoding input(s): ",
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

read_selected_character_csv <- function(path, columns) {
  header <- readr::read_csv(
    path,
    n_max = 0L,
    col_types = cols(.default = col_character()),
    show_col_types = FALSE,
    progress = FALSE
  )
  assert_columns(names(header), columns, basename(path))
  readr::read_csv(
    path,
    col_select = all_of(columns),
    col_types = cols(.default = col_character()),
    show_col_types = FALSE,
    progress = FALSE
  )
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
    stop("Could not atomically replace a local matching output.", call. = FALSE)
  }
  invisible(path)
}

message("Reading local candidate keys and public address points...")
prepared <- read_selected_character_csv(
  PREPARED_FILE,
  c("source_county", "address_for_geocoding", "geocoding_candidate")
)
unique_addresses <- read_selected_character_csv(
  UNIQUE_ADDRESS_FILE,
  c("address_id", "address_for_geocoding", "geocoding_candidate")
)
reference <- readr::read_csv(
  REFERENCE_FILE,
  col_types = cols(
    object_id = col_character(),
    full_address = col_character(),
    address_number = col_character(),
    address_type = col_character(),
    address_unit = col_character(),
    county = col_character(),
    city = col_character(),
    pct_number = col_character(),
    hex_id = col_integer(),
    h3_index = col_character(),
    longitude = col_double(),
    latitude = col_double(),
    .default = col_skip()
  ),
  show_col_types = FALSE,
  progress = FALSE
)
assert_columns(
  names(reference),
  c(
    "object_id", "full_address", "address_number", "address_type",
    "address_unit", "county", "city", "pct_number", "hex_id",
    "h3_index", "longitude", "latitude"
  ),
  basename(REFERENCE_FILE)
)

contract_header <- readr::read_csv(
  TRAVIS_CONTRACT_FILE,
  n_max = 0L,
  col_types = cols(.default = col_character()),
  show_col_types = FALSE,
  progress = FALSE
)
contract_columns <- names(contract_header)
required_contract_columns <- c(
  "address_id", "address_for_geocoding", "result_id", "loc_name",
  "status", "score", "match_addr", "addr_type", "longitude", "latitude"
)
assert_columns(
  contract_columns,
  required_contract_columns,
  basename(TRAVIS_CONTRACT_FILE)
)
if (anyDuplicated(contract_columns)) {
  stop("The Travis geocode contract has duplicate columns.", call. = FALSE)
}

prepared_candidates <- prepared |>
  dplyr::filter(
    tolower(trimws(.data$source_county)) == "williamson",
    as_flag(.data$geocoding_candidate),
    !is.na(.data$address_for_geocoding),
    trimws(.data$address_for_geocoding) != ""
  ) |>
  dplyr::distinct(.data$address_for_geocoding) |>
  dplyr::arrange(.data$address_for_geocoding) |>
  dplyr::pull(.data$address_for_geocoding)

candidates <- unique_addresses |>
  dplyr::filter(
    as_flag(.data$geocoding_candidate),
    !is.na(.data$address_for_geocoding),
    trimws(.data$address_for_geocoding) != ""
  ) |>
  dplyr::mutate(
    address_id_numeric = suppressWarnings(as.numeric(.data$address_id))
  ) |>
  dplyr::arrange(.data$address_id_numeric, .data$address_for_geocoding)

if (
  nrow(candidates) == 0L ||
    anyNA(candidates$address_id_numeric) ||
    any(candidates$address_id_numeric != floor(candidates$address_id_numeric)) ||
    anyDuplicated(candidates$address_id) ||
    anyDuplicated(candidates$address_for_geocoding)
) {
  stop(
    "Williamson candidate addresses require unique integer IDs and unique ",
    "nonblank keys.",
    call. = FALSE
  )
}
if (!setequal(prepared_candidates, candidates$address_for_geocoding)) {
  stop(
    "The prepared filings and unique-address file do not contain the same ",
    "Williamson candidate set.",
    call. = FALSE
  )
}

if (
  nrow(reference) == 0L ||
    anyNA(reference[c("object_id", "hex_id", "h3_index")]) ||
    anyDuplicated(reference$object_id) ||
    any(!is.finite(reference$longitude)) ||
    any(!is.finite(reference$latitude)) ||
    any(!dplyr::between(reference$longitude, -180, 180)) ||
    any(!dplyr::between(reference$latitude, -90, 90))
) {
  stop(
    "The local public address reference has invalid or duplicate spatial keys.",
    call. = FALSE
  )
}
reference <- reference |>
  dplyr::filter(
    !is.na(.data$full_address),
    trimws(.data$full_address) != ""
  ) |>
  dplyr::arrange(suppressWarnings(as.numeric(.data$object_id)), .data$object_id)
if (nrow(reference) == 0L) {
  stop("The local reference has no usable public full addresses.", call. = FALSE)
}

extract_primary_address_one <- function(address) {
  if (is.na(address) || !nzchar(trimws(address))) return(NA_character_)
  parts <- trimws(unlist(strsplit(address, ",", fixed = TRUE)))
  parts <- parts[nzchar(parts)]
  if (length(parts) == 0L) return(NA_character_)

  house_number <- grepl(
    "^[[:space:]]*[0-9]+[A-Za-z]?(?:[-/][0-9A-Za-z]+)?\\b",
    parts,
    perl = TRUE
  )
  starting_positions <- which(house_number)
  start <- if (length(starting_positions) == 0L) {
    1L
  } else {
    starting_positions[[1L]]
  }
  selected <- parts[[start]]

  if (start < length(parts)) {
    unit_pattern <- paste0(
      "^[[:space:]]*(?:#|APT\\b|APARTMENT\\b|UNIT\\b|STE\\b|",
      "SUITE\\b|BLDG\\b|BUILDING\\b|LOT\\b|SPACE\\b|RM\\b|",
      "ROOM\\b|FL\\b|FLOOR\\b)"
    )
    for (index in seq.int(start + 1L, length(parts))) {
      if (!grepl(unit_pattern, parts[[index]], ignore.case = TRUE, perl = TRUE)) {
        break
      }
      selected <- paste(selected, parts[[index]])
    }
  }
  selected
}

extract_primary_address <- function(address) {
  vapply(address, extract_primary_address_one, character(1))
}

basic_address_key <- function(address) {
  value <- as.character(address)
  value[is.na(value)] <- ""
  value <- stringi::stri_trans_general(value, "Latin-ASCII")
  value <- toupper(value)
  value <- gsub("&", " AND ", value, fixed = TRUE)
  value <- gsub("#", " HASH ", value, fixed = TRUE)
  value <- gsub("[^A-Z0-9]+", " ", value)
  value <- trimws(gsub("[[:space:]]+", " ", value))
  value[value == ""] <- NA_character_
  value
}

normalize_city <- function(city) {
  value <- as.character(city)
  value[is.na(value)] <- ""
  value <- stringi::stri_trans_general(value, "Latin-ASCII")
  value <- toupper(value)
  value <- gsub("&", " AND ", value, fixed = TRUE)
  value <- gsub("[^A-Z]+", " ", value)
  value <- trimws(gsub("[[:space:]]+", " ", value))
  value <- sub("^CITY OF[[:space:]]+", "", value)
  value[value %in% c("", "UNKNOWN", "UNK", "N A", "NA")] <- NA_character_
  value
}

# JP2 exports concatenate STREET + CITY + TX + ZIP without commas. Derive the
# city vocabulary from the official public reference and strip only a known
# terminal city, so street-name words are never guessed to be a locality.
known_city_keys <- sort(unique(stats::na.omit(normalize_city(reference$city))))
known_city_keys <- known_city_keys[
  known_city_keys != "UNINCORPORATED WILLIAMSON COUNTY"
]
known_city_keys <- sort(known_city_keys, decreasing = TRUE)
known_city_pattern <- paste(
  gsub("[[:space:]]+", "[[:space:]]+", known_city_keys),
  collapse = "|"
)

extract_known_terminal_city <- function(address) {
  if (length(known_city_keys) == 0L) {
    return(rep(NA_character_, length(address)))
  }
  value <- toupper(as.character(address))
  value[is.na(value)] <- ""
  value <- gsub('["“”]+$', "", trimws(value), perl = TRUE)
  matches <- stringr::str_match(
    value,
    paste0(
      "(?:^|[[:space:],])(",
      known_city_pattern,
      ")[[:space:],]+TX[[:space:]]+[0-9]{5}",
      "(?:-[0-9]{4})?[[:space:]]*$"
    )
  )[, 2L]
  normalize_city(matches)
}

strip_known_terminal_city_state_zip <- function(address) {
  if (length(known_city_keys) == 0L) return(address)
  value <- as.character(address)
  value <- gsub('["“”]+$', "", trimws(value), perl = TRUE)
  value <- sub(
    paste0(
      "[[:space:],]+(?:",
      known_city_pattern,
      ")[[:space:],]+TX[[:space:]]+[0-9]{5}",
      "(?:-[0-9]{4})?[[:space:]]*$"
    ),
    "",
    value,
    ignore.case = TRUE,
    perl = TRUE
  )
  trimws(value)
}

valid_explicit_city <- function(city) {
  if (is.na(city) || !nzchar(trimws(city))) return(FALSE)
  value <- trimws(city)
  nchar(value) >= 2L &&
    nchar(value) <= 60L &&
    grepl("^[A-Za-z][A-Za-z .'-]*$", value) &&
    !grepl(
      paste0(
        "^(?:TX|TEXAS|USA|UNIT|APT|APARTMENT|SUITE|STE|BUILDING|",
        "BLDG|LOT|SPACE|ROOM|RM|FLOOR|FL)$"
      ),
      value,
      ignore.case = TRUE,
      perl = TRUE
    )
}

extract_explicit_city_one <- function(address) {
  if (is.na(address) || !nzchar(trimws(address))) return(NA_character_)
  parts <- trimws(unlist(strsplit(address, ",", fixed = TRUE)))
  # Some report cells retain a surrounding spreadsheet quote. It is only
  # wrapper punctuation and does not weaken the comma + TX/ZIP delimiter test.
  parts <- gsub('^["“”]+|["“”]+$', "", parts, perl = TRUE)
  parts <- trimws(parts)
  parts <- parts[nzchar(parts)]
  if (length(parts) < 2L) return(NA_character_)

  tx_zip_pattern <- "^TX[[:space:]]+[0-9]{5}(?:-[0-9]{4})?$"
  tx_zip_segments <- which(grepl(
    tx_zip_pattern,
    parts,
    ignore.case = TRUE,
    perl = TRUE
  ))
  for (index in tx_zip_segments) {
    if (index > 1L && valid_explicit_city(parts[[index - 1L]])) {
      return(normalize_city(parts[[index - 1L]]))
    }
  }

  # Also accept one comma-delimited segment of the form CITY TX 78701.
  # Requiring that segment boundary avoids guessing a city from an unparsed
  # street line that merely happens to end in TX + ZIP.
  city_tx_zip_pattern <- paste0(
    "^([A-Za-z][A-Za-z .'-]*?)[[:space:]]+TX[[:space:]]+",
    "[0-9]{5}(?:-[0-9]{4})?$"
  )
  for (index in seq.int(2L, length(parts))) {
    match <- regexec(
      city_tx_zip_pattern,
      parts[[index]],
      ignore.case = TRUE,
      perl = TRUE
    )
    captures <- regmatches(parts[[index]], match)[[1L]]
    if (length(captures) == 2L && valid_explicit_city(captures[[2L]])) {
      return(normalize_city(captures[[2L]]))
    }
  }
  NA_character_
}

extract_explicit_city <- function(address) {
  vapply(address, extract_explicit_city_one, character(1))
}

public_city_indeterminate <- function(city_key) {
  is.na(city_key) |
    city_key == "UNINCORPORATED WILLIAMSON COUNTY"
}

normalized_address_key <- function(address) {
  value <- basic_address_key(address)
  replacements <- c(
    NORTHEAST = "NE",
    NORTHWEST = "NW",
    SOUTHEAST = "SE",
    SOUTHWEST = "SW",
    NORTH = "N",
    SOUTH = "S",
    EAST = "E",
    WEST = "W",
    STREET = "ST",
    ROAD = "RD",
    AVENUE = "AVE",
    BOULEVARD = "BLVD",
    DRIVE = "DR",
    LANE = "LN",
    COURT = "CT",
    CIRCLE = "CIR",
    PARKWAY = "PKWY",
    HIGHWAY = "HWY",
    TRAIL = "TRL",
    PLACE = "PL",
    TERRACE = "TER",
    TURNPIKE = "TPKE",
    EXPRESSWAY = "EXPY",
    FREEWAY = "FWY",
    JUNCTION = "JCT",
    APARTMENT = "UNIT",
    APT = "UNIT",
    HASH = "UNIT",
    SUITE = "UNIT",
    STE = "UNIT",
    BUILDING = "UNIT",
    BLDG = "UNIT",
    LOT = "UNIT",
    SPACE = "UNIT",
    ROOM = "UNIT",
    RM = "UNIT",
    FLOOR = "UNIT",
    FL = "UNIT"
  )
  for (token in names(replacements)) {
    value <- gsub(
      paste0("\\b", token, "\\b"),
      replacements[[token]],
      value,
      perl = TRUE
    )
  }
  value <- trimws(gsub("[[:space:]]+", " ", value))
  value[value == ""] <- NA_character_
  value
}

unit_ignored_key <- function(normalized_key) {
  value <- sub("[[:space:]]+UNIT(?:[[:space:]]+.*)?$", "", normalized_key)
  value <- trimws(value)
  value[value == ""] <- NA_character_
  value
}

usable_street_key <- function(key) {
  !is.na(key) & grepl("^[0-9]+[A-Z]*(?:[[:space:]]|$)", key, perl = TRUE)
}

clean_unit_value <- function(unit) {
  value <- basic_address_key(unit)
  value <- sub(
    "^(?:HASH|APARTMENT|APT|UNIT|SUITE|STE|BUILDING|BLDG|LOT|SPACE|ROOM|RM|FLOOR|FL)[[:space:]]+",
    "",
    value,
    perl = TRUE
  )
  value[value == ""] <- NA_character_
  value
}

candidates$primary_address <- strip_known_terminal_city_state_zip(
  extract_primary_address(candidates$address_for_geocoding)
)
candidates$explicit_city <- dplyr::coalesce(
  extract_explicit_city(candidates$address_for_geocoding),
  extract_known_terminal_city(candidates$address_for_geocoding)
)
candidates$exact_key <- basic_address_key(candidates$primary_address)
candidates$normalized_key <- normalized_address_key(candidates$primary_address)
candidates$unit_ignored_key <- unit_ignored_key(candidates$normalized_key)

reference$city_key <- normalize_city(reference$city)
reference$primary_address <- extract_primary_address(reference$full_address)
reference$unit_value <- clean_unit_value(reference$address_unit)
reference$primary_with_unit <- ifelse(
  is.na(reference$unit_value),
  reference$primary_address,
  paste(reference$primary_address, "UNIT", reference$unit_value)
)

reference_variants <- dplyr::bind_rows(
  tibble::tibble(
    reference_index = seq_len(nrow(reference)),
    address_variant = reference$primary_address
  ),
  tibble::tibble(
    reference_index = seq_len(nrow(reference)),
    address_variant = reference$primary_with_unit
  )
) |>
  dplyr::mutate(
    exact_key = basic_address_key(.data$address_variant),
    normalized_key = normalized_address_key(.data$address_variant),
    unit_ignored_key = unit_ignored_key(.data$normalized_key)
  )

build_lookup <- function(key_column) {
  key_rows <- reference_variants |>
    dplyr::transmute(
      key = .data[[key_column]],
      reference_index = .data$reference_index
    ) |>
    dplyr::filter(usable_street_key(.data$key)) |>
    dplyr::distinct(.data$key, .data$reference_index)
  grouped <- split(key_rows$reference_index, key_rows$key)
  lapply(grouped, function(indices) {
    indices <- sort(unique(as.integer(indices)))
    list(
      reference_indices = indices,
      reference_rows = length(indices)
    )
  })
}

message("Building conservative local address-key indexes...")
exact_lookup <- build_lookup("exact_key")
normalized_lookup <- build_lookup("normalized_key")
unit_ignored_lookup <- build_lookup("unit_ignored_key")

resolve_candidate <- function(
  exact_key,
  normalized_key,
  base_key,
  explicit_city
) {
  if (!usable_street_key(exact_key)) {
    return(list(
      reference_index = NA_integer_,
      method = NA_character_,
      score = 0,
      resolution = "unusable_primary_address"
    ))
  }

  stages <- list(
    list(
      key = exact_key,
      lookup = exact_lookup,
      method = "exact_primary_address",
      score = 100,
      ambiguous = "ambiguous_exact_primary_address"
    ),
    list(
      key = normalized_key,
      lookup = normalized_lookup,
      method = "normalized_suffix_or_unit_marker",
      score = 95,
      ambiguous = "ambiguous_normalized_address"
    ),
    list(
      key = base_key,
      lookup = unit_ignored_lookup,
      method = "normalized_unit_ignored",
      score = 92,
      ambiguous = "ambiguous_unit_ignored_address"
    )
  )

  saw_concrete_city_mismatch <- FALSE
  for (stage in stages) {
    if (!usable_street_key(stage$key)) next
    candidate_match <- stage$lookup[[stage$key]]
    if (is.null(candidate_match)) next

    compatible_indices <- candidate_match$reference_indices
    if (!is.na(explicit_city)) {
      public_city_keys <- reference$city_key[compatible_indices]
      city_compatible <- public_city_indeterminate(public_city_keys) |
        public_city_keys == explicit_city
      if (!any(city_compatible)) {
        saw_concrete_city_mismatch <- TRUE
        next
      }
      compatible_indices <- compatible_indices[city_compatible]
    }

    # Coordinates may differ among units or duplicated official records. For a
    # hex-level outcome, the safe criterion is one and only one analysis hex.
    # reference is already in stable numeric OBJECTID order, so the first
    # compatible row is a deterministic representative coordinate.
    compatible_hexes <- unique(reference$hex_id[compatible_indices])
    if (length(compatible_hexes) != 1L) {
      return(list(
        reference_index = NA_integer_,
        method = NA_character_,
        score = 0,
        resolution = stage$ambiguous
      ))
    }
    return(list(
      reference_index = compatible_indices[[1L]],
      method = stage$method,
      score = stage$score,
      resolution = stage$method
    ))
  }

  if (saw_concrete_city_mismatch) {
    return(list(
      reference_index = NA_integer_,
      method = NA_character_,
      score = 0,
      resolution = "concrete_city_mismatch"
    ))
  }

  list(
    reference_index = NA_integer_,
    method = NA_character_,
    score = 0,
    resolution = "no_local_reference_match"
  )
}

message("Matching candidate keys locally; no address data are transmitted...")
resolved <- lapply(
  seq_len(nrow(candidates)),
  function(index) {
    resolve_candidate(
      candidates$exact_key[[index]],
      candidates$normalized_key[[index]],
      candidates$unit_ignored_key[[index]],
      candidates$explicit_city[[index]]
    )
  }
)
resolution <- tibble::tibble(
  reference_index = vapply(
    resolved,
    function(result) result$reference_index,
    integer(1)
  ),
  match_method = vapply(
    resolved,
    function(result) result$method,
    character(1)
  ),
  score = vapply(
    resolved,
    function(result) result$score,
    numeric(1)
  ),
  resolution = vapply(
    resolved,
    function(result) result$resolution,
    character(1)
  )
)
matched <- !is.na(resolution$reference_index)

wide_output <- tibble::as_tibble(stats::setNames(
  rep(list(rep(NA_character_, nrow(candidates))), length(contract_columns)),
  contract_columns
))
wide_output$address_id <- as.character(candidates$address_id)
wide_output$address_for_geocoding <- candidates$address_for_geocoding
wide_output$result_id <- as.character(candidates$address_id)
wide_output$loc_name <- ifelse(
  matched,
  paste0("williamson_public_address_points_local_", resolution$match_method),
  "williamson_public_address_points_local_unresolved"
)
wide_output$status <- ifelse(matched, "M", "U")
wide_output$score <- as.character(resolution$score)

matched_reference <- reference[rep(1L, nrow(candidates)), , drop = FALSE]
if (any(matched)) {
  matched_reference[matched, ] <- reference[
    resolution$reference_index[matched],
    ,
    drop = FALSE
  ]
}

assign_if_present <- function(column, value) {
  if (column %in% names(wide_output)) {
    wide_output[[column]] <<- ifelse(matched, as.character(value), NA_character_)
  }
}

assign_if_present("match_addr", matched_reference$full_address)
assign_if_present("long_label", matched_reference$full_address)
assign_if_present("short_label", matched_reference$full_address)
assign_if_present("place_addr", matched_reference$full_address)
assign_if_present("st_addr", matched_reference$full_address)
assign_if_present(
  "addr_type",
  dplyr::coalesce(matched_reference$address_type, "PointAddress")
)
assign_if_present("type_field", matched_reference$address_type)
assign_if_present("add_num", matched_reference$address_number)
assign_if_present("unit_name", matched_reference$address_unit)
assign_if_present("city", matched_reference$city)
assign_if_present("subregion", rep("Williamson", nrow(candidates)))
assign_if_present("region", rep("Texas", nrow(candidates)))
assign_if_present("region_abbr", rep("TX", nrow(candidates)))
assign_if_present("country", rep("USA", nrow(candidates)))
assign_if_present("cntry_name", rep("United States", nrow(candidates)))
assign_if_present("distance", rep("0", nrow(candidates)))
assign_if_present("x", matched_reference$longitude)
assign_if_present("y", matched_reference$latitude)
assign_if_present("display_x", matched_reference$longitude)
assign_if_present("display_y", matched_reference$latitude)
assign_if_present("xmin", matched_reference$longitude)
assign_if_present("xmax", matched_reference$longitude)
assign_if_present("ymin", matched_reference$latitude)
assign_if_present("ymax", matched_reference$latitude)
assign_if_present("longitude", matched_reference$longitude)
assign_if_present("latitude", matched_reference$latitude)

if (
  nrow(wide_output) != nrow(candidates) ||
    !identical(names(wide_output), contract_columns) ||
    anyNA(wide_output$address_for_geocoding) ||
    anyDuplicated(wide_output$address_for_geocoding) ||
    any(!wide_output$status %in% c("M", "U")) ||
    any(matched & suppressWarnings(as.numeric(wide_output$score)) < 90) ||
    any(matched & (
      is.na(wide_output$longitude) | is.na(wide_output$latitude)
    )) ||
    any(!matched & (
      !is.na(wide_output$longitude) | !is.na(wide_output$latitude)
    ))
) {
  stop("The local geocode output failed its wide-contract checks.", call. = FALSE)
}

resolution_levels <- c(
  "exact_primary_address",
  "normalized_suffix_or_unit_marker",
  "normalized_unit_ignored",
  "ambiguous_exact_primary_address",
  "ambiguous_normalized_address",
  "ambiguous_unit_ignored_address",
  "concrete_city_mismatch",
  "unusable_primary_address",
  "no_local_reference_match"
)
resolution_counts <- table(factor(
  resolution$resolution,
  levels = resolution_levels
))

qa <- tibble::tibble(
  metric = c(
    "generated_at_utc",
    "prepared_input_sha256",
    "unique_candidate_input_sha256",
    "public_reference_input_sha256",
    "travis_wide_contract_sha256",
    "wide_contract_columns",
    "candidate_addresses",
    "candidate_addresses_with_explicit_city",
    "public_reference_points_with_addresses",
    "matched_addresses",
    "unresolved_addresses",
    "match_rate",
    "distinct_public_hexes_used",
    paste0("resolution_", resolution_levels)
  ),
  value = c(
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    hash_file(PREPARED_FILE),
    hash_file(UNIQUE_ADDRESS_FILE),
    hash_file(REFERENCE_FILE),
    hash_file(TRAVIS_CONTRACT_FILE),
    as.character(length(contract_columns)),
    as.character(nrow(candidates)),
    as.character(sum(!is.na(candidates$explicit_city))),
    as.character(nrow(reference)),
    as.character(sum(matched)),
    as.character(sum(!matched)),
    format(mean(matched), digits = 8, trim = TRUE),
    as.character(dplyr::n_distinct(
      reference$hex_id[resolution$reference_index[matched]],
      na.rm = TRUE
    )),
    as.character(unname(resolution_counts))
  )
)

atomic_write_csv(wide_output, OUTPUT_FILE)
atomic_write_csv(qa, QA_FILE)

message(
  "Wrote one local geocode-contract row for each of ",
  nrow(candidates),
  " candidate addresses; matched ",
  sum(matched),
  " and left ",
  sum(!matched),
  " unresolved."
)
