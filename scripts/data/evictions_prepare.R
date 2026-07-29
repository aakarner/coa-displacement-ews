################################################################################
# Prepare Eviction Filing Addresses for Geocoding
################################################################################
#
# This standalone script reads Travis County JP eviction-defendant workbook
# exports, performs basic quality control, and prepares defendant
# correspondence addresses for geocoding.
#
# INPUTS:
#   - data/Alex Karner Eviction Report 1-1-20 to 5-22-26 (1).xlsx
#       JP4 standalone export
#   - data/Odyssey-JobOutput-May 20, 2026 16-54-45-3728695-1 (1).xlsx
#       JP1, JP2, JP3, and JP5 export
#
# OUTPUTS:
#   - output/eviction_filings_prepared_for_geocoding.csv
#       Case/defendant-level records with cleaned address fields and QC flags
#   - output/eviction_unique_addresses_for_geocoding.csv
#       One row per unique cleaned address, with candidate/QC flags and row counts
#   - output/eviction_unique_addresses_for_geocoding.rds
#       Character vector of unique plausible street addresses to pass to a geocoder
#   - output/eviction_address_qc_summary.csv
#       Overall QC metrics
#   - output/eviction_address_qc_by_source.csv
#       Source-file/court-level QC metrics
#   - output/eviction_addresses_geocoded.rds (optional)
#       sf object of ArcGIS geocoding results, when geocoding is enabled
#   - output/eviction_addresses_geocoded.csv (optional)
#       Non-spatial CSV copy of ArcGIS geocoding results with longitude/latitude
#
# OPTIONAL ARCGIS GEOCODING:
#   Geocoding is off by default so that data prep can run without spending
#   credits. To geocode from R, install arcgisgeocode and arcgisutils, authenticate
#   with one of the supported ArcGIS credential methods below, then run:
#
#     Sys.setenv(GEOCODE_EVICTION_ADDRESSES = "true")
#     source("scripts/data/evictions_prepare.R")
#
#   Supported authentication environment variables:
#     - ARCGIS_API_KEY
#     - ARCGIS_CLIENT and ARCGIS_SECRET
#     - ARCGIS_USER and ARCGIS_PASSWORD
#     - ARCGIS_CLIENT for interactive OAuth via browser auth_code()
#
#   Other useful controls:
#     - ARCGIS_AUTH_METHOD: auto, key, client, user, code
#     - ARCGIS_FOR_STORAGE: true/false; default true
#     - ARCGIS_BATCH_SIZE: optional ArcGIS batch size
#     - ARCGIS_CHUNK_SIZE: local resumable chunk size; default 1000
#     - ARCGIS_SOURCE_COUNTRY: default USA
################################################################################

required_packages <- c("readxl", "dplyr", "stringr", "readr", "janitor", "lubridate")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install missing package(s) before running this script: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(lubridate)
  library(readr)
  library(readxl)
  library(stringr)
})

source("R/utils.R")

print_header("02a - PREPARE EVICTION ADDRESSES FOR GEOCODING")

DATA_DIR <- "data"
OUTPUT_DIR <- "output"
GEOCODE_ADDRESSES <- tolower(Sys.getenv("GEOCODE_EVICTION_ADDRESSES", "false")) %in%
  c("true", "t", "1", "yes", "y")
ARCGIS_AUTH_METHOD <- tolower(Sys.getenv("ARCGIS_AUTH_METHOD", "auto"))
ARCGIS_FOR_STORAGE <- tolower(Sys.getenv("ARCGIS_FOR_STORAGE", "true")) %in%
  c("true", "t", "1", "yes", "y")
ARCGIS_SOURCE_COUNTRY <- Sys.getenv("ARCGIS_SOURCE_COUNTRY", "USA")
ARCGIS_BATCH_SIZE <- suppressWarnings(as.integer(Sys.getenv("ARCGIS_BATCH_SIZE", NA_character_)))
ARCGIS_CHUNK_SIZE <- suppressWarnings(as.integer(Sys.getenv("ARCGIS_CHUNK_SIZE", "1000")))

if (is.na(ARCGIS_CHUNK_SIZE) || ARCGIS_CHUNK_SIZE < 1) {
  ARCGIS_CHUNK_SIZE <- 1000
}

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

EVICTION_FILES <- c(
  jp4 = file.path(DATA_DIR, "Alex Karner Eviction Report 1-1-20 to 5-22-26 (1).xlsx"),
  jp_1_2_3_5 = file.path(DATA_DIR, "Odyssey-JobOutput-May 20, 2026 16-54-45-3728695-1 (1).xlsx")
)

missing_files <- EVICTION_FILES[!file.exists(EVICTION_FILES)]
if (length(missing_files) > 0) {
  stop(
    "Missing expected eviction workbook(s):\n",
    paste(missing_files, collapse = "\n"),
    call. = FALSE
  )
}

court_to_jp <- c(
  "310" = "JP1",
  "320" = "JP2",
  "330" = "JP3",
  "340" = "JP4",
  "350" = "JP5"
)

read_eviction_workbook <- function(path, source_group) {
  print_progress(paste0("Reading ", basename(path), "..."))

  read_excel(
    path,
    sheet = "JP_EvictionDefendantsReport",
    skip = 2,
    col_types = "text",
    .name_repair = janitor::make_clean_names
  ) %>%
    select(-any_of("x")) %>%
    mutate(
      source_group = source_group,
      source_file = basename(path),
      .before = 1
    )
}

parse_excel_or_text_date <- function(x) {
  x <- str_squish(as.character(x))
  x[x == ""] <- NA_character_

  excel_serial <- suppressWarnings(as.numeric(x))
  out <- as.Date(rep(NA_real_, length(x)), origin = "1970-01-01")

  serial_idx <- !is.na(excel_serial) & excel_serial > 20000 & excel_serial < 80000
  out[serial_idx] <- as.Date(excel_serial[serial_idx], origin = "1899-12-30")

  text_idx <- !serial_idx & !is.na(x)
  if (any(text_idx)) {
    out[text_idx] <- as.Date(parse_date_time(
      x[text_idx],
      orders = c("ymd", "mdy", "dmy", "Ymd HMS", "mdy HMS"),
      quiet = TRUE
    ))
  }

  out
}

clean_address_for_geocoding <- function(address) {
  address %>%
    as.character() %>%
    str_replace_all(regex("<br\\s*/?>", ignore_case = TRUE), ", ") %>%
    str_replace_all(regex("</?[^>]+>", ignore_case = TRUE), " ") %>%
    str_replace_all("&amp;", "&") %>%
    str_replace_all("&nbsp;", " ") %>%
    str_replace_all("[\r\n\t]", " ") %>%
    str_replace_all("\\s*,\\s*", ", ") %>%
    str_replace_all(",\\s*,+", ", ") %>%
    str_squish() %>%
    str_remove("^,\\s*") %>%
    str_remove(",\\s*$") %>%
    str_to_upper() %>%
    # Drop leading attention/care-of fragments when the street address follows.
    str_replace("^([^,]+,\\s*)+(?=\\d+\\w?\\b)", "")
}

get_arcgis_token <- function(auth_method = "auto") {
  if (!requireNamespace("arcgisutils", quietly = TRUE)) {
    stop(
      "Install arcgisutils before geocoding: install.packages('arcgisutils')",
      call. = FALSE
    )
  }

  auth_method <- match.arg(auth_method, c("auto", "key", "client", "user", "code"))

  if (auth_method == "auto") {
    if (nzchar(Sys.getenv("ARCGIS_API_KEY"))) {
      auth_method <- "key"
    } else if (nzchar(Sys.getenv("ARCGIS_CLIENT")) && nzchar(Sys.getenv("ARCGIS_SECRET"))) {
      auth_method <- "client"
    } else if (nzchar(Sys.getenv("ARCGIS_USER")) && nzchar(Sys.getenv("ARCGIS_PASSWORD"))) {
      auth_method <- "user"
    } else if (nzchar(Sys.getenv("ARCGIS_CLIENT")) && interactive()) {
      auth_method <- "code"
    } else {
      stop(
        "No ArcGIS credentials found. Set ARCGIS_API_KEY, ARCGIS_CLIENT/ARCGIS_SECRET, ",
        "ARCGIS_USER/ARCGIS_PASSWORD, or ARCGIS_CLIENT for interactive OAuth.",
        call. = FALSE
      )
    }
  }

  print_progress(paste0("Authenticating with ArcGIS using method: ", auth_method))

  token <- switch(
    auth_method,
    key = arcgisutils::auth_key(),
    client = arcgisutils::auth_client(),
    user = arcgisutils::auth_user(),
    code = arcgisutils::auth_code()
  )

  arcgisutils::set_arc_token(token)
  token
}

as_geocode_sf <- function(geocoded) {
  if (inherits(geocoded, "sf")) {
    return(geocoded)
  }

  if ("geometry" %in% names(geocoded) && inherits(geocoded$geometry, "sfc")) {
    return(sf::st_as_sf(geocoded, sf_column_name = "geometry"))
  }

  stop(
    "Geocoded results do not contain an sf geometry column.",
    call. = FALSE
  )
}

sf_to_geocode_csv <- function(geocoded) {
  geocoded_sf <- as_geocode_sf(geocoded)
  coords <- sf::st_coordinates(sf::st_geometry(geocoded_sf))

  geocoded_sf %>%
    sf::st_drop_geometry() %>%
    mutate(
      longitude = coords[, "X"],
      latitude = coords[, "Y"]
    )
}

geocode_unique_addresses <- function(unique_addresses_table) {
  geocode_packages <- c("arcgisgeocode", "arcgisutils", "sf")
  missing_geocode_packages <- geocode_packages[
    !vapply(geocode_packages, requireNamespace, logical(1), quietly = TRUE)
  ]

  if (length(missing_geocode_packages) > 0) {
    stop(
      "Install missing package(s) before geocoding: ",
      paste(missing_geocode_packages, collapse = ", "),
      call. = FALSE
    )
  }

  addresses_to_geocode <- unique_addresses_table %>%
    filter(geocoding_candidate) %>%
    select(address_id, address_for_geocoding)

  if (nrow(addresses_to_geocode) == 0) {
    print_progress("No candidate addresses available to geocode.")
    return(invisible(NULL))
  }

  token <- get_arcgis_token(ARCGIS_AUTH_METHOD)
  geocode_cache_dir <- file.path(OUTPUT_DIR, "eviction_geocode_cache")
  dir.create(geocode_cache_dir, showWarnings = FALSE, recursive = TRUE)

  chunks <- split(
    addresses_to_geocode,
    ceiling(seq_len(nrow(addresses_to_geocode)) / ARCGIS_CHUNK_SIZE)
  )

  print_progress(paste0(
    "Geocoding ",
    nrow(addresses_to_geocode),
    " unique addresses in ",
    length(chunks),
    " local chunk(s)..."
  ))

  chunk_outputs <- vector("list", length(chunks))

  for (i in seq_along(chunks)) {
    chunk_file <- file.path(
      geocode_cache_dir,
      sprintf("eviction_geocode_chunk_%04d.rds", i)
    )

    if (file.exists(chunk_file)) {
      print_progress(paste0("Loading cached geocode chunk ", i, " of ", length(chunks), "..."))
      chunk_outputs[[i]] <- readRDS(chunk_file)
      next
    }

    chunk <- chunks[[i]]
    print_progress(paste0("Geocoding chunk ", i, " of ", length(chunks), "..."))

    geocode_args <- list(
      single_line = chunk$address_for_geocoding,
      source_country = ARCGIS_SOURCE_COUNTRY,
      for_storage = ARCGIS_FOR_STORAGE,
      token = token,
      .progress = TRUE
    )

    if (!is.na(ARCGIS_BATCH_SIZE)) {
      geocode_args$batch_size <- ARCGIS_BATCH_SIZE
    }

    chunk_geocoded <- do.call(arcgisgeocode::geocode_addresses, geocode_args)

    if (nrow(chunk_geocoded) != nrow(chunk)) {
      warning(
        "ArcGIS returned ",
        nrow(chunk_geocoded),
        " result(s) for ",
        nrow(chunk),
        " input address(es) in chunk ",
        i,
        ". Joining by row order for returned rows only.",
        call. = FALSE
      )
    }

    chunk_index <- chunk %>%
      slice_head(n = nrow(chunk_geocoded))

    chunk_geocoded <- bind_cols(chunk_index, chunk_geocoded) %>%
      as_geocode_sf()

    saveRDS(chunk_geocoded, chunk_file)
    chunk_outputs[[i]] <- chunk_geocoded
  }

  geocoded <- do.call(rbind, chunk_outputs) %>%
    as_geocode_sf()

  saveRDS(
    geocoded,
    file.path(OUTPUT_DIR, "eviction_addresses_geocoded.rds")
  )

  write_csv(
    sf_to_geocode_csv(geocoded),
    file.path(OUTPUT_DIR, "eviction_addresses_geocoded.csv"),
    na = ""
  )

  print_progress(paste0("Saved geocoded addresses: ", nrow(geocoded)))

  invisible(geocoded)
}

print_progress("Combining workbook rows...")
eviction_raw <- bind_rows(
  lapply(names(EVICTION_FILES), function(source_group) {
    read_eviction_workbook(EVICTION_FILES[[source_group]], source_group)
  })
)

print_progress("Cleaning fields and preparing address strings...")
eviction_prepared <- eviction_raw %>%
  mutate(
    across(where(is.character), ~ na_if(str_squish(.x), "")),
    file_date = parse_excel_or_text_date(file_date),
    court = str_squish(court),
    jp_district = unname(court_to_jp[court]),
    jp_district = if_else(
      is.na(jp_district),
      str_extract(case_number, regex("(?<=J)\\d", ignore_case = TRUE)) %>%
        paste0("JP", .),
      jp_district
    ),
    address_raw = correspondence_address,
    address_for_geocoding = clean_address_for_geocoding(correspondence_address),
    address_for_geocoding = na_if(address_for_geocoding, ""),
    address_key = address_for_geocoding,
    missing_address = is.na(address_for_geocoding),
    po_box_address = str_detect(
      coalesce(address_for_geocoding, ""),
      regex("\\bP\\.?\\s*O\\.?\\s*BOX\\b|\\bPOST OFFICE BOX\\b", ignore_case = TRUE)
    ),
    has_tx_zip = str_detect(coalesce(address_for_geocoding, ""), "\\bTX\\s+\\d{5}(-\\d{4})?\\b"),
    has_placeholder_zip = str_detect(coalesce(address_for_geocoding, ""), "\\b[A-Z]{2}\\s+0{5}\\b"),
    has_house_number = str_detect(coalesce(address_for_geocoding, ""), "^\\d+\\w?\\b"),
    invalid_placeholder_address = str_detect(
      coalesce(address_for_geocoding, ""),
      regex("\\b(TRANSIENT|UNKNOWN|HOMELESS|ADDRESS UNKNOWN|NO ADDRESS)\\b", ignore_case = TRUE)
    ) | has_placeholder_zip,
    likely_out_of_state = !missing_address &
      str_detect(coalesce(address_for_geocoding, ""), "\\b[A-Z]{2}\\s+\\d{5}(-\\d{4})?\\b") &
      !has_tx_zip,
    geocoding_candidate = !missing_address &
      !po_box_address &
      !invalid_placeholder_address &
      has_house_number,
    duplicate_case_defendant = duplicated(paste(case_number, defendant_name)) |
      duplicated(paste(case_number, defendant_name), fromLast = TRUE)
  ) %>%
  select(
    source_group,
    source_file,
    court,
    jp_district,
    case_type,
    case_number,
    file_date,
    case_status,
    defendant_name,
    address_raw,
    address_for_geocoding,
    address_key,
    missing_address,
    po_box_address,
    has_tx_zip,
    has_placeholder_zip,
    has_house_number,
    invalid_placeholder_address,
    likely_out_of_state,
    geocoding_candidate,
    duplicate_case_defendant
  )

print_progress("Building unique address table and vector...")
unique_addresses_table <- eviction_prepared %>%
  filter(!missing_address) %>%
  group_by(address_for_geocoding) %>%
  summarize(
    filing_defendant_rows = n(),
    geocoding_candidate = any(geocoding_candidate, na.rm = TRUE),
    po_box_address = any(po_box_address, na.rm = TRUE),
    has_tx_zip = any(has_tx_zip, na.rm = TRUE),
    has_placeholder_zip = any(has_placeholder_zip, na.rm = TRUE),
    has_house_number = any(has_house_number, na.rm = TRUE),
    invalid_placeholder_address = any(invalid_placeholder_address, na.rm = TRUE),
    likely_out_of_state = any(likely_out_of_state, na.rm = TRUE),
    jp_districts = paste(sort(unique(na.omit(jp_district))), collapse = "; "),
    .groups = "drop"
  ) %>%
  arrange(desc(geocoding_candidate), desc(filing_defendant_rows), address_for_geocoding) %>%
  mutate(address_id = row_number(), .before = 1)

unique_addresses <- unique_addresses_table %>%
  filter(geocoding_candidate) %>%
  pull(address_for_geocoding)

qc_summary <- tibble::tibble(
  metric = c(
    "source_files",
    "total_rows",
    "unique_case_numbers",
    "duplicate_case_defendant_rows",
    "missing_addresses",
    "po_box_addresses",
    "placeholder_addresses",
    "addresses_without_tx_zip",
    "addresses_without_house_number",
    "likely_out_of_state_addresses",
    "all_unique_cleaned_addresses",
    "unique_geocoding_candidate_addresses",
    "min_file_date",
    "max_file_date"
  ),
  value = c(
    length(unique(eviction_prepared$source_file)),
    nrow(eviction_prepared),
    n_distinct(eviction_prepared$case_number, na.rm = TRUE),
    sum(eviction_prepared$duplicate_case_defendant, na.rm = TRUE),
    sum(eviction_prepared$missing_address, na.rm = TRUE),
    sum(eviction_prepared$po_box_address, na.rm = TRUE),
    sum(eviction_prepared$invalid_placeholder_address, na.rm = TRUE),
    sum(!eviction_prepared$missing_address & !eviction_prepared$has_tx_zip, na.rm = TRUE),
    sum(!eviction_prepared$missing_address & !eviction_prepared$has_house_number, na.rm = TRUE),
    sum(eviction_prepared$likely_out_of_state, na.rm = TRUE),
    n_distinct(eviction_prepared$address_for_geocoding, na.rm = TRUE),
    length(unique_addresses),
    as.character(min(eviction_prepared$file_date, na.rm = TRUE)),
    as.character(max(eviction_prepared$file_date, na.rm = TRUE))
  )
)

qc_by_source <- eviction_prepared %>%
  group_by(source_group, source_file, court, jp_district) %>%
  summarize(
    rows = n(),
    unique_case_numbers = n_distinct(case_number, na.rm = TRUE),
    missing_addresses = sum(missing_address, na.rm = TRUE),
    po_box_addresses = sum(po_box_address, na.rm = TRUE),
    placeholder_addresses = sum(invalid_placeholder_address, na.rm = TRUE),
    addresses_without_tx_zip = sum(!missing_address & !has_tx_zip, na.rm = TRUE),
    addresses_without_house_number = sum(!missing_address & !has_house_number, na.rm = TRUE),
    likely_out_of_state_addresses = sum(likely_out_of_state, na.rm = TRUE),
    geocoding_candidate_addresses = n_distinct(address_for_geocoding[geocoding_candidate], na.rm = TRUE),
    min_file_date = min(file_date, na.rm = TRUE),
    max_file_date = max(file_date, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(jp_district, source_file)

print_progress("Writing outputs...")
write_csv(
  eviction_prepared,
  file.path(OUTPUT_DIR, "eviction_filings_prepared_for_geocoding.csv"),
  na = ""
)
write_csv(
  unique_addresses_table,
  file.path(OUTPUT_DIR, "eviction_unique_addresses_for_geocoding.csv"),
  na = ""
)
saveRDS(
  unique_addresses,
  file.path(OUTPUT_DIR, "eviction_unique_addresses_for_geocoding.rds")
)
write_csv(
  qc_summary,
  file.path(OUTPUT_DIR, "eviction_address_qc_summary.csv"),
  na = ""
)
write_csv(
  qc_by_source,
  file.path(OUTPUT_DIR, "eviction_address_qc_by_source.csv"),
  na = ""
)

if (GEOCODE_ADDRESSES) {
  print_header("ARCGIS GEOCODING")
  geocode_unique_addresses(unique_addresses_table)
} else {
  print_progress("ArcGIS geocoding is disabled. Set GEOCODE_EVICTION_ADDRESSES=true to enable it.")
}

print_progress(paste0("Prepared ", nrow(eviction_prepared), " filing-defendant rows."))
print_progress(paste0("Unique cleaned addresses: ", nrow(unique_addresses_table)))
print_progress(paste0("Unique candidate addresses ready for geocoding: ", length(unique_addresses)))
print_progress("Done.")
