################################################################################
# 02l - Process Ownership Change and Property Transactions
################################################################################
#
# Builds two distinct smoke-signal domains:
#   1. transaction_pressure_index: arms-length/warranty transfer volume,
#      current-unit exposure, and change from an equal prior window.
#   2. ownership_change_index: corporate acquisition intensity, unit exposure,
#      and net corporate acquisition direction.
#
# Travis uses cached deed parties. Hays and Williamson use annual owner
# snapshots for corporate ownership change. All transaction sources are filtered
# to the fixed analysis as-of date.
#
# Optional environment variable:
#   EWS_LANDLORD_MAPPER_DIR=/path/to/landlord-mapper
#
# Outputs:
#   output/ownership_transaction_features_by_parcel.rds
#   output/ownership_transaction_features_by_hex.rds/.csv
#   output/ownership_transaction_source_qa.csv
#   output/transaction_event_type_qa.csv
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
  library(data.table)
  library(dplyr)
  library(lubridate)
  library(readr)
  library(sf)
  library(tibble)
})

print_header("02l - OWNERSHIP CHANGE AND PROPERTY TRANSACTIONS")

OUTPUT_DIR <- project_path("output")
RAW_APPRAISAL_DIR <- project_path(
  "data", "raw_parcels", "appraisal_history"
)
PARCELS_FILE <- file.path(OUTPUT_DIR, "residential_parcels_for_hex_sf.rds")
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")
WILLIAMSON_OWNERS_FILE <- project_path(
  "data", "raw_parcels", "williamson", "wcad_owners.csv"
)
WILLIAMSON_PROPERTY_FILE <- project_path(
  "data", "raw_parcels", "williamson", "wcad_property_certified.csv"
)
WILLIAMSON_SALES_FILE <- project_path(
  "data", "raw_parcels", "williamson",
  "wcad_sales_history_certified.csv"
)

default_landlord_mapper_dir <- normalizePath(
  file.path(project_path(), "..", "landlord-mapper"),
  winslash = "/",
  mustWork = FALSE
)
LANDLORD_MAPPER_DIR <- Sys.getenv(
  "EWS_LANDLORD_MAPPER_DIR",
  unset = default_landlord_mapper_dir
)
TRAVIS_DEEDS_FILE <- file.path(
  LANDLORD_MAPPER_DIR,
  "output",
  "travis_deeds.csv"
)

required_files <- c(
  PARCELS_FILE,
  HEX_GRID_FILE,
  WILLIAMSON_OWNERS_FILE,
  WILLIAMSON_PROPERTY_FILE,
  WILLIAMSON_SALES_FILE,
  TRAVIS_DEEDS_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(
    "Ownership/transaction processing is missing required source file(s):\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}

window_years <- as.integer(EWS_CONFIG$transaction_recent_years)
analysis_as_of <- EWS_CONFIG$transaction_analysis_as_of_date
recent_start <- analysis_as_of %m-% years(window_years) + days(1)
previous_start <- recent_start %m-% years(window_years)
snapshot_years <- seq.int(
  EWS_CONFIG$appraisal_current_year - window_years,
  EWS_CONFIG$appraisal_current_year
)

if (length(snapshot_years) != window_years + 1L) {
  stop("Ownership snapshot window is not internally consistent.", call. = FALSE)
}

blank_to_na <- function(x) {
  x <- trimws(as.character(x))
  x[x == "" | toupper(x) == "NA"] <- NA_character_
  x
}

clean_name <- function(x) {
  x <- toupper(blank_to_na(x))
  x <- gsub("[[:punct:]]+", " ", x)
  x <- gsub("[[:space:]]+", " ", x)
  trimws(x)
}

corporate_name_pattern <- paste(
  "\\bLLC\\b", "\\bL L C\\b", "\\bLTD\\b", "\\bLIMITED\\b",
  "\\bLP\\b", "\\bLLP\\b", "\\bLLLP\\b", "\\bINC\\b",
  "\\bCORP", "\\bCOMPANY\\b", "\\bHOLDING", "\\bPROPERT",
  "\\bINVEST", "\\bCAPITAL\\b", "\\bASSET", "\\bMANAGE",
  "\\bBANK\\b", "\\bREIT\\b", "\\bDEVELOP", "\\bVENTURE",
  "\\bPARTN", "\\bEQUIT", "\\bREALTY\\b", "\\bREAL ESTATE\\b",
  "\\bMORTGAGE\\b", "\\bFOUNDATION\\b",
  sep = "|"
)

is_corporate_name <- function(x) {
  cleaned <- clean_name(x)
  !is.na(cleaned) & grepl(corporate_name_pattern, cleaned, perl = TRUE)
}

collapse_ids <- function(x) {
  x <- sort(unique(blank_to_na(x)))
  x <- x[!is.na(x)]
  if (length(x) == 0) NA_character_ else paste(x, collapse = ";")
}

parse_ymd_date <- function(primary, fallback = NULL) {
  primary <- blank_to_na(primary)
  out <- as.IDate(substr(primary, 1L, 10L), format = "%Y-%m-%d")
  if (!is.null(fallback)) {
    fallback <- blank_to_na(fallback)
    fallback_date <- as.IDate(
      substr(fallback, 1L, 10L),
      format = "%Y-%m-%d"
    )
    out[is.na(out)] <- fallback_date[is.na(out)]
  }
  out
}

parse_mdy_date <- function(primary, fallback = NULL) {
  primary <- blank_to_na(primary)
  primary <- sub("[[:space:]].*$", "", primary)
  out <- as.IDate(primary, format = "%m/%d/%Y")
  if (!is.null(fallback)) {
    fallback <- blank_to_na(fallback)
    fallback <- sub("[[:space:]].*$", "", fallback)
    fallback_date <- as.IDate(fallback, format = "%m/%d/%Y")
    out[is.na(out)] <- fallback_date[is.na(out)]
  }
  out
}

read_zip_table <- function(archive, member, select = NULL, nrows = Inf) {
  command <- paste("unzip -p", shQuote(archive), shQuote(member))
  data.table::fread(
    cmd = command,
    select = select,
    nrows = nrows,
    colClasses = "character",
    fill = TRUE,
    showProgress = FALSE
  )
}

extract_nested_zip <- function(outer_archive, member) {
  listing <- utils::unzip(outer_archive, list = TRUE)
  size <- listing$Length[match(member, listing$Name)]
  if (is.na(size)) {
    stop("Nested archive member not found: ", member, call. = FALSE)
  }
  input <- unz(outer_archive, member, open = "rb")
  on.exit(close(input), add = TRUE)
  bytes <- readBin(input, what = "raw", n = size)
  destination <- tempfile(fileext = ".zip")
  writeBin(bytes, destination)
  destination
}

locate_nested_table <- function(outer_archive, nested_pattern) {
  listing <- utils::unzip(outer_archive, list = TRUE)
  nested_members <- listing$Name[
    grepl("[.]zip$", listing$Name, ignore.case = TRUE) &
      grepl(nested_pattern, basename(listing$Name), ignore.case = TRUE)
  ]
  if (length(nested_members) == 0) {
    stop(
      "No nested member matching ", nested_pattern,
      " in ", outer_archive,
      call. = FALSE
    )
  }
  inner_archive <- extract_nested_zip(outer_archive, nested_members[[1]])
  data_members <- utils::unzip(inner_archive, list = TRUE)$Name
  data_members <- data_members[
    grepl("[.](txt|csv)$", data_members, ignore.case = TRUE)
  ]
  if (length(data_members) == 0) {
    stop("No table found in nested archive.", call. = FALSE)
  }
  list(archive = inner_archive, member = data_members[[1]])
}

normalize_robust <- function(x) {
  normalize_robust_to_100(as.numeric(x))
}

################################################################################
# Parcel-to-hex analysis universe
################################################################################

print_progress("Preparing parcel-to-hex denominators...")
parcels_sf <- load_output(PARCELS_FILE, "residential parcels for hex analysis")
hex_grid <- load_output(HEX_GRID_FILE, "hexagonal grid")
if (!inherits(parcels_sf, "sf") || !inherits(hex_grid, "sf")) {
  stop("Parcel and hex inputs must both be sf objects.", call. = FALSE)
}

parcels_sf <- st_transform(parcels_sf, st_crs(hex_grid))
parcels_joined <- st_join(
  parcels_sf,
  hex_grid %>% select(hex_id),
  join = st_within,
  left = FALSE
)
if (anyDuplicated(parcels_joined$parcel_id)) {
  stop("Parcel-to-hex join produced duplicate parcel assignments.", call. = FALSE)
}

parcel_map <- as.data.table(st_drop_geometry(parcels_joined))
unit_col <- c(
  "property_units_targeted", "units_calibrated_targeted",
  "property_units", "units_calibrated"
)
unit_col <- unit_col[unit_col %in% names(parcel_map)][[1]]
corporate_col <- c("is_corporate_owned", "has_financialized_owner")
corporate_col <- corporate_col[corporate_col %in% names(parcel_map)][[1]]
parcel_map[, residential_units := suppressWarnings(as.numeric(get(unit_col)))]
parcel_map[!is.finite(residential_units) | residential_units <= 0,
  residential_units := 1]
parcel_map[, current_corporate_owned := as.logical(get(corporate_col))]
parcel_map <- parcel_map[, .(
  parcel_id = as.character(parcel_id),
  source_county = as.character(source_county),
  hex_id = as.character(hex_id),
  residential_units,
  current_corporate_owned,
  transaction_window_complete = source_county != "Williamson"
)]
setkey(parcel_map, parcel_id)

parcel_target_qa <- parcel_map[, .(
  target_parcels = .N,
  target_units = sum(residential_units)
), by = source_county]

################################################################################
# Transaction events
################################################################################

print_progress("Reading and filtering Travis deed events...")
travis_deeds <- data.table::fread(
  TRAVIS_DEEDS_FILE,
  select = c(
    "deed_deedID", "deed_deedType", "deed_deedDt",
    "deed_deedRecordedDt", "deed_instrumentNum", "deed_sellerLine",
    "deed_buyerLine", "deed_pID"
  ),
  colClasses = "character",
  showProgress = TRUE
)
travis_deeds[, parcel_id := blank_to_na(deed_pID)]
travis_deeds <- travis_deeds[
  parcel_id %in% parcel_map[source_county == "Travis", parcel_id]
]
travis_deeds[, transaction_date := parse_ymd_date(
  deed_deedDt,
  deed_deedRecordedDt
)]
travis_deeds[, event_type := toupper(blank_to_na(deed_deedType))]
travis_deeds[, market_transfer := event_type %in% c("WD", "SW")]
travis_deeds[, event_id := blank_to_na(deed_instrumentNum)]
travis_deeds[is.na(event_id), event_id := blank_to_na(deed_deedID)]
travis_deeds[is.na(event_id), event_id := paste(
  parcel_id, transaction_date, event_type,
  clean_name(deed_buyerLine), clean_name(deed_sellerLine),
  sep = "|"
)]
travis_deeds[, `:=`(
  buyer_name = blank_to_na(deed_buyerLine),
  seller_name = blank_to_na(deed_sellerLine),
  buyer_corporate = is_corporate_name(deed_buyerLine),
  seller_corporate = is_corporate_name(deed_sellerLine),
  source_county = "Travis",
  transaction_source = "TCAD deeds"
)]
travis_candidates <- travis_deeds[
  !is.na(transaction_date) &
    transaction_date >= as.IDate(previous_start) &
    transaction_date <= as.IDate(analysis_as_of)
]

print_progress("Reading and filtering Hays sales history...")
hays_sales_archive <- file.path(
  RAW_APPRAISAL_DIR,
  "hays",
  as.character(EWS_CONFIG$appraisal_current_year),
  paste0("hays_", EWS_CONFIG$appraisal_current_year, ".zip")
)
hays_sales_location <- locate_nested_table(hays_sales_archive, "SALES")
hays_sales <- read_zip_table(
  hays_sales_location$archive,
  hays_sales_location$member,
  select = c(
    "QuickRefID", "SaleDate", "DeedDate", "InstrumentNumber",
    "PrevOwnerName", "InstrumentType", "DeedType"
  )
)
hays_sales[, parcel_id := paste0("HAYS:", blank_to_na(QuickRefID))]
hays_sales <- hays_sales[
  parcel_id %in% parcel_map[source_county == "Hays", parcel_id]
]
hays_sales[, transaction_date := parse_mdy_date(DeedDate, SaleDate)]
hays_sales[, event_type := toupper(blank_to_na(InstrumentType))]
hays_sales[, market_transfer := event_type %in% c(
  "GWD", "GWDVL", "SWD", "SWDVL", "WD", "WDVL", "CWD"
)]
hays_sales[, event_id := blank_to_na(InstrumentNumber)]
hays_sales[is.na(event_id), event_id := paste(
  parcel_id, transaction_date, event_type, clean_name(PrevOwnerName),
  sep = "|"
)]
hays_sales[, `:=`(
  buyer_name = NA_character_,
  seller_name = blank_to_na(PrevOwnerName),
  buyer_corporate = NA,
  seller_corporate = is_corporate_name(PrevOwnerName),
  source_county = "Hays",
  transaction_source = "Hays SALES history"
)]
hays_candidates <- hays_sales[
  !is.na(transaction_date) &
    transaction_date >= as.IDate(previous_start) &
    transaction_date <= as.IDate(analysis_as_of)
]

print_progress("Reading and filtering Williamson certified sales history...")
williamson_property <- data.table::fread(
  WILLIAMSON_PROPERTY_FILE,
  select = c("PropertyID", "QuickRefID"),
  colClasses = "character",
  showProgress = FALSE
)
williamson_sales <- data.table::fread(
  WILLIAMSON_SALES_FILE,
  select = c(
    "PropertyID", "OwnershipTransferID", "SaleDate", "SaleTypeCode",
    "TransferValidityCode"
  ),
  colClasses = "character",
  showProgress = FALSE
)
williamson_sales[williamson_property, QuickRefID := i.QuickRefID,
  on = "PropertyID"]
williamson_sales[, parcel_id := paste0(
  "WILLIAMSON:", blank_to_na(QuickRefID)
)]
williamson_sales <- williamson_sales[
  parcel_id %in% parcel_map[source_county == "Williamson", parcel_id]
]
williamson_sales[, transaction_date := parse_mdy_date(SaleDate)]
williamson_sales[, event_type := toupper(blank_to_na(SaleTypeCode))]
williamson_sales[, market_transfer :=
  TransferValidityCode == "VALID" & grepl("^AL", event_type)]
williamson_sales[, `:=`(
  event_id = blank_to_na(OwnershipTransferID),
  buyer_name = NA_character_,
  seller_name = NA_character_,
  buyer_corporate = NA,
  seller_corporate = NA,
  source_county = "Williamson",
  transaction_source = "WCAD certified sales history"
)]
williamson_candidates <- williamson_sales[
  !is.na(transaction_date) &
    transaction_date >= as.IDate(previous_start) &
    transaction_date <= as.IDate(analysis_as_of)
]

transaction_candidates <- rbindlist(list(
  travis_candidates[, .(
    source_county, parcel_id, transaction_date, event_type,
    market_transfer, event_id, buyer_name, seller_name,
    buyer_corporate, seller_corporate, transaction_source
  )],
  hays_candidates[, .(
    source_county, parcel_id, transaction_date, event_type,
    market_transfer, event_id, buyer_name, seller_name,
    buyer_corporate, seller_corporate, transaction_source
  )],
  williamson_candidates[, .(
    source_county, parcel_id, transaction_date, event_type,
    market_transfer, event_id, buyer_name, seller_name,
    buyer_corporate, seller_corporate, transaction_source
  )]
), use.names = TRUE, fill = TRUE)

transaction_candidates[, analysis_window := fifelse(
  transaction_date >= as.IDate(recent_start),
  "recent",
  "previous"
)]
event_type_qa <- transaction_candidates[, .(
  candidate_events = .N,
  market_transfer_events = sum(market_transfer, na.rm = TRUE),
  unique_parcels = uniqueN(parcel_id)
), by = .(source_county, analysis_window, event_type)][
  order(source_county, analysis_window, -market_transfer_events, -candidate_events)
]
transaction_source_year_qa <- transaction_candidates[, .(
  candidate_events = .N,
  market_transfer_events = sum(market_transfer, na.rm = TRUE),
  unique_parcels = uniqueN(parcel_id)
), by = .(
  source_county,
  transaction_year = as.integer(format(transaction_date, "%Y"))
)][order(source_county, transaction_year)]

transaction_events <- unique(
  transaction_candidates[market_transfer == TRUE],
  by = c("source_county", "parcel_id", "transaction_date", "event_id")
)
rm(
  travis_deeds, travis_candidates, hays_sales, hays_candidates,
  williamson_sales, williamson_candidates, transaction_candidates
)
invisible(gc())

transaction_by_parcel <- transaction_events[, .(
  transaction_recent_count = sum(analysis_window == "recent"),
  transaction_previous_count = sum(analysis_window == "previous")
), by = .(source_county, parcel_id)]

################################################################################
# Corporate ownership-change events
################################################################################

summarise_owner_snapshot <- function(
    data,
    county,
    tax_year,
    parcel_col,
    owner_id_col,
    owner_name_col) {
  data <- as.data.table(data)
  data[, parcel_id := blank_to_na(get(parcel_col))]
  data[, owner_id := blank_to_na(get(owner_id_col))]
  data[, owner_name := blank_to_na(get(owner_name_col))]
  data <- data[!is.na(parcel_id)]
  data[, .(
    owner_signature = collapse_ids(owner_id),
    corporate_owner = any(is_corporate_name(owner_name)),
    owner_name_available = any(!is.na(owner_name))
  ), by = parcel_id][, `:=`(
    source_county = county,
    tax_year = as.integer(tax_year)
  )]
}

parse_hays_owner_snapshot <- function(tax_year, target_ids) {
  outer_archive <- file.path(
    RAW_APPRAISAL_DIR,
    "hays",
    as.character(tax_year),
    paste0("hays_", tax_year, ".zip")
  )
  location <- locate_nested_table(outer_archive, "OWNER")
  owners <- read_zip_table(
    location$archive,
    location$member,
    select = c("QuickRefID", "OwnerID", "OwnerName")
  )
  owners[, parcel_key := paste0("HAYS:", blank_to_na(QuickRefID))]
  owners <- owners[parcel_key %in% target_ids]
  summarise_owner_snapshot(
    owners,
    county = "Hays",
    tax_year = tax_year,
    parcel_col = "parcel_key",
    owner_id_col = "OwnerID",
    owner_name_col = "OwnerName"
  )
}

parse_williamson_report_snapshot <- function(tax_year, target_ids) {
  archive <- file.path(
    RAW_APPRAISAL_DIR,
    "williamson",
    as.character(tax_year),
    paste0("williamson_", tax_year, ".zip")
  )
  member <- utils::unzip(archive, list = TRUE)$Name[[1]]
  awk_program <- paste(
    'BEGIN{OFS="|"}',
    '/^PID:[[:space:]]*/{pid=$0;',
    'gsub(/[[:cntrl:]]/, "", pid);',
    'getline;getline;getline;',
    'name=substr($0,1,31);',
    'gsub(/[[:cntrl:]]/, "", name);',
    'gsub(/^[[:space:]]+|[[:space:]]+$/, "", name);',
    'print pid,name}'
  )
  command <- paste(
    "unzip -p", shQuote(archive), shQuote(member),
    "| awk", shQuote(awk_program)
  )
  owners <- data.table::fread(
    cmd = command,
    sep = "|",
    header = FALSE,
    col.names = c("pid_line", "owner_name"),
    colClasses = "character",
    quote = "",
    fill = TRUE,
    showProgress = FALSE
  )
  owners[, quick_ref := sub(
    "^PID:[[:space:]]*([^[:space:]]+).*$",
    "\\1",
    pid_line,
    perl = TRUE
  )]
  owners[, owner_id := ifelse(
    grepl("[(]O[0-9]+[)]", pid_line),
    sub("^.*[(](O[0-9]+)[)].*$", "\\1", pid_line, perl = TRUE),
    NA_character_
  )]
  owners[, parcel_key := paste0("WILLIAMSON:", quick_ref)]
  owners <- owners[parcel_key %in% target_ids]
  summarise_owner_snapshot(
    owners,
    county = "Williamson",
    tax_year = tax_year,
    parcel_col = "parcel_key",
    owner_id_col = "owner_id",
    owner_name_col = "owner_name"
  )
}

parse_williamson_current_snapshot <- function(tax_year, target_ids) {
  owners <- data.table::fread(
    WILLIAMSON_OWNERS_FILE,
    select = c("QuickRefID", "OwnerID", "FullName", "PrimaryOwner"),
    colClasses = "character",
    showProgress = FALSE
  )
  owners <- owners[
    is.na(PrimaryOwner) | PrimaryOwner %in% c("1", "TRUE", "true")
  ]
  owners[, parcel_key := paste0(
    "WILLIAMSON:", blank_to_na(QuickRefID)
  )]
  owners <- owners[parcel_key %in% target_ids]
  summarise_owner_snapshot(
    owners,
    county = "Williamson",
    tax_year = tax_year,
    parcel_col = "parcel_key",
    owner_id_col = "OwnerID",
    owner_name_col = "FullName"
  )
}

print_progress("Building Hays and Williamson owner-snapshot transitions...")
hays_target_ids <- parcel_map[source_county == "Hays", parcel_id]
williamson_target_ids <- parcel_map[source_county == "Williamson", parcel_id]

hays_snapshots <- rbindlist(
  lapply(
    snapshot_years,
    parse_hays_owner_snapshot,
    target_ids = hays_target_ids
  ),
  use.names = TRUE,
  fill = TRUE
)
williamson_snapshots <- rbindlist(list(
  parse_williamson_report_snapshot(
    snapshot_years[[1]],
    williamson_target_ids
  ),
  parse_williamson_report_snapshot(
    snapshot_years[[2]],
    williamson_target_ids
  ),
  parse_williamson_current_snapshot(
    snapshot_years[[3]],
    williamson_target_ids
  )
), use.names = TRUE, fill = TRUE)

snapshot_data <- rbindlist(
  list(hays_snapshots, williamson_snapshots),
  use.names = TRUE,
  fill = TRUE
)
snapshot_grid <- rbindlist(list(
  data.table::CJ(
    source_county = "Hays",
    parcel_id = hays_target_ids,
    tax_year = snapshot_years,
    unique = TRUE
  ),
  data.table::CJ(
    source_county = "Williamson",
    parcel_id = williamson_target_ids,
    tax_year = snapshot_years,
    unique = TRUE
  )
))
snapshot_grid[snapshot_data, `:=`(
  owner_signature = i.owner_signature,
  corporate_owner = i.corporate_owner,
  owner_name_available = i.owner_name_available
), on = .(source_county, parcel_id, tax_year)]
setorder(snapshot_grid, source_county, parcel_id, tax_year)
snapshot_grid[, `:=`(
  prior_owner_signature = shift(owner_signature),
  prior_corporate_owner = shift(corporate_owner),
  prior_tax_year = shift(tax_year)
), by = .(source_county, parcel_id)]
snapshot_transitions <- snapshot_grid[
  !is.na(prior_tax_year) & tax_year == prior_tax_year + 1L
]
snapshot_transitions[, pair_covered :=
  !is.na(owner_signature) & !is.na(prior_owner_signature) &
    !is.na(corporate_owner) & !is.na(prior_corporate_owner)]
snapshot_transitions[, owner_changed :=
  pair_covered & owner_signature != prior_owner_signature]
snapshot_transitions[, corporate_acquisition :=
  owner_changed & corporate_owner]
snapshot_transitions[, corporate_disposition :=
  owner_changed & prior_corporate_owner]

snapshot_ownership_by_parcel <- snapshot_transitions[, .(
  ownership_source_covered = all(pair_covered),
  corporate_acquisition_source_covered = all(pair_covered),
  corporate_disposition_source_covered = all(pair_covered),
  ownership_change_recent_count = if (all(pair_covered)) {
    sum(owner_changed)
  } else {
    NA_integer_
  },
  corporate_acquisition_recent_count = if (all(pair_covered)) {
    sum(corporate_acquisition)
  } else {
    NA_integer_
  },
  corporate_disposition_recent_count = if (all(pair_covered)) {
    sum(corporate_disposition)
  } else {
    NA_integer_
  }
), by = .(source_county, parcel_id)]

travis_ownership_by_parcel <- transaction_events[
  source_county == "Travis" & analysis_window == "recent",
  .(
    ownership_source_covered = TRUE,
    ownership_change_recent_count = .N
  ),
  by = .(source_county, parcel_id)
]
travis_ownership_by_parcel[parcel_map, current_corporate_owned :=
  i.current_corporate_owned, on = .(source_county, parcel_id)]
travis_ownership_by_parcel[, `:=`(
  corporate_acquisition_source_covered = !is.na(current_corporate_owned),
  corporate_disposition_source_covered = FALSE,
  corporate_acquisition_recent_count = fifelse(
    !is.na(current_corporate_owned),
    as.integer(current_corporate_owned),
    NA_integer_
  ),
  corporate_disposition_recent_count = NA_integer_,
  current_corporate_owned = NULL
)]

ownership_by_parcel <- rbindlist(
  list(snapshot_ownership_by_parcel, travis_ownership_by_parcel),
  use.names = TRUE,
  fill = TRUE
)

################################################################################
# Complete parcel and hex summaries
################################################################################

print_progress("Aggregating transaction and ownership measures to hexagons...")
parcel_features <- copy(parcel_map)
parcel_features[transaction_by_parcel, `:=`(
  transaction_recent_count = i.transaction_recent_count,
  transaction_previous_count = i.transaction_previous_count
), on = .(source_county, parcel_id)]
parcel_features[, `:=`(
  transaction_recent_count = fcoalesce(transaction_recent_count, 0L),
  transaction_previous_count = fcoalesce(transaction_previous_count, 0L),
  transaction_source_covered = TRUE
)]

parcel_features[ownership_by_parcel, `:=`(
  ownership_source_covered = i.ownership_source_covered,
  corporate_acquisition_source_covered =
    i.corporate_acquisition_source_covered,
  corporate_disposition_source_covered =
    i.corporate_disposition_source_covered,
  ownership_change_recent_count = i.ownership_change_recent_count,
  corporate_acquisition_recent_count = i.corporate_acquisition_recent_count,
  corporate_disposition_recent_count = i.corporate_disposition_recent_count
), on = .(source_county, parcel_id)]
parcel_features[source_county == "Travis" & is.na(ownership_source_covered),
  `:=`(
    ownership_source_covered = TRUE,
    corporate_acquisition_source_covered =
      !is.na(current_corporate_owned),
    corporate_disposition_source_covered = FALSE,
    ownership_change_recent_count = 0L,
    corporate_acquisition_recent_count = fifelse(
      !is.na(current_corporate_owned),
      0L,
      NA_integer_
    ),
    corporate_disposition_recent_count = NA_integer_
  )]

parcel_features[, corporate_net_acquisition_source_covered :=
  corporate_acquisition_source_covered %in% TRUE &
    corporate_disposition_source_covered %in% TRUE]
parcel_features[, corporate_net_acquisition_recent_count := fifelse(
  corporate_net_acquisition_source_covered,
  corporate_acquisition_recent_count - corporate_disposition_recent_count,
  NA_integer_
)]

hex_summary <- parcel_features[, .(
  transaction_source_parcels = sum(transaction_source_covered),
  transaction_source_units = sum(
    residential_units[transaction_source_covered],
    na.rm = TRUE
  ),
  transaction_window_complete_parcels = sum(transaction_window_complete),
  transaction_recent_count = sum(transaction_recent_count, na.rm = TRUE),
  transaction_previous_count = sum(transaction_previous_count, na.rm = TRUE),
  transaction_recent_parcels = sum(transaction_recent_count > 0),
  transaction_previous_parcels = sum(transaction_previous_count > 0),
  transaction_recent_units_exposed = sum(
    residential_units[transaction_recent_count > 0],
    na.rm = TRUE
  ),
  transaction_previous_units_exposed = sum(
    residential_units[transaction_previous_count > 0],
    na.rm = TRUE
  ),
  ownership_source_parcels = sum(ownership_source_covered %in% TRUE),
  ownership_source_units = sum(
    residential_units[ownership_source_covered %in% TRUE],
    na.rm = TRUE
  ),
  corporate_acquisition_source_parcels = sum(
    corporate_acquisition_source_covered %in% TRUE
  ),
  corporate_acquisition_source_units = sum(
    residential_units[corporate_acquisition_source_covered %in% TRUE],
    na.rm = TRUE
  ),
  corporate_disposition_source_parcels = sum(
    corporate_disposition_source_covered %in% TRUE
  ),
  corporate_net_acquisition_source_parcels = sum(
    corporate_net_acquisition_source_covered %in% TRUE
  ),
  ownership_change_recent_count = sum(
    ownership_change_recent_count[ownership_source_covered %in% TRUE],
    na.rm = TRUE
  ),
  corporate_acquisition_recent_count = sum(
    corporate_acquisition_recent_count[
      corporate_acquisition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  ),
  corporate_disposition_recent_count = sum(
    corporate_disposition_recent_count[
      corporate_disposition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  ),
  corporate_net_acquisition_recent_count = sum(
    corporate_net_acquisition_recent_count[
      corporate_net_acquisition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  ),
  corporate_acquisition_recent_parcels = sum(
    corporate_acquisition_recent_count > 0 &
      corporate_acquisition_source_covered %in% TRUE,
    na.rm = TRUE
  ),
  corporate_acquisition_recent_units_exposed = sum(
    residential_units[
      corporate_acquisition_recent_count > 0 &
        corporate_acquisition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  )
), by = hex_id]

hex_summary[, transaction_window_coverage_pct :=
  100 * transaction_window_complete_parcels / transaction_source_parcels]
hex_summary[, transaction_window_complete :=
  transaction_window_complete_parcels == transaction_source_parcels]

hex_summary[, transaction_recent_per_100_parcels := fifelse(
  transaction_window_complete & transaction_source_parcels > 0,
  100 * transaction_recent_count / transaction_source_parcels,
  NA_real_
)]
hex_summary[, transaction_previous_per_100_parcels := fifelse(
  transaction_window_complete & transaction_source_parcels > 0,
  100 * transaction_previous_count / transaction_source_parcels,
  NA_real_
)]
hex_summary[, transaction_recent_per_100_units := fifelse(
  transaction_window_complete &
    transaction_source_units >= EWS_CONFIG$minimum_residential_units_for_rates,
  100 * transaction_recent_count / transaction_source_units,
  NA_real_
)]
hex_summary[, transaction_recent_unit_exposure_pct := fifelse(
  transaction_window_complete & transaction_source_units > 0,
  100 * transaction_recent_units_exposed / transaction_source_units,
  NA_real_
)]
hex_summary[, transaction_rate_change_per_100_parcels :=
  fifelse(
    transaction_window_complete,
    transaction_recent_per_100_parcels -
      transaction_previous_per_100_parcels,
    NA_real_
  )]
hex_summary[, transaction_log_count_change := fifelse(
  transaction_window_complete,
  log1p(transaction_recent_count) - log1p(transaction_previous_count),
  NA_real_
)]

hex_summary[, ownership_history_coverage_pct := fifelse(
  transaction_source_parcels > 0,
  100 * ownership_source_parcels / transaction_source_parcels,
  NA_real_
)]
hex_summary[, ownership_change_recent_per_100_parcels := fifelse(
  ownership_source_parcels > 0,
  100 * ownership_change_recent_count / ownership_source_parcels,
  NA_real_
)]
hex_summary[, corporate_acquisition_recent_per_100_parcels := fifelse(
  corporate_acquisition_source_parcels > 0,
  100 * corporate_acquisition_recent_count /
    corporate_acquisition_source_parcels,
  NA_real_
)]
hex_summary[, corporate_net_acquisition_recent_per_100_parcels := fifelse(
  corporate_net_acquisition_source_parcels > 0,
  100 * corporate_net_acquisition_recent_count /
    corporate_net_acquisition_source_parcels,
  NA_real_
)]
hex_summary[, corporate_acquisition_recent_share := fifelse(
  corporate_acquisition_source_parcels > 0 &
    ownership_change_recent_count > 0,
  corporate_acquisition_recent_count / ownership_change_recent_count,
  fifelse(corporate_acquisition_source_parcels > 0, 0, NA_real_)
)]
hex_summary[, corporate_acquisition_recent_unit_exposure_pct := fifelse(
  corporate_acquisition_source_units > 0,
  100 * corporate_acquisition_recent_units_exposed /
    corporate_acquisition_source_units,
  NA_real_
)]

hex_summary[, transaction_pressure_index := rowMeans(cbind(
  normalize_robust(transaction_recent_per_100_parcels),
  normalize_robust(transaction_recent_unit_exposure_pct),
  normalize_robust(pmax(transaction_rate_change_per_100_parcels, 0)),
  normalize_robust(pmax(transaction_log_count_change, 0))
), na.rm = FALSE)]
hex_summary[
  transaction_window_complete == FALSE,
  transaction_pressure_index := NA_real_
]
hex_summary[, ownership_change_index := rowMeans(cbind(
  normalize_robust(corporate_acquisition_recent_per_100_parcels),
  normalize_robust(corporate_acquisition_recent_unit_exposure_pct),
  normalize_robust(pmax(
    corporate_net_acquisition_recent_per_100_parcels,
    0
  )),
  normalize_robust(corporate_acquisition_recent_share)
), na.rm = TRUE)]

all_hexes <- as.data.table(st_drop_geometry(hex_grid))[, .(hex_id)]
all_hexes[, hex_id := as.character(hex_id)]
all_hexes <- merge(
  all_hexes,
  hex_summary,
  by = "hex_id",
  all.x = TRUE,
  sort = FALSE
)
all_hexes[, `:=`(
  transaction_analysis_as_of_date = analysis_as_of,
  transaction_recent_window_start = recent_start,
  transaction_previous_window_start = previous_start,
  ownership_snapshot_start_year = min(snapshot_years),
  ownership_snapshot_end_year = max(snapshot_years)
)]

################################################################################
# QA and outputs
################################################################################

source_qa <- parcel_features[, .(
  target_parcels = .N,
  target_units = sum(residential_units),
  transaction_source_parcels = sum(transaction_source_covered),
  transaction_window_complete_parcels = sum(transaction_window_complete),
  transaction_window_coverage_pct = 100 * mean(transaction_window_complete),
  parcels_with_recent_transaction = sum(transaction_recent_count > 0),
  recent_transaction_events = sum(transaction_recent_count),
  previous_transaction_events = sum(transaction_previous_count),
  ownership_source_parcels = sum(ownership_source_covered %in% TRUE),
  ownership_source_coverage_pct = 100 * mean(
    ownership_source_covered %in% TRUE
  ),
  corporate_acquisition_source_coverage_pct = 100 * mean(
    corporate_acquisition_source_covered %in% TRUE
  ),
  corporate_disposition_source_coverage_pct = 100 * mean(
    corporate_disposition_source_covered %in% TRUE
  ),
  ownership_change_events = sum(
    ownership_change_recent_count[ownership_source_covered %in% TRUE],
    na.rm = TRUE
  ),
  corporate_acquisition_events = sum(
    corporate_acquisition_recent_count[
      corporate_acquisition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  ),
  corporate_disposition_events = sum(
    corporate_disposition_recent_count[
      corporate_disposition_source_covered %in% TRUE
    ],
    na.rm = TRUE
  )
), by = source_county]
source_qa[, `:=`(
  pipeline_analysis_as_of_date = EWS_CONFIG$analysis_as_of_date,
  analysis_as_of_date = analysis_as_of,
  recent_window_start = recent_start,
  previous_window_start = previous_start,
  ownership_snapshot_years = paste(snapshot_years, collapse = ";"),
  transaction_window_status = fcase(
    source_county == "Williamson",
    "incomplete: certified history omits 2022-2023 and most of 2024",
    default = "complete comparable recent and previous windows"
  ),
  corporate_acquisition_method = fcase(
    source_county == "Travis",
    "recent market deed plus current corporate owner",
    default = "annual owner-snapshot transition"
  ),
  corporate_disposition_method = fcase(
    source_county == "Travis",
    "unavailable: recent deed party names are blank",
    default = "annual owner-snapshot transition"
  )
)]

if (nrow(parcel_map) != sum(parcel_target_qa$target_parcels)) {
  stop("Parcel denominator QA failed.", call. = FALSE)
}
if (any(source_qa$transaction_source_parcels != source_qa$target_parcels)) {
  stop("Transaction source does not cover the complete parcel universe.", call. = FALSE)
}
if (any(source_qa[
  source_county %in% c("Travis", "Hays"),
  transaction_window_coverage_pct
] < 100)) {
  stop("A complete Travis or Hays transaction window was marked incomplete.", call. = FALSE)
}
if (source_qa[
  source_county == "Williamson",
  transaction_window_coverage_pct
] != 0) {
  stop("Williamson's known transaction-history gap was not preserved.", call. = FALSE)
}
if (any(source_qa$ownership_source_coverage_pct < 95)) {
  stop("Ownership history coverage fell below 95% for a county.", call. = FALSE)
}
if (any(!is.finite(hex_summary[
  transaction_window_complete == TRUE,
  transaction_pressure_index
]))) {
  stop("Complete-window transaction indices contain non-finite values.", call. = FALSE)
}
if (any(!is.na(hex_summary[
  transaction_window_complete == FALSE,
  transaction_pressure_index
]))) {
  stop("Incomplete transaction windows must retain missing indices.", call. = FALSE)
}
if (any(!is.finite(hex_summary$ownership_change_index))) {
  stop("Ownership change index contains non-finite values.", call. = FALSE)
}

save_output(
  as_tibble(parcel_features),
  file.path(OUTPUT_DIR, "ownership_transaction_features_by_parcel.rds"),
  "ownership and transaction parcel features"
)
save_output(
  as_tibble(all_hexes),
  file.path(OUTPUT_DIR, "ownership_transaction_features_by_hex.rds"),
  "ownership and transaction hex features"
)
readr::write_csv(
  as_tibble(all_hexes),
  file.path(OUTPUT_DIR, "ownership_transaction_features_by_hex.csv")
)
readr::write_csv(
  as_tibble(source_qa),
  file.path(OUTPUT_DIR, "ownership_transaction_source_qa.csv")
)
readr::write_csv(
  as_tibble(event_type_qa),
  file.path(OUTPUT_DIR, "transaction_event_type_qa.csv")
)
readr::write_csv(
  as_tibble(transaction_source_year_qa),
  file.path(OUTPUT_DIR, "transaction_source_year_qa.csv")
)

cat("\nCounty source QA:\n")
print(as_tibble(source_qa))
cat("\nHex index summaries:\n")
print(as_tibble(hex_summary)[, c(
  "transaction_pressure_index",
  "ownership_change_index",
  "transaction_recent_per_100_parcels",
  "corporate_acquisition_recent_per_100_parcels",
  "ownership_history_coverage_pct"
)])
cat("\nOwnership and transaction features complete.\n")
