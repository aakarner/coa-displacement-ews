################################################################################
# Audit Historical Ownership and Transaction Sources
################################################################################
#
# Measures whether existing county appraisal and landlord-mapper extracts can
# support ownership-change and transaction-pressure features for the exact
# residential parcel universe used by the EWS. This script writes aggregate QA
# only; owner names and row-level deed records remain in ignored raw data.
#
# Optional environment variable:
#   EWS_LANDLORD_MAPPER_DIR=/path/to/landlord-mapper
#
# Outputs:
#   output/ownership_transaction_source_audit.csv
#   output/ownership_snapshot_coverage.csv
#   output/ownership_change_interval_audit.csv
#   output/transaction_history_coverage.csv
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
  library(readr)
  library(tibble)
})

print_header("02k - OWNERSHIP AND TRANSACTION SOURCE AUDIT")

OUTPUT_DIR <- project_path("output")
RAW_APPRAISAL_DIR <- project_path(
  "data", "raw_parcels", "appraisal_history"
)
TARGET_PARCELS_FILE <- file.path(
  OUTPUT_DIR,
  "residential_parcels_unit_targeted.rds"
)
WILLIAMSON_CURRENT_OWNERS_FILE <- project_path(
  "data", "raw_parcels", "williamson", "wcad_owners.csv"
)
WILLIAMSON_CURRENT_PROPERTY_FILE <- project_path(
  "data", "raw_parcels", "williamson", "wcad_property_certified.csv"
)
WILLIAMSON_CURRENT_SALES_FILE <- project_path(
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
LANDLORD_TRANSACTION_FILE <- file.path(
  LANDLORD_MAPPER_DIR,
  "output",
  "austin_parcel_year_land_transactions.csv"
)

required_files <- c(
  TARGET_PARCELS_FILE,
  WILLIAMSON_CURRENT_OWNERS_FILE,
  WILLIAMSON_CURRENT_PROPERTY_FILE,
  WILLIAMSON_CURRENT_SALES_FILE,
  LANDLORD_TRANSACTION_FILE
)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(
    "Ownership/transaction audit is missing required source file(s):\n- ",
    paste(missing_files, collapse = "\n- "),
    call. = FALSE
  )
}

years <- as.integer(EWS_CONFIG$appraisal_years)
recent_start_year <- min(years)
as_of_year <- as.integer(format(
  EWS_CONFIG$transaction_analysis_as_of_date,
  "%Y"
))

blank_to_na <- function(x) {
  x <- trimws(as.character(x))
  x[x == "" | toupper(x) == "NA"] <- NA_character_
  x
}

collapse_ids <- function(x) {
  x <- sort(unique(blank_to_na(x)))
  x <- x[!is.na(x)]
  if (length(x) == 0) NA_character_ else paste(x, collapse = ";")
}

first_existing <- function(x, candidates) {
  available <- if (is.data.frame(x) || data.table::is.data.table(x)) {
    names(x)
  } else {
    as.character(x)
  }
  hit <- candidates[candidates %in% available]
  if (length(hit) == 0) NA_character_ else hit[[1]]
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

locate_nested_member <- function(outer_archive, nested_pattern) {
  listing <- utils::unzip(outer_archive, list = TRUE)
  nested <- listing$Name[
    grepl("[.]zip$", listing$Name, ignore.case = TRUE) &
      grepl(nested_pattern, basename(listing$Name), ignore.case = TRUE)
  ]
  if (length(nested) == 0) {
    stop(
      "No nested member matching ", nested_pattern,
      " in ", outer_archive,
      call. = FALSE
    )
  }

  inner_archive <- extract_nested_zip(outer_archive, nested[[1]])
  inner_listing <- utils::unzip(inner_archive, list = TRUE)
  data_members <- inner_listing$Name[
    grepl("[.](txt|csv)$", inner_listing$Name, ignore.case = TRUE)
  ]
  if (length(data_members) == 0) {
    stop("No tabular member found in ", nested[[1]], call. = FALSE)
  }

  list(archive = inner_archive, member = data_members[[1]])
}

summarise_owner_rows <- function(
    data,
    county,
    tax_year,
    source,
    parcel_col,
    owner_id_col,
    owner_name_col = NA_character_) {
  data <- as.data.table(data)
  data[, parcel_id := blank_to_na(get(parcel_col))]
  data[, owner_key := blank_to_na(get(owner_id_col))]
  if (!is.na(owner_name_col) && owner_name_col %in% names(data)) {
    data[, owner_name := blank_to_na(get(owner_name_col))]
  } else {
    data[, owner_name := NA_character_]
  }
  data[is.na(owner_key), owner_key := owner_name]
  data <- data[!is.na(parcel_id)]

  data[, .(
    owner_signature = collapse_ids(owner_key),
    owner_id_available = any(!is.na(owner_key)),
    owner_name_available = any(!is.na(owner_name))
  ), by = parcel_id][, `:=`(
    source_county = county,
    tax_year = as.integer(tax_year),
    source = source
  )]
}

parse_hays_owner_year <- function(tax_year, target_ids) {
  outer_archive <- file.path(
    RAW_APPRAISAL_DIR,
    "hays",
    as.character(tax_year),
    paste0("hays_", tax_year, ".zip")
  )
  if (!file.exists(outer_archive)) {
    stop("Missing Hays annual archive: ", outer_archive, call. = FALSE)
  }

  location <- locate_nested_member(outer_archive, "OWNER")
  header <- names(read_zip_table(
    location$archive,
    location$member,
    nrows = 0
  ))
  quick_col <- first_existing(header, c("QuickRefID", "Quick Ref ID"))
  owner_id_col <- first_existing(
    header,
    c("OwnerID", "OwnerQuickRefID", "OwnerPropertyNumber")
  )
  owner_name_col <- first_existing(header, c("OwnerName", "Owner Name"))
  if (is.na(quick_col) || is.na(owner_id_col)) {
    stop("Hays owner schema is missing parcel or owner identifiers.", call. = FALSE)
  }

  selected <- unique(na.omit(c(quick_col, owner_id_col, owner_name_col)))
  owners <- read_zip_table(
    location$archive,
    location$member,
    select = selected
  )
  owners[, parcel_key := paste0("HAYS:", blank_to_na(get(quick_col)))]
  owners <- owners[parcel_key %in% target_ids]

  source_label <- if (tax_year == 2022L) {
    "Hays supplemental annual OWNER export"
  } else {
    "Hays annual OWNER export"
  }
  summarise_owner_rows(
    owners,
    county = "Hays",
    tax_year = tax_year,
    source = source_label,
    parcel_col = "parcel_key",
    owner_id_col = owner_id_col,
    owner_name_col = owner_name_col
  )
}

parse_williamson_report_owner_year <- function(tax_year, target_ids) {
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
  owners[, owner_quick_ref := ifelse(
    grepl("[(]O[0-9]+[)]", pid_line),
    sub("^.*[(](O[0-9]+)[)].*$", "\\1", pid_line, perl = TRUE),
    NA_character_
  )]
  owners[, parcel_key := paste0("WILLIAMSON:", quick_ref)]
  owners <- owners[parcel_key %in% target_ids]

  summarise_owner_rows(
    owners,
    county = "Williamson",
    tax_year = tax_year,
    source = "Williamson certified roll report",
    parcel_col = "parcel_key",
    owner_id_col = "owner_quick_ref",
    owner_name_col = "owner_name"
  )
}

parse_williamson_current_owners <- function(tax_year, target_ids) {
  owners <- data.table::fread(
    WILLIAMSON_CURRENT_OWNERS_FILE,
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

  summarise_owner_rows(
    owners,
    county = "Williamson",
    tax_year = tax_year,
    source = "Williamson current certified owners",
    parcel_col = "parcel_key",
    owner_id_col = "OwnerID",
    owner_name_col = "FullName"
  )
}

print_progress("Loading the residential parcel analysis universe...")
target <- as.data.table(readRDS(TARGET_PARCELS_FILE))
target <- target[, .(
  parcel_id = as.character(parcel_id),
  source_county = as.character(source_county),
  owner_names = as.character(owner_names)
)]
target[, transaction_key := fifelse(
  source_county == "Travis",
  paste0("TRAVIS:", parcel_id),
  parcel_id
)]

target_counts <- target[, .(target_parcels = .N), by = source_county]
hays_target_ids <- target[source_county == "Hays", parcel_id]
williamson_target_ids <- target[source_county == "Williamson", parcel_id]

################################################################################
# Annual owner snapshots
################################################################################

print_progress("Auditing Hays annual owner snapshots...")
hays_snapshots <- rbindlist(
  lapply(years, parse_hays_owner_year, target_ids = hays_target_ids),
  use.names = TRUE,
  fill = TRUE
)

print_progress("Auditing Williamson annual owner snapshots...")
williamson_snapshots <- rbindlist(list(
  parse_williamson_report_owner_year(2023L, williamson_target_ids),
  parse_williamson_report_owner_year(2024L, williamson_target_ids),
  parse_williamson_current_owners(2025L, williamson_target_ids)
), use.names = TRUE, fill = TRUE)

travis_current <- target[source_county == "Travis", .(
  parcel_id,
  owner_signature = blank_to_na(owner_names),
  owner_id_available = FALSE,
  owner_name_available = !is.na(blank_to_na(owner_names)),
  source_county,
  tax_year = as.integer(EWS_CONFIG$appraisal_current_year),
  source = "Travis current owner extract"
)]

owner_snapshots <- rbindlist(
  list(hays_snapshots, williamson_snapshots, travis_current),
  use.names = TRUE,
  fill = TRUE
)
setcolorder(
  owner_snapshots,
  c(
    "source_county", "tax_year", "parcel_id", "owner_signature",
    "owner_id_available", "owner_name_available", "source"
  )
)

owner_coverage <- owner_snapshots[, .(
  parcels_with_owner_rows = uniqueN(parcel_id),
  parcels_with_owner_id = uniqueN(parcel_id[owner_id_available]),
  parcels_with_owner_name = uniqueN(parcel_id[owner_name_available]),
  source = paste(sort(unique(source)), collapse = "; ")
), by = .(source_county, tax_year)]
owner_coverage[target_counts, target_parcels := i.target_parcels,
  on = "source_county"]

williamson_unavailable_years <- data.table(
  source_county = "Williamson",
  tax_year = c(2021L, 2022L),
  parcels_with_owner_rows = 0L,
  parcels_with_owner_id = 0L,
  parcels_with_owner_name = 0L,
  source = paste0(
    "Unavailable: ASMNT OwnerQuickRefID duplicates the parcel identifier"
  ),
  target_parcels = length(williamson_target_ids)
)
owner_coverage <- rbindlist(
  list(owner_coverage, williamson_unavailable_years),
  use.names = TRUE,
  fill = TRUE
)
owner_coverage[, owner_row_coverage_pct :=
  100 * parcels_with_owner_rows / target_parcels]
owner_coverage[, owner_id_coverage_pct :=
  100 * parcels_with_owner_id / target_parcels]
setorder(owner_coverage, source_county, tax_year)

change_counties <- c("Hays", "Williamson")
change_grid <- rbindlist(lapply(change_counties, function(county) {
  data.table::CJ(
    source_county = county,
    parcel_id = target[source_county == county, parcel_id],
    tax_year = years,
    unique = TRUE
  )
}))
change_grid[owner_snapshots, `:=`(
  owner_signature = i.owner_signature,
  owner_id_available = i.owner_id_available,
  owner_name_available = i.owner_name_available
), on = .(source_county, parcel_id, tax_year)]
setorder(change_grid, source_county, parcel_id, tax_year)
change_grid[, `:=`(
  prior_tax_year = shift(tax_year),
  prior_owner_signature = shift(owner_signature)
), by = .(source_county, parcel_id)]
change_grid <- change_grid[
  !is.na(prior_tax_year) & tax_year == prior_tax_year + 1L
]
change_grid[, comparable :=
  !is.na(owner_signature) & !is.na(prior_owner_signature)]
change_grid[, owner_changed :=
  comparable & owner_signature != prior_owner_signature]

owner_change_audit <- change_grid[, .(
  target_parcels = .N,
  parcels_with_both_snapshots = sum(comparable),
  snapshot_pair_coverage_pct = 100 * mean(comparable),
  parcels_with_owner_change = sum(owner_changed),
  owner_change_share_pct = if (sum(comparable) > 0) {
    100 * sum(owner_changed) / sum(comparable)
  } else {
    NA_real_
  }
), by = .(
  source_county,
  prior_tax_year,
  current_tax_year = tax_year
)]
setorder(owner_change_audit, source_county, prior_tax_year)

################################################################################
# Transaction event coverage
################################################################################

print_progress("Auditing cached Travis and Hays transaction events...")
transactions <- data.table::fread(
  LANDLORD_TRANSACTION_FILE,
  select = c(
    "county", "parcel_id", "transaction_year", "transaction_count",
    "corporate_buyer_transaction_count",
    "corporate_seller_transaction_count", "transaction_source"
  ),
  colClasses = "character",
  showProgress = FALSE
)
transactions[, transaction_year := suppressWarnings(as.integer(
  transaction_year
))]
transactions[, transaction_count := suppressWarnings(as.integer(
  transaction_count
))]
transactions <- transactions[
  county %in% c("Travis", "Hays") &
    parcel_id %in% target$transaction_key
]
transactions <- transactions[
  is.na(transaction_year) | transaction_year <= as_of_year
]

transaction_coverage_existing <- transactions[, .(
  target_parcels = target_counts[
    match(.BY$county, source_county),
    target_parcels
  ],
  parcels_represented = uniqueN(parcel_id),
  parcels_with_any_transaction = uniqueN(
    parcel_id[!is.na(transaction_year) & transaction_count > 0]
  ),
  parcels_with_recent_transaction = uniqueN(
    parcel_id[
      !is.na(transaction_year) &
        transaction_year >= recent_start_year &
        transaction_count > 0
    ]
  ),
  recent_transaction_events = sum(
    transaction_count[
      !is.na(transaction_year) & transaction_year >= recent_start_year
    ],
    na.rm = TRUE
  ),
  minimum_transaction_year = if (any(!is.na(transaction_year))) {
    min(transaction_year, na.rm = TRUE)
  } else {
    NA_integer_
  },
  maximum_transaction_year = if (any(!is.na(transaction_year))) {
    max(transaction_year, na.rm = TRUE)
  } else {
    NA_integer_
  },
  buyer_name_available = FALSE,
  seller_name_available = .BY$county == "Hays",
  comparable_transaction_window_complete = TRUE,
  source = paste(sort(unique(transaction_source)), collapse = "; ")
), by = county]

print_progress("Auditing current Williamson certified sales history...")
williamson_property <- data.table::fread(
  WILLIAMSON_CURRENT_PROPERTY_FILE,
  select = c("PropertyID", "QuickRefID"),
  colClasses = "character",
  showProgress = FALSE
)
williamson_sales <- data.table::fread(
  WILLIAMSON_CURRENT_SALES_FILE,
  select = c(
    "PropertyID", "SaleDate", "TransferValidityCode",
    "OwnershipTransferID"
  ),
  colClasses = "character",
  showProgress = FALSE
)
williamson_sales[williamson_property, QuickRefID := i.QuickRefID,
  on = "PropertyID"]
williamson_sales[, parcel_id := paste0(
  "WILLIAMSON:", blank_to_na(QuickRefID)
)]
williamson_sales[, transaction_date := as.IDate(
  SaleDate,
  format = "%m/%d/%Y"
)]
williamson_sales <- williamson_sales[
  parcel_id %in% williamson_target_ids &
    !is.na(transaction_date) &
    transaction_date <= as.IDate(EWS_CONFIG$analysis_as_of_date) &
    (is.na(TransferValidityCode) | TransferValidityCode == "VALID")
]
williamson_sales[, transaction_year := as.integer(format(
  transaction_date,
  "%Y"
))]

williamson_tx_coverage <- williamson_sales[, .(
  county = "Williamson",
  target_parcels = length(williamson_target_ids),
  parcels_represented = length(williamson_target_ids),
  parcels_with_any_transaction = uniqueN(parcel_id),
  parcels_with_recent_transaction = uniqueN(
    parcel_id[transaction_year >= recent_start_year]
  ),
  recent_transaction_events = sum(transaction_year >= recent_start_year),
  minimum_transaction_year = min(transaction_year, na.rm = TRUE),
  maximum_transaction_year = max(transaction_year, na.rm = TRUE),
  buyer_name_available = FALSE,
  seller_name_available = FALSE,
  comparable_transaction_window_complete = FALSE,
  source = "WCAD Sales History - Certified (kdj3-9hpg)"
)]

transaction_coverage <- rbindlist(
  list(transaction_coverage_existing, williamson_tx_coverage),
  use.names = TRUE,
  fill = TRUE
)
transaction_coverage[, any_transaction_parcel_coverage_pct :=
  100 * parcels_with_any_transaction / target_parcels]
transaction_coverage[, recent_transaction_parcel_share_pct :=
  100 * parcels_with_recent_transaction / target_parcels]
setorder(transaction_coverage, county)

################################################################################
# Source-level decision audit
################################################################################

source_audit <- tribble(
  ~source_county, ~domain, ~local_source, ~coverage_status,
  ~main_limitation, ~recommended_use,
  "Travis", "ownership_change",
  "Current owners plus TCAD deed events",
  "proxy_only",
  "Recent deed buyer and seller names are blank and no annual owner snapshots are available",
  "Infer corporate entry only for recent market-deed parcels currently corporate-owned; leave dispositions unavailable",
  "Travis", "transaction_pressure",
  "Cached TCAD deeds extract",
  "ready",
  "Deed records include non-market transfers that require type filtering",
  "Build normalized recent deed-volume and valid-transfer measures",
  "Hays", "ownership_change",
  "Annual OWNER exports, 2021-2025",
  "ready_2023_2025",
  "The comparable feature window begins after the supplementary 2022 export",
  "Build direct owner-change and corporate-entry flags from 2023-2025",
  "Hays", "transaction_pressure",
  "2025 SALES history export",
  "ready_with_caveat",
  "Prior owner is present but buyer name is absent",
  "Build transaction volume; infer corporate entry from annual owner snapshots",
  "Williamson", "ownership_change",
  "2023-2024 certified reports and 2025 current owners",
  "ready_2023_2025",
  "The 2021-2022 ASMNT OwnerQuickRefID field duplicates the parcel ID and is not an owner identity",
  "Build direct 2023-2025 owner-change and corporate-entry flags",
  "Williamson", "transaction_pressure",
  "WCAD Sales History - Certified (kdj3-9hpg)",
  "incomplete",
  "The current table omits 2022-2023 and nearly all 2024 sales",
  "Leave transaction pressure missing until a complete comparable history is acquired"
) %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

readr::write_csv(
  as_tibble(owner_coverage),
  file.path(OUTPUT_DIR, "ownership_snapshot_coverage.csv")
)
readr::write_csv(
  as_tibble(owner_change_audit),
  file.path(OUTPUT_DIR, "ownership_change_interval_audit.csv")
)
readr::write_csv(
  as_tibble(transaction_coverage),
  file.path(OUTPUT_DIR, "transaction_history_coverage.csv")
)
readr::write_csv(
  source_audit,
  file.path(OUTPUT_DIR, "ownership_transaction_source_audit.csv")
)

cat("\nOwner snapshot coverage:\n")
print(as_tibble(owner_coverage))
cat("\nOwner change interval audit:\n")
print(as_tibble(owner_change_audit))
cat("\nTransaction history coverage:\n")
print(as_tibble(transaction_coverage))
cat("\nOwnership and transaction source audit complete.\n")
