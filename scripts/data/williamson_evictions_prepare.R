################################################################################
# Prepare Williamson County JP1/JP2 Eviction Records for Part 3
################################################################################
#
# The Williamson exports use two report layouts. JP1 stores address continuation
# lines as separate physical spreadsheet rows; JP2 uses a seven-row report
# preamble and contains a small number of literal duplicate rows. This stage
# preserves party/address evidence while creating county-and-court namespaced
# case identifiers for case-level counting.
################################################################################

required_packages <- c(
  "digest", "dplyr", "janitor", "lubridate", "readr", "readxl", "stringr"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Install missing package(s) before preparing Williamson evictions: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

source(here::here("R", "utils.R"))
source(here::here("R", "analysis_config.R"))
source(here::here("R", "williamson_eviction_ingest.R"))

print_header("PREPARE WILLIAMSON EVICTION ADDRESSES")

OUTPUT_DIR <- here::here("output")
SOURCE_CONFIG_FILE <- here::here("config", "eviction_sources.csv")
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(SOURCE_CONFIG_FILE)) {
  stop("Missing eviction source configuration: ", SOURCE_CONFIG_FILE, call. = FALSE)
}

source_config <- read_csv(
  SOURCE_CONFIG_FILE,
  col_types = cols(
    source_id = col_character(),
    source_county = col_character(),
    jp_district = col_character(),
    path = col_character(),
    sheet = col_character(),
    parser = col_character(),
    source_period_start = col_date(),
    source_period_end = col_date(),
    coverage_geography = col_character(),
    notes = col_character()
  ),
  show_col_types = FALSE
)

print_progress("Reading and normalizing the Williamson JP1/JP2 workbooks...")
prepared <- williamson_prepare_eviction_records(
  source_config,
  analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
)

source_period_check <- prepared |>
  group_by(.data$source_id) |>
  summarize(
    records = n(),
    min_file_date = min(.data$file_date),
    max_file_date = max(.data$file_date),
    declared_start = first(.data$source_period_start),
    declared_end = first(.data$source_period_end),
    dates_within_declared_period = all(
      .data$file_date >= .data$source_period_start &
        .data$file_date <= .data$source_period_end
    ),
    .groups = "drop"
  )
if (any(!source_period_check$dates_within_declared_period)) {
  stop(
    "At least one Williamson filing date falls outside its configured source period.",
    call. = FALSE
  )
}

canonical_case_check <- prepared |>
  filter(!is.na(.data$case_uid)) |>
  group_by(.data$case_uid) |>
  summarize(
    filing_dates = n_distinct(.data$file_date),
    source_jps = n_distinct(.data$jp_district),
    .groups = "drop"
  )
if (any(canonical_case_check$filing_dates != 1L) ||
    any(canonical_case_check$source_jps != 1L)) {
  stop(
    "A canonical Williamson case has conflicting dates or court provenance.",
    call. = FALSE
  )
}

unique_addresses <- prepared |>
  filter(!.data$missing_address) |>
  group_by(.data$address_for_geocoding) |>
  summarize(
    filing_party_rows = n(),
    canonical_cases = n_distinct(.data$case_uid, na.rm = TRUE),
    geocoding_candidate = any(.data$geocoding_candidate),
    po_box_address = any(.data$po_box_address),
    has_tx_zip = any(.data$has_tx_zip),
    has_placeholder_zip = any(.data$has_placeholder_zip),
    has_house_number = any(.data$has_house_number),
    invalid_placeholder_address = any(.data$invalid_placeholder_address),
    likely_out_of_state = any(.data$likely_out_of_state),
    jp_districts = paste(
      sort(unique(stats::na.omit(.data$jp_district))),
      collapse = "; "
    ),
    .groups = "drop"
  ) |>
  arrange(
    desc(.data$geocoding_candidate),
    desc(.data$filing_party_rows),
    .data$address_for_geocoding
  ) |>
  mutate(address_id = row_number(), .before = 1)

candidate_addresses <- unique_addresses |>
  filter(.data$geocoding_candidate) |>
  pull(.data$address_for_geocoding)

qc_summary <- tibble::tibble(
  metric = c(
    "source_files",
    "logical_party_rows",
    "canonical_cases",
    "nonstandard_case_number_rows_excluded_from_case_inventory",
    "source_exact_duplicate_rows",
    "jp1_address_continuation_rows_folded",
    "rows_after_analysis_cutoff",
    "missing_addresses",
    "po_box_addresses",
    "placeholder_addresses",
    "likely_out_of_state_addresses",
    "all_unique_cleaned_addresses",
    "unique_geocoding_candidate_addresses",
    "min_file_date",
    "max_file_date"
  ),
  value = c(
    as.character(c(
    n_distinct(prepared$source_file),
    nrow(prepared),
    n_distinct(prepared$case_uid, na.rm = TRUE),
    sum(!prepared$case_number_format_valid),
    sum(prepared$source_exact_duplicate),
    sum(prepared$continuation_rows),
    sum(prepared$after_analysis_cutoff),
    sum(prepared$missing_address),
    sum(prepared$po_box_address),
    sum(prepared$invalid_placeholder_address),
    sum(prepared$likely_out_of_state),
    n_distinct(prepared$address_for_geocoding, na.rm = TRUE),
    length(candidate_addresses)
    )),
    as.character(min(prepared$file_date)),
    as.character(max(prepared$file_date))
  )
)

qc_by_source <- prepared |>
  group_by(
    .data$source_id,
    .data$source_county,
    .data$source_file,
    .data$jp_district
  ) |>
  summarize(
    logical_party_rows = n(),
    canonical_cases = n_distinct(.data$case_uid, na.rm = TRUE),
    nonstandard_case_number_rows = sum(!.data$case_number_format_valid),
    source_exact_duplicate_rows = sum(.data$source_exact_duplicate),
    address_continuation_rows_folded = sum(.data$continuation_rows),
    rows_after_analysis_cutoff = sum(.data$after_analysis_cutoff),
    missing_addresses = sum(.data$missing_address),
    geocoding_candidate_addresses = n_distinct(
      .data$address_for_geocoding[.data$geocoding_candidate],
      na.rm = TRUE
    ),
    min_file_date = min(.data$file_date),
    max_file_date = max(.data$file_date),
    .groups = "drop"
  ) |>
  arrange(.data$jp_district, .data$min_file_date)

ingest_issues <- prepared |>
  filter(!is.na(.data$ingest_issue)) |>
  select(
    "source_id", "source_file", "source_row_start", "source_row_end",
    "source_record_id", "source_record_hash", "jp_district", "file_date",
    "case_number_format_valid", "source_exact_duplicate", "missing_address",
    "ingest_issue"
  )

print_progress("Writing ignored, address-bearing working outputs and non-sensitive QA...")
write_csv(
  prepared,
  file.path(OUTPUT_DIR, "williamson_eviction_filings_prepared_for_geocoding.csv"),
  na = ""
)
write_csv(
  unique_addresses,
  file.path(OUTPUT_DIR, "williamson_eviction_unique_addresses_for_geocoding.csv"),
  na = ""
)
write_csv(
  qc_summary,
  file.path(OUTPUT_DIR, "williamson_eviction_address_qc_summary.csv"),
  na = ""
)
write_csv(
  qc_by_source,
  file.path(OUTPUT_DIR, "williamson_eviction_address_qc_by_source.csv"),
  na = ""
)
write_csv(
  source_period_check,
  file.path(OUTPUT_DIR, "williamson_eviction_source_period_qa.csv"),
  na = ""
)
write_csv(
  ingest_issues,
  file.path(OUTPUT_DIR, "williamson_eviction_ingest_issues.csv"),
  na = ""
)

print_progress(paste0("Prepared logical party rows: ", nrow(prepared)))
print_progress(paste0(
  "Canonical county/court/case IDs: ",
  n_distinct(prepared$case_uid, na.rm = TRUE)
))
print_progress(paste0(
  "Unique Williamson addresses requiring a geocode record: ",
  length(candidate_addresses)
))
print_header("WILLIAMSON EVICTION PREPARATION COMPLETE")
