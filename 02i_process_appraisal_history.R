################################################################################
# 02i - Process Historical County Appraisal Values
################################################################################
#
# Builds a 2021-2025 parcel-year panel for the current Austin residential parcel
# universe, then aggregates the values and inflation-adjusted trends to hexagons.
# Large source archives and normalized county-year extracts are cached below
# data/raw_parcels/appraisal_history/.
#
# Optional environment variables:
#   EWS_APPRAISAL_YEARS=2021,2022,2023,2024,2025
#   EWS_APPRAISAL_CURRENT_YEAR=2025
#   EWS_APPRAISAL_COUNTIES=Travis,Hays,Williamson
#   EWS_APPRAISAL_REFRESH=true
#   EWS_APPRAISAL_DOWNLOAD_OPTIONAL=false
#
# Outputs:
#   - output/appraisal_values_by_parcel_year.rds/.csv
#   - output/appraisal_values_by_hex_year.rds/.csv
#   - output/appraisal_value_trends_by_hex.rds/.csv
#   - output/appraisal_county_land_values_by_account_year.rds
#   - output/appraisal_panel_source_qa.csv
#   - output/appraisal_panel_spatial_qa.csv
#   - output/appraisal_hays_2022_source_comparison.csv
#   - output/appraisal_hays_2022_source_comparison_qa.csv
#
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
  library(sf)
  library(tidyr)
})

print_header("02i - HISTORICAL APPRAISAL VALUES")

OUTPUT_DIR <- project_path("output")
RAW_DIR <- project_path("data", "raw_parcels", "appraisal_history")
SOURCE_MANIFEST <- project_path("config", "appraisal_sources.csv")
PARSER_VERSIONS <- c(
  travis_ears_nested_zip = "3",
  hays_property_export_zip = "4",
  hays_certified_fixed_zip = "2",
  williamson_certified_report_zip = "3",
  williamson_current_csv = "4"
)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(RAW_DIR, showWarnings = FALSE, recursive = TRUE)

env_flag <- function(name, default = FALSE) {
  value <- tolower(trimws(Sys.getenv(name, unset = as.character(default))))
  value %in% c("1", "true", "t", "yes", "y")
}

parse_county_list <- function(value, default) {
  if (!nzchar(value)) return(default)
  values <- trimws(strsplit(value, ",", fixed = TRUE)[[1]])
  values[nzchar(values)]
}

REFRESH <- env_flag("EWS_APPRAISAL_REFRESH", FALSE)
DOWNLOAD_OPTIONAL <- env_flag("EWS_APPRAISAL_DOWNLOAD_OPTIONAL", FALSE)
SELECTED_COUNTIES <- parse_county_list(
  Sys.getenv("EWS_APPRAISAL_COUNTIES", unset = ""),
  c("Travis", "Hays", "Williamson")
)

sum_or_na <- function(x) {
  if (length(x) == 0 || all(is.na(x))) NA_real_ else sum(x, na.rm = TRUE)
}

max_or_na <- function(x) {
  if (length(x) == 0 || all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)
}

first_or_na <- function(x) {
  x <- x[!is.na(x) & nzchar(trimws(as.character(x)))]
  if (length(x) == 0) NA_character_ else as.character(x[[1]])
}

collapse_values <- function(x) {
  x <- sort(unique(trimws(as.character(x))))
  x <- x[!is.na(x) & nzchar(x)]
  if (length(x) == 0) NA_character_ else paste(x, collapse = "|")
}

empty_normalized_values <- function() {
  tibble(
    parcel_id = character(),
    source_county = character(),
    tax_year = integer(),
    source_account_id = character(),
    source_long_account_id = character(),
    category_codes = character(),
    situs_address_year = character(),
    land_sqft_year = double(),
    improvement_sqft_year = double(),
    land_market_value = double(),
    improvement_market_value = double(),
    total_market_value = double(),
    assessed_value = double(),
    source_previous_total_market_value = double(),
    data_source = character()
  )
}

empty_county_values <- function() {
  tibble(
    source_account_id = character(),
    land_market_value = double(),
    improvement_market_value = double(),
    source_county = character(),
    tax_year = integer()
  )
}

as_value <- function(x) {
  suppressWarnings(as.numeric(gsub("[^0-9.-]", "", as.character(x))))
}

normalize_account <- function(x) {
  x <- trimws(as.character(x))
  x <- sub("^0+([0-9])", "\\1", x)
  x[x %in% c("", "NA", "NULL")] <- NA_character_
  x
}

pick_column <- function(data, candidates, required = FALSE) {
  normalized_names <- tolower(gsub("[^a-z0-9]", "", names(data)))
  normalized_candidates <- tolower(gsub("[^a-z0-9]", "", candidates))
  match_index <- match(normalized_candidates, normalized_names, nomatch = 0L)
  match_index <- match_index[match_index > 0L]

  if (length(match_index) == 0) {
    if (required) {
      stop(
        "Required column not found. Tried: ",
        paste(candidates, collapse = ", "),
        call. = FALSE
      )
    }
    return(rep(NA_character_, nrow(data)))
  }

  data[[match_index[[1]]]]
}

target_signature <- function(ids) {
  path <- tempfile(fileext = ".txt")
  on.exit(unlink(path), add = TRUE)
  writeLines(sort(unique(ids)), path, useBytes = TRUE)
  unname(tools::md5sum(path))
}

source_signature <- function(path) {
  info <- file.info(path)
  paste(info$size, as.numeric(info$mtime), sep = ":")
}

download_source <- function(url, destination) {
  if (file.exists(destination) && !REFRESH) return(destination)
  if (!nzchar(url)) stop("No URL configured for ", destination, call. = FALSE)

  dir.create(dirname(destination), showWarnings = FALSE, recursive = TRUE)
  partial <- paste0(destination, ".part")
  if (file.exists(partial)) unlink(partial)

  print_progress(paste0("Downloading ", basename(destination), "..."))
  old_timeout <- getOption("timeout")
  on.exit(options(timeout = old_timeout), add = TRUE)
  options(timeout = max(7200, old_timeout))
  utils::download.file(url, partial, mode = "wb", method = "libcurl")

  if (!file.rename(partial, destination)) {
    stop("Could not move completed download to: ", destination, call. = FALSE)
  }
  destination
}

archive_member <- function(archive, pattern = "\\.(txt|csv)$", largest = TRUE) {
  members <- utils::unzip(archive, list = TRUE)
  candidates <- members[grepl(pattern, members$Name, ignore.case = TRUE), , drop = FALSE]
  candidates <- candidates[!grepl("(^|/)__MACOSX/", candidates$Name), , drop = FALSE]
  if (nrow(candidates) == 0) {
    stop("No matching data member found in ", archive, call. = FALSE)
  }
  if (largest) candidates <- candidates[order(candidates$Length, decreasing = TRUE), ]
  candidates$Name[[1]]
}

extract_nested_zip <- function(outer_archive, extract_dir, preferred_pattern = NULL) {
  members <- utils::unzip(outer_archive, list = TRUE)
  candidates <- members[grepl("\\.zip$", members$Name, ignore.case = TRUE), , drop = FALSE]
  candidates <- candidates[!grepl("(^|/)__MACOSX/", candidates$Name), , drop = FALSE]
  if (!is.null(preferred_pattern)) {
    preferred <- candidates[grepl(preferred_pattern, basename(candidates$Name), ignore.case = TRUE), , drop = FALSE]
    if (nrow(preferred) > 0) candidates <- preferred
  }
  if (nrow(candidates) == 0) {
    stop("No nested ZIP found in ", outer_archive, call. = FALSE)
  }

  candidates <- candidates[order(candidates$Length, decreasing = TRUE), ]
  member <- candidates$Name[[1]]
  destination <- file.path(extract_dir, member)
  if (!file.exists(destination) || REFRESH) {
    dir.create(dirname(destination), showWarnings = FALSE, recursive = TRUE)
    utils::unzip(outer_archive, files = member, exdir = extract_dir, overwrite = TRUE)
  }
  destination
}

read_zipped_table <- function(archive, member) {
  command <- paste("unzip -p", shQuote(archive), shQuote(member))
  data.table::fread(
    cmd = command,
    header = TRUE,
    colClasses = "character",
    fill = TRUE,
    showProgress = FALSE,
    data.table = FALSE
  )
}

locate_hays_property_table <- function(archive, extract_dir) {
  members <- utils::unzip(archive, list = TRUE)
  direct <- members[grepl("\\.(txt|csv)$", members$Name, ignore.case = TRUE), , drop = FALSE]
  direct <- direct[grepl("property", basename(direct$Name), ignore.case = TRUE), , drop = FALSE]
  direct <- direct[
    !grepl("owner|improvement|land|sales|segment", basename(direct$Name), ignore.case = TRUE),
    , drop = FALSE
  ]
  if (nrow(direct) > 0) {
    direct <- direct[order(direct$Length, decreasing = TRUE), ]
    return(list(archive = archive, member = direct$Name[[1]]))
  }

  nested_members <- members[grepl("\\.zip$", members$Name, ignore.case = TRUE), , drop = FALSE]
  nested_names <- basename(nested_members$Name)
  nested_candidates <- nested_members[
    grepl("property", nested_names, ignore.case = TRUE) &
      !grepl(
        "owner|improvement|land|sales|segment|segement",
        nested_names,
        ignore.case = TRUE
      ),
    , drop = FALSE
  ]
  if (nrow(nested_candidates) == 0) {
    stop("No nested Hays property ZIP found in ", archive, call. = FALSE)
  }
  nested_candidates <- nested_candidates[
    order(nested_candidates$Length, decreasing = TRUE),
  ]
  nested_member <- nested_candidates$Name[[1]]
  nested <- file.path(extract_dir, nested_member)
  if (!file.exists(nested) || REFRESH) {
    dir.create(dirname(nested), showWarnings = FALSE, recursive = TRUE)
    utils::unzip(archive, files = nested_member, exdir = extract_dir, overwrite = TRUE)
  }
  list(archive = nested, member = archive_member(nested))
}

parse_travis_ears <- function(archive, tax_year, target_ids, extract_dir) {
  outer_members <- utils::unzip(archive, list = TRUE)
  if (any(grepl("\\.zip$", outer_members$Name, ignore.case = TRUE))) {
    data_archive <- extract_nested_zip(archive, extract_dir, preferred_pattern = "ears")
  } else {
    data_archive <- archive
  }

  data_members <- utils::unzip(data_archive, list = TRUE)
  csv_members <- data_members[grepl("\\.csv$", data_members$Name, ignore.case = TRUE), , drop = FALSE]
  ajr_members <- csv_members[
    grepl("AJR_RECORDS", basename(csv_members$Name), ignore.case = TRUE),
    , drop = FALSE
  ]
  if (nrow(ajr_members) > 0) {
    ajr_members <- ajr_members[order(ajr_members$Length, decreasing = TRUE), ]
    member <- ajr_members$Name[[1]]
  } else {
    member <- archive_member(data_archive, pattern = "\\.csv$")
  }
  command <- paste(
    "unzip -p", shQuote(data_archive), shQuote(member),
    "| LC_ALL=C grep '^AJR,'"
  )

  indexes <- c(1, 2, 4, 5, 6, 7, 8, 9, 10, 22, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38)
  column_names <- c(
    "record_type", "source_tax_year", "taxing_unit_id", "taxing_unit_type",
    "county_fund", "long_account", "short_account", "parent_account",
    "situs_address", "improvement_sqft", "land_unit", "land_size",
    "category_code", "previous_category_code", "previous_total_market_value",
    "totally_exempt_value", "land_market_value", "improvement_market_value",
    "mineral_market_value", "personal_market_value"
  )

  values <- data.table::fread(
    cmd = command,
    header = FALSE,
    select = indexes,
    col.names = column_names,
    colClasses = "character",
    fill = TRUE,
    showProgress = TRUE
  )
  source_rows <- nrow(values)

  values[, short_account := normalize_account(short_account)]
  values[, long_account := normalize_account(long_account)]
  county_rows <- values[taxing_unit_id == "227000"]
  if (nrow(county_rows) == 0) {
    county_rows <- values[taxing_unit_type == "00" & county_fund %in% c("", "A", NA_character_)]
  }
  rm(values)

  numeric_columns <- c(
    "improvement_sqft", "land_size", "previous_total_market_value",
    "totally_exempt_value", "land_market_value", "improvement_market_value",
    "mineral_market_value", "personal_market_value"
  )
  county_rows[, (numeric_columns) := lapply(.SD, as_value), .SDcols = numeric_columns]
  county_rows[, county_account_id := fifelse(
    !is.na(short_account), short_account, long_account
  )]
  county_data <- county_rows[!is.na(county_account_id), .(
    land_market_value = sum_or_na(land_market_value),
    improvement_market_value = sum_or_na(improvement_market_value)
  ), by = .(source_account_id = county_account_id)]
  county_data[, `:=`(
    source_county = "Travis",
    tax_year = as.integer(tax_year)
  )]

  county_rows[, source_account_id := fifelse(
    short_account %in% target_ids,
    short_account,
    fifelse(long_account %in% target_ids, long_account, short_account)
  )]
  matched <- county_rows[source_account_id %in% target_ids]
  matched[, land_sqft_component := fcase(
    land_unit == "1", land_size * 43560,
    land_unit == "2", land_size,
    default = NA_real_
  )]
  matched[, total_market_component := rowSums(
    .SD,
    na.rm = TRUE
  ), .SDcols = c(
    "totally_exempt_value", "land_market_value", "improvement_market_value",
    "mineral_market_value", "personal_market_value"
  )]
  matched[
    is.na(totally_exempt_value) & is.na(land_market_value) &
      is.na(improvement_market_value) & is.na(mineral_market_value) &
      is.na(personal_market_value),
    total_market_component := NA_real_
  ]

  output <- matched[, .(
    source_long_account_id = first_or_na(long_account),
    category_codes = collapse_values(category_code),
    situs_address_year = first_or_na(situs_address),
    land_sqft_year = max_or_na(land_sqft_component),
    improvement_sqft_year = max_or_na(improvement_sqft),
    land_market_value = sum_or_na(land_market_value),
    improvement_market_value = sum_or_na(improvement_market_value),
    total_market_value = sum_or_na(total_market_component),
    assessed_value = NA_real_,
    source_previous_total_market_value = sum_or_na(previous_total_market_value)
  ), by = source_account_id]

  output[, `:=`(
    parcel_id = source_account_id,
    source_county = "Travis",
    tax_year = as.integer(tax_year),
    data_source = "Travis CAD certified EARS"
  )]

  list(
    data = as_tibble(output),
    county_data = as_tibble(county_data),
    source_rows = source_rows,
    source_accounts = data.table::uniqueN(county_rows$short_account),
    matched_source_rows = nrow(matched)
  )
}

parse_hays_export <- function(archive, tax_year, target_ids, extract_dir) {
  location <- locate_hays_property_table(archive, extract_dir)
  values <- read_zipped_table(location$archive, location$member)
  source_rows <- nrow(values)

  quick_ref <- normalize_account(pick_column(
    values,
    c("QuickRefID", "Quick Ref ID", "GeoID"),
    required = TRUE
  ))
  parcel_id <- paste0("HAYS:", quick_ref)

  current_land <- pick_column(values, c("CurrLandValue", "CurrentLandValue"))
  fallback_land <- pick_column(values, c("LandValue"))
  current_improvement <- pick_column(
    values,
    c("CurrImprovmentValue", "CurrImprovementValue", "CurrentImprovementValue")
  )
  fallback_improvement <- pick_column(values, c("ImprovmentValue", "ImprovementValue"))
  current_total <- pick_column(values, c("CurrMarketValue", "CurrentMarketValue"))
  fallback_total <- pick_column(values, c("MarketValue", "TotalMarketValue"))
  current_assessed <- pick_column(values, c("CurrAssessedValue", "CurrentAssessedValue"))
  fallback_assessed <- pick_column(values, c("AssessedValue", "TotalAssessedValue"))

  coalesce_value <- function(primary, fallback) {
    primary <- as_value(primary)
    fallback <- as_value(fallback)
    ifelse(is.na(primary), fallback, primary)
  }

  land_values <- coalesce_value(current_land, fallback_land)
  improvement_values <- coalesce_value(current_improvement, fallback_improvement)
  total_values <- coalesce_value(current_total, fallback_total)
  assessed_values <- coalesce_value(current_assessed, fallback_assessed)

  county_data <- tibble(
    source_account_id = quick_ref,
    land_market_value = land_values,
    improvement_market_value = improvement_values
  ) %>%
    filter(!is.na(source_account_id), grepl("^R[0-9]+$", source_account_id)) %>%
    group_by(source_account_id) %>%
    summarise(
      land_market_value = max_or_na(land_market_value),
      improvement_market_value = max_or_na(improvement_market_value),
      .groups = "drop"
    ) %>%
    mutate(source_county = "Hays", tax_year = as.integer(tax_year))

  matched <- tibble(
    parcel_id = parcel_id,
    source_account_id = quick_ref,
    source_long_account_id = as.character(pick_column(values, c("PropertyID"))),
    category_codes = NA_character_,
    situs_address_year = as.character(pick_column(values, c("Situs", "SitusAddress"))),
    land_sqft_year = NA_real_,
    improvement_sqft_year = as_value(pick_column(
      values,
      c("SquareFootage", "TotalSqFtLivingArea", "LivingArea")
    )),
    land_market_value = land_values,
    improvement_market_value = improvement_values,
    total_market_value = total_values,
    assessed_value = assessed_values,
    source_previous_total_market_value = as_value(fallback_total)
  ) %>%
    filter(parcel_id %in% target_ids) %>%
    mutate(
      total_market_value = if_else(
        is.na(total_market_value),
        land_market_value + improvement_market_value,
        total_market_value
      )
    ) %>%
    group_by(parcel_id) %>%
    summarise(
      source_account_id = first_or_na(source_account_id),
      source_long_account_id = first_or_na(source_long_account_id),
      category_codes = collapse_values(category_codes),
      situs_address_year = first_or_na(situs_address_year),
      land_sqft_year = max_or_na(land_sqft_year),
      improvement_sqft_year = max_or_na(improvement_sqft_year),
      land_market_value = sum_or_na(land_market_value),
      improvement_market_value = sum_or_na(improvement_market_value),
      total_market_value = sum_or_na(total_market_value),
      assessed_value = sum_or_na(assessed_value),
      source_previous_total_market_value = sum_or_na(source_previous_total_market_value),
      .groups = "drop"
    ) %>%
    mutate(
      source_county = "Hays",
      tax_year = as.integer(tax_year),
      data_source = "Hays CAD certified property export"
    )

  if (
    nrow(matched) > 0 &&
      !any(
        !is.na(matched$land_market_value) |
          !is.na(matched$improvement_market_value) |
          !is.na(matched$total_market_value)
      )
  ) {
    stop(
      "Hays ", tax_year,
      " matched parcel IDs but contained no appraisal values; check the nested property-table selection.",
      call. = FALSE
    )
  }

  list(
    data = matched,
    county_data = county_data,
    source_rows = source_rows,
    source_accounts = dplyr::n_distinct(quick_ref),
    matched_source_rows = sum(parcel_id %in% target_ids)
  )
}

parse_hays_certified_fixed <- function(archive, tax_year, target_ids) {
  member <- archive_member(archive, pattern = "\\.txt$")
  command <- paste("unzip -p", shQuote(archive), shQuote(member))
  connection <- pipe(command, open = "r")
  on.exit(close(connection), add = TRUE)

  chunks <- list()
  source_rows <- 0L
  chunk_index <- 0L
  observed_record_lengths <- integer()

  repeat {
    records <- readLines(connection, n = 25000L, warn = FALSE)
    if (length(records) == 0L) break
    source_rows <- source_rows + length(records)
    observed_record_lengths <- c(
      observed_record_lengths,
      nchar(records[seq_len(min(length(records), 10L))], type = "bytes")
    )

    source_account_id <- trimws(substr(records, 1L, 10L))
    real_property <- grepl("^R[0-9]+$", source_account_id)
    if (!any(real_property)) next

    records <- records[real_property]
    source_account_id <- source_account_id[real_property]
    fixed_value <- function(start, end) {
      suppressWarnings(as.numeric(substr(records, start, end)))
    }

    chunk_index <- chunk_index + 1L
    chunks[[chunk_index]] <- data.table(
      source_account_id = source_account_id,
      land_market_value = fixed_value(526L, 535L) + fixed_value(536L, 545L),
      improvement_market_value = fixed_value(579L, 588L) + fixed_value(589L, 598L),
      assessed_value = fixed_value(599L, 608L)
    )[, total_market_value := land_market_value + improvement_market_value]
  }

  if (!length(chunks)) {
    stop("No Hays real-property records found in ", archive, call. = FALSE)
  }
  if (any(observed_record_lengths != 4096L)) {
    stop(
      "Unexpected Hays certified fixed-width record length in ", archive,
      "; expected 4096 bytes.",
      call. = FALSE
    )
  }

  county_values <- rbindlist(chunks, use.names = TRUE)
  complete_value_record <- complete.cases(
    county_values[, .(land_market_value, improvement_market_value, assessed_value)]
  )
  auxiliary_rows <- county_values[!complete_value_record]
  county_values <- county_values[complete_value_record]
  if (anyDuplicated(county_values$source_account_id)) {
    stop(
      "Duplicate Hays real-property account IDs in certified fixed-width export.",
      call. = FALSE
    )
  }
  target_accounts <- sub("^HAYS:", "", target_ids)
  matched <- county_values[source_account_id %in% target_accounts] %>%
    as_tibble() %>%
    transmute(
      parcel_id = paste0("HAYS:", source_account_id),
      source_county = "Hays",
      tax_year = as.integer(tax_year),
      source_account_id,
      source_long_account_id = NA_character_,
      category_codes = NA_character_,
      situs_address_year = NA_character_,
      land_sqft_year = NA_real_,
      improvement_sqft_year = NA_real_,
      land_market_value,
      improvement_market_value,
      total_market_value,
      assessed_value,
      source_previous_total_market_value = NA_real_,
      data_source = "Hays CAD certified fixed-width appraisal roll"
    )

  list(
    data = matched,
    county_data = county_values %>%
      as_tibble() %>%
      mutate(source_county = "Hays", tax_year = as.integer(tax_year)),
    source_rows = source_rows,
    source_accounts = nrow(county_values),
    matched_source_rows = nrow(matched),
    auxiliary_source_rows = nrow(auxiliary_rows)
  )
}

write_hays_source_comparison <- function(certified, postcert, output_dir) {
  value_columns <- c(
    "land_market_value", "improvement_market_value",
    "total_market_value", "assessed_value"
  )

  certified_values <- certified %>%
    select(parcel_id, all_of(value_columns)) %>%
    rename_with(~ paste0("certified_", .x), all_of(value_columns)) %>%
    mutate(certified_present = TRUE)
  postcert_values <- postcert %>%
    select(parcel_id, all_of(value_columns)) %>%
    rename_with(~ paste0("postcert_", .x), all_of(value_columns)) %>%
    mutate(postcert_present = TRUE)

  comparison <- full_join(certified_values, postcert_values, by = "parcel_id") %>%
    mutate(
      certified_present = coalesce(certified_present, FALSE),
      postcert_present = coalesce(postcert_present, FALSE)
    )

  for (value_name in value_columns) {
    certified_name <- paste0("certified_", value_name)
    postcert_name <- paste0("postcert_", value_name)
    comparison[[paste0("delta_", value_name)]] <-
      comparison[[certified_name]] - comparison[[postcert_name]]
    comparison[[paste0("pct_delta_", value_name)]] <- ifelse(
      comparison[[postcert_name]] == 0,
      NA_real_,
      comparison[[paste0("delta_", value_name)]] /
        comparison[[postcert_name]] * 100
    )
  }

  comparison <- comparison %>% arrange(parcel_id)
  readr::write_csv(
    comparison,
    file.path(output_dir, "appraisal_hays_2022_source_comparison.csv")
  )

  comparison_qa <- bind_rows(lapply(value_columns, function(value_name) {
    certified_value <- comparison[[paste0("certified_", value_name)]]
    postcert_value <- comparison[[paste0("postcert_", value_name)]]
    delta <- comparison[[paste0("delta_", value_name)]]
    both_present <- comparison$certified_present & comparison$postcert_present &
      !is.na(certified_value) & !is.na(postcert_value)

    tibble(
      metric = value_name,
      certified_parcels = sum(comparison$certified_present),
      postcert_parcels = sum(comparison$postcert_present),
      parcels_in_both = sum(both_present),
      identical_values = sum(certified_value[both_present] == postcert_value[both_present]),
      changed_values = sum(certified_value[both_present] != postcert_value[both_present]),
      certified_missing_postcert_present = sum(
        !comparison$certified_present & comparison$postcert_present
      ),
      postcert_missing_certified_present = sum(
        comparison$certified_present & !comparison$postcert_present
      ),
      median_delta_certified_minus_postcert = median(delta[both_present], na.rm = TRUE),
      median_absolute_delta = median(abs(delta[both_present]), na.rm = TRUE),
      maximum_absolute_delta = max(abs(delta[both_present]), na.rm = TRUE)
    )
  })) %>%
    mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

  readr::write_csv(
    comparison_qa,
    file.path(output_dir, "appraisal_hays_2022_source_comparison_qa.csv")
  )
  invisible(comparison_qa)
}

extract_report_money <- function(lines, label) {
  hits <- grep(
    paste0("\\b", label, "[[:space:]:]*\\$\\s*[-(]?[0-9,]+"),
    lines,
    value = TRUE,
    perl = TRUE
  )
  if (length(hits) == 0) return(NA_real_)
  remainder <- sub(
    paste0(".*\\b", label, "[[:space:]:]*\\$\\s*"),
    "",
    hits[[1]],
    perl = TRUE
  )
  matched <- regmatches(
    remainder,
    regexpr("[-(]?[0-9][0-9,]*", remainder, perl = TRUE)
  )
  value <- as_value(matched)
  if (grepl("^\\(", matched)) -value else value
}

parse_williamson_report_county_values <- function(archive, member, tax_year) {
  awk_program <- paste(
    'function money(s,t){sub(/^.*\\$/, "", s);',
    'sub(/^[[:space:]]*/, "", s);',
    'if(match(s,/^[0-9,]+/)){t=substr(s,RSTART,RLENGTH);',
    'gsub(/,/,"",t);return t+0}return 0}',
    'function emit(){if(id ~ /^R[0-9]+$/)',
    'printf "%s\\t%.0f\\t%.0f\\n",id,lh+ln+ag+tm,ih+inh}',
    '/^PID:[[:space:]]*/{emit();id=$2;lh=ln=ag=tm=ih=inh=0}',
    '/LANDHS[[:space:]]+\\$/{lh=money($0)}',
    '/LANDNHS[[:space:]]+\\$/{ln=money($0)}',
    '/IMPHS[[:space:]]+\\$/{ih=money($0)}',
    '/IMPNHS[[:space:]]+\\$/{inh=money($0)}',
    '/AGMKT[[:space:]]+\\$/{ag=money($0)}',
    '/TIMMKT[[:space:]]+\\$/{tm=money($0)}',
    'END{emit()}',
    collapse = " "
  )
  command <- paste(
    "unzip -p", shQuote(archive), shQuote(member),
    "| awk", shQuote(awk_program)
  )
  values <- data.table::fread(
    cmd = command,
    header = FALSE,
    sep = "\t",
    col.names = c(
      "source_account_id", "land_market_value", "improvement_market_value"
    ),
    colClasses = c("character", "numeric", "numeric"),
    showProgress = TRUE,
    data.table = FALSE
  )

  values %>%
    as_tibble() %>%
    group_by(source_account_id) %>%
    summarise(
      land_market_value = max_or_na(land_market_value),
      improvement_market_value = max_or_na(improvement_market_value),
      .groups = "drop"
    ) %>%
    mutate(source_county = "Williamson", tax_year = as.integer(tax_year))
}

parse_williamson_report <- function(archive, tax_year, target_ids) {
  member <- archive_member(archive, pattern = "\\.(txt|dat)$")
  command <- paste("unzip -p", shQuote(archive), shQuote(member))
  connection <- pipe(command, open = "r", encoding = "UTF-8")
  on.exit(close(connection), add = TRUE)

  records <- list()
  record_index <- 0L
  source_accounts <- 0L
  matched_source_rows <- 0L
  carry <- character()

  process_pages <- function(lines, final = FALSE) {
    lines <- sub("^[[:cntrl:]]+", "", lines)
    starts <- grep("^PID:\\s*", lines, perl = TRUE)
    if (length(starts) == 0) return(lines)

    process_count <- if (final) length(starts) else max(0L, length(starts) - 1L)
    if (process_count == 0L) return(lines[starts[[1]]:length(lines)])

    for (index in seq_len(process_count)) {
      page_start <- starts[[index]]
      page_end <- if (index < length(starts)) starts[[index + 1L]] - 1L else length(lines)
      page <- lines[page_start:page_end]
      account <- sub("^PID:\\s*([^[:space:]]+).*$", "\\1", page[[1]], perl = TRUE)
      source_accounts <<- source_accounts + 1L
      if (!account %in% target_ids) next

      matched_source_rows <<- matched_source_rows + 1L
      land_hs <- extract_report_money(page, "LANDHS")
      land_nhs <- extract_report_money(page, "LANDNHS")
      improvement_hs <- extract_report_money(page, "IMPHS")
      improvement_nhs <- extract_report_money(page, "IMPNHS")
      agricultural_market <- extract_report_money(page, "AGMKT")
      timber_market <- extract_report_money(page, "TIMMKT")

      market_components <- c(
        land_hs, land_nhs, improvement_hs, improvement_nhs,
        agricultural_market, timber_market
      )
      land_components <- c(land_hs, land_nhs, agricultural_market, timber_market)
      improvement_components <- c(improvement_hs, improvement_nhs)

      assessed_lines <- grep("ASSESSED\\s*\\$", page, value = TRUE, perl = TRUE)
      assessed_lines <- assessed_lines[!grepl("LAST YEARS", assessed_lines, fixed = TRUE)]
      assessed <- if (length(assessed_lines) > 0) {
        extract_report_money(assessed_lines, "ASSESSED")
      } else {
        NA_real_
      }

      record_index <<- record_index + 1L
      records[[record_index]] <<- tibble(
        parcel_id = paste0("WILLIAMSON:", account),
        source_account_id = account,
        source_long_account_id = NA_character_,
        category_codes = NA_character_,
        situs_address_year = NA_character_,
        land_sqft_year = NA_real_,
        improvement_sqft_year = NA_real_,
        land_market_value = sum_or_na(land_components),
        improvement_market_value = sum_or_na(improvement_components),
        total_market_value = sum_or_na(market_components),
        assessed_value = assessed,
        source_previous_total_market_value = extract_report_money(page, "LAST YEARS ASSESSED"),
        source_county = "Williamson",
        tax_year = as.integer(tax_year),
        data_source = "Williamson CAD certified parcel report"
      )
    }

    if (final) character() else lines[starts[[process_count + 1L]]:length(lines)]
  }

  repeat {
    chunk <- readLines(connection, n = 100000L, warn = FALSE)
    if (length(chunk) == 0) break
    carry <- process_pages(c(carry, chunk), final = FALSE)
  }
  process_pages(carry, final = TRUE)

  county_data <- parse_williamson_report_county_values(archive, member, tax_year)
  matched_data <- bind_rows(records)
  reconciliation <- matched_data %>%
    select(source_account_id, parsed_land_market_value = land_market_value) %>%
    inner_join(county_data, by = "source_account_id")
  if (
    nrow(reconciliation) > 0L &&
      any(
        abs(
          reconciliation$parsed_land_market_value -
            reconciliation$land_market_value
        ) > 0.5,
        na.rm = TRUE
      )
  ) {
    stop(
      "Williamson county-wide report parser did not reconcile to target records.",
      call. = FALSE
    )
  }

  list(
    data = matched_data,
    county_data = county_data,
    source_rows = source_accounts,
    source_accounts = source_accounts,
    matched_source_rows = matched_source_rows
  )
}

parse_williamson_table_export <- function(archive, tax_year, target_ids) {
  members <- utils::unzip(archive, list = TRUE)
  value_members <- members[
    tolower(basename(members$Name)) == "final_values.txt",
    , drop = FALSE
  ]
  if (nrow(value_members) == 0) {
    stop("Final_Values.txt not found in ", archive, call. = FALSE)
  }

  member <- value_members$Name[[1]]
  command <- paste("unzip -p", shQuote(archive), shQuote(member))
  values <- data.table::fread(
    cmd = command,
    colClasses = "character",
    fill = TRUE,
    showProgress = TRUE,
    data.table = FALSE
  )
  source_rows <- nrow(values)
  quick_ref <- normalize_account(pick_column(values, c("QuickRefID"), required = TRUE))
  source_year <- as.integer(as_value(pick_column(values, c("AdHocTaxYear"))))
  final_land <- as_value(pick_column(values, c("FinalLand")))
  final_improvement <- as_value(pick_column(values, c("FinalBuilding")))
  final_total <- as_value(pick_column(values, c("FinalTotal", "MarketValue")))
  base_value <- as_value(pick_column(values, c("BaseValue")))

  county_data <- tibble(
    source_account_id = quick_ref,
    land_market_value = final_land,
    improvement_market_value = final_improvement,
    source_year = source_year
  ) %>%
    filter(
      !is.na(source_account_id),
      grepl("^R[0-9]+$", source_account_id),
      is.na(source_year) | source_year == tax_year
    ) %>%
    group_by(source_account_id) %>%
    summarise(
      land_market_value = max_or_na(land_market_value),
      improvement_market_value = max_or_na(improvement_market_value),
      .groups = "drop"
    ) %>%
    mutate(source_county = "Williamson", tax_year = as.integer(tax_year))

  matched <- tibble(
    parcel_id = paste0("WILLIAMSON:", quick_ref),
    source_account_id = quick_ref,
    source_long_account_id = as.character(pick_column(values, c("PropertyID"))),
    category_codes = NA_character_,
    situs_address_year = NA_character_,
    land_sqft_year = NA_real_,
    improvement_sqft_year = NA_real_,
    land_market_value = final_land,
    improvement_market_value = final_improvement,
    total_market_value = final_total,
    assessed_value = NA_real_,
    source_previous_total_market_value = base_value
  ) %>%
    filter(quick_ref %in% target_ids, is.na(source_year) | source_year == tax_year) %>%
    group_by(parcel_id) %>%
    summarise(
      source_account_id = first_or_na(source_account_id),
      source_long_account_id = first_or_na(source_long_account_id),
      category_codes = collapse_values(category_codes),
      situs_address_year = first_or_na(situs_address_year),
      land_sqft_year = max_or_na(land_sqft_year),
      improvement_sqft_year = max_or_na(improvement_sqft_year),
      land_market_value = sum_or_na(land_market_value),
      improvement_market_value = sum_or_na(improvement_market_value),
      total_market_value = sum_or_na(total_market_value),
      assessed_value = sum_or_na(assessed_value),
      source_previous_total_market_value = sum_or_na(source_previous_total_market_value),
      .groups = "drop"
    ) %>%
    mutate(
      source_county = "Williamson",
      tax_year = as.integer(tax_year),
      data_source = "Williamson CAD certified final values export"
    )

  list(
    data = matched,
    county_data = county_data,
    source_rows = source_rows,
    source_accounts = dplyr::n_distinct(quick_ref),
    matched_source_rows = sum(quick_ref %in% target_ids & (is.na(source_year) | source_year == tax_year))
  )
}

parse_williamson_archive <- function(archive, tax_year, target_ids) {
  members <- utils::unzip(archive, list = TRUE)
  if (any(tolower(basename(members$Name)) == "final_values.txt")) {
    parse_williamson_table_export(archive, tax_year, target_ids)
  } else {
    parse_williamson_report(archive, tax_year, target_ids)
  }
}

parse_williamson_current <- function(path, tax_year, target_ids) {
  required_columns <- c(
    "PropertyID", "QuickRefID", "TotalAssessedValue", "TotalImpMktValue",
    "TotalLandMktValue", "TotalPropMktValue", "TotalSqFtLivingArea",
    "SitusAddress", "Acres"
  )
  values <- data.table::fread(
    path,
    select = required_columns,
    colClasses = "character",
    fill = Inf,
    showProgress = TRUE,
    data.table = FALSE
  )
  quick_ref <- normalize_account(pick_column(values, c("QuickRefID"), required = TRUE))
  parcel_id <- paste0("WILLIAMSON:", quick_ref)
  current_land <- as_value(pick_column(values, c("TotalLandMktValue")))
  current_improvement <- as_value(pick_column(values, c("TotalImpMktValue")))
  current_total <- as_value(pick_column(values, c("TotalPropMktValue")))
  current_assessed <- as_value(pick_column(values, c("TotalAssessedValue")))
  current_land_sqft <- as_value(pick_column(values, c("Acres"))) * 43560

  county_data <- tibble(
    source_account_id = quick_ref,
    land_market_value = current_land,
    improvement_market_value = current_improvement
  ) %>%
    filter(!is.na(source_account_id), grepl("^R[0-9]+$", source_account_id)) %>%
    group_by(source_account_id) %>%
    summarise(
      land_market_value = max_or_na(land_market_value),
      improvement_market_value = max_or_na(improvement_market_value),
      .groups = "drop"
    ) %>%
    mutate(source_county = "Williamson", tax_year = as.integer(tax_year))

  matched <- tibble(
    parcel_id = parcel_id,
    source_account_id = quick_ref,
    source_long_account_id = as.character(pick_column(values, c("PropertyID"))),
    category_codes = NA_character_,
    situs_address_year = as.character(pick_column(values, c("SitusAddress", "Situs"))),
    land_sqft_year = current_land_sqft,
    improvement_sqft_year = as_value(pick_column(values, c("TotalSqFtLivingArea"))),
    land_market_value = current_land,
    improvement_market_value = current_improvement,
    total_market_value = current_total,
    assessed_value = current_assessed,
    source_previous_total_market_value = NA_real_
  ) %>%
    filter(parcel_id %in% target_ids) %>%
    group_by(parcel_id) %>%
    summarise(
      source_account_id = first_or_na(source_account_id),
      source_long_account_id = first_or_na(source_long_account_id),
      category_codes = collapse_values(category_codes),
      situs_address_year = first_or_na(situs_address_year),
      land_sqft_year = max_or_na(land_sqft_year),
      improvement_sqft_year = max_or_na(improvement_sqft_year),
      land_market_value = sum_or_na(land_market_value),
      improvement_market_value = sum_or_na(improvement_market_value),
      total_market_value = sum_or_na(total_market_value),
      assessed_value = sum_or_na(assessed_value),
      source_previous_total_market_value = sum_or_na(source_previous_total_market_value),
      .groups = "drop"
    ) %>%
    mutate(
      source_county = "Williamson",
      tax_year = as.integer(tax_year),
      data_source = "Williamson CAD certified property export"
    )

  list(
    data = matched,
    county_data = county_data,
    source_rows = nrow(values),
    source_accounts = dplyr::n_distinct(quick_ref),
    matched_source_rows = sum(parcel_id %in% target_ids)
  )
}

manifest <- readr::read_csv(SOURCE_MANIFEST, show_col_types = FALSE) %>%
  mutate(
    tax_year = as.integer(tax_year),
    local_path = if_else(is.na(local_path), "", local_path),
    comparison_path = if_else(is.na(comparison_path), "", comparison_path),
    url = if_else(is.na(url), "", url),
    required = if_else(is.na(required), TRUE, as.logical(required))
  ) %>%
  filter(
    county %in% SELECTED_COUNTIES,
    tax_year %in% EWS_CONFIG$appraisal_years
  ) %>%
  arrange(county, tax_year)

expected_sources <- tidyr::crossing(
  county = SELECTED_COUNTIES,
  tax_year = EWS_CONFIG$appraisal_years
)
missing_sources <- anti_join(expected_sources, manifest, by = c("county", "tax_year"))
if (nrow(missing_sources) > 0) {
  stop(
    "Missing appraisal source configuration for: ",
    paste0(missing_sources$county, " ", missing_sources$tax_year, collapse = ", "),
    call. = FALSE
  )
}

parcels <- load_output(
  file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.rds"),
  "calibrated residential parcel universe"
) %>%
  as_tibble() %>%
  transmute(
    parcel_id = as.character(parcel_id),
    source_county = as.character(source_county),
    current_units = as.numeric(units_calibrated),
    current_land_sqft = as.numeric(land_sqft),
    current_improvement_sqft = as.numeric(improvement_sqft),
    current_situs_address = as.character(situs_address),
    lat = as.numeric(lat),
    lon = as.numeric(lon)
  ) %>%
  filter(source_county %in% SELECTED_COUNTIES) %>%
  distinct(parcel_id, .keep_all = TRUE)

if (anyDuplicated(parcels$parcel_id)) {
  stop("Current residential parcel IDs are not unique.", call. = FALSE)
}

hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid")
parcel_coordinates <- parcels %>% filter(!is.na(lat), !is.na(lon))
parcel_hex <- st_as_sf(parcel_coordinates, coords = c("lon", "lat"), crs = 4326) %>%
  st_transform(st_crs(hex_grid)) %>%
  st_join(hex_grid %>% select(hex_id), join = st_within, left = TRUE) %>%
  st_drop_geometry() %>%
  select(parcel_id, hex_id) %>%
  distinct(parcel_id, .keep_all = TRUE)

parcels <- parcels %>% left_join(parcel_hex, by = "parcel_id")
spatial_qa <- parcels %>%
  group_by(source_county) %>%
  summarise(
    current_parcels = n(),
    parcels_missing_coordinates = sum(is.na(lat) | is.na(lon)),
    parcels_outside_or_missing_hex = sum(is.na(hex_id)),
    parcel_hex_match_pct = (current_parcels - parcels_outside_or_missing_hex) /
      current_parcels * 100,
    .groups = "drop"
  ) %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)
readr::write_csv(spatial_qa, file.path(OUTPUT_DIR, "appraisal_panel_spatial_qa.csv"))

signatures <- parcels %>%
  group_by(source_county) %>%
  summarise(signature = target_signature(parcel_id), .groups = "drop")

normalized_results <- vector("list", nrow(manifest))
county_results <- vector("list", nrow(manifest))
qa_results <- vector("list", nrow(manifest))

for (index in seq_len(nrow(manifest))) {
  source_row <- manifest[index, ]
  county <- source_row$county[[1]]
  tax_year <- source_row$tax_year[[1]]
  source_type <- source_row$source_type[[1]]
  parser_version <- unname(PARSER_VERSIONS[[source_type]])
  if (is.null(parser_version) || is.na(parser_version)) {
    stop("No parser version configured for source_type: ", source_type, call. = FALSE)
  }
  source_required <- isTRUE(source_row$required[[1]])
  target_ids <- parcels$parcel_id[parcels$source_county == county]
  signature <- signatures$signature[signatures$source_county == county][[1]]
  county_dir <- file.path(RAW_DIR, tolower(county), tax_year)
  dir.create(county_dir, showWarnings = FALSE, recursive = TRUE)

  source_error <- NA_character_
  if (nzchar(source_row$local_path[[1]])) {
    source_path <- project_path(source_row$local_path[[1]])
    if (!file.exists(source_path)) {
      stop("Configured local source does not exist: ", source_path, call. = FALSE)
    }
  } else {
    archive_path <- file.path(
      county_dir,
      paste0(tolower(county), "_", tax_year, ".zip")
    )
    if (!source_required && !DOWNLOAD_OPTIONAL && !file.exists(archive_path)) {
      source_error <- paste0("Optional archive not present at ", archive_path)
      source_path <- NA_character_
    } else {
      source_path <- tryCatch(
        download_source(source_row$url[[1]], archive_path),
        error = function(error) {
          if (source_required) stop(error)
          source_error <<- conditionMessage(error)
          warning(
            county, " ", tax_year, " source unavailable; retaining explicit missing parcel-years. ",
            source_error,
            call. = FALSE
          )
          NA_character_
        }
      )
    }
  }

  cache_path <- file.path(
    county_dir,
    paste0("normalized_", tolower(county), "_", tax_year, "_v", parser_version, ".rds")
  )

  if (is.na(source_path)) {
    parsed <- list(
      data = empty_normalized_values(),
      county_data = empty_county_values(),
      source_rows = NA_integer_,
      source_accounts = NA_integer_,
      matched_source_rows = NA_integer_
    )
    normalized_results[[index]] <- parsed$data
    county_results[[index]] <- parsed$county_data
    qa_results[[index]] <- tibble(
      county = county,
      tax_year = tax_year,
      source_type = source_type,
      source_status = "unavailable_optional_source",
      source_note = source_error,
      source_rows_or_accounts = NA_integer_,
      source_accounts = NA_integer_,
      matched_source_rows = NA_integer_,
      target_parcels = length(target_ids),
      matched_target_parcels = 0L,
      parcel_coverage_pct = 0,
      source_file = NA_character_,
      normalized_cache = NA_character_,
      parser_version = parser_version
    )
    next
  }

  current_source_signature <- source_signature(source_path)
  cache <- NULL
  if (file.exists(cache_path) && !REFRESH) cache <- readRDS(cache_path)

  cache_valid <- is.list(cache) &&
    identical(cache$parser_version, parser_version) &&
    identical(cache$target_signature, signature) &&
    identical(cache$source_signature, current_source_signature) &&
    all(names(empty_normalized_values()) %in% names(cache$parsed$data)) &&
    all(names(empty_county_values()) %in% names(cache$parsed$county_data))

  if (cache_valid) {
    print_progress(paste0("Using cached normalized extract: ", county, " ", tax_year))
    parsed <- cache$parsed
  } else {
    print_progress(paste0("Parsing ", county, " ", tax_year, "..."))
    parsed <- switch(
      source_type,
      travis_ears_nested_zip = parse_travis_ears(
        source_path, tax_year, target_ids, county_dir
      ),
      hays_property_export_zip = parse_hays_export(
        source_path, tax_year, target_ids, county_dir
      ),
      hays_certified_fixed_zip = parse_hays_certified_fixed(
        source_path, tax_year, target_ids
      ),
      williamson_certified_report_zip = parse_williamson_archive(
        source_path, tax_year, sub("^WILLIAMSON:", "", target_ids)
      ),
      williamson_current_csv = parse_williamson_current(
        source_path, tax_year, target_ids
      ),
      stop("Unsupported source_type: ", source_type, call. = FALSE)
    )
    saveRDS(
      list(
        parser_version = parser_version,
        target_signature = signature,
        source_signature = current_source_signature,
        parsed = parsed
      ),
      cache_path
    )
  }

  parsed$data <- parsed$data %>%
    select(
      parcel_id, source_county, tax_year, source_account_id,
      source_long_account_id, category_codes, situs_address_year,
      land_sqft_year, improvement_sqft_year, land_market_value,
      improvement_market_value, total_market_value, assessed_value,
      source_previous_total_market_value, data_source
    ) %>%
    distinct(parcel_id, tax_year, .keep_all = TRUE)
  parsed$county_data <- parsed$county_data %>%
    select(
      source_account_id, land_market_value, improvement_market_value,
      source_county, tax_year
    ) %>%
    distinct(source_county, source_account_id, tax_year, .keep_all = TRUE)

  comparison_path <- source_row$comparison_path[[1]]
  if (county == "Hays" && tax_year == 2022L && nzchar(comparison_path)) {
    comparison_source <- project_path(comparison_path)
    if (!file.exists(comparison_source)) {
      warning(
        "Configured Hays 2022 comparison source does not exist: ",
        comparison_source,
        call. = FALSE
      )
    } else {
      postcert <- parse_hays_export(
        comparison_source, tax_year, target_ids, county_dir
      )$data
      write_hays_source_comparison(parsed$data, postcert, OUTPUT_DIR)
    }
  }

  normalized_results[[index]] <- parsed$data
  county_results[[index]] <- parsed$county_data
  qa_results[[index]] <- tibble(
    county = county,
    tax_year = tax_year,
    source_type = source_type,
    source_status = "available",
    source_note = source_row$notes[[1]],
    source_rows_or_accounts = parsed$source_rows,
    source_accounts = parsed$source_accounts,
    matched_source_rows = parsed$matched_source_rows,
    target_parcels = length(target_ids),
    matched_target_parcels = n_distinct(parsed$data$parcel_id),
    parcel_coverage_pct = matched_target_parcels / target_parcels * 100,
    source_file = source_path,
    normalized_cache = cache_path,
    parser_version = parser_version
  )
}

normalized_values <- bind_rows(normalized_results)
if (anyDuplicated(normalized_values[c("parcel_id", "tax_year")])) {
  stop("Normalized appraisal values contain duplicate parcel-year records.", call. = FALSE)
}

county_values <- bind_rows(county_results)
if (anyDuplicated(county_values[c("source_county", "source_account_id", "tax_year")])) {
  stop("County appraisal values contain duplicate account-year records.", call. = FALSE)
}
save_output(
  county_values,
  file.path(OUTPUT_DIR, "appraisal_county_land_values_by_account_year.rds"),
  "county appraisal land values by account-year"
)

current_cpi <- unname(
  EWS_CONFIG$appraisal_cpi_u[[as.character(EWS_CONFIG$appraisal_current_year)]]
)
cpi_table <- tibble(
  tax_year = as.integer(names(EWS_CONFIG$appraisal_cpi_u)),
  cpi_u = as.numeric(EWS_CONFIG$appraisal_cpi_u)
) %>%
  filter(tax_year %in% EWS_CONFIG$appraisal_years)

parcel_panel <- tidyr::crossing(
  parcels,
  tax_year = EWS_CONFIG$appraisal_years
) %>%
  left_join(
    normalized_values %>% select(-source_county),
    by = c("parcel_id", "tax_year")
  ) %>%
  left_join(cpi_table, by = "tax_year") %>%
  mutate(
    value_available = !is.na(total_market_value) |
      !is.na(land_market_value) |
      !is.na(improvement_market_value),
    land_market_value_real = land_market_value * current_cpi / cpi_u,
    improvement_market_value_real = improvement_market_value * current_cpi / cpi_u,
    total_market_value_real = total_market_value * current_cpi / cpi_u,
    assessed_value_real = assessed_value * current_cpi / cpi_u,
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
  ) %>%
  arrange(parcel_id, tax_year)

save_output(
  parcel_panel,
  file.path(OUTPUT_DIR, "appraisal_values_by_parcel_year.rds"),
  "appraisal parcel-year panel"
)
data.table::fwrite(
  parcel_panel,
  file.path(OUTPUT_DIR, "appraisal_values_by_parcel_year.csv"),
  na = ""
)

hex_year_values <- parcel_panel %>%
  filter(!is.na(hex_id)) %>%
  group_by(hex_id, tax_year) %>%
  summarise(
    appraisal_parcels_total = n_distinct(parcel_id),
    appraisal_parcels_with_values = sum(value_available),
    appraisal_parcel_coverage_pct = appraisal_parcels_with_values / appraisal_parcels_total * 100,
    current_units_total = sum_or_na(current_units),
    current_units_with_values = sum_or_na(if_else(value_available, current_units, NA_real_)),
    appraisal_unit_coverage_pct = current_units_with_values / current_units_total * 100,
    current_land_sqft_total = sum_or_na(current_land_sqft),
    current_land_sqft_with_values = sum_or_na(if_else(value_available, current_land_sqft, NA_real_)),
    land_market_value = sum_or_na(land_market_value),
    improvement_market_value = sum_or_na(improvement_market_value),
    total_market_value = sum_or_na(total_market_value),
    assessed_value = sum_or_na(assessed_value),
    land_market_value_real = sum_or_na(land_market_value_real),
    improvement_market_value_real = sum_or_na(improvement_market_value_real),
    total_market_value_real = sum_or_na(total_market_value_real),
    assessed_value_real = sum_or_na(assessed_value_real),
    median_parcel_total_market_value_real = if (
      all(is.na(total_market_value_real))
    ) NA_real_ else median(total_market_value_real, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    land_value_real_per_current_land_sqft = if_else(
      !is.na(current_land_sqft_with_values) & current_land_sqft_with_values > 0,
      land_market_value_real / current_land_sqft_with_values,
      NA_real_
    ),
    total_value_real_per_current_unit = if_else(
      !is.na(current_units_with_values) & current_units_with_values > 0,
      total_market_value_real / current_units_with_values,
      NA_real_
    ),
    analysis_as_of_date = EWS_CONFIG$analysis_as_of_date
  )

hex_year_panel <- tidyr::crossing(
  hex_id = hex_grid$hex_id,
  tax_year = EWS_CONFIG$appraisal_years
) %>%
  left_join(hex_year_values, by = c("hex_id", "tax_year")) %>%
  left_join(cpi_table, by = "tax_year") %>%
  left_join(hex_grid %>% select(hex_id, geometry), by = "hex_id") %>%
  st_as_sf() %>%
  arrange(hex_id, tax_year)

save_output(
  hex_year_panel,
  file.path(OUTPUT_DIR, "appraisal_values_by_hex_year.rds"),
  "appraisal hex-year panel"
)
hex_year_panel %>%
  st_drop_geometry() %>%
  data.table::fwrite(
    file.path(OUTPUT_DIR, "appraisal_values_by_hex_year.csv"),
    na = ""
  )

annualized_log_change <- function(current, previous, years_elapsed) {
  if (
    is.na(current) || is.na(previous) || !is.finite(current) ||
      !is.finite(previous) || current <= 0 || previous <= 0 ||
      is.na(years_elapsed) || years_elapsed <= 0
  ) {
    return(NA_real_)
  }
  100 * (log(current) - log(previous)) / years_elapsed
}

trend_from_panel <- function(data) {
  data <- data %>% arrange(tax_year)
  years <- EWS_CONFIG$appraisal_years
  first_year <- min(years)
  current_year <- EWS_CONFIG$appraisal_current_year
  midpoint_year <- years[[ceiling(length(years) / 2)]]
  previous_year <- max(years[years < current_year])

  value_at <- function(column, year) {
    value <- data[[column]][data$tax_year == year]
    if (length(value) == 1 && is.finite(value[[1]])) value[[1]] else NA_real_
  }

  growth_metrics <- function(column) {
    first_value <- value_at(column, first_year)
    midpoint_value <- value_at(column, midpoint_year)
    current_value <- value_at(column, current_year)
    previous_value <- value_at(column, previous_year)
    recent_growth <- annualized_log_change(
      current_value, midpoint_value, current_year - midpoint_year
    )
    prior_growth <- annualized_log_change(
      midpoint_value, first_value, midpoint_year - first_year
    )
    c(
      current = current_value,
      growth_long = annualized_log_change(
        current_value, first_value, current_year - first_year
      ),
      growth_recent = recent_growth,
      growth_prior = prior_growth,
      acceleration = recent_growth - prior_growth,
      growth_one_year = annualized_log_change(current_value, previous_value, 1)
    )
  }

  land <- growth_metrics("land_value_real_per_current_land_sqft")
  total <- growth_metrics("total_value_real_per_current_unit")
  coverage <- data$appraisal_parcel_coverage_pct[!is.na(data$appraisal_parcel_coverage_pct)]
  current_units <- value_at("current_units_total", current_year)
  years_available <- sum(
    is.finite(data$land_value_real_per_current_land_sqft) &
      data$land_value_real_per_current_land_sqft > 0
  )
  minimum_coverage <- if (length(coverage) == 0) NA_real_ else min(coverage)

  tibble(
    appraisal_current_year = current_year,
    land_value_real_per_current_land_sqft = land[["current"]],
    land_value_growth_long_annualized_pct = land[["growth_long"]],
    land_value_growth_recent_annualized_pct = land[["growth_recent"]],
    land_value_growth_prior_annualized_pct = land[["growth_prior"]],
    land_value_acceleration_pp = land[["acceleration"]],
    land_value_growth_one_year_pct = land[["growth_one_year"]],
    total_value_real_per_current_unit = total[["current"]],
    total_value_growth_long_annualized_pct = total[["growth_long"]],
    total_value_growth_recent_annualized_pct = total[["growth_recent"]],
    total_value_growth_prior_annualized_pct = total[["growth_prior"]],
    total_value_acceleration_pp = total[["acceleration"]],
    total_value_growth_one_year_pct = total[["growth_one_year"]],
    appraisal_years_available = years_available,
    appraisal_min_parcel_coverage_pct = minimum_coverage,
    appraisal_trend_reliable = years_available == length(years) &
      !is.na(minimum_coverage) &
      minimum_coverage >= EWS_CONFIG$appraisal_min_parcel_coverage * 100 &
      !is.na(current_units) &
      current_units >= EWS_CONFIG$minimum_residential_units_for_rates
  )
}

hex_trend_values <- hex_year_panel %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  group_modify(~trend_from_panel(.x)) %>%
  ungroup()

hex_trends <- hex_grid %>%
  select(hex_id, geometry) %>%
  left_join(hex_trend_values, by = "hex_id") %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)

save_output(
  hex_trends,
  file.path(OUTPUT_DIR, "appraisal_value_trends_by_hex.rds"),
  "appraisal value trends by hex"
)
hex_trends %>%
  st_drop_geometry() %>%
  data.table::fwrite(
    file.path(OUTPUT_DIR, "appraisal_value_trends_by_hex.csv"),
    na = ""
  )

source_qa <- bind_rows(qa_results) %>%
  mutate(analysis_as_of_date = EWS_CONFIG$analysis_as_of_date)
readr::write_csv(source_qa, file.path(OUTPUT_DIR, "appraisal_panel_source_qa.csv"))

print_header("STEP 02i COMPLETE")
cat(paste0("Appraisal years: ", paste(EWS_CONFIG$appraisal_years, collapse = ", "), "\n"))
cat(paste0("Parcel-year rows: ", format(nrow(parcel_panel), big.mark = ","), "\n"))
cat(paste0("Hex-year rows: ", format(nrow(hex_year_panel), big.mark = ","), "\n"))
cat(
  paste0(
    "Reliable appraisal trends: ",
    format(sum(hex_trends$appraisal_trend_reliable, na.rm = TRUE), big.mark = ","),
    " hexagons\n"
  )
)
