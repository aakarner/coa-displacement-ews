################################################################################
# Williamson County Eviction Workbook Ingestion Helpers
################################################################################

williamson_eviction_required_columns <- function(data, required, data_name) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      data_name,
      " is missing required column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
}

williamson_eviction_blank <- function(x) {
  value <- stringr::str_squish(as.character(x))
  is.na(value) | value == ""
}

williamson_eviction_parse_date <- function(x) {
  value <- stringr::str_squish(as.character(x))
  value[value == ""] <- NA_character_

  excel_serial <- suppressWarnings(as.numeric(value))
  parsed <- as.Date(rep(NA_real_, length(value)), origin = "1970-01-01")
  serial_row <- !is.na(excel_serial) &
    excel_serial > 20000 &
    excel_serial < 80000
  parsed[serial_row] <- as.Date(
    excel_serial[serial_row],
    origin = "1899-12-30"
  )

  text_row <- !serial_row & !is.na(value)
  if (any(text_row)) {
    parsed[text_row] <- as.Date(lubridate::parse_date_time(
      value[text_row],
      orders = c("ymd", "mdy", "dmy", "Ymd HMS", "mdy HMS"),
      quiet = TRUE
    ))
  }
  parsed
}

# This intentionally matches the normalization used for the reviewed Travis
# geocode registry. Keeping the key definition identical makes keyed joins and
# exact candidate-set validation possible without re-geocoding Travis records.
williamson_clean_address_for_geocoding <- function(address) {
  address |>
    as.character() |>
    stringr::str_replace_all(
      stringr::regex("<br\\s*/?>", ignore_case = TRUE),
      ", "
    ) |>
    stringr::str_replace_all(
      stringr::regex("</?[^>]+>", ignore_case = TRUE),
      " "
    ) |>
    stringr::str_replace_all("&amp;", "&") |>
    stringr::str_replace_all("&nbsp;", " ") |>
    stringr::str_replace_all("[\r\n\t]", " ") |>
    stringr::str_replace_all("\\s*,\\s*", ", ") |>
    stringr::str_replace_all(",\\s*,+", ", ") |>
    stringr::str_squish() |>
    stringr::str_remove("^,\\s*") |>
    stringr::str_remove(",\\s*$") |>
    stringr::str_to_upper() |>
    stringr::str_replace("^([^,]+,\\s*)+(?=\\d+\\w?\\b)", "")
}

williamson_eviction_record_hash <- function(...) {
  fields <- list(...)
  row_count <- length(fields[[1]])
  vapply(
    seq_len(row_count),
    function(row) {
      digest::digest(
        paste(
          vapply(
            fields,
            function(field) {
              value <- as.character(field[[row]])
              if (is.na(value)) "<NA>" else value
            },
            character(1)
          ),
          collapse = "\u001f"
        ),
        algo = "sha256",
        serialize = FALSE
      )
    },
    character(1)
  )
}

williamson_read_jp1_evictions <- function(source_row) {
  path <- source_row$path[[1]]
  sheet <- source_row$sheet[[1]]
  raw <- readxl::read_excel(
    path,
    sheet = sheet,
    col_types = "text",
    .name_repair = janitor::make_clean_names
  )
  williamson_eviction_required_columns(
    raw,
    c(
      "case_nbr", "status", "file_date", "plaintiff", "defendant",
      "defendant_address"
    ),
    basename(path)
  )

  raw <- raw |>
    dplyr::mutate(
      source_physical_row = dplyr::row_number() + 1L,
      parsed_file_date = williamson_eviction_parse_date(.data$file_date),
      record_row = !is.na(.data$parsed_file_date) &
        !williamson_eviction_blank(.data$status),
      continuation_row = is.na(.data$parsed_file_date) &
        williamson_eviction_blank(.data$status) &
        williamson_eviction_blank(.data$file_date) &
        williamson_eviction_blank(.data$plaintiff) &
        williamson_eviction_blank(.data$defendant) &
        williamson_eviction_blank(.data$defendant_address) &
        !williamson_eviction_blank(.data$case_nbr)
    )

  nonblank_dates <- !williamson_eviction_blank(raw$file_date)
  if (any(nonblank_dates & is.na(raw$parsed_file_date))) {
    stop("JP1 contains a nonblank filing date that cannot be parsed.", call. = FALSE)
  }
  if (nrow(raw) == 0L || !raw$record_row[[1]]) {
    stop("JP1 must begin with a dated filing record.", call. = FALSE)
  }
  if (any(!raw$record_row & !raw$continuation_row)) {
    bad_rows <- raw$source_physical_row[!raw$record_row & !raw$continuation_row]
    stop(
      "JP1 contains unexpected report-row shapes at physical row(s): ",
      paste(bad_rows, collapse = ", "),
      call. = FALSE
    )
  }

  logical_records <- raw |>
    dplyr::mutate(logical_record = cumsum(.data$record_row)) |>
    dplyr::group_by(.data$logical_record) |>
    dplyr::summarize(
      source_row_start = min(.data$source_physical_row[.data$record_row]),
      source_row_end = max(.data$source_physical_row),
      case_number_raw = dplyr::first(.data$case_nbr[.data$record_row]),
      status_raw = dplyr::first(.data$status[.data$record_row]),
      file_date = dplyr::first(.data$parsed_file_date[.data$record_row]),
      plaintiff_name = dplyr::first(.data$plaintiff[.data$record_row]),
      defendant_name = dplyr::first(.data$defendant[.data$record_row]),
      address_line1_raw = dplyr::first(
        .data$defendant_address[.data$record_row]
      ),
      address_continuation_raw = {
        continuation <- stringr::str_squish(
          as.character(.data$case_nbr[.data$continuation_row])
        )
        continuation <- continuation[!is.na(continuation) & continuation != ""]
        if (length(continuation) == 0L) {
          NA_character_
        } else {
          paste(continuation, collapse = ", ")
        }
      },
      continuation_rows = sum(.data$continuation_row),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      address_raw = dplyr::case_when(
        !williamson_eviction_blank(.data$address_line1_raw) &
          !williamson_eviction_blank(.data$address_continuation_raw) ~
            paste(.data$address_line1_raw, .data$address_continuation_raw, sep = ", "),
        !williamson_eviction_blank(.data$address_line1_raw) ~
          as.character(.data$address_line1_raw),
        !williamson_eviction_blank(.data$address_continuation_raw) ~
          as.character(.data$address_continuation_raw),
        TRUE ~ NA_character_
      )
    )

  logical_records |>
    dplyr::mutate(
      source_id = source_row$source_id[[1]],
      source_group = "williamson_jp1",
      source_county = source_row$source_county[[1]],
      source_file = basename(path),
      source_sheet = sheet,
      jp_district = source_row$jp_district[[1]],
      source_period_start = as.Date(source_row$source_period_start[[1]]),
      source_period_end = as.Date(source_row$source_period_end[[1]]),
      source_exact_duplicate = FALSE,
      .before = 1
    )
}

williamson_read_jp2_evictions <- function(source_row) {
  path <- source_row$path[[1]]
  sheet <- source_row$sheet[[1]]
  raw <- suppressMessages(readxl::read_excel(
    path,
    sheet = sheet,
    skip = 7,
    col_types = "text",
    .name_repair = janitor::make_clean_names
  )) |>
    dplyr::select(-dplyr::matches("^x[0-9]*$"))
  williamson_eviction_required_columns(
    raw,
    c(
      "case_nbr", "status", "file_date", "plaintiff", "defendant",
      "defendant_address"
    ),
    basename(path)
  )

  records <- raw |>
    dplyr::transmute(
      source_row_start = dplyr::row_number() + 8L,
      source_row_end = .data$source_row_start,
      case_number_raw = .data$case_nbr,
      status_raw = .data$status,
      file_date = williamson_eviction_parse_date(.data$file_date),
      plaintiff_name = .data$plaintiff,
      defendant_name = .data$defendant,
      address_line1_raw = .data$defendant_address,
      address_continuation_raw = NA_character_,
      continuation_rows = 0L,
      address_raw = .data$defendant_address
    )
  if (any(is.na(records$file_date))) {
    stop("JP2 contains an unparseable or missing filing date.", call. = FALSE)
  }

  duplicate_key <- paste(
    records$case_number_raw,
    records$status_raw,
    records$file_date,
    records$plaintiff_name,
    records$defendant_name,
    records$address_raw,
    sep = "\u001f"
  )
  records |>
    dplyr::mutate(
      source_id = source_row$source_id[[1]],
      source_group = "williamson_jp2",
      source_county = source_row$source_county[[1]],
      source_file = basename(path),
      source_sheet = sheet,
      jp_district = source_row$jp_district[[1]],
      source_period_start = as.Date(source_row$source_period_start[[1]]),
      source_period_end = as.Date(source_row$source_period_end[[1]]),
      source_exact_duplicate = duplicated(duplicate_key),
      .before = 1
    )
}

williamson_prepare_eviction_records <- function(
  source_config,
  analysis_as_of_date
) {
  williamson_eviction_required_columns(
    source_config,
    c(
      "source_id", "source_county", "jp_district", "path", "sheet",
      "parser", "source_period_start", "source_period_end"
    ),
    "Eviction source configuration"
  )
  sources <- source_config |>
    dplyr::filter(.data$source_county == "Williamson") |>
    dplyr::mutate(
      source_period_start = as.Date(.data$source_period_start),
      source_period_end = as.Date(.data$source_period_end)
    )
  if (nrow(sources) == 0L || anyDuplicated(sources$source_id)) {
    stop("Williamson source configuration must contain unique source IDs.", call. = FALSE)
  }
  missing_paths <- sources$path[!file.exists(sources$path)]
  if (length(missing_paths) > 0L) {
    stop(
      "Missing configured Williamson workbook(s): ",
      paste(missing_paths, collapse = ", "),
      call. = FALSE
    )
  }

  records <- dplyr::bind_rows(lapply(seq_len(nrow(sources)), function(index) {
    source_row <- sources[index, , drop = FALSE]
    parser <- source_row$parser[[1]]
    if (identical(parser, "williamson_jp1_continuation")) {
      williamson_read_jp1_evictions(source_row)
    } else if (identical(parser, "williamson_jp2_report")) {
      williamson_read_jp2_evictions(source_row)
    } else {
      stop("Unsupported Williamson parser: ", parser, call. = FALSE)
    }
  }))

  records <- records |>
    dplyr::mutate(
      dplyr::across(
        dplyr::where(is.character),
        ~ dplyr::na_if(stringr::str_squish(.x), "")
      ),
      case_number = stringr::str_to_upper(.data$case_number_raw),
      case_number_format_valid = dplyr::case_when(
        .data$jp_district == "JP1" ~ stringr::str_detect(
          dplyr::coalesce(.data$case_number, ""),
          "^1JC-[0-9]{2}-[0-9]{4}$"
        ),
        .data$jp_district == "JP2" ~ stringr::str_detect(
          dplyr::coalesce(.data$case_number, ""),
          "^2JE-[0-9]{2}-[0-9]{4}$"
        ),
        TRUE ~ FALSE
      ),
      case_uid = dplyr::if_else(
        .data$case_number_format_valid,
        paste(.data$source_county, .data$jp_district, .data$case_number, sep = ":"),
        NA_character_
      ),
      source_record_id = paste0(.data$source_id, ":ROW", .data$source_row_start),
      case_type = "Eviction",
      case_status = .data$status_raw,
      court = paste0("WILLIAMSON_", .data$jp_district),
      outcome_year = as.integer(format(.data$file_date, "%Y")),
      after_analysis_cutoff = .data$file_date > as.Date(analysis_as_of_date),
      address_for_geocoding = williamson_clean_address_for_geocoding(
        .data$address_raw
      ),
      address_for_geocoding = dplyr::na_if(.data$address_for_geocoding, ""),
      address_key = .data$address_for_geocoding,
      missing_address = is.na(.data$address_for_geocoding),
      po_box_address = stringr::str_detect(
        dplyr::coalesce(.data$address_for_geocoding, ""),
        stringr::regex(
          "\\bP\\.?\\s*O\\.?\\s*BOX\\b|\\bPOST OFFICE BOX\\b",
          ignore_case = TRUE
        )
      ),
      has_tx_zip = stringr::str_detect(
        dplyr::coalesce(.data$address_for_geocoding, ""),
        "\\bTX\\s+\\d{5}(-\\d{4})?\\b"
      ),
      has_placeholder_zip = stringr::str_detect(
        dplyr::coalesce(.data$address_for_geocoding, ""),
        "\\b[A-Z]{2}\\s+0{5}\\b"
      ),
      has_house_number = stringr::str_detect(
        dplyr::coalesce(.data$address_for_geocoding, ""),
        "^\\d+\\w?\\b"
      ),
      invalid_placeholder_address = stringr::str_detect(
        dplyr::coalesce(.data$address_for_geocoding, ""),
        stringr::regex(
          "\\b(TRANSIENT|UNKNOWN|HOMELESS|ADDRESS UNKNOWN|NO ADDRESS)\\b",
          ignore_case = TRUE
        )
      ) | .data$has_placeholder_zip,
      likely_out_of_state = !.data$missing_address &
        stringr::str_detect(
          dplyr::coalesce(.data$address_for_geocoding, ""),
          "\\b[A-Z]{2}\\s+\\d{5}(-\\d{4})?\\b"
        ) &
        !.data$has_tx_zip,
      geocoding_candidate = !.data$missing_address &
        !.data$po_box_address &
        !.data$invalid_placeholder_address &
        !.data$likely_out_of_state &
        .data$has_house_number,
      ingest_issue = dplyr::case_when(
        !.data$case_number_format_valid ~ "nonstandard_case_number_manual_review",
        .data$source_exact_duplicate ~ "exact_duplicate_source_row",
        .data$missing_address ~ "missing_address",
        TRUE ~ NA_character_
      )
    )

  duplicate_case_party_key <- paste(
    records$case_uid,
    records$defendant_name,
    sep = "\u001f"
  )
  records |>
    dplyr::mutate(
      duplicate_case_defendant = !is.na(.data$case_uid) &
        (duplicated(duplicate_case_party_key) |
          duplicated(duplicate_case_party_key, fromLast = TRUE)),
      source_record_hash = williamson_eviction_record_hash(
        .data$source_id,
        .data$source_row_start,
        .data$case_number_raw,
        .data$file_date,
        .data$status_raw,
        .data$plaintiff_name,
        .data$defendant_name,
        .data$address_raw
      )
    ) |>
    dplyr::select(
      "source_id", "source_group", "source_county", "source_file",
      "source_sheet", "source_row_start", "source_row_end",
      "source_period_start", "source_period_end", "source_record_id",
      "source_record_hash", "source_exact_duplicate", "court",
      "jp_district", "case_type", "case_uid", "case_number",
      "case_number_raw", "case_number_format_valid", "file_date",
      "outcome_year", "after_analysis_cutoff", "case_status", "status_raw",
      "plaintiff_name", "defendant_name", "address_line1_raw",
      "address_continuation_raw", "continuation_rows", "address_raw",
      "address_for_geocoding", "address_key", "missing_address",
      "po_box_address", "has_tx_zip", "has_placeholder_zip",
      "has_house_number", "invalid_placeholder_address",
      "likely_out_of_state", "geocoding_candidate", "duplicate_case_defendant",
      "ingest_issue"
    )
}
