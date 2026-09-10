################################################################################
# Historical City-Jurisdiction Coverage for Demolition Outcomes
################################################################################

demolition_coverage_baseline_required_columns <- c(
  "OBJECTID",
  "JURISDICTION_TYPE",
  "JURISDICTION_DATE"
)

demolition_coverage_action_required_columns <- c(
  "OBJECTID",
  "JURISDICTION_CASE_NUMBER",
  "ORDINANCE_NUMBER",
  "JURISDICTION_DESCRIPTION",
  "EFFECTIVE_DATE",
  "JURISDICTION_TYPE",
  "GLOBALID"
)

demolition_coverage_normalize_text <- function(value) {
  value <- toupper(trimws(as.character(value)))
  value[value == ""] <- NA_character_
  value
}

demolition_coverage_assert_columns <- function(data, required, data_name) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      data_name,
      " is missing required column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

demolition_coverage_assert_sf <- function(data, data_name) {
  if (!inherits(data, "sf")) {
    stop(data_name, " must be an sf object.", call. = FALSE)
  }
  if (nrow(data) == 0L) {
    stop(data_name, " has no rows.", call. = FALSE)
  }
  if (is.na(sf::st_crs(data))) {
    stop(data_name, " has no coordinate reference system.", call. = FALSE)
  }
  if (any(sf::st_is_empty(data))) {
    stop(data_name, " contains empty geometries.", call. = FALSE)
  }
  invisible(TRUE)
}

demolition_coverage_parse_baseline_date <- function(value) {
  text <- trimws(as.character(value))
  text[text == ""] <- NA_character_
  result <- rep(as.Date(NA), length(text))

  iso <- !is.na(text) & grepl("^[0-9]{4}-[0-9]{2}-[0-9]{2}$", text)
  result[iso] <- suppressWarnings(as.Date(text[iso]))

  # The City baseline layer stores its four historical dates as MMDDYY. All
  # baseline components predate 1970, so an explicit 1900s century avoids the
  # platform-dependent two-digit-year pivot that would turn 1968 into 2068.
  compact <- !is.na(text) & grepl("^[0-9]{6}$", text)
  if (any(compact)) {
    compact_text <- text[compact]
    compact_iso <- paste0(
      1900L + as.integer(substr(compact_text, 5L, 6L)),
      "-",
      substr(compact_text, 1L, 2L),
      "-",
      substr(compact_text, 3L, 4L)
    )
    result[compact] <- suppressWarnings(as.Date(compact_iso))
  }

  unparsed <- !is.na(text) & is.na(result)
  if (any(unparsed)) {
    stop(
      "Could not parse historical baseline date value(s): ",
      paste(sort(unique(text[unparsed])), collapse = ", "),
      call. = FALSE
    )
  }
  result
}

demolition_coverage_parse_action_date <- function(
  value,
  timezone = "America/Chicago"
) {
  if (!is.character(timezone) || length(timezone) != 1L || is.na(timezone)) {
    stop("timezone must be one non-missing character value.", call. = FALSE)
  }
  if (!timezone %in% OlsonNames()) {
    stop("Unknown timezone: ", timezone, call. = FALSE)
  }

  if (inherits(value, "Date")) {
    return(as.Date(value))
  }
  if (inherits(value, "POSIXt")) {
    return(as.Date(value, tz = timezone))
  }

  if (is.numeric(value)) {
    numeric_value <- as.numeric(value)
    result <- rep(as.Date(NA), length(numeric_value))
    valid <- is.finite(numeric_value)
    # ArcGIS JSON/GeoJSON encodes dates as epoch milliseconds. Supporting
    # epoch seconds as well makes the parser safe for alternate readers.
    seconds <- numeric_value[valid]
    # ArcGIS returns one numeric date field in one unit. Detect the unit for the
    # vector as a whole: per-value magnitude misclassifies pre-1967 epoch
    # milliseconds (and dates close to 1970) as seconds.
    numeric_field_is_milliseconds <- any(abs(seconds) >= 1e10)
    if (numeric_field_is_milliseconds) seconds <- seconds / 1000
    timestamp <- as.POSIXct(seconds, origin = "1970-01-01", tz = timezone)
    result[valid] <- as.Date(timestamp, tz = timezone)
    return(result)
  }

  text <- trimws(as.character(value))
  text[text == ""] <- NA_character_
  numeric_text <- !is.na(text) & grepl("^-?[0-9]+(?:\\.[0-9]+)?$", text)
  result <- rep(as.Date(NA), length(text))
  if (any(numeric_text)) {
    result[numeric_text] <- demolition_coverage_parse_action_date(
      as.numeric(text[numeric_text]),
      timezone = timezone
    )
  }
  date_text <- !is.na(text) & !numeric_text
  if (any(date_text)) {
    result[date_text] <- suppressWarnings(as.Date(text[date_text]))
  }

  unparsed <- !is.na(text) & is.na(result)
  if (any(unparsed)) {
    stop(
      "Could not parse annexation effective-date value(s): ",
      paste(sort(unique(text[unparsed])), collapse = ", "),
      call. = FALSE
    )
  }
  result
}

demolition_coverage_optional_column <- function(data, column, default = NA) {
  if (column %in% names(data)) data[[column]] else rep(default, nrow(data))
}

demolition_coverage_prepare_events <- function(
  historical_baselines,
  dated_actions,
  analysis_crs = 5070,
  timezone = "America/Chicago"
) {
  demolition_coverage_assert_sf(
    historical_baselines,
    "Historical jurisdiction baselines"
  )
  demolition_coverage_assert_sf(dated_actions, "Dated jurisdiction actions")
  demolition_coverage_assert_columns(
    historical_baselines,
    demolition_coverage_baseline_required_columns,
    "Historical jurisdiction baselines"
  )
  demolition_coverage_assert_columns(
    dated_actions,
    demolition_coverage_action_required_columns,
    "Dated jurisdiction actions"
  )

  baseline_dates <- demolition_coverage_parse_baseline_date(
    historical_baselines$JURISDICTION_DATE
  )
  action_dates <- demolition_coverage_parse_action_date(
    dated_actions$EFFECTIVE_DATE,
    timezone = timezone
  )

  baseline_events <- historical_baselines |>
    sf::st_make_valid() |>
    sf::st_transform(analysis_crs) |>
    dplyr::transmute(
      event_source = "historical_baseline",
      source_precedence = 1L,
      event_object_id = as.character(.data$OBJECTID),
      event_global_id = as.character(demolition_coverage_optional_column(
        historical_baselines,
        "GLOBALID",
        NA_character_
      )),
      jurisdiction_case_number = NA_character_,
      ordinance_number = NA_character_,
      jurisdiction_description = paste0(
        "Historical City jurisdiction baseline dated ",
        baseline_dates
      ),
      jurisdiction_type = demolition_coverage_normalize_text(
        .data$JURISDICTION_TYPE
      ),
      effective_date = baseline_dates
    )

  action_events <- dated_actions |>
    sf::st_make_valid() |>
    sf::st_transform(analysis_crs) |>
    dplyr::transmute(
      event_source = "dated_action",
      source_precedence = 2L,
      event_object_id = as.character(.data$OBJECTID),
      event_global_id = as.character(.data$GLOBALID),
      jurisdiction_case_number = as.character(
        .data$JURISDICTION_CASE_NUMBER
      ),
      ordinance_number = as.character(.data$ORDINANCE_NUMBER),
      jurisdiction_description = as.character(
        .data$JURISDICTION_DESCRIPTION
      ),
      jurisdiction_type = demolition_coverage_normalize_text(
        .data$JURISDICTION_TYPE
      ),
      effective_date = action_dates
    )

  events <- rbind(baseline_events, action_events)
  events$event_record_id <- paste(
    events$event_source,
    events$event_object_id,
    sep = ":"
  )
  if (anyNA(events$event_object_id) || any(events$event_object_id == "")) {
    stop("Jurisdiction events contain missing OBJECTID values.", call. = FALSE)
  }
  if (anyDuplicated(events$event_record_id)) {
    stop("Jurisdiction event record identifiers are not unique.", call. = FALSE)
  }

  events
}

demolition_coverage_collapse_aligned <- function(value) {
  value <- trimws(as.character(value))
  value[is.na(value) | value == ""] <- "<NA>"
  paste(value, collapse = "||")
}

demolition_coverage_resolve_state <- function(
  event_indexes,
  events,
  cutoff_date,
  supported_jurisdiction_types
) {
  event_indexes <- event_indexes[
    !is.na(events$effective_date[event_indexes]) &
      events$effective_date[event_indexes] <= cutoff_date
  ]

  if (length(event_indexes) == 0L) {
    return(list(
      source_covered = NA,
      coverage_jurisdiction_type = NA_character_,
      coverage_candidate_types = NA_character_,
      coverage_status = "unresolved_no_effective_dated_record",
      coverage_tied = FALSE,
      coverage_ambiguous = FALSE,
      latest_effective_date = as.Date(NA),
      latest_record_count = 0L,
      latest_distinct_type_count = 0L,
      latest_record_sources = NA_character_,
      latest_record_ids = NA_character_,
      latest_event_types = NA_character_,
      latest_case_numbers = NA_character_,
      latest_ordinance_numbers = NA_character_,
      latest_jurisdiction_descriptions = NA_character_,
      latest_global_ids = NA_character_
    ))
  }

  latest_date <- max(events$effective_date[event_indexes])
  latest_indexes <- event_indexes[
    events$effective_date[event_indexes] == latest_date
  ]
  # An effective-dated action is a state change and therefore takes precedence
  # over a historical baseline component on the same date.
  highest_precedence <- max(events$source_precedence[latest_indexes])
  latest_indexes <- latest_indexes[
    events$source_precedence[latest_indexes] == highest_precedence
  ]
  latest_indexes <- latest_indexes[order(events$event_record_id[latest_indexes])]

  raw_types <- events$jurisdiction_type[latest_indexes]
  valid_types <- sort(unique(stats::na.omit(raw_types)))
  has_missing_type <- anyNA(raw_types)
  distinct_type_count <- length(valid_types)
  tied <- length(latest_indexes) > 1L
  ambiguous <- has_missing_type || distinct_type_count > 1L

  resolved_type <- if (!has_missing_type && distinct_type_count == 1L) {
    valid_types[[1]]
  } else {
    NA_character_
  }
  coverage_values <- valid_types %in% supported_jurisdiction_types
  source_covered <- if (has_missing_type || length(coverage_values) == 0L) {
    NA
  } else if (length(unique(coverage_values)) == 1L) {
    coverage_values[[1]]
  } else {
    NA
  }

  coverage_status <- if (has_missing_type) {
    "ambiguous_latest_record_missing_type"
  } else if (distinct_type_count > 1L) {
    "ambiguous_latest_records_conflicting_types"
  } else if (tied) {
    "resolved_tied_records_same_type"
  } else {
    "resolved_single_record"
  }

  list(
    source_covered = source_covered,
    coverage_jurisdiction_type = resolved_type,
    coverage_candidate_types = if (length(valid_types) == 0L) {
      NA_character_
    } else {
      paste(valid_types, collapse = "|")
    },
    coverage_status = coverage_status,
    coverage_tied = tied,
    coverage_ambiguous = ambiguous,
    latest_effective_date = latest_date,
    latest_record_count = length(latest_indexes),
    latest_distinct_type_count = distinct_type_count,
    # Double pipes delimit aligned values in every metadata column. Missing
    # values use an explicit token so records in a tie can be reconstructed.
    latest_record_sources = demolition_coverage_collapse_aligned(
      events$event_source[latest_indexes]
    ),
    latest_record_ids = demolition_coverage_collapse_aligned(
      events$event_record_id[latest_indexes]
    ),
    latest_event_types = demolition_coverage_collapse_aligned(raw_types),
    latest_case_numbers = demolition_coverage_collapse_aligned(
      events$jurisdiction_case_number[latest_indexes]
    ),
    latest_ordinance_numbers = demolition_coverage_collapse_aligned(
      events$ordinance_number[latest_indexes]
    ),
    latest_jurisdiction_descriptions = demolition_coverage_collapse_aligned(
      events$jurisdiction_description[latest_indexes]
    ),
    latest_global_ids = demolition_coverage_collapse_aligned(
      events$event_global_id[latest_indexes]
    )
  )
}

demolition_coverage_first_date <- function(dates, condition) {
  matching_dates <- dates[condition]
  if (length(matching_dates) == 0L) as.Date(NA) else min(matching_dates)
}

demolition_coverage_period_candidate_types <- function(values) {
  values <- stats::na.omit(as.character(values))
  if (length(values) == 0L) return(NA_character_)
  types <- sort(unique(unlist(strsplit(values, "|", fixed = TRUE))))
  types <- types[!is.na(types) & types != ""]
  if (length(types) == 0L) NA_character_ else paste(types, collapse = "|")
}

# Resolve the jurisdiction state on the first observed day and immediately
# after every effective-dated event through the final observed day. A period is
# covered only when every resulting state is both resolved and supported.
demolition_coverage_resolve_period <- function(
  event_indexes,
  events,
  period_start_date,
  period_end_date,
  supported_jurisdiction_types
) {
  period_start_date <- as.Date(period_start_date)
  period_end_date <- as.Date(period_end_date)
  if (period_start_date > period_end_date) {
    stop("Coverage period start cannot follow its end.", call. = FALSE)
  }

  event_dates <- events$effective_date[event_indexes]
  change_dates <- sort(unique(event_dates[
    !is.na(event_dates) &
      event_dates > period_start_date &
      event_dates <= period_end_date
  ]))
  checkpoint_dates <- c(period_start_date, change_dates)
  checkpoint_states <- lapply(
    checkpoint_dates,
    function(checkpoint_date) {
      demolition_coverage_resolve_state(
        event_indexes = event_indexes,
        events = events,
        cutoff_date = checkpoint_date,
        supported_jurisdiction_types = supported_jurisdiction_types
      )
    }
  )
  checkpoint_states <- dplyr::bind_rows(checkpoint_states) |>
    dplyr::mutate(checkpoint_date = checkpoint_dates, .before = 1L)

  period_resolved <- !anyNA(checkpoint_states$source_covered)
  period_covered <- if (period_resolved) {
    all(checkpoint_states$source_covered)
  } else {
    NA
  }
  resolved_types <- sort(unique(stats::na.omit(
    checkpoint_states$coverage_jurisdiction_type
  )))
  period_type <- if (period_resolved && length(resolved_types) == 1L) {
    resolved_types[[1]]
  } else {
    NA_character_
  }
  period_status <- if (!period_resolved) {
    "unresolved_during_observed_period"
  } else if (period_covered) {
    "resolved_supported_continuously"
  } else if (all(!checkpoint_states$source_covered)) {
    "resolved_unsupported_continuously"
  } else {
    "resolved_supported_only_part_of_period"
  }

  end_state <- demolition_coverage_resolve_state(
    event_indexes = event_indexes,
    events = events,
    cutoff_date = period_end_date,
    supported_jurisdiction_types = supported_jurisdiction_types
  )

  # The unprefixed latest-record fields remain explicit aliases of end-state
  # metadata for backward compatibility. Period-level classification fields
  # never overwrite the independently resolved end state used by snapshot QA.
  list(
    source_covered = period_covered,
    coverage_period_resolved = period_resolved,
    coverage_jurisdiction_type = period_type,
    coverage_candidate_types = demolition_coverage_period_candidate_types(
      checkpoint_states$coverage_candidate_types
    ),
    coverage_status = period_status,
    coverage_tied = any(checkpoint_states$coverage_tied),
    coverage_ambiguous = any(checkpoint_states$coverage_ambiguous),
    period_checkpoint_count = nrow(checkpoint_states),
    period_effective_change_date_count = length(change_dates),
    period_first_failure_date = demolition_coverage_first_date(
      checkpoint_states$checkpoint_date,
      is.na(checkpoint_states$source_covered) |
        !checkpoint_states$source_covered
    ),
    period_first_uncovered_date = demolition_coverage_first_date(
      checkpoint_states$checkpoint_date,
      checkpoint_states$source_covered %in% FALSE
    ),
    period_first_unresolved_date = demolition_coverage_first_date(
      checkpoint_states$checkpoint_date,
      is.na(checkpoint_states$source_covered)
    ),
    end_state_source_covered = end_state$source_covered,
    end_state_jurisdiction_type = end_state$coverage_jurisdiction_type,
    end_state_candidate_types = end_state$coverage_candidate_types,
    end_state_status = end_state$coverage_status,
    end_state_tied = end_state$coverage_tied,
    end_state_ambiguous = end_state$coverage_ambiguous,
    end_state_latest_effective_date = end_state$latest_effective_date,
    end_state_latest_record_count = end_state$latest_record_count,
    end_state_latest_distinct_type_count =
      end_state$latest_distinct_type_count,
    end_state_latest_record_sources = end_state$latest_record_sources,
    end_state_latest_record_ids = end_state$latest_record_ids,
    end_state_latest_event_types = end_state$latest_event_types,
    end_state_latest_case_numbers = end_state$latest_case_numbers,
    end_state_latest_ordinance_numbers = end_state$latest_ordinance_numbers,
    end_state_latest_jurisdiction_descriptions =
      end_state$latest_jurisdiction_descriptions,
    end_state_latest_global_ids = end_state$latest_global_ids,
    latest_effective_date = end_state$latest_effective_date,
    latest_record_count = end_state$latest_record_count,
    latest_distinct_type_count = end_state$latest_distinct_type_count,
    latest_record_sources = end_state$latest_record_sources,
    latest_record_ids = end_state$latest_record_ids,
    latest_event_types = end_state$latest_event_types,
    latest_case_numbers = end_state$latest_case_numbers,
    latest_ordinance_numbers = end_state$latest_ordinance_numbers,
    latest_jurisdiction_descriptions =
      end_state$latest_jurisdiction_descriptions,
    latest_global_ids = end_state$latest_global_ids
  )
}

validate_demolition_historical_coverage <- function(
  coverage,
  hex_ids,
  years,
  observed_through_date = NULL,
  source_start_date = NULL
) {
  required_columns <- c(
    "hex_id",
    "outcome_year",
    "coverage_period_start_date",
    "coverage_period_end_date",
    "coverage_as_of_date",
    "coverage_period_complete",
    "coverage_start_capped",
    "coverage_cutoff_capped",
    "source_covered",
    "coverage_period_resolved",
    "coverage_jurisdiction_type",
    "coverage_candidate_types",
    "coverage_status",
    "coverage_tied",
    "coverage_ambiguous",
    "period_checkpoint_count",
    "period_effective_change_date_count",
    "period_first_failure_date",
    "period_first_uncovered_date",
    "period_first_unresolved_date",
    "end_state_source_covered",
    "end_state_jurisdiction_type",
    "end_state_candidate_types",
    "end_state_status",
    "end_state_tied",
    "end_state_ambiguous",
    "end_state_latest_effective_date",
    "end_state_latest_record_count",
    "end_state_latest_distinct_type_count",
    "end_state_latest_record_sources",
    "end_state_latest_record_ids",
    "end_state_latest_event_types",
    "end_state_latest_case_numbers",
    "end_state_latest_ordinance_numbers",
    "end_state_latest_jurisdiction_descriptions",
    "end_state_latest_global_ids",
    "latest_effective_date",
    "latest_record_count",
    "latest_distinct_type_count",
    "latest_record_sources",
    "latest_record_ids",
    "latest_event_types",
    "latest_case_numbers",
    "latest_ordinance_numbers",
    "latest_jurisdiction_descriptions",
    "latest_global_ids",
    "coverage_basis"
  )
  demolition_coverage_assert_columns(
    coverage,
    required_columns,
    "Historical demolition coverage"
  )

  years <- sort(unique(as.integer(years)))
  hex_ids <- unique(hex_ids)
  expected_rows <- length(hex_ids) * length(years)
  if (nrow(coverage) != expected_rows) {
    stop(
      "Historical demolition coverage is incomplete: expected ",
      expected_rows,
      " rows but found ",
      nrow(coverage),
      ".",
      call. = FALSE
    )
  }
  if (anyNA(coverage[c(
    "hex_id",
    "outcome_year",
    "coverage_period_start_date",
    "coverage_period_end_date",
    "coverage_as_of_date"
  )])) {
    stop("Coverage keys and period dates cannot be missing.", call. = FALSE)
  }
  if (anyDuplicated(coverage[c("hex_id", "outcome_year")])) {
    stop("Historical demolition coverage has duplicate hex-year rows.", call. = FALSE)
  }
  if (!setequal(coverage$hex_id, hex_ids)) {
    stop("Historical demolition coverage has unexpected hex IDs.", call. = FALSE)
  }
  if (!setequal(as.integer(coverage$outcome_year), years)) {
    stop("Historical demolition coverage has unexpected years.", call. = FALSE)
  }

  logical_columns <- c(
    "coverage_period_complete",
    "coverage_start_capped",
    "coverage_cutoff_capped",
    "source_covered",
    "coverage_period_resolved",
    "coverage_tied",
    "coverage_ambiguous",
    "end_state_source_covered",
    "end_state_tied",
    "end_state_ambiguous"
  )
  non_logical <- logical_columns[
    !vapply(coverage[logical_columns], is.logical, logical(1))
  ]
  if (length(non_logical) > 0L) {
    stop(
      "Coverage flags must be logical: ",
      paste(non_logical, collapse = ", "),
      call. = FALSE
    )
  }
  if (anyNA(coverage[c(
    "coverage_period_complete",
    "coverage_start_capped",
    "coverage_cutoff_capped",
    "coverage_period_resolved",
    "coverage_tied",
    "coverage_ambiguous",
    "end_state_tied",
    "end_state_ambiguous"
  )])) {
    stop("Non-coverage status flags cannot be missing.", call. = FALSE)
  }

  year_start <- as.Date(paste0(coverage$outcome_year, "-01-01"))
  year_end <- as.Date(paste0(coverage$outcome_year, "-12-31"))
  expected_start <- year_start
  if (!is.null(source_start_date)) {
    source_start_date <- as.Date(source_start_date)
    expected_start <- pmax(year_start, source_start_date)
  }
  expected_end <- year_end
  if (!is.null(observed_through_date)) {
    observed_through_date <- as.Date(observed_through_date)
    expected_end <- pmin(year_end, observed_through_date)
  }
  if (any(expected_start > expected_end)) {
    stop("Requested years include no observable coverage period.", call. = FALSE)
  }
  if (!identical(as.Date(coverage$coverage_period_start_date), expected_start)) {
    stop("Historical coverage period start dates are incorrect.", call. = FALSE)
  }
  if (!identical(as.Date(coverage$coverage_period_end_date), expected_end)) {
    stop("Historical coverage period end dates are incorrect.", call. = FALSE)
  }
  if (!identical(as.Date(coverage$coverage_as_of_date), expected_end)) {
    stop("Historical coverage cutoff dates are incorrect.", call. = FALSE)
  }
  if (!identical(
    coverage$coverage_period_complete,
    expected_start == year_start & expected_end == year_end
  )) {
    stop("Historical coverage completeness flags are incorrect.", call. = FALSE)
  }
  if (!identical(coverage$coverage_start_capped, expected_start > year_start)) {
    stop("Historical coverage source-start cap flags are incorrect.", call. = FALSE)
  }
  if (!identical(coverage$coverage_cutoff_capped, expected_end < year_end)) {
    stop("Historical coverage cap flags are incorrect.", call. = FALSE)
  }
  if (!identical(
    coverage$coverage_period_resolved,
    !is.na(coverage$source_covered)
  )) {
    stop("Period coverage resolution flags are inconsistent.", call. = FALSE)
  }
  if (any(
    coverage$period_checkpoint_count < 1L |
      coverage$period_effective_change_date_count < 0L |
      coverage$period_checkpoint_count !=
        coverage$period_effective_change_date_count + 1L
  )) {
    stop("Coverage checkpoint counts are inconsistent.", call. = FALSE)
  }
  failure_date_columns <- c(
    "period_first_failure_date",
    "period_first_uncovered_date",
    "period_first_unresolved_date"
  )
  for (column in failure_date_columns) {
    value <- as.Date(coverage[[column]])
    outside_period <- !is.na(value) &
      (value < expected_start | value > expected_end)
    if (any(outside_period)) {
      stop(column, " contains a date outside its coverage period.", call. = FALSE)
    }
  }
  if (any(coverage$source_covered %in% TRUE &
    !is.na(coverage$period_first_failure_date))) {
    stop("Continuously covered periods contain a failure date.", call. = FALSE)
  }
  if (any(coverage$source_covered %in% FALSE &
    is.na(coverage$period_first_uncovered_date))) {
    stop("Resolved uncovered periods lack an uncovered date.", call. = FALSE)
  }
  if (any(!coverage$coverage_period_resolved &
    is.na(coverage$period_first_unresolved_date))) {
    stop("Unresolved periods lack an unresolved date.", call. = FALSE)
  }
  if (any(
    coverage$latest_record_count == 0L &
      (!is.na(coverage$source_covered) |
        !is.na(coverage$coverage_jurisdiction_type))
  )) {
    stop("Unresolved coverage rows contain a resolved state.", call. = FALSE)
  }
  if (any(
    coverage$end_state_ambiguous &
      !is.na(coverage$end_state_jurisdiction_type)
  )) {
    stop("Ambiguous end states contain a resolved type.", call. = FALSE)
  }
  if (!identical(
    coverage$latest_effective_date,
    coverage$end_state_latest_effective_date
  ) || !identical(
    coverage$latest_record_count,
    coverage$end_state_latest_record_count
  )) {
    stop("Legacy latest-record aliases differ from end-state metadata.", call. = FALSE)
  }
  if (any(
    coverage$end_state_latest_effective_date >
      coverage$coverage_period_end_date,
    na.rm = TRUE
  )) {
    stop("Coverage contains an action after its annual cutoff.", call. = FALSE)
  }

  invisible(TRUE)
}

#' Replay effective-dated City jurisdiction actions to a complete hex-year grid.
#'
#' @param hex_grid Analysis hexagons as an sf object with a unique `hex_id`.
#' @param historical_baselines City historical-jurisdiction baseline polygons.
#' @param dated_actions City annexation-history action polygons.
#' @param years Calendar years to return.
#' @param observed_through_date Optional latest observable date. Each annual
#'   period ends at `min(December 31, observed_through_date)`, preventing
#'   actions after a partial final-year cutoff from entering coverage.
#' @param source_start_date Optional first observable permit-source date. Each
#'   annual period starts at `max(January 1, source_start_date)`.
#' @param supported_jurisdiction_types Types treated as covered by the permit
#'   source. ETJ variants are intentionally opt-in rather than silently assumed.
#' @return A complete, non-spatial tibble with one row per hex and year.
#'   `source_covered` is true only when every jurisdiction state throughout the
#'   observed period is resolved and supported. `end_state_*` fields preserve
#'   the independently resolved state on the period's final day for QA.
build_demolition_historical_hex_coverage <- function(
  hex_grid,
  historical_baselines,
  dated_actions,
  years,
  observed_through_date = NULL,
  source_start_date = NULL,
  supported_jurisdiction_types = c("FULL", "LTD", "2MILE"),
  analysis_crs = 5070,
  timezone = "America/Chicago"
) {
  demolition_coverage_assert_sf(hex_grid, "Hex grid")
  demolition_coverage_assert_columns(hex_grid, "hex_id", "Hex grid")
  if (anyNA(hex_grid$hex_id) || anyDuplicated(hex_grid$hex_id)) {
    stop("Hex grid must have unique, non-missing hex_id values.", call. = FALSE)
  }

  years <- sort(unique(as.integer(years)))
  if (length(years) == 0L || anyNA(years)) {
    stop("years must contain one or more valid integer years.", call. = FALSE)
  }
  if (any(years < 1900L | years > 9999L)) {
    stop("years contains an unsupported calendar year.", call. = FALSE)
  }

  supported_jurisdiction_types <- demolition_coverage_normalize_text(
    supported_jurisdiction_types
  )
  supported_jurisdiction_types <- sort(unique(stats::na.omit(
    supported_jurisdiction_types
  )))
  if (length(supported_jurisdiction_types) == 0L) {
    stop(
      "supported_jurisdiction_types must contain at least one type.",
      call. = FALSE
    )
  }

  if (!is.null(observed_through_date)) {
    if (length(observed_through_date) != 1L) {
      stop("observed_through_date must have length one.", call. = FALSE)
    }
    observed_through_date <- as.Date(observed_through_date)
    if (is.na(observed_through_date)) {
      stop("observed_through_date must be a valid date.", call. = FALSE)
    }
  }

  if (!is.null(source_start_date)) {
    if (length(source_start_date) != 1L) {
      stop("source_start_date must have length one.", call. = FALSE)
    }
    source_start_date <- as.Date(source_start_date)
    if (is.na(source_start_date)) {
      stop("source_start_date must be a valid date.", call. = FALSE)
    }
  }

  requested_year_start <- as.Date(paste0(years, "-01-01"))
  requested_year_end <- as.Date(paste0(years, "-12-31"))
  requested_period_start <- requested_year_start
  requested_period_end <- requested_year_end
  if (!is.null(source_start_date)) {
    requested_period_start <- pmax(
      requested_period_start,
      source_start_date
    )
  }
  if (!is.null(observed_through_date)) {
    requested_period_end <- pmin(
      requested_period_end,
      observed_through_date
    )
  }
  if (any(requested_period_start > requested_period_end)) {
    invalid_years <- years[requested_period_start > requested_period_end]
    stop(
      "Requested year(s) have no dates inside the observable source window: ",
      paste(invalid_years, collapse = ", "),
      ".",
      call. = FALSE
    )
  }

  events <- demolition_coverage_prepare_events(
    historical_baselines = historical_baselines,
    dated_actions = dated_actions,
    analysis_crs = analysis_crs,
    timezone = timezone
  )
  hex_points <- suppressWarnings(sf::st_point_on_surface(
    hex_grid |>
      sf::st_make_valid() |>
      sf::st_transform(analysis_crs)
  ))
  event_matches <- sf::st_intersects(hex_points, events)

  annual_rows <- lapply(years, function(outcome_year) {
    year_start <- as.Date(paste0(outcome_year, "-01-01"))
    year_end <- as.Date(paste0(outcome_year, "-12-31"))
    period_start_date <- year_start
    if (!is.null(source_start_date)) {
      period_start_date <- max(year_start, source_start_date)
    }
    period_end_date <- year_end
    if (!is.null(observed_through_date)) {
      period_end_date <- min(year_end, observed_through_date)
    }
    states <- lapply(event_matches, demolition_coverage_resolve_period,
      events = events,
      period_start_date = period_start_date,
      period_end_date = period_end_date,
      supported_jurisdiction_types = supported_jurisdiction_types
    )
    state_rows <- dplyr::bind_rows(states)
    dplyr::bind_cols(
      tibble::tibble(
        .hex_order = seq_len(nrow(hex_grid)),
        hex_id = hex_grid$hex_id,
        outcome_year = outcome_year,
        coverage_period_start_date = period_start_date,
        coverage_period_end_date = period_end_date,
        coverage_as_of_date = period_end_date,
        coverage_period_complete =
          period_start_date == year_start & period_end_date == year_end,
        coverage_start_capped = period_start_date > year_start,
        coverage_cutoff_capped = period_end_date < year_end
      ),
      state_rows
    )
  })

  coverage <- dplyr::bind_rows(annual_rows) |>
    dplyr::mutate(
      coverage_basis =
        "effective_dated_city_jurisdiction_continuous_period_replay"
    ) |>
    dplyr::arrange(.data$.hex_order, .data$outcome_year) |>
    dplyr::select(-dplyr::all_of(".hex_order"))

  validate_demolition_historical_coverage(
    coverage = coverage,
    hex_ids = hex_grid$hex_id,
    years = years,
    observed_through_date = observed_through_date,
    source_start_date = source_start_date
  )
  attr(coverage, "supported_jurisdiction_types") <-
    supported_jurisdiction_types
  attr(coverage, "event_input_qa") <- tibble::tibble(
    baseline_records = nrow(historical_baselines),
    dated_action_records = nrow(dated_actions),
    dated_actions_missing_effective_date = sum(
      events$event_source == "dated_action" & is.na(events$effective_date)
    ),
    dated_actions_missing_jurisdiction_type = sum(
      events$event_source == "dated_action" & is.na(events$jurisdiction_type)
    )
  )
  coverage
}

demolition_coverage_classify_current_snapshot <- function(
  hex_grid,
  current_jurisdictions,
  supported_jurisdiction_types = c("FULL", "LTD", "2MILE"),
  city_name = "CITY OF AUSTIN",
  city_name_column = "city_name",
  jurisdiction_type_column = "jurisdiction_type",
  analysis_crs = 5070
) {
  demolition_coverage_assert_sf(hex_grid, "Hex grid")
  demolition_coverage_assert_columns(hex_grid, "hex_id", "Hex grid")
  demolition_coverage_assert_sf(current_jurisdictions, "Current jurisdictions")
  demolition_coverage_assert_columns(
    current_jurisdictions,
    c(city_name_column, jurisdiction_type_column),
    "Current jurisdictions"
  )

  supported_jurisdiction_types <- sort(unique(stats::na.omit(
    demolition_coverage_normalize_text(supported_jurisdiction_types)
  )))
  normalized_city_name <- demolition_coverage_normalize_text(city_name)
  city_values <- demolition_coverage_normalize_text(
    current_jurisdictions[[city_name_column]]
  )
  city_jurisdictions <- current_jurisdictions[
    !is.na(city_values) & city_values == normalized_city_name,
  ] |>
    sf::st_make_valid() |>
    sf::st_transform(analysis_crs)
  if (nrow(city_jurisdictions) == 0L) {
    stop("Current snapshot has no polygons for ", city_name, ".", call. = FALSE)
  }
  city_types <- demolition_coverage_normalize_text(
    city_jurisdictions[[jurisdiction_type_column]]
  )

  hex_points <- suppressWarnings(sf::st_point_on_surface(
    hex_grid |>
      sf::st_make_valid() |>
      sf::st_transform(analysis_crs)
  ))
  matches <- sf::st_intersects(hex_points, city_jurisdictions)
  rows <- lapply(matches, function(index) {
    types <- sort(unique(stats::na.omit(city_types[index])))
    has_missing_type <- anyNA(city_types[index])
    coverage_values <- types %in% supported_jurisdiction_types
    covered <- if (length(index) == 0L) {
      FALSE
    } else if (has_missing_type || length(coverage_values) == 0L) {
      NA
    } else if (length(unique(coverage_values)) == 1L) {
      coverage_values[[1]]
    } else {
      NA
    }
    tibble::tibble(
      current_source_covered = covered,
      current_jurisdiction_type = if (
        !has_missing_type && length(types) == 1L
      ) types[[1]] else NA_character_,
      current_candidate_types = if (length(types) > 0L) {
        paste(types, collapse = "|")
      } else {
        NA_character_
      },
      current_polygon_matches = length(index),
      current_ambiguous = has_missing_type || length(types) > 1L
    )
  })

  dplyr::bind_cols(
    tibble::tibble(hex_id = hex_grid$hex_id),
    dplyr::bind_rows(rows)
  )
}

#' Compare one replayed year with a current City jurisdiction snapshot.
#'
#' This is a QA comparison, not a historical fallback. Differences remain in
#' the returned detail table for review.
compare_demolition_coverage_to_current_snapshot <- function(
  coverage,
  hex_grid,
  current_jurisdictions,
  comparison_year = max(coverage$outcome_year),
  current_snapshot_as_of_date = NULL,
  supported_jurisdiction_types = c("FULL", "LTD", "2MILE"),
  city_name = "CITY OF AUSTIN",
  city_name_column = "city_name",
  jurisdiction_type_column = "jurisdiction_type",
  analysis_crs = 5070
) {
  demolition_coverage_assert_columns(
    coverage,
    c(
      "hex_id",
      "outcome_year",
      "coverage_as_of_date",
      "end_state_source_covered",
      "end_state_jurisdiction_type",
      "end_state_candidate_types",
      "end_state_status",
      "end_state_ambiguous"
    ),
    "Historical demolition coverage"
  )
  comparison_year <- as.integer(comparison_year)
  if (length(comparison_year) != 1L || is.na(comparison_year)) {
    stop("comparison_year must be one valid integer year.", call. = FALSE)
  }
  replay <- coverage |>
    dplyr::filter(.data$outcome_year == comparison_year)
  if (nrow(replay) == 0L) {
    stop("Coverage has no rows for comparison_year.", call. = FALSE)
  }
  if (anyDuplicated(replay$hex_id)) {
    stop("Coverage has duplicate hex rows for comparison_year.", call. = FALSE)
  }

  snapshot <- demolition_coverage_classify_current_snapshot(
    hex_grid = hex_grid,
    current_jurisdictions = current_jurisdictions,
    supported_jurisdiction_types = supported_jurisdiction_types,
    city_name = city_name,
    city_name_column = city_name_column,
    jurisdiction_type_column = jurisdiction_type_column,
    analysis_crs = analysis_crs
  )
  if (!is.null(current_snapshot_as_of_date)) {
    current_snapshot_as_of_date <- as.Date(current_snapshot_as_of_date)
    if (length(current_snapshot_as_of_date) != 1L ||
      is.na(current_snapshot_as_of_date)) {
      stop("current_snapshot_as_of_date must be a valid date.", call. = FALSE)
    }
  } else {
    current_snapshot_as_of_date <- as.Date(NA)
  }

  replay |>
    dplyr::transmute(
      hex_id = .data$hex_id,
      outcome_year = .data$outcome_year,
      replay_as_of_date = .data$coverage_as_of_date,
      replay_source_covered = .data$end_state_source_covered,
      replay_jurisdiction_type = .data$end_state_jurisdiction_type,
      replay_candidate_types = .data$end_state_candidate_types,
      replay_status = .data$end_state_status,
      replay_ambiguous = .data$end_state_ambiguous
    ) |>
    dplyr::left_join(snapshot, by = "hex_id") |>
    dplyr::mutate(
      current_snapshot_as_of_date = current_snapshot_as_of_date,
      cutoff_dates_aligned = dplyr::if_else(
        is.na(.data$current_snapshot_as_of_date),
        NA,
        .data$replay_as_of_date == .data$current_snapshot_as_of_date
      ),
      coverage_agreement = dplyr::if_else(
        is.na(.data$replay_source_covered) |
          is.na(.data$current_source_covered),
        NA,
        .data$replay_source_covered == .data$current_source_covered
      ),
      comparison_status = dplyr::case_when(
        is.na(.data$replay_source_covered) ~ "replay_unresolved",
        is.na(.data$current_source_covered) ~ "current_snapshot_ambiguous",
        .data$coverage_agreement ~ "coverage_agrees",
        TRUE ~ "coverage_disagrees"
      )
    )
}

summarize_demolition_coverage_snapshot_qa <- function(comparison) {
  demolition_coverage_assert_columns(
    comparison,
    c(
      "hex_id",
      "outcome_year",
      "replay_source_covered",
      "current_source_covered",
      "coverage_agreement",
      "comparison_status"
    ),
    "Demolition coverage snapshot comparison"
  )
  comparison |>
    dplyr::summarise(
      outcome_year = dplyr::first(.data$outcome_year),
      hexes = dplyr::n(),
      replay_covered = sum(.data$replay_source_covered %in% TRUE),
      replay_uncovered = sum(.data$replay_source_covered %in% FALSE),
      replay_unresolved = sum(is.na(.data$replay_source_covered)),
      current_covered = sum(.data$current_source_covered %in% TRUE),
      current_uncovered = sum(.data$current_source_covered %in% FALSE),
      current_ambiguous = sum(is.na(.data$current_source_covered)),
      comparable_hexes = sum(!is.na(.data$coverage_agreement)),
      agreeing_hexes = sum(.data$coverage_agreement %in% TRUE),
      disagreeing_hexes = sum(.data$coverage_agreement %in% FALSE),
      agreement_rate = mean(.data$coverage_agreement, na.rm = TRUE),
      .groups = "drop"
    )
}
