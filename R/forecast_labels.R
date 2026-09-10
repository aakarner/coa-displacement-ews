################################################################################
# Part 3 Forward Outcome Labels
################################################################################

required_outcome_panel_columns <- function(count_column) {
  c(
    "hex_id",
    "outcome_year",
    count_column,
    "source_covered",
    "period_complete",
    "measurement_complete",
    "count_observed",
    "coverage_reason",
    "analysis_as_of_date"
  )
}

assert_required_columns <- function(data, required, object_name) {
  missing <- setdiff(required, names(data))
  if (length(missing) > 0L) {
    stop(
      object_name,
      " is missing required column(s): ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

is_nonnegative_whole_number <- function(value) {
  is.na(value) | (is.finite(value) & value >= 0 & value == floor(value))
}

validate_complete_outcome_panel <- function(
  panel,
  count_column,
  panel_name = "outcome panel"
) {
  assert_required_columns(
    panel,
    required_outcome_panel_columns(count_column),
    panel_name
  )

  if (nrow(panel) == 0L) {
    stop(panel_name, " has no rows.", call. = FALSE)
  }
  if (any(is.na(panel$hex_id)) || any(is.na(panel$outcome_year))) {
    stop(panel_name, " has missing hex_id or outcome_year values.", call. = FALSE)
  }
  if (anyDuplicated(panel[c("hex_id", "outcome_year")])) {
    stop(panel_name, " has duplicate hex-year rows.", call. = FALSE)
  }

  logical_columns <- c(
    "source_covered",
    "period_complete",
    "measurement_complete",
    "count_observed"
  )
  invalid_logical <- logical_columns[
    !vapply(panel[logical_columns], is.logical, logical(1))
  ]
  if (length(invalid_logical) > 0L) {
    stop(
      panel_name,
      " must store logical values in: ",
      paste(invalid_logical, collapse = ", "),
      call. = FALSE
    )
  }
  if (anyNA(panel[logical_columns])) {
    stop(panel_name, " has missing coverage/completeness flags.", call. = FALSE)
  }
  analysis_dates <- unique(as.Date(panel$analysis_as_of_date))
  if (length(analysis_dates) != 1L || anyNA(analysis_dates)) {
    stop(panel_name, " must record exactly one analysis_as_of_date.", call. = FALSE)
  }

  expected_observed <- panel$source_covered &
    panel$period_complete &
    panel$measurement_complete
  if (!identical(panel$count_observed, expected_observed)) {
    stop(
      panel_name,
      " must define count_observed as source_covered AND period_complete ",
      "AND measurement_complete.",
      call. = FALSE
    )
  }

  counts <- panel[[count_column]]
  if (!is.numeric(counts) || any(!is_nonnegative_whole_number(counts))) {
    stop(
      panel_name,
      " contains a negative, non-finite, or non-integer event count.",
      call. = FALSE
    )
  }
  if (any(panel$count_observed & is.na(counts))) {
    stop(panel_name, " has an observed label with a missing count.", call. = FALSE)
  }
  if (any(!panel$count_observed & !is.na(counts))) {
    stop(
      panel_name,
      " has a modeling count for an uncovered or incomplete period.",
      call. = FALSE
    )
  }

  hexes <- unique(panel$hex_id)
  years <- sort(unique(as.integer(panel$outcome_year)))
  if (length(years) == 0L || any(diff(years) != 1L)) {
    stop(panel_name, " outcome years must form one consecutive sequence.", call. = FALSE)
  }
  expected_rows <- length(hexes) * length(years)
  if (nrow(panel) != expected_rows) {
    stop(
      panel_name,
      " is not a complete hex-by-year grid: expected ",
      expected_rows,
      " rows but found ",
      nrow(panel),
      ".",
      call. = FALSE
    )
  }

  invisible(list(
    hexes = length(hexes),
    first_year = min(years),
    last_year = max(years),
    observed_rows = sum(panel$count_observed),
    analysis_as_of_date = analysis_dates[[1]]
  ))
}

build_expected_forecast_label_domain <- function(
  panel,
  count_column,
  outcome_id,
  horizons
) {
  validate_complete_outcome_panel(
    panel,
    count_column,
    paste0(outcome_id, " panel")
  )

  if (
    length(outcome_id) != 1L ||
      is.na(outcome_id) ||
      !nzchar(as.character(outcome_id))
  ) {
    stop("outcome_id must be one non-missing value.", call. = FALSE)
  }

  horizons <- sort(unique(as.integer(horizons)))
  if (length(horizons) == 0L || anyNA(horizons) || any(horizons < 1L)) {
    stop("Forecast horizons must be positive integers.", call. = FALSE)
  }

  complete_years <- sort(unique(panel$outcome_year[panel$period_complete]))
  if (length(complete_years) == 0L) {
    stop(outcome_id, " has no complete outcome periods.", call. = FALSE)
  }
  origin_years <- seq.int(
    min(complete_years) - 1L,
    max(panel$outcome_year) - 1L
  )

  tidyr::expand_grid(
    hex_id = sort(unique(panel$hex_id)),
    forecast_origin_year = origin_years,
    outcome_id = as.character(outcome_id),
    horizon_years = horizons
  ) |>
    dplyr::arrange(
      .data$outcome_id,
      .data$horizon_years,
      .data$forecast_origin_year,
      .data$hex_id
    )
}

build_forward_count_labels <- function(
  panel,
  count_column,
  outcome_id,
  horizons
) {
  expected_domain <- build_expected_forecast_label_domain(
    panel,
    count_column,
    outcome_id,
    horizons
  )

  horizons <- sort(unique(expected_domain$horizon_years))
  origin_years <- sort(unique(expected_domain$forecast_origin_year))
  panel_analysis_as_of_date <- unique(as.Date(panel$analysis_as_of_date))[[1]]

  coverage_by_hex <- panel |>
    dplyr::arrange(.data$hex_id, .data$outcome_year) |>
    dplyr::group_by(.data$hex_id) |>
    dplyr::slice_tail(n = 1L) |>
    dplyr::ungroup() |>
    dplyr::transmute(
      .data$hex_id,
      source_covered_at_panel_vintage = .data$source_covered,
      source_coverage_reference_year = .data$outcome_year
    )

  panel_for_join <- panel |>
    dplyr::select(
      "hex_id",
      "outcome_year",
      annual_source_covered = "source_covered",
      "period_complete",
      "measurement_complete",
      "count_observed",
      event_count = dplyr::all_of(count_column)
    ) |>
    dplyr::mutate(panel_row_present = TRUE)

  labels <- lapply(
    horizons,
    function(horizon) {
      required_periods <- tidyr::expand_grid(
        hex_id = unique(panel$hex_id),
        forecast_origin_year = origin_years,
        lead_year = seq_len(horizon)
      ) |>
        dplyr::mutate(
          outcome_year = .data$forecast_origin_year + .data$lead_year
        ) |>
        dplyr::left_join(
          coverage_by_hex,
          by = "hex_id"
        ) |>
        dplyr::left_join(
          panel_for_join,
          by = c("hex_id", "outcome_year")
        )

      required_periods |>
        dplyr::group_by(.data$hex_id, .data$forecast_origin_year) |>
        dplyr::summarise(
          required_years = dplyr::n(),
          panel_years_present = sum(.data$panel_row_present %in% TRUE),
          source_covered_at_panel_vintage = dplyr::first(
            .data$source_covered_at_panel_vintage
          ),
          source_coverage_reference_year = dplyr::first(
            .data$source_coverage_reference_year
          ),
          covered_years = sum(.data$annual_source_covered %in% TRUE),
          complete_years = sum(.data$period_complete %in% TRUE),
          measurement_complete_years = sum(
            .data$measurement_complete %in% TRUE
          ),
          observed_years = sum(.data$count_observed %in% TRUE),
          observed_event_sum = sum(.data$event_count, na.rm = TRUE),
          unavailable_years = paste(
            .data$outcome_year[!(.data$count_observed %in% TRUE)],
            collapse = "|"
          ),
          .groups = "drop"
        ) |>
        dplyr::mutate(
          outcome_id = outcome_id,
          horizon_years = as.integer(horizon),
          analysis_as_of_date = .env$panel_analysis_as_of_date,
          forecast_origin_date = as.Date(
            paste0(.data$forecast_origin_year, "-12-31")
          ),
          target_window_start = as.Date(
            paste0(.data$forecast_origin_year + 1L, "-01-01")
          ),
          target_window_end = as.Date(
            paste0(.data$forecast_origin_year + horizon, "-12-31")
          ),
          window_source_coverage_complete =
            .data$panel_years_present == .data$required_years &
            .data$covered_years == .data$required_years,
          period_complete =
            .data$panel_years_present == .data$required_years &
            .data$complete_years == .data$required_years,
          measurement_complete =
            .data$panel_years_present == .data$required_years &
            .data$measurement_complete_years == .data$required_years,
          label_observed =
            .data$panel_years_present == .data$required_years &
            .data$window_source_coverage_complete &
            .data$period_complete &
            .data$measurement_complete &
            .data$observed_years == .data$required_years,
          outcome_count = dplyr::if_else(
            .data$label_observed,
            as.integer(.data$observed_event_sum),
            NA_integer_
          ),
          unavailable_years = dplyr::if_else(
            .data$label_observed,
            NA_character_,
            .data$unavailable_years
          )
        ) |>
        dplyr::select(
          "hex_id",
          "forecast_origin_year",
          "forecast_origin_date",
          "analysis_as_of_date",
          "outcome_id",
          "horizon_years",
          "target_window_start",
          "target_window_end",
          "outcome_count",
          "label_observed",
          "source_covered_at_panel_vintage",
          "source_coverage_reference_year",
          "window_source_coverage_complete",
          "period_complete",
          "measurement_complete",
          "required_years",
          "panel_years_present",
          "covered_years",
          "complete_years",
          "measurement_complete_years",
          "observed_years",
          "unavailable_years"
        )
    }
  ) |>
    dplyr::bind_rows()

  labels |>
    dplyr::arrange(
      .data$forecast_origin_year,
      .data$hex_id,
      .data$outcome_id,
      .data$horizon_years
    )
}

validate_pilot_label_contract <- function(
  labels,
  outcome_spec,
  horizons,
  expected_analysis_as_of_date = NULL,
  expected_domain = NULL
) {
  assert_required_columns(
    outcome_spec,
    c("proxy_id", "pilot_scope", "horizons_years"),
    "forecast outcome specification"
  )
  assert_required_columns(
    labels,
    c(
      "hex_id", "forecast_origin_year", "outcome_id", "horizon_years",
      "forecast_origin_date", "analysis_as_of_date", "target_window_start",
      "target_window_end",
      "outcome_count", "label_observed",
      "source_covered_at_panel_vintage", "source_coverage_reference_year",
      "window_source_coverage_complete", "period_complete",
      "measurement_complete", "required_years", "panel_years_present",
      "covered_years", "complete_years", "measurement_complete_years",
      "observed_years", "unavailable_years"
    ),
    "forecast labels"
  )

  if (nrow(labels) == 0L) {
    stop("Forecast labels have no rows.", call. = FALSE)
  }
  if (is.null(expected_domain)) {
    stop(
      "A source-derived expected forecast label domain is required.",
      call. = FALSE
    )
  }
  label_analysis_dates <- unique(as.Date(labels$analysis_as_of_date))
  if (length(label_analysis_dates) != 1L || anyNA(label_analysis_dates)) {
    stop(
      "Forecast labels combine missing or inconsistent analysis_as_of_date values.",
      call. = FALSE
    )
  }
  if (
    !is.null(expected_analysis_as_of_date) &&
      label_analysis_dates[[1]] != as.Date(expected_analysis_as_of_date)
  ) {
    stop(
      "Forecast labels were built for ",
      label_analysis_dates[[1]],
      ", not the configured analysis date ",
      as.Date(expected_analysis_as_of_date),
      ".",
      call. = FALSE
    )
  }
  key_columns <- c(
    "hex_id", "forecast_origin_year", "outcome_id", "horizon_years"
  )
  if (anyNA(labels[key_columns])) {
    stop("Forecast labels have missing task-key values.", call. = FALSE)
  }
  if (anyDuplicated(
    labels[key_columns]
  )) {
    stop("Forecast labels have duplicate task rows.", call. = FALSE)
  }
  assert_required_columns(
    expected_domain,
    key_columns,
    "expected forecast label domain"
  )
  if (nrow(expected_domain) == 0L) {
    stop("Expected forecast label domain has no rows.", call. = FALSE)
  }
  if (anyNA(expected_domain[key_columns])) {
    stop(
      "Expected forecast label domain has missing task-key values.",
      call. = FALSE
    )
  }
  if (anyDuplicated(expected_domain[key_columns])) {
    stop(
      "Expected forecast label domain has duplicate task keys.",
      call. = FALSE
    )
  }

  logical_columns <- c(
    "label_observed", "source_covered_at_panel_vintage",
    "window_source_coverage_complete", "period_complete",
    "measurement_complete"
  )
  invalid_logical <- logical_columns[
    !vapply(labels[logical_columns], is.logical, logical(1))
  ]
  if (length(invalid_logical) > 0L || anyNA(labels[logical_columns])) {
    stop(
      "Forecast labels must have complete logical values in: ",
      paste(logical_columns, collapse = ", "),
      call. = FALSE
    )
  }

  integer_columns <- c(
    "forecast_origin_year", "horizon_years", "required_years",
    "source_coverage_reference_year",
    "panel_years_present", "covered_years", "complete_years",
    "measurement_complete_years", "observed_years"
  )
  invalid_integer <- vapply(
    labels[integer_columns],
    function(value) {
      !is.numeric(value) ||
        anyNA(value) ||
        any(!is.finite(value)) ||
        any(value != floor(value))
    },
    logical(1)
  )
  if (any(invalid_integer)) {
    stop(
      "Forecast labels contain invalid integer bookkeeping fields: ",
      paste(integer_columns[invalid_integer], collapse = ", "),
      call. = FALSE
    )
  }

  horizons <- sort(unique(as.integer(horizons)))
  if (
    length(horizons) == 0L ||
      anyNA(horizons) ||
      any(horizons < 1L)
  ) {
    stop("Configured forecast horizons must be positive integers.", call. = FALSE)
  }
  if (any(!labels$horizon_years %in% horizons)) {
    stop("Forecast labels contain an unconfigured horizon.", call. = FALSE)
  }
  if (any(labels$required_years != labels$horizon_years)) {
    stop("required_years must equal horizon_years for every label.", call. = FALSE)
  }
  bookkeeping_columns <- c(
    "panel_years_present", "covered_years", "complete_years",
    "measurement_complete_years", "observed_years"
  )
  if (any(vapply(
    labels[bookkeeping_columns],
    function(value) any(value < 0L | value > labels$required_years),
    logical(1)
  ))) {
    stop("Forecast label bookkeeping counts are outside [0, required_years].", call. = FALSE)
  }

  expected_origin_date <- as.Date(
    paste0(labels$forecast_origin_year, "-12-31")
  )
  if (anyNA(labels$forecast_origin_date) || any(
    as.Date(labels$forecast_origin_date) != expected_origin_date
  )) {
    stop(
      "Forecast origin dates must be December 31 of forecast_origin_year.",
      call. = FALSE
    )
  }
  expected_window_start <- as.Date(
    paste0(labels$forecast_origin_year + 1L, "-01-01")
  )
  expected_window_end <- as.Date(
    paste0(
      labels$forecast_origin_year + labels$horizon_years,
      "-12-31"
    )
  )
  if (
    anyNA(labels$target_window_start) ||
      anyNA(labels$target_window_end) ||
      any(as.Date(labels$target_window_start) != expected_window_start) ||
      any(as.Date(labels$target_window_end) != expected_window_end)
  ) {
    stop(
      "Forecast target windows do not match their origin and horizon.",
      call. = FALSE
    )
  }

  expected_window_coverage <-
    labels$panel_years_present == labels$required_years &
    labels$covered_years == labels$required_years
  expected_period_complete <-
    labels$panel_years_present == labels$required_years &
    labels$complete_years == labels$required_years
  expected_measurement_complete <-
    labels$panel_years_present == labels$required_years &
    labels$measurement_complete_years == labels$required_years
  expected_label_observed <-
    expected_window_coverage &
    expected_period_complete &
    expected_measurement_complete &
    labels$observed_years == labels$required_years

  if (!identical(
    labels$window_source_coverage_complete,
    expected_window_coverage
  )) {
    stop(
      "window_source_coverage_complete is inconsistent with annual coverage.",
      call. = FALSE
    )
  }
  if (!identical(labels$period_complete, expected_period_complete)) {
    stop("period_complete is inconsistent with annual completeness.", call. = FALSE)
  }
  if (!identical(
    labels$measurement_complete,
    expected_measurement_complete
  )) {
    stop(
      "measurement_complete is inconsistent with annual measurement flags.",
      call. = FALSE
    )
  }
  if (!identical(labels$label_observed, expected_label_observed)) {
    stop(
      "label_observed is inconsistent with coverage, period, or measurement flags.",
      call. = FALSE
    )
  }
  coverage_states <- labels |>
    dplyr::group_by(.data$outcome_id, .data$hex_id) |>
    dplyr::summarise(
      coverage_states = dplyr::n_distinct(
        .data$source_covered_at_panel_vintage
      ),
      reference_years = dplyr::n_distinct(
        .data$source_coverage_reference_year
      ),
      .groups = "drop"
    )
  if (any(
    coverage_states$coverage_states != 1L |
      coverage_states$reference_years != 1L
  )) {
    stop(
      "Panel-vintage source coverage must be stable across label rows for ",
      "each outcome/hex.",
      call. = FALSE
    )
  }
  if (any(labels$label_observed & is.na(labels$outcome_count))) {
    stop("An observed forecast label has a missing count.", call. = FALSE)
  }
  if (any(!labels$label_observed & !is.na(labels$outcome_count))) {
    stop("An unavailable forecast label has a non-missing count.", call. = FALSE)
  }
  if (any(!is_nonnegative_whole_number(labels$outcome_count))) {
    stop("Forecast labels contain invalid counts.", call. = FALSE)
  }
  if (any(labels$label_observed & !is.na(labels$unavailable_years))) {
    stop("Observed labels must not list unavailable years.", call. = FALSE)
  }
  if (any(!labels$label_observed & (
    is.na(labels$unavailable_years) | labels$unavailable_years == ""
  ))) {
    stop("Unavailable labels must list at least one unavailable year.", call. = FALSE)
  }

  pilot_outcomes <- outcome_spec$proxy_id[outcome_spec$pilot_scope == "pilot"]
  expected_tasks <- tidyr::expand_grid(
    outcome_id = pilot_outcomes,
    horizon_years = sort(unique(as.integer(horizons)))
  ) |>
    dplyr::arrange(.data$outcome_id, .data$horizon_years)
  actual_tasks <- labels |>
    dplyr::distinct(.data$outcome_id, .data$horizon_years) |>
    dplyr::arrange(.data$outcome_id, .data$horizon_years)

  domain_tasks <- expected_domain |>
    dplyr::distinct(.data$outcome_id, .data$horizon_years) |>
    dplyr::arrange(.data$outcome_id, .data$horizon_years)
  task_key_columns <- c("outcome_id", "horizon_years")
  task_sets_match <- function(left, right) {
    nrow(left) == nrow(right) &&
      nrow(dplyr::anti_join(left, right, by = task_key_columns)) == 0L &&
      nrow(dplyr::anti_join(right, left, by = task_key_columns)) == 0L
  }

  if (!task_sets_match(expected_tasks, domain_tasks)) {
    stop(
      "Expected forecast label domain does not match the configured pilot ",
      "outcome/horizon tasks.",
      call. = FALSE
    )
  }
  if (!task_sets_match(expected_tasks, actual_tasks)) {
    stop(
      "Forecast labels do not match the configured pilot outcome/horizon tasks.",
      call. = FALSE
    )
  }

  expected_keys <- expected_domain |>
    dplyr::select(dplyr::all_of(key_columns))
  actual_keys <- labels |>
    dplyr::select(dplyr::all_of(key_columns))
  missing_keys <- dplyr::anti_join(
    expected_keys,
    actual_keys,
    by = key_columns
  )
  unexpected_keys <- dplyr::anti_join(
    actual_keys,
    expected_keys,
    by = key_columns
  )
  if (nrow(missing_keys) > 0L || nrow(unexpected_keys) > 0L) {
    stop(
      "Forecast labels do not match the source-derived expected panel ",
      "domain: ",
      nrow(missing_keys),
      " expected task key(s) are missing and ",
      nrow(unexpected_keys),
      " unexpected task key(s) are present.",
      call. = FALSE
    )
  }

  task_groups <- split(
    expected_domain,
    interaction(
      expected_domain$outcome_id,
      expected_domain$horizon_years,
      drop = TRUE,
      lex.order = TRUE
    )
  )
  for (task in task_groups) {
    task_hexes <- unique(task$hex_id)
    task_origins <- sort(unique(task$forecast_origin_year))
    expected_rows <- length(task_hexes) * length(task_origins)
    if (nrow(task) != expected_rows) {
      stop(
        "Expected forecast task domain is not a complete hex-by-origin grid: ",
        task$outcome_id[[1]],
        " at ",
        task$horizon_years[[1]],
        " year(s).",
        call. = FALSE
      )
    }
    if (length(task_origins) == 0L || any(diff(task_origins) != 1L)) {
      stop(
        "Expected forecast task origin years must be consecutive.",
        call. = FALSE
      )
    }
  }

  for (outcome_id in pilot_outcomes) {
    outcome_labels <- labels[labels$outcome_id == outcome_id, , drop = FALSE]
    horizon_groups <- split(outcome_labels, outcome_labels$horizon_years)
    reference_keys <- horizon_groups[[1]][c("hex_id", "forecast_origin_year")]
    reference_keys <- reference_keys[order(
      reference_keys$forecast_origin_year,
      reference_keys$hex_id
    ), , drop = FALSE]
    for (horizon_group in horizon_groups[-1]) {
      comparison_keys <- horizon_group[c("hex_id", "forecast_origin_year")]
      comparison_keys <- comparison_keys[order(
        comparison_keys$forecast_origin_year,
        comparison_keys$hex_id
      ), , drop = FALSE]
      if (!identical(reference_keys, comparison_keys)) {
        stop(
          "Configured horizons do not share the same hex-origin grid for ",
          outcome_id,
          ".",
          call. = FALSE
        )
      }
    }

    cumulative_counts <- outcome_labels |>
      dplyr::filter(.data$label_observed) |>
      dplyr::select(
        "hex_id", "forecast_origin_year", "horizon_years", "outcome_count"
      ) |>
      dplyr::arrange(
        .data$hex_id,
        .data$forecast_origin_year,
        .data$horizon_years
      ) |>
      dplyr::group_by(.data$hex_id, .data$forecast_origin_year) |>
      dplyr::summarise(
        cumulative_non_decreasing = all(diff(.data$outcome_count) >= 0L),
        .groups = "drop"
      )
    if (any(!cumulative_counts$cumulative_non_decreasing)) {
      stop(
        "Longer cumulative-horizon counts are smaller than shorter counts for ",
        outcome_id,
        ".",
        call. = FALSE
      )
    }
  }

  origin_status <- labels |>
    dplyr::group_by(
      .data$outcome_id,
      .data$horizon_years,
      .data$forecast_origin_year
    ) |>
    dplyr::summarise(
      candidate_hexes = dplyr::n(),
      source_covered_hexes = sum(
        .data$window_source_coverage_complete
      ),
      observed_labels = sum(.data$label_observed),
      origin_has_observed_labels = .data$observed_labels > 0L,
      origin_has_full_covered_geography =
        .data$source_covered_hexes > 0L &
        .data$observed_labels == .data$source_covered_hexes,
      .groups = "drop"
    )

  task_status <- origin_status |>
    dplyr::group_by(.data$outcome_id, .data$horizon_years) |>
    dplyr::summarise(
      label_rows = sum(.data$candidate_hexes),
      observed_labels = sum(.data$observed_labels),
      origins_with_observed_labels = sum(.data$origin_has_observed_labels),
      origins_with_full_covered_geography = sum(
        .data$origin_has_full_covered_geography
      ),
      first_origin_with_observed_labels = if (
        any(.data$origin_has_observed_labels)
      ) {
        min(.data$forecast_origin_year[.data$origin_has_observed_labels])
      } else {
        NA_integer_
      },
      last_origin_with_observed_labels = if (
        any(.data$origin_has_observed_labels)
      ) {
        max(.data$forecast_origin_year[.data$origin_has_observed_labels])
      } else {
        NA_integer_
      },
      .groups = "drop"
    ) |>
    dplyr::mutate(
      check_status = dplyr::if_else(
        .data$observed_labels > 0L,
        "pass",
        "fail"
      )
    )

  if (any(task_status$check_status != "pass")) {
    stop("At least one pilot forecast task has no observed labels.", call. = FALSE)
  }
  task_status
}

build_forecast_label_qa <- function(labels) {
  labels |>
    dplyr::group_by(
      .data$outcome_id,
      .data$horizon_years,
      .data$forecast_origin_year,
      .data$target_window_start,
      .data$target_window_end
    ) |>
    dplyr::summarise(
      candidate_hexes = dplyr::n(),
      panel_vintage_source_covered_hexes = sum(
        .data$source_covered_at_panel_vintage
      ),
      window_source_coverage_complete_hexes = sum(
        .data$window_source_coverage_complete
      ),
      period_complete_hexes = sum(.data$period_complete),
      measurement_complete_hexes = sum(.data$measurement_complete),
      labels_observed = sum(.data$label_observed),
      labels_unavailable = sum(!.data$label_observed),
      positive_hexes = sum(.data$outcome_count > 0L, na.rm = TRUE),
      zero_event_hexes = sum(.data$outcome_count == 0L, na.rm = TRUE),
      total_events = if (any(.data$label_observed)) {
        sum(.data$outcome_count, na.rm = TRUE)
      } else {
        NA_integer_
      },
      .groups = "drop"
    ) |>
    dplyr::mutate(
      label_coverage_pct = 100 * .data$labels_observed / .data$candidate_hexes,
      availability_status = dplyr::case_when(
        .data$labels_observed == 0L ~ "unavailable",
        .data$labels_observed <
          .data$window_source_coverage_complete_hexes ~
          "partially_available_within_target_window_coverage",
        TRUE ~ "available_for_all_target_window_covered_hexes"
      ),
      check_status = dplyr::if_else(
        .data$labels_observed + .data$labels_unavailable ==
          .data$candidate_hexes &
          .data$positive_hexes + .data$zero_event_hexes ==
          .data$labels_observed,
        "pass",
        "fail"
      )
    ) |>
    dplyr::arrange(
      .data$forecast_origin_year,
      .data$horizon_years,
      .data$outcome_id
    )
}

build_forecast_labels_wide <- function(labels) {
  labels |>
    dplyr::select(
      "hex_id",
      "forecast_origin_year",
      "forecast_origin_date",
      "analysis_as_of_date",
      "outcome_id",
      "horizon_years",
      "outcome_count",
      "label_observed",
      "source_covered_at_panel_vintage",
      "source_coverage_reference_year",
      "window_source_coverage_complete",
      "period_complete",
      "measurement_complete"
    ) |>
    tidyr::pivot_wider(
      names_from = c("outcome_id", "horizon_years"),
      values_from = c(
        "outcome_count",
        "label_observed",
        "source_covered_at_panel_vintage",
        "source_coverage_reference_year",
        "window_source_coverage_complete",
        "period_complete",
        "measurement_complete"
      ),
      names_glue = "{outcome_id}_{horizon_years}y_{.value}",
      values_fill = list(
        outcome_count = NA_integer_,
        label_observed = FALSE,
        source_covered_at_panel_vintage = FALSE,
        source_coverage_reference_year = NA_integer_,
        window_source_coverage_complete = FALSE,
        period_complete = FALSE,
        measurement_complete = FALSE
      )
    ) |>
    dplyr::arrange(.data$forecast_origin_year, .data$hex_id)
}

build_joint_common_support_qa <- function(labels, pilot_outcomes) {
  observed_wide <- labels |>
    dplyr::filter(.data$outcome_id %in% pilot_outcomes) |>
    dplyr::select(
      "hex_id",
      "forecast_origin_year",
      "horizon_years",
      "outcome_id",
      "label_observed"
    ) |>
    tidyr::pivot_wider(
      names_from = "outcome_id",
      values_from = "label_observed",
      values_fill = FALSE
    )

  missing_outcomes <- setdiff(pilot_outcomes, names(observed_wide))
  if (length(missing_outcomes) > 0L) {
    stop(
      "Common-support QA is missing pilot outcome(s): ",
      paste(missing_outcomes, collapse = ", "),
      call. = FALSE
    )
  }

  observed_matrix <- as.data.frame(observed_wide[pilot_outcomes])
  observed_wide$joint_label_observed <- apply(
    observed_matrix,
    1L,
    function(value) all(value %in% TRUE)
  )

  observed_wide |>
    dplyr::group_by(.data$horizon_years, .data$forecast_origin_year) |>
    dplyr::summarise(
      candidate_hexes = dplyr::n(),
      joint_labels_observed = sum(.data$joint_label_observed),
      joint_labels_unavailable = sum(!.data$joint_label_observed),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      joint_label_coverage_pct =
        100 * .data$joint_labels_observed / .data$candidate_hexes,
      availability_status = dplyr::if_else(
        .data$joint_labels_observed > 0L,
        "joint_labels_available",
        "joint_labels_unavailable"
      ),
      modeling_sufficiency_status = dplyr::if_else(
        .data$joint_labels_observed > 0L,
        "not_assessed_until_predictor_panel",
        "insufficient_no_joint_labels"
      ),
      check_status = dplyr::if_else(
        .data$joint_labels_observed + .data$joint_labels_unavailable ==
          .data$candidate_hexes,
        "pass",
        "fail"
      )
    ) |>
    dplyr::arrange(.data$forecast_origin_year, .data$horizon_years)
}
