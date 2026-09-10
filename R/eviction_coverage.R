################################################################################
# Part 3 Eviction Source-Coverage Helpers
################################################################################

eviction_coverage_required_columns <- function(data, required, data_name) {
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

eviction_coverage_validate_intervals <- function(source_intervals) {
  ordered <- source_intervals |>
    dplyr::arrange(
      .data$source_county,
      .data$jp_district,
      .data$source_period_start,
      .data$source_period_end
    ) |>
    dplyr::group_by(.data$source_county, .data$jp_district) |>
    dplyr::mutate(
      prior_end = dplyr::lag(cummax(as.numeric(.data$source_period_end))),
      gap_after_prior_interval = !is.na(.data$prior_end) &
        as.numeric(.data$source_period_start) > .data$prior_end + 1
    ) |>
    dplyr::ungroup()
  if (any(ordered$gap_after_prior_interval)) {
    gap_groups <- ordered |>
      dplyr::filter(.data$gap_after_prior_interval) |>
      dplyr::distinct(.data$source_county, .data$jp_district) |>
      dplyr::transmute(
        key = paste(.data$source_county, .data$jp_district, sep = "/")
      ) |>
      dplyr::pull(.data$key)
    stop(
      "Eviction source periods contain an unmodeled gap for: ",
      paste(gap_groups, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(source_intervals)
}

#' Build row-specific eviction source coverage for every hex-year.
#'
#' Travis coverage is the intersection of all configured Travis JP periods.
#' Williamson coverage follows the configured JP source periods and the
#' versioned official precinct assignment. Counties or JP periods without a
#' supplied source remain explicitly uncovered rather than receiving zeroes.
#'
#' @param hex_counties One row per analysis hex with `source_county`.
#' @param outcome_years Calendar years included in the panel.
#' @param source_config Versioned eviction source inventory.
#' @param williamson_jp_reference Versioned Williamson hex/JP assignment.
#' @param analysis_as_of_date Requested analysis cutoff.
#' @return One row per requested hex-year with source-specific coverage dates.
build_eviction_hex_year_coverage <- function(
  hex_counties,
  outcome_years,
  source_config,
  williamson_jp_reference,
  analysis_as_of_date
) {
  eviction_coverage_required_columns(
    hex_counties,
    c("hex_id", "source_county"),
    "Hex county assignment"
  )
  eviction_coverage_required_columns(
    source_config,
    c(
      "source_id", "source_county", "jp_district",
      "source_period_start", "source_period_end"
    ),
    "Eviction source configuration"
  )
  eviction_coverage_required_columns(
    williamson_jp_reference,
    c(
      "hex_id", "effective_start_date", "effective_end_date",
      "jp_district", "boundary_vintage", "assignment_status"
    ),
    "Williamson JP hex reference"
  )
  if (anyNA(hex_counties$hex_id) || anyDuplicated(hex_counties$hex_id)) {
    stop("Hex county assignment must contain unique nonmissing IDs.", call. = FALSE)
  }

  outcome_years <- sort(unique(as.integer(outcome_years)))
  analysis_as_of_date <- as.Date(analysis_as_of_date)
  if (length(outcome_years) == 0L || anyNA(outcome_years) ||
      is.na(analysis_as_of_date)) {
    stop("Outcome years and analysis cutoff must be valid.", call. = FALSE)
  }

  sources <- source_config |>
    dplyr::mutate(
      source_period_start = as.Date(.data$source_period_start),
      source_period_end = as.Date(.data$source_period_end)
    )
  if (anyNA(sources[c(
    "source_id", "source_county", "jp_district",
    "source_period_start", "source_period_end"
  )]) || any(sources$source_period_start > sources$source_period_end)) {
    stop("Eviction source configuration contains invalid periods.", call. = FALSE)
  }
  eviction_coverage_validate_intervals(sources)

  source_intervals <- sources |>
    dplyr::group_by(.data$source_county, .data$jp_district) |>
    dplyr::summarize(
      source_interval_start = min(.data$source_period_start),
      source_interval_end = max(.data$source_period_end),
      coverage_source_ids = paste(
        sort(unique(.data$source_id)),
        collapse = "|"
      ),
      .groups = "drop"
    )

  travis_intervals <- source_intervals |>
    dplyr::filter(.data$source_county == "Travis")
  expected_travis_jps <- paste0("JP", 1:5)
  if (!setequal(travis_intervals$jp_district, expected_travis_jps)) {
    stop("Travis coverage requires configured JP1 through JP5 sources.", call. = FALSE)
  }
  travis_county_interval <- travis_intervals |>
    dplyr::summarize(
      coverage_start_date = max(.data$source_interval_start),
      coverage_end_date = min(.data$source_interval_end),
      coverage_source_ids = paste(
        sort(unique(unlist(strsplit(.data$coverage_source_ids, "\\|")))),
        collapse = "|"
      )
    )
  if (travis_county_interval$coverage_start_date >
      travis_county_interval$coverage_end_date) {
    stop("Configured Travis JP periods have no common coverage interval.", call. = FALSE)
  }

  years <- tibble::tibble(
    outcome_year = outcome_years,
    period_start = as.Date(paste0(outcome_years, "-01-01")),
    period_end = as.Date(paste0(outcome_years, "-12-31"))
  )
  base <- tidyr::crossing(
    hex_counties |>
      dplyr::select("hex_id", "source_county"),
    years
  )

  jp_reference <- williamson_jp_reference |>
    dplyr::mutate(
      effective_start_date = as.Date(.data$effective_start_date),
      effective_end_date = as.Date(.data$effective_end_date)
    )
  williamson_year_reference <- tidyr::crossing(
    jp_reference,
    years |>
      dplyr::select("outcome_year", "period_start", "period_end")
  ) |>
    dplyr::filter(
      .data$effective_start_date <= .data$period_start,
      is.na(.data$effective_end_date) |
        .data$effective_end_date >= .data$period_end
    ) |>
    dplyr::select(
      "hex_id", "outcome_year", coverage_jp_district = "jp_district",
      coverage_boundary_vintage = "boundary_vintage",
      precinct_assignment_status = "assignment_status"
    )
  expected_williamson_rows <- sum(
    base$source_county == "Williamson"
  )
  if (nrow(williamson_year_reference) != expected_williamson_rows ||
      anyDuplicated(williamson_year_reference[c("hex_id", "outcome_year")])) {
    stop(
      "Williamson JP reference does not assign exactly one boundary vintage ",
      "to every requested Williamson hex-year.",
      call. = FALSE
    )
  }

  williamson_intervals <- source_intervals |>
    dplyr::filter(.data$source_county == "Williamson") |>
    dplyr::select(
      coverage_jp_district = "jp_district",
      "source_interval_start",
      "source_interval_end",
      "coverage_source_ids"
    )

  coverage <- base |>
    dplyr::left_join(
      williamson_year_reference,
      by = c("hex_id", "outcome_year")
    ) |>
    dplyr::left_join(
      williamson_intervals,
      by = "coverage_jp_district"
    ) |>
    dplyr::mutate(
      coverage_jp_district = dplyr::if_else(
        .data$source_county == "Travis",
        "ALL",
        .data$coverage_jp_district
      ),
      coverage_boundary_vintage = dplyr::if_else(
        .data$source_county == "Travis",
        "countywide_all_jps",
        .data$coverage_boundary_vintage
      ),
      coverage_source_ids = dplyr::if_else(
        .data$source_county == "Travis",
        travis_county_interval$coverage_source_ids[[1]],
        .data$coverage_source_ids
      ),
      source_interval_start = dplyr::if_else(
        .data$source_county == "Travis",
        travis_county_interval$coverage_start_date[[1]],
        .data$source_interval_start
      ),
      source_interval_end = dplyr::if_else(
        .data$source_county == "Travis",
        travis_county_interval$coverage_end_date[[1]],
        .data$source_interval_end
      ),
      coverage_start_date = pmax(
        .data$period_start,
        .data$source_interval_start,
        na.rm = FALSE
      ),
      coverage_end_date = pmin(
        .data$period_end,
        .data$source_interval_end,
        analysis_as_of_date,
        na.rm = FALSE
      ),
      source_covered = !is.na(.data$coverage_start_date) &
        !is.na(.data$coverage_end_date) &
        .data$coverage_start_date <= .data$coverage_end_date,
      uncovered_reason = dplyr::case_when(
        .data$source_covered ~ NA_character_,
        .data$source_county == "Hays" ~ "county_source_not_supplied",
        .data$source_county == "Williamson" &
          .data$precinct_assignment_status != "assigned" ~
            "williamson_precinct_unassigned",
        .data$source_county == "Williamson" ~
          "williamson_jp_not_supplied_for_period",
        TRUE ~ "county_source_not_supplied"
      )
    ) |>
    dplyr::select(
      "hex_id", "outcome_year", "source_covered",
      "coverage_jp_district", "coverage_boundary_vintage",
      "coverage_source_ids", "coverage_start_date", "coverage_end_date",
      "uncovered_reason"
    ) |>
    dplyr::arrange(.data$hex_id, .data$outcome_year)

  expected_rows <- nrow(hex_counties) * length(outcome_years)
  if (nrow(coverage) != expected_rows ||
      anyDuplicated(coverage[c("hex_id", "outcome_year")]) ||
      anyNA(coverage$source_covered) ||
      any(!coverage$source_covered & is.na(coverage$uncovered_reason))) {
    stop("Generated eviction hex-year coverage is structurally invalid.", call. = FALSE)
  }
  coverage
}
