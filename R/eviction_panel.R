################################################################################
# Part 3 Eviction Outcome Panel Helpers
################################################################################

eviction_panel_required_columns <- function(data, required, data_name) {
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

eviction_panel_normalize_case_number <- function(case_number) {
  normalized <- toupper(trimws(as.character(case_number)))
  normalized[normalized == ""] <- NA_character_
  normalized
}

eviction_panel_format_values <- function(values) {
  values <- sort(unique(as.character(stats::na.omit(values))))
  if (length(values) == 0L) NA_character_ else paste(values, collapse = "|")
}

eviction_panel_format_hex_row_counts <- function(hex_keys) {
  hex_keys <- as.character(stats::na.omit(hex_keys))
  if (length(hex_keys) == 0L) {
    return(NA_character_)
  }

  counts <- sort(table(hex_keys), decreasing = TRUE)
  paste(names(counts), as.integer(counts), sep = ":", collapse = "|")
}

#' Resolve one and only one analysis hex for each eviction filing case.
#'
#' The source workbooks contain defendant-level rows, so a filed case can occur
#' more than once. Cases whose reliable addresses land in more than one hex are
#' deliberately left unresolved: choosing a row, the highest score, or a modal
#' hex could silently assign a co-defendant's correspondence address as the
#' eviction location. The returned issue table makes those cases reviewable.
#'
#' @param source_filings All source filing rows used to establish the case/date
#'   inventory.
#' @param filings_hex Reliably geocoded filing rows assigned to the analysis
#'   grid.
#' @param reliably_geocoded_case_numbers Case numbers with at least one reliable
#'   point geocode, including points outside the analysis grid. Supplying this
#'   separates outside-grid cases from cases with no reliable location.
#' @param reliably_geocoded_outside_grid_case_numbers Case numbers with at
#'   least one reliable point geocode outside the analysis grid. A case with
#'   both an in-grid and an outside-grid address remains unresolved rather than
#'   silently selecting the in-grid co-defendant address.
#' @param reliably_geocoded_outside_study_case_numbers Case numbers with at
#'   least one reliable point inside the grid but outside the selected City,
#'   county, or court study geography. Mixed in/out evidence remains unresolved.
#' @return A list with `cases`, `assigned_cases`, `uncertain_hex_years`,
#'   `issues`, and `row_qc`.
resolve_eviction_case_hexes <- function(
  source_filings,
  filings_hex,
  reliably_geocoded_case_numbers = NULL,
  reliably_geocoded_outside_grid_case_numbers = NULL,
  reliably_geocoded_outside_study_case_numbers = NULL
) {
  if (inherits(source_filings, "sf")) {
    source_filings <- sf::st_drop_geometry(source_filings)
  }
  if (inherits(filings_hex, "sf")) {
    filings_hex <- sf::st_drop_geometry(filings_hex)
  }

  eviction_panel_required_columns(
    source_filings,
    c("case_number", "file_date"),
    "Source eviction filings"
  )
  eviction_panel_required_columns(
    filings_hex,
    c("case_number", "file_date", "hex_id"),
    "Hex-assigned eviction filings"
  )

  source_county_column <- if ("source_county" %in% names(source_filings)) {
    as.character(source_filings$source_county)
  } else {
    rep("unspecified", nrow(source_filings))
  }
  source_jp_column <- if ("jp_district" %in% names(source_filings)) {
    as.character(source_filings$jp_district)
  } else {
    rep(NA_character_, nrow(source_filings))
  }

  source_rows <- source_filings |>
    dplyr::transmute(
      case_number = eviction_panel_normalize_case_number(.data$case_number),
      file_date = as.Date(.data$file_date),
      source_county = source_county_column,
      source_jp_district = source_jp_column
    )

  invalid_source_rows <- source_rows |>
    dplyr::summarize(
      missing_case_number_rows = sum(is.na(.data$case_number)),
      missing_file_date_rows = sum(is.na(.data$file_date)),
      .groups = "drop"
    )

  source_cases <- source_rows |>
    dplyr::filter(!is.na(.data$case_number)) |>
    dplyr::group_by(.data$case_number) |>
    dplyr::summarize(
      n_source_rows = dplyr::n(),
      n_filing_dates = dplyr::n_distinct(.data$file_date, na.rm = TRUE),
      filing_dates = eviction_panel_format_values(.data$file_date),
      n_source_counties = dplyr::n_distinct(
        .data$source_county,
        na.rm = TRUE
      ),
      source_counties = eviction_panel_format_values(.data$source_county),
      source_county = if (
        dplyr::n_distinct(.data$source_county, na.rm = TRUE) == 1L
      ) eviction_panel_format_values(.data$source_county) else NA_character_,
      n_source_jps = dplyr::n_distinct(
        .data$source_jp_district,
        na.rm = TRUE
      ),
      source_jp_districts = eviction_panel_format_values(
        .data$source_jp_district
      ),
      source_jp_district = if (
        dplyr::n_distinct(.data$source_jp_district, na.rm = TRUE) == 1L
      ) {
        eviction_panel_format_values(.data$source_jp_district)
      } else {
        NA_character_
      },
      file_date = if (
        dplyr::n_distinct(.data$file_date, na.rm = TRUE) == 1L
      ) min(.data$file_date, na.rm = TRUE) else as.Date(NA),
      .groups = "drop"
    )

  reliably_geocoded_case_numbers <- eviction_panel_normalize_case_number(
    reliably_geocoded_case_numbers
  )
  reliably_geocoded_case_numbers <- unique(
    stats::na.omit(reliably_geocoded_case_numbers)
  )
  reliably_geocoded_outside_grid_case_numbers <-
    eviction_panel_normalize_case_number(
      reliably_geocoded_outside_grid_case_numbers
    )
  reliably_geocoded_outside_grid_case_numbers <- unique(
    stats::na.omit(reliably_geocoded_outside_grid_case_numbers)
  )
  reliably_geocoded_outside_study_case_numbers <-
    eviction_panel_normalize_case_number(
      reliably_geocoded_outside_study_case_numbers
    )
  reliably_geocoded_outside_study_case_numbers <- unique(
    stats::na.omit(reliably_geocoded_outside_study_case_numbers)
  )

  address_column <- if ("address_for_geocoding" %in% names(filings_hex)) {
    as.character(filings_hex$address_for_geocoding)
  } else {
    rep(NA_character_, nrow(filings_hex))
  }

  hex_rows <- filings_hex |>
    dplyr::transmute(
      case_number = eviction_panel_normalize_case_number(.data$case_number),
      hex_key = as.character(.data$hex_id),
      address_for_geocoding = address_column
    ) |>
    dplyr::filter(!is.na(.data$case_number))

  hex_evidence <- hex_rows |>
    dplyr::group_by(.data$case_number) |>
    dplyr::summarize(
      n_hex_rows = dplyr::n(),
      n_candidate_hexes = dplyr::n_distinct(.data$hex_key, na.rm = TRUE),
      candidate_hex_ids = eviction_panel_format_values(.data$hex_key),
      candidate_hex_row_counts = eviction_panel_format_hex_row_counts(
        .data$hex_key
      ),
      n_distinct_geocoded_addresses = dplyr::n_distinct(
        .data$address_for_geocoding,
        na.rm = TRUE
      ),
      assigned_hex_key = if (
        dplyr::n_distinct(.data$hex_key, na.rm = TRUE) == 1L
      ) eviction_panel_format_values(.data$hex_key) else NA_character_,
      .groups = "drop"
    )

  cases <- source_cases |>
    dplyr::full_join(hex_evidence, by = "case_number") |>
    dplyr::mutate(
      n_source_rows = dplyr::coalesce(.data$n_source_rows, 0L),
      n_filing_dates = dplyr::coalesce(.data$n_filing_dates, 0L),
      n_source_counties = dplyr::coalesce(.data$n_source_counties, 0L),
      n_source_jps = dplyr::coalesce(.data$n_source_jps, 0L),
      n_hex_rows = dplyr::coalesce(.data$n_hex_rows, 0L),
      n_candidate_hexes = dplyr::coalesce(.data$n_candidate_hexes, 0L),
      n_distinct_geocoded_addresses = dplyr::coalesce(
        .data$n_distinct_geocoded_addresses,
        0L
      ),
      outcome_year = suppressWarnings(as.integer(format(.data$file_date, "%Y"))),
      has_reliable_geocode = .data$case_number %in%
        reliably_geocoded_case_numbers,
      has_reliable_outside_grid_location = .data$case_number %in%
        reliably_geocoded_outside_grid_case_numbers,
      has_reliable_outside_study_location = .data$case_number %in%
        reliably_geocoded_outside_study_case_numbers,
      assignment_status = dplyr::case_when(
        .data$n_source_rows == 0L ~ "excluded_not_in_source_inventory",
        .data$n_source_counties != 1L ~
          "excluded_inconsistent_source_county",
        .data$n_source_jps > 1L ~ "excluded_inconsistent_source_jp",
        .data$n_filing_dates == 0L ~ "excluded_missing_filing_date",
        .data$n_filing_dates > 1L ~ "excluded_inconsistent_filing_dates",
        .data$n_candidate_hexes == 0L &
          .data$has_reliable_outside_study_location ~
            "excluded_reliable_geocode_outside_study_geography",
        .data$n_candidate_hexes == 0L & .data$has_reliable_geocode ~
          "excluded_reliable_geocode_outside_grid",
        .data$n_candidate_hexes == 0L ~ "excluded_no_reliable_location",
        .data$has_reliable_outside_study_location ~
          "excluded_mixed_inside_outside_study_geography",
        .data$has_reliable_outside_grid_location ~
          "excluded_mixed_inside_outside_grid",
        .data$n_candidate_hexes > 1L ~ "excluded_multiple_hexes",
        TRUE ~ "assigned_unique_hex"
      ),
      assigned_hex_key = dplyr::if_else(
        .data$assignment_status == "assigned_unique_hex",
        .data$assigned_hex_key,
        NA_character_
      )
    ) |>
    dplyr::arrange(.data$file_date, .data$case_number)

  assigned_cases <- cases |>
    dplyr::filter(.data$assignment_status == "assigned_unique_hex") |>
    dplyr::select(
      "case_number",
      "file_date",
      "outcome_year",
      "assigned_hex_key",
      "source_county",
      "source_jp_district"
    )

  issues <- cases |>
    dplyr::filter(.data$assignment_status != "assigned_unique_hex") |>
    dplyr::select(
      "case_number",
      "file_date",
      "outcome_year",
      "assignment_status",
      "source_county",
      "source_jp_district",
      "n_source_counties",
      "source_counties",
      "n_source_jps",
      "source_jp_districts",
      "n_source_rows",
      "n_filing_dates",
      "filing_dates",
      "has_reliable_geocode",
      "has_reliable_outside_grid_location",
      "has_reliable_outside_study_location",
      "n_hex_rows",
      "n_candidate_hexes",
      "candidate_hex_ids",
      "candidate_hex_row_counts",
      "n_distinct_geocoded_addresses"
    )

  uncertain_hex_years <- cases |>
    dplyr::filter(.data$assignment_status %in% c(
      "excluded_multiple_hexes",
      "excluded_mixed_inside_outside_grid",
      "excluded_mixed_inside_outside_study_geography"
    )) |>
    dplyr::select(
      "case_number",
      "outcome_year",
      "source_county",
      "source_jp_district"
    ) |>
    dplyr::inner_join(
      hex_rows |>
        dplyr::distinct(.data$case_number, .data$hex_key),
      by = "case_number"
    ) |>
    dplyr::count(
      assigned_hex_key = .data$hex_key,
      .data$outcome_year,
      .data$source_county,
      .data$source_jp_district,
      name = "unresolved_candidate_cases"
    )

  row_qc <- invalid_source_rows |>
    dplyr::mutate(
      source_rows = nrow(source_filings),
      hex_rows = nrow(filings_hex),
      source_cases = nrow(source_cases),
      .before = 1
    )

  list(
    cases = cases,
    assigned_cases = assigned_cases,
    uncertain_hex_years = uncertain_hex_years,
    issues = issues,
    row_qc = row_qc
  )
}

#' Assign each analysis hex to exactly one county using its point on surface.
#'
#' @param hex_grid Analysis grid as an sf object with `hex_id`.
#' @param county_boundaries County polygons as an sf object with `county`.
#' @param analysis_crs Projected CRS used for the point-on-surface operation.
#' @return A non-spatial table with one row per hex and `source_county`.
assign_eviction_hex_counties <- function(
  hex_grid,
  county_boundaries,
  analysis_crs = 3083
) {
  eviction_panel_required_columns(hex_grid, "hex_id", "Hex grid")
  eviction_panel_required_columns(
    county_boundaries,
    "county",
    "County boundaries"
  )
  if (!inherits(hex_grid, "sf") || !inherits(county_boundaries, "sf")) {
    stop("Hex grid and county boundaries must both be sf objects.", call. = FALSE)
  }
  if (anyNA(hex_grid$hex_id) || anyDuplicated(hex_grid$hex_id)) {
    stop("Hex grid IDs must be complete and unique.", call. = FALSE)
  }

  counties <- county_boundaries |>
    dplyr::select("county") |>
    sf::st_transform(analysis_crs)
  hex_points <- suppressWarnings(
    hex_grid |>
      sf::st_transform(analysis_crs) |>
      sf::st_point_on_surface()
  )
  assignment <- hex_points |>
    dplyr::select("hex_id") |>
    sf::st_join(counties, join = sf::st_within, left = TRUE)

  if (nrow(assignment) != nrow(hex_grid) || anyDuplicated(assignment$hex_id)) {
    stop(
      "County polygons assigned at least one hex to multiple counties.",
      call. = FALSE
    )
  }

  assignment |>
    sf::st_drop_geometry() |>
    dplyr::transmute(
      .data$hex_id,
      source_county = as.character(.data$county)
    )
}

#' Build the complete hex-year eviction outcome panel.
#'
#' `eviction_cases` is intentionally missing for incomplete calendar years and
#' uncovered geography. Partial counts remain available only in
#' `eviction_cases_observed_to_date`, which prevents a partial year from being
#' used accidentally as a completed forecasting label.
#'
#' @param hex_counties Table with unique `hex_id` and `source_county` rows.
#' @param assigned_cases Output from `resolve_eviction_case_hexes()`.
#' @param source_start_date First date represented anywhere in the source panel.
#' @param observed_through_date Last requested date represented in the panel.
#' @param analysis_as_of_date Requested analysis cutoff recorded for provenance.
#' @param uncertain_hex_years Candidate hex-years for cases that have multiple
#'   reliable candidate hexes. Their labels remain unavailable.
#' @param covered_county Legacy single-county coverage used only when
#'   `hex_year_coverage` is not supplied.
#' @param hex_year_coverage Optional complete row-specific coverage table with
#'   dates and JP provenance for every hex-year.
#' @return One row per analysis hex and outcome year.
build_complete_eviction_panel <- function(
  hex_counties,
  assigned_cases,
  source_start_date,
  observed_through_date,
  analysis_as_of_date = observed_through_date,
  uncertain_hex_years = NULL,
  covered_county = "Travis",
  hex_year_coverage = NULL
) {
  eviction_panel_required_columns(
    hex_counties,
    c("hex_id", "source_county"),
    "Hex county assignment"
  )
  eviction_panel_required_columns(
    assigned_cases,
    c("case_number", "outcome_year", "assigned_hex_key"),
    "Assigned eviction cases"
  )
  if (is.null(uncertain_hex_years)) {
    uncertain_hex_years <- tibble::tibble(
      assigned_hex_key = character(),
      outcome_year = integer(),
      unresolved_candidate_cases = integer()
    )
  }
  eviction_panel_required_columns(
    uncertain_hex_years,
    c("assigned_hex_key", "outcome_year", "unresolved_candidate_cases"),
    "Uncertain eviction case locations"
  )
  if (anyNA(hex_counties$hex_id) || anyDuplicated(hex_counties$hex_id)) {
    stop("Hex county assignment must have one row per nonmissing hex ID.", call. = FALSE)
  }

  source_start_date <- as.Date(source_start_date)
  observed_through_date <- as.Date(observed_through_date)
  analysis_as_of_date <- as.Date(analysis_as_of_date)
  if (
    is.na(source_start_date) ||
      is.na(observed_through_date) ||
      is.na(analysis_as_of_date) ||
      observed_through_date > analysis_as_of_date ||
      observed_through_date < source_start_date
  ) {
    stop("Eviction source coverage dates are invalid.", call. = FALSE)
  }

  first_year <- as.integer(format(source_start_date, "%Y"))
  last_year <- as.integer(format(observed_through_date, "%Y"))
  outcome_years <- seq.int(first_year, last_year)

  if (is.null(hex_year_coverage)) {
    hex_year_coverage <- tidyr::expand_grid(
      hex_id = hex_counties$hex_id,
      outcome_year = outcome_years
    ) |>
      dplyr::left_join(hex_counties, by = "hex_id") |>
      dplyr::mutate(
        source_covered = .data$source_county == covered_county,
        source_covered = dplyr::coalesce(.data$source_covered, FALSE),
        coverage_jp_district = dplyr::if_else(
          .data$source_covered,
          "ALL",
          NA_character_
        ),
        coverage_boundary_vintage = dplyr::if_else(
          .data$source_covered,
          "legacy_single_county",
          NA_character_
        ),
        coverage_source_ids = dplyr::if_else(
          .data$source_covered,
          "legacy_source",
          NA_character_
        ),
        coverage_start_date = dplyr::if_else(
          .data$source_covered,
          pmax(
            source_start_date,
            as.Date(paste0(.data$outcome_year, "-01-01"))
          ),
          as.Date(NA)
        ),
        coverage_end_date = dplyr::if_else(
          .data$source_covered,
          pmin(
            observed_through_date,
            as.Date(paste0(.data$outcome_year, "-12-31"))
          ),
          as.Date(NA)
        ),
        uncovered_reason = dplyr::if_else(
          .data$source_covered,
          NA_character_,
          "outside_source_geography"
        )
      ) |>
      dplyr::select(
        "hex_id", "outcome_year", "source_covered",
        "coverage_jp_district", "coverage_boundary_vintage",
        "coverage_source_ids", "coverage_start_date", "coverage_end_date",
        "uncovered_reason"
      )
  }
  eviction_panel_required_columns(
    hex_year_coverage,
    c(
      "hex_id", "outcome_year", "source_covered",
      "coverage_jp_district", "coverage_boundary_vintage",
      "coverage_source_ids", "coverage_start_date", "coverage_end_date",
      "uncovered_reason"
    ),
    "Eviction hex-year coverage"
  )
  expected_coverage_rows <- nrow(hex_counties) * length(outcome_years)
  if (nrow(hex_year_coverage) != expected_coverage_rows ||
      anyDuplicated(hex_year_coverage[c("hex_id", "outcome_year")]) ||
      !setequal(hex_year_coverage$hex_id, hex_counties$hex_id) ||
      !setequal(hex_year_coverage$outcome_year, outcome_years)) {
    stop(
      "Eviction hex-year coverage must contain exactly one row for every ",
      "requested hex-year.",
      call. = FALSE
    )
  }
  hex_year_coverage <- hex_year_coverage |>
    dplyr::mutate(
      coverage_start_date = as.Date(.data$coverage_start_date),
      coverage_end_date = as.Date(.data$coverage_end_date)
    )

  assigned_counts <- assigned_cases |>
    dplyr::filter(
      !is.na(.data$outcome_year),
      .data$outcome_year %in% outcome_years
    ) |>
    dplyr::group_by(.data$assigned_hex_key, .data$outcome_year) |>
    dplyr::summarize(
      observed_count = dplyr::n_distinct(.data$case_number),
      .groups = "drop"
    )

  uncertain_counts <- uncertain_hex_years |>
    dplyr::filter(
      !is.na(.data$outcome_year),
      .data$outcome_year %in% outcome_years
    ) |>
    dplyr::group_by(.data$assigned_hex_key, .data$outcome_year) |>
    dplyr::summarize(
      unresolved_candidate_cases = sum(.data$unresolved_candidate_cases),
      .groups = "drop"
    )

  panel <- tidyr::expand_grid(
    hex_id = hex_counties$hex_id,
    outcome_year = outcome_years
  ) |>
    dplyr::left_join(hex_counties, by = "hex_id") |>
    dplyr::mutate(hex_key = as.character(.data$hex_id)) |>
    dplyr::left_join(
      hex_year_coverage,
      by = c("hex_id", "outcome_year")
    ) |>
    dplyr::left_join(
      assigned_counts,
      by = c("hex_key" = "assigned_hex_key", "outcome_year")
    ) |>
    dplyr::left_join(
      uncertain_counts,
      by = c("hex_key" = "assigned_hex_key", "outcome_year")
    ) |>
    dplyr::mutate(
      period_start = as.Date(paste0(.data$outcome_year, "-01-01")),
      period_end = as.Date(paste0(.data$outcome_year, "-12-31")),
      source_covered = dplyr::coalesce(.data$source_covered, FALSE),
      county_assignment_method = "hex_point_on_surface",
      period_complete = .data$source_covered &
        .data$coverage_start_date <= .data$period_start &
        .data$coverage_end_date >= .data$period_end,
      period_complete = dplyr::coalesce(.data$period_complete, FALSE),
      unresolved_candidate_cases = dplyr::coalesce(
        as.integer(.data$unresolved_candidate_cases),
        0L
      ),
      measurement_complete = .data$unresolved_candidate_cases == 0L,
      count_observed = .data$source_covered &
        .data$period_complete &
        .data$measurement_complete,
      eviction_cases_observed_to_date = dplyr::if_else(
        .data$source_covered,
        dplyr::coalesce(as.integer(.data$observed_count), 0L),
        NA_integer_
      ),
      eviction_cases = dplyr::if_else(
        .data$count_observed,
        .data$eviction_cases_observed_to_date,
        NA_integer_
      ),
      observed_through_date = dplyr::if_else(
        .data$source_covered,
        .data$coverage_end_date,
        as.Date(NA)
      ),
      observed_from_date = dplyr::if_else(
        .data$source_covered,
        .data$coverage_start_date,
        as.Date(NA)
      ),
      analysis_as_of_date = analysis_as_of_date,
      coverage_reason = dplyr::case_when(
        !.data$source_covered ~ .data$uncovered_reason,
        !.data$period_complete ~ "covered_partial_year",
        !.data$measurement_complete ~
          "covered_complete_year_with_ambiguous_case_location",
        TRUE ~ "covered_complete"
      )
    ) |>
    dplyr::select(
      "hex_id",
      "outcome_year",
      "eviction_cases",
      "source_county",
      "county_assignment_method",
      "coverage_jp_district",
      "coverage_boundary_vintage",
      "coverage_source_ids",
      "source_covered",
      "period_complete",
      "measurement_complete",
      "count_observed",
      "coverage_reason",
      "unresolved_candidate_cases",
      "eviction_cases_observed_to_date",
      "observed_from_date",
      "observed_through_date",
      "analysis_as_of_date"
    ) |>
    dplyr::arrange(.data$hex_id, .data$outcome_year)

  validate_complete_eviction_panel(
    panel,
    expected_hex_ids = hex_counties$hex_id,
    expected_years = outcome_years
  )
  panel
}

#' Validate structural and missingness invariants for an eviction panel.
validate_complete_eviction_panel <- function(
  panel,
  expected_hex_ids = unique(panel$hex_id),
  expected_years = unique(panel$outcome_year)
) {
  eviction_panel_required_columns(
    panel,
    c(
      "hex_id", "outcome_year", "eviction_cases", "source_covered",
      "period_complete", "measurement_complete", "count_observed",
      "coverage_reason", "unresolved_candidate_cases",
      "eviction_cases_observed_to_date", "source_county",
      "county_assignment_method", "coverage_jp_district",
      "coverage_boundary_vintage", "coverage_source_ids",
      "observed_from_date",
      "observed_through_date", "analysis_as_of_date"
    ),
    "Complete eviction panel"
  )

  expected_rows <- length(unique(expected_hex_ids)) * length(unique(expected_years))
  if (nrow(panel) != expected_rows) {
    stop("Eviction panel is not a complete hex-year Cartesian product.", call. = FALSE)
  }
  if (anyDuplicated(panel[c("hex_id", "outcome_year")])) {
    stop("Eviction panel has duplicate hex-year rows.", call. = FALSE)
  }
  if (
    !setequal(panel$hex_id, expected_hex_ids) ||
      !setequal(panel$outcome_year, expected_years)
  ) {
    stop("Eviction panel does not contain the expected hexes and years.", call. = FALSE)
  }
  if (
      anyNA(panel$source_covered) ||
      anyNA(panel$period_complete) ||
      anyNA(panel$measurement_complete) ||
      anyNA(panel$count_observed) ||
      anyNA(panel$coverage_reason) ||
      anyNA(panel$analysis_as_of_date)
  ) {
    stop("Eviction panel coverage fields may not be missing.", call. = FALSE)
  }
  if (any(panel$count_observed != (
    panel$source_covered &
      panel$period_complete &
      panel$measurement_complete
  ))) {
    stop(
      "count_observed must equal source_covered AND period_complete AND ",
      "measurement_complete.",
      call. = FALSE
    )
  }
  if (any(is.na(panel$eviction_cases) != !panel$count_observed)) {
    stop("Only covered, complete hex-years may contain eviction_cases.", call. = FALSE)
  }
  if (any(!is.na(panel$eviction_cases) & panel$eviction_cases < 0L)) {
    stop("Eviction case counts may not be negative.", call. = FALSE)
  }
  if (any(!is.na(panel$eviction_cases) & panel$eviction_cases %% 1 != 0)) {
    stop("Eviction case counts must be integers.", call. = FALSE)
  }
  if (any(!panel$source_covered & !is.na(panel$eviction_cases_observed_to_date))) {
    stop("Uncovered hex-years must not receive zero or observed eviction counts.", call. = FALSE)
  }
  if (any(panel$source_covered & is.na(panel$eviction_cases_observed_to_date))) {
    stop("Covered hex-years must contain their observed-to-date count.", call. = FALSE)
  }
  if (any(
    panel$source_covered &
      (
        is.na(panel$coverage_jp_district) |
          is.na(panel$coverage_boundary_vintage) |
          is.na(panel$coverage_source_ids) |
          is.na(panel$observed_from_date) |
          is.na(panel$observed_through_date)
      )
  )) {
    stop("Covered hex-years require complete source provenance and dates.", call. = FALSE)
  }
  if (any(
    !panel$source_covered &
      (!is.na(panel$observed_from_date) | !is.na(panel$observed_through_date))
  )) {
    stop("Uncovered hex-years may not contain observed coverage dates.", call. = FALSE)
  }
  if (any(
    panel$measurement_complete != (panel$unresolved_candidate_cases == 0L)
  )) {
    stop(
      "Eviction measurement_complete flags do not match unresolved cases.",
      call. = FALSE
    )
  }
  if (length(unique(as.Date(panel$analysis_as_of_date))) != 1L) {
    stop("Eviction panel must record one analysis_as_of_date.", call. = FALSE)
  }

  invisible(panel)
}

#' Create annual QA that reconciles case assignments to panel totals.
summarize_complete_eviction_panel <- function(
  panel,
  resolved_cases,
  covered_hex_ids = NULL
) {
  coverage_keys <- panel |>
    dplyr::transmute(
      assigned_hex_key = as.character(.data$hex_id),
      .data$outcome_year,
      target_source_covered = .data$source_covered
    )

  assignment_by_year <- resolved_cases$cases |>
    dplyr::left_join(
      coverage_keys,
      by = c("assigned_hex_key", "outcome_year")
    ) |>
    dplyr::mutate(
      assigned_inside_coverage = .data$assignment_status == "assigned_unique_hex" &
        dplyr::coalesce(.data$target_source_covered, FALSE),
      assigned_outside_coverage = .data$assignment_status == "assigned_unique_hex" &
        !dplyr::coalesce(.data$target_source_covered, FALSE)
    ) |>
    dplyr::filter(!is.na(.data$outcome_year)) |>
    dplyr::group_by(.data$outcome_year) |>
    dplyr::summarize(
      source_cases = dplyr::n_distinct(.data$case_number),
      assigned_inside_coverage_cases = sum(.data$assigned_inside_coverage),
      assigned_outside_coverage_cases = sum(.data$assigned_outside_coverage),
      ambiguous_multi_hex_cases = sum(
        .data$assignment_status == "excluded_multiple_hexes"
      ),
      ambiguous_mixed_inside_outside_cases = sum(
        .data$assignment_status %in% c(
          "excluded_mixed_inside_outside_grid",
          "excluded_mixed_inside_outside_study_geography"
        )
      ),
      unassigned_or_other_excluded_cases = sum(
        !.data$assignment_status %in% c(
          "assigned_unique_hex",
          "excluded_multiple_hexes",
          "excluded_mixed_inside_outside_grid",
          "excluded_mixed_inside_outside_study_geography"
        )
      ),
      .groups = "drop"
    )

  panel |>
    dplyr::group_by(.data$outcome_year) |>
    dplyr::summarize(
      panel_hexes = dplyr::n(),
      source_covered_hexes = sum(.data$source_covered),
      period_complete = all(.data$period_complete[.data$source_covered]),
      measurement_complete_hexes = sum(.data$measurement_complete),
      hexes_with_ambiguous_case_location = sum(!.data$measurement_complete),
      unresolved_candidate_case_hex_links = sum(
        .data$unresolved_candidate_cases
      ),
      count_observed_hexes = sum(.data$count_observed),
      panel_eviction_cases = sum(.data$eviction_cases, na.rm = TRUE),
      panel_eviction_cases_observed_to_date = sum(
        .data$eviction_cases_observed_to_date,
        na.rm = TRUE
      ),
      .groups = "drop"
    ) |>
    dplyr::left_join(assignment_by_year, by = "outcome_year") |>
    dplyr::mutate(
      assigned_count_reconciles = .data$panel_eviction_cases_observed_to_date ==
        .data$assigned_inside_coverage_cases
    ) |>
    dplyr::arrange(.data$outcome_year)
}
