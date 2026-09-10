################################################################################
# Residential Demolition Outcome Panel Helpers
################################################################################
#
# These helpers turn the City of Austin issued-permit extract into a complete
# H3-by-year outcome panel.  A usable annual outcome is written only when both
# the geographic source-coverage flag and the calendar-period completeness flag
# are true.  Counts observed during partial years are retained in a separate
# field so they cannot be mistaken for complete annual labels.

demolition_required_columns <- c(
  "Permit Num",
  "Permit Class Mapped",
  "Work Class",
  "Issued Date",
  "Latitude",
  "Longitude",
  "Jurisdiction"
)

normalize_demolition_text <- function(x) {
  stringr::str_to_upper(stringr::str_squish(as.character(x)))
}

parse_demolition_date <- function(x) {
  if (inherits(x, "Date")) return(x)
  suppressWarnings(lubridate::ymd(as.character(x), quiet = TRUE))
}

validate_demolition_inputs <- function(raw_permits, hex_grid, hex_coverage) {
  missing_columns <- setdiff(demolition_required_columns, names(raw_permits))
  if (length(missing_columns) > 0L) {
    stop(
      "Demolition source is missing required column(s): ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  if (!inherits(hex_grid, "sf") || !"hex_id" %in% names(hex_grid)) {
    stop("hex_grid must be an sf object with a hex_id column.", call. = FALSE)
  }
  if (anyNA(hex_grid$hex_id) || anyDuplicated(hex_grid$hex_id)) {
    stop("hex_grid hex_id values must be complete and unique.", call. = FALSE)
  }
  if (is.na(sf::st_crs(hex_grid))) {
    stop("hex_grid must have a coordinate reference system.", call. = FALSE)
  }
  coverage_columns <- c(
    "hex_id",
    "outcome_year",
    "source_covered",
    "coverage_resolved",
    "coverage_period_resolved",
    "coverage_period_start_date",
    "coverage_period_end_date",
    "coverage_period_complete",
    "coverage_status",
    "coverage_jurisdiction_type",
    "end_state_source_covered",
    "end_state_status",
    "geographic_coverage_reason",
    "coverage_basis",
    "city_study_geography",
    "city_boundary_snapshot_date",
    "city_hex_assignment_method",
    "hex_center_inside_current_austin_full",
    "hex_intersects_current_austin_full",
    "source_covered_before_city_filter"
  )
  if (
    !is.data.frame(hex_coverage) ||
      !all(coverage_columns %in% names(hex_coverage))
  ) {
    stop(
      "hex_coverage must be a table with complete historical coverage fields.",
      call. = FALSE
    )
  }
  if (
    anyNA(hex_coverage[c("hex_id", "outcome_year", "source_covered")]) ||
      anyDuplicated(hex_coverage[c("hex_id", "outcome_year")])
  ) {
    stop(
      "hex_coverage must have one non-missing coverage row per hex-year.",
      call. = FALSE
    )
  }
  if (
    anyNA(hex_coverage[c(
      "coverage_resolved",
      "coverage_period_resolved",
      "coverage_period_start_date",
      "coverage_period_end_date",
      "coverage_period_complete"
    )]) ||
      !identical(
        hex_coverage$coverage_resolved,
        hex_coverage$coverage_period_resolved
      ) ||
      any(hex_coverage$source_covered & !hex_coverage$coverage_resolved)
  ) {
    stop(
      "hex_coverage full-period coverage flags are inconsistent.",
      call. = FALSE
    )
  }
  fixed_geography_columns <- c(
    "city_study_geography",
    "city_boundary_snapshot_date",
    "city_hex_assignment_method",
    "hex_center_inside_current_austin_full",
    "hex_intersects_current_austin_full",
    "source_covered_before_city_filter"
  )
  if (
    anyNA(hex_coverage[fixed_geography_columns]) ||
      dplyr::n_distinct(hex_coverage$city_study_geography) != 1L ||
      dplyr::n_distinct(hex_coverage$city_boundary_snapshot_date) != 1L ||
      dplyr::n_distinct(hex_coverage$city_hex_assignment_method) != 1L ||
      any(
        hex_coverage$hex_center_inside_current_austin_full &
          !hex_coverage$hex_intersects_current_austin_full
      ) ||
      any(
        hex_coverage$source_covered &
          !hex_coverage$hex_center_inside_current_austin_full
      ) ||
      any(
        hex_coverage$hex_center_inside_current_austin_full &
          hex_coverage$source_covered !=
            hex_coverage$source_covered_before_city_filter
      )
  ) {
    stop(
      "hex_coverage is inconsistent with the fixed City study geography.",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

prepare_residential_demolition_permits <- function(
  raw_permits,
  observed_through_date,
  panel_years
) {
  observed_through_date <- as.Date(observed_through_date)
  if (is.na(observed_through_date)) {
    stop("observed_through_date must be a valid Date.", call. = FALSE)
  }
  panel_years <- sort(unique(as.integer(panel_years)))
  if (length(panel_years) == 0L || anyNA(panel_years)) {
    stop("panel_years must contain valid years.", call. = FALSE)
  }

  selected <- raw_permits %>%
    dplyr::mutate(
      source_row_id = dplyr::row_number(),
      permit_id = normalize_demolition_text(.data[["Permit Num"]]),
      issue_date = parse_demolition_date(.data[["Issued Date"]]),
      outcome_year = lubridate::year(.data$issue_date),
      latitude = suppressWarnings(as.numeric(.data[["Latitude"]])),
      longitude = suppressWarnings(as.numeric(.data[["Longitude"]])),
      source_jurisdiction = normalize_demolition_text(
        .data[["Jurisdiction"]]
      ),
      is_demolition = normalize_demolition_text(
        .data[["Work Class"]]
      ) == "DEMOLITION",
      is_residential = normalize_demolition_text(
        .data[["Permit Class Mapped"]]
      ) == "RESIDENTIAL",
      permit_id_missing = is.na(.data$permit_id) | .data$permit_id == "",
      issue_date_missing = is.na(.data$issue_date),
      valid_coordinates =
        is.finite(.data$latitude) &
        is.finite(.data$longitude) &
        dplyr::between(.data$latitude, -90, 90) &
        dplyr::between(.data$longitude, -180, 180)
    ) %>%
    dplyr::filter(.data$is_demolition, .data$is_residential)

  duplicate_groups <- selected %>%
    dplyr::filter(!.data$permit_id_missing) %>%
    dplyr::group_by(.data$permit_id) %>%
    dplyr::summarise(
      source_rows = dplyr::n(),
      issue_dates = dplyr::n_distinct(.data$issue_date, na.rm = TRUE),
      coordinate_pairs = dplyr::n_distinct(
        paste(.data$latitude, .data$longitude, sep = ","),
        na.rm = TRUE
      ),
      .groups = "drop"
    ) %>%
    dplyr::filter(.data$source_rows > 1L)

  deduplicated <- selected %>%
    dplyr::filter(!.data$permit_id_missing) %>%
    dplyr::arrange(
      .data$permit_id,
      .data$issue_date_missing,
      dplyr::desc(.data$valid_coordinates),
      .data$source_row_id
    ) %>%
    dplyr::distinct(.data$permit_id, .keep_all = TRUE)

  eligible <- deduplicated %>%
    dplyr::filter(
      !.data$issue_date_missing,
      .data$issue_date <= observed_through_date,
      .data$outcome_year %in% panel_years
    )

  list(
    selected = selected,
    duplicate_groups = duplicate_groups,
    deduplicated = deduplicated,
    eligible = eligible,
    metrics = c(
      raw_source_rows = nrow(raw_permits),
      residential_demolition_rows = nrow(selected),
      residential_demolition_rows_missing_permit_id = sum(
        selected$permit_id_missing
      ),
      unique_residential_demolition_permits = nrow(deduplicated),
      duplicate_permit_ids = nrow(duplicate_groups),
      duplicate_source_rows_removed = nrow(selected) -
        sum(selected$permit_id_missing) - nrow(deduplicated),
      duplicate_permit_ids_with_conflicting_issue_dates = sum(
        duplicate_groups$issue_dates > 1L
      ),
      duplicate_permit_ids_with_conflicting_coordinates = sum(
        duplicate_groups$coordinate_pairs > 1L
      ),
      unique_permits_missing_issue_date = sum(deduplicated$issue_date_missing),
      unique_permits_after_observed_through_date = sum(
        !deduplicated$issue_date_missing &
          deduplicated$issue_date > observed_through_date
      ),
      unique_permits_outside_panel_years = sum(
        !deduplicated$issue_date_missing &
          deduplicated$issue_date <= observed_through_date &
          !deduplicated$outcome_year %in% panel_years
      ),
      eligible_unique_permits = nrow(eligible)
    )
  )
}

assign_demolition_permits_to_hex <- function(
  eligible_permits,
  hex_grid,
  hex_coverage,
  fixed_study_boundary,
  analysis_crs = 3083
) {
  if (
    !inherits(fixed_study_boundary, "sf") ||
      nrow(fixed_study_boundary) == 0L ||
      is.na(sf::st_crs(fixed_study_boundary)) ||
      any(sf::st_is_empty(fixed_study_boundary))
  ) {
    stop(
      "fixed_study_boundary must be a non-empty sf polygon with a CRS.",
      call. = FALSE
    )
  }
  permits <- eligible_permits %>%
    dplyr::mutate(
      match_status = dplyr::if_else(
        .data$valid_coordinates,
        "pending_spatial_match",
        "invalid_coordinates"
      ),
      point_inside_current_austin_full = NA,
      inside_city_study_hex = NA,
      matched_source_covered = NA
    )
  permits$hex_id <- rep(hex_grid$hex_id[NA_integer_], nrow(permits))

  valid_index <- which(permits$valid_coordinates)
  if (length(valid_index) > 0L) {
    permit_points <- permits[valid_index, , drop = FALSE] %>%
      sf::st_as_sf(
        coords = c("longitude", "latitude"),
        crs = 4326,
        remove = FALSE
      ) %>%
      sf::st_transform(analysis_crs)
    boundary_projected <- fixed_study_boundary %>%
      sf::st_make_valid() %>%
      sf::st_transform(analysis_crs)
    point_inside_city <- lengths(
      sf::st_covered_by(permit_points, boundary_projected)
    ) > 0L
    permits$point_inside_current_austin_full[valid_index] <- point_inside_city
    permits$match_status[valid_index[!point_inside_city]] <-
      "outside_current_austin_full_purpose"

    inside_city_index <- which(point_inside_city)
    if (length(inside_city_index) == 0L) {
      return(permits)
    }

    grid_projected <- hex_grid %>%
      sf::st_make_valid() %>%
      sf::st_transform(analysis_crs)
    match_list <- sf::st_within(
      permit_points[inside_city_index, , drop = FALSE],
      grid_projected
    )
    match_count <- lengths(match_list)
    single_match <- which(match_count == 1L)
    inside_valid_index <- valid_index[inside_city_index]

    permits$match_status[inside_valid_index[match_count == 0L]] <- "outside_grid"
    permits$match_status[inside_valid_index[match_count > 1L]] <-
      "multiple_hex_matches"

    if (length(single_match) > 0L) {
      matched_grid_index <- vapply(
        match_list[single_match],
        function(index) index[[1]],
        integer(1)
      )
      matched_hex_id <- hex_grid$hex_id[matched_grid_index]
      permit_coverage_key <- paste(
        matched_hex_id,
        permits$outcome_year[inside_valid_index[single_match]],
        sep = "::"
      )
      coverage_key <- paste(
        hex_coverage$hex_id,
        hex_coverage$outcome_year,
        sep = "::"
      )
      coverage_index <- match(permit_coverage_key, coverage_key)
      if (anyNA(coverage_index)) {
        stop("Matched hex-year is missing from the coverage table.", call. = FALSE)
      }
      covered <- hex_coverage$source_covered[coverage_index]
      center_selected <- hex_coverage[[
        "hex_center_inside_current_austin_full"
      ]][coverage_index]
      permits$hex_id[inside_valid_index[single_match]] <- matched_hex_id
      permits$inside_city_study_hex[inside_valid_index[single_match]] <-
        center_selected
      permits$matched_source_covered[inside_valid_index[single_match]] <-
        covered
      permits$match_status[inside_valid_index[single_match]] <- dplyr::case_when(
        !center_selected ~
          "inside_city_point_outside_center_selected_city_hexes",
        covered ~ "mapped_covered_hex",
        TRUE ~ "mapped_uncovered_hex"
      )
    }
  }

  permits
}

build_demolition_annual_panel <- function(
  assigned_permits,
  hex_coverage,
  panel_years,
  complete_years,
  source_start_date,
  observed_through_date,
  analysis_as_of_date
) {
  panel_years <- sort(unique(as.integer(panel_years)))
  complete_years <- sort(unique(as.integer(complete_years)))
  analysis_source_start_date <- as.Date(source_start_date)
  analysis_observed_through_date <- as.Date(observed_through_date)
  analysis_as_of_date <- as.Date(analysis_as_of_date)
  if (
    is.na(analysis_source_start_date) ||
      is.na(analysis_observed_through_date) ||
      is.na(analysis_as_of_date) ||
      analysis_observed_through_date > analysis_as_of_date
  ) {
    stop("Demolition panel provenance dates are invalid.", call. = FALSE)
  }

  mapped_counts <- assigned_permits %>%
    dplyr::filter(.data$match_status == "mapped_covered_hex") %>%
    dplyr::count(
      .data$hex_id,
      .data$outcome_year,
      name = "residential_demolition_permits_observed_to_date"
    )

  year_metadata <- tibble::tibble(outcome_year = panel_years) %>%
    dplyr::mutate(
      period_complete = .data$outcome_year %in% complete_years,
      observed_from_date = pmax(
        .env$analysis_source_start_date,
        as.Date(paste0(.data$outcome_year, "-01-01"))
      ),
      observed_through_date = as.Date(dplyr::if_else(
        .data$outcome_year < lubridate::year(
          .env$analysis_observed_through_date
        ),
        paste0(.data$outcome_year, "-12-31"),
        as.character(.env$analysis_observed_through_date)
      ))
    )

  panel <- hex_coverage %>%
    dplyr::filter(.data$outcome_year %in% panel_years) %>%
    dplyr::left_join(
      year_metadata,
      by = "outcome_year",
      relationship = "many-to-one"
    ) %>%
    dplyr::left_join(
      mapped_counts,
      by = c("hex_id", "outcome_year"),
      relationship = "one-to-one"
    ) %>%
    dplyr::mutate(
      residential_demolition_permits_observed_to_date = dplyr::if_else(
        .data$source_covered,
        dplyr::coalesce(
          .data$residential_demolition_permits_observed_to_date,
          0L
        ),
        NA_integer_
      ),
      measurement_complete = TRUE,
      count_observed = .data$source_covered &
        .data$period_complete &
        .data$measurement_complete,
      residential_demolition_permits = dplyr::if_else(
        .data$count_observed,
        .data$residential_demolition_permits_observed_to_date,
        NA_integer_
      ),
      coverage_reason = dplyr::case_when(
        !.data$source_covered ~ .data$geographic_coverage_reason,
        .data$coverage_start_capped & .data$coverage_cutoff_capped ~ paste0(
          .data$geographic_coverage_reason,
          ";partial_source_start_",
          .data$coverage_period_start_date,
          ";partial_year_through_",
          .data$coverage_period_end_date
        ),
        .data$coverage_start_capped ~ paste0(
          .data$geographic_coverage_reason,
          ";partial_source_start_",
          .data$coverage_period_start_date
        ),
        .data$coverage_cutoff_capped ~ paste0(
          .data$geographic_coverage_reason,
          ";partial_year_through_",
          .data$coverage_period_end_date
        ),
        TRUE ~ .data$geographic_coverage_reason
      ),
      observed_from_date = dplyr::if_else(
        .data$source_covered,
        .data$observed_from_date,
        as.Date(NA)
      ),
      analysis_as_of_date = .env$analysis_as_of_date
    ) %>%
    dplyr::select(
      "hex_id",
      "outcome_year",
      "residential_demolition_permits",
      "residential_demolition_permits_observed_to_date",
      "city_study_geography",
      "city_boundary_snapshot_date",
      "city_hex_assignment_method",
      "hex_center_inside_current_austin_full",
      "hex_intersects_current_austin_full",
      "source_covered",
      "source_covered_before_city_filter",
      "coverage_resolved",
      "coverage_period_resolved",
      "coverage_jurisdiction_type",
      "coverage_candidate_types",
      "coverage_status",
      "coverage_tied",
      "coverage_ambiguous",
      "coverage_period_start_date",
      "coverage_period_end_date",
      "coverage_as_of_date",
      "coverage_period_complete",
      "coverage_start_capped",
      "coverage_cutoff_capped",
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
      "latest_effective_date",
      "coverage_basis",
      "period_complete",
      "measurement_complete",
      "count_observed",
      "observed_from_date",
      "observed_through_date",
      "analysis_as_of_date",
      "coverage_reason"
    ) %>%
    dplyr::arrange(.data$hex_id, .data$outcome_year)

  expected_coverage_start <- pmax(
    as.Date(paste0(panel$outcome_year, "-01-01")),
    analysis_source_start_date
  )
  expected_coverage_end <- pmin(
    as.Date(paste0(panel$outcome_year, "-12-31")),
    analysis_observed_through_date
  )
  if (any(
    panel$coverage_period_start_date != expected_coverage_start |
      panel$coverage_period_end_date != expected_coverage_end |
      panel$coverage_period_complete != panel$period_complete
  )) {
    stop(
      "Historical coverage periods do not match demolition source periods.",
      call. = FALSE
    )
  }

  panel
}

build_demolition_qa <- function(
  prepared,
  assigned_permits,
  panel,
  hex_coverage,
  panel_years
) {
  match_levels <- c(
    "mapped_covered_hex",
    "mapped_uncovered_hex",
    "inside_city_point_outside_center_selected_city_hexes",
    "outside_current_austin_full_purpose",
    "outside_grid",
    "multiple_hex_matches",
    "invalid_coordinates"
  )
  match_counts <- assigned_permits %>%
    dplyr::count(.data$outcome_year, .data$match_status) %>%
    tidyr::complete(
      outcome_year = panel_years,
      match_status = match_levels,
      fill = list(n = 0L)
    ) %>%
    tidyr::pivot_wider(
      names_from = "match_status",
      values_from = "n",
      values_fill = 0L
    )

  source_year_counts <- assigned_permits %>%
    dplyr::count(.data$outcome_year, name = "source_unique_permits")
  panel_year_counts <- panel %>%
    dplyr::group_by(.data$outcome_year) %>%
    dplyr::summarise(
      period_complete = dplyr::first(.data$period_complete),
      observed_through_date = dplyr::first(.data$observed_through_date),
      source_covered_hexes = sum(.data$source_covered),
      count_observed_hexes = sum(.data$count_observed),
      panel_observed_to_date_total = sum(
        .data$residential_demolition_permits_observed_to_date,
        na.rm = TRUE
      ),
      panel_complete_label_total = if (
        dplyr::first(.data$period_complete)
      ) {
        sum(.data$residential_demolition_permits, na.rm = TRUE)
      } else {
        NA_integer_
      },
      .groups = "drop"
    )

  annual_qa <- tibble::tibble(outcome_year = panel_years) %>%
    dplyr::left_join(source_year_counts, by = "outcome_year") %>%
    dplyr::left_join(match_counts, by = "outcome_year") %>%
    dplyr::left_join(panel_year_counts, by = "outcome_year") %>%
    dplyr::mutate(
      dplyr::across(
        dplyr::all_of(c("source_unique_permits", match_levels)),
        ~ dplyr::coalesce(as.integer(.x), 0L)
      ),
      classified_unique_permits =
        .data$mapped_covered_hex +
        .data$mapped_uncovered_hex +
        .data$inside_city_point_outside_center_selected_city_hexes +
        .data$outside_current_austin_full_purpose +
        .data$outside_grid +
        .data$multiple_hex_matches +
        .data$invalid_coordinates,
      source_classification_difference =
        .data$source_unique_permits - .data$classified_unique_permits,
      panel_mapping_difference =
        .data$mapped_covered_hex - .data$panel_observed_to_date_total,
      reconciliation_pass =
        .data$source_classification_difference == 0L &
        .data$panel_mapping_difference == 0L
    ) %>%
    dplyr::select(
      "outcome_year",
      "period_complete",
      "observed_through_date",
      "source_unique_permits",
      "mapped_covered_hex",
      "mapped_uncovered_hex",
      "inside_city_point_outside_center_selected_city_hexes",
      "outside_current_austin_full_purpose",
      "outside_grid",
      "multiple_hex_matches",
      "invalid_coordinates",
      "panel_observed_to_date_total",
      "panel_complete_label_total",
      "source_covered_hexes",
      "count_observed_hexes",
      "source_classification_difference",
      "panel_mapping_difference",
      "reconciliation_pass"
    )

  unmatched_qa <- assigned_permits %>%
    dplyr::filter(.data$match_status != "mapped_covered_hex") %>%
    dplyr::mutate(
      source_jurisdiction = dplyr::coalesce(
        .data$source_jurisdiction,
        "MISSING"
      )
    ) %>%
    dplyr::count(
      .data$outcome_year,
      .data$source_jurisdiction,
      .data$match_status,
      name = "unique_permits"
    ) %>%
    dplyr::left_join(
      source_year_counts,
      by = "outcome_year",
      relationship = "many-to-one"
    ) %>%
    dplyr::mutate(
      share_of_year_unique_permits = dplyr::if_else(
        .data$source_unique_permits > 0L,
        .data$unique_permits / .data$source_unique_permits,
        NA_real_
      )
    ) %>%
    dplyr::arrange(
      .data$outcome_year,
      .data$match_status,
      .data$source_jurisdiction
    )

  spatial_metrics <- c(
    grid_hexes = dplyr::n_distinct(hex_coverage$hex_id),
    fixed_study_hexes = dplyr::n_distinct(
      hex_coverage$hex_id[
        hex_coverage$hex_center_inside_current_austin_full
      ]
    ),
    outside_fixed_study_hexes = dplyr::n_distinct(
      hex_coverage$hex_id[
        !hex_coverage$hex_center_inside_current_austin_full
      ]
    ),
    coverage_hex_years = nrow(hex_coverage),
    fixed_study_hex_years = sum(
      hex_coverage$hex_center_inside_current_austin_full
    ),
    source_covered_hex_years_before_city_filter = sum(
      hex_coverage$source_covered_before_city_filter
    ),
    source_covered_hex_years = sum(hex_coverage$source_covered),
    source_uncovered_hex_years = sum(!hex_coverage$source_covered),
    unresolved_coverage_hex_years = sum(!hex_coverage$coverage_resolved),
    ambiguous_coverage_hex_years = sum(hex_coverage$coverage_ambiguous),
    eligible_permits_with_invalid_coordinates = sum(
      assigned_permits$match_status == "invalid_coordinates"
    ),
    eligible_permits_outside_grid = sum(
      assigned_permits$match_status == "outside_grid"
    ),
    eligible_permits_with_multiple_hex_matches = sum(
      assigned_permits$match_status == "multiple_hex_matches"
    ),
    eligible_permits_mapped_to_uncovered_hex = sum(
      assigned_permits$match_status == "mapped_uncovered_hex"
    ),
    eligible_permits_inside_city_outside_center_selected_hexes = sum(
      assigned_permits$match_status ==
        "inside_city_point_outside_center_selected_city_hexes"
    ),
    eligible_permits_outside_current_austin_full_purpose = sum(
      assigned_permits$match_status ==
        "outside_current_austin_full_purpose"
    ),
    eligible_permits_mapped_to_covered_hex = sum(
      assigned_permits$match_status == "mapped_covered_hex"
    ),
    panel_rows = nrow(panel),
    panel_complete_label_rows = sum(panel$count_observed),
    panel_partial_or_uncovered_rows = sum(!panel$count_observed)
  )
  source_metrics <- c(prepared$metrics, spatial_metrics)
  source_qa <- tibble::tibble(
    metric = names(source_metrics),
    value = as.numeric(source_metrics),
    detail = dplyr::case_when(
      .data$metric == "source_covered_hex_years" ~ paste(
        "Within the fixed current-FULL center-selected study cells, coverage",
        "replays every effective-dated City jurisdiction state over the",
        "observed annual interval; only continuously resolved FULL, LTD, or",
        "2MILE periods are covered."
      ),
      .data$metric == "panel_complete_label_rows" ~
        paste(
          "Rows receive annual labels only when source_covered,",
          "period_complete, and measurement_complete are all true."
        ),
      .data$metric == "duplicate_source_rows_removed" ~
        "Permit Num is normalized and used as the unique permit outcome identifier.",
      TRUE ~ NA_character_
    )
  )

  list(
    source_qa = source_qa,
    annual_qa = annual_qa,
    unmatched_qa = unmatched_qa
  )
}

validate_demolition_panel <- function(
  panel,
  annual_qa,
  expected_hexes,
  panel_years,
  complete_years
) {
  expected_rows <- as.integer(expected_hexes) * length(panel_years)
  if (nrow(panel) != expected_rows) {
    stop(
      "Demolition panel has ",
      nrow(panel),
      " rows; expected ",
      expected_rows,
      ".",
      call. = FALSE
    )
  }
  if (anyDuplicated(panel[c("hex_id", "outcome_year")])) {
    stop("Demolition panel has duplicate hex-year rows.", call. = FALSE)
  }
  fixed_geography_columns <- c(
    "city_study_geography",
    "city_boundary_snapshot_date",
    "city_hex_assignment_method",
    "hex_center_inside_current_austin_full",
    "hex_intersects_current_austin_full",
    "source_covered_before_city_filter"
  )
  if (
    !all(fixed_geography_columns %in% names(panel)) ||
      anyNA(panel[fixed_geography_columns])
  ) {
    stop(
      "Demolition panel lacks complete fixed-geography provenance.",
      call. = FALSE
    )
  }
  fixed_hex_reference <- panel %>%
    dplyr::distinct(
      .data$hex_id,
      .data$city_study_geography,
      .data$city_boundary_snapshot_date,
      .data$city_hex_assignment_method,
      .data$hex_center_inside_current_austin_full,
      .data$hex_intersects_current_austin_full
    )
  if (
    nrow(fixed_hex_reference) != expected_hexes ||
      anyDuplicated(fixed_hex_reference$hex_id) ||
      any(
        fixed_hex_reference$hex_center_inside_current_austin_full &
          !fixed_hex_reference$hex_intersects_current_austin_full
      ) ||
      any(
        panel$source_covered &
          !panel$hex_center_inside_current_austin_full
      ) ||
      any(
        panel$hex_center_inside_current_austin_full &
          panel$source_covered != panel$source_covered_before_city_filter
      )
  ) {
    stop(
      "Demolition panel is inconsistent with the fixed City study geography.",
      call. = FALSE
    )
  }
  outside_fixed_study <- !panel$hex_center_inside_current_austin_full
  if (
    any(!is.na(
      panel$residential_demolition_permits[outside_fixed_study]
    )) ||
      any(!is.na(
        panel$residential_demolition_permits_observed_to_date[
          outside_fixed_study
        ]
      ))
  ) {
    stop(
      "Rows outside the fixed City study geography contain demolition counts.",
      call. = FALSE
    )
  }
  if (!setequal(unique(panel$outcome_year), panel_years)) {
    stop("Demolition panel years do not match panel_years.", call. = FALSE)
  }
  if (!setequal(
    sort(unique(panel$outcome_year[panel$period_complete])),
    sort(as.integer(complete_years))
  )) {
    stop("Demolition period_complete flags are incorrect.", call. = FALSE)
  }
  if (!identical(
    panel$count_observed,
    panel$source_covered &
      panel$period_complete &
      panel$measurement_complete
  )) {
    stop("Demolition count_observed flags are inconsistent.", call. = FALSE)
  }
  if (any(!is.na(panel$residential_demolition_permits[!panel$count_observed]))) {
    stop("Incomplete or uncovered rows contain annual outcome labels.", call. = FALSE)
  }
  if (any(is.na(panel$residential_demolition_permits[panel$count_observed]))) {
    stop("Observed demolition rows have missing annual outcome labels.", call. = FALSE)
  }
  if (any(
    !is.na(panel$residential_demolition_permits_observed_to_date) &
      !panel$source_covered
  )) {
    stop("Uncovered demolition rows contain zero-filled counts.", call. = FALSE)
  }
  if (
    anyNA(panel$measurement_complete) ||
      any(!panel$measurement_complete)
  ) {
    stop(
      "Demolition measurement_complete must be true for every annual row; ",
      "coverage uncertainty belongs in the coverage fields.",
      call. = FALSE
    )
  }
  if (
    anyNA(panel$analysis_as_of_date) ||
      length(unique(as.Date(panel$analysis_as_of_date))) != 1L
  ) {
    stop("Demolition panel must record one analysis_as_of_date.", call. = FALSE)
  }
  count_columns <- c(
    "residential_demolition_permits",
    "residential_demolition_permits_observed_to_date"
  )
  if (any(vapply(
    panel[count_columns],
    function(x) any(x < 0, na.rm = TRUE),
    logical(1)
  ))) {
    stop("Demolition panel contains negative counts.", call. = FALSE)
  }
  if (!all(annual_qa$reconciliation_pass)) {
    bad_years <- annual_qa$outcome_year[!annual_qa$reconciliation_pass]
    stop(
      "Demolition annual reconciliation failed for: ",
      paste(bad_years, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

build_demolition_panel_artifacts <- function(
  raw_permits,
  hex_grid,
  hex_coverage,
  fixed_study_boundary,
  source_start_date = as.Date("2009-10-02"),
  observed_through_date = as.Date("2026-04-01"),
  analysis_as_of_date = observed_through_date,
  panel_years = 2009:2026,
  complete_years = 2010:2025,
  analysis_crs = 3083
) {
  validate_demolition_inputs(raw_permits, hex_grid, hex_coverage)
  if (
    nrow(hex_coverage) != nrow(hex_grid) * length(unique(panel_years)) ||
      !setequal(hex_coverage$hex_id, hex_grid$hex_id) ||
      !setequal(hex_coverage$outcome_year, panel_years)
  ) {
    stop(
      "Historical demolition coverage does not match the requested grid/years.",
      call. = FALSE
    )
  }
  prepared <- prepare_residential_demolition_permits(
    raw_permits = raw_permits,
    observed_through_date = observed_through_date,
    panel_years = panel_years
  )
  assigned <- assign_demolition_permits_to_hex(
    eligible_permits = prepared$eligible,
    hex_grid = hex_grid,
    hex_coverage = hex_coverage,
    fixed_study_boundary = fixed_study_boundary,
    analysis_crs = analysis_crs
  )
  if (
    any(assigned$match_status == "pending_spatial_match") ||
      any(
        assigned$match_status == "mapped_covered_hex" &
          (
            !assigned$point_inside_current_austin_full |
              !assigned$inside_city_study_hex |
              !assigned$matched_source_covered
          )
      ) ||
      any(
        assigned$match_status ==
          "inside_city_point_outside_center_selected_city_hexes" &
          (
            !assigned$point_inside_current_austin_full |
              assigned$inside_city_study_hex
          )
      ) ||
      any(
        assigned$match_status == "outside_current_austin_full_purpose" &
          assigned$point_inside_current_austin_full
      )
  ) {
    stop(
      "Demolition permit assignments violate the fixed City geography.",
      call. = FALSE
    )
  }
  panel <- build_demolition_annual_panel(
    assigned_permits = assigned,
    hex_coverage = hex_coverage,
    panel_years = panel_years,
    complete_years = complete_years,
    source_start_date = source_start_date,
    observed_through_date = observed_through_date,
    analysis_as_of_date = analysis_as_of_date
  )
  qa <- build_demolition_qa(
    prepared = prepared,
    assigned_permits = assigned,
    panel = panel,
    hex_coverage = hex_coverage,
    panel_years = panel_years
  )
  validate_demolition_panel(
    panel = panel,
    annual_qa = qa$annual_qa,
    expected_hexes = nrow(hex_grid),
    panel_years = panel_years,
    complete_years = complete_years
  )

  c(
    list(panel = panel),
    qa,
    list(coverage = hex_coverage, assigned_permits = assigned)
  )
}
