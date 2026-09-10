# Paired retrospective demolition features. Requires demolition_panel.R and
# demolition_coverage_history.R, which remain unchanged and independently usable.

part2_demolition_windows <- function(as_of, recent_years = 2L) {
  as_of <- as.Date(as_of)
  if (length(as_of) != 1L || is.na(as_of) || !identical(as.integer(recent_years), 2L)) {
    stop("Part 2 demolition requires one cutoff and two-year windows.", call. = FALSE)
  }
  recent_start <- lubridate::`%m-%`(as_of, lubridate::years(recent_years)) + 1
  previous_start <- lubridate::`%m-%`(recent_start, lubridate::years(recent_years))
  tibble::tibble(analysis_as_of_date = as_of, event_window = c("previous", "recent"),
    window_start = c(previous_start, recent_start), window_end = c(recent_start - 1, as_of))
}

part2_prepare_demolitions <- function(raw, observed_through_date) {
  if (!all(c(demolition_required_columns, "Description") %in% names(raw))) {
    stop("Demolition raw schema lacks required fields, including Description.", call. = FALSE)
  }
  raw$Description <- as.character(raw$Description)
  raw$is_total_demolition <- stringr::str_detect(
    dplyr::coalesce(raw$Description, ""), stringr::regex("total\\s+demo", ignore_case = TRUE))
  prepared <- prepare_residential_demolition_permits(raw,
    observed_through_date = observed_through_date, panel_years = 1900:9999)
  conflicting_total <- prepared$selected %>%
    dplyr::filter(!permit_id_missing) %>% dplyr::group_by(permit_id) %>%
    dplyr::summarise(total_flags = dplyr::n_distinct(is_total_demolition), .groups = "drop") %>%
    dplyr::filter(total_flags > 1L)
  conflicts <- prepared$duplicate_groups %>% dplyr::filter(issue_dates > 1L | coordinate_pairs > 1L)
  if (nrow(conflicts) || nrow(conflicting_total)) {
    stop("Conflicting duplicate demolition permit IDs require review before processing.", call. = FALSE)
  }
  legacy_selected <- stringr::str_detect(dplyr::coalesce(raw[["Work Class"]], ""),
      stringr::regex("^demolition$", ignore_case = TRUE)) &
    stringr::str_detect(dplyr::coalesce(raw[["Permit Class Mapped"]], ""),
      stringr::regex("residential", ignore_case = TRUE))
  strict_selected <- seq_len(nrow(raw)) %in% prepared$selected$source_row_id
  raw_audit <- tibble::tibble(source_row_id = seq_len(nrow(raw)),
    permit_id = normalize_demolition_text(raw[["Permit Num"]]),
    issue_date = parse_demolition_date(raw[["Issued Date"]]),
    work_class = raw[["Work Class"]], permit_class = raw[["Permit Class Mapped"]],
    legacy_selected = legacy_selected, selected_residential_demolition = strict_selected,
    total_demolition_description = raw$is_total_demolition,
    source_disposition = dplyr::case_when(
      !strict_selected ~ "not_residential_demolition",
      is.na(permit_id) | permit_id == "" ~ "missing_permit_id",
      is.na(issue_date) ~ "missing_issue_date",
      issue_date > observed_through_date ~ "after_source_cutoff",
      !source_row_id %in% prepared$eligible$source_row_id ~ "duplicate_permit_row",
      TRUE ~ "eligible_unique_permit"))
  list(events = prepared$eligible, source_audit = raw_audit,
       duplicate_groups = prepared$duplicate_groups,
       source_qa = tibble::tibble(metric = names(prepared$metrics), value = unname(prepared$metrics)),
       classification_disagreements = raw_audit %>%
         dplyr::filter(legacy_selected != selected_residential_demolition))
}

part2_demolition_city_reference <- function(hex_grid, current_jurisdictions,
                                           snapshot_date = as.Date("2026-04-29"), analysis_crs = 3083) {
  boundary <- current_jurisdictions %>%
    dplyr::filter(toupper(trimws(city_name)) == "CITY OF AUSTIN",
                  toupper(trimws(jurisdiction_type)) == "FULL") %>%
    sf::st_make_valid() %>% sf::st_transform(analysis_crs)
  if (!nrow(boundary)) stop("Current City FULL boundary is empty.", call. = FALSE)
  boundary <- sf::st_sf(city_study_geography = "current_austin_full_purpose_fixed",
                        geometry = sf::st_union(boundary))
  grid <- sf::st_transform(sf::st_make_valid(hex_grid), analysis_crs)
  centers <- suppressWarnings(sf::st_point_on_surface(grid))
  reference <- tibble::tibble(hex_id = as.character(hex_grid$hex_id),
    hex_center_inside_current_austin_full = lengths(sf::st_within(centers, boundary)) > 0L,
    hex_intersects_current_austin_full = lengths(sf::st_intersects(grid, boundary)) > 0L,
    hex_wholly_inside_current_austin_full = lengths(sf::st_covered_by(grid, boundary)) > 0L,
    city_boundary_snapshot_date = snapshot_date,
    city_study_geography = "current_austin_full_purpose_fixed",
    city_hex_assignment_method = "hex_point_on_surface_within_current_city_full") %>%
    dplyr::mutate(hex_straddles_current_austin_full =
      hex_intersects_current_austin_full & !hex_wholly_inside_current_austin_full)
  list(reference = reference, boundary = boundary)
}

part2_assign_demolitions <- function(events, hex_grid, city_reference, boundary, analysis_crs = 3083) {
  audit <- events %>% dplyr::mutate(hex_id = NA_character_, candidate_hex_ids = NA_character_,
    point_inside_current_austin_full = NA, inside_city_study_hex = NA,
    spatial_status = "invalid_coordinates")
  valid <- which(audit$valid_coordinates)
  ambiguous <- tibble::tibble(permit_id = character(), hex_id = character(), issue_date = as.Date(character()))
  if (!length(valid)) return(list(events = audit, ambiguous_hex_links = ambiguous))
  points <- audit[valid, ] %>% sf::st_as_sf(coords = c("longitude", "latitude"), crs = 4326,
    remove = FALSE) %>% sf::st_transform(analysis_crs)
  grid <- sf::st_transform(sf::st_make_valid(hex_grid), analysis_crs)
  inside <- lengths(sf::st_covered_by(points, sf::st_transform(boundary, analysis_crs))) > 0L
  matches <- sf::st_covered_by(points, grid)
  audit$point_inside_current_austin_full[valid] <- inside
  audit$candidate_hex_ids[valid] <- vapply(matches, function(rows) {
    if (!length(rows)) NA_character_ else paste(hex_grid$hex_id[rows], collapse = "|")
  }, character(1))
  audit$spatial_status[valid] <- dplyr::case_when(!inside ~ "outside_current_austin_full_purpose",
    lengths(matches) == 0L ~ "outside_grid", lengths(matches) > 1L ~ "multiple_hex_matches",
    TRUE ~ "pending_single_hex")
  singles <- which(lengths(matches) == 1L)
  if (length(singles)) {
    ids <- as.character(hex_grid$hex_id[vapply(matches[singles], `[[`, integer(1), 1L)])
    audit$hex_id[valid[singles]] <- ids
    center_selected <- city_reference$hex_center_inside_current_austin_full[
      match(ids, city_reference$hex_id)]
    audit$inside_city_study_hex[valid[singles]] <- center_selected
    selected_rows <- singles[inside[singles]]
    audit$spatial_status[valid[selected_rows]] <- ifelse(
      center_selected[inside[singles]], "mapped_current_city_study_hex",
      "inside_city_point_outside_center_selected_hex")
  }
  multiples <- which(inside & lengths(matches) > 1L)
  if (length(multiples)) {
    ambiguous <- dplyr::bind_rows(lapply(multiples, function(row) tibble::tibble(
      permit_id = audit$permit_id[valid[row]], hex_id = as.character(hex_grid$hex_id[matches[[row]]]),
      issue_date = audit$issue_date[valid[row]])))
  }
  list(events = audit, ambiguous_hex_links = ambiguous)
}

part2_demolition_coverage_context <- function(hex_grid, historical_baselines, dated_actions, analysis_crs = 5070) {
  events <- demolition_coverage_prepare_events(historical_baselines, dated_actions,
    analysis_crs = analysis_crs, timezone = "America/Chicago")
  points <- suppressWarnings(sf::st_point_on_surface(sf::st_transform(sf::st_make_valid(hex_grid), analysis_crs)))
  matches <- sf::st_intersects(points, events)
  keys <- vapply(matches, paste, character(1), collapse = "|")
  unique_keys <- !duplicated(keys)
  # Identical spatial event histories have identical period states. Resolve
  # each distinct history once without skipping any effective-date checkpoint.
  list(events = events, histories = matches[unique_keys], history_index = match(keys, keys[unique_keys]))
}

part2_demolition_window_coverage <- function(hex_grid, context, windows, city_reference,
  source_start = as.Date("2009-10-02"), source_end = as.Date("2026-04-01")) {
  dplyr::bind_rows(lapply(seq_len(nrow(windows)), function(row) {
    period <- windows[row, ]
    states <- dplyr::bind_rows(lapply(context$histories, demolition_coverage_resolve_period,
      events = context$events, period_start_date = period$window_start, period_end_date = period$window_end,
      supported_jurisdiction_types = c("FULL", "LTD", "2MILE")))
    dplyr::bind_cols(tibble::tibble(hex_id = as.character(hex_grid$hex_id)),
      states[context$history_index, ]) %>%
      dplyr::left_join(city_reference, by = "hex_id") %>% dplyr::mutate(
        analysis_as_of_date = period$analysis_as_of_date, event_window = period$event_window,
        window_start = period$window_start, window_end = period$window_end,
        source_period_complete = window_start >= source_start & window_end <= source_end,
        historical_source_covered = source_covered,
        window_usable = hex_center_inside_current_austin_full & source_period_complete &
          dplyr::coalesce(historical_source_covered, FALSE),
        coverage_reason = dplyr::case_when(
          !hex_center_inside_current_austin_full ~ "outside_fixed_city_study",
          !source_period_complete ~ "source_temporal_coverage_incomplete",
          is.na(historical_source_covered) ~ "historical_jurisdiction_unresolved",
          !historical_source_covered ~ "historical_jurisdiction_not_continuously_supported",
          TRUE ~ "covered_continuously"),
        coverage_basis = "effective_dated_city_jurisdiction_full_rolling_interval")
  }))
}

part2_demolition_features <- function(hex_grid, assigned, coverage, windows) {
  features <- tibble::tibble(hex_id = as.character(hex_grid$hex_id), area_km2 = as.numeric(hex_grid$area_km2))
  metadata <- coverage %>% dplyr::select(hex_id, dplyr::any_of(c(
    "hex_center_inside_current_austin_full", "hex_intersects_current_austin_full",
    "hex_wholly_inside_current_austin_full", "hex_straddles_current_austin_full",
    "city_boundary_snapshot_date", "city_study_geography", "city_hex_assignment_method"))) %>%
    dplyr::distinct()
  features <- features %>% dplyr::left_join(metadata, by = "hex_id")
  if (anyDuplicated(features$hex_id) || anyNA(features$hex_id)) stop("Invalid demolition hex grid IDs.")
  for (window in c("previous", "recent")) {
    interval <- windows[windows$event_window == window, ]
    cov <- coverage %>% dplyr::filter(event_window == window)
    if (nrow(interval) != 1L || anyDuplicated(cov$hex_id) || !setequal(features$hex_id, cov$hex_id)) {
      stop("Demolition window coverage must enumerate the complete grid.", call. = FALSE)
    }
    events <- assigned$events %>% dplyr::filter(issue_date >= interval$window_start,
      issue_date <= interval$window_end, spatial_status == "mapped_current_city_study_hex")
    counts <- events %>% dplyr::group_by(hex_id) %>% dplyr::summarise(
      observed_permits = dplyr::n_distinct(permit_id),
      observed_total_permits = dplyr::n_distinct(permit_id[is_total_demolition]), .groups = "drop")
    ambiguity <- assigned$ambiguous_hex_links %>% dplyr::filter(issue_date >= interval$window_start,
      issue_date <= interval$window_end) %>% dplyr::count(hex_id, name = "ambiguous_permits")
    values <- cov %>% dplyr::left_join(counts, by = "hex_id") %>%
      dplyr::left_join(ambiguity, by = "hex_id") %>% dplyr::mutate(
        observed_permits = dplyr::coalesce(observed_permits, 0L),
        observed_total_permits = dplyr::coalesce(observed_total_permits, 0L),
        ambiguous_permits = dplyr::coalesce(ambiguous_permits, 0L),
        count_available = window_usable & ambiguous_permits == 0L,
        permits = dplyr::if_else(count_available, observed_permits, NA_integer_),
        total_permits = dplyr::if_else(count_available, observed_total_permits, NA_integer_))
    values <- values[match(features$hex_id, values$hex_id), ]
    for (name in c("observed_permits", "observed_total_permits", "ambiguous_permits", "count_available",
                   "permits", "total_permits", "coverage_reason", "historical_source_covered")) {
      features[[paste0("demo_", window, "_", name)]] <- values[[name]]
    }
  }
  features %>% dplyr::mutate(
    analysis_as_of_date = unique(windows$analysis_as_of_date),
    demolition_previous_window_start = windows$window_start[windows$event_window == "previous"],
    demolition_recent_window_start = windows$window_start[windows$event_window == "recent"],
    demolition_window_years = 2L,
    demolition_area_basis = "canonical_full_hex_area_not_city_clipped",
    demolition_comparison_ready = demo_previous_count_available & demo_recent_count_available &
      is.finite(area_km2) & area_km2 > 0,
    demo_latest_24mo = demo_recent_permits, demo_previous_24mo = demo_previous_permits,
    demo_total_latest_24mo = demo_recent_total_permits, demo_total_previous_24mo = demo_previous_total_permits,
    demo_recent_density = dplyr::if_else(demolition_comparison_ready, demo_latest_24mo / area_km2, NA_real_),
    demo_trend = dplyr::if_else(demolition_comparison_ready, log1p(demo_latest_24mo) - log1p(demo_previous_24mo), NA_real_),
    demo_trend_positive = pmax(demo_trend, 0),
    demo_total_recent_density = dplyr::if_else(demolition_comparison_ready, demo_total_latest_24mo / area_km2, NA_real_))
}
