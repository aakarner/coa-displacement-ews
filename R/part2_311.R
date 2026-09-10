# Retrospective selected 311 counts. Geographic screens are POC assumptions,
# not a certification of 311 service coverage or all-request geocoding.
part2_311_components <- function() c("sr_311_smoke_signal_latest_12mo_per_100_units",
  "sr_311_smoke_signal_latest_12mo_density", "sr_311_smoke_signal_latest_12mo_rate_change_per_100_units")

part2_311_windows <- function(cutoff) {
  cutoff <- as.Date(cutoff)
  if (length(cutoff) != 1L || is.na(cutoff)) stop("Invalid 311 cutoff.")
  latest <- lubridate::`%m-%`(cutoff, lubridate::years(1)) + 1L
  list(analysis_as_of_date = cutoff, latest_12mo_start = latest,
       previous_12mo_start = lubridate::`%m-%`(latest, lubridate::years(1)))
}

part2_311_validate_cache <- function(cache, types) {
  required <- c("sr_number", "sr_type_desc", "sr_created_date", "sr_location_lat", "sr_location_long")
  if (!is.list(cache) || !identical(cache$schema_version, 1L) || !isTRUE(cache$complete) ||
      !is.data.frame(cache$data) || !nrow(cache$data) || !all(required %in% names(cache$data)) ||
      anyNA(types$sr_type_desc) || anyDuplicated(types$sr_type_desc) ||
      !identical(sort(cache$selected_type_descriptions), sort(types$sr_type_desc))) {
    stop("311 cache/type query contract is incomplete or inconsistent.")
  }
  if (length(cache$start_date) != 1L || length(cache$analysis_as_of_date) != 1L ||
      is.na(as.Date(cache$start_date)) || is.na(as.Date(cache$analysis_as_of_date)) ||
      as.Date(cache$start_date) > as.Date(cache$analysis_as_of_date)) stop("Invalid 311 source dates.")
  if (anyNA(cache$data$sr_number) || any(!nzchar(trimws(cache$data$sr_number)))) stop("Missing 311 request ID.")
  if (anyNA(cache$data$sr_type_desc) || any(!cache$data$sr_type_desc %in% types$sr_type_desc)) {
    stop("311 cache contains unconfigured request types.")
  }
  invisible(TRUE)
}

part2_311_prepare_events <- function(cache, types, grid, city, crs = 3083) {
  part2_311_validate_cache(cache, types)
  if (anyNA(grid$hex_id) || anyDuplicated(grid$hex_id)) stop("Invalid hex IDs.")
  raw <- cache$data
  events <- data.frame(sr_number = as.character(raw$sr_number),
    sr_type_desc = as.character(raw$sr_type_desc),
    sr_created_date = as.Date(substr(as.character(raw$sr_created_date), 1L, 10L)),
    latitude = suppressWarnings(as.numeric(raw$sr_location_lat)),
    longitude = suppressWarnings(as.numeric(raw$sr_location_long)), stringsAsFactors = FALSE)
  # Collapse only identical event identity/date/type/coordinate records. Status
  # revisions do not create a second request; conflicting event attributes stop.
  duplicates <- events[duplicated(events$sr_number) | duplicated(events$sr_number, fromLast = TRUE), ]
  events <- dplyr::distinct(events)
  if (anyDuplicated(events$sr_number)) stop("Conflicting duplicated 311 event IDs; reconcile before proceeding.")
  events$event_date_valid <- !is.na(events$sr_created_date) &
    events$sr_created_date >= as.Date(cache$start_date) &
    events$sr_created_date <= as.Date(cache$analysis_as_of_date)
  events$coordinate_valid <- is.finite(events$latitude) & is.finite(events$longitude) &
    events$latitude >= 29.8 & events$latitude <= 30.7 &
    events$longitude >= -98.3 & events$longitude <= -97.2
  events$hex_id <- grid$hex_id[rep(NA_integer_, nrow(events))]
  events$spatial_match_count <- 0L
  events$spatial_assignment <- "invalid_coordinate"
  events$hex_candidates <- NA_character_
  events$event_inside_current_city <- NA
  valid <- which(events$coordinate_valid)
  if (length(valid)) {
    points <- sf::st_as_sf(events[valid, ], coords = c("longitude", "latitude"), crs = 4326)
    points <- sf::st_transform(points, crs)
    projected <- sf::st_transform(grid, crs)
    within <- sf::st_within(points, projected)
    border <- which(lengths(within) == 0L)
    matches <- within
    if (length(border)) matches[border] <- sf::st_intersects(points[border, ], projected)
    n <- lengths(matches)
    events$spatial_match_count[valid] <- n
    events$spatial_assignment[valid] <- ifelse(n == 0L, "outside_grid",
      ifelse(n > 1L, "ambiguous_unassigned", ifelse(lengths(within) == 1L, "within_unique", "boundary_unique")))
    single <- which(n == 1L)
    events$hex_id[valid[single]] <- grid$hex_id[vapply(matches[single], `[`, integer(1), 1L)]
    events$hex_candidates[valid] <- vapply(matches, function(ids)
      if (length(ids)) paste(grid$hex_id[ids], collapse = "|") else NA_character_, character(1))
    events$event_inside_current_city[valid] <- lengths(sf::st_covered_by(points, sf::st_transform(city, crs))) > 0L
  }
  list(events = events, duplicates = duplicates,
       qa = data.frame(raw_rows = nrow(raw), distinct_events = nrow(events),
         exact_duplicate_rows_removed = nrow(raw) - nrow(events),
         invalid_date_rows = sum(!events$event_date_valid), invalid_coordinates = sum(!events$coordinate_valid),
         mapped_events = sum(!is.na(events$hex_id)), ambiguous_events = sum(events$spatial_match_count > 1L),
         mapped_outside_current_city = sum(!is.na(events$hex_id) & !events$event_inside_current_city, na.rm = TRUE)))
}

part2_311_geography <- function(grid, city, historical_baselines, dated_actions,
                               cutoffs = as.Date(c("2025-04-01", "2026-04-01")), crs = 3083) {
  projected <- sf::st_transform(sf::st_make_valid(grid), crs)
  city <- sf::st_transform(sf::st_make_valid(city), crs)
  centers <- suppressWarnings(sf::st_point_on_surface(projected))
  current <- data.frame(hex_id = grid$hex_id,
    sr_311_in_current_city_scope = lengths(sf::st_covered_by(centers, city)) > 0L,
    sr_311_hex_intersects_city = lengths(sf::st_intersects(projected, city)) > 0L,
    sr_311_hex_wholly_inside_city = lengths(sf::st_covered_by(projected, city)) > 0L)
  current$sr_311_boundary_straddling_hex <- current$sr_311_hex_intersects_city & !current$sr_311_hex_wholly_inside_city
  # Reuse geometry/state replay only. FULL is the deliberately conservative POC
  # screen; demolition permit authority in LTD/2MILE is NOT borrowed for 311.
  jurisdiction_events <- demolition_coverage_prepare_events(historical_baselines, dated_actions, analysis_crs = crs)
  matches <- sf::st_intersects(centers, jurisdiction_events)
  by_date <- lapply(seq_along(cutoffs), function(i) {
    w <- part2_311_windows(cutoffs[i])
    states <- dplyr::bind_rows(lapply(matches, demolition_coverage_resolve_period,
      events = jurisdiction_events, period_start_date = w$previous_12mo_start,
      period_end_date = w$analysis_as_of_date, supported_jurisdiction_types = "FULL"))
    names(states) <- paste0("sr_311_geography_", names(states))
    dplyr::bind_cols(current, states) %>% dplyr::mutate(analysis_as_of_date = cutoffs[i],
      sr_311_geography_period_start = w$previous_12mo_start,
      sr_311_geography_period_end = w$analysis_as_of_date,
      sr_311_geography_assumption = "fixed_current_city_centers_and_continuous_historical_FULL_not_verified_311_service_area")
  })
  dplyr::bind_rows(by_date)
}

part2_311_snapshot <- function(events, support, geography, cutoff, source_start, source_end,
                              query_complete = TRUE, minimum_units = 20) {
  w <- part2_311_windows(cutoff)
  if (anyDuplicated(support$hex_id) || anyDuplicated(geography$hex_id) ||
      !setequal(support$hex_id, geography$hex_id) || anyNA(support$hex_id)) stop("311 support/geography mismatch.")
  if (!all(c("area_km2", "residential_units") %in% names(support))) stop("Missing fixed 311 denominators.")
  if (any(!is.finite(support$area_km2) | support$area_km2 <= 0) ||
      any(!is.na(support$residential_units) & (!is.finite(support$residential_units) | support$residential_units < 0))) {
    stop("Invalid fixed 311 denominators.")
  }
  membership <- events %>% dplyr::mutate(analysis_as_of_date = w$analysis_as_of_date,
    event_window = dplyr::case_when(
      event_date_valid & sr_created_date >= w$latest_12mo_start & sr_created_date <= cutoff ~ "recent",
      event_date_valid & sr_created_date >= w$previous_12mo_start & sr_created_date < w$latest_12mo_start ~ "previous",
      TRUE ~ "outside"), mapped_city_event = !is.na(hex_id) & event_inside_current_city %in% TRUE)
  counts <- membership %>% dplyr::filter(!is.na(hex_id), event_window != "outside") %>%
    dplyr::group_by(hex_id) %>% dplyr::summarise(
      sr_311_recent_mapped_before_city_filter = sum(event_window == "recent"),
      sr_311_previous_mapped_before_city_filter = sum(event_window == "previous"),
      sr_311_recent_observed_count = sum(event_window == "recent" & mapped_city_event),
      sr_311_previous_observed_count = sum(event_window == "previous" & mapped_city_event), .groups = "drop")
  ambiguous <- membership %>% dplyr::filter(event_window != "outside", spatial_match_count > 1L,
                                            event_inside_current_city %in% TRUE)
  ambiguous_ids <- unique(unlist(strsplit(ambiguous$hex_candidates, "|", fixed = TRUE)))
  complete <- isTRUE(query_complete) && as.Date(source_start) <= w$previous_12mo_start && as.Date(source_end) >= cutoff
  features <- support %>% dplyr::left_join(geography, by = "hex_id") %>%
    dplyr::left_join(counts, by = "hex_id") %>% dplyr::mutate(
      dplyr::across(dplyr::all_of(setdiff(names(counts), "hex_id")), ~dplyr::coalesce(.x, 0L)),
      analysis_as_of_date = w$analysis_as_of_date,
      sr_311_latest_12mo_start = w$latest_12mo_start, sr_311_previous_12mo_start = w$previous_12mo_start,
      sr_311_query_complete = isTRUE(query_complete), sr_311_query_window_complete = complete,
      sr_311_all_requests_coverage_verified = FALSE,
      sr_311_ambiguous_assignment_affects_window = as.character(hex_id) %in% ambiguous_ids,
      sr_311_poc_coverage_usable = sr_311_in_current_city_scope &
        sr_311_geography_source_covered %in% TRUE & complete & !sr_311_ambiguous_assignment_affects_window,
      sr_311_coverage_reason = dplyr::case_when(
        !sr_311_in_current_city_scope ~ "outside_fixed_city_center_scope",
        !complete ~ "query_window_incomplete",
        !sr_311_geography_source_covered %in% TRUE ~ "historical_FULL_screen_failed_or_unresolved",
        sr_311_ambiguous_assignment_affects_window ~ "ambiguous_spatial_assignment",
        TRUE ~ "usable_mapped_selected_request_proxy_not_all_requests"),
      sr_311_smoke_signal_latest_12mo = dplyr::if_else(sr_311_poc_coverage_usable, sr_311_recent_observed_count, NA_integer_),
      sr_311_smoke_signal_previous_12mo = dplyr::if_else(sr_311_poc_coverage_usable, sr_311_previous_observed_count, NA_integer_),
      sr_311_valid_zero_latest = sr_311_poc_coverage_usable & sr_311_recent_observed_count == 0L,
      sr_311_valid_zero_previous = sr_311_poc_coverage_usable & sr_311_previous_observed_count == 0L,
      sr_311_rate_units_denominator = dplyr::if_else(is.finite(residential_units) & residential_units >= minimum_units,
                                                   residential_units, NA_real_),
      sr_311_smoke_signal_latest_12mo_per_100_units = 100 * sr_311_smoke_signal_latest_12mo / sr_311_rate_units_denominator,
      sr_311_smoke_signal_latest_12mo_density = sr_311_smoke_signal_latest_12mo / area_km2,
      sr_311_smoke_signal_latest_12mo_rate_change_per_100_units = 100 *
        (sr_311_smoke_signal_latest_12mo - sr_311_smoke_signal_previous_12mo) / sr_311_rate_units_denominator,
      sr_311_smoke_signal_latest_12mo_change_pct = dplyr::if_else(sr_311_smoke_signal_previous_12mo > 0,
        100 * (sr_311_smoke_signal_latest_12mo / sr_311_smoke_signal_previous_12mo - 1), NA_real_),
      sr_311_rate_minimum_units = minimum_units,
      sr_311_denominator_basis = "fixed_full_hex_area_and_promoted_residential_units_no_extra_legacy_cap")
  membership <- membership %>% dplyr::left_join(features %>%
    dplyr::select(hex_id, sr_311_poc_coverage_usable), by = "hex_id") %>%
    dplyr::mutate(contributes_to_feature = event_window != "outside" & mapped_city_event & sr_311_poc_coverage_usable %in% TRUE)
  list(features = features, membership = membership)
}
