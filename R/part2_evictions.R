# Paired recorded/mapped filing proxies, not completed eviction/displacement.
part2_eviction_components <- function() c("eviction_latest_12mo_per_100_units",
  "eviction_latest_12mo_rate_change_per_100_units")

part2_eviction_windows <- function(cutoff, history_start = as.Date("2022-01-01")) {
  cutoff <- as.Date(cutoff); history_start <- as.Date(history_start)
  if (length(cutoff) != 1L || is.na(cutoff) || length(history_start) != 1L || is.na(history_start)) stop("Invalid eviction dates.")
  recent <- lubridate::`%m-%`(cutoff, lubridate::years(1)) + 1L
  previous <- lubridate::`%m-%`(recent, lubridate::years(1))
  if (history_start > previous) stop("Eviction history must contain both rolling windows.")
  list(cutoff = cutoff, history_start = history_start, recent_start = recent, previous_start = previous,
    history_days = as.integer(cutoff - history_start + 1L), recent_days = as.integer(cutoff - recent + 1L),
    previous_days = as.integer(recent - previous))
}

part2_eviction_read_filings <- function(path, county, registry) {
  x <- readr::read_csv(path, col_types = readr::cols(.default = readr::col_character()), show_col_types = FALSE)
  required <- c("case_number", "file_date", "jp_district", "address_for_geocoding", "geocoding_candidate")
  eviction_panel_required_columns(x, required, "Prepared eviction filings")
  case <- eviction_panel_normalize_case_number(x$case_number)
  jp <- toupper(trimws(x$jp_district))
  if (anyNA(case) || anyNA(jp) || any(!jp %in% paste0("JP", 1:5))) stop("Missing/invalid namespaced case identity.")
  uid <- eviction_panel_normalize_case_number(paste(county, jp, case, sep = ":"))
  identity_valid <- rep(TRUE, nrow(x))
  if ("case_uid" %in% names(x)) {
    identity_valid <- !is.na(x$case_uid) & nzchar(x$case_uid)
    if (any(eviction_panel_normalize_case_number(x$case_uid[identity_valid]) != uid[identity_valid])) stop("Prepared case UID disagrees with county/court namespace.")
    # Invalid upstream identifiers stay uncountable. A row-specific audit key
    # preserves their candidate location/date for uncertainty, never a case ID.
    uid[!identity_valid] <- paste0(toupper(county), ":UNIDENTIFIED_SOURCE_ROW:", which(!identity_valid))
  }
  data.frame(case_number = uid, case_number_raw = case, file_date = as.Date(x$file_date),
    source_county = county, jp_district = jp, address_for_geocoding = x$address_for_geocoding,
    geocoding_candidate = toupper(x$geocoding_candidate) %in% c("TRUE", "T", "1"),
    case_identity_valid = identity_valid, geocode_registry = unname(registry), stringsAsFactors = FALSE)
}

part2_eviction_read_geocodes <- function(path, registry, filings) {
  x <- readr::read_csv(path, col_types = readr::cols(address_for_geocoding = readr::col_character(),
    status = readr::col_character(), score = readr::col_double(), longitude = readr::col_double(),
    latitude = readr::col_double(), .default = readr::col_skip()), show_col_types = FALSE)
  candidates <- unique(filings$address_for_geocoding[filings$geocoding_candidate & !is.na(filings$address_for_geocoding) &
                                                     nzchar(filings$address_for_geocoding)])
  if (anyNA(x$address_for_geocoding) || anyDuplicated(x$address_for_geocoding) ||
      !setequal(x$address_for_geocoding, candidates)) stop("Geocode registry does not exactly cover prepared candidate addresses.")
  dplyr::mutate(x, geocode_registry = registry)
}

part2_eviction_city_reference <- function(grid, city, crs = 3083) {
  projected <- sf::st_transform(grid, crs); city <- sf::st_transform(city, crs)
  centers <- suppressWarnings(sf::st_point_on_surface(projected))
  data.frame(hex_id = grid$hex_id,
    eviction_inside_current_city = lengths(sf::st_covered_by(centers, city)) > 0L,
    eviction_hex_intersects_city = lengths(sf::st_intersects(projected, city)) > 0L,
    eviction_boundary_straddling_hex = lengths(sf::st_intersects(projected, city)) > 0L &
      lengths(sf::st_covered_by(projected, city)) == 0L)
}

part2_eviction_source_coverage <- function(hex_counties, source_config, jp_reference, cutoff,
                                         history_start = as.Date("2022-01-01")) {
  w <- part2_eviction_windows(cutoff, history_start)
  # Historical counts remain diagnostics, not a pre-window eligibility gate.
  check_start <- w$previous_start
  years <- seq(as.integer(format(check_start, "%Y")), as.integer(format(cutoff, "%Y")))
  # This uses source-interval metadata, never annual outcome counts or an
  # annual-completeness label. The final requested year ends on April 1.
  segments <- build_eviction_hex_year_coverage(hex_counties, years, source_config, jp_reference, cutoff) %>%
    dplyr::mutate(requested_start = pmax(as.Date(paste0(outcome_year, "-01-01")), check_start),
      requested_end = pmin(as.Date(paste0(outcome_year, "-12-31")), cutoff),
      segment_complete = source_covered & !is.na(coverage_start_date) & !is.na(coverage_end_date) &
        coverage_start_date <= requested_start & coverage_end_date >= requested_end)
  summary <- segments %>% dplyr::group_by(hex_id) %>% dplyr::summarise(
    eviction_scored_window_source_covered = all(segment_complete),
    eviction_source_ids = eviction_panel_format_values(coverage_source_ids),
    eviction_court_vintages = eviction_panel_format_values(coverage_boundary_vintage),
    eviction_coverage_jps = eviction_panel_format_values(coverage_jp_district),
    eviction_source_uncovered_reason = eviction_panel_format_values(uncovered_reason), .groups = "drop")
  list(segments = segments, coverage = summary)
}

part2_eviction_resolve <- function(filings, geocodes, grid, hex_counties, city, city_reference,
                                  annual_coverage, history_start = as.Date("2022-01-01"),
                                  max_cutoff = as.Date("2026-04-01"), crs = 3083) {
  # Keep ALL rows/dates of a potentially relevant case. Filtering rows first
  # could hide a conflicting filing date on the other side of the cutoff.
  relevant <- filings %>% dplyr::filter(is.na(file_date) | (file_date >= history_start & file_date <= max_cutoff)) %>%
    dplyr::distinct(case_number)
  source <- dplyr::semi_join(filings, relevant, by = "case_number")
  reliable <- source %>% dplyr::inner_join(geocodes, by = c("geocode_registry", "address_for_geocoding"), na_matches = "never") %>%
    dplyr::filter(status %in% c("M", "T"), score >= 90, is.finite(longitude), is.finite(latitude),
                  longitude >= -180, longitude <= 180, latitude >= -90, latitude <= 90)
  reliable$row_id <- seq_len(nrow(reliable))
  target <- sf::st_transform(grid, crs) %>% dplyr::select(hex_id) %>%
    dplyr::left_join(dplyr::rename(hex_counties, target_source_county = source_county), by = "hex_id") %>%
    dplyr::left_join(city_reference, by = "hex_id")
  points <- sf::st_transform(sf::st_as_sf(reliable, coords = c("longitude", "latitude"), crs = 4326, remove = FALSE), crs)
  points$point_inside_current_city <- lengths(sf::st_covered_by(points, sf::st_transform(city, crs))) > 0L
  # Intersections retain edge ties as multiple candidates for the conservative
  # case resolver instead of silently assigning/dropping a boundary point.
  matches <- sf::st_intersects(points, target)
  points$point_hex_matches <- lengths(matches)
  evidence <- sf::st_join(points, target, join = sf::st_intersects, left = TRUE) %>% sf::st_drop_geometry() %>%
    dplyr::mutate(outcome_year = as.integer(format(file_date, "%Y"))) %>%
    dplyr::left_join(annual_coverage %>% dplyr::select(hex_id, outcome_year, source_covered, coverage_jp_district),
                     by = c("hex_id", "outcome_year")) %>%
    dplyr::mutate(inside_source_county = !is.na(hex_id) & source_county == target_source_county,
      inside_source_jp = dplyr::case_when(!inside_source_county ~ FALSE, source_county == "Travis" ~ TRUE,
        source_county == "Williamson" ~ dplyr::coalesce(source_covered, FALSE) & jp_district == coverage_jp_district,
        TRUE ~ FALSE),
      inside_source_geography = point_inside_current_city & eviction_inside_current_city %in% TRUE &
        inside_source_county & inside_source_jp %in% TRUE,
      source_geography_status = dplyr::case_when(is.na(hex_id) ~ "outside_grid",
        !point_inside_current_city ~ "outside_city_point", !eviction_inside_current_city ~ "outside_city_center_scope",
        !inside_source_county ~ "cross_county_hex", !inside_source_jp %in% TRUE ~ "outside_effective_supplied_court",
        TRUE ~ "inside_source_geography"))
  accepted <- evidence %>% dplyr::filter(inside_source_geography)
  resolved <- resolve_eviction_case_hexes(source, accepted,
    reliably_geocoded_case_numbers = unique(reliable$case_number),
    reliably_geocoded_outside_grid_case_numbers = unique(evidence$case_number[is.na(evidence$hex_id)]),
    reliably_geocoded_outside_study_case_numbers = unique(evidence$case_number[!is.na(evidence$hex_id) & !evidence$inside_source_geography]))
  invalid_ids <- if ("case_identity_valid" %in% names(source)) unique(source$case_number[!source$case_identity_valid]) else character()
  invalid <- resolved$cases$case_number %in% invalid_ids
  resolved$cases$assignment_status[invalid] <- "excluded_missing_valid_case_identifier"
  resolved$cases$assigned_hex_key[invalid] <- NA_character_
  resolved$assigned_cases <- dplyr::filter(resolved$assigned_cases, !case_number %in% invalid_ids)
  resolved$issues <- dplyr::filter(resolved$cases, assignment_status != "assigned_unique_hex")
  # All physical candidates are retained for date/county/court conflicts too,
  # not just the three location-conflict statuses in the legacy annual helper.
  candidates <- evidence %>% dplyr::filter(!is.na(hex_id), point_inside_current_city,
                                         eviction_inside_current_city %in% TRUE) %>%
    dplyr::distinct(case_number, hex_id)
  list(source = source, cases = resolved$cases, assigned = resolved$assigned_cases,
       issues = resolved$issues, row_qc = resolved$row_qc, evidence = evidence, candidates = candidates)
}

part2_eviction_uncertainty <- function(cases, candidates, source, cutoff, history_start) {
  possible <- source %>% dplyr::filter(is.na(file_date) | (file_date >= history_start & file_date <= cutoff)) %>%
    dplyr::distinct(case_number)
  ambiguous <- cases %>% dplyr::filter(assignment_status %in% c("excluded_inconsistent_source_county",
    "excluded_inconsistent_source_jp", "excluded_missing_valid_case_identifier", "excluded_missing_filing_date", "excluded_inconsistent_filing_dates",
    "excluded_multiple_hexes", "excluded_mixed_inside_outside_grid", "excluded_mixed_inside_outside_study_geography")) %>%
    dplyr::semi_join(possible, by = "case_number") %>% dplyr::select(case_number, assignment_status)
  dplyr::inner_join(ambiguous, candidates, by = "case_number") %>% dplyr::distinct(case_number, hex_id, .keep_all = TRUE)
}

part2_eviction_snapshot <- function(resolved, support, coverage, cutoff, history_start = as.Date("2022-01-01"), minimum_units = 20) {
  w <- part2_eviction_windows(cutoff, history_start)
  if (anyDuplicated(support$hex_id) || anyDuplicated(coverage$hex_id) || !setequal(support$hex_id, coverage$hex_id)) stop("Eviction support/coverage mismatch.")
  if (any(!is.finite(support$area_km2) | support$area_km2 <= 0) || any(is.na(support$residential_units) | !is.finite(support$residential_units) | support$residential_units < 0)) stop("Invalid fixed eviction denominators.")
  membership <- resolved$cases %>% dplyr::mutate(analysis_as_of_date = cutoff,
    in_history = !is.na(file_date) & file_date >= history_start & file_date <= cutoff,
    event_window = dplyr::case_when(!is.na(file_date) & file_date >= w$recent_start & file_date <= cutoff ~ "recent",
      !is.na(file_date) & file_date >= w$previous_start & file_date < w$recent_start ~ "previous", TRUE ~ "outside"),
    hex_id = support$hex_id[match(assigned_hex_key, as.character(support$hex_id))])
  counts <- membership %>% dplyr::filter(assignment_status == "assigned_unique_hex", in_history) %>%
    dplyr::group_by(hex_id) %>% dplyr::summarise(eviction_history_observed_cases = dplyr::n(),
      eviction_recent_observed_cases = sum(event_window == "recent"), eviction_previous_observed_cases = sum(event_window == "previous"), .groups = "drop")
  uncertainty <- part2_eviction_uncertainty(resolved$cases, resolved$candidates, resolved$source, cutoff, w$previous_start)
  uncertain_counts <- uncertainty %>% dplyr::count(hex_id, name = "eviction_unresolved_candidate_cases")
  f <- support %>% dplyr::left_join(coverage, by = "hex_id") %>% dplyr::left_join(counts, by = "hex_id") %>%
    dplyr::left_join(uncertain_counts, by = "hex_id") %>% dplyr::mutate(
      dplyr::across(dplyr::all_of(c("eviction_history_observed_cases", "eviction_recent_observed_cases", "eviction_previous_observed_cases", "eviction_unresolved_candidate_cases")), ~dplyr::coalesce(.x, 0L)),
      analysis_as_of_date = cutoff, eviction_history_start = history_start,
      eviction_eligibility_window_start = w$previous_start,
      eviction_eligibility_window_end = cutoff,
      eviction_eligibility_rule = "rolling_scored_24_months_v1",
      eviction_history_days = w$history_days, eviction_recent_window_days = w$recent_days,
      eviction_previous_window_days = w$previous_days, eviction_recent_window_start = w$recent_start,
      eviction_previous_window_start = w$previous_start,
      eviction_source_covered = eviction_inside_current_city & eviction_scored_window_source_covered,
      eviction_count_observed = eviction_source_covered & eviction_unresolved_candidate_cases == 0L,
      eviction_coverage_reason = dplyr::case_when(!eviction_inside_current_city ~ "outside_fixed_city_center_scope",
        !eviction_scored_window_source_covered ~ "county_court_or_scored_window_uncovered",
        eviction_unresolved_candidate_cases > 0L ~ "localizable_case_ambiguity_in_scored_window", TRUE ~ "covered_mapped_filing_proxy"),
      eviction_cases_total = dplyr::if_else(eviction_count_observed, eviction_history_observed_cases, NA_integer_),
      eviction_cases_latest_12mo = dplyr::if_else(eviction_count_observed, eviction_recent_observed_cases, NA_integer_),
      eviction_cases_previous_12mo = dplyr::if_else(eviction_count_observed, eviction_previous_observed_cases, NA_integer_),
      eviction_valid_zero_recent = eviction_count_observed & eviction_recent_observed_cases == 0L,
      eviction_rate_units_denominator = dplyr::if_else(residential_units >= minimum_units, residential_units, NA_real_),
      eviction_latest_12mo_per_100_units = 100 * eviction_cases_latest_12mo / eviction_rate_units_denominator,
      eviction_latest_12mo_rate_change_per_100_units = 100 *
        (eviction_cases_latest_12mo - eviction_cases_previous_12mo) / eviction_rate_units_denominator,
      eviction_cases_latest_12mo_change_pct = dplyr::if_else(eviction_cases_previous_12mo > 0,
        100 * (eviction_cases_latest_12mo / eviction_cases_previous_12mo - 1), NA_real_),
      eviction_recent_share = dplyr::if_else(eviction_cases_total > 0, eviction_cases_latest_12mo / eviction_cases_total, NA_real_),
      eviction_all_filing_locations_complete = FALSE, eviction_rate_minimum_units = minimum_units,
      eviction_measure = "recorded_unique_reliably_mapped_filings_not_completed_displacement",
      eviction_history_rule = "since_2022_diagnostics_only_not_eligibility_or_scoring")
  membership <- membership %>% dplyr::left_join(dplyr::select(f, hex_id, eviction_count_observed), by = "hex_id") %>%
    dplyr::mutate(contributes_to_feature = in_history & assignment_status == "assigned_unique_hex" & eviction_count_observed %in% TRUE)
  list(features = f, membership = membership, uncertainty = uncertainty)
}

part2_eviction_court_window_qa <- function(resolved, cutoff, history_start = as.Date("2022-01-01")) {
  w <- part2_eviction_windows(cutoff, history_start)
  periods <- data.frame(window = c("history", "previous", "recent"),
    start = c(history_start, w$previous_start, w$recent_start),
    end = c(cutoff, w$recent_start - 1L, cutoff))
  dplyr::bind_rows(lapply(seq_len(nrow(periods)), function(i) {
    inventory <- resolved$source %>%
      dplyr::filter(is.na(file_date) | (file_date >= periods$start[i] & file_date <= periods$end[i])) %>%
      dplyr::distinct(case_number, source_county, jp_district) %>%
      dplyr::left_join(dplyr::select(resolved$cases, case_number, assignment_status, has_reliable_geocode), by = "case_number")
    inventory %>% dplyr::group_by(source_county, jp_district) %>% dplyr::summarise(
      potentially_in_window_case_or_row_units = dplyr::n(),
      identified_unique_cases = sum(assignment_status != "excluded_missing_valid_case_identifier"),
      reliable_mapped_cases = sum(assignment_status == "assigned_unique_hex"),
      no_reliable_location_cases = sum(!has_reliable_geocode & assignment_status != "excluded_missing_valid_case_identifier"),
      no_reliable_location_unidentified_rows = sum(!has_reliable_geocode & assignment_status == "excluded_missing_valid_case_identifier"),
      conflicting_or_missing_date_cases = sum(assignment_status %in% c("excluded_missing_filing_date", "excluded_inconsistent_filing_dates")),
      unidentified_source_rows = sum(assignment_status == "excluded_missing_valid_case_identifier"),
      excluded_or_ambiguous_identified_cases = sum(!assignment_status %in% c("assigned_unique_hex", "excluded_missing_valid_case_identifier")), .groups = "drop") %>%
      dplyr::mutate(analysis_as_of_date = cutoff, window = periods$window[i],
        window_start = periods$start[i], window_end = periods$end[i],
        no_reliable_location_share = no_reliable_location_cases / identified_unique_cases,
        reliable_mapped_share = reliable_mapped_cases / identified_unique_cases,
        inventory_definition = "any original candidate date in window or missing date; all source geography; shares use identified unique cases only; unidentified rows reported separately",
        all_filing_locations_complete = FALSE)
  }))
}
