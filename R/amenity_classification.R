# Shared sales-tax opening classification for Part 1 and retrospective Part 2.
# These are event-date eligibility rules, not evidence of publication by cutoff.

amenity_windows <- function(as_of, window_months = 18L) {
  as_of <- as.Date(as_of)
  if (length(as_of) != 1L || is.na(as_of) || length(window_months) != 1L ||
      is.na(window_months) || window_months <= 0 || window_months != as.integer(window_months)) {
    stop("Invalid amenity date/window configuration.", call. = FALSE)
  }
  recent_start <- lubridate::`%m-%`(as_of, lubridate::period(month = window_months)) + 1
  previous_start <- lubridate::`%m-%`(recent_start, lubridate::period(month = window_months))
  list(analysis_as_of_date = as_of, previous_window_start = previous_start,
       recent_window_start = recent_start, window_months = as.integer(window_months))
}

amenity_normalize_name <- function(x) {
  x <- stringr::str_to_upper(dplyr::coalesce(as.character(x), ""))
  x <- stringr::str_replace_all(x, "[[:punct:]]+", " ")
  x <- stringr::str_replace_all(x, "\\b(LLC|L L C|INC|CORP|LTD|LP|COMPANY|CO)\\b", " ")
  stringr::str_squish(x)
}

amenity_normalize_address <- function(x) {
  x <- stringr::str_to_upper(dplyr::coalesce(as.character(x), ""))
  x <- stringr::str_replace(x, ",.*$", "")
  x <- stringr::str_replace_all(x, "[[:punct:]]+", " ")
  x <- stringr::str_replace(x, "\\b(SUITE|STE|UNIT|APT|APARTMENT|ROOM|RM)\\b.*$", "")
  replacements <- c(" STREET\\b" = " ST", " ROAD\\b" = " RD", " AVENUE\\b" = " AVE",
                    " BOULEVARD\\b" = " BLVD", " DRIVE\\b" = " DR", " HIGHWAY\\b" = " HWY",
                    " LANE\\b" = " LN", " PARKWAY\\b" = " PKWY")
  for (pattern in names(replacements)) x <- stringr::str_replace_all(x, pattern, replacements[[pattern]])
  stringr::str_squish(x)
}

# Mechanical source-format cleanup only, retaining the original street in audit.
# Do not infer a new street, city, ZIP, or coordinates from a business name.
amenity_geocode_street <- function(x) {
  x <- stringr::str_squish(as.character(x))
  x <- stringr::str_replace(x, "^NA\\s+(?=[0-9])", "")
  stringr::str_replace(x, "^([0-9]+[A-Z]?)\\s+\\1\\b\\s*", "\\1 ")
}

amenity_validate_sales <- function(sales) {
  required <- c("tp_number", "loc_number", "loc_name", "address_number", "address_text",
                "permit_date", "juris_city", "loc_city", "loc_state", "loc_zip", "loc_county",
                "naics", "first_sale_date", "out_of_business_date")
  if (!all(required %in% names(sales))) stop("Amenity source schema is incomplete.", call. = FALSE)
  if (any(is.na(sales$tp_number) | !nzchar(trimws(sales$tp_number)) |
          is.na(sales$loc_number) | !nzchar(trimws(sales$loc_number)))) {
    stop("Amenity source has missing stable IDs.", call. = FALSE)
  }
  sales <- dplyr::distinct(sales)
  if (anyDuplicated(paste(sales$tp_number, sales$loc_number, sep = ":"))) {
    stop("Amenity source has conflicting duplicate stable IDs.", call. = FALSE)
  }
  sales
}

amenity_classify_sales <- function(sales, taxonomy, analysis_as_of, window_months = 18L) {
  sales <- amenity_validate_sales(sales)
  if (anyDuplicated(taxonomy$naics)) stop("Amenity taxonomy has duplicate NAICS.", call. = FALSE)
  windows <- amenity_windows(analysis_as_of, window_months)
  analysis_as_of <- windows$analysis_as_of_date
  previous_start <- windows$previous_window_start
  recent_start <- windows$recent_window_start
  county_lookup <- c(`105` = "Hays", `227` = "Travis", `246` = "Williamson")
  institutional_pattern <- paste("SODEXO", "ARAMARK", "COMPASS GROUP", "CHARTWELLS",
    "DELAWARE NORTH", "LEVY", "SCHOOL", "UNIVERSITY", "HOSPITAL", "MEDICAL CENTER", sep = "|")
  parse_date <- function(x) as.Date(substr(as.character(x), 1L, 10L))
  result <- sales %>%
    dplyr::transmute(
      event_id = paste(tp_number, loc_number, sep = ":"), taxpayer_number = tp_number,
      location_number = loc_number, location_name = loc_name,
      street = stringr::str_squish(paste(address_number, address_text)),
      city = dplyr::coalesce(juris_city, loc_city), state = loc_state,
      zip = stringr::str_sub(loc_zip, 1L, 5L), county_code = loc_county,
      county = unname(county_lookup[loc_county]), naics = as.character(naics),
      permit_date = parse_date(permit_date), first_sale_date = parse_date(first_sale_date),
      out_of_business_date = parse_date(out_of_business_date)) %>%
    dplyr::left_join(taxonomy, by = "naics") %>%
    dplyr::mutate(
      opening_date = first_sale_date,
      normalized_name = amenity_normalize_name(location_name),
      street_key = amenity_normalize_address(street), address_key = paste(street_key, zip, sep = "|"),
      name_filter_pass = !name_filter_required |
        stringr::str_detect(normalized_name, stringr::regex(dplyr::coalesce(name_pattern, "$^"))),
      home_business_flag = stringr::str_detect(stringr::str_to_upper(dplyr::coalesce(street, "")),
                                              "\\b(APT|APARTMENT|TRLR|TRAILER|LOT)\\b"),
      institutional_flag = category == "full_service_restaurant" &
        stringr::str_detect(normalized_name, stringr::regex(institutional_pattern)),
      category_classified = dplyr::if_else(category == "cafe" & !name_filter_pass,
                                           "other_snack_non_alcoholic", category),
      # Legacy name retained for compatibility; permit date is not publication date.
      record_available_as_of = is.na(permit_date) | permit_date <= analysis_as_of,
      source_eligible = !is.na(opening_date) & record_available_as_of &
        opening_date >= previous_start & opening_date <= analysis_as_of,
      core_index_eligible = include_in_index & name_filter_pass & !home_business_flag &
        !institutional_flag & source_eligible,
      event_window = dplyr::case_when(
        opening_date >= recent_start & opening_date <= analysis_as_of ~ "recent",
        opening_date >= previous_start & opening_date < recent_start ~ "previous", TRUE ~ "outside"),
      active_as_of = is.na(out_of_business_date) | out_of_business_date > analysis_as_of,
      first_of_month_flag = lubridate::day(opening_date) == 1L,
      january_first_flag = lubridate::month(opening_date) == 1L & lubridate::day(opening_date) == 1L,
      permit_lag_days = as.integer(permit_date - first_sale_date),
      permit_after_cutoff_flag = !is.na(permit_date) & permit_date > analysis_as_of)
  if (any(is.na(result$county))) stop("Sales-tax rows contain an unmapped county code.", call. = FALSE)
  result
}

# Preserve the archived historical establishment record when the stable ID is
# present in both sources. Later-only IDs supplement, never duplicate, it.
# A single reconciled event ledger is then windowed identically for both dates.
amenity_reconcile_sources <- function(archive, live, archive_id, live_id) {
  archive <- amenity_validate_sales(archive)
  live <- amenity_validate_sales(live)
  archive$event_id <- paste(archive$tp_number, archive$loc_number, sep = ":")
  live$event_id <- paste(live$tp_number, live$loc_number, sep = ":")
  columns <- setdiff(names(archive), "event_id")
  shared <- dplyr::inner_join(archive, live, by = "event_id", suffix = c("_archive", "_live"))
  conflicts <- dplyr::bind_rows(lapply(columns, function(column) {
    older <- as.character(shared[[paste0(column, "_archive")]])
    newer <- as.character(shared[[paste0(column, "_live")]])
    changed <- (is.na(older) != is.na(newer)) |
      (!is.na(older) & !is.na(newer) & older != newer)
    tibble::tibble(event_id = shared$event_id[changed], field = column,
                   archive_value = older[changed], live_value = newer[changed])
  }))
  dispositions <- dplyr::full_join(
    dplyr::transmute(archive, event_id, in_archive = TRUE),
    dplyr::transmute(live, event_id, in_live = TRUE), by = "event_id") %>%
    dplyr::mutate(
      in_archive = dplyr::coalesce(in_archive, FALSE), in_live = dplyr::coalesce(in_live, FALSE),
      disposition = dplyr::case_when(in_archive & in_live ~ "shared_archive_retained",
                                     in_archive ~ "archive_only_retained", TRUE ~ "live_only_supplement"),
      selected_source_id = dplyr::if_else(in_archive, archive_id, live_id),
      source_fields_changed = event_id %in% conflicts$event_id)
  ledger <- dplyr::bind_rows(archive, dplyr::anti_join(live, archive, by = "event_id")) %>%
    dplyr::left_join(dispositions, by = "event_id") %>% dplyr::arrange(event_id)
  if (anyDuplicated(ledger$event_id)) stop("Reconciled amenity ledger has duplicate IDs.", call. = FALSE)
  list(ledger = ledger, dispositions = dispositions, field_changes = conflicts)
}

# Multiple taxpayer/outlet IDs can describe the same opening. Collapse only an
# exact name + full street (including unit) + ZIP + category + opening-date match,
# after date/permit eligibility. Never merge on address alone or fuzzy names.
amenity_deduplicate_events <- function(categorized) {
  eligible <- categorized %>% dplyr::filter(core_index_eligible) %>%
    dplyr::mutate(fingerprint_complete =
      !is.na(location_name) & nzchar(trimws(location_name)) &
      !is.na(street) & nzchar(trimws(street)) & !stringr::str_detect(street, "^NA(?: |$)") &
      !is.na(zip) & nzchar(trimws(zip)) & !is.na(naics) & nzchar(trimws(naics)) &
      !is.na(opening_date),
      opening_fingerprint = paste(
      stringr::str_squish(stringr::str_to_upper(location_name)),
      stringr::str_squish(stringr::str_to_upper(street)), zip, naics,
      as.character(opening_date), sep = "|"),
      opening_fingerprint = dplyr::if_else(fingerprint_complete, opening_fingerprint,
                                            paste0("incomplete_individual_id:", event_id))) %>%
    dplyr::arrange(!in_archive, is.na(permit_date), permit_date, event_id) %>%
    dplyr::group_by(opening_fingerprint) %>%
    dplyr::mutate(opening_id = dplyr::first(event_id), alias_group_size = dplyr::n(),
                   duplicate_opening_alias = event_id != opening_id) %>% dplyr::ungroup()
  result <- categorized %>% dplyr::left_join(
    eligible %>% dplyr::select(event_id, opening_id, alias_group_size, duplicate_opening_alias),
    by = "event_id") %>% dplyr::mutate(
      duplicate_opening_alias = dplyr::coalesce(duplicate_opening_alias, FALSE),
      core_index_eligible_before_alias_dedup = core_index_eligible,
      core_index_eligible = core_index_eligible & !duplicate_opening_alias)
  list(categorized = result, alias_audit = eligible %>% dplyr::filter(alias_group_size > 1L))
}
