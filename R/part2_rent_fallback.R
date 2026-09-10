# Isolated historical rent-series audit. This does not alter production ACS
# selection, scoring, caches, globals, or files. One geography must support all
# six vintages; the two dates may not mix geography-specific rent series.
part2_rent_required_years <- function() c(2013L, 2014L, 2018L, 2019L, 2023L, 2024L)

part2_rent_check_limit <- function(relative_moe_limit) {
  if (!is.numeric(relative_moe_limit) || length(relative_moe_limit) != 1L ||
      !is.finite(relative_moe_limit) || relative_moe_limit < 0) {
    stop("Rent relative-MOE limit must be one finite nonnegative number.", call. = FALSE)
  }
  invisible(TRUE)
}

part2_rent_complete_candidates <- function(candidates) {
  required <- c("hex_id", "acs_year", "source_geography", "source_geoid", "estimate", "moe")
  if (!is.data.frame(candidates) || !nrow(candidates) || !all(required %in% names(candidates)) ||
      anyDuplicated(names(candidates)) || !is.atomic(candidates$hex_id) || is.factor(candidates$hex_id) ||
      anyNA(candidates$hex_id) || any(!nzchar(trimws(as.character(candidates$hex_id)))) ||
      (is.numeric(candidates$hex_id) && any(!is.finite(candidates$hex_id))) ||
      !is.numeric(candidates$acs_year) || anyNA(candidates$acs_year) ||
      any(!candidates$acs_year %in% part2_rent_required_years()) ||
      !is.character(candidates$source_geography) || anyNA(candidates$source_geography) ||
      any(!candidates$source_geography %in% c("block_group", "tract")) ||
      !is.character(candidates$source_geoid) || !is.numeric(candidates$estimate) || !is.numeric(candidates$moe)) {
    stop("Invalid historical rent candidate schema or keys.", call. = FALSE)
  }
  # Different GEOIDs do not make multiple candidates for the same geography and
  # year safe: the spatial assignment must already have resolved to one source.
  keys <- c("hex_id", "acs_year", "source_geography")
  if (anyDuplicated(candidates[keys])) stop("Duplicate historical rent hex/year/geography keys.", call. = FALSE)
  if ("candidate_present" %in% names(candidates) &&
      (!is.logical(candidates$candidate_present) || anyNA(candidates$candidate_present))) {
    stop("Invalid rent candidate_present placeholder flag.", call. = FALSE)
  }
  hexes <- sort(unique(candidates$hex_id))
  years <- part2_rent_required_years()
  expected <- data.frame(hex_id = rep(hexes, each = 2L * length(years)),
    acs_year = rep(rep(years, each = 2L), times = length(hexes)),
    source_geography = rep(c("block_group", "tract"), times = length(hexes) * length(years)))
  key <- function(x) paste(x$hex_id, x$acs_year, x$source_geography, sep = "|")
  position <- match(key(expected), key(candidates))
  result <- as.data.frame(candidates[position, , drop = FALSE])
  present <- !is.na(position)
  if ("candidate_present" %in% names(candidates)) present <- present & candidates$candidate_present[position] %in% TRUE
  for (field in keys) result[[field]] <- expected[[field]]
  result$candidate_present <- present
  rownames(result) <- NULL
  result
}

part2_select_rent_series <- function(candidates, relative_moe_limit = .30) {
  part2_rent_check_limit(relative_moe_limit)
  candidates <- part2_rent_complete_candidates(candidates)
  estimate_valid <- is.finite(candidates$estimate) & candidates$estimate > 0
  moe_valid <- is.finite(candidates$moe) & candidates$moe >= 0
  ratio_valid <- candidates$candidate_present & estimate_valid & moe_valid
  candidates$relative_moe <- NA_real_
  candidates$relative_moe[ratio_valid] <- candidates$moe[ratio_valid] / candidates$estimate[ratio_valid]
  candidates$reliable <- ratio_valid & is.finite(candidates$relative_moe) &
    candidates$relative_moe <= relative_moe_limit
  candidates$reliable[is.na(candidates$reliable)] <- FALSE
  # Missing provenance is audited separately: the stipulated reliability rule is
  # about the estimate and its own MOE, not a guessed replacement GEOID.
  candidates$source_geoid_missing <- is.na(candidates$source_geoid) | !nzchar(trimws(candidates$source_geoid))
  candidates$failure_reason <- ifelse(!candidates$candidate_present, "missing_vintage",
    ifelse(is.na(candidates$estimate), "estimate_missing",
      ifelse(!is.finite(candidates$estimate), "estimate_nonfinite",
        ifelse(candidates$estimate <= 0, "estimate_nonpositive",
          ifelse(is.na(candidates$moe), "moe_missing",
            ifelse(!is.finite(candidates$moe), "moe_nonfinite",
              ifelse(candidates$moe < 0, "moe_negative",
                ifelse(!is.finite(candidates$relative_moe), "relative_moe_nonfinite",
                  ifelse(!candidates$reliable, "relative_moe_above_limit", "reliable")))))))))
  hexes <- sort(unique(candidates$hex_id))
  selection <- data.frame(hex_id = hexes)
  years <- part2_rent_required_years()
  date_years <- list(`2025` = c(2013L, 2018L, 2023L), `2026` = c(2014L, 2019L, 2024L))
  for (geography in c("block_group", "tract")) {
    prefix <- if (geography == "block_group") "bg" else "tract"
    # Completion above guarantees hex-major, year-major order with one row per
    # geography/year. Scan the full table once per geography, then inspect only
    # six values per hex rather than repeatedly scanning all candidates per hex.
    reliability <- matrix(candidates$reliable[candidates$source_geography == geography],
      nrow = length(hexes), ncol = length(years), byrow = TRUE)
    for (date in names(date_years)) selection[[paste0(prefix, "_reliable_", date)]] <-
      rowSums(reliability[, match(date_years[[date]], years), drop = FALSE]) == length(date_years[[date]])
    selection[[paste0(prefix, "_reliable_both")]] <- selection[[paste0(prefix, "_reliable_2025")]] &
      selection[[paste0(prefix, "_reliable_2026")]]
    selection[[paste0(prefix, "_failed_years")]] <- vapply(seq_along(hexes), function(row) {
      paste(years[!reliability[row, ]], collapse = ";")
    }, character(1))
  }
  selection$selected_geography <- ifelse(selection$bg_reliable_both, "block_group",
    ifelse(selection$tract_reliable_both, "tract", NA_character_))
  selection$fallback_reason <- ifelse(selection$bg_reliable_both, "block_group_reliable_all_six_vintages",
    ifelse(selection$tract_reliable_both, "block_group_series_unreliable_using_tract",
      "neither_geography_reliable_all_six_vintages"))
  selection$relative_moe_limit <- relative_moe_limit
  list(selection = selection, candidates = candidates)
}

part2_rent_series_features <- function(candidates, selection, cpi, base_year = 2024L) {
  if (!is.data.frame(selection) || !nrow(selection) ||
      !all(c("hex_id", "selected_geography", "relative_moe_limit") %in% names(selection)) ||
      anyNA(selection$hex_id) || anyDuplicated(selection$hex_id) ||
      !is.character(selection$selected_geography) ||
      any(!is.na(selection$selected_geography) & !selection$selected_geography %in% c("block_group", "tract")) ||
      length(unique(selection$relative_moe_limit)) != 1L) stop("Invalid fixed historical rent selection.", call. = FALSE)
  limit <- unique(selection$relative_moe_limit)
  part2_rent_check_limit(limit)
  reviewed <- part2_select_rent_series(candidates, relative_moe_limit = limit)
  expected <- reviewed$selection
  if (!setequal(selection$hex_id, expected$hex_id) ||
      !identical(selection$selected_geography[match(expected$hex_id, selection$hex_id)], expected$selected_geography)) {
    stop("Rent selection does not match the validated fixed six-vintage rule.", call. = FALSE)
  }
  candidates <- reviewed$candidates
  selection <- expected
  if (!is.numeric(base_year) || length(base_year) != 1L || !is.finite(base_year) ||
      base_year != as.integer(base_year) || !is.numeric(cpi) || is.null(names(cpi)) ||
      anyNA(names(cpi)) || any(!nzchar(names(cpi))) || anyDuplicated(names(cpi)) ||
      any(!is.finite(cpi) | cpi <= 0) ||
      !all(as.character(c(part2_rent_required_years(), base_year)) %in% names(cpi))) {
    stop("Rent CPI must provide unique positive finite values for the six vintages and dollar base year.", call. = FALSE)
  }
  dates <- as.Date(c("2025-04-01", "2026-04-01"))
  date_years <- list(c(earliest = 2013L, previous = 2018L, current = 2023L),
    c(earliest = 2014L, previous = 2019L, current = 2024L))
  output <- lapply(seq_along(dates), function(i) {
    result <- data.frame(hex_id = selection$hex_id, analysis_as_of_date = dates[[i]],
      selected_geography = selection$selected_geography,
      rent_level = NA_real_, rent_growth = NA_real_, rent_acceleration = NA_real_,
      rent_series_supported = !is.na(selection$selected_geography),
      fallback_reason = selection$fallback_reason, relative_moe_limit = limit,
      rent_dollar_base_year = as.integer(base_year))
    years <- date_years[[i]]
    for (position in c("current", "previous", "earliest")) {
      year <- unname(years[[position]])
      vintage <- candidates[candidates$acs_year == year, ]
      wanted <- paste(selection$hex_id, selection$selected_geography, sep = "|")
      available <- paste(vintage$hex_id, vintage$source_geography, sep = "|")
      row <- match(wanted, available)
      estimate <- vintage$estimate[row]
      moe <- vintage$moe[row]
      multiplier <- unname(cpi[as.character(base_year)] / cpi[as.character(year)])
      result[[paste0("acs_year_", position)]] <- year
      result[[paste0("source_geoid_", position)]] <- vintage$source_geoid[row]
      result[[paste0("source_geoid_", position, "_missing")]] <- ifelse(result$rent_series_supported,
        vintage$source_geoid_missing[row], NA)
      result[[paste0("rent_", position, "_nominal")]] <- estimate
      result[[paste0("rent_", position, "_moe_nominal")]] <- moe
      result[[paste0("rent_", position, "_real")]] <- estimate * multiplier
      result[[paste0("rent_", position, "_moe_real")]] <- moe * multiplier
    }
    supported <- result$rent_series_supported
    result$rent_level[supported] <- result$rent_current_real[supported]
    result$rent_growth[supported] <- 100 * (log(result$rent_current_real[supported]) - log(result$rent_previous_real[supported])) / 5
    previous_growth <- 100 * (log(result$rent_previous_real[supported]) - log(result$rent_earliest_real[supported])) / 5
    result$rent_acceleration[supported] <- result$rent_growth[supported] - previous_growth
    if (any(!is.finite(as.matrix(result[supported, c("rent_level", "rent_growth", "rent_acceleration")])))) {
      stop("Chosen historical rent series produced nonfinite real-dollar features.", call. = FALSE)
    }
    result
  })
  result <- do.call(rbind, output)
  rownames(result) <- NULL
  result
}
