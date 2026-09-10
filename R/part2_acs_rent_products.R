# Adapt a validated six-vintage rent series to the existing Part 2 ACS schema.
part2_acs_rent_products <- function(candidates, selection, series, grid, cpi) {
  ids <- grid$hex_id; dates <- as.Date(c("2025-04-01", "2026-04-01"))
  stopifnot(!anyNA(ids), !anyDuplicated(ids), setequal(selection$hex_id, ids),
    !anyDuplicated(selection$hex_id), nrow(series) == 2L * length(ids),
    !anyDuplicated(series[c("hex_id", "analysis_as_of_date")]))
  selection <- selection[match(ids, selection$hex_id), ]
  add_geometry <- function(x) {
    if (inherits(grid, "sf")) sf::st_sf(x, geometry = sf::st_geometry(grid)[match(x$hex_id, ids)]) else x
  }
  result <- lapply(seq_along(dates), function(i) {
    date <- dates[[i]]; years <- if (i == 1L) c(2013L, 2018L, 2023L) else c(2014L, 2019L, 2024L)
    s <- series[series$analysis_as_of_date == date, ]; s <- s[match(ids, s$hex_id), ]
    stopifnot(identical(s$selected_geography, selection$selected_geography))
    vintages <- lapply(years, function(year) {
      c <- candidates[candidates$acs_year == year, ]
      row <- match(paste(ids, selection$selected_geography, sep = "|"), paste(c$hex_id, c$source_geography, sep = "|"))
      supported <- s$rent_series_supported
      stopifnot(all(c$reliable[row[supported]]))
      out <- data.frame(hex_id = ids, median_rent = c$estimate[row], median_rent_moe = c$moe[row],
        median_rent_source_geoid = c$source_geoid[row], median_rent_source_geography = selection$selected_geography,
        median_rent_source_residential_share = c$source_residential_share[row],
        median_rent_source_assignment_method = c$source_assignment_method[row],
        acs_year = year, acs_survey = "acs5", analysis_as_of_date = date, acs_dollar_base_year = 2024L,
        acs_median_primary_geography = "block_group", acs_median_fallback_geography = "tract",
        cpi_u = unname(cpi[as.character(year)]),
        median_rent_real = c$estimate[row] * unname(cpi["2024"] / cpi[as.character(year)]),
        median_rent_moe_real = c$moe[row] * unname(cpi["2024"] / cpi[as.character(year)]),
        median_rent_relative_moe = c$relative_moe[row], median_rent_reliable = supported,
        acs_rent_series_supported = supported, acs_rent_fallback_reason = selection$fallback_reason,
        acs_rent_source_rule = "block_group_all_six_reliable_else_tract_all_six_else_missing")
      stopifnot(all(is.na(out$median_rent[!supported])), all(is.na(out$median_rent_moe[!supported])))
      out
    })
    vintage <- do.call(rbind, vintages)
    current <- vintages[[3]]; previous <- vintages[[2]]; earliest <- vintages[[1]]
    trend <- data.frame(hex_id = ids, acs_rent_current_year = years[3], acs_rent_previous_year = years[2],
      acs_rent_earliest_year = years[1], acs_rent_recent_interval_years = 5L, acs_rent_prior_interval_years = 5L,
      acs_rent_dollar_base_year = 2024L, acs_rent_source_geoid = current$median_rent_source_geoid,
      acs_rent_source_geography = s$selected_geography,
      acs_rent_source_residential_share = current$median_rent_source_residential_share,
      acs_rent_source_assignment_method = current$median_rent_source_assignment_method,
      acs_rent_current = s$rent_current_nominal, acs_rent_current_real = s$rent_level,
      acs_rent_growth_recent_annualized_pct = s$rent_growth,
      acs_rent_growth_prior_annualized_pct = 100 * (log(s$rent_previous_real) - log(s$rent_earliest_real)) / 5,
      acs_rent_growth_long_annualized_pct = 100 * (log(s$rent_current_real) - log(s$rent_earliest_real)) / 10,
      acs_rent_acceleration_pp = s$rent_acceleration,
      acs_rent_relative_moe_current = current$median_rent_relative_moe,
      acs_rent_relative_moe_max = pmax(current$median_rent_relative_moe, previous$median_rent_relative_moe, earliest$median_rent_relative_moe),
      acs_rent_vintages_available = ifelse(s$rent_series_supported, 3L, 0L),
      acs_rent_moe_vintages_available = ifelse(s$rent_series_supported, 3L, 0L),
      acs_rent_trend_reliable = s$rent_series_supported, analysis_as_of_date = date, acs_dollar_base_year = 2024L,
      acs_rent_series_supported = s$rent_series_supported,
      acs_rent_bg_reliable_both = selection$bg_reliable_both, acs_rent_tract_reliable_both = selection$tract_reliable_both,
      acs_rent_fallback_reason = selection$fallback_reason, acs_rent_relative_moe_limit = selection$relative_moe_limit,
      acs_rent_source_rule = "block_group_all_six_reliable_else_tract_all_six_else_missing")
    for (position in c("current", "previous", "earliest")) {
      trend[[paste0("acs_rent_", position, "_source_geoid")]] <- s[[paste0("source_geoid_", position)]]
      for (suffix in c("nominal", "moe_nominal", "real", "moe_real"))
        trend[[paste0("acs_rent_", position, "_", suffix)]] <- s[[paste0("rent_", position, "_", suffix)]]
    }
    list(vintage = add_geometry(vintage), trends = add_geometry(trend))
  })
  names(result) <- as.character(dates)
  result
}
