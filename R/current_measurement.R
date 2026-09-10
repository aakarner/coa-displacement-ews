# Single-cutoff adapters for the same complete-component measurement contract
# used in Part 2. The temporal sample is not a current-snapshot eligibility gate.
current_rent_series <- function(candidates, ids, years, cpi, base_year = 2024L,
                                relative_moe_limit = .30) {
  stopifnot(length(years) == 3L, all(diff(years) == 5L), !anyDuplicated(ids),
    all(c("hex_id", "acs_year", "source_geography", "estimate", "moe", "source_geoid") %in% names(candidates)),
    is.finite(relative_moe_limit), relative_moe_limit >= 0,
    all(as.character(c(years, base_year)) %in% names(cpi)), all(is.finite(cpi) & cpi > 0))
  x <- candidates[candidates$acs_year %in% years, ]
  stopifnot(!anyDuplicated(x[c("hex_id", "acs_year", "source_geography")]))
  key <- function(x) paste(x$hex_id, x$acs_year, x$source_geography, sep = "|")
  rows <- lapply(c("block_group", "tract"), function(g) lapply(years, function(y)
    match(paste(ids, y, g, sep = "|"), key(x))))
  reliable <- lapply(rows, function(r) Reduce(`&`, lapply(r, function(i) {
    is.finite(x$estimate[i]) & x$estimate[i] > 0 & is.finite(x$moe[i]) & x$moe[i] >= 0 &
      x$moe[i] / x$estimate[i] <= relative_moe_limit
  })))
  geography <- ifelse(reliable[[1]], "block_group", ifelse(reliable[[2]], "tract", NA_character_))
  out <- data.frame(hex_id = ids, acs_rent_source_geography = geography,
    acs_rent_trend_reliable = !is.na(geography), acs_rent_series_supported = !is.na(geography),
    acs_rent_relative_moe_limit = relative_moe_limit, acs_dollar_base_year = base_year,
    acs_rent_source_selection_scope = "current_snapshot_three_vintages")
  roles <- c("earliest", "previous", "current")
  for (i in seq_along(years)) {
    pos <- match(paste(ids, years[i], geography, sep = "|"), key(x))
    multiplier <- unname(cpi[as.character(base_year)] / cpi[as.character(years[i])])
    out[[paste0("acs_rent_", roles[i], "_year")]] <- years[i]
    out[[paste0("acs_rent_", roles[i], "_real")]] <- x$estimate[pos] * multiplier
    out[[paste0("acs_rent_", roles[i], "_moe_real")]] <- x$moe[pos] * multiplier
    out[[paste0("acs_rent_", roles[i], "_source_geoid")]] <- x$source_geoid[pos]
  }
  out$acs_rent_growth_recent_annualized_pct <- 100 * log(out$acs_rent_current_real / out$acs_rent_previous_real) / 5
  out$acs_rent_growth_previous_annualized_pct <- 100 * log(out$acs_rent_previous_real / out$acs_rent_earliest_real) / 5
  out$acs_rent_acceleration_pp <- out$acs_rent_growth_recent_annualized_pct - out$acs_rent_growth_previous_annualized_pct
  out$acs_rent_growth_recent_for_clustering <- out$acs_rent_growth_recent_annualized_pct
  out$acs_rent_acceleration_for_clustering <- out$acs_rent_acceleration_pp
  out
}

current_ownership_features <- function(panel, support, tax_year = 2025L, minimum_units = 20,
                                       minimum_coverage = .95) {
  # Jointly known corporate/financialized flags in THIS year only. Unknown
  # parcels are not assigned noncorporate status, nor required in the prior year.
  p <- panel[panel$tax_year == tax_year & !is.na(panel$hex_id), ]
  stopifnot(!anyDuplicated(p$parcel_id), !anyDuplicated(support$hex_id),
    all(is.finite(p$residential_units) & p$residential_units >= 0),
    all(p$hex_id %in% support$hex_id))
  known <- !is.na(p$is_corporate_owned) & !is.na(p$has_financialized_owner)
  full <- ownership_summarise(p, "hex_id")
  observed <- ownership_summarise(p[known, ], "hex_id")
  out <- support[c("hex_id", "area_km2", "residential_units")]
  a <- match(out$hex_id, full$hex_id); b <- match(out$hex_id, observed$hex_id)
  zero_empty <- function(x) { x[is.na(x)] <- 0; x }
  stopifnot(isTRUE(all.equal(out$residential_units, zero_empty(full$residential_units[a]), check.attributes = FALSE)))
  out$residential_parcels <- zero_empty(full$residential_parcels[a])
  out$ownership_observed_parcels <- zero_empty(observed$residential_parcels[b])
  out$ownership_observed_units <- zero_empty(observed$residential_units[b])
  out$ownership_parcel_coverage <- ownership_safe_share(out$ownership_observed_parcels, out$residential_parcels)
  out$ownership_unit_coverage <- ownership_safe_share(out$ownership_observed_units, out$residential_units)
  out$ownership_current_usable <- out$ownership_observed_units >= minimum_units &
    out$ownership_parcel_coverage >= minimum_coverage & out$ownership_unit_coverage >= minimum_coverage
  out$ownership_current_usable[is.na(out$ownership_current_usable)] <- FALSE
  out$pct_corporate_units <- observed$pct_corporate_units[b]
  out$corporate_owned_units_per_km2 <- observed$corporate_owned_units[b] / out$area_km2
  out$pct_financialized_owner_parcels <- observed$pct_financialized_owner_parcels[b]
  out$corporate_owned_units <- observed$corporate_owned_units[b]
  out$corporate_owned_parcels <- observed$corporate_owned_parcels[b]
  out$financialized_owner_parcels <- observed$financialized_owner_parcels_observed[b]
  out$pct_corporate_parcels <- 100 * out$corporate_owned_parcels / out$ownership_observed_parcels
  for (column in part2_index_components("ownership_pressure_index")) out[[column]][!out$ownership_current_usable] <- NA_real_
  out$ownership_tax_year <- tax_year
  out$ownership_support_scope <- "jointly_known_parcels_current_year_only"
  out
}

current_measurement_eligibility <- function(x) {
  gates <- c("in_current_city_scope", "minimum_unit_support", "ownership_current_usable",
    "sr_311_poc_coverage_usable", "demolition_comparison_ready", "eviction_count_observed",
    "amenity_retrospective_usable", "all_required_components_available")
  stopifnot(all(gates %in% names(x)), all(vapply(x[gates], is.logical, logical(1))), !anyNA(x[gates]))
  x$primary_cluster_eligible <- rowSums(as.matrix(x[gates])) == length(gates)
  x$primary_exclusion <- "included"
  for (g in rev(gates)) x$primary_exclusion[!x[[g]]] <- g
  x
}
