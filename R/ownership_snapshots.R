# Historical ownership import and fixed-support summaries for Part 2.
# This module never calls the current feature builder or replaces Part 1 files.

ownership_sha256 <- function(path) digest::digest(path, algo = "sha256", file = TRUE)

ownership_assert_hash <- function(path, expected) {
  if (!file.exists(path) || !identical(ownership_sha256(path), expected)) {
    stop("Ownership input is missing or has changed: ", path, call. = FALSE)
  }
  invisible(TRUE)
}

ownership_bool <- function(x) {
  value <- toupper(trimws(as.character(x)))
  if (any(!is.na(value) & !value %in% c("TRUE", "FALSE", "NA", ""))) {
    stop("Invalid ownership boolean; expected TRUE/FALSE/NA.", call. = FALSE)
  }
  out <- rep(NA, length(value))
  out[which(value == "TRUE")] <- TRUE
  out[which(value == "FALSE")] <- FALSE
  out
}

ownership_validate_rows <- function(rows, targets, years, rule_version) {
  required <- c("source_county", "tax_year", "parcel_id", "owner_ids", "owner_names",
                "is_owner_occupied", "has_financialized_owner", "is_corporate_owned",
                "classification_status", "classification_rule_version", "source_snapshot_id")
  if (!all(required %in% names(rows))) stop("Ownership snapshot schema is incomplete.")
  if (anyNA(rows$parcel_id) || anyDuplicated(rows[c("parcel_id", "tax_year")])) {
    stop("Missing or duplicate parcel-year ownership keys.")
  }
  if (!setequal(unique(rows$tax_year), years) ||
      !identical(unique(rows$classification_rule_version), rule_version)) {
    stop("Ownership year or classifier contract changed.")
  }
  for (year in years) {
    r <- rows[rows$tax_year == year, ]
    if (!setequal(r$parcel_id, targets$parcel_id) || nrow(r) != nrow(targets)) {
      stop("Ownership snapshot does not enumerate every target parcel in ", year)
    }
    if (any(r$source_county != targets$source_county[match(r$parcel_id, targets$parcel_id)])) {
      stop("Ownership source county disagrees with EWS parcel county.")
    }
  }
  bools <- c("is_owner_occupied", "has_financialized_owner", "is_corporate_owned")
  allowed_status <- c("matched_classified", "matched_ambiguous", "matched_evidence_insufficient",
    "matched_owner_missing", "matched_owner_partial_missing", "matched_owner_suppressed",
    "source_parcel_not_found", "source_snapshot_unavailable")
  if (anyNA(rows$classification_status) || any(!rows$classification_status %in% allowed_status)) {
    stop("Ownership classification status is missing or outside the source contract.")
  }
  for (field in bools) rows[[field]] <- ownership_bool(rows[[field]])
  missing <- rows$classification_status %in% c("source_parcel_not_found", "source_snapshot_unavailable")
  if (any(!is.na(rows$is_corporate_owned[missing])) ||
      any(!is.na(rows$has_financialized_owner[missing]))) {
    stop("Missing source parcel was assigned an ownership classification.")
  }
  complete <- rows$classification_status == "matched_classified"
  if (any(!stats::complete.cases(rows[complete, bools]))) {
    stop("Complete classification has missing flags.")
  }
  if (any(rows$is_corporate_owned %in% TRUE &
          (rows$is_owner_occupied %in% TRUE | rows$has_financialized_owner %in% FALSE))) {
    stop("Corporate flag contradicts owner occupancy or financialized status.")
  }
  rows
}

ownership_safe_share <- function(numerator, denominator) {
  ifelse(denominator > 0, numerator / denominator, NA_real_)
}

# Full-support results retain strict estimates, observed lower/upper bounds,
# and coverage. Unknown ownership is never added to the noncorporate count.
ownership_summarise <- function(p, groups) {
  p |>
    dplyr::mutate(.ownership_units = residential_units) |>
    dplyr::group_by(dplyr::across(dplyr::all_of(groups))) |>
    dplyr::summarise(
      residential_parcels = dplyr::n(),
      residential_units = sum(.ownership_units),
      matched_parcels = sum(!classification_status %in% c("source_parcel_not_found", "source_snapshot_unavailable")),
      matched_units = sum(.ownership_units[!classification_status %in% c("source_parcel_not_found", "source_snapshot_unavailable")]),
      complete_parcels = sum(classification_status == "matched_classified"),
      complete_units = sum(.ownership_units[classification_status == "matched_classified"]),
      corporate_unknown_parcels = sum(is.na(is_corporate_owned)),
      corporate_unknown_units = sum(.ownership_units[is.na(is_corporate_owned)]),
      corporate_owned_parcels_observed = sum(is_corporate_owned %in% TRUE),
      corporate_owned_units_observed = sum(.ownership_units[is_corporate_owned %in% TRUE]),
      financialized_unknown_parcels = sum(is.na(has_financialized_owner)),
      financialized_owner_parcels_observed = sum(has_financialized_owner %in% TRUE),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      parcel_match_rate = ownership_safe_share(matched_parcels, residential_parcels),
      unit_match_rate = ownership_safe_share(matched_units, residential_units),
      parcel_complete_rate = ownership_safe_share(complete_parcels, residential_parcels),
      unit_complete_rate = ownership_safe_share(complete_units, residential_units),
      pct_corporate_units_lower = 100 * ownership_safe_share(corporate_owned_units_observed, residential_units),
      pct_corporate_units_upper = 100 * ownership_safe_share(corporate_owned_units_observed + corporate_unknown_units, residential_units),
      pct_corporate_units = dplyr::if_else(corporate_unknown_units == 0, pct_corporate_units_lower, NA_real_),
      corporate_owned_units = dplyr::if_else(corporate_unknown_units == 0, corporate_owned_units_observed, NA_real_),
      corporate_owned_parcels = dplyr::if_else(corporate_unknown_parcels == 0, as.numeric(corporate_owned_parcels_observed), NA_real_),
      pct_financialized_owner_parcels = dplyr::if_else(financialized_unknown_parcels == 0,
        100 * ownership_safe_share(financialized_owner_parcels_observed, residential_parcels), NA_real_)
    )
}

ownership_hex_outputs <- function(panel, grid, minimum_coverage = .95, minimum_units = 20) {
  grid <- sf::st_drop_geometry(grid)[c("hex_id", "area_km2")]
  mapped <- panel[!is.na(panel$hex_id), ]
  years <- sort(unique(panel$tax_year))
  stopifnot(length(years) == 2L)
  skeleton <- tidyr::crossing(hex_id = grid$hex_id, tax_year = years) |>
    dplyr::left_join(grid, by = "hex_id")
  full <- skeleton |>
    dplyr::left_join(ownership_summarise(mapped, c("hex_id", "tax_year")), by = c("hex_id", "tax_year")) |>
    dplyr::mutate(
      has_residential_support = !is.na(residential_parcels),
      corporate_owned_units_per_km2 = corporate_owned_units / area_km2,
      corporate_unit_density_lower = corporate_owned_units_observed / area_km2,
      corporate_unit_density_upper = (corporate_owned_units_observed + corporate_unknown_units) / area_km2,
      full_ownership_complete = has_residential_support & corporate_unknown_parcels == 0 & financialized_unknown_parcels == 0,
      residential_parcels = dplyr::coalesce(residential_parcels, 0L),
      residential_units = dplyr::coalesce(residential_units, 0)
    )
  pair_support <- mapped |>
    dplyr::group_by(parcel_id) |>
    dplyr::summarise(common_owner_support = dplyr::n() == 2L &
      all(!is.na(is_corporate_owned) & !is.na(has_financialized_owner)), .groups = "drop")
  common_panel <- mapped |>
    dplyr::inner_join(pair_support[pair_support$common_owner_support, ], by = "parcel_id")
  common <- skeleton |>
    dplyr::left_join(ownership_summarise(common_panel, c("hex_id", "tax_year")), by = c("hex_id", "tax_year")) |>
    dplyr::rename(common_parcels = residential_parcels, common_units = residential_units) |>
    dplyr::left_join(full[c("hex_id", "tax_year", "residential_parcels", "residential_units")], by = c("hex_id", "tax_year")) |>
    dplyr::mutate(
      common_parcels = dplyr::coalesce(common_parcels, 0L),
      common_units = dplyr::coalesce(common_units, 0),
      common_parcel_coverage = ownership_safe_share(common_parcels, residential_parcels),
      common_unit_coverage = ownership_safe_share(common_units, residential_units),
      corporate_owned_units_per_km2 = corporate_owned_units / area_km2,
      comparison_ready = common_parcels > 0 & common_units >= minimum_units &
        dplyr::coalesce(common_parcel_coverage >= minimum_coverage, FALSE) &
        dplyr::coalesce(common_unit_coverage >= minimum_coverage, FALSE)
    )
  features <- c("pct_corporate_units", "corporate_owned_units_per_km2", "pct_financialized_owner_parcels")
  change <- common |>
    dplyr::select(hex_id, tax_year, dplyr::all_of(features), comparison_ready,
                  common_parcels, common_units, common_parcel_coverage, common_unit_coverage) |>
    tidyr::pivot_wider(names_from = tax_year,
      values_from = -c(hex_id, tax_year), names_glue = "{.value}_{tax_year}")
  for (field in features) change[[paste0("delta_", field)]] <-
    change[[paste0(field, "_", years[2])]] - change[[paste0(field, "_", years[1])]]
  change$comparison_ready <- change[[paste0("comparison_ready_", years[1])]] &
    change[[paste0("comparison_ready_", years[2])]]
  list(full = full, common = common, change = change, pair_support = pair_support)
}
