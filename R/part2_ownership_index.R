# Assemble the selected ownership index from reviewed, already-built summaries.
# No classification, parcel processing, imputation, or source reconciliation here.
part2_ownership_components <- function() {
  c("pct_corporate_units", "corporate_owned_units_per_km2", "pct_financialized_owner_parcels")
}

# Read JSON with simplifyVector = FALSE. Normalize duplicate relative/absolute
# references, but reject competing hashes for the same underlying input.
part2_ownership_manifest_entries <- function(manifest) {
  entries <- list()
  visit <- function(x) {
    if (!is.list(x)) return(invisible(NULL))
    if (is.character(x$path) && length(x$path) == 1L &&
        is.character(x$sha256) && length(x$sha256) == 1L) {
      entries[[length(entries) + 1L]] <<- data.frame(path = x$path, sha256 = x$sha256)
    }
    invisible(lapply(x, visit))
  }
  visit(manifest)
  if (!length(entries)) stop("Ownership manifest contains no pinned files.")
  entries <- do.call(rbind, entries)
  if (any(!file.exists(entries$path))) stop("Pinned ownership file is missing.")
  entries$path <- normalizePath(entries$path, winslash = "/", mustWork = TRUE)
  entries <- unique(entries)
  if (anyDuplicated(entries$path)) stop("Ownership manifest has conflicting checksums for one path.")
  entries[order(entries$path), , drop = FALSE]
}

part2_ownership_verify_hashes <- function(entries) {
  if (!all(c("path", "sha256") %in% names(entries)) || any(!file.exists(entries$path))) {
    stop("Missing pinned ownership files or checksum schema.")
  }
  actual <- vapply(entries$path, digest::digest, character(1), file = TRUE, algo = "sha256")
  if (any(actual != entries$sha256)) {
    stop("Pinned ownership checksum mismatch: ", paste(entries$path[actual != entries$sha256], collapse = ", "))
  }
  data.frame(path = entries$path, sha256 = entries$sha256, verified = TRUE)
}

part2_ownership_assert_numeric <- function(actual, expected, label) {
  same_missing <- identical(is.na(actual), is.na(expected))
  observed <- !is.na(expected)
  if (!is.numeric(actual) || !same_missing || any(!is.finite(actual[!is.na(actual)])) ||
      any(abs(actual[observed] - expected[observed]) > 1e-8 * pmax(1, abs(expected[observed])))) {
    stop("Ownership component/denominator contract differs: ", label, call. = FALSE)
  }
  invisible(TRUE)
}

part2_ownership_panel_order <- function(panel, grid, years, label) {
  if (!is.data.frame(panel) || !all(c("hex_id", "tax_year") %in% names(panel)) ||
      anyNA(panel$hex_id) || anyNA(panel$tax_year) || anyDuplicated(panel[c("hex_id", "tax_year")])) {
    stop("Invalid or duplicate ", label, " hex-year keys.")
  }
  expected <- unlist(lapply(years, function(year) paste(grid$hex_id, year, sep = "|")), use.names = FALSE)
  keys <- paste(panel$hex_id, panel$tax_year, sep = "|")
  if (!setequal(keys, expected) || length(keys) != length(expected)) {
    stop(label, " must enumerate the full canonical grid at both tax years.")
  }
  panel <- as.data.frame(panel[match(expected, keys), , drop = FALSE])
  panel$hex_id <- rep(grid$hex_id, length(years))
  rownames(panel) <- NULL
  panel
}

part2_prepare_ownership_index <- function(common, full, grid, variants,
    tax_years = c(2024L, 2025L), cutoffs = as.Date(c("2025-04-01", "2026-04-01")),
    minimum_units = 20, minimum_coverage = .95) {
  if (inherits(grid, "sf")) grid <- sf::st_drop_geometry(grid)
  if (!is.data.frame(grid) || !all(c("hex_id", "area_km2") %in% names(grid)) ||
      anyNA(grid$hex_id) || anyDuplicated(grid$hex_id) ||
      any(!is.finite(grid$area_km2) | grid$area_km2 <= 0)) stop("Invalid canonical ownership grid.")
  cutoffs <- as.Date(cutoffs)
  if (!identical(as.integer(tax_years), c(2024L, 2025L)) ||
      !identical(cutoffs, as.Date(c("2025-04-01", "2026-04-01")))) {
    stop("Ownership tax-year/cutoff mapping differs from the reviewed pair.")
  }
  components <- part2_ownership_components()
  required <- c(components, "area_km2", "common_parcels", "common_units", "residential_parcels",
    "residential_units", "common_parcel_coverage", "common_unit_coverage", "comparison_ready",
    "corporate_owned_units", "corporate_unknown_units", "corporate_unknown_parcels",
    "financialized_owner_parcels_observed", "financialized_unknown_parcels")
  if (!all(required %in% names(common)) || !all(c("area_km2", "residential_parcels", "residential_units",
    "has_residential_support", "full_ownership_complete", "corporate_unknown_units",
    "corporate_unknown_parcels", "financialized_unknown_parcels") %in% names(full))) {
    stop("Incomplete ownership support/component schema.")
  }
  common <- part2_ownership_panel_order(common, grid, tax_years, "Common-support ownership")
  full <- part2_ownership_panel_order(full, grid, tax_years, "Full-support ownership")
  area <- rep(as.numeric(grid$area_km2), length(tax_years))
  part2_ownership_assert_numeric(common$area_km2, area, "canonical common area")
  part2_ownership_assert_numeric(full$area_km2, area, "canonical full area")
  common$area_km2 <- area
  for (field in c("residential_parcels", "residential_units")) {
    part2_ownership_assert_numeric(common[[field]], full[[field]], paste("full support", field))
  }
  fixed_fields <- c("common_parcels", "common_units", "residential_parcels", "residential_units",
    "common_parcel_coverage", "common_unit_coverage")
  earlier <- seq_len(nrow(grid)); later <- earlier + nrow(grid)
  for (field in fixed_fields) {
    part2_ownership_assert_numeric(common[[field]][earlier], common[[field]][later], paste("fixed", field))
  }
  if (any(!is.finite(common$common_parcels) | common$common_parcels < 0) ||
      any(!is.finite(common$common_units) | common$common_units < 0) ||
      any(common$common_parcels > common$residential_parcels) ||
      any(common$common_units > common$residential_units + 1e-8)) stop("Invalid common ownership support.")
  share <- function(x, d) ifelse(d > 0, x / d, NA_real_)
  part2_ownership_assert_numeric(common$common_parcel_coverage,
    share(common$common_parcels, common$residential_parcels), "common parcel coverage proportion")
  part2_ownership_assert_numeric(common$common_unit_coverage,
    share(common$common_units, common$residential_units), "common unit coverage proportion")
  expected_corporate <- ifelse(common$corporate_unknown_units == 0,
    100 * share(common$corporate_owned_units, common$common_units), NA_real_)
  expected_density <- ifelse(common$corporate_unknown_units == 0,
    common$corporate_owned_units / common$area_km2, NA_real_)
  expected_financialized <- ifelse(common$financialized_unknown_parcels == 0,
    100 * share(common$financialized_owner_parcels_observed, common$common_parcels), NA_real_)
  for (component in components) part2_ownership_assert_numeric(common[[component]],
    switch(component, pct_corporate_units = expected_corporate,
      corporate_owned_units_per_km2 = expected_density,
      pct_financialized_owner_parcels = expected_financialized), component)
  if (any(common$pct_corporate_units < 0 | common$pct_corporate_units > 100 + 1e-8, na.rm = TRUE) ||
      any(common$pct_financialized_owner_parcels < 0 | common$pct_financialized_owner_parcels > 100 + 1e-8, na.rm = TRUE) ||
      any(common$corporate_owned_units_per_km2 < 0, na.rm = TRUE)) stop("Ownership component scale is invalid.")
  screen <- common$common_parcels > 0 & common$common_units >= minimum_units &
    common$common_parcel_coverage >= minimum_coverage & common$common_unit_coverage >= minimum_coverage
  screen[is.na(screen)] <- FALSE
  if (!is.logical(common$comparison_ready) || anyNA(common$comparison_ready) ||
      any(common$comparison_ready != screen)) stop("Original ownership common-support screen differs.")
  common$ownership_component_evidence_complete <- rowSums(is.finite(as.matrix(common[components]))) == 3L &
    common$corporate_unknown_parcels %in% 0 & common$corporate_unknown_units %in% 0 &
    common$financialized_unknown_parcels %in% 0
  common$ownership_comparison_ready <- screen & common$ownership_component_evidence_complete
  common$ownership_has_residential_support <- full$has_residential_support
  common$ownership_coverage_status <- ifelse(!full$has_residential_support, "no_residential_support",
    ifelse(common$common_parcels == 0, "no_jointly_known_parcels",
      ifelse(common$common_units < minimum_units, "common_units_below_20",
        ifelse(!screen, "common_coverage_below_95pct",
          ifelse(!common$ownership_component_evidence_complete, "required_ownership_evidence_unknown",
            "provisional_common_support_ready")))))
  for (component in components) {
    common[[paste0(component, "_common_support_unmasked")]] <- common[[component]]
    common[[component]][!common$ownership_comparison_ready] <- NA_real_
  }
  # The common cohort has known flags by construction; retain the full-support
  # unknowns and bounds too, so an apparently complete cohort never hides gaps.
  full_fields <- setdiff(names(full), c("hex_id", "tax_year", "area_km2"))
  for (field in full_fields) common[[paste0("full_", field)]] <- full[[field]]
  expected_variants <- c("certified_only", "source_agreement", "prior_2024_certified_only")
  if (!is.list(variants) || !setequal(names(variants), expected_variants)) {
    stop("Missing ownership source sensitivity variants.")
  }
  for (variant in expected_variants) {
    v <- variants[[variant]]
    if (!is.data.frame(v) || !all(c("hex_id", paste0("comparison_ready_", tax_years),
      unlist(lapply(components, paste0, "_", tax_years))) %in% names(v)) ||
        anyNA(v$hex_id) || anyDuplicated(v$hex_id) || !setequal(v$hex_id, grid$hex_id)) {
      stop("Invalid ownership sensitivity schema: ", variant)
    }
    v <- v[match(grid$hex_id, v$hex_id), , drop = FALSE]
    ready <- unlist(v[paste0("comparison_ready_", tax_years)], use.names = FALSE)
    if (!is.logical(ready) || anyNA(ready)) stop("Unknown sensitivity readiness: ", variant)
    prefix <- paste0("ownership_", variant, "_")
    common[[paste0(prefix, "comparison_ready")]] <- ready
    common[[paste0(prefix, "readiness_differs_from_main")]] <- ready != common$ownership_comparison_ready
    differs <- rep(FALSE, nrow(common))
    for (component in components) {
      x <- unlist(v[paste0(component, "_", tax_years)], use.names = FALSE)
      main <- common[[paste0(component, "_common_support_unmasked")]]
      differs <- differs | xor(is.na(x), is.na(main)) |
        (!is.na(x) & !is.na(main) & abs(x - main) > 1e-8 * pmax(1, abs(main)))
    }
    common[[paste0(prefix, "components_differ_from_main")]] <- differs
  }
  common$analysis_as_of_date <- rep(cutoffs, each = nrow(grid))
  common$ownership_source_variant <- "main"
  common$ownership_source_basis <- "county_appraisal_vintages_with_same_year_verified_williamson_gis"
  common$ownership_retrospective_reconstruction <- TRUE
  common$ownership_support_basis <- "fixed_promoted_parcels_coordinates_and_validated_units_jointly_known_both_years"
  common$ownership_area_basis <- "canonical_full_hex_area_not_common_parcel_area"
  common$ownership_common_support_provisional <- TRUE
  common$ownership_minimum_common_units <- minimum_units
  common$ownership_minimum_common_coverage <- minimum_coverage
  common
}

part2_ownership_changes <- function(earlier, later) {
  if (!identical(earlier$hex_id, later$hex_id) || anyDuplicated(earlier$hex_id)) {
    stop("Ownership change inputs must share the ordered canonical grid.")
  }
  measures <- c(part2_ownership_components(), "ownership_pressure_index")
  fields <- c("ownership_comparison_ready", measures)
  output <- data.frame(hex_id = earlier$hex_id)
  for (field in fields) {
    output[[paste0(field, "_2025")]] <- earlier[[field]]
    output[[paste0(field, "_2026")]] <- later[[field]]
  }
  output$ownership_change_available <- earlier$ownership_comparison_ready & later$ownership_comparison_ready
  for (measure in measures) output[[paste0("delta_", measure)]] <- ifelse(output$ownership_change_available,
    later[[measure]] - earlier[[measure]], NA_real_)
  output$ownership_change_direction <- ifelse(!output$ownership_change_available, "unavailable",
    ifelse(output$delta_ownership_pressure_index > 1e-9, "increase",
      ifelse(output$delta_ownership_pressure_index < -1e-9, "decrease", "unchanged")))
  output
}
