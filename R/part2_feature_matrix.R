# Join already-scored Part 2 inputs without refitting any scoring or clusters.
part2_matrix_contract <- function() {
  list(
    acs = list(date = "analysis_as_of_date", reference = "acs_scaling_reference_as_of_date",
      scores = list(rent_pressure_citywide_index = paste0("acs_score_", c("rent_level", "rent_growth", "rent_acceleration")),
        demographic_vulnerability_index = paste0("acs_score_", c("low_income", "renters", "poverty", "rent_burden", "low_college")))),
    amenities = list(date = "amenity_analysis_as_of_date", reference = "amenity_scaling_reference_as_of_date",
      scores = list(amenity_change_index = paste0("amenity_", c("cafe", "full_service_restaurant", "drinking_place"), "_score"))),
    sr311 = list(date = "analysis_as_of_date", reference = "sr_311_pressure_index_scaling_reference_as_of_date",
      scores = list(sr_311_pressure_index = paste0(c("sr_311_smoke_signal_latest_12mo_per_100_units",
        "sr_311_smoke_signal_latest_12mo_density", "sr_311_smoke_signal_latest_12mo_rate_change_per_100_units"), "_score"))),
    demolitions = list(date = "analysis_as_of_date", reference = "demolition_pressure_index_scaling_reference_as_of_date",
      scores = list(demolition_pressure_index = paste0(c("demo_recent_density", "demo_trend_positive", "demo_total_recent_density"), "_score"))),
    evictions = list(date = "analysis_as_of_date", reference = "eviction_pressure_index_scaling_reference_as_of_date",
      scores = list(eviction_pressure_index = paste0(c("eviction_latest_12mo_per_100_units",
        "eviction_latest_12mo_rate_change_per_100_units"), "_score"))),
    ownership = list(date = "analysis_as_of_date", reference = "ownership_pressure_index_scaling_reference_as_of_date",
      scores = list(ownership_pressure_index = paste0(c("pct_corporate_units", "corporate_owned_units_per_km2",
        "pct_financialized_owner_parcels"), "_score"))))
}

part2_matrix_indices <- function() {
  c("rent_pressure_citywide_index", "demographic_vulnerability_index", "demolition_pressure_index",
    "eviction_pressure_index", "sr_311_pressure_index", "ownership_pressure_index", "amenity_change_index")
}

part2_matrix_order <- function(x, ids, cutoffs, date_column, label) {
  if (!is.data.frame(x) || !all(c("hex_id", date_column) %in% names(x)) ||
      anyNA(x$hex_id) || anyNA(x[[date_column]]) || !inherits(x[[date_column]], "Date")) {
    stop("Invalid hex/date schema: ", label, call. = FALSE)
  }
  keys <- paste(x$hex_id, x[[date_column]], sep = "|")
  expected <- unlist(lapply(as.character(cutoffs), function(date) paste(ids, date, sep = "|")), use.names = FALSE)
  if (anyDuplicated(keys) || length(keys) != length(expected) || !setequal(keys, expected)) {
    stop("Incomplete, extra, or duplicated hex/date keys: ", label, call. = FALSE)
  }
  x <- as.data.frame(x[match(expected, keys), , drop = FALSE])
  x$hex_id <- rep(ids, length(cutoffs))
  rownames(x) <- NULL
  x
}

part2_matrix_flag <- function(x, name) {
  if (!name %in% names(x) || !is.logical(x[[name]]) || anyNA(x[[name]])) {
    stop("Missing or unknown required coverage flag: ", name, call. = FALSE)
  }
  x[[name]]
}

part2_assemble_feature_matrix <- function(domains, support,
    cutoffs = as.Date(c("2025-04-01", "2026-04-01")), minimum_units = 20) {
  specification <- part2_matrix_contract()
  indices <- part2_matrix_indices()
  if (!identical(cutoffs, as.Date(c("2025-04-01", "2026-04-01"))) ||
      !identical(sort(names(domains)), sort(names(specification)))) stop("Wrong paired-matrix source contract.")
  required_support <- c("hex_id", "area_km2", "residential_units", "source_county")
  if (!is.data.frame(support) || !nrow(support) || !all(required_support %in% names(support)) ||
      !is.integer(support$hex_id) || anyNA(support$hex_id) || anyDuplicated(support$hex_id) ||
      any(!is.finite(support$area_km2) | support$area_km2 <= 0) ||
      !is.numeric(support$residential_units) || any(!is.finite(support$residential_units) & !is.na(support$residential_units)) ||
      any(support$residential_units < 0, na.rm = TRUE) || anyNA(support$source_county) ||
      length(minimum_units) != 1L || !is.finite(minimum_units) || minimum_units <= 0) stop("Invalid fixed matrix support.")
  ids <- support$hex_id
  n <- length(ids)
  paired <- support[rep(seq_len(n), 2L), required_support, drop = FALSE]
  paired$analysis_as_of_date <- rep(cutoffs, each = n)
  rownames(paired) <- NULL
  for (domain in names(specification)) {
    spec <- specification[[domain]]
    x <- part2_matrix_order(domains[[domain]], ids, cutoffs, spec$date, domain)
    if (!spec$reference %in% names(x) || anyNA(x[[spec$reference]]) ||
        any(as.Date(x[[spec$reference]]) != cutoffs[1])) stop("Earlier scoring reference not frozen: ", domain)
    for (field in intersect(c("area_km2", "residential_units"), names(x))) {
      if (!isTRUE(all.equal(as.numeric(x[[field]]), as.numeric(paired[[field]]), tolerance = 1e-8))) {
        stop("Fixed support differs: ", domain, "/", field)
      }
    }
    for (index in names(spec$scores)) {
      columns <- spec$scores[[index]]
      if (!all(c(index, columns) %in% names(x))) stop("Missing scored index/terms: ", index)
      for (column in c(index, columns)) {
        values <- x[[column]]
        if (!is.numeric(values) || any(!is.finite(values) & !is.na(values)) ||
            any(values < -1e-8 | values > 100 + 1e-8, na.rm = TRUE)) stop("Invalid 0-100 score: ", column)
      }
      terms <- as.matrix(x[columns])
      count <- rowSums(is.finite(terms))
      # No row/date-specific reweighting: missing any required term makes the
      # entire index unavailable. Every included observation has one recipe.
      expected_index <- rowMeans(terms)
      if (!isTRUE(all.equal(x[[index]], expected_index, check.attributes = FALSE, tolerance = 1e-8))) {
        stop("Composite disagrees with its scored terms: ", index)
      }
      paired[[index]] <- x[[index]]
      paired[[paste0(index, "_terms_available")]] <- count
      paired[[paste0(index, "_terms_total")]] <- ncol(terms)
      paired[[paste0(index, "_terms_complete")]] <- count == ncol(terms)
      paired[[paste0(index, "_availability_signature")]] <- apply(is.finite(terms), 1L,
        function(row) paste(as.integer(row), collapse = ""))
    }
    domains[[domain]] <- x
  }
  paired$in_current_city_scope <- part2_matrix_flag(domains$sr311, "sr_311_in_current_city_scope")
  for (check in list(list(domains$demolitions, "hex_center_inside_current_austin_full"),
                    list(domains$evictions, "eviction_inside_current_city"))) {
    if (!identical(paired$in_current_city_scope, part2_matrix_flag(check[[1]], check[[2]]))) {
      stop("Event streams disagree on the fixed city-center mask.")
    }
  }
  paired$boundary_straddling_hex <- part2_matrix_flag(domains$sr311, "sr_311_boundary_straddling_hex")
  if (!identical(paired$in_current_city_scope[seq_len(n)], paired$in_current_city_scope[n + seq_len(n)]) ||
      !identical(paired$boundary_straddling_hex[seq_len(n)], paired$boundary_straddling_hex[n + seq_len(n)])) {
    stop("City mask changes between snapshots.")
  }
  paired$minimum_unit_support <- is.finite(paired$residential_units) & paired$residential_units >= minimum_units
  paired$ownership_comparison_ready <- part2_matrix_flag(domains$ownership, "ownership_comparison_ready")
  paired$sr_311_poc_coverage_usable <- part2_matrix_flag(domains$sr311, "sr_311_poc_coverage_usable")
  paired$demolition_comparison_ready <- part2_matrix_flag(domains$demolitions, "demolition_comparison_ready")
  paired$eviction_poc_coverage_usable <- part2_matrix_flag(domains$evictions, "eviction_count_observed")
  paired$amenity_retrospective_usable <- part2_matrix_flag(domains$amenities, "amenity_retrospective_usable")
  paired$all_seven_indices_available <- rowSums(is.finite(as.matrix(paired[indices]))) == length(indices)
  paired$all_required_components_available <- rowSums(as.matrix(
    paired[paste0(indices, "_terms_complete")])) == length(indices)
  stopifnot(identical(paired$all_seven_indices_available, paired$all_required_components_available))
  gates <- c("in_current_city_scope", "minimum_unit_support", "ownership_comparison_ready",
    "sr_311_poc_coverage_usable", "demolition_comparison_ready", "eviction_poc_coverage_usable",
    "amenity_retrospective_usable", "all_seven_indices_available")
  paired$eligible_this_snapshot <- rowSums(as.matrix(paired[gates])) == length(gates)
  earlier <- paired[seq_len(n), , drop = FALSE]
  later <- paired[n + seq_len(n), , drop = FALSE]
  eligibility <- support[required_support]
  eligibility$in_current_city_scope <- earlier$in_current_city_scope
  eligibility$boundary_straddling_hex <- earlier$boundary_straddling_hex
  reasons <- c("outside_fixed_city", "insufficient_fixed_units", "ownership_support_unusable",
    "sr311_coverage_unusable", "demolition_coverage_unusable", "eviction_coverage_unusable",
    "amenity_reconstruction_unusable", "missing_required_index")
  fail_columns <- paste0("exclude_", reasons)
  for (i in seq_along(gates)) eligibility[[fail_columns[i]]] <- !earlier[[gates[i]]] | !later[[gates[i]]]
  eligibility$eligible_2025 <- earlier$eligible_this_snapshot
  eligibility$eligible_2026 <- later$eligible_this_snapshot
  eligibility$common_comparison_ready <- eligibility$eligible_2025 & eligibility$eligible_2026
  eligibility$primary_exclusion <- "included"
  for (i in rev(seq_along(reasons))) eligibility$primary_exclusion[eligibility[[fail_columns[i]]]] <- reasons[i]
  for (index in indices) {
    signature <- paste0(index, "_availability_signature")
    eligibility[[paste0(index, "_same_term_availability")]] <- earlier[[signature]] == later[[signature]]
    eligibility[[paste0(index, "_available_both")]] <- is.finite(earlier[[index]]) & is.finite(later[[index]])
  }
  paired$common_comparison_ready <- rep(eligibility$common_comparison_ready, 2L)
  paired$primary_exclusion <- rep(eligibility$primary_exclusion, 2L)
  paired$scaling_reference_as_of_date <- cutoffs[1]
  paired$retrospective_reconstruction <- TRUE
  paired$cluster_standardization_fitted <- FALSE
  paired$measurement_version <- "part2-fixed-components-v2"
  if ("acs_rent_source_geography" %in% names(domains$acs)) {
    geography <- domains$acs$acs_rent_source_geography
    eligible <- eligibility$common_comparison_ready
    if (anyNA(geography[rep(eligible, 2L)]) ||
        any(geography[seq_len(n)][eligible] != geography[n + seq_len(n)][eligible])) {
      stop("Included rent histories change geographic level between snapshots.")
    }
    paired$rent_source_geography <- geography
    eligibility$rent_source_geography <- geography[seq_len(n)]
  }
  exclusions <- do.call(rbind, lapply(seq_along(reasons), function(i) {
    keep <- eligibility[[fail_columns[i]]]
    data.frame(hex_id = ids[keep], reason = rep(reasons[i], sum(keep)), source_county = support$source_county[keep])
  }))
  exclusion_summary <- data.frame(reason = c(reasons, "included"),
    primary_hexes = as.integer(table(factor(eligibility$primary_exclusion, levels = c(reasons, "included")))),
    nonexclusive_hexes = c(vapply(fail_columns, function(column) sum(eligibility[[column]]), integer(1)),
      sum(eligibility$common_comparison_ready)))
  matrices <- lapply(list(earlier, later), function(x) x[eligibility$common_comparison_ready,
    c("hex_id", "analysis_as_of_date", indices), drop = FALSE])
  names(matrices) <- as.character(cutoffs)
  if (!nrow(matrices[[1]])) stop("No common eligible cells; inspect source coverage before clustering.")
  list(paired = paired, eligibility = eligibility, exclusions = exclusions,
    exclusion_summary = exclusion_summary, matrices = matrices)
}

# A source artifact must be listed in a complete, verifiable source manifest.
part2_matrix_verify_source <- function(manifest_path, feature_path, expected_status) {
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  if (!identical(manifest$status, expected_status) || !length(manifest$inputs) || !length(manifest$outputs)) {
    stop("Incomplete source manifest: ", manifest_path)
  }
  records <- c(manifest$inputs, manifest$outputs)
  if (any(!vapply(records, function(x) is.character(x$path) && length(x$path) == 1L &&
      is.character(x$sha256) && length(x$sha256) == 1L, logical(1)))) stop("Unpinned source file.")
  # Include embedded upstream pins, notably ownership's classifier/source
  # inventory, as well as immediate inputs/outputs. Do not follow arbitrary
  # external URLs or infer paths from other prose metadata.
  pinned <- list()
  visit <- function(x) {
    if (!is.list(x)) return(invisible(NULL))
    if (is.character(x$path) && length(x$path) == 1L &&
        is.character(x$sha256) && length(x$sha256) == 1L) {
      pinned[[length(pinned) + 1L]] <<- data.frame(path = x$path, sha256 = x$sha256)
    }
    invisible(lapply(x, visit))
  }
  visit(manifest)
  entries <- do.call(rbind, pinned)
  if (any(!file.exists(entries$path))) stop("Missing pinned source file.")
  entries$path <- normalizePath(entries$path, mustWork = TRUE)
  entries <- unique(entries)
  if (anyDuplicated(entries$path)) stop("Conflicting pinned hashes for a source file.")
  if (!normalizePath(feature_path, mustWork = TRUE) %in% vapply(manifest$outputs,
      function(x) normalizePath(x$path, mustWork = TRUE), character(1))) stop("Feature is not a pinned source output.")
  actual <- vapply(entries$path, function(path) digest::digest(file = path, algo = "sha256"), character(1))
  if (any(actual != entries$sha256)) stop("Source checksum mismatch: ", paste(entries$path[actual != entries$sha256], collapse = ", "))
  entries$source_manifest <- manifest_path
  entries$verified <- TRUE
  entries
}
