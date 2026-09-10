source("R/part2_feature_matrix.R")
fails <- function(expr) stopifnot(inherits(tryCatch({force(expr); NULL}, error = identity), "error"))
cutoffs <- as.Date(c("2025-04-01", "2026-04-01"))
support <- data.frame(hex_id = 1:6, area_km2 = rep(.1, 6), residential_units = c(100, 10, 100, 100, 100, 100),
  source_county = c("Travis", "Travis", "Williamson", "Travis", "Travis", "Hays"))
domains <- lapply(part2_matrix_contract(), function(spec) {
  x <- data.frame(hex_id = rep(1:6, 2))
  x[[spec$date]] <- rep(cutoffs, each = 6)
  x[[spec$reference]] <- cutoffs[1]
  for (index in names(spec$scores)) for (column in c(index, spec$scores[[index]])) x[[column]] <- 20
  x
})
domains$sr311$sr_311_in_current_city_scope <- rep(c(rep(TRUE, 5), FALSE), 2)
domains$demolitions$hex_center_inside_current_austin_full <- domains$sr311$sr_311_in_current_city_scope
domains$evictions$eviction_inside_current_city <- domains$sr311$sr_311_in_current_city_scope
domains$sr311$sr_311_boundary_straddling_hex <- FALSE
domains$sr311$sr_311_poc_coverage_usable <- TRUE
domains$demolitions$demolition_comparison_ready <- TRUE
domains$evictions$eviction_count_observed <- TRUE
domains$ownership$ownership_comparison_ready <- TRUE
domains$ownership$ownership_comparison_ready[9] <- FALSE
domains$amenities$amenity_retrospective_usable <- TRUE
domains$amenities$amenity_window_complete <- NA # not a completeness gate
rent <- part2_matrix_contract()$acs$scores$rent_pressure_citywide_index
domains$acs[4, c("rent_pressure_citywide_index", rent)] <- NA_real_
# Partial components must make the entire index unavailable at that date.
# A changed missingness pattern can be reported but cannot enter the sample.
domains$acs[5, rent[1]] <- NA_real_
domains$acs[11, rent[2]] <- NA_real_
domains$acs[c(5, 11), "rent_pressure_citywide_index"] <- NA_real_
result <- part2_assemble_feature_matrix(domains, support)
stopifnot(identical(result$matrices[[1]]$hex_id, 1L),
  identical(result$matrices[[1]]$hex_id, result$matrices[[2]]$hex_id),
  identical(result$eligibility$primary_exclusion,
    c("included", "insufficient_fixed_units", "ownership_support_unusable", "missing_required_index", "missing_required_index", "outside_fixed_city")),
  !result$eligibility$rent_pressure_citywide_index_same_term_availability[5],
  result$paired$rent_pressure_citywide_index_terms_available[5] == 2,
  nrow(result$paired) == 12, sum(result$exclusion_summary$primary_hexes) == 6,
  !any(result$paired$cluster_standardization_fitted))
stopifnot(all(result$paired$all_required_components_available[result$paired$common_comparison_ready]),
          all(result$paired$measurement_version == "part2-fixed-components-v2"))
bad <- domains; bad$acs$rent_pressure_citywide_index[5] <- 20
fails(part2_assemble_feature_matrix(bad, support))
# Source order/ID storage may differ; canonical integer ID order is restored.
shuffled <- lapply(domains, function(x) {x$hex_id <- as.character(x$hex_id); x[12:1, ]})
stopifnot(identical(result, part2_assemble_feature_matrix(shuffled, support)))
bad <- domains; bad$acs <- bad$acs[-1, ]; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$acs[1, ] <- bad$acs[2, ]; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$acs$analysis_as_of_date[1] <- as.Date("2024-04-01"); fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$ownership$ownership_comparison_ready <- NULL; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$evictions$eviction_count_observed[1] <- NA; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$demolitions$hex_center_inside_current_austin_full[1] <- FALSE; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$acs$acs_scaling_reference_as_of_date <- cutoffs[2]; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$acs$rent_pressure_citywide_index[1] <- Inf; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$acs$rent_pressure_citywide_index[1] <- 30; fails(part2_assemble_feature_matrix(bad, support))
bad <- domains; bad$sr311$residential_units <- rep(999, 12); fails(part2_assemble_feature_matrix(bad, support))
cat("Part 2 matrix synthetic checks passed: paired keys, support, frozen scoring, coverage, missingness, common sample and availability signatures.\n")

# Manifest fixtures verify immediate and nested pins, failed provenance and
# feature registration. Only this newly created temporary test directory is removed.
fixture <- tempfile("part2-matrix-manifest-")
dir.create(fixture)
input_path <- file.path(fixture, "input.rds")
output_path <- file.path(fixture, "features.rds")
nested_path <- file.path(fixture, "upstream.rds")
manifest_path <- file.path(fixture, "manifest.json")
saveRDS(1L, input_path); saveRDS(2L, output_path); saveRDS(3L, nested_path)
entry <- function(path) list(path = path, sha256 = digest::digest(file = path, algo = "sha256"))
manifest <- list(status = "complete", inputs = list(entry(input_path)), outputs = list(entry(output_path)),
  pinned_upstream_files = list(entry(nested_path)))
write_manifest <- function(x) jsonlite::write_json(x, manifest_path, auto_unbox = TRUE)
write_manifest(manifest)
stopifnot(nrow(part2_matrix_verify_source(manifest_path, output_path, "complete")) == 3L)
fails(part2_matrix_verify_source(manifest_path, output_path, "not_complete"))
fails(part2_matrix_verify_source(manifest_path, input_path, "complete"))
saveRDS(9L, nested_path)
fails(part2_matrix_verify_source(manifest_path, output_path, "complete"))
saveRDS(3L, nested_path)
saveRDS(9L, input_path)
fails(part2_matrix_verify_source(manifest_path, output_path, "complete"))
saveRDS(1L, input_path)
bad_manifest <- manifest; bad_manifest$inputs[[1]]$sha256 <- NULL; write_manifest(bad_manifest)
fails(part2_matrix_verify_source(manifest_path, output_path, "complete"))
bad_manifest <- manifest; bad_manifest$pinned_upstream_files[[2]] <- list(path = input_path, sha256 = "conflict")
write_manifest(bad_manifest)
fails(part2_matrix_verify_source(manifest_path, output_path, "complete"))
unlink(fixture, recursive = TRUE)
cat("Part 2 matrix manifest fixture checks passed, including nested upstream pins.\n")
