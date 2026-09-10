# Read-only integration audit. Run only after the historical analysis is complete:
# Rscript tests/test_part2_cluster_outputs.R
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
# No production cluster helper is sourced: preprocessing, distances, label
# correspondence, ARI, transition counts, and optimizer replay are independent.
audit_permutations <- local({
  values <- seq_len(7L); result <- matrix(0L, nrow = factorial(7L), ncol = 7L)
  for (r in seq_len(nrow(result))) {
    result[r, ] <- values
    if (r == nrow(result)) break
    pivot <- max(which(values[-length(values)] < values[-1L]))
    swap <- max(which(values > values[pivot]))
    temporary <- values[pivot]; values[pivot] <- values[swap]; values[swap] <- temporary
    tail <- seq.int(pivot + 1L, length(values)); values[tail] <- rev(values[tail])
  }
  stopifnot(nrow(unique(result)) == factorial(7L))
  result
})
audit_ari <- function(a, b) {
  stopifnot(length(a) == length(b), !anyNA(a), !anyNA(b))
  n <- length(a)
  if (n < 2L) return(1)
  cross <- table(a, b); combinations <- function(x) choose(x, 2)
  observed <- sum(combinations(cross))
  first <- sum(combinations(rowSums(cross))); second <- sum(combinations(colSums(cross)))
  chance <- first * second / combinations(n); upper <- (first + second) / 2
  if (upper == chance) 1 else (observed - chance) / (upper - chance)
}
audit_alignment <- function(reference, candidate, k = 7L) {
  stopifnot(k == 7L, length(reference) == length(candidate), length(reference) > 0L)
  labels <- paste0("C", seq_len(k))
  overlap <- table(reference = factor(reference, levels = labels), candidate = factor(candidate, levels = labels))
  lookup <- cbind(as.vector(audit_permutations), rep(seq_len(k), each = nrow(audit_permutations)))
  objectives <- rowSums(matrix(as.numeric(overlap[lookup]), nrow = nrow(audit_permutations)))
  best <- which(objectives == max(objectives))
  permutation <- audit_permutations[best[1], ]
  mapping <- data.frame(candidate_cluster = labels, aligned_cluster = labels[permutation])
  list(mapping = mapping, aligned_cluster = mapping$aligned_cluster[match(candidate, mapping$candidate_cluster)],
    overlap_table = overlap, matched_hexes = as.integer(max(objectives)),
    match_share = max(objectives) / length(reference), optimal_permutation_count = length(best),
    tie_break_policy = "lexicographically_smallest_candidate_to_reference_label_permutation")
}
audit_transitions <- function(from, to, k = 7L) {
  stopifnot(length(from) == length(to)); labels <- paste0("C", seq_len(k))
  # Explicit cell indicators avoid relying on production tabulation/order code.
  result <- do.call(rbind, lapply(labels, function(a) do.call(rbind, lapply(labels, function(b) {
    count <- sum(from == a & to == b); from_n <- sum(from == a); to_n <- sum(to == b)
    data.frame(from_cluster = a, to_cluster = b, hexes = count, from_total = from_n, to_total = to_n,
      share_of_from = if (from_n) count / from_n else NA_real_,
      share_of_to = if (to_n) count / to_n else NA_real_,
      share_of_all = if (length(from)) count / length(from) else NA_real_, same_cluster = a == b)
  }))))
  rownames(result) <- NULL
  result
}
audit_kmeans <- function(z, k = 7L, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  fit <- stats::kmeans(z, centers = k, iter.max = 500L, nstart = 100L, algorithm = "Lloyd")
  rownames(fit$centers) <- paste0("C", seq_len(k))
  list(centers = fit$centers, cluster = paste0("C", fit$cluster))
}
root <- "output/part2/clusters"
read_csv_output <- function(name) read_csv(file.path(root, name), show_col_types = FALSE)
expect_equal <- function(actual, expected, tolerance = 1e-10) {
  stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE, tolerance = tolerance)))
}
hash_checks <- 0L
verify_files <- function(entries) {
  stopifnot(all(c("path", "sha256") %in% names(entries)), !anyNA(entries$path),
    !anyNA(entries$sha256), all(file.exists(entries$path)))
  actual <- vapply(entries$path, digest::digest, character(1), file = TRUE, algo = "sha256")
  if (any(actual != entries$sha256)) stop("Changed historical-cluster dependency/output: ",
    paste(entries$path[actual != entries$sha256], collapse = ", "))
  hash_checks <<- hash_checks + nrow(entries)
}
manifest <- jsonlite::read_json(file.path(root, "part2_cluster_run_manifest.json"), simplifyVector = TRUE)
stopifnot(identical(manifest$status, "historical_cluster_comparison_complete_v2"),
  isTRUE(manifest$retrospective_reconstruction), isTRUE(manifest$ml_work_paused),
  isTRUE(manifest$canonical_part1_unchanged), identical(manifest$risk_ranking, FALSE),
  manifest$existing_artifacts_preserved > 0L, manifest$audit_hexes == 7027L, manifest$common_hexes > 7L)
verify_files(bind_rows(manifest$inputs, manifest$outputs))
before <- readRDS(file.path(root, "existing_artifacts_before.rds"))
after <- read_csv_output("existing_artifacts_preservation_after.csv")
stopifnot(nrow(before) == manifest$existing_artifacts_preserved, nrow(after) == nrow(before), !anyDuplicated(before$path),
  setequal(before$path, after$path))
expect_equal(before$sha256, after$sha256[match(before$path, after$path)])
part2_roots <- normalizePath(c("output/part2", "figures/part2"), mustWork = TRUE)
stopifnot(!any(vapply(before$path, function(path) any(startsWith(path, paste0(part2_roots, "/"))), logical(1))),
  all(normalizePath(c("output/hex_grid.rds", "output/hex_features.rds",
    "output/corporate_ownership_by_hex.rds", "output/amenity_cluster_assignments.csv"), mustWork = TRUE) %in% before$path))
source("tests/current_preservation_policy.R")
verify_files(current_preservation_entries(after))
matrix_verified <- read_csv_output("matrix_source_hash_verification.csv")
stopifnot(all(matrix_verified$verified))
verify_files(matrix_verified)
matrix_manifest <- jsonlite::read_json("output/part2/matrix/part2_matrix_run_manifest.json", simplifyVector = TRUE)
stopifnot(matrix_manifest$status == "paired_seven_feature_matrix_complete_v2")
verify_files(bind_rows(matrix_manifest$inputs, matrix_manifest$outputs))
upstream_verified <- read_csv("output/part2/matrix/part2_source_hash_verification.csv", show_col_types = FALSE)
stopifnot(all(upstream_verified$verified), any(grepl("/data/raw_acs/", upstream_verified$path, fixed = TRUE)),
  any(grepl("/data/raw_311/", upstream_verified$path, fixed = TRUE)))
verify_files(upstream_verified)

dates <- as.Date(c("2025-04-01", "2026-04-01")); labels <- paste0("C", 1:7)
features <- c("rent_pressure_citywide_index", "demographic_vulnerability_index", "demolition_pressure_index",
  "eviction_pressure_index", "sr_311_pressure_index", "ownership_pressure_index", "amenity_change_index")
grid <- readRDS("output/hex_grid.rds")
eligibility <- readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
paired <- readRDS("output/part2/matrix/part2_features_paired.rds")
inputs <- lapply(dates, function(date) readRDS(file.path("output/part2/matrix", paste0("part2_analysis_matrix_", date, ".rds"))))
model <- readRDS(file.path(root, "part2_cluster_models.rds"))
all_assignments <- readRDS(file.path(root, "part2_cluster_assignments.rds"))
ids <- inputs[[1]]$hex_id; n <- length(ids)
stopifnot(n > 7L, n == manifest$common_hexes, n == matrix_manifest$common_eligible_hexes,
  is.integer(ids), !anyDuplicated(ids), identical(ids, inputs[[2]]$hex_id),
  identical(model$schema_version, "part2-historical-clusters-v2"),
  identical(model$measurement_version, "part2-fixed-components-v2"),
  identical(model$training_hex_ids, ids), identical(model$features, features), identical(model$dates, dates), model$k == 7L,
  isTRUE(model$baseline_is_distinct_from_part1), identical(model$risk_ranking, FALSE),
  identical(model$later_scaling, "frozen_2025_means_and_sample_standard_deviations"),
  identical(eligibility$hex_id, grid$hex_id), identical(all_assignments$hex_id, grid$hex_id),
  identical(ids, eligibility$hex_id[eligibility$common_comparison_ready]),
  all(vapply(names(eligibility), function(field) identical(all_assignments[[field]], eligibility[[field]]), logical(1))))
stopifnot(all(paired$measurement_version == "part2-fixed-components-v2"),
  all(paired$all_required_components_available[paired$common_comparison_ready]))
# Independently verify the corrected measurement contract from each stream's
# component scores; no mean-available terms or eviction history share can enter.
component_contract <- list(
  rent_pressure_citywide_index = c("acs_score_rent_level", "acs_score_rent_growth", "acs_score_rent_acceleration"),
  demographic_vulnerability_index = paste0("acs_score_", c("low_income", "renters", "poverty", "rent_burden", "low_college")),
  demolition_pressure_index = paste0(c("demo_recent_density", "demo_trend_positive", "demo_total_recent_density"), "_score"),
  eviction_pressure_index = paste0(c("eviction_latest_12mo_per_100_units", "eviction_latest_12mo_rate_change_per_100_units"), "_score"),
  sr_311_pressure_index = paste0(c("sr_311_smoke_signal_latest_12mo_per_100_units", "sr_311_smoke_signal_latest_12mo_density",
    "sr_311_smoke_signal_latest_12mo_rate_change_per_100_units"), "_score"),
  ownership_pressure_index = paste0(c("pct_corporate_units", "corporate_owned_units_per_km2", "pct_financialized_owner_parcels"), "_score"),
  amenity_change_index = paste0("amenity_", c("cafe", "full_service_restaurant", "drinking_place"), "_score"))
component_sources <- c(rent_pressure_citywide_index = "output/part2/acs/acs_features_paired.rds",
  demographic_vulnerability_index = "output/part2/acs/acs_features_paired.rds",
  demolition_pressure_index = "output/part2/demolitions/demolition_features_paired.rds",
  eviction_pressure_index = "output/part2/evictions/eviction_features_paired.rds",
  sr_311_pressure_index = "output/part2/311/311_features_paired.rds",
  ownership_pressure_index = "output/part2/ownership_index/ownership_features_paired.rds",
  amenity_change_index = "output/part2/amenities/amenity_features_paired.rds")
stopifnot(identical(names(component_contract), features))
for (index in features) {
  x <- readRDS(component_sources[[index]])
  date_field <- if (index == "amenity_change_index") "amenity_analysis_as_of_date" else "analysis_as_of_date"
  keys <- paste(x$hex_id, x[[date_field]])
  wanted <- paste(paired$hex_id, paired$analysis_as_of_date)
  stopifnot(!anyDuplicated(keys), setequal(keys, wanted), all(component_contract[[index]] %in% names(x)))
  x <- x[match(wanted, keys), ]
  scores <- as.matrix(x[component_contract[[index]]])
  expected_index <- rowSums(scores) / ncol(scores)
  expect_equal(x[[index]], expected_index)
  expect_equal(paired[[index]], expected_index)
  count <- rowSums(is.finite(scores))
  expect_equal(paired[[paste0(index, "_terms_available")]], count)
  expect_equal(paired[[paste0(index, "_terms_total")]], rep(ncol(scores), nrow(paired)))
  expected_signature <- unname(apply(is.finite(scores), 1L, function(row) paste(as.integer(row), collapse = "")))
  stopifnot(identical(paired[[paste0(index, "_availability_signature")]], expected_signature),
    all(count[paired$common_comparison_ready] == ncol(scores)))
  if (index %in% c("eviction_pressure_index", "sr_311_pressure_index")) {
    recent_column <- if (index == "eviction_pressure_index") "eviction_cases_latest_12mo" else "sr_311_smoke_signal_latest_12mo"
    previous_column <- if (index == "eviction_pressure_index") "eviction_cases_previous_12mo" else "sr_311_smoke_signal_previous_12mo"
    change_column <- if (index == "eviction_pressure_index") "eviction_latest_12mo_rate_change_per_100_units" else
      "sr_311_smoke_signal_latest_12mo_rate_change_per_100_units"
    keep <- paired$common_comparison_ready
    stopifnot(all(paired$residential_units[keep] >= 20))
    expected_change <- 100 * (x[[recent_column]][keep] - x[[previous_column]][keep]) / paired$residential_units[keep]
    expect_equal(x[[change_column]][keep], expected_change)
    zero_change <- keep & is.finite(x[[change_column]]) & x[[change_column]] == 0
    expect_equal(x[[paste0(change_column, "_score")]][zero_change], rep(50, sum(zero_change)))
  }
}
all_complete <- Reduce(`&`, lapply(features, function(index)
  paired[[paste0(index, "_terms_available")]] == length(component_contract[[index]])))
stopifnot(identical(paired$all_required_components_available, all_complete))
# The corrected rent source level is a single pair-wide choice, not a different
# geography selected separately for each snapshot or each growth term.
rent_candidates <- readRDS("output/part2/acs/acs_rent_source_candidates.rds")
rent_selection <- readRDS("output/part2/acs/acs_rent_source_selection.rds")
required_rent_years <- c(2013L, 2014L, 2018L, 2019L, 2023L, 2024L)
stopifnot(nrow(rent_candidates) == nrow(grid) * 12L,
  !anyDuplicated(rent_candidates[c("hex_id", "acs_year", "source_geography")]))
rent_candidates$independent_reliable <- is.finite(rent_candidates$estimate) & rent_candidates$estimate > 0 &
  is.finite(rent_candidates$moe) & rent_candidates$moe >= 0 & rent_candidates$moe / rent_candidates$estimate <= .30
rent_support <- rent_candidates %>% group_by(hex_id, source_geography) %>% summarise(
  supported = n() == 6L && setequal(acs_year, required_rent_years) && all(independent_reliable), .groups = "drop")
bg <- rent_support[rent_support$source_geography == "block_group", ]
tr <- rent_support[rent_support$source_geography == "tract", ]; tr <- tr[match(bg$hex_id, tr$hex_id), ]
chosen_level <- ifelse(bg$supported, "block_group", ifelse(tr$supported, "tract", NA_character_))
stopifnot(identical(rent_selection$selected_geography, chosen_level[match(rent_selection$hex_id, bg$hex_id)]))
acs <- readRDS("output/part2/acs/acs_features_paired.rds")
acs <- acs[match(paste(paired$hex_id, paired$analysis_as_of_date), paste(acs$hex_id, acs$analysis_as_of_date)), ]
expected_level <- chosen_level[match(acs$hex_id, bg$hex_id)]
stopifnot(identical(acs$acs_rent_source_geography, expected_level),
  identical(acs$acs_rent_series_supported, !is.na(expected_level)),
  all(acs$acs_rent_complete[paired$common_comparison_ready]),
  all(acs$acs_vulnerability_complete[paired$common_comparison_ready]))
expect_equal(acs$acs_rent_complete, paired$rent_pressure_citywide_index_terms_available == 3L)
expect_equal(acs$acs_vulnerability_complete, paired$demographic_vulnerability_index_terms_available == 5L)
new_fields <- setdiff(names(all_assignments), c(names(eligibility), "assignment_status"))
excluded <- !all_assignments$common_comparison_ready
stopifnot(all(vapply(all_assignments[excluded, new_fields], function(x) all(is.na(x)), logical(1))),
  all(all_assignments$assignment_status[!excluded] == "assigned_common_retrospective_sample"),
  all(all_assignments$assignment_status[excluded] == paste0("excluded_", eligibility$primary_exclusion[excluded])))
assignments <- all_assignments[match(ids, all_assignments$hex_id), ]
unit_weights <- eligibility$residential_units[match(ids, eligibility$hex_id)]
raw <- lapply(inputs, function(x) as.matrix(x[features]))
for (i in 1:2) {
  stopifnot(identical(names(inputs[[i]]), c("hex_id", "analysis_as_of_date", features)),
    all(inputs[[i]]$analysis_as_of_date == dates[i]), all(is.finite(raw[[i]])))
  source <- paired[paired$analysis_as_of_date == dates[i], ]
  expect_equal(raw[[i]], as.matrix(source[match(ids, source$hex_id), features]))
}
# Compute preprocessing independently, without applying the implementation helper.
center <- colMeans(raw[[1]])
sample_sd <- sqrt(colSums(sweep(raw[[1]], 2, center, "-")^2) / (n - 1))
expect_equal(model$scaling$center, center)
expect_equal(model$scaling$scale, sample_sd)
stopifnot(identical(model$scaling$features, features), model$scaling$reference_date == dates[1],
  model$scaling$training_n == n, model$scaling$scale_definition == "sample_standard_deviation_n_minus_1")
z <- lapply(raw, function(x) sweep(sweep(x, 2, center, "-"), 2, sample_sd, "/"))
expect_equal(colMeans(z[[1]]), rep(0, 7))
expect_equal(apply(z[[1]], 2, sd), rep(1, 7))
# Equivalent SD arithmetic differs by a few machine-epsilon units. Validate it
# independently above, then use the saved binary parameters for exact seeded
# optimizer replay; otherwise tied-SSE starts can choose a label permutation.
stored_z <- lapply(raw, function(x) sweep(sweep(x, 2, model$scaling$center, "-"), 2, model$scaling$scale, "/"))
expect_equal(stored_z, z)
z <- stored_z
manual_assignment <- function(x, centers) {
  stopifnot(identical(rownames(centers), labels), identical(colnames(centers), features))
  distances <- sapply(seq_len(7), function(j) sqrt(rowSums((x - matrix(centers[j, ], nrow(x), 7, byrow = TRUE))^2)))
  first <- max.col(-distances, ties.method = "first")
  nearest <- distances[cbind(seq_len(nrow(x)), first)]
  second <- apply(distances, 1, function(values) sort(values)[2])
  data.frame(cluster = labels[first], distance = nearest, second = second,
    margin = ifelse(second == 0, 0, 1 - nearest / second))
}
expected_seeds <- 49L + 10000L * (0:19)
stopifnot(identical(model$optimizer_seeds, expected_seeds), length(model$optimizer_fits) == 2L)
optimizer_qa <- read_csv_output("part2_cluster_optimizer_robustness.csv")
stopifnot(nrow(optimizer_qa) == 40L, !anyDuplicated(optimizer_qa[c("analysis_as_of_date", "seed")]))
for (i in 1:2) {
  fits <- model$optimizer_fits[[i]]
  stopifnot(length(fits) == 20L, identical(vapply(fits, `[[`, integer(1), "seed"), expected_seeds))
  best_index <- which.min(vapply(fits, `[[`, numeric(1), "tot.withinss"))
  best <- if (i == 1L) model$baseline else model$later_refit
  stopifnot(identical(best, fits[[best_index]]))
  for (fit in fits) {
    stopifnot(fit$algorithm == "Lloyd", fit$nstart == 100L, fit$iter_max == 500L,
      fit$iter <= 500L, all(fit$size > 0))
    predicted <- manual_assignment(z[[i]], fit$centers)
    stopifnot(identical(predicted$cluster, fit$cluster))
    expect_equal(fit$tot.withinss, sum(predicted$distance^2))
    total_ss <- sum(sweep(z[[i]], 2L, colMeans(z[[i]]), "-")^2)
    expect_equal(fit$totss, total_ss)
    expect_equal(fit$betweenss, total_ss - sum(predicted$distance^2))
    expect_equal(fit$fit$centers, fit$centers)
    stopifnot(identical(labels[fit$fit$cluster], fit$cluster))
    expect_equal(unname(fit$size), as.integer(table(factor(fit$cluster, levels = labels))))
    for (label in labels) {
      keep <- fit$cluster == label
      expect_equal(fit$centers[label, ], colMeans(z[[i]][keep, , drop = FALSE]))
      expect_equal(fit$withinss[[label]], sum(predicted$distance[keep]^2))
    }
    qa <- optimizer_qa[optimizer_qa$analysis_as_of_date == dates[i] & optimizer_qa$seed == fit$seed, ]
    expect_equal(qa$tot_withinss, fit$tot.withinss)
    expect_equal(qa$relative_excess_withinss, fit$tot.withinss / best$tot.withinss - 1)
    expect_equal(qa$adjusted_rand, audit_ari(best$cluster, fit$cluster))
    alignment <- audit_alignment(best$cluster, fit$cluster, 7)
    expect_equal(qa$matched_share, alignment$match_share)
    expect_equal(qa$optimal_label_permutations, alignment$optimal_permutation_count)
    expected_warnings <- paste(fit$warnings, collapse = " | ")
    stopifnot("fitting_warnings" %in% names(qa),
      if (nzchar(expected_warnings)) qa$fitting_warnings == expected_warnings else
        is.na(qa$fitting_warnings) || qa$fitting_warnings == "")
  }
  # Independently replay the chosen 100-start seed at each date.
  replay <- audit_kmeans(z[[i]], k = 7L, seed = best$seed)
  expect_equal(replay$centers, best$centers)
  stopifnot(identical(replay$cluster, best$cluster))
}
baseline <- manual_assignment(z[[1]], model$baseline$centers)
fixed <- manual_assignment(z[[2]], model$baseline$centers)
refit <- manual_assignment(z[[2]], model$later_refit$centers)
stopifnot(identical(assignments$cluster_2025, baseline$cluster),
  identical(assignments$cluster_2026_fixed, fixed$cluster), identical(assignments$cluster_2026_refit_raw, refit$cluster))
distance_thresholds <- vapply(labels, function(label) as.numeric(quantile(baseline$distance[baseline$cluster == label], .95, type = 7)), numeric(1))
margin_threshold <- as.numeric(quantile(baseline$margin, .1, type = 7))
stored_thresholds <- model$baseline_thresholds
expect_equal(stored_thresholds$distance_thresholds$distance_threshold,
  distance_thresholds[match(stored_thresholds$distance_thresholds$cluster, labels)])
expect_equal(stored_thresholds$margin_threshold, margin_threshold)
for (i in 1:2) {
  observed <- if (i == 1L) baseline else fixed
  expect_equal(assignments[[if (i == 1L) "distance_2025" else "distance_2026_fixed"]], observed$distance)
  expect_equal(assignments[[if (i == 1L) "margin_2025" else "margin_2026_fixed"]], observed$margin)
  stopifnot(identical(assignments[[if (i == 1L) "low_margin_2025" else "low_margin_2026_fixed"]], observed$margin < margin_threshold),
    identical(assignments[[if (i == 1L) "far_from_baseline_2025" else "far_from_baseline_2026"]],
      observed$distance > unname(distance_thresholds[match(observed$cluster, labels)])))
}
alignment <- audit_alignment(fixed$cluster, refit$cluster, k = 7L)
stopifnot(identical(alignment, model$alignment),
  identical(assignments$cluster_2026_refit_aligned, alignment$aligned_cluster),
  identical(manual_assignment(z[[2]], model$later_refit_aligned_centers)$cluster, alignment$aligned_cluster),
  identical(assignments$moved_fixed, baseline$cluster != fixed$cluster),
  identical(assignments$fixed_refit_disagree, fixed$cluster != alignment$aligned_cluster))
expect_equal(read_csv_output("part2_refit_label_mapping.csv"), alignment$mapping)

comparison_pairs <- list(temporal_fixed = list(baseline$cluster, fixed$cluster),
  structural_later_fixed_vs_refit = list(fixed$cluster, alignment$aligned_cluster),
  combined_baseline_vs_later_refit = list(baseline$cluster, alignment$aligned_cluster))
transitions <- read_csv_output("part2_cluster_transitions.csv")
comparison_summary <- read_csv_output("part2_cluster_comparison_summary.csv")
recovery <- read_csv_output("part2_cluster_recovery.csv")
stopifnot(nrow(transitions) == 147L, nrow(comparison_summary) == 3L, nrow(recovery) == 21L)
for (comparison in names(comparison_pairs)) {
  pair <- comparison_pairs[[comparison]]
  expected <- audit_transitions(pair[[1]], pair[[2]], 7L)
  observed <- transitions[transitions$comparison == comparison, names(expected)]
  expect_equal(observed, expected)
  stopifnot(sum(observed$hexes) == n)
  changed <- pair[[1]] != pair[[2]]
  qa <- comparison_summary[comparison_summary$comparison == comparison, ]
  expect_equal(qa$changed_label_hexes, sum(changed)); expect_equal(qa$same_label_hexes, sum(!changed))
  expect_equal(qa$same_label_share, mean(!changed)); expect_equal(qa$adjusted_rand, audit_ari(pair[[1]], pair[[2]]))
  expect_equal(qa$fixed_units_changed_label, sum(unit_weights[changed]))
  expect_equal(qa$fixed_units_changed_label_share, sum(unit_weights[changed]) / sum(unit_weights))
  for (label in labels) {
    a <- pair[[1]] == label; b <- pair[[2]] == label
    q <- recovery[recovery$comparison == comparison & recovery$cluster == label, ]
    expect_equal(q$earlier_or_reference_hexes, sum(a)); expect_equal(q$later_or_candidate_hexes, sum(b))
    expect_equal(q$overlap_hexes, sum(a & b)); expect_equal(q$share_reference_retained, if (any(a)) sum(a & b) / sum(a) else NA_real_)
    expect_equal(q$jaccard, if (any(a | b)) sum(a & b) / sum(a | b) else NA_real_)
  }
}
availability <- eligibility[match(ids, eligibility$hex_id), paste0(features, "_same_term_availability")]
changed_terms <- unname(rowSums(!as.matrix(availability)) > 0L)
stopifnot(identical(assignments$any_changed_term_availability, changed_terms))
availability_transitions <- read_csv_output("part2_cluster_transitions_by_availability.csv")
availability_qa <- read_csv_output("part2_cluster_component_availability_qa.csv")
stopifnot(nrow(availability_transitions) == 98L, nrow(availability_qa) >= 1L,
  !any(changed_terms), all(availability_qa$hexes[availability_qa$any_changed_term_availability %in% TRUE] == 0L))
for (changed in c(FALSE, TRUE)) {
  keep <- changed_terms == changed
  expected <- audit_transitions(baseline$cluster[keep], fixed$cluster[keep], k = 7L)
  expect_equal(availability_transitions[availability_transitions$any_changed_term_availability == changed, names(expected)], expected)
  qa <- availability_qa[availability_qa$any_changed_term_availability == changed, ]
  if (any(keep)) {
    stopifnot(nrow(qa) == 1L)
    expect_equal(qa$hexes, sum(keep)); expect_equal(qa$moved_fixed_hexes, sum(assignments$moved_fixed[keep]))
    expect_equal(qa$moved_fixed_share, mean(assignments$moved_fixed[keep]))
    expect_equal(qa$fixed_refit_disagreement_share, mean(assignments$fixed_refit_disagree[keep]))
  } else if (nrow(qa)) {
    stopifnot(nrow(qa) == 1L, qa$hexes == 0L, qa$moved_fixed_hexes == 0L,
      is.na(qa$moved_fixed_share), is.na(qa$fixed_refit_disagreement_share))
  }
}
groups <- list(baseline_2025 = baseline$cluster, fixed_2026 = fixed$cluster, refit_2026_aligned = alignment$aligned_cluster)
profiles <- read_csv_output("part2_cluster_profiles_long.csv")
stopifnot(nrow(profiles) == 147L, !anyDuplicated(profiles[c("solution", "cluster", "feature")]))
for (solution in names(groups)) for (label in labels) {
  i <- if (solution == "baseline_2025") 1L else 2L
  keep <- groups[[solution]] == label
  p <- profiles[profiles$solution == solution & profiles$cluster == label, ]; p <- p[match(features, p$feature), ]
  expect_equal(p$n_hexes, rep(sum(keep), 7)); expect_equal(p$fixed_residential_units, rep(sum(unit_weights[keep]), 7))
  expect_equal(p$mean_index, if (any(keep)) colMeans(raw[[i]][keep, , drop = FALSE]) else rep(NA_real_, 7))
  expect_equal(p$mean_z, if (any(keep)) colMeans(z[[i]][keep, , drop = FALSE]) else rep(NA_real_, 7))
}
dz <- z[[2]] - z[[1]]
expect_equal(assignments$feature_shift_distance, sqrt(rowSums(dz^2)))
largest <- max.col(abs(dz), ties.method = "first")
stopifnot(identical(assignments$largest_standardized_change_feature, features[largest]))
expect_equal(assignments$largest_standardized_change, dz[cbind(seq_len(n), largest)])
contributions <- read_csv_output("part2_transition_feature_contributions.csv")
stopifnot(nrow(contributions) == sum(assignments$moved_fixed) * 7L,
  setequal(unique(contributions$hex_id), ids[assignments$moved_fixed]),
  !anyDuplicated(contributions[c("hex_id", "feature")]),
  all(is.na(assignments$boundary_crossing_top_feature[!assignments$moved_fixed])))
for (row in which(assignments$moved_fixed)) {
  from <- baseline$cluster[row]; to <- fixed$cluster[row]
  from_center <- model$baseline$centers[from, ]; to_center <- model$baseline$centers[to, ]
  expected <- 2 * dz[row, ] * (to_center - from_center)
  c <- contributions[contributions$hex_id == ids[row], ]; c <- c[match(features, c$feature), ]
  stopifnot(all(c$from_cluster == from), all(c$to_cluster == to))
  expect_equal(c$standardized_feature_change, dz[row, ])
  expect_equal(c$change_in_squared_distance_advantage, expected)
  advantage <- function(values) sum((values - from_center)^2) - sum((values - to_center)^2)
  expect_equal(sum(c$change_in_squared_distance_advantage), advantage(z[[2]][row, ]) - advantage(z[[1]][row, ]))
  stopifnot(assignments$boundary_crossing_top_feature[row] == features[which.max(expected)])
}

# Observed amenity totals count overlapping hex-event catchment links, not unique
# openings. Other domains assign the retained events to disjoint hexes.
prevalence <- read_csv_output("part2_cluster_observed_event_prevalence.csv")
stopifnot(nrow(prevalence) == 84L, "count_sum_definition" %in% names(prevalence))
domain_paths <- c(evictions = "output/part2/evictions/eviction_features_paired.rds",
  demolitions = "output/part2/demolitions/demolition_features_paired.rds",
  sr311 = "output/part2/311/311_features_paired.rds", amenities = "output/part2/amenities/amenity_features_paired.rds")
domain_fields <- c(evictions = "eviction_cases_latest_12mo", demolitions = "demo_latest_24mo",
  sr311 = "sr_311_smoke_signal_latest_12mo", amenities = "amenity_recent_opening_events")
for (domain in names(domain_paths)) {
  source <- readRDS(domain_paths[[domain]])
  date_column <- if (domain == "amenities") "amenity_analysis_as_of_date" else "analysis_as_of_date"
  expected_definition <- if (domain == "amenities") "hex_event_exposure_links_within_800m_not_unique_openings" else "disjoint_hex_mapped_events"
  for (solution in names(groups)) {
    i <- if (solution == "baseline_2025") 1L else 2L
    x <- source[source[[date_column]] == dates[i], ]; values <- x[[domain_fields[[domain]]]][match(ids, x$hex_id)]
    stopifnot(all(is.finite(values)), all(values >= 0))
    for (label in labels) {
      keep <- groups[[solution]] == label
      q <- prevalence[prevalence$domain == domain & prevalence$solution == solution & prevalence$cluster == label, ]
      stopifnot(nrow(q) == 1L, q$count_sum_definition == expected_definition)
      expect_equal(q$hexes, sum(keep)); expect_equal(q$observed_count_sum, sum(values[keep]))
      expect_equal(q$positive_observed_hexes, sum(values[keep] > 0))
      expect_equal(q$positive_observed_share, if (any(keep)) mean(values[keep] > 0) else NA_real_)
    }
  }
}

# The same heldout hexes are used at both dates. Spatial samples include entire
# H3-r7 parents among the common sample, and correspondence is learned on TRAIN.
memberships <- read_csv_output("part2_cluster_holdout_membership.csv")
cell_replicates <- readRDS(file.path(root, "part2_cluster_heldout_cell_replicates.rds"))
replicates <- read_csv_output("part2_cluster_resampling_replicates.csv")
cluster_replicates <- read_csv_output("part2_cluster_heldout_cluster_replicates.csv")
stopifnot(!anyDuplicated(memberships[c("scheme", "replicate", "hex_id")]),
  !anyDuplicated(cell_replicates[c("scheme", "replicate", "analysis_as_of_date", "hex_id")]),
  nrow(replicates) == 140L, nrow(cluster_replicates) == 980L)
parent_blocks <- h3jsr::get_parent(as.character(grid$h3_index[match(ids, grid$hex_id)]), res = 7L)
schemes <- c("random_hex", "h3_parent_r7")
for (s in seq_along(schemes)) {
  scheme <- schemes[s]; number <- if (s == 1L) 50L else 20L
  blocks <- if (s == 1L) as.character(ids) else parent_blocks
  stopifnot(setequal(unique(memberships$replicate[memberships$scheme == scheme]), seq_len(number)))
  for (r in seq_len(number)) {
    membership <- memberships[memberships$scheme == scheme & memberships$replicate == r, ]
    holdout <- match(membership$hex_id, ids); train <- setdiff(seq_len(n), holdout)
    stopifnot(!anyNA(holdout), all(membership$block_id == blocks[holdout]),
      setequal(holdout, which(blocks %in% unique(blocks[holdout]))))
    # Reproduce membership independently from the documented seed convention.
    set.seed(1000000L + 1000L * s + r, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    block_sizes <- table(blocks); order <- sample.int(length(block_sizes))
    prefix <- which.min(abs(cumsum(as.integer(block_sizes[order])) - round(.2 * n)))
    prefix <- max(1L, min(prefix, length(block_sizes) - 1L))
    expected_holdout <- which(blocks %in% names(block_sizes)[order[seq_len(prefix)]])
    stopifnot(identical(holdout, expected_holdout))
    for (i in 1:2) {
      reference <- if (i == 1L) baseline$cluster else alignment$aligned_cluster
      c <- cell_replicates[cell_replicates$scheme == scheme & cell_replicates$replicate == r & cell_replicates$analysis_as_of_date == dates[i], ]
      q <- replicates[replicates$scheme == scheme & replicates$replicate == r & replicates$analysis_as_of_date == dates[i], ]
      stopifnot(identical(c$hex_id, ids[holdout]), identical(c$reference_cluster, reference[holdout]),
        identical(c$recovered, c$reference_cluster == c$resampled_cluster), nrow(q) == 1L,
        q$training_hexes == length(train), q$heldout_hexes == length(holdout))
      expect_equal(q$heldout_matched_share, mean(c$recovered))
      expect_equal(q$heldout_adjusted_rand, audit_ari(c$reference_cluster, c$resampled_cluster))
      stopifnot(!anyNA(c$recovered), !anyNA(c$reference_cluster), !anyNA(c$resampled_cluster),
        all(c$reference_cluster %in% labels), all(c$resampled_cluster %in% labels))
      if (i == 2L) stopifnot(all(is.na(c$fixed_transition_recovered)), all(is.na(c$fixed_switch_recovered)))
      if (i == 1L) stopifnot(!anyNA(c$fixed_transition_recovered), !anyNA(c$fixed_switch_recovered))
      for (label in labels) {
        a <- c$reference_cluster == label; b <- c$resampled_cluster == label
        cq <- cluster_replicates[cluster_replicates$scheme == scheme & cluster_replicates$replicate == r &
          cluster_replicates$analysis_as_of_date == dates[i] & cluster_replicates$cluster == label, ]
        stopifnot(nrow(cq) == 1L)
        expect_equal(cq$reference_heldout_hexes, sum(a)); expect_equal(cq$predicted_heldout_hexes, sum(b)); expect_equal(cq$overlap_hexes, sum(a & b))
        expect_equal(cq$recovery_share, if (any(a)) sum(a & b) / sum(a) else NA_real_)
        expect_equal(cq$jaccard, if (any(a | b)) sum(a & b) / sum(a | b) else NA_real_)
      }
      if (r == 1L) {
        fit <- audit_kmeans(z[[i]][train, , drop = FALSE], k = 7L,
          seed = 2000000L + 10000L * s + 100L * r + i)
        prediction <- manual_assignment(z[[i]], fit$centers)$cluster
        training_mapping <- audit_alignment(reference[train], prediction[train], k = 7L)
        aligned <- training_mapping$mapping$aligned_cluster[match(prediction, training_mapping$mapping$candidate_cluster)]
        stopifnot(identical(aligned[holdout], c$resampled_cluster))
        expect_equal(q$training_matched_share, training_mapping$match_share)
        expect_equal(q$optimal_training_label_permutations, training_mapping$optimal_permutation_count)
        if (i == 1L) {
          prediction_later <- manual_assignment(z[[2]], fit$centers)$cluster
          aligned_later <- training_mapping$mapping$aligned_cluster[match(prediction_later, training_mapping$mapping$candidate_cluster)]
          stopifnot(identical(c$fixed_transition_recovered, (aligned == baseline$cluster & aligned_later == fixed$cluster)[holdout]),
            identical(c$fixed_switch_recovered, ((aligned != aligned_later) == assignments$moved_fixed)[holdout]))
        }
      }
    }
  }
}
cell_summary <- read_csv_output("part2_cluster_heldout_cell_summary.csv")
expected_summary <- cell_replicates %>% group_by(scheme, analysis_as_of_date, hex_id) %>% summarise(
  heldout_replicates = n(), assignment_recovery_share = mean(recovered),
  fixed_transition_recovery_share = if (all(is.na(fixed_transition_recovered))) NA_real_ else mean(fixed_transition_recovered),
  fixed_switch_recovery_share = if (all(is.na(fixed_switch_recovered))) NA_real_ else mean(fixed_switch_recovered), .groups = "drop")
expect_equal(cell_summary, expected_summary)
for (scheme in schemes) {
  earlier <- cell_summary[cell_summary$scheme == scheme & cell_summary$analysis_as_of_date == dates[1], ]
  position <- match(ids, earlier$hex_id)
  expected_n <- ifelse(is.na(position), 0L, earlier$heldout_replicates[position])
  expect_equal(assignments[[paste0(scheme, "_heldout_replicates")]], expected_n)
  expect_equal(assignments[[paste0(scheme, "_transition_recovery_share")]], earlier$fixed_transition_recovery_share[position])
  expect_equal(assignments[[paste0(scheme, "_switch_recovery_share")]], earlier$fixed_switch_recovery_share[position])
  stopifnot(all(is.na(assignments[[paste0(scheme, "_transition_recovery_share")]][expected_n == 0])),
    all(is.na(assignments[[paste0(scheme, "_switch_recovery_share")]][expected_n == 0])))
  # Cells never selected into a spatial holdout must stay unknown; their count
  # depends on the corrected eligible sample and is not an old-run constant.
  never_heldout <- ids[expected_n == 0L]
  stopifnot(!any(cell_replicates$scheme == scheme & cell_replicates$hex_id %in% never_heldout))
}
resampling_summary <- read_csv_output("part2_cluster_resampling_summary.csv")
expected_resampling <- replicates %>% group_by(scheme, analysis_as_of_date) %>% summarise(replicates = n(),
  median_heldout_ari = median(heldout_adjusted_rand), p10_heldout_ari = quantile(heldout_adjusted_rand, .1),
  min_heldout_ari = min(heldout_adjusted_rand), median_heldout_matched_share = median(heldout_matched_share), .groups = "drop")
expect_equal(resampling_summary, expected_resampling)
cat("Corrected historical cluster output audit passed:", nrow(grid), "audit cells;", n, "paired analysis cells;",
  "40 optimizer fits; 140 paired-date holdout fits; never-heldout cells kept unknown;",
  hash_checks, "current checksum verifications;", nrow(before),
  "artifacts preserved at the historical run; approved later Part 1 replacements exempted from present-day preservation.\n")
