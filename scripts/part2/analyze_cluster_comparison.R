# Retrospective Part 2 comparison; isolated from canonical Part 1 and Part 3.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
source("R/pipeline.R")
source("R/part2_feature_matrix.R")
source("R/part2_clusters.R")
root <- "output/part2/clusters"
dir.create(root, recursive = TRUE, showWarnings = FALSE)
dates <- as.Date(c("2025-04-01", "2026-04-01"))
k <- 7L; labels <- paste0("C", seq_len(k)); features <- part2_matrix_indices()
matrix_root <- "output/part2/matrix"
matrix_manifest_path <- file.path(matrix_root, "part2_matrix_run_manifest.json")
paired_path <- file.path(matrix_root, "part2_features_paired.rds")
eligibility_path <- file.path(matrix_root, "part2_eligibility_by_hex.rds")
matrix_paths <- file.path(matrix_root, paste0("part2_analysis_matrix_", dates, ".rds"))
cat("Verifying the paired feature contract and preserving existing artifacts...\n")
verified <- part2_matrix_verify_source(matrix_manifest_path, paired_path, "paired_seven_feature_matrix_complete_v2")
protected <- c(list.files("output", pattern = "[.](rds|csv|json)$", recursive = TRUE, full.names = TRUE),
  list.files("figures", recursive = TRUE, full.names = TRUE))
protected <- protected[!startsWith(protected, "output/part2/") & !startsWith(protected, "figures/part2/")]
protected <- sort(unique(normalizePath(protected, mustWork = TRUE)))
before <- build_file_manifest(protected, require_all = TRUE, hash_files = TRUE)
saveRDS(before, file.path(root, "existing_artifacts_before.rds"))
write_csv(verified, file.path(root, "matrix_source_hash_verification.csv"))
inputs <- lapply(matrix_paths, readRDS)
paired <- readRDS(paired_path); eligibility <- readRDS(eligibility_path)
grid <- readRDS("output/hex_grid.rds")
ids <- inputs[[1]]$hex_id; n <- length(ids)
stopifnot(n > k, is.integer(ids), !anyDuplicated(ids), identical(ids, inputs[[2]]$hex_id),
  identical(eligibility$hex_id, grid$hex_id), identical(ids, eligibility$hex_id[eligibility$common_comparison_ready]),
  identical(names(inputs[[1]]), c("hex_id", "analysis_as_of_date", features)),
  identical(names(inputs[[1]]), names(inputs[[2]])))
stopifnot(all(paired$measurement_version == "part2-fixed-components-v2"),
  all(paired$all_required_components_available[paired$common_comparison_ready]))
for (i in 1:2) stopifnot(all(inputs[[i]]$analysis_as_of_date == dates[i]),
  all(is.finite(as.matrix(inputs[[i]][features]))))
support <- eligibility[match(ids, eligibility$hex_id), ]
unit_weights <- support$residential_units
scaling <- part2_cluster_fit_scaling(inputs[[1]], features, dates[1])
z <- lapply(inputs, part2_cluster_apply_scaling, scaling = scaling)

# Several full nstart=100 fits guard against adopting a weaker local optimum.
# The minimum-SSE run is the primary solution, with deterministic first-seed ties.
optimizer_seeds <- 49L + 10000L * (0:19)
optimizer_fits <- lapply(seq_along(z), function(i) {
  cat("Fitting", as.character(dates[i]), ": 20 seeded 100-start k=7 solutions...\n")
  lapply(optimizer_seeds, function(seed) part2_cluster_kmeans(z[[i]], k, seed, nstart = 100L, iter_max = 500L))
})
best <- lapply(optimizer_fits, function(fits) fits[[which.min(vapply(fits, function(fit) fit$tot.withinss, numeric(1)))]])
a0 <- part2_cluster_assign(z[[1]], best[[1]]$centers)
stopifnot(identical(a0$cluster, best[[1]]$cluster))
thresholds <- part2_cluster_fit_thresholds(a0, labels)
a <- part2_cluster_assign(z[[1]], best[[1]]$centers, thresholds)
b <- part2_cluster_assign(z[[2]], best[[1]]$centers, thresholds)
later_refit <- part2_cluster_assign(z[[2]], best[[2]]$centers)
stopifnot(identical(later_refit$cluster, best[[2]]$cluster))
# Match on the SAME later observations. Do not maximize staying since 2025.
alignment <- part2_cluster_align(b$cluster, later_refit$cluster, k)
c_labels <- alignment$aligned_cluster
aligned_centers <- best[[2]]$centers[match(alignment$mapping$candidate_cluster[
  match(labels, alignment$mapping$aligned_cluster)], rownames(best[[2]]$centers)), , drop = FALSE]
rownames(aligned_centers) <- labels
stopifnot(identical(part2_cluster_assign(z[[2]], aligned_centers)$cluster, c_labels))
model <- list(schema_version = "part2-historical-clusters-v2", dates = dates, features = features, k = k,
  measurement_version = "part2-fixed-components-v2",
  training_hex_ids = ids, scaling = scaling, baseline_thresholds = thresholds,
  baseline = best[[1]], later_refit = best[[2]], later_refit_aligned_centers = aligned_centers,
  alignment = alignment, optimizer_seeds = optimizer_seeds,
  optimizer_fits = optimizer_fits, later_scaling = "frozen_2025_means_and_sample_standard_deviations",
  baseline_is_distinct_from_part1 = TRUE, risk_ranking = FALSE)
saveRDS(model, file.path(root, "part2_cluster_models.rds"))
write_csv(alignment$mapping, file.path(root, "part2_refit_label_mapping.csv"))

availability_columns <- paste0(features, "_same_term_availability")
stopifnot(all(availability_columns %in% names(support)))
changed_components <- rowSums(!as.matrix(support[availability_columns])) > 0L
stopifnot(!any(changed_components))
assignments <- data.frame(hex_id = ids, cluster_2025 = a$cluster, cluster_2026_fixed = b$cluster,
  cluster_2026_refit_raw = later_refit$cluster, cluster_2026_refit_aligned = c_labels,
  moved_fixed = a$cluster != b$cluster, fixed_refit_disagree = b$cluster != c_labels,
  low_margin_2025 = a$low_margin, low_margin_2026_fixed = b$low_margin,
  far_from_baseline_2025 = a$far_from_baseline, far_from_baseline_2026 = b$far_from_baseline,
  distance_2025 = a$distance_to_centroid, distance_2026_fixed = b$distance_to_centroid,
  margin_2025 = a$separation_margin, margin_2026_fixed = b$separation_margin,
  any_changed_term_availability = changed_components)
dz <- z[[2]] - z[[1]]
largest <- max.col(abs(dz), ties.method = "first")
assignments$feature_shift_distance <- sqrt(rowSums(dz^2))
assignments$largest_standardized_change_feature <- features[largest]
assignments$largest_standardized_change <- dz[cbind(seq_len(n), largest)]
assignments$boundary_crossing_top_feature <- NA_character_
# Exact change in squared-distance advantage, for old versus new centroid;
# descriptive feature contribution, not a causal explanation of displacement.
crossing <- do.call(rbind, lapply(which(assignments$moved_fixed), function(row) {
  from <- a$cluster[row]; to <- b$cluster[row]
  contribution <- 2 * dz[row, ] * (best[[1]]$centers[to, ] - best[[1]]$centers[from, ])
  assignments$boundary_crossing_top_feature[row] <<- features[which.max(contribution)]
  data.frame(hex_id = ids[row], from_cluster = from, to_cluster = to, feature = features,
    standardized_feature_change = as.numeric(dz[row, ]),
    change_in_squared_distance_advantage = as.numeric(contribution))
}))
if (is.null(crossing)) crossing <- data.frame(hex_id = integer(), from_cluster = character(), to_cluster = character(),
  feature = character(), standardized_feature_change = numeric(), change_in_squared_distance_advantage = numeric())
write_csv(crossing, file.path(root, "part2_transition_feature_contributions.csv"))

transition_sets <- list(temporal_fixed = list(a$cluster, b$cluster),
  structural_later_fixed_vs_refit = list(b$cluster, c_labels),
  combined_baseline_vs_later_refit = list(a$cluster, c_labels))
transition_tables <- bind_rows(lapply(names(transition_sets), function(comparison) {
  x <- transition_sets[[comparison]]
  part2_cluster_transitions(x[[1]], x[[2]], k) %>% mutate(comparison = comparison, .before = 1)
}))
write_csv(transition_tables, file.path(root, "part2_cluster_transitions.csv"))
comparison_summary <- bind_rows(lapply(names(transition_sets), function(comparison) {
  x <- transition_sets[[comparison]]; same <- x[[1]] == x[[2]]
  data.frame(comparison = comparison, hexes = n, same_label_hexes = sum(same), changed_label_hexes = sum(!same),
    same_label_share = mean(same), adjusted_rand = part2_cluster_ari(x[[1]], x[[2]]),
    fixed_units_changed_label = sum(unit_weights[!same]), fixed_units_changed_label_share = sum(unit_weights[!same]) / sum(unit_weights))
}))
write_csv(comparison_summary, file.path(root, "part2_cluster_comparison_summary.csv"))
recovery <- bind_rows(lapply(names(transition_sets), function(comparison) {
  x <- transition_sets[[comparison]]
  bind_rows(lapply(labels, function(label) {
    from <- x[[1]] == label; to <- x[[2]] == label
    data.frame(comparison = comparison, cluster = label, earlier_or_reference_hexes = sum(from),
      later_or_candidate_hexes = sum(to), overlap_hexes = sum(from & to),
      share_reference_retained = if (any(from)) sum(from & to) / sum(from) else NA_real_,
      jaccard = if (any(from | to)) sum(from & to) / sum(from | to) else NA_real_)
  }))
}))
write_csv(recovery, file.path(root, "part2_cluster_recovery.csv"))
groups <- list(baseline_2025 = a$cluster, fixed_2026 = b$cluster, refit_2026_aligned = c_labels)
profiles <- bind_rows(lapply(names(groups), function(solution) {
  i <- if (solution == "baseline_2025") 1L else 2L
  bind_rows(lapply(labels, function(label) {
    keep <- groups[[solution]] == label
    data.frame(solution = solution, cluster = label, feature = features, n_hexes = sum(keep),
      fixed_residential_units = sum(unit_weights[keep]),
      mean_index = if (any(keep)) colMeans(inputs[[i]][keep, features, drop = FALSE]) else NA_real_,
      mean_z = if (any(keep)) colMeans(z[[i]][keep, , drop = FALSE]) else NA_real_)
  }))
}))
write_csv(profiles, file.path(root, "part2_cluster_profiles_long.csv"))
centroid_drift <- bind_rows(lapply(labels, function(label) {
  shift <- aligned_centers[label, ] - best[[1]]$centers[label, ]
  data.frame(cluster = label, centroid_shift_distance_2025_sd = sqrt(sum(shift^2)),
    largest_centroid_shift_feature = features[which.max(abs(shift))],
    largest_centroid_shift_2025_sd = shift[which.max(abs(shift))])
}))
write_csv(centroid_drift, file.path(root, "part2_refit_centroid_drift.csv"))
internal <- bind_rows(lapply(names(groups), function(solution) {
  i <- if (solution == "baseline_2025") 1L else 2L
  cluster_ids <- match(groups[[solution]], labels)
  silhouettes <- cluster::silhouette(cluster_ids, dist(z[[i]]))
  sizes <- as.integer(table(factor(groups[[solution]], levels = labels)))
  centers <- if (solution == "refit_2026_aligned") aligned_centers else best[[1]]$centers
  loss <- sum((z[[i]] - centers[cluster_ids, , drop = FALSE])^2)
  data.frame(solution = solution, hexes = n, k = k, mean_silhouette = mean(silhouettes[, "sil_width"]),
    negative_silhouette_share = mean(silhouettes[, "sil_width"] < 0),
    min_cluster_hexes = min(sizes), max_cluster_hexes = max(sizes),
    squared_distance_to_assigned_centers = loss, loss_per_hex = loss / n)
}))
write_csv(internal, file.path(root, "part2_cluster_internal_diagnostics.csv"))

# Raw-event prevalence keeps a high relative composite from being mistaken for
# universal observed events in every cell bearing a cluster label.
domain_paths <- c(evictions = "output/part2/evictions/eviction_features_paired.rds",
  demolitions = "output/part2/demolitions/demolition_features_paired.rds",
  sr311 = "output/part2/311/311_features_paired.rds", amenities = "output/part2/amenities/amenity_features_paired.rds")
domain_fields <- c(evictions = "eviction_cases_latest_12mo", demolitions = "demo_latest_24mo",
  sr311 = "sr_311_smoke_signal_latest_12mo", amenities = "amenity_recent_opening_events")
event_prevalence <- bind_rows(lapply(names(domain_paths), function(domain) {
  x <- readRDS(domain_paths[domain]); date_col <- if (domain == "amenities") "amenity_analysis_as_of_date" else "analysis_as_of_date"
  stopifnot(all(c("hex_id", date_col, domain_fields[domain]) %in% names(x)))
  bind_rows(lapply(names(groups), function(solution) {
    i <- if (solution == "baseline_2025") 1L else 2L
    snapshot <- x[x[[date_col]] == dates[i], ]; values <- snapshot[[domain_fields[domain]]][match(ids, snapshot$hex_id)]
    stopifnot(length(values) == n, !anyNA(values), all(is.finite(values)), all(values >= 0))
    bind_rows(lapply(labels, function(label) {
      keep <- groups[[solution]] == label
      data.frame(solution = solution, cluster = label, domain = domain, hexes = sum(keep),
        positive_observed_hexes = sum(values[keep] > 0),
        positive_observed_share = if (any(keep)) mean(values[keep] > 0) else NA_real_,
        observed_count_sum = sum(values[keep]), count_sum_definition = if (domain == "amenities")
          "hex_event_exposure_links_within_800m_not_unique_openings" else "disjoint_hex_mapped_events")
    }))
  }))
}))
write_csv(event_prevalence, file.path(root, "part2_cluster_observed_event_prevalence.csv"))
composition_qa <- assignments %>% group_by(any_changed_term_availability) %>% summarise(hexes = n(),
  moved_fixed_hexes = sum(moved_fixed), moved_fixed_share = mean(moved_fixed),
  fixed_refit_disagreement_share = mean(fixed_refit_disagree), .groups = "drop")
write_csv(composition_qa, file.path(root, "part2_cluster_component_availability_qa.csv"))
composition_transitions <- bind_rows(lapply(c(FALSE, TRUE), function(changed) {
  keep <- changed_components == changed
  part2_cluster_transitions(a$cluster[keep], b$cluster[keep], k) %>%
    mutate(any_changed_term_availability = changed, .before = 1)
}))
write_csv(composition_transitions, file.path(root, "part2_cluster_transitions_by_availability.csv"))
feature_changes <- bind_rows(lapply(c(FALSE, TRUE), function(moved) {
  keep <- assignments$moved_fixed == moved
  data.frame(moved_fixed = moved, hexes = sum(keep), feature = features,
    mean_index_change = if (any(keep)) colMeans(as.matrix(inputs[[2]][keep, features]) - as.matrix(inputs[[1]][keep, features])) else NA_real_,
    mean_standardized_change = if (any(keep)) colMeans(dz[keep, , drop = FALSE]) else NA_real_,
    mean_absolute_standardized_change = if (any(keep)) colMeans(abs(dz[keep, , drop = FALSE])) else NA_real_)
}))
write_csv(feature_changes, file.path(root, "part2_cluster_feature_change_by_movement.csv"))

# Conditional optimizer and spatial/random centroid robustness. Full-period
# 2025 standardization stays fixed; these are not predictive accuracy estimates.
optimizer_qa <- bind_rows(lapply(1:2, function(i) {
  reference <- best[[i]]$cluster
  bind_rows(lapply(seq_along(optimizer_fits[[i]]), function(r) {
    fit <- optimizer_fits[[i]][[r]]
    aligned <- part2_cluster_align(reference, fit$cluster, k)
    data.frame(analysis_as_of_date = dates[i], seed = optimizer_seeds[r], tot_withinss = fit$tot.withinss,
      relative_excess_withinss = fit$tot.withinss / best[[i]]$tot.withinss - 1,
      adjusted_rand = part2_cluster_ari(reference, fit$cluster),
      matched_share = aligned$match_share, optimal_label_permutations = aligned$optimal_permutation_count,
      fitting_warnings = paste(fit$warnings, collapse = " | "))
  }))
}))
write_csv(optimizer_qa, file.path(root, "part2_cluster_optimizer_robustness.csv"))
blocks <- h3jsr::get_parent(as.character(grid$h3_index[match(ids, grid$hex_id)]), res = 7L)
stopifnot(length(blocks) == n, !anyNA(blocks), length(unique(blocks)) > 5L)
select_holdout <- function(blocks, target_n, seed) {
  set.seed(seed, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  sizes <- table(blocks); ordered <- sample.int(length(sizes))
  take <- which.min(abs(cumsum(as.integer(sizes[ordered])) - target_n))
  take <- max(1L, min(take, length(sizes) - 1L))
  which(blocks %in% names(sizes)[ordered[seq_len(take)]])
}
schemes <- list(random_hex = list(blocks = as.character(ids), replicates = 50L),
  h3_parent_r7 = list(blocks = blocks, replicates = 20L))
stability <- list(); cell_stability <- list(); cluster_stability <- list(); memberships <- list()
counter <- 0L
for (scheme in names(schemes)) {
  specification <- schemes[[scheme]]
  cat("Checking", scheme, "conditional stability:", specification$replicates, "paired holdouts...\n")
  for (r in seq_len(specification$replicates)) {
    holdout <- select_holdout(specification$blocks, round(.2 * n), 1000000L + 1000L * match(scheme, names(schemes)) + r)
    train <- setdiff(seq_len(n), holdout)
    memberships[[length(memberships) + 1L]] <- data.frame(scheme = scheme, replicate = r,
      hex_id = ids[holdout], block_id = specification$blocks[holdout])
    for (i in 1:2) {
      fit <- part2_cluster_kmeans(z[[i]][train, , drop = FALSE], k,
        seed = 2000000L + 10000L * match(scheme, names(schemes)) + 100L * r + i,
        nstart = 100L, iter_max = 500L)
      predicted <- part2_cluster_assign(z[[i]], fit$centers)$cluster
      reference <- if (i == 1L) a$cluster else c_labels
      # Learn correspondence on training cells only, then evaluate holdouts.
      match_labels <- part2_cluster_align(reference[train], predicted[train], k)
      aligned <- match_labels$mapping$aligned_cluster[match(predicted, match_labels$mapping$candidate_cluster)]
      counter <- counter + 1L
      stability[[counter]] <- data.frame(scheme = scheme, replicate = r, analysis_as_of_date = dates[i],
        training_hexes = length(train), heldout_hexes = length(holdout),
        heldout_adjusted_rand = part2_cluster_ari(reference[holdout], aligned[holdout]),
        heldout_matched_share = mean(reference[holdout] == aligned[holdout]),
        training_matched_share = match_labels$match_share,
        optimal_training_label_permutations = match_labels$optimal_permutation_count)
      per_cell <- data.frame(scheme = scheme, replicate = r, analysis_as_of_date = dates[i],
        hex_id = ids[holdout], reference_cluster = reference[holdout], resampled_cluster = aligned[holdout],
        recovered = reference[holdout] == aligned[holdout],
        fixed_transition_recovered = NA, fixed_switch_recovered = NA)
      if (i == 1L) {
        predicted_later <- part2_cluster_assign(z[[2]], fit$centers)$cluster
        aligned_later <- match_labels$mapping$aligned_cluster[match(predicted_later, match_labels$mapping$candidate_cluster)]
        per_cell$fixed_transition_recovered <- (aligned == a$cluster & aligned_later == b$cluster)[holdout]
        per_cell$fixed_switch_recovered <- ((aligned != aligned_later) == assignments$moved_fixed)[holdout]
      }
      cell_stability[[counter]] <- per_cell
      cluster_stability[[counter]] <- bind_rows(lapply(labels, function(label) {
        ref <- reference[holdout] == label; prediction <- aligned[holdout] == label
        data.frame(scheme = scheme, replicate = r, analysis_as_of_date = dates[i], cluster = label,
          reference_heldout_hexes = sum(ref), predicted_heldout_hexes = sum(prediction),
          overlap_hexes = sum(ref & prediction),
          recovery_share = if (any(ref)) sum(ref & prediction) / sum(ref) else NA_real_,
          jaccard = if (any(ref | prediction)) sum(ref & prediction) / sum(ref | prediction) else NA_real_)
      }))
    }
    if (r %% 10L == 0L) cat("  completed", r, "paired", scheme, "replicates\n")
  }
}
stability <- bind_rows(stability); cell_stability <- bind_rows(cell_stability)
cluster_stability <- bind_rows(cluster_stability)
write_csv(stability, file.path(root, "part2_cluster_resampling_replicates.csv"))
write_csv(bind_rows(memberships), file.path(root, "part2_cluster_holdout_membership.csv"))
saveRDS(cell_stability, file.path(root, "part2_cluster_heldout_cell_replicates.rds"))
write_csv(cluster_stability, file.path(root, "part2_cluster_heldout_cluster_replicates.csv"))
stability_summary <- stability %>% group_by(scheme, analysis_as_of_date) %>% summarise(replicates = n(),
  median_heldout_ari = median(heldout_adjusted_rand), p10_heldout_ari = quantile(heldout_adjusted_rand, .1),
  min_heldout_ari = min(heldout_adjusted_rand), median_heldout_matched_share = median(heldout_matched_share),
  .groups = "drop")
write_csv(stability_summary, file.path(root, "part2_cluster_resampling_summary.csv"))
cell_summary <- cell_stability %>% group_by(scheme, analysis_as_of_date, hex_id) %>% summarise(
  heldout_replicates = n(), assignment_recovery_share = mean(recovered),
  fixed_transition_recovery_share = if (all(is.na(fixed_transition_recovered))) NA_real_ else mean(fixed_transition_recovered),
  fixed_switch_recovery_share = if (all(is.na(fixed_switch_recovered))) NA_real_ else mean(fixed_switch_recovered), .groups = "drop")
write_csv(cell_summary, file.path(root, "part2_cluster_heldout_cell_summary.csv"))
for (scheme in names(schemes)) {
  subset <- cell_summary[cell_summary$scheme == scheme & cell_summary$analysis_as_of_date == dates[1], ]
  m <- match(assignments$hex_id, subset$hex_id)
  assignments[[paste0(scheme, "_heldout_replicates")]] <- ifelse(is.na(m), 0L, subset$heldout_replicates[m])
  assignments[[paste0(scheme, "_transition_recovery_share")]] <- subset$fixed_transition_recovery_share[m]
  assignments[[paste0(scheme, "_switch_recovery_share")]] <- subset$fixed_switch_recovery_share[m]
}
all_assignments <- eligibility %>% left_join(assignments, by = "hex_id") %>% mutate(
  assignment_status = if_else(common_comparison_ready, "assigned_common_retrospective_sample", paste0("excluded_", primary_exclusion)))
stopifnot(identical(all_assignments$hex_id, grid$hex_id),
  all(is.na(all_assignments$cluster_2025[!all_assignments$common_comparison_ready])))
saveRDS(all_assignments, file.path(root, "part2_cluster_assignments.rds"))
write_csv(all_assignments, file.path(root, "part2_cluster_assignments.csv"))
movement_qa <- assignments %>% group_by(moved_fixed) %>% summarise(hexes = n(),
  low_margin_2025_share = mean(low_margin_2025), low_margin_2026_share = mean(low_margin_2026_fixed),
  far_from_baseline_2026_share = mean(far_from_baseline_2026),
  changed_term_availability_share = mean(any_changed_term_availability),
  median_feature_shift_distance = median(feature_shift_distance),
  median_random_transition_recovery = median(random_hex_transition_recovery_share, na.rm = TRUE),
  median_spatial_transition_recovery = median(h3_parent_r7_transition_recovery_share, na.rm = TRUE), .groups = "drop")
write_csv(movement_qa, file.path(root, "part2_cluster_movement_qa.csv"))

after <- build_file_manifest(before$path, require_all = TRUE, hash_files = TRUE)
stopifnot(setequal(before$path, after$path),
  identical(unname(before$sha256), unname(after$sha256[match(before$path, after$path)])))
write_csv(after, file.path(root, "existing_artifacts_preservation_after.csv"))
manifest <- list(schema_version = 2L, status = "historical_cluster_comparison_complete_v2",
  measurement_version = "part2-fixed-components-v2",
  generated_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  git_commit = system2("git", c("rev-parse", "HEAD"), stdout = TRUE),
  cutoffs = as.character(dates), audit_hexes = nrow(grid), common_hexes = n, features = features, k = k,
  scaling = "2025 common-sample mean/sample-SD frozen for all later assignments, later centroid refits and conditional resampling",
  fitting = "Lloyd kmeans; minimum within-SS across20 seeded fits/date,100starts/fit,500iterations maximum",
  optimizer_seeds = optimizer_seeds,
  label_alignment = "Exact one-to-one maximum overlap of refit2026 versus coeval fixed2026; deterministic lexicographic ties, no risk ordering",
  optimal_refit_label_permutations = alignment$optimal_permutation_count,
  thresholds = "Earlier assigned-cluster95th percentile distance and earlier overall10th percentile margin; reference flags, not probabilities",
  resampling = "50 paired random-cell and20 paired whole-H3r7-block approximate20% holdouts/date; train-only label mapping; frozen preprocessing; conditional robustness, not predictive validation",
  retrospective_reconstruction = TRUE, ml_work_paused = TRUE, canonical_part1_unchanged = TRUE,
  existing_artifacts_preserved = nrow(before), risk_ranking = FALSE,
  preservation_scope = "Non-Part2 outputs and figures; prior failed Part2 run overwritten, not archived",
  limitations = c("Covered subset, not all Austin; no individual displacement prediction or causal interpretation",
    "Fixed current support, overlapping ACS releases, mapped event universes and historical source gaps remain",
    "All included observations have the full fixed component recipe; no availability-dependent weighting",
    "Rent fallback keeps geographic level fixed across six vintages, not historically harmonized Census boundaries",
    "Signed event rate-change scores use50 for no change; relative composites are not risk probabilities",
    "k7 is inherited for this proof of concept, not reselected or claimed optimal for both dates",
    "Refit and random/spatial robustness are descriptive sensitivity checks conditional on existing scoring and2025scaling",
    "Automatically aligned labels and cluster profiles are tentative types, not validated risk categories"),
  runtime = list(R = R.version.string, packages = sapply(c("dplyr", "sf", "cluster", "h3jsr"), function(p) as.character(packageVersion(p)))),
  inputs = build_file_manifest(c(matrix_manifest_path, matrix_paths, paired_path, eligibility_path, domain_paths,
    "output/hex_grid.rds", "R/pipeline.R", "R/part2_feature_matrix.R", "R/part2_clusters.R",
    "scripts/part2/analyze_cluster_comparison.R"), require_all = TRUE, hash_files = TRUE),
  outputs = build_file_manifest(list.files(root, full.names = TRUE, pattern = "[.](rds|csv)$"), require_all = TRUE, hash_files = TRUE))
jsonlite::write_json(manifest, file.path(root, "part2_cluster_run_manifest.json"), auto_unbox = TRUE, pretty = TRUE, na = "null", digits = NA)
print(comparison_summary); print(stability_summary); print(movement_qa)
cat("Historical cluster comparison complete. Canonical artifacts preserved:", nrow(before), "\n")
