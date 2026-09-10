# Synthetic, file-free tests for the paired historical clustering mechanics.
source("R/part2_clusters.R")
expect_error <- function(expr, pattern = NULL) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"))
  if (!is.null(pattern)) stopifnot(grepl(pattern, conditionMessage(result)))
}
expect_equal <- function(actual, expected) stopifnot(isTRUE(all.equal(actual, expected, check.attributes = FALSE)))
data <- data.frame(a = c(-10, -9, -8, 0, 1, 2, 10, 11, 12),
  b = c(-8, -10, -9, 0, 2, 1, 9, 12, 10), analysis_as_of_date = as.Date("2025-04-01"))
features <- c("a", "b")
scaling <- part2_cluster_fit_scaling(data, features)
z <- part2_cluster_apply_scaling(data, scaling)
expect_equal(scaling$center, colMeans(data[features]))
expect_equal(scaling$scale, vapply(data[features], stats::sd, numeric(1)))
expect_equal(colMeans(z), c(0, 0))
expect_equal(apply(z, 2, stats::sd), c(1, 1))
stopifnot(identical(colnames(z), features), identical(scaling$training_n, 9L))
later <- data
later$a <- later$a + 10
later$b <- 2 * later$b
later$analysis_as_of_date <- as.Date("2026-04-01")
later_z <- part2_cluster_apply_scaling(later, scaling)
expect_equal(colMeans(later_z), (colMeans(later[features]) - scaling$center) / scaling$scale)
stopifnot(abs(mean(later_z[, "a"])) > 1, abs(stats::sd(later_z[, "b"]) - 2) < 1e-12)
expect_equal(part2_cluster_apply_scaling(later[c("b", "a", "analysis_as_of_date")], scaling), later_z)
expect_error(part2_cluster_fit_scaling(later, features), "only on the baseline")
bad <- data; bad$analysis_as_of_date <- as.Date("2024-04-01")
expect_error(part2_cluster_apply_scaling(bad, scaling), "future")
bad <- data; bad$a <- 1
expect_error(part2_cluster_fit_scaling(bad, features), "zero sample")
expect_error(part2_cluster_fit_scaling(data[1, ], features), "two baseline rows")
for (value in c(NA_real_, NaN, Inf, -Inf)) {
  bad <- data; bad$a[1] <- value
  expect_error(part2_cluster_fit_scaling(bad, features), "finite")
  expect_error(part2_cluster_apply_scaling(bad, scaling), "finite")
}
bad <- scaling; bad$scale[1] <- 0
expect_error(part2_cluster_apply_scaling(data, bad), "scaling contract")
bad <- scaling; names(bad$center) <- rev(features)
expect_error(part2_cluster_apply_scaling(data, bad), "scaling contract")
bad <- data; bad$a <- as.character(bad$a)
expect_error(part2_cluster_fit_scaling(bad, features), "numeric")
expect_error(part2_cluster_fit_scaling(data, c("a", "a")), "features")
expect_error(part2_cluster_fit_scaling(data, c("a", "missing")), "features")
expect_error(part2_cluster_matrix(matrix(1:4, 2)), "feature names")

set.seed(314)
seed_before <- .Random.seed
fit <- part2_cluster_kmeans(z, k = 3, seed = 21)
stopifnot(identical(.Random.seed, seed_before), identical(fit$seed, 21L),
  identical(fit$nstart, 100L), identical(fit$iter_max, 500L), identical(fit$algorithm, "Lloyd"),
  identical(rownames(fit$centers), paste0("C", 1:3)), identical(colnames(fit$centers), features),
  identical(part2_cluster_assign(z, fit$centers)$cluster, fit$cluster),
  identical(fit, part2_cluster_kmeans(z, k = 3, seed = 21)), all(fit$size == 3),
  all(fit$cluster %in% paste0("C", 1:3)))
expect_equal(fit$tot.withinss, sum(fit$withinss))
expect_equal(fit$totss, fit$betweenss + fit$tot.withinss)
for (cluster in fit$labels) {
  expect_equal(fit$centers[cluster, ], colMeans(z[fit$cluster == cluster, , drop = FALSE]))
  expect_equal(fit$withinss[[cluster]], sum((sweep(z[fit$cluster == cluster, , drop = FALSE], 2, fit$centers[cluster, ], "-"))^2))
}
# A fit must also restore absence of the global RNG seed.
test_no_seed_side_effect <- function() {
  saved_seed <- .Random.seed
  on.exit(assign(".Random.seed", saved_seed, envir = .GlobalEnv))
  rm(".Random.seed", envir = .GlobalEnv)
  invisible(part2_cluster_kmeans(z, k = 3, seed = 21))
  stopifnot(!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
}
test_no_seed_side_effect()
expect_error(part2_cluster_kmeans(z, k = 1), "at least two")
expect_error(part2_cluster_kmeans(z, k = 1e20), "at least two")
expect_error(part2_cluster_kmeans(z, k = 9), "insufficient distinct")
expect_error(part2_cluster_kmeans(z, k = 3, seed = -1), "seed/settings")
expect_error(part2_cluster_kmeans(z, k = 3, nstart = 0), "seed/settings")
expect_error(part2_cluster_kmeans(z, k = 3, iter_max = .5), "seed/settings")
repeated <- matrix(rep(c(0, 0, 1, 1), 3), ncol = 2, byrow = TRUE, dimnames = list(NULL, features))
expect_error(part2_cluster_kmeans(repeated, k = 3), "insufficient distinct")

# Exact ties choose the smallest neutral label irrespective of centroid row order.
centers <- matrix(c(0, 10), ncol = 1, dimnames = list(c("C1", "C2"), "a"))
points <- matrix(c(0, 5, 11), ncol = 1, dimnames = list(NULL, "a"))
assigned <- part2_cluster_assign(points, centers)
stopifnot(identical(assigned$cluster, c("C1", "C1", "C2")),
  identical(assigned, part2_cluster_assign(points, centers[2:1, , drop = FALSE])),
  all(is.na(assigned$low_margin)), all(is.na(assigned$far_from_baseline)))
expect_equal(assigned$distance_to_centroid, c(0, 5, 1))
expect_equal(assigned$second_centroid_distance, c(10, 5, 11))
expect_equal(assigned$separation_margin, c(1, 0, 10 / 11))
duplicate_centers <- centers; duplicate_centers[, 1] <- 0
zero_tie <- part2_cluster_assign(points[1, , drop = FALSE], duplicate_centers)
stopifnot(zero_tie$cluster == "C1", zero_tie$separation_margin == 0,
  zero_tie$distance_to_centroid == 0, zero_tie$second_centroid_distance == 0)
threshold_training <- data.frame(cluster = c("C1", "C1", "C2", "C2"),
  distance_to_centroid = c(0, 2, 0, 4), separation_margin = c(.1, .2, .7, .9))
thresholds <- part2_cluster_fit_thresholds(threshold_training, c("C1", "C2"))
expect_equal(thresholds$distance_thresholds$distance_threshold, c(1.9, 3.8))
expect_equal(thresholds$margin_threshold, .13)
flagged <- part2_cluster_assign(points, centers, thresholds)
stopifnot(identical(flagged$far_from_baseline, c(FALSE, TRUE, FALSE)),
  identical(flagged$low_margin, c(FALSE, TRUE, FALSE)))
baseline_assignment <- part2_cluster_assign(z, fit$centers)
baseline_thresholds <- part2_cluster_fit_thresholds(baseline_assignment, fit$labels)
baseline_reassignment <- part2_cluster_assign(z, fit$centers, baseline_thresholds)
stopifnot(identical(baseline_reassignment$cluster, fit$cluster))
expect_error(part2_cluster_fit_thresholds(threshold_training[1:2, ], c("C1", "C2")), "every cluster")
bad <- thresholds; bad$margin_threshold <- 1.1
expect_error(part2_cluster_assign(points, centers, bad), "threshold contract")
bad <- centers; colnames(bad) <- "b"
expect_error(part2_cluster_assign(points, bad), "feature order")
bad <- centers; rownames(bad) <- c("high risk", "low risk")
expect_error(part2_cluster_assign(points, bad), "neutral cluster labels")
expect_error(part2_cluster_assign(z, fit$centers[, 2:1, drop = FALSE]), "feature order")

# ARI is permutation invariant, including the two degenerate perfect partitions.
stopifnot(part2_cluster_ari(c(1, 1, 2, 2), c("b", "b", "a", "a")) == 1,
  part2_cluster_ari(rep("a", 4), rep("b", 4)) == 1,
  part2_cluster_ari(1:4, letters[1:4]) == 1,
  part2_cluster_ari("a", "b") == 1,
  part2_cluster_ari(rep("a", 4), letters[1:4]) == 0)
expect_equal(part2_cluster_ari(c(1, 1, 2, 2), c(1, 2, 1, 2)), -.5)
expect_error(part2_cluster_ari(c(1, NA), c(1, 2)), "partition")
expect_error(part2_cluster_ari(1:3, 1:2), "same ordered rows")

reference <- rep(c("C1", "C2", "C3"), c(3, 4, 2))
candidate <- c(C1 = "C2", C2 = "C3", C3 = "C1")[reference]
aligned <- part2_cluster_align(reference, candidate, k = 3)
stopifnot(identical(aligned$aligned_cluster, reference), aligned$matched_hexes == 9L,
  aligned$match_share == 1, aligned$optimal_permutation_count == 1L)
expect_equal(aligned$mapping$aligned_cluster, c("C3", "C1", "C2"))
expect_equal(part2_cluster_ari(reference, candidate), 1)
stopifnot(nrow(part2_cluster_permutations(1:7)) == 5040L)
# An entirely tied 2x2 overlap uses the identity, not a random mapping.
tie <- part2_cluster_align(c("C1", "C1", "C2", "C2"), c("C1", "C2", "C1", "C2"), k = 2)
stopifnot(tie$optimal_permutation_count == 2L, tie$matched_hexes == 2L,
  identical(tie$mapping$aligned_cluster, c("C1", "C2")))
missing_group <- part2_cluster_align(c("C1", "C1", "C2", "C2"), rep("C3", 4), k = 3)
stopifnot(identical(dim(missing_group$overlap_table), c(3L, 3L)),
  sum(missing_group$overlap_table["C3", ]) == 0,
  all(missing_group$overlap_table[, "C1"] == 0),
  missing_group$matched_hexes == 2L, missing_group$optimal_permutation_count == 4L)
expect_error(part2_cluster_align(reference, candidate[-1], k = 3), "same ordered shared")
expect_error(part2_cluster_align(c("C1", "C4"), c("C1", "C2"), k = 3), "partition")
expect_error(part2_cluster_align("C1", "C1", k = 9), "at most eight")

# Complete transition tables retain zero flows and absent origin/destination groups.
transitions <- part2_cluster_transitions(c("C1", "C1", "C2"), c("C1", "C2", "C2"), k = 3)
stopifnot(nrow(transitions) == 9L, sum(transitions$hexes) == 3L,
  identical(transitions$from_cluster, rep(paste0("C", 1:3), each = 3)),
  identical(transitions$to_cluster, rep(paste0("C", 1:3), 3)),
  all(is.na(transitions$share_of_from[transitions$from_cluster == "C3"])),
  all(is.na(transitions$share_of_to[transitions$to_cluster == "C3"])))
expect_equal(transitions$hexes, c(1, 1, 0, 0, 1, 0, 0, 0, 0))
expect_equal(transitions$share_of_from[1:3], c(.5, .5, 0))
expect_equal(sum(transitions$share_of_all), 1)
empty_transitions <- part2_cluster_transitions(character(), character(), k = 2)
stopifnot(nrow(empty_transitions) == 4L, all(empty_transitions$hexes == 0), all(is.na(empty_transitions$share_of_all)))
expect_error(part2_cluster_transitions("C1", character(), k = 2), "same ordered shared")
profiles <- part2_cluster_profiles(data[1:3, ], c("C1", "C1", "C2"), features, k = 3)
stopifnot(nrow(profiles) == 6L, all(profiles$n[profiles$cluster == "C3"] == 0),
  all(is.na(profiles$mean[profiles$cluster == "C3"])),
  all(is.na(profiles$sd[profiles$cluster == "C2"])))
expect_equal(profiles$mean[profiles$cluster == "C1" & profiles$feature == "a"], -9.5)
expect_error(part2_cluster_profiles(data, c("C1", "C2"), features, k = 3), "ordered feature rows")
cat("Historical cluster helper synthetic tests passed.\n")
