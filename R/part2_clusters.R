# Pure historical-cluster helpers. These never read/write files or import the
# current Part 1 model/labels. Both dates use the earlier mean/sample-SD scale.
part2_cluster_labels <- function(k) {
  if (!is.numeric(k) || length(k) != 1L || is.na(k) || !is.finite(k) || k < 2 ||
      k > .Machine$integer.max || k != as.integer(k)) {
    stop("Cluster count must be a single integer of at least two.", call. = FALSE)
  }
  paste0("C", seq_len(k))
}

part2_cluster_matrix <- function(x, label = "Cluster matrix") {
  if (!(is.matrix(x) || is.data.frame(x)) || !nrow(x) || !ncol(x) ||
      (is.data.frame(x) && any(!vapply(x, is.numeric, logical(1)))) ||
      (is.matrix(x) && !is.numeric(x))) stop(label, " must be nonempty and numeric.", call. = FALSE)
  x <- as.matrix(x)
  if (is.null(colnames(x)) || anyNA(colnames(x)) || any(!nzchar(colnames(x))) || anyDuplicated(colnames(x))) {
    stop(label, " requires unique, nonmissing feature names.", call. = FALSE)
  }
  storage.mode(x) <- "double"
  if (any(!is.finite(x))) stop(label, " must contain only finite values; mask eligibility before clustering.", call. = FALSE)
  x
}

part2_cluster_feature_matrix <- function(data, features) {
  if (!is.character(features) || !length(features) || anyNA(features) ||
      any(!nzchar(features)) || anyDuplicated(features) ||
      !(is.matrix(data) || is.data.frame(data)) || !all(features %in% colnames(data))) {
    stop("Invalid or missing historical cluster features.", call. = FALSE)
  }
  part2_cluster_matrix(data[, features, drop = FALSE])
}

part2_cluster_fit_scaling <- function(data, features, reference_date = as.Date("2025-04-01")) {
  reference_date <- as.Date(reference_date)
  if (length(reference_date) != 1L || is.na(reference_date)) stop("Invalid scaling reference date.")
  if (is.data.frame(data) && "analysis_as_of_date" %in% names(data) &&
      (anyNA(data$analysis_as_of_date) || any(as.Date(data$analysis_as_of_date) != reference_date))) {
    stop("Fit historical cluster preprocessing only on the baseline date.", call. = FALSE)
  }
  x <- part2_cluster_feature_matrix(data, features)
  if (nrow(x) < 2L) stop("At least two baseline rows are required for sample-SD scaling.")
  center <- colMeans(x)
  scale <- apply(x, 2, stats::sd)
  if (any(!is.finite(center)) || any(!is.finite(scale) | scale <= 0)) {
    stop("Baseline features have invalid or zero sample standard deviation.", call. = FALSE)
  }
  list(schema_version = "part2-cluster-scaling-v1", features = features,
    reference_date = reference_date, center = center, scale = scale, training_n = nrow(x),
    scale_definition = "sample_standard_deviation_n_minus_1")
}

part2_cluster_apply_scaling <- function(data, scaling) {
  if (!is.list(scaling) || !identical(scaling$schema_version, "part2-cluster-scaling-v1") ||
      !identical(scaling$scale_definition, "sample_standard_deviation_n_minus_1") ||
      !is.numeric(scaling$center) || !is.numeric(scaling$scale) ||
      !identical(names(scaling$center), scaling$features) || !identical(names(scaling$scale), scaling$features) ||
      any(!is.finite(scaling$center)) || any(!is.finite(scaling$scale) | scaling$scale <= 0) ||
      length(scaling$reference_date) != 1L || is.na(scaling$reference_date)) {
    stop("Invalid frozen historical cluster scaling contract.", call. = FALSE)
  }
  if (is.data.frame(data) && "analysis_as_of_date" %in% names(data) &&
      (anyNA(data$analysis_as_of_date) || any(as.Date(data$analysis_as_of_date) < scaling$reference_date))) {
    stop("Cannot use future historical cluster preprocessing.", call. = FALSE)
  }
  x <- part2_cluster_feature_matrix(data, scaling$features)
  scaled <- sweep(sweep(x, 2, scaling$center, "-"), 2, scaling$scale, "/")
  if (any(!is.finite(scaled))) stop("Nonfinite standardized historical cluster feature.")
  scaled
}

part2_cluster_validate_labels <- function(x, labels = NULL, label = "Cluster labels", allow_empty = FALSE) {
  if (!is.atomic(x) || is.matrix(x) || is.list(x) || anyNA(x) ||
      (!allow_empty && !length(x)) || (is.numeric(x) && any(!is.finite(x)))) stop("Invalid ", label, ".")
  x <- as.character(x)
  if (any(!nzchar(trimws(x))) || (!is.null(labels) && any(!x %in% labels))) stop("Invalid ", label, ".")
  x
}

part2_cluster_assign <- function(z, centers, thresholds = NULL) {
  z <- part2_cluster_matrix(z)
  centers <- part2_cluster_matrix(centers, "Centroid matrix")
  labels <- part2_cluster_labels(nrow(centers))
  if (!identical(colnames(z), colnames(centers)) || is.null(rownames(centers)) ||
      anyDuplicated(rownames(centers)) || !setequal(rownames(centers), labels)) {
    stop("Centroid feature order or neutral cluster labels differ.", call. = FALSE)
  }
  # Canonical row order also makes all exact distance ties choose C1, C2, ... .
  centers <- centers[labels, , drop = FALSE]
  distances <- vapply(seq_len(nrow(centers)), function(j) {
    sqrt(rowSums(sweep(z, 2, centers[j, ], "-")^2))
  }, numeric(nrow(z)))
  distances <- matrix(distances, nrow = nrow(z), ncol = nrow(centers), dimnames = list(NULL, labels))
  if (any(!is.finite(distances))) stop("Nonfinite centroid distance.")
  first <- max.col(-distances, ties.method = "first")
  nearest <- distances[cbind(seq_len(nrow(z)), first)]
  second <- apply(distances, 1, function(d) sort(d, partial = 2)[[2]])
  # Coincident centroids yield a zero/zero ratio: no separation, hence margin 0.
  margin <- ifelse(second == 0, 0, pmin(1, pmax(0, 1 - nearest / second)))
  result <- data.frame(cluster = labels[first], distance_to_centroid = nearest,
    second_centroid_distance = second, separation_margin = margin,
    distance_threshold = NA_real_, margin_threshold = NA_real_,
    low_margin = NA, far_from_baseline = NA)
  if (!is.null(thresholds)) {
    if (!is.list(thresholds) || !identical(thresholds$schema_version, "part2-cluster-thresholds-v1") ||
        !identical(thresholds$labels, labels) || !identical(thresholds$distance_quantile, .95) ||
        !identical(thresholds$margin_quantile, .10) || !identical(thresholds$quantile_type, 7L) ||
        !identical(thresholds$margin_definition, "1_minus_nearest_over_second_distance_not_a_probability") ||
        !is.data.frame(thresholds$distance_thresholds) ||
        !all(c("cluster", "distance_threshold") %in% names(thresholds$distance_thresholds)) ||
        anyDuplicated(thresholds$distance_thresholds$cluster) ||
        !setequal(thresholds$distance_thresholds$cluster, labels) ||
        any(!is.finite(thresholds$distance_thresholds$distance_threshold) | thresholds$distance_thresholds$distance_threshold < 0) ||
        !is.numeric(thresholds$margin_threshold) || length(thresholds$margin_threshold) != 1L ||
        !is.finite(thresholds$margin_threshold) || thresholds$margin_threshold < 0 || thresholds$margin_threshold > 1) {
      stop("Invalid baseline assignment threshold contract.", call. = FALSE)
    }
    result$distance_threshold <- thresholds$distance_thresholds$distance_threshold[
      match(result$cluster, thresholds$distance_thresholds$cluster)]
    result$margin_threshold <- thresholds$margin_threshold
    result$low_margin <- result$separation_margin < result$margin_threshold
    result$far_from_baseline <- result$distance_to_centroid > result$distance_threshold
  }
  result
}

part2_cluster_fit_thresholds <- function(assignments, labels) {
  if (!is.character(labels) || !identical(labels, part2_cluster_labels(length(labels))) ||
      !is.data.frame(assignments) || !all(c("cluster", "distance_to_centroid", "separation_margin") %in% names(assignments))) {
    stop("Invalid baseline assignment/label schema.", call. = FALSE)
  }
  cluster <- part2_cluster_validate_labels(assignments$cluster, labels)
  distance <- assignments$distance_to_centroid; margin <- assignments$separation_margin
  if (!is.numeric(distance) || any(!is.finite(distance) | distance < 0) ||
      !is.numeric(margin) || any(!is.finite(margin) | margin < 0 | margin > 1) || !setequal(cluster, labels)) {
    stop("Baseline thresholds require finite assignments and every cluster.", call. = FALSE)
  }
  list(schema_version = "part2-cluster-thresholds-v1", labels = labels,
    distance_quantile = .95, margin_quantile = .10, quantile_type = 7L,
    margin_definition = "1_minus_nearest_over_second_distance_not_a_probability",
    distance_thresholds = data.frame(cluster = labels, distance_threshold = vapply(labels, function(label) {
      as.numeric(stats::quantile(distance[cluster == label], .95, type = 7, names = FALSE))
    }, numeric(1))),
    margin_threshold = as.numeric(stats::quantile(margin, .10, type = 7, names = FALSE)), training_n = length(cluster))
}

part2_cluster_kmeans <- function(z, k = 7L, seed = 2025L, nstart = 100L, iter_max = 500L) {
  z <- part2_cluster_matrix(z)
  labels <- part2_cluster_labels(k)
  integer_scalar <- function(x, minimum) is.numeric(x) && length(x) == 1L &&
    !is.na(x) && is.finite(x) && x >= minimum && x <= .Machine$integer.max && x == as.integer(x)
  if (!integer_scalar(seed, 0) || !integer_scalar(nstart, 1) || !integer_scalar(iter_max, 1) ||
      nrow(z) <= k || nrow(unique(z)) < k) stop("Invalid k-means seed/settings or insufficient distinct training rows.")
  # Explicit RNG convention plus restoration makes this seeded helper reproducible
  # without consuming the caller's random stream (including sampling diagnostics).
  old_kind <- RNGkind()
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  on.exit({
    do.call(RNGkind, as.list(old_kind))
    if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv) else
      if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) rm(".Random.seed", envir = .GlobalEnv)
  }, add = TRUE)
  set.seed(as.integer(seed), kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  warnings <- character()
  fit <- withCallingHandlers(stats::kmeans(z, centers = as.integer(k), nstart = as.integer(nstart),
    iter.max = as.integer(iter_max), algorithm = "Lloyd"), warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")
    })
  if (any(!is.finite(fit$centers)) || any(fit$size == 0L) || !is.finite(fit$tot.withinss) ||
      fit$iter > iter_max || (!is.null(fit$ifault) && fit$ifault != 0L)) {
    stop("Selected historical k-means solution is invalid or did not converge.", call. = FALSE)
  }
  centers <- fit$centers
  rownames(centers) <- labels
  assignment <- part2_cluster_assign(z, centers)
  raw_cluster <- labels[fit$cluster]
  # This explicit guard keeps baseline self-assignment exact. Rare exact ties
  # with a different Lloyd allocation require a reviewed solution, not silent
  # post-fit reassignment that would change the centroid means or cluster sizes.
  if (!identical(assignment$cluster, raw_cluster)) {
    stop("Lloyd training allocation differs from deterministic nearest-centroid self-assignment.", call. = FALSE)
  }
  list(schema_version = "part2-cluster-kmeans-v1", k = as.integer(k), labels = labels,
    seed = as.integer(seed), nstart = as.integer(nstart), iter_max = as.integer(iter_max), algorithm = "Lloyd",
    centers = centers, cluster = raw_cluster, size = stats::setNames(fit$size, labels),
    tot.withinss = fit$tot.withinss, withinss = stats::setNames(fit$withinss, labels),
    betweenss = fit$betweenss, totss = fit$totss, iter = fit$iter,
    warnings = unique(warnings), fit = fit)
}

part2_cluster_ari <- function(a, b) {
  a <- part2_cluster_validate_labels(a, label = "first partition")
  b <- part2_cluster_validate_labels(b, label = "second partition")
  if (length(a) != length(b)) stop("ARI partitions must refer to the same ordered rows.")
  n <- length(a)
  if (n < 2L) return(1)
  contingency <- table(a, b)
  pairs <- function(x) x * (x - 1) / 2
  index <- sum(pairs(contingency)); rows <- sum(pairs(rowSums(contingency))); cols <- sum(pairs(colSums(contingency)))
  expected <- rows * cols / pairs(n)
  upper <- (rows + cols) / 2
  denominator <- upper - expected
  # Both one-block partitions and both all-singleton partitions are identical.
  if (denominator == 0) return(1)
  (index - expected) / denominator
}

part2_cluster_permutations <- function(values) {
  if (length(values) == 1L) return(matrix(values, nrow = 1L))
  do.call(rbind, lapply(values, function(first) {
    cbind(first, part2_cluster_permutations(values[values != first]))
  }))
}

part2_cluster_align <- function(reference, candidate, k = 7L) {
  labels <- part2_cluster_labels(k)
  if (k > 8L) stop("Exact permutation alignment is restricted to at most eight clusters.")
  reference <- part2_cluster_validate_labels(reference, labels, "reference partition")
  candidate <- part2_cluster_validate_labels(candidate, labels, "candidate partition")
  if (length(reference) != length(candidate)) stop("Alignment requires the same ordered shared hexes.")
  overlap <- table(reference = factor(reference, levels = labels), candidate = factor(candidate, levels = labels))
  permutations <- part2_cluster_permutations(seq_len(k))
  matches <- apply(permutations, 1, function(permutation) sum(overlap[cbind(permutation, seq_len(k))]))
  best <- which(matches == max(matches))
  # Permutations are enumerated lexicographically in candidate-label order.
  permutation <- permutations[best[1], ]
  mapping <- data.frame(candidate_cluster = labels, aligned_cluster = labels[permutation])
  list(mapping = mapping, aligned_cluster = mapping$aligned_cluster[match(candidate, mapping$candidate_cluster)],
    overlap_table = overlap, matched_hexes = as.integer(max(matches)), match_share = max(matches) / length(reference),
    optimal_permutation_count = length(best),
    tie_break_policy = "lexicographically_smallest_candidate_to_reference_label_permutation")
}

part2_cluster_transitions <- function(from, to, k = 7L) {
  labels <- part2_cluster_labels(k)
  from <- part2_cluster_validate_labels(from, labels, "origin labels", allow_empty = TRUE)
  to <- part2_cluster_validate_labels(to, labels, "destination labels", allow_empty = TRUE)
  if (length(from) != length(to)) stop("Transitions require the same ordered shared hexes.")
  counts <- table(from = factor(from, levels = labels), to = factor(to, levels = labels))
  result <- expand.grid(to_cluster = labels, from_cluster = labels, stringsAsFactors = FALSE)[c("from_cluster", "to_cluster")]
  result$hexes <- as.integer(counts[cbind(match(result$from_cluster, labels), match(result$to_cluster, labels))])
  result$from_total <- as.integer(rowSums(counts)[match(result$from_cluster, labels)])
  result$to_total <- as.integer(colSums(counts)[match(result$to_cluster, labels)])
  result$share_of_from <- ifelse(result$from_total > 0, result$hexes / result$from_total, NA_real_)
  result$share_of_to <- ifelse(result$to_total > 0, result$hexes / result$to_total, NA_real_)
  result$share_of_all <- if (length(from)) result$hexes / length(from) else NA_real_
  result$same_cluster <- result$from_cluster == result$to_cluster
  result
}

part2_cluster_profiles <- function(data, assignments, features, k = 7L) {
  labels <- part2_cluster_labels(k)
  assignments <- part2_cluster_validate_labels(assignments, labels, "profile assignments")
  x <- part2_cluster_feature_matrix(data, features)
  if (nrow(x) != length(assignments)) stop("Profile assignments must match the ordered feature rows.")
  do.call(rbind, lapply(labels, function(cluster) {
    do.call(rbind, lapply(features, function(feature) {
      values <- x[assignments == cluster, feature]
      data.frame(cluster = cluster, feature = feature, n = length(values),
        mean = if (length(values)) mean(values) else NA_real_,
        median = if (length(values)) stats::median(values) else NA_real_,
        sd = if (length(values) > 1L) stats::sd(values) else NA_real_,
        min = if (length(values)) min(values) else NA_real_,
        max = if (length(values)) max(values) else NA_real_)
    }))
  }))
}
