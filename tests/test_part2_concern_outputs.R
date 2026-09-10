# Independent, read-only integration audit. Run from the repository root after
# analyze_cluster_concern.R; do not source its implementation or scoring helpers.
suppressPackageStartupMessages(library(readr))

checks <- 0L
check <- function(ok, label) {
  if (length(ok) == 0L || anyNA(ok) || !all(ok)) stop(label, call. = FALSE)
  checks <<- checks + 1L
}
equal <- function(actual, expected, label, tolerance = 1e-10) {
  check(isTRUE(all.equal(unname(actual), unname(expected), tolerance = tolerance,
    check.attributes = FALSE)), label)
}
read_table <- function(name) read_csv(file.path(root, paste0("part2_", name, ".csv")),
  show_col_types = FALSE)
compare_table <- function(actual, expected, keys, label) {
  check(setequal(names(actual), names(expected)), paste(label, "columns"))
  check(nrow(actual) == nrow(expected), paste(label, "rows"))
  check(!anyDuplicated(actual[keys]) && !anyDuplicated(expected[keys]), paste(label, "unique keys"))
  order_by <- function(x) do.call(order, unname(as.list(x[keys])))
  actual <- actual[order_by(actual), names(expected), drop = FALSE]
  expected <- expected[order_by(expected), , drop = FALSE]
  for (name in names(expected)) equal(actual[[name]], expected[[name]], paste(label, name))
}
hash_count <- 0L
verify_manifest_files <- function(files, label) {
  check(nrow(files) > 0L && !anyDuplicated(files$path), paste(label, "unique files"))
  check(all(file.exists(files$path)), paste(label, "files exist"))
  hashes <- vapply(files$path, digest::digest, character(1), file = TRUE, algo = "sha256")
  check(identical(unname(hashes), files$sha256), paste(label, "SHA-256 pins"))
  equal(as.numeric(file.info(files$path)$size), as.numeric(files$size), paste(label, "file sizes"), 0)
  hash_count <<- hash_count + nrow(files)
}

root <- "output/part2/interpretation"
config_path <- "config/part2_cluster_interpretation.json"
model_path <- "output/part2/clusters/part2_cluster_models.rds"
source_path <- "output/part2/clusters/part2_cluster_assignments.rds"
eviction_path <- "output/part2/evictions/eviction_features_paired.rds"
config <- jsonlite::fromJSON(config_path)
model <- readRDS(model_path)
source_assignments <- readRDS(source_path)
assignments <- readRDS(file.path(root, "part2_concern_assignments.rds"))
eligibility <- readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
manifest <- jsonlite::fromJSON(file.path(root, "manifest.json"))
cluster_manifest <- jsonlite::fromJSON("output/part2/clusters/part2_cluster_run_manifest.json")
matrix_manifest <- jsonlite::fromJSON("output/part2/matrix/part2_matrix_run_manifest.json")
local_inputs <- c(config_path, model_path, source_path, eviction_path,
  "output/part2/matrix/part2_eligibility_by_hex.rds", "output/corporate_ownership_by_hex.rds",
  "output/part2/acs/acs_features_paired.rds", "output/part2/demolitions/demolition_features_paired.rds")
before_hashes <- vapply(local_inputs, digest::digest, character(1), file = TRUE, algo = "sha256")

# These are the explicitly reviewed human categories. C numbers are nominal;
# their numerical order is not used to calculate either rank or direction.
approved_rank <- c(C3 = 1L, C6 = 2L, C2 = 2L, C5 = 3L, C4 = 1L, C1 = 4L, C7 = 2L)
tier_names <- c("Low", "Moderate", "High", "Very high")
mapping <- config$clusters
check(config$schema_version == 1L && config$measurement_version == "part2-fixed-components-v2",
  "reviewed mapping uses corrected measurement contract")
check(model$schema_version == "part2-historical-clusters-v2" &&
  model$measurement_version == config$measurement_version && model$k == 7L,
  "corrected seven-cluster model")
check(!anyDuplicated(mapping$cluster) && setequal(mapping$cluster, names(approved_rank)) &&
  setequal(mapping$cluster, rownames(model$baseline$centers)), "exactly seven reviewed cluster identities")
equal(mapping$concern_rank, approved_rank[mapping$cluster], "approved human category ranks", 0)
equal(mapping$concern, tier_names[approved_rank[mapping$cluster]], "approved human category names", 0)
check(all(nzchar(mapping$profile)) && all(nzchar(mapping$rationale)) && nzchar(config$rationale),
  "each interpretation includes a profile and rationale")
check(config$baseline_centers_sha256 == digest::digest(model$baseline$centers, algo = "sha256") &&
  manifest$baseline_centers_sha256 == config$baseline_centers_sha256, "interpretation pins exact baseline centers")
compare_table(read_table("cluster_interpretation"), mapping, "cluster", "published interpretation")
check(identical(model$risk_ranking, FALSE) && identical(manifest$automatic_cluster_ids_are_ordinal, FALSE) &&
  identical(manifest$qualitative_categories_are_ordered, TRUE) &&
  identical(manifest$concern_is_probability, FALSE), "nominal identities versus ordered, nonprobabilistic interpretation")
check(manifest$status == "qualitative_concern_interpretation_complete" &&
  identical(manifest$labels_reviewed_against_corrected_profiles, TRUE) &&
  identical(manifest$mapping_rationale, config$rationale) && length(manifest$limitations) >= 4L,
  "interpretation completion and substantive-review limitations")
check(cluster_manifest$status == "historical_cluster_comparison_complete_v2" &&
  matrix_manifest$status == "paired_seven_feature_matrix_complete_v2", "corrected upstream source runs")

# The interpretation is additive: it must not change any source column or row.
check(identical(assignments$hex_id, source_assignments$hex_id) && is.integer(assignments$hex_id) &&
  !anyDuplicated(assignments$hex_id), "full-grid integer IDs and source order preserved")
for (name in names(source_assignments)) check(identical(assignments[[name]], source_assignments[[name]]),
  paste("unchanged source column", name))
equal(assignments$hex_id, eligibility$hex_id, "same full-grid eligibility IDs", 0)
equal(assignments$common_comparison_ready, eligibility$common_comparison_ready, "same paired inclusion mask", 0)
included <- assignments$common_comparison_ready
check(!anyNA(included) && nrow(assignments) == cluster_manifest$audit_hexes, "full audit grid")
n <- sum(included)
check(n == manifest$common_hexes && n == cluster_manifest$common_hexes &&
  n == matrix_manifest$common_eligible_hexes && n == length(model$training_hex_ids), "common cohort conserved")
check(identical(assignments$hex_id[included], model$training_hex_ids), "same fitted training cells")
expected <- source_assignments
for (snapshot in c("2025", "2026_fixed", "2026_refit_aligned")) {
  ranks <- unname(approved_rank[source_assignments[[paste0("cluster_", snapshot)]]])
  expected[[paste0("concern_rank_", snapshot)]] <- ranks
  expected[[paste0("concern_", snapshot)]] <- tier_names[ranks]
  equal(assignments[[paste0("concern_rank_", snapshot)]], ranks, paste(snapshot, "mapped rank"), 0)
  equal(assignments[[paste0("concern_", snapshot)]], tier_names[ranks], paste(snapshot, "mapped category"), 0)
}
delta <- expected$concern_rank_2026_fixed - expected$concern_rank_2025
direction <- rep(NA_character_, nrow(expected))
direction[included & delta > 0] <- "Higher"
direction[included & delta < 0] <- "Lower"
direction[included & delta == 0] <- "Same tier"
size <- rep(NA_character_, nrow(expected))
size[included & abs(delta) >= 2] <- "Two or more tiers"
size[included & abs(delta) == 1] <- "One tier"
size[included & delta == 0] <- "Different cluster, same tier"
size[included & !expected$moved_fixed] <- "Same cluster"
equal(assignments$concern_rank_change, delta, "later minus earlier approved ordinal tier", 0)
equal(assignments$concern_direction, direction, "independent temporal concern direction", 0)
equal(assignments$transition_size, size, "independent transition size", 0)
new_columns <- setdiff(names(assignments), names(source_assignments))
check(length(new_columns) == 9L && all(vapply(assignments[!included, new_columns],
  function(x) all(is.na(x)), logical(1))), "all excluded derived fields remain unknown, not Low or Same")
check(all(is.finite(delta[included])) && all(abs(delta[included]) <= 3), "ordinal steps are bounded")
csv_assignments <- read_table("concern_assignments")
compare_table(csv_assignments, as.data.frame(assignments), "hex_id", "RDS/CSV assignment parity")

# Independently associate the source assignments with the frozen model. This
# also prevents an interpretation of different/stale labels with matching IDs.
matrices <- lapply(c("2025-04-01", "2026-04-01"), function(date)
  readRDS(paste0("output/part2/matrix/part2_analysis_matrix_", date, ".rds")))
for (m in matrices) check(identical(m$hex_id, model$training_hex_ids) &&
  all(is.finite(as.matrix(m[model$features]))), "finite complete fitted matrix and training order")
baseline_values <- as.matrix(matrices[[1]][model$features])
equal(model$scaling$center, colMeans(baseline_values), "model scaling means from the same baseline")
equal(model$scaling$scale, apply(baseline_values, 2, sd), "model scaling sample SDs from the same baseline")
z <- lapply(matrices, function(m) sweep(sweep(as.matrix(m[model$features]), 2,
  model$scaling$center, "-"), 2, model$scaling$scale, "/"))
nearest <- function(values, centers) {
  distances <- vapply(seq_len(nrow(centers)), function(i)
    rowSums(sweep(values, 2, centers[i, ], "-")^2), numeric(nrow(values)))
  rownames(centers)[max.col(-distances, ties.method = "first")]
}
equal(assignments$cluster_2025[included], nearest(z[[1]], model$baseline$centers), "source baseline nearest centers", 0)
equal(assignments$cluster_2026_fixed[included], nearest(z[[2]], model$baseline$centers), "source later frozen assignment", 0)
later_raw <- nearest(z[[2]], model$later_refit$centers)
equal(assignments$cluster_2026_refit_raw[included], later_raw, "source later refit assignment", 0)
aligned <- model$alignment$mapping$aligned_cluster[match(later_raw, model$alignment$mapping$candidate_cluster)]
equal(assignments$cluster_2026_refit_aligned[included], aligned, "source coeval label alignment", 0)
check(all(assignments$moved_fixed[included] ==
  (assignments$cluster_2025[included] != assignments$cluster_2026_fixed[included])), "movement means changed cluster identity")

x <- assignments[included, ]
units <- readRDS("output/corporate_ownership_by_hex.rds")
fixed_units <- units$residential_units[match(x$hex_id, units$hex_id)]
equal(x$residential_units, fixed_units, "unchanged canonical promoted residential units")
check(all(is.finite(fixed_units) & fixed_units >= 20) && all(x$in_current_city_scope), "fixed units floor and city scope")
from <- unname(approved_rank[x$cluster_2025])
to <- unname(approved_rank[x$cluster_2026_fixed])
change <- to - from
moved <- x$cluster_2025 != x$cluster_2026_fixed
count_table <- as.data.frame(table(from = factor(from, levels = 1:4), to = factor(to, levels = 1:4)))
count_table <- count_table[count_table$Freq > 0, ]
expected_transitions <- data.frame(concern_2025 = tier_names[as.integer(count_table$from)],
  concern_2026_fixed = tier_names[as.integer(count_table$to)], concern_rank_2025 = as.integer(count_table$from),
  concern_rank_2026_fixed = as.integer(count_table$to), hexes = count_table$Freq, share_of_all = count_table$Freq / n)
transitions <- read_table("concern_transitions")
compare_table(transitions, expected_transitions, c("concern_rank_2025", "concern_rank_2026_fixed"), "tier transition counts")
check(sum(transitions$hexes) == n && sum(transitions$hexes[transitions$concern_rank_2025 ==
  transitions$concern_rank_2026_fixed]) == sum(change == 0), "transition table conservation")

groups <- list(Higher = which(change > 0), Lower = which(change < 0), `Same tier` = which(change == 0))
expected_directions <- do.call(rbind, lapply(names(groups), function(label) {
  i <- groups[[label]]
  data.frame(concern_direction = label, hexes = length(i), fixed_residential_units = sum(fixed_units[i]),
    share_of_all = length(i) / n, fixed_units_share = sum(fixed_units[i]) / sum(fixed_units))
}))
directions <- read_table("concern_directions")
compare_table(directions, expected_directions, "concern_direction", "direction counts and fixed unit weights")
equal(sum(directions$fixed_residential_units), sum(fixed_units), "all fixed units counted exactly once")
equal(c(sum(directions$share_of_all), sum(directions$fixed_units_share)), c(1, 1), "direction shares sum to one")
expected_sizes <- as.data.frame(table(transition_size = size[included]), stringsAsFactors = FALSE)
names(expected_sizes)[2] <- "hexes"
expected_sizes$transition_size <- as.character(expected_sizes$transition_size)
expected_sizes$share_of_all <- expected_sizes$hexes / n
sizes <- read_table("concern_transition_sizes")
compare_table(sizes, expected_sizes, "transition_size", "transition size partition")
expected_steps <- data.frame(concern_rank_change = sort(unique(change)))
expected_steps$hexes <- vapply(expected_steps$concern_rank_change, function(v) sum(change == v), integer(1))
expected_steps$share_of_all <- expected_steps$hexes / n
compare_table(read_table("concern_step_distribution"), expected_steps, "concern_rank_change", "signed step distribution")
check(sum(moved & change == 0) + sum(change != 0) == sum(moved) &&
  sum(!moved) + sum(moved & change == 0) == sum(change == 0), "same-tier changes are not falsely upward/downward")

# Obtain paired event counts directly, not from the produced bands or annual
# panels. The maximum is over the two recent windows, not their sum.
evictions <- readRDS(eviction_path)
dates <- as.Date(c("2025-04-01", "2026-04-01"))
check(identical(sort(unique(evictions$analysis_as_of_date)), dates) &&
  !anyDuplicated(evictions[c("hex_id", "analysis_as_of_date")]), "unique exact-date eviction records")
paired_events <- lapply(dates, function(date) {
  y <- evictions[evictions$analysis_as_of_date == date, ]
  y[match(x$hex_id, y$hex_id), ]
})
for (y in paired_events) {
  check(identical(y$hex_id, x$hex_id), "one eviction record per included cell/date")
  equal(y$residential_units, fixed_units, "same fixed units in eviction source")
  for (column in c("eviction_cases_latest_12mo", "eviction_cases_previous_12mo"))
    check(all(is.finite(y[[column]]) & y[[column]] >= 0 & y[[column]] == floor(y[[column]])),
      paste("known nonnegative integer", column))
}
recent <- do.call(cbind, lapply(paired_events, function(y) y$eviction_cases_latest_12mo))
previous <- do.call(cbind, lapply(paired_events, function(y) y$eviction_cases_previous_12mo))
max_recent <- apply(recent, 1, max)
first_filing <- rowSums(previous == 0 & recent > 0) > 0
filing_band <- cut(max_recent, breaks = c(-Inf, 0, 1, 4, Inf), labels = c("0", "1", "2-4", "5+"))
unit_band <- cut(fixed_units, breaks = c(20, 50, 100, Inf), right = FALSE,
  labels = c("20-49", "50-99", "100+"))
check(!anyNA(filing_band) && !anyNA(unit_band), "all common cells have a supported diagnostic band")
band_groups <- split(seq_len(n), interaction(filing_band, unit_band, drop = TRUE))
expected_small <- do.call(rbind, lapply(band_groups, function(i) data.frame(
  filing_band = as.character(filing_band[i[1]]), unit_band = as.character(unit_band[i[1]]),
  hexes = length(i), moved = sum(moved[i]), moved_share = mean(moved[i]),
  higher = sum(change[i] > 0), lower = sum(change[i] < 0), first_filing_cells = sum(first_filing[i]))))
small <- read_table("small_count_movement")
compare_table(small, expected_small, c("filing_band", "unit_band"), "independent small-count diagnostic")
equal(colSums(as.data.frame(small[c("hexes", "moved", "higher", "lower", "first_filing_cells")])),
  c(n, sum(moved), sum(change > 0), sum(change < 0), sum(first_filing)), "small-count conservation", 0)
check(all(small$higher + small$lower <= small$moved & small$moved <= small$hexes &
  small$first_filing_cells <= small$hexes & small$moved_share >= 0 & small$moved_share <= 1) &&
  all(small$first_filing_cells[small$filing_band == "0"] == 0), "small-count bounds and zero-band invariant")

thresholds <- c(20, 50, 100)
expected_units <- do.call(rbind, lapply(thresholds, function(threshold) {
  i <- which(fixed_units >= threshold)
  data.frame(minimum_fixed_units = threshold, hexes = length(i), moved = sum(moved[i]),
    moved_share = mean(moved[i]), higher_share = mean(change[i] > 0), lower_share = mean(change[i] < 0),
    interpretation = "Subset of the same fitted model, not a refit with a different unit cutoff")
}))
unit_support <- read_table("unit_support_sensitivity")
compare_table(unit_support, expected_units, "minimum_fixed_units", "same-model unit threshold subsets")
unit_support <- unit_support[order(unit_support$minimum_fixed_units), ]
check(all(diff(unit_support$hexes) <= 0) && all(diff(unit_support$moved) <= 0) &&
  unit_support$hexes[1] == n && unit_support$moved[1] == sum(moved), "nested unit thresholds and full-cohort first row")
check(all(unit_support$higher_share + unit_support$lower_share <= unit_support$moved_share + 1e-12) &&
  all(unit_support$moved <= unit_support$hexes), "unit-subset directional and cluster movement bounds")

# Check the factual anchors in the written rationales, not the subjective
# judgment that those profiles deserve their reviewed qualitative category.
baseline_clusters <- x$cluster_2025
centers <- model$baseline$centers
signal_features <- setdiff(model$features, "demographic_vulnerability_index")
check(all(centers["C3", signal_features] < 0), "C3 six signal indices below their baseline means")
check(which.max(centers[, "sr_311_pressure_index"]) == match("C6", rownames(centers)) &&
  centers["C6", "demographic_vulnerability_index"] > 0, "C6 request activity and vulnerability anchor")
check(which.max(centers[, "ownership_pressure_index"]) == match("C2", rownames(centers)) &&
  centers["C2", "demographic_vulnerability_index"] > 0, "C2 ownership and vulnerability anchor")
demo <- readRDS("output/part2/demolitions/demolition_features_paired.rds")
demo <- demo[demo$analysis_as_of_date == dates[1], ]
demo <- demo[match(x$hex_id, demo$hex_id), ]
check(all(demo$demo_latest_24mo[baseline_clusters == "C5"] > 0) &&
  which.max(centers["C5", ]) == match("demolition_pressure_index", colnames(centers)), "C5 mapped recent permit and dominant demolition anchor")
rate <- 100 * recent[, 1] / fixed_units
rates_by_cluster <- tapply(rate, baseline_clusters, mean)
check(all(recent[baseline_clusters == "C1", 1] > 0) && round(rates_by_cluster["C1"], 1) == 15.4 &&
  all(round(rates_by_cluster[names(rates_by_cluster) != "C1"], 1) <= 1.8) &&
  mean(100 * (recent[baseline_clusters == "C1", 1] - previous[baseline_clusters == "C1", 1]) /
    fixed_units[baseline_clusters == "C1"]) > 0 && centers["C1", "demographic_vulnerability_index"] > 0,
  "C1 filing concentration, relative rate, positive change and vulnerability anchors")
acs <- readRDS("output/part2/acs/acs_features_paired.rds")
acs <- acs[acs$analysis_as_of_date == dates[1], ]
acs <- acs[match(x$hex_id, acs$hex_id), ]
check(mean(acs$acs_rent_current_real[baseline_clusters == "C4"]) > mean(acs$acs_rent_current_real) &&
  mean(acs$acs_rent_growth_recent_annualized_pct[baseline_clusters == "C4"]) > 0 &&
  which.min(centers[, "demographic_vulnerability_index"]) == match("C4", rownames(centers)),
  "C4 higher, rising rent and lowest vulnerability anchors")
check(which.max(centers[, "amenity_change_index"]) == match("C7", rownames(centers)), "C7 nearby amenity exposure anchor")

# Exact reviewed-run fixture supplements the recomputation above. A future
# model must obtain a new reviewed hash/mapping, not silently reuse these labels.
reviewed_hash <- "a0b242512ecc72f353fa7bec45653024e0a5fd0f1543f6d4d7d1c376e7f8a441"
if (identical(config$baseline_centers_sha256, reviewed_hash)) {
  equal(c(n, sum(moved), sum(change > 0), sum(change < 0), sum(change == 0)),
    c(2351, 822, 335, 262, 1754), "reviewed corrected-run count fixture", 0)
  equal(c(sum(moved & change == 0), sum(abs(change) == 1), sum(abs(change) >= 2)),
    c(225, 354, 243), "reviewed same-tier/one-tier/multitier fixture", 0)
}

# Pin both the immediate interpretation inputs and the upstream corrected model
# and canonical support. No production file is rewritten during this audit.
expected_inputs <- normalizePath(c(config_path, model_path, source_path, eviction_path,
  "R/pipeline.R", "scripts/part2/analyze_cluster_concern.R"))
check(setequal(manifest$inputs$path, expected_inputs), "complete immediate interpretation source registry")
check(setequal(manifest$outputs$path, normalizePath(list.files(root, full.names = TRUE,
  pattern = "[.](csv|rds)$"))) && nrow(manifest$outputs) == 9L, "complete nine-artifact output registry")
for (kind in c("inputs", "outputs")) {
  verify_manifest_files(manifest[[kind]], paste("interpretation", kind))
  verify_manifest_files(cluster_manifest[[kind]], paste("corrected cluster", kind))
  verify_manifest_files(matrix_manifest[[kind]], paste("corrected matrix", kind))
}
after_hashes <- vapply(local_inputs, digest::digest, character(1), file = TRUE, algo = "sha256")
check(identical(before_hashes, after_hashes), "test preserves every directly inspected source")
cat("Concern interpretation audit passed:", checks, "checks and", hash_count, "checksum pins;",
  n, "paired cells;", sum(change > 0), "higher,", sum(change < 0), "lower,", sum(change == 0),
  "same tier. Human mapping, source assignments, fixed-unit subsets and small-count diagnostics independently match.\n")
