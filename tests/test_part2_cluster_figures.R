# Read-only checks for corrected figures, run only after visualization completes.
suppressPackageStartupMessages({library(dplyr); library(readr)})
root <- "figures/part2"
manifest <- jsonlite::read_json(file.path(root, "part2_cluster_figure_manifest.json"), simplifyVector = TRUE)
analysis <- jsonlite::read_json("output/part2/clusters/part2_cluster_run_manifest.json", simplifyVector = TRUE)
assignments <- readRDS("output/part2/clusters/part2_cluster_assignments.rds")
eligible <- readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
stopifnot(manifest$schema_version == 2L, manifest$status == "paired_cluster_static_figures_complete",
  manifest$measurement_version == "part2-fixed-components-v2",
  analysis$status == "historical_cluster_comparison_complete_v2",
  manifest$sample_cells == sum(eligible$common_comparison_ready), manifest$sample_cells == analysis$common_hexes,
  manifest$fixed_city_center_cells == sum(eligible$in_current_city_scope),
  manifest$excluded_city_center_cells == sum(eligible$in_current_city_scope & !eligible$common_comparison_ready),
  manifest$sample_cells + manifest$excluded_city_center_cells == manifest$fixed_city_center_cells,
  identical(manifest$cluster_labels, paste0("C", 1:7)), identical(manifest$labels_are_risk_ranks, FALSE),
  isTRUE(manifest$analysis_inputs_unchanged), isTRUE(manifest$prior_failed_part2_figures_overwritten))
for (kind in c("inputs", "outputs")) {
  files <- manifest[[kind]]
  stopifnot(nrow(files) > 0L, !anyDuplicated(files$path), all(file.exists(files$path)))
  current <- vapply(files$path, digest::digest, character(1), file = TRUE, algo = "sha256")
  stopifnot(identical(unname(current), files$sha256))
}
palette <- read_csv(file.path(root, "part2_cluster_palette.csv"), show_col_types = FALSE)
stopifnot(identical(palette$cluster, paste0("C", 1:7)),
  identical(palette$color, c("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#332288")),
  all(palette$meaning == "neutral cluster identity; not ordered risk"))
transitions <- read_csv(file.path(root, "part2_cluster_transition_plot_data.csv"), show_col_types = FALSE)
x <- assignments[assignments$common_comparison_ready, ]
stopifnot(nrow(transitions) == 49L, sum(transitions$hexes) == nrow(x),
  !anyDuplicated(transitions[c("cluster_2025", "cluster_2026_fixed")]))
for (i in seq_len(nrow(transitions))) {
  from <- x$cluster_2025 == transitions$cluster_2025[i]
  to <- x$cluster_2026_fixed == transitions$cluster_2026_fixed[i]
  stopifnot(transitions$hexes[i] == sum(from & to), transitions$baseline_cluster_hexes[i] == sum(from),
    abs(transitions$row_share[i] - sum(from & to) / sum(from)) < 1e-12)
}
stopifnot(sum(transitions$hexes[transitions$cluster_2025 != transitions$cluster_2026_fixed]) == sum(x$moved_fixed),
  all(abs(tapply(transitions$row_share, transitions$cluster_2025, sum) - 1) < 1e-12))
# Inspect PNG signature/IHDR directly, without changing or resampling images.
pngs <- file.path(root, c("part2_cluster_comparison_maps.png", "part2_cluster_transition_heatmap.png", "part2_cluster_profile_heatmap.png"))
dimensions <- matrix(c(3200L, 2000L, 2100L, 1800L, 3400L, 1500L), ncol = 2L, byrow = TRUE)
for (i in seq_along(pngs)) {
  bytes <- readBin(pngs[i], what = "raw", n = 24L)
  stopifnot(identical(as.integer(bytes[1:8]), c(137L, 80L, 78L, 71L, 13L, 10L, 26L, 10L)),
    rawToChar(bytes[13:16]) == "IHDR", file.info(pngs[i])$size > 10000)
  integer32 <- function(x) sum(as.integer(x) * 256^(3:0))
  stopifnot(identical(c(integer32(bytes[17:20]), integer32(bytes[21:24])), as.numeric(dimensions[i, ])))
}
cat("Corrected cluster figures validated for", nrow(x), "paired cells; all input/output hashes and PNG dimensions match.\n")
