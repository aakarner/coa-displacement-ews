# Static scientific figures for the paired, covered-subset cluster comparison.
# This script overwrites only the authorized figures/part2 artifacts. No Part 1 labels, risk
# ordering, model fitting, or interactive map/site updates are performed here.
suppressPackageStartupMessages({
  library(dplyr); library(ggplot2); library(readr); library(sf); library(tidyr)
})
source("R/pipeline.R")

figure_dir <- "figures/part2"
assignment_path <- "output/part2/clusters/part2_cluster_assignments.rds"
profile_path <- "output/part2/clusters/part2_cluster_profiles_long.csv"
grid_path <- "output/hex_grid.rds"
eligibility_path <- "output/part2/matrix/part2_eligibility_by_hex.rds"
paired_path <- "output/part2/matrix/part2_features_paired.rds"
cluster_manifest_path <- "output/part2/clusters/part2_cluster_run_manifest.json"
script_path <- "scripts/part2/visualize_cluster_comparison.R"
input_paths <- c(assignment_path, profile_path, grid_path, eligibility_path, paired_path,
  cluster_manifest_path, script_path, "R/pipeline.R")
before <- build_file_manifest(input_paths, require_all = TRUE, hash_files = TRUE)

assignments <- readRDS(assignment_path)
profiles <- read_csv(profile_path, show_col_types = FALSE)
grid <- readRDS(grid_path)
eligibility <- readRDS(eligibility_path)
paired <- readRDS(paired_path)
cluster_manifest <- jsonlite::read_json(cluster_manifest_path, simplifyVector = TRUE)
cluster_columns <- c("cluster_2025", "cluster_2026_fixed", "cluster_2026_refit_aligned")
cluster_levels <- paste0("C", 1:7)
solution_levels <- c("baseline_2025", "fixed_2026", "refit_2026_aligned")
solution_labels <- c(baseline_2025 = "2025 baseline\nOriginal cluster definitions",
  fixed_2026 = "2026 fixed assignment\nUnchanged 2025 definitions",
  refit_2026_aligned = "2026 aligned refit\nNew definitions, matched labels")
feature_labels <- c(
  rent_pressure_citywide_index = "Rent\npressure",
  demographic_vulnerability_index = "Demographic\nvulnerability",
  demolition_pressure_index = "Demolition\nactivity",
  eviction_pressure_index = "Recorded\neviction filings",
  sr_311_pressure_index = "Selected\n311 calls",
  ownership_pressure_index = "Corporate\nownership",
  amenity_change_index = "Amenity\nopenings")

stopifnot(nrow(grid) == 7027L, is.integer(grid$hex_id),
  all(c("hex_id", "common_comparison_ready", cluster_columns, "moved_fixed") %in% names(assignments)),
  !anyDuplicated(assignments$hex_id), setequal(assignments$hex_id, grid$hex_id),
  !anyDuplicated(eligibility$hex_id), setequal(eligibility$hex_id, grid$hex_id),
  all(c("hex_id", "in_current_city_scope", "common_comparison_ready") %in% names(eligibility)))
assignments <- assignments[match(grid$hex_id, assignments$hex_id), ]
eligibility <- eligibility[match(grid$hex_id, eligibility$hex_id), ]
stopifnot(identical(assignments$common_comparison_ready, eligibility$common_comparison_ready),
  !anyNA(assignments$common_comparison_ready), !anyNA(eligibility$in_current_city_scope),
  sum(assignments$common_comparison_ready) > length(cluster_levels),
  sum(assignments$common_comparison_ready) == cluster_manifest$common_hexes,
  sum(eligibility$in_current_city_scope) == 6060L,
  all(!assignments$common_comparison_ready | eligibility$in_current_city_scope))
eligible <- assignments %>% filter(common_comparison_ready)
paired_sample <- paired[paired$hex_id %in% eligible$hex_id, ]
stopifnot(nrow(paired_sample) == 2L * nrow(eligible),
  !anyDuplicated(paired_sample[c("hex_id", "analysis_as_of_date")]),
  identical(cluster_manifest$status, "historical_cluster_comparison_complete_v2"),
  all(paired_sample$measurement_version == "part2-fixed-components-v2"),
  all(paired_sample$all_required_components_available),
  identical(cluster_manifest$features, names(feature_labels)))
for (index in names(feature_labels)) {
  available <- paired_sample[[paste0(index, "_terms_available")]]
  total <- paired_sample[[paste0(index, "_terms_total")]]
  stopifnot(length(available) == nrow(paired_sample), length(total) == nrow(paired_sample),
    all(is.finite(paired_sample[[index]])), all(is.finite(total) & total > 0),
    all(available == total), length(unique(total)) == 1L)
}
for (column in cluster_columns) {
  stopifnot(all(eligible[[column]] %in% cluster_levels), !anyNA(eligible[[column]]),
    all(is.na(assignments[[column]][!assignments$common_comparison_ready])))
}
stopifnot(all(eligible$moved_fixed == (eligible$cluster_2025 != eligible$cluster_2026_fixed)),
  all(c("solution", "cluster", "feature", "mean_z") %in% names(profiles)),
  setequal(profiles$solution, solution_levels), setequal(profiles$cluster, cluster_levels),
  setequal(profiles$feature, names(feature_labels)),
  nrow(profiles) == length(solution_levels) * length(cluster_levels) * length(feature_labels),
  nrow(distinct(profiles, solution, cluster, feature)) == nrow(profiles),
  is.numeric(profiles$mean_z), all(is.finite(profiles$mean_z) | is.na(profiles$mean_z)),
  all(is.finite(profiles$mean_z[profiles$solution != "fixed_2026"])))

palette <- c(C1 = "#0072B2", C2 = "#E69F00", C3 = "#009E73", C4 = "#D55E00",
  C5 = "#CC79A7", C6 = "#56B4E9", C7 = "#332288")
excluded_label <- "Excluded city cell"
map_palette <- c(palette, setNames("#D9DEE3", excluded_label))
ink <- "#172B4D"; secondary <- "#52606D"
base_theme <- theme_minimal(base_size = 11, base_family = "sans") + theme(
  plot.background = element_rect(fill = "white", color = NA),
  plot.title = element_text(face = "bold", color = ink, size = 18, margin = margin(b = 7)),
  plot.subtitle = element_text(color = secondary, size = 11, lineheight = 1.15, margin = margin(b = 12)),
  plot.caption = element_text(color = secondary, size = 9, hjust = 0, lineheight = 1.15, margin = margin(t = 12)),
  strip.text = element_text(face = "bold", color = ink, size = 11, lineheight = 1.1, margin = margin(b = 9)),
  panel.grid = element_blank(), legend.title = element_text(face = "bold", color = ink),
  legend.text = element_text(color = secondary),
  plot.margin = margin(16, 18, 14, 18))
sample_text <- paste0(format(nrow(eligible), big.mark = ","), " common eligible cells")
excluded_count <- sum(eligibility$in_current_city_scope & !eligibility$common_comparison_ready)

# All panels use exactly the same full-hex footprint, map extent, and palette.
# Gray cells are the remaining fixed City-center grid cells, not a low-risk class.
map_attributes <- assignments %>% select(hex_id, common_comparison_ready, all_of(cluster_columns)) %>%
  mutate(in_current_city_scope = eligibility$in_current_city_scope) %>%
  filter(in_current_city_scope) %>%
  pivot_longer(all_of(cluster_columns), names_to = "assignment", values_to = "cluster") %>%
  mutate(solution = factor(solution_levels[match(assignment, cluster_columns)], levels = solution_levels),
    map_class = factor(if_else(common_comparison_ready, cluster, excluded_label), levels = names(map_palette)))
map_grid <- st_transform(grid %>% select(hex_id), 3083)
map_data <- map_grid %>% inner_join(map_attributes, by = "hex_id", relationship = "one-to-many")
stopifnot(nrow(map_data) == 6060L * 3L)
maps <- ggplot(map_data) +
  geom_sf(aes(fill = map_class), color = NA, linewidth = 0) +
  facet_wrap(vars(solution), nrow = 1, labeller = as_labeller(solution_labels)) +
  scale_fill_manual(values = map_palette, breaks = names(map_palette), drop = FALSE, name = NULL) +
  coord_sf(datum = NA, expand = FALSE) + base_theme +
  theme(axis.text = element_blank(), axis.title = element_blank(), axis.ticks = element_blank(),
    panel.spacing = grid::unit(1.2, "lines"), legend.position = "bottom", legend.key.height = grid::unit(4, "mm")) +
  guides(fill = guide_legend(nrow = 1, override.aes = list(color = NA))) +
  labs(title = "One year of new data: unchanged definitions versus a new fit",
    subtitle = paste0(sample_text, " at April 1, 2025 and April 1, 2026. Colors identify clusters, not a risk ranking."),
    caption = paste0("Gray: ", format(excluded_count, big.mark = ","), " fixed City-center cells excluded from the paired sample. Whole-hex geometry is held fixed.\n",
      "Fixed assignment measures movement against 2025 definitions; refit labels maximize overlap with the coeval fixed-2026 assignments.\n",
      "This retrospective comparison describes a covered subset of places, not individual displacement."))

transitions <- eligible %>% count(cluster_2025, cluster_2026_fixed, name = "hexes") %>%
  complete(cluster_2025 = cluster_levels, cluster_2026_fixed = cluster_levels, fill = list(hexes = 0L)) %>%
  group_by(cluster_2025) %>% mutate(baseline_cluster_hexes = sum(hexes), row_share = hexes / baseline_cluster_hexes) %>%
  ungroup() %>% mutate(baseline = factor(cluster_2025, levels = rev(cluster_levels)),
    later = factor(cluster_2026_fixed, levels = cluster_levels),
    cell_label = paste0(format(hexes, big.mark = ",", trim = TRUE), "\n", sprintf("%.1f%%", 100 * row_share)),
    label_color = if_else(row_share >= .55, "white", ink))
stopifnot(sum(transitions$hexes) == nrow(eligible), all(is.finite(transitions$row_share)))
retained_share <- mean(!eligible$moved_fixed)
transition_plot <- ggplot(transitions, aes(x = later, y = baseline)) +
  geom_tile(aes(fill = row_share), color = "white", linewidth = .8) +
  geom_tile(data = filter(transitions, cluster_2025 == cluster_2026_fixed), fill = NA, color = ink, linewidth = .8) +
  geom_text(aes(label = cell_label, color = label_color), size = 3.6, lineheight = 1.15) +
  scale_color_identity() +
  scale_fill_gradient(low = "#F2F6FA", high = "#155A88", limits = c(0, 1),
    labels = scales::label_percent(accuracy = 1), name = "Share of\n2025 cluster") +
  scale_x_discrete(drop = FALSE, position = "top") + scale_y_discrete(drop = FALSE) +
  coord_fixed() + base_theme + theme(axis.text = element_text(color = ink, size = 11),
    axis.title = element_text(color = secondary), legend.position = "right") +
  labs(title = "Where cells move under unchanged cluster definitions",
    subtitle = paste0(sprintf("%.1f%%", 100 * retained_share), " keep their 2025 cluster; ",
      sprintf("%.1f%%", 100 * (1 - retained_share)), " change. ", sample_text, "."),
    x = "2026 assignment to fixed 2025 clusters", y = "2025 baseline cluster",
    caption = "Each tile shows cell count and percentage of its 2025 row. Outlined diagonal tiles retain the same cluster.\nNumbers describe covered geographic cells, not people or confirmed displacement.")

profiles <- profiles %>% mutate(solution = factor(solution, levels = solution_levels),
  cluster = factor(cluster, levels = rev(cluster_levels)), feature = factor(feature, levels = names(feature_labels)))
color_limit <- max(1, ceiling(max(abs(profiles$mean_z), na.rm = TRUE) * 2) / 2)
profiles <- profiles %>% mutate(cell_label = if_else(is.na(mean_z), "—", sprintf("%.1f", if_else(abs(mean_z) < .05, 0, mean_z))),
  label_color = if_else(!is.na(mean_z) & abs(mean_z) > .6 * color_limit, "white", ink))
profile_plot <- ggplot(profiles, aes(x = feature, y = cluster)) +
  geom_tile(aes(fill = mean_z), color = "white", linewidth = .7) +
  geom_text(aes(label = cell_label, color = label_color), size = 3.2) +
  facet_wrap(vars(solution), nrow = 1, labeller = as_labeller(solution_labels)) +
  scale_color_identity() +
  scale_fill_gradient2(low = "#2166AC", mid = "#FAFAFA", high = "#B35806", midpoint = 0,
    limits = c(-color_limit, color_limit), na.value = "#D9DEE3", name = "2025 standard-\ndeviation units") +
  scale_x_discrete(labels = feature_labels, drop = FALSE) + scale_y_discrete(drop = FALSE) +
  base_theme + theme(axis.text.x = element_text(color = secondary, size = 9, angle = 40, hjust = 1),
    axis.text.y = element_text(color = ink, size = 10),
    panel.spacing = grid::unit(1.25, "lines"), legend.position = "bottom") +
  labs(title = "How cluster profiles compare on the same 2025 scale",
    subtitle = "Baseline centers, 2026 fixed-group means, and aligned 2026 refit centers. All seven indices use 2025 mean/SD standardization.",
    x = NULL, y = NULL,
    caption = "0 is the 2025 sample average; positive/negative values are above/below that average. Colors here show feature values, not cluster identity.\nC1–C7 are neutral labels from this corrected fit, not risk ranks or the earlier fit's cluster meanings. Each cell contributes equally.\nAll indices require complete fixed recipes at both dates. Rent uses one reliable block-group or tract level across all six vintages.\nProfiles describe the covered paired subset, not individual displacement; matching refit labels does not make the cluster definitions identical.")

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
figure_paths <- file.path(figure_dir, c("part2_cluster_comparison_maps.png", "part2_cluster_transition_heatmap.png", "part2_cluster_profile_heatmap.png"))
plots <- list(maps, transition_plot, profile_plot)
dimensions <- data.frame(width = c(16, 10.5, 17), height = c(10, 9, 7.5))
for (i in seq_along(plots)) {
  ggsave(figure_paths[i], plots[[i]], device = ragg::agg_png, width = dimensions$width[i], height = dimensions$height[i],
    units = "in", dpi = 200, bg = "white", limitsize = TRUE)
}
palette_path <- file.path(figure_dir, "part2_cluster_palette.csv")
write_csv(data.frame(cluster = cluster_levels, color = unname(palette), meaning = "neutral cluster identity; not ordered risk"), palette_path)
transition_path <- file.path(figure_dir, "part2_cluster_transition_plot_data.csv")
write_csv(select(transitions, cluster_2025, cluster_2026_fixed, hexes, baseline_cluster_hexes, row_share), transition_path)
after <- build_file_manifest(input_paths, require_all = TRUE, hash_files = TRUE)
stopifnot(identical(before$path, after$path), identical(before$sha256, after$sha256))
manifest <- list(schema_version = 2L, status = "paired_cluster_static_figures_complete",
  measurement_version = "part2-fixed-components-v2",
  created_at_utc = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
  sample_cells = nrow(eligible), fixed_city_center_cells = sum(eligibility$in_current_city_scope),
  excluded_city_center_cells = excluded_count, cluster_labels = cluster_levels,
  labels_are_risk_ranks = FALSE, alignment = "maximize shared-cell overlap with coeval fixed-2026 assignments, not feature-risk order",
  maps = "same full-hex geometry, City-center background, extent, sample, and categorical palette across all three panels",
  transition = "2025 to fixed-definition 2026; counts and within-baseline-cluster percentages",
  profiles = "2025 mean/SD units throughout; baseline centers, 2026 fixed-group means, aligned 2026 refit centers",
  measurement = "complete fixed component recipes at both dates; rent block-group-or-tract level selected once across all six vintages",
  prior_failed_part2_figures_overwritten = TRUE,
  interpretation = "retrospective covered-subset place-level comparison, not individual displacement",
  analysis_inputs_unchanged = TRUE, dpi = 200L, figure_dimensions_inches = dimensions,
  inputs = before, outputs = build_file_manifest(c(figure_paths, palette_path, transition_path), require_all = TRUE, hash_files = TRUE),
  runtime = list(R = R.version.string, ggplot2 = as.character(packageVersion("ggplot2")),
    sf = as.character(packageVersion("sf")), ragg = as.character(packageVersion("ragg"))))
jsonlite::write_json(manifest, file.path(figure_dir, "part2_cluster_figure_manifest.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")
cat("Saved three paired-cluster scientific figures under figures/part2/.\n")
