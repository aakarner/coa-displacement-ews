################################################################################
# Illustrate the ACS Dasymetric Allocation Workflow
################################################################################
#
# Creates a conceptual, reproducible example of the production allocation:
#   1. ACS additive estimates originate in block groups.
#   2. 2020 Census blocks control their within-block-group distribution.
#   3. Appraisal-based residential evidence distributes each block to hexes.
#   4. Additive counts are divided among hexes, while medians are assigned from
#      the dominant residential block group with tract fallback when suppressed.
#
# The geometries and values below are synthetic. The allocation arithmetic
# follows the hierarchy documented in R/acs_dasymetric.R.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(sf)
  library(tidyr)
})

source(here::here("R", "utils.R"))

FIGURES_DIR <- here::here("figures")
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

bg_palette <- c("Block group A" = "#74A9CF", "Block group B" = "#F4A582")
median_palette <- c(
  "Block group A: $1,100" = "#2166AC",
  "Tract fallback: $1,350" = "#B2182B"
)

polygon_from_xy <- function(x, y, crs = 3857) {
  st_polygon(list(cbind(x, y))) %>%
    st_sfc(crs = crs)
}

integerize_to_total <- function(x, total) {
  integer_values <- floor(x)
  remainder <- as.integer(round(total - sum(integer_values)))
  if (remainder > 0L) {
    add_index <- order(x - integer_values, decreasing = TRUE)[
      seq_len(remainder)
    ]
    integer_values[add_index] <- integer_values[add_index] + 1L
  }
  integer_values
}

tract <- st_sf(
  tract_id = "Example tract",
  median_rent = 1350,
  geometry = polygon_from_xy(
    c(0, 12, 12, 0, 0),
    c(0, 0, 8, 8, 0)
  )
)

block_groups <- st_sf(
  bg_id = c("Block group A", "Block group B"),
  acs_population = c(1500, 1100),
  acs_median_rent = c(1100, NA_real_),
  block_population_control = c(1200, 850),
  block_housing_control = c(520, 390),
  geometry = st_sfc(
    st_polygon(list(cbind(
      c(0, 6.3, 5.7, 6.2, 0, 0),
      c(0, 0, 4, 8, 8, 0)
    ))),
    st_polygon(list(cbind(
      c(6.3, 12, 12, 6.2, 5.7, 6.3),
      c(0, 0, 8, 8, 4, 0)
    ))),
    crs = st_crs(tract)
  )
)

block_grid <- st_make_grid(tract, n = c(6, 4)) %>%
  st_sf(grid_id = seq_along(.), geometry = .)

blocks <- suppressWarnings(st_intersection(block_grid, block_groups)) %>%
  mutate(
    block_id = paste0("B", row_number()),
    center = st_point_on_surface(geometry),
    center_x = st_coordinates(center)[, 1],
    center_y = st_coordinates(center)[, 2],
    raw_population = case_when(
      bg_id == "Block group A" ~
        8 + 55 * exp(-((center_x - 3.1)^2 + (center_y - 5.4)^2) / 5),
      TRUE ~
        7 + 48 * exp(-((center_x - 9.0)^2 + (center_y - 2.6)^2) / 4.5)
    ),
    raw_housing = case_when(
      bg_id == "Block group A" ~
        6 + 40 * exp(-((center_x - 3.3)^2 + (center_y - 5.1)^2) / 5.5),
      TRUE ~
        5 + 34 * exp(-((center_x - 8.8)^2 + (center_y - 2.8)^2) / 5)
    )
  ) %>%
  st_drop_geometry() %>%
  group_by(bg_id) %>%
  mutate(
    block_population = raw_population / sum(raw_population) *
      first(block_population_control),
    block_housing_units = raw_housing / sum(raw_housing) *
      first(block_housing_control),
    block_population_display = integerize_to_total(
      block_population,
      first(block_population_control)
    )
  ) %>%
  ungroup() %>%
  select(
    block_id,
    bg_id,
    block_population,
    block_population_display,
    block_housing_units
  ) %>%
  left_join(
    suppressWarnings(st_intersection(block_grid, block_groups)) %>%
      mutate(block_id = paste0("B", row_number())) %>%
      select(block_id, geometry),
    by = "block_id"
  ) %>%
  st_as_sf()

hexes <- st_make_grid(
  tract,
  cellsize = 2.05,
  square = FALSE,
  flat_topped = TRUE
) %>%
  st_sf(hex_id = paste0("H", seq_along(.)), geometry = .) %>%
  filter(
    lengths(st_within(suppressWarnings(st_point_on_surface(.)), tract)) > 0
  )

support_points <- tibble(
  x = c(
    1.1, 1.7, 2.4, 2.9, 3.4, 3.8, 4.4, 4.9, 2.0, 3.2, 3.7,
    6.9, 7.5, 8.1, 8.8, 9.4, 10.0, 10.6, 11.1, 8.4, 9.6, 10.4
  ),
  y = c(
    5.0, 5.7, 4.9, 6.5, 5.8, 4.4, 5.2, 6.1, 2.2, 4.3, 5.4,
    6.6, 5.7, 3.4, 2.3, 3.0, 1.9, 2.8, 1.4, 6.5, 5.0, 4.2
  ),
  support_type = c(
    rep("Reported floor area", 9),
    "Unit-count fallback",
    "Parcel-count fallback",
    rep("Reported floor area", 8),
    "Unit-count fallback",
    rep("Reported floor area", 2)
  ),
  support_value = c(
    1100, 7200, 2400, 18000, 3600, 9500, 1500, 5100, 2600,
    3000, 1000,
    1300, 4800, 2200, 12500, 6800, 28000, 4100, 17000,
    5000, 1900, 3300
  )
) %>%
  st_as_sf(coords = c("x", "y"), crs = st_crs(tract), remove = FALSE) %>%
  st_join(blocks %>% select(block_id), join = st_within, left = FALSE) %>%
  st_join(hexes %>% select(hex_id), join = st_within, left = FALSE)

mixed_support_blocks <- support_points %>%
  st_drop_geometry() %>%
  distinct(block_id, support_type) %>%
  count(block_id, name = "support_basis_count") %>%
  filter(support_basis_count > 1L)

if (nrow(mixed_support_blocks) == 0L) {
  stop("Panel 3 must include at least one mixed-support Census block.", call. = FALSE)
}

block_point_fallback <- suppressWarnings(
  blocks %>%
    filter(!block_id %in% support_points$block_id) %>%
    st_point_on_surface()
) %>%
  st_join(hexes %>% select(hex_id), join = st_within, left = FALSE) %>%
  transmute(
    block_id,
    hex_id,
    x = st_coordinates(geometry)[, 1],
    y = st_coordinates(geometry)[, 2],
    support_type = "Census block point fallback",
    support_value = 1
  )

block_hex_support <- bind_rows(
  support_points %>%
    st_drop_geometry() %>%
    select(block_id, hex_id, support_type, support_value),
  block_point_fallback %>%
    st_drop_geometry() %>%
    select(block_id, hex_id, support_type, support_value)
) %>%
  group_by(block_id, hex_id) %>%
  summarise(
    support_value = sum(support_value),
    allocation_method = if_else(
      all(support_type == "Census block point fallback"),
      "Census block point fallback",
      "Residential parcel support"
    ),
    .groups = "drop"
  ) %>%
  group_by(block_id) %>%
  mutate(within_block_weight = support_value / sum(support_value)) %>%
  ungroup() %>%
  left_join(
    blocks %>%
      st_drop_geometry() %>%
      select(block_id, bg_id, block_population, block_housing_units),
    by = "block_id"
  ) %>%
  mutate(
    block_population_contribution =
      block_population * within_block_weight,
    block_housing_contribution =
      block_housing_units * within_block_weight
  )

hex_allocations <- block_hex_support %>%
  group_by(hex_id, bg_id) %>%
  summarise(
    population_control = sum(block_population_contribution),
    housing_control = sum(block_housing_contribution),
    .groups = "drop"
  ) %>%
  left_join(
    block_groups %>%
      st_drop_geometry() %>%
      select(bg_id, acs_population),
    by = "bg_id"
  ) %>%
  group_by(bg_id) %>%
  mutate(
    bg_population_weight = population_control / sum(population_control),
    allocated_population_component = acs_population * bg_population_weight
  ) %>%
  ungroup()

dominant_bg_supported <- hex_allocations %>%
  group_by(hex_id) %>%
  arrange(desc(housing_control), bg_id, .by_group = TRUE) %>%
  slice_head(n = 1) %>%
  ungroup() %>%
  transmute(
    hex_id,
    dominant_bg = bg_id
  )

dominant_bg_point_fallback <- suppressWarnings(
  hexes %>%
    filter(!hex_id %in% dominant_bg_supported$hex_id) %>%
    st_point_on_surface() %>%
    st_join(
      block_groups %>% select(bg_id),
      join = st_within,
      left = TRUE
    )
) %>%
  st_drop_geometry() %>%
  filter(!is.na(bg_id)) %>%
  transmute(
    hex_id,
    dominant_bg = bg_id
  )

dominant_bg <- bind_rows(
  dominant_bg_supported,
  dominant_bg_point_fallback
) %>%
  distinct(hex_id, .keep_all = TRUE) %>%
  mutate(
    median_source = if_else(
      dominant_bg == "Block group A",
      "Block group A: $1,100",
      "Tract fallback: $1,350"
    ),
    assigned_median_rent = if_else(dominant_bg == "Block group A", 1100, 1350)
  )

hex_results <- hexes %>%
  left_join(
    hex_allocations %>%
      group_by(hex_id) %>%
      summarise(
        allocated_population = sum(allocated_population_component),
        .groups = "drop"
      ),
    by = "hex_id"
  ) %>%
  left_join(dominant_bg, by = "hex_id") %>%
  mutate(
    allocated_population = replace_na(allocated_population, 0),
    median_source = factor(
      median_source,
      levels = names(median_palette)
    )
  )

if (
  any(
    abs(
      hex_allocations %>%
        st_drop_geometry() %>%
        group_by(bg_id) %>%
        summarise(weight_sum = sum(bg_population_weight), .groups = "drop") %>%
        pull(weight_sum) -
        1
    ) > 1e-8
  )
) {
  stop("Synthetic block-group allocation weights do not sum to one.", call. = FALSE)
}

if (
  abs(
    sum(hex_results$allocated_population) -
      sum(block_groups$acs_population)
  ) > 1e-6
) {
  stop("Synthetic hex allocations do not preserve ACS source totals.", call. = FALSE)
}

display_control_check <- blocks %>%
  st_drop_geometry() %>%
  group_by(bg_id) %>%
  summarise(
    displayed_total = sum(block_population_display),
    .groups = "drop"
  ) %>%
  left_join(
    block_groups %>%
      st_drop_geometry() %>%
      select(bg_id, block_population_control),
    by = "bg_id"
  )

if (
  any(
    display_control_check$displayed_total !=
      display_control_check$block_population_control
  )
) {
  stop("Displayed block controls do not sum to their stated totals.", call. = FALSE)
}

panel_theme <- theme_void(base_size = 11) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 12,
      color = "#1F2933",
      margin = margin(b = 3)
    ),
    plot.subtitle = element_text(
      size = 9,
      color = "#52606D",
      lineheight = 1.05,
      margin = margin(b = 8)
    ),
    plot.margin = margin(6, 8, 6, 8),
    legend.position = "none"
  )

p_source <- ggplot() +
  geom_sf(
    data = block_groups,
    aes(fill = bg_id),
    color = "white",
    linewidth = 1.2
  ) +
  geom_sf(data = tract, fill = NA, color = "#172B4D", linewidth = 1) +
  annotate(
    "text",
    x = 2.75,
    y = 5.15,
    label = "BLOCK GROUP A\nPopulation: 1,500\nMedian rent: $1,100",
    size = 3.8,
    fontface = "bold",
    lineheight = 1.08,
    color = "#172B4D"
  ) +
  annotate(
    "text",
    x = 9.0,
    y = 3.15,
    label = "BLOCK GROUP B\nPopulation: 1,100\nMedian rent: suppressed",
    size = 3.8,
    fontface = "bold",
    lineheight = 1.08,
    color = "#4A1D17"
  ) +
  annotate(
    "label",
    x = 8.9,
    y = 7.3,
    label = "TRACT MEDIAN RENT: $1,350",
    size = 3.1,
    fontface = "bold",
    linewidth = 0.2,
    label.padding = unit(0.18, "lines"),
    fill = "white",
    color = "#172B4D"
  ) +
  scale_fill_manual(values = bg_palette, guide = "none") +
  coord_sf(xlim = c(0, 12), ylim = c(0, 8), expand = FALSE) +
  labs(
    title = "1. Begin with ACS source geographies",
    subtitle = paste(
      "Most inputs come from block groups;",
      "tract medians are retained as fallbacks.",
      sep = "\n"
    )
  ) +
  panel_theme

block_labels <- blocks %>%
  mutate(
    label = number(block_population_display, accuracy = 1),
    geometry = st_point_on_surface(geometry)
  )

p_blocks <- ggplot() +
  geom_sf(
    data = blocks,
    aes(fill = block_population),
    color = "white",
    linewidth = 0.6
  ) +
  geom_sf(data = block_groups, fill = NA, color = "#344563", linewidth = 0.8) +
  geom_sf_text(
    data = block_labels,
    aes(label = label),
    size = 2.7,
    color = "#172B4D",
    check_overlap = TRUE
  ) +
  annotate(
    "label",
    x = 4.0,
    y = 0.65,
    label = "2020 block controls: A = 1,200; B = 850",
    size = 2.8,
    label.padding = unit(0.15, "lines"),
    linewidth = 0.15,
    fill = alpha("white", 0.9),
    color = "#172B4D"
  ) +
  scale_fill_gradient(
    low = "#F7F8FA",
    high = "#D95F59",
    guide = "none"
  ) +
  coord_sf(xlim = c(0, 12), ylim = c(0, 8), expand = FALSE) +
  labs(
    title = "2. Use Census blocks as controls",
    subtitle = paste(
      "2020 block counts determine within-group allocation shares;",
      "they need not equal the current ACS estimates.",
      sep = "\n"
    )
  ) +
  panel_theme

p_support <- ggplot() +
  geom_sf(data = blocks, fill = "#F4F5F7", color = "white", linewidth = 0.5) +
  geom_sf(data = hexes, fill = NA, color = "#7A869A", linewidth = 0.55) +
  geom_sf(data = block_groups, fill = NA, color = "#344563", linewidth = 0.75) +
  geom_sf(
    data = support_points,
    aes(shape = support_type, size = support_value),
    fill = "#FFB000",
    color = "#172B4D",
    stroke = 0.5,
    alpha = 0.9
  ) +
  geom_sf(
    data = block_point_fallback,
    shape = 4,
    size = 2.2,
    stroke = 0.8,
    color = "#6554C0"
  ) +
  scale_shape_manual(
    values = c(
      "Reported floor area" = 21,
      "Unit-count fallback" = 24,
      "Parcel-count fallback" = 22
    ),
    guide = "none"
  ) +
  scale_size_continuous(
    range = c(1.8, 6.5),
    trans = "sqrt",
    guide = "none"
  ) +
  coord_sf(xlim = c(0, 12), ylim = c(0, 8), expand = FALSE) +
  labs(
    title = "3. Allocate each block using residential evidence",
    subtitle = paste(
      "Apply the hierarchy separately to each parcel;",
      "support bases may mix within a Census block.",
      sep = "\n"
    )
  ) +
  panel_theme

hex_labels <- hex_results %>%
  filter(allocated_population >= 30) %>%
  mutate(
    label = number(allocated_population, accuracy = 1),
    geometry = st_point_on_surface(geometry)
  )

p_hex <- ggplot() +
  geom_sf(
    data = hex_results,
    aes(fill = allocated_population, color = median_source),
    linewidth = 0.9
  ) +
  geom_sf_text(
    data = hex_labels,
    aes(label = label),
    size = 2.65,
    fontface = "bold",
    color = "#172B4D",
    check_overlap = TRUE
  ) +
  geom_sf(data = tract, fill = NA, color = "#172B4D", linewidth = 0.9) +
  scale_fill_gradient(
    low = "#EDF8E9",
    high = "#238B45",
    guide = "none"
  ) +
  scale_color_manual(
    values = median_palette,
    na.value = "#C1C7D0",
    guide = "none",
    drop = FALSE
  ) +
  coord_sf(xlim = c(0, 12), ylim = c(0, 8), expand = FALSE) +
  labs(
    title = "4. Produce hex-level estimates",
    subtitle = paste(
      "Divide counts proportionally. Assign each median from the dominant",
      "residential block group, using the tract only when suppressed.",
      sep = "\n"
    )
  ) +
  panel_theme

support_legend <- ggplot() +
  annotate(
    "text",
    x = 0,
    y = 7.6,
    label = "PANEL 3 KEY",
    hjust = 0,
    size = 3.25,
    fontface = "bold",
    color = "#172B4D"
  ) +
  annotate(
    "point",
    x = rep(0.12, 4),
    y = c(6.7, 5.65, 4.6, 3.55),
    shape = c(21, 24, 22, 4),
    size = c(3.4, 3.4, 3.4, 3),
    stroke = 0.65,
    fill = c("#FFB000", "#FFB000", "#FFB000", NA),
    color = c("#172B4D", "#172B4D", "#172B4D", "#6554C0")
  ) +
  annotate(
    "text",
    x = rep(0.32, 4),
    y = c(6.7, 5.65, 4.6, 3.55),
    label = c(
      "reported floor area",
      "unit-count fallback",
      "parcel-count fallback",
      "block-point fallback"
    ),
    hjust = 0,
    size = 2.65,
    color = "#172B4D"
  ) +
  coord_cartesian(xlim = c(0, 2.4), ylim = c(0, 8), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(24, 8, 8, 8))

hex_legend <- ggplot() +
  annotate(
    "text",
    x = 0,
    y = 7.6,
    label = "PANEL 4 KEY",
    hjust = 0,
    size = 3.25,
    fontface = "bold",
    color = "#172B4D"
  ) +
  annotate(
    "rect",
    xmin = c(0, 0.28, 0.56),
    xmax = c(0.25, 0.53, 0.81),
    ymin = 6.3,
    ymax = 6.7,
    fill = c("#EDF8E9", "#86BE8B", "#238B45"),
    color = NA
  ) +
  annotate(
    "text",
    x = 0,
    y = 6.05,
    label = "Hex fill: allocated population\n(white = 0; light to dark)",
    hjust = 0,
    vjust = 1,
    lineheight = 1.05,
    size = 2.65,
    color = "#172B4D"
  ) +
  annotate(
    "segment",
    x = c(0, 0),
    xend = c(0.8, 0.8),
    y = c(4.55, 3.15),
    yend = c(4.55, 3.15),
    linewidth = 1.25,
    color = c(
      median_palette[["Block group A: $1,100"]],
      median_palette[["Tract fallback: $1,350"]]
    )
  ) +
  annotate(
    "text",
    x = c(0, 0),
    y = c(4.3, 2.9),
    label = c(
      "Blue border:\nblock-group median $1,100",
      "Red border:\ntract fallback $1,350"
    ),
    hjust = 0,
    vjust = 1,
    lineheight = 1.05,
    size = 2.65,
    color = c(
      median_palette[["Block group A: $1,100"]],
      median_palette[["Tract fallback: $1,350"]]
    ),
    fontface = "bold"
  ) +
  coord_cartesian(xlim = c(0, 2.4), ylim = c(0, 8), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(24, 8, 8, 8))

top_row <- plot_spacer() + p_source + p_blocks + plot_spacer() +
  plot_layout(widths = c(0.42, 1, 1, 0.42))
bottom_row <- support_legend + p_support + p_hex + hex_legend +
  plot_layout(widths = c(0.42, 1, 1, 0.42))

workflow_figure <- top_row / bottom_row +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(
    title = "How ACS estimates are translated to the EWS hex grid",
    subtitle = paste(
      "Dasymetric allocation uses Census blocks and appraisal-based residential",
      "evidence instead of distributing estimates by land area."
    ),
    caption = paste(
      "Conceptual example; values and geometries are illustrative.",
      "The production output retains allocation weights, source geography,",
      "fallback method, and ACS margin of error for audit."
    ),
    theme = theme(
      plot.title = element_text(
        face = "bold",
        size = 20,
        color = "#172B4D",
        margin = margin(b = 4)
      ),
      plot.subtitle = element_text(
        size = 11.5,
        color = "#52606D",
        margin = margin(b = 10)
      ),
      plot.caption = element_text(
        size = 8.5,
        color = "#6B778C",
        hjust = 0,
        margin = margin(t = 8)
      ),
      plot.margin = margin(14, 16, 10, 16)
    )
  )

png_file <- file.path(FIGURES_DIR, "09_dasymetric_allocation_workflow.png")
pdf_file <- file.path(FIGURES_DIR, "09_dasymetric_allocation_workflow.pdf")

ggsave(
  png_file,
  workflow_figure,
  width = 16,
  height = 9,
  dpi = 300,
  bg = "white"
)
ggsave(
  pdf_file,
  workflow_figure,
  width = 16,
  height = 9,
  device = pdf,
  bg = "white"
)

print_progress(paste0("Saved dasymetric workflow figure: ", png_file))
print_progress(paste0("Saved vector workflow figure: ", pdf_file))
