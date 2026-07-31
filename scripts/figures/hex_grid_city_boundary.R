################################################################################
# Map the Current EWS Hex Grid and City of Austin Boundary
################################################################################
#
# The production grid was generated from the 2021 Census Austin place polygon.
# This figure compares that fixed H3 grid with the City jurisdiction boundary
# dated April 29, 2026, which is used by current boundary audits.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
  library(sf)
  library(tibble)
})

source(here::here("R", "utils.R"))
source(here::here("R", "analysis_config.R"))

OUTPUT_DIR <- here::here("output")
FIGURES_DIR <- here::here("figures")
HEX_GRID_FILE <- file.path(OUTPUT_DIR, "hex_grid.rds")
JURISDICTIONS_FILE <- here::here(
  "data",
  "BOUNDARIES_jurisdictions_20260429.geojson"
)
ANALYSIS_CRS <- 5070

dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

required_files <- c(HEX_GRID_FILE, JURISDICTIONS_FILE)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0L) {
  stop(
    "Missing required grid-map input(s): ",
    paste(missing_files, collapse = ", "),
    call. = FALSE
  )
}

sf_use_s2(FALSE)

hex_grid <- load_output(HEX_GRID_FILE, "current EWS hex grid") %>%
  st_make_valid() %>%
  st_transform(ANALYSIS_CRS)

city_jurisdictions <- st_read(JURISDICTIONS_FILE, quiet = TRUE) %>%
  st_make_valid() %>%
  filter(city_name == "CITY OF AUSTIN") %>%
  st_transform(ANALYSIS_CRS)

city_boundary <- city_jurisdictions %>%
  filter(jurisdiction_type == "FULL") %>%
  summarise(
    boundary_name = "Austin full-purpose boundary",
    .groups = "drop"
  )

city_limited <- city_jurisdictions %>%
  filter(jurisdiction_type == "LTD") %>%
  summarise(
    boundary_name = "Austin limited-purpose jurisdiction",
    .groups = "drop"
  )

if (nrow(city_boundary) != 1L || nrow(city_limited) != 1L) {
  stop("City jurisdiction boundaries did not dissolve as expected.", call. = FALSE)
}

hex_intersects_city <- lengths(st_intersects(hex_grid, city_boundary)) > 0L
hex_intersects_limited <- lengths(st_intersects(hex_grid, city_limited)) > 0L
hex_points <- suppressWarnings(st_point_on_surface(hex_grid))
hex_center_in_city <- lengths(st_within(hex_points, city_boundary)) > 0L

hex_map <- hex_grid %>%
  mutate(
    map_class = case_when(
      hex_intersects_city ~ "Intersects full-purpose boundary",
      hex_intersects_limited ~ "Limited-purpose jurisdiction only",
      TRUE ~ "Outside both City jurisdictions"
    )
  )

grid_union <- st_union(hex_grid)
city_uncovered <- suppressWarnings(st_difference(city_boundary, grid_union)) %>%
  st_as_sf() %>%
  mutate(map_class = "Full-purpose area outside grid")

city_area_km2 <- as.numeric(st_area(city_boundary)) / 1e6
city_uncovered_km2 <- sum(as.numeric(st_area(city_uncovered))) / 1e6
city_covered_pct <- 100 * (city_area_km2 - city_uncovered_km2) / city_area_km2
median_hex_area_km2 <- median(hex_grid$area_km2, na.rm = TRUE)

if (
  sum(hex_intersects_city) <= 0L ||
    sum(hex_center_in_city) <= 0L ||
    city_covered_pct <= 0 ||
    city_covered_pct > 100
) {
  stop("Grid and City boundary overlap checks failed.", call. = FALSE)
}

summary_text <- paste0(
  comma(nrow(hex_grid)),
  " total H3 cells\n",
  comma(sum(hex_intersects_city)),
  " intersect full-purpose Austin\n",
  comma(sum(!hex_intersects_city & hex_intersects_limited)),
  " intersect limited-purpose Austin only\n",
  comma(sum(!hex_intersects_city & !hex_intersects_limited)),
  " intersect neither jurisdiction\n",
  number(median_hex_area_km2, accuracy = 0.001),
  " km2 median cell area\n",
  number(city_covered_pct, accuracy = 0.1),
  "% of full-purpose area covered"
)

map_palette <- c(
  "Intersects full-purpose boundary" = "#B8D8D8",
  "Limited-purpose jurisdiction only" = "#F3D58A",
  "Outside both City jurisdictions" = "#E7EBF0",
  "Full-purpose area outside grid" = "#F4A6A6"
)

map_theme <- theme_void(base_size = 11) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 13,
      color = "#172B4D",
      margin = margin(b = 3)
    ),
    plot.subtitle = element_text(
      size = 9.5,
      color = "#52606D",
      lineheight = 1.05,
      margin = margin(b = 8)
    ),
    plot.margin = margin(6, 8, 6, 8),
    legend.position = "none"
  )

main_map <- ggplot() +
  geom_sf(
    data = hex_map,
    aes(fill = map_class),
    color = "#8C99A8",
    linewidth = 0.07
  ) +
  geom_sf(
    data = city_uncovered,
    aes(fill = map_class),
    color = NA
  ) +
  geom_sf(
    data = city_boundary,
    fill = NA,
    color = "#C62828",
    linewidth = 0.85
  ) +
  scale_fill_manual(values = map_palette, drop = FALSE) +
  coord_sf(datum = NA, expand = FALSE) +
  labs(
    title = "Complete Current Study Grid",
    subtitle = paste(
      "Gold cells are outside full-purpose Austin but intersect",
      "the City's limited-purpose jurisdiction."
    )
  ) +
  map_theme

downtown_point <- st_sfc(
  st_point(c(-97.7431, 30.2672)),
  crs = 4326
) %>%
  st_transform(ANALYSIS_CRS)
downtown_xy <- st_coordinates(downtown_point)[1, ]
zoom_half_width <- 3000
zoom_half_height <- 2400
zoom_xlim <- downtown_xy[["X"]] + c(-zoom_half_width, zoom_half_width)
zoom_ylim <- downtown_xy[["Y"]] + c(-zoom_half_height, zoom_half_height)

zoom_map <- ggplot() +
  geom_sf(
    data = hex_map,
    aes(fill = map_class),
    color = "#65758B",
    linewidth = 0.28
  ) +
  geom_sf(
    data = city_boundary,
    fill = NA,
    color = "#C62828",
    linewidth = 1
  ) +
  annotate(
    "segment",
    x = zoom_xlim[[1]] + 350,
    xend = zoom_xlim[[1]] + 1350,
    y = zoom_ylim[[1]] + 350,
    yend = zoom_ylim[[1]] + 350,
    linewidth = 1.2,
    color = "#172B4D"
  ) +
  annotate(
    "text",
    x = zoom_xlim[[1]] + 850,
    y = zoom_ylim[[1]] + 520,
    label = "1 km",
    size = 3,
    fontface = "bold",
    color = "#172B4D"
  ) +
  scale_fill_manual(values = map_palette, drop = FALSE) +
  coord_sf(
    xlim = zoom_xlim,
    ylim = zoom_ylim,
    datum = NA,
    expand = FALSE
  ) +
  labs(
    title = "Grid Detail",
    subtitle = "Central Austin; each outlined polygon is one analysis cell."
  ) +
  map_theme

metrics_panel <- ggplot() +
  annotate(
    "text",
    x = 0,
    y = 1,
    label = "CURRENT GRID",
    hjust = 0,
    vjust = 1,
    size = 3.7,
    fontface = "bold",
    color = "#172B4D"
  ) +
  annotate(
    "text",
    x = 0,
    y = 0.82,
    label = summary_text,
    hjust = 0,
    vjust = 1,
    lineheight = 1.18,
    size = 3.35,
    color = "#344563"
  ) +
  annotate(
    "rect",
    xmin = c(0, 0, 0, 0),
    xmax = c(0.09, 0.09, 0.09, 0.09),
    ymin = c(-0.01, -0.13, -0.25, -0.37),
    ymax = c(0.07, -0.05, -0.17, -0.29),
    fill = unname(map_palette),
    color = "#8C99A8",
    linewidth = 0.2
  ) +
  annotate(
    "text",
    x = 0.12,
    y = c(0.03, -0.09, -0.21, -0.33),
    label = names(map_palette),
    hjust = 0,
    size = 2.55,
    color = "#344563"
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-0.41, 1), clip = "off") +
  theme_void() +
  theme(plot.margin = margin(8, 10, 8, 10))

right_column <- zoom_map / metrics_panel +
  plot_layout(heights = c(0.68, 0.32))

grid_figure <- main_map + right_column +
  plot_layout(widths = c(1.18, 0.82)) +
  plot_annotation(
    title = "Current EWS Hex Grid and City of Austin Boundary",
    subtitle = paste(
      "H3 resolution 9 grid generated from the 2021 Census Austin place;",
      "compared with the City's full- and limited-purpose jurisdictions dated April 29, 2026."
    ),
    caption = paste(
      "Complete cells are retained rather than clipped.",
      "The red outline is the current full-purpose boundary;",
      "red fill identifies full-purpose areas outside the fixed grid."
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

png_file <- file.path(FIGURES_DIR, "10_hex_grid_city_boundary.png")
pdf_file <- file.path(FIGURES_DIR, "10_hex_grid_city_boundary.pdf")

ggsave(
  png_file,
  grid_figure,
  width = 16,
  height = 9,
  dpi = 300,
  bg = "white"
)
ggsave(
  pdf_file,
  grid_figure,
  width = 16,
  height = 9,
  device = pdf,
  bg = "white"
)

print_progress(paste0("Saved current grid map: ", png_file))
print_progress(paste0("Saved vector grid map: ", pdf_file))
