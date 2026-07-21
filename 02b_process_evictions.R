################################################################################
# 02b - Process Eviction Filing Data to Hexagonal Grid
################################################################################
#
# Reads prepared all-JP eviction filing records and ArcGIS geocoding results from
# 02a_prepare_eviction_addresses.R, assigns geocoded defendant addresses to the
# Austin H3 grid, writes source-specific summaries, and generates high-level
# eviction figures.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(forcats)
  library(ggplot2)
  library(lubridate)
  library(readr)
  library(scales)
  library(sf)
  library(stringr)
  library(tigris)
  library(tidyr)
  library(viridis)
})

source(here::here("R/utils.R"))
source(here::here("R/analysis_config.R"))

print_header("02b - PROCESS EVICTION FILINGS")

OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
FIGURES_DIR <- here::here("figures")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

if (!exists("hex_grid")) {
  hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid")
}

eviction_filings_file <- file.path(OUTPUT_DIR, "eviction_filings_prepared_for_geocoding.csv")
eviction_geocoded_file <- file.path(OUTPUT_DIR, "eviction_addresses_geocoded.csv")

if (!file.exists(eviction_filings_file) || !file.exists(eviction_geocoded_file)) {
  stop(
    "Prepared eviction/geocode outputs not found. Run 02a_prepare_eviction_addresses.R first.",
    call. = FALSE
  )
}

print_progress("Loading prepared full JP eviction filings and geocoded defendant addresses...")

eviction_filings_raw <- read_csv(eviction_filings_file, show_col_types = FALSE) %>%
  mutate(
    file_date = as.Date(file_date),
    filing_year = year(file_date),
    filing_month = floor_date(file_date, "month")
  ) %>%
  filter(
    !is.na(file_date),
    file_date <= EWS_CONFIG$analysis_as_of_date
  )

eviction_geocoded <- read_csv(
  eviction_geocoded_file,
  col_types = cols(
    address_id = col_double(),
    address_for_geocoding = col_character(),
    status = col_character(),
    score = col_double(),
    match_addr = col_character(),
    addr_type = col_character(),
    longitude = col_double(),
    latitude = col_double(),
    .default = col_skip()
  ),
  show_col_types = FALSE
) %>%
  rename(
    geocode_status = status,
    geocode_score = score,
    geocode_match_addr = match_addr,
    geocode_addr_type = addr_type
  ) %>%
  select(
    address_id,
    address_for_geocoding,
    geocode_status,
    geocode_score,
    geocode_match_addr,
    geocode_addr_type,
    longitude,
    latitude
  )

eviction_filings <- eviction_filings_raw %>%
  left_join(eviction_geocoded, by = "address_for_geocoding") %>%
  mutate(
    geocoded = !is.na(longitude) & !is.na(latitude),
    reliable_geocode = geocoded & geocode_status %in% c("M", "T") & geocode_score >= 90
  )

eviction_geocode_qc <- eviction_filings %>%
  summarize(
    filing_defendant_rows = n(),
    unique_cases = n_distinct(case_number, na.rm = TRUE),
    rows_with_candidate_address = sum(geocoding_candidate, na.rm = TRUE),
    rows_geocoded = sum(geocoded, na.rm = TRUE),
    rows_reliable_geocode = sum(reliable_geocode, na.rm = TRUE),
    unmatched_rows = sum(geocode_status == "U", na.rm = TRUE),
    missing_coordinates = sum(is.na(longitude) | is.na(latitude), na.rm = TRUE),
    min_file_date = min(file_date, na.rm = TRUE),
    max_file_date = max(file_date, na.rm = TRUE)
  )

write_csv(eviction_geocode_qc, file.path(OUTPUT_DIR, "eviction_full_geocode_qc.csv"))

eviction_filings_for_join <- eviction_filings %>%
  filter(geocoded) %>%
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326, remove = FALSE) %>%
  st_transform(st_crs(hex_grid))

eviction_filings_hex <- eviction_filings_for_join %>%
  st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE)

save_output(
  eviction_filings_hex,
  file.path(OUTPUT_DIR, "eviction_filings_full_geocoded_hex.rds"),
  "full geocoded eviction filing-defendant rows assigned to hexagons"
)

eviction_filings_hex %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "eviction_filings_full_geocoded_hex.csv"))

eviction_monthly_summary <- eviction_filings_hex %>%
  st_drop_geometry() %>%
  group_by(filing_month, jp_district) %>%
  summarize(
    eviction_defendant_rows = n(),
    eviction_cases = n_distinct(case_number),
    .groups = "drop"
  )

write_csv(eviction_monthly_summary, file.path(OUTPUT_DIR, "eviction_filings_monthly_by_jp.csv"))

eviction_annual_hex <- eviction_filings_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id, filing_year) %>%
  summarize(
    eviction_defendant_rows = n(),
    eviction_cases = n_distinct(case_number),
    eviction_cases_final_status = n_distinct(case_number[case_status == "Final Status"]),
    eviction_cases_dismissed = n_distinct(case_number[case_status == "Dismissed"]),
    .groups = "drop"
  ) %>%
  arrange(hex_id, filing_year)

write_csv(eviction_annual_hex, file.path(OUTPUT_DIR, "eviction_filings_by_hex_year.csv"))

max_eviction_date <- max(eviction_filings_hex$file_date, na.rm = TRUE)
latest_12mo_start <- EWS_CONFIG$analysis_as_of_date %m-% years(1) + days(1)
previous_12mo_start <- latest_12mo_start %m-% years(1)

hex_eviction_summary <- eviction_filings_hex %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarize(
    eviction_defendant_rows_total = n(),
    eviction_cases_total = n_distinct(case_number),
    eviction_cases_2020 = n_distinct(case_number[filing_year == 2020]),
    eviction_cases_2021 = n_distinct(case_number[filing_year == 2021]),
    eviction_cases_2022 = n_distinct(case_number[filing_year == 2022]),
    eviction_cases_2023 = n_distinct(case_number[filing_year == 2023]),
    eviction_cases_2024 = n_distinct(case_number[filing_year == 2024]),
    eviction_cases_2025 = n_distinct(case_number[filing_year == 2025]),
    eviction_cases_2026 = n_distinct(case_number[filing_year == 2026]),
    eviction_cases_latest_12mo = n_distinct(
      case_number[
        file_date >= latest_12mo_start &
          file_date <= EWS_CONFIG$analysis_as_of_date
      ]
    ),
    eviction_cases_previous_12mo = n_distinct(
      case_number[file_date >= previous_12mo_start & file_date < latest_12mo_start]
    ),
    eviction_final_status_cases_total = n_distinct(case_number[case_status == "Final Status"]),
    eviction_dismissed_cases_total = n_distinct(case_number[case_status == "Dismissed"]),
    first_eviction_file_date = min(file_date, na.rm = TRUE),
    last_eviction_file_date = max(file_date, na.rm = TRUE),
    jp_districts_with_evictions = paste(sort(unique(na.omit(jp_district))), collapse = "; "),
    .groups = "drop"
  ) %>%
  mutate(
    eviction_cases_latest_12mo_change_pct = if_else(
      eviction_cases_previous_12mo > 0,
      100 * (eviction_cases_latest_12mo / eviction_cases_previous_12mo - 1),
      NA_real_
    ),
    eviction_analysis_as_of = EWS_CONFIG$analysis_as_of_date,
    eviction_latest_12mo_start = latest_12mo_start,
    eviction_previous_12mo_start = previous_12mo_start
  )

save_output(
  hex_eviction_summary,
  file.path(OUTPUT_DIR, "eviction_filings_by_hex_summary.rds"),
  "eviction filing hex summary"
)

write_csv(hex_eviction_summary, file.path(OUTPUT_DIR, "eviction_filings_by_hex_summary.csv"))

print_progress("Creating eviction filing visualizations...")

p_evictions_monthly <- eviction_monthly_summary %>%
  group_by(filing_month) %>%
  summarize(eviction_cases = sum(eviction_cases), .groups = "drop") %>%
  ggplot(aes(x = filing_month, y = eviction_cases)) +
  geom_line(color = "#2f6f7e", linewidth = 0.7) +
  geom_point(color = "#2f6f7e", size = 1.1) +
  scale_y_continuous(labels = comma) +
  labs(
    title = "Eviction Filings by Month",
    subtitle = paste0(
      "All Travis County JP districts, ",
      format(min(eviction_filings$file_date, na.rm = TRUE), "%b %d, %Y"),
      " through ",
      format(max(eviction_filings$file_date, na.rm = TRUE), "%b %d, %Y")
    ),
    x = NULL,
    y = "Cases"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold")
  )

ggsave(
  file.path(FIGURES_DIR, "02_eviction_filings_by_month.png"),
  p_evictions_monthly,
  width = 10,
  height = 5,
  dpi = 300,
  bg = "white"
)

p_evictions_jp_year <- eviction_filings_hex %>%
  st_drop_geometry() %>%
  distinct(case_number, jp_district, filing_year) %>%
  count(filing_year, jp_district, name = "eviction_cases") %>%
  ggplot(aes(x = factor(filing_year), y = eviction_cases, fill = jp_district)) +
  geom_col(position = "dodge") +
  scale_y_continuous(labels = comma) +
  scale_fill_viridis_d(option = "mako", end = 0.85) +
  labs(
    title = "Eviction Filings by JP District and Year",
    x = NULL,
    y = "Cases",
    fill = "JP district"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold")
  )

ggsave(
  file.path(FIGURES_DIR, "02_eviction_filings_by_jp_year.png"),
  p_evictions_jp_year,
  width = 10,
  height = 6,
  dpi = 300,
  bg = "white"
)

p_eviction_status <- eviction_filings %>%
  distinct(case_number, case_status) %>%
  count(case_status, name = "eviction_cases", sort = TRUE) %>%
  mutate(case_status = forcats::fct_reorder(case_status, eviction_cases)) %>%
  ggplot(aes(x = eviction_cases, y = case_status)) +
  geom_col(fill = "#6c8a4d") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Eviction Case Status",
    subtitle = "All Travis County JP districts",
    x = "Cases",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold")
  )

ggsave(
  file.path(FIGURES_DIR, "02_eviction_case_status.png"),
  p_eviction_status,
  width = 8,
  height = 5,
  dpi = 300,
  bg = "white"
)

hex_evictions_map <- hex_grid %>%
  left_join(hex_eviction_summary, by = "hex_id") %>%
  mutate(eviction_cases_total = replace_na(eviction_cases_total, 0))

eviction_map_context <- list()
if (
  requireNamespace("ggspatial", quietly = TRUE) &&
    requireNamespace("rosm", quietly = TRUE) &&
    requireNamespace("prettymapr", quietly = TRUE)
) {
  eviction_map_context <- c(
    eviction_map_context,
    list(ggspatial::annotation_map_tile(type = "cartolight", zoomin = -1, alpha = 0.75))
  )
}

eviction_jurisdictions_file <- file.path(DATA_DIR, "BOUNDARIES_jurisdictions_20260429.geojson")
eviction_context_boundaries <- NULL
if (file.exists(eviction_jurisdictions_file)) {
  eviction_context_boundaries <- st_read(eviction_jurisdictions_file, quiet = TRUE) %>%
    st_transform(st_crs(hex_grid))
}

eviction_context_roads <- tryCatch(
  {
    roads(state = "TX", county = "Travis County") %>%
      filter(RTTYP %in% c("I", "S", "U")) %>%
      st_transform(st_crs(hex_grid))
  },
  error = function(e) {
    print_progress("WARNING: Could not load major roads for eviction map context.")
    NULL
  }
)

p_evictions_hex <- ggplot() +
  eviction_map_context +
  geom_sf(data = hex_grid, fill = NA, color = "grey82", linewidth = 0.08) +
  {if (!is.null(eviction_context_boundaries)) {
    geom_sf(data = eviction_context_boundaries, fill = NA, color = "grey35", linewidth = 0.25)
  }} +
  {if (!is.null(eviction_context_roads)) {
    geom_sf(data = eviction_context_roads, color = "grey45", linewidth = 0.18, alpha = 0.6)
  }} +
  geom_sf(
    data = hex_evictions_map %>% filter(eviction_cases_total > 0),
    aes(fill = eviction_cases_total),
    color = NA,
    alpha = 0.82
  ) +
  scale_fill_viridis_c(option = "magma", trans = "sqrt", labels = comma) +
  ggthemes::theme_map() +
  labs(
    title = "Geocoded Eviction Filings by Hexagon",
    subtitle = "Count of unique cases by defendant address location; roads and jurisdiction boundaries shown for context",
    fill = "Cases"
  ) +
  theme(
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title = element_text(face = "bold")
  )

ggsave(
  file.path(FIGURES_DIR, "02_eviction_filings_by_hex.png"),
  p_evictions_hex,
  width = 8,
  height = 8,
  dpi = 300,
  bg = "white"
)

print_progress(
  paste0(
    "Processed ",
    comma(nrow(eviction_filings)),
    " full filing-defendant rows and assigned ",
    comma(nrow(eviction_filings_hex)),
    " geocoded row(s) to hexagons."
  )
)

print_header("02b COMPLETE")
