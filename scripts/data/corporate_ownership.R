################################################################################
# Process Corporate Ownership / Residential Parcel Data to Hexagonal Grid
################################################################################
#
# Reads prebuilt residential parcel universe files, assigns parcel points to the
# Austin H3 grid, aggregates corporate ownership and residential parcel metrics,
# and writes source-specific outputs and figures.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(forcats)
  library(ggplot2)
  library(purrr)
  library(readr)
  library(scales)
  library(sf)
  library(tidyr)
  library(viridis)
})

source(here::here("R/utils.R"))

print_header("02c - PROCESS CORPORATE PARCELS")

OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
FIGURES_DIR <- here::here("figures")

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)

if (!exists("hex_grid")) {
  hex_grid <- load_output(file.path(OUTPUT_DIR, "hex_grid.rds"), "hexagonal grid")
}

residential_parcel_files <- c(
  Travis = file.path(DATA_DIR, "residential_parcels_for_hex.csv"),
  Williamson = file.path(DATA_DIR, "williamson_residential_parcels_for_hex.csv"),
  Hays = file.path(DATA_DIR, "hays_residential_parcels_for_hex.csv")
)
jurisdictions_file <- file.path(DATA_DIR, "BOUNDARIES_jurisdictions_20260429.geojson")
calibrated_units_file <- file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.rds")
targeted_units_file <- file.path(OUTPUT_DIR, "residential_parcels_unit_targeted.rds")
promoted_units_file <- file.path(OUTPUT_DIR, "residential_parcels_unit_promoted.rds")
unit_surface <- tolower(
  Sys.getenv("EWS_UNIT_SURFACE", unset = "promoted")
)

if (!unit_surface %in% c("promoted", "baseline")) {
  stop("EWS_UNIT_SURFACE must be 'promoted' or 'baseline'.", call. = FALSE)
}

if (unit_surface == "promoted") {
  if (!file.exists(promoted_units_file)) {
    stop(
      "Canonical promoted unit surface is missing: ",
      promoted_units_file,
      ". Build and promote the unit hierarchy through _targets.R, or set ",
      "EWS_UNIT_SURFACE=baseline for an explicit historical/bootstrap run.",
      call. = FALSE
    )
  }
  if (!file.exists(targeted_units_file)) {
    stop(
      "Cannot validate the promoted unit surface without ",
      targeted_units_file,
      ".",
      call. = FALSE
    )
  }

  residential_parcels_raw <- load_output(
    promoted_units_file,
    "promoted residential parcel unit hierarchy"
  )
  targeted_baseline <- load_output(
    targeted_units_file,
    "targeted residential parcel unit baseline"
  ) %>%
    transmute(
      parcel_id = as.character(parcel_id),
      current_baseline_targeted_units = coalesce(
        as.numeric(units_calibrated_targeted),
        as.numeric(units_calibrated),
        0
      )
    )

  required_promotion_columns <- c(
    "parcel_id",
    "promotion_baseline_targeted_units",
    "units_calibrated_targeted",
    "unit_model_promotion_version"
  )
  missing_promotion_columns <- setdiff(
    required_promotion_columns,
    names(residential_parcels_raw)
  )
  if (length(missing_promotion_columns) > 0L) {
    stop(
      "Promoted parcel surface is missing: ",
      paste(missing_promotion_columns, collapse = ", "),
      call. = FALSE
    )
  }

  promotion_validation <- residential_parcels_raw %>%
    st_drop_geometry() %>%
    transmute(
      parcel_id = as.character(parcel_id),
      promotion_baseline_targeted_units = as.numeric(
        promotion_baseline_targeted_units
      )
    ) %>%
    inner_join(
      targeted_baseline,
      by = "parcel_id",
      relationship = "one-to-one"
    )

  if (
    nrow(promotion_validation) != nrow(residential_parcels_raw) ||
      nrow(promotion_validation) != nrow(targeted_baseline) ||
      anyDuplicated(promotion_validation$parcel_id) ||
      any(
        abs(
          promotion_validation$promotion_baseline_targeted_units -
            promotion_validation$current_baseline_targeted_units
        ) > 1e-9
      )
  ) {
    stop(
      "Promoted units do not match the current 02e targeted baseline. ",
      "Re-run 02p through 02t and 02v before rebuilding canonical outputs.",
      call. = FALSE
    )
  }

  residential_parcels_raw <- residential_parcels_raw %>%
    mutate(
      property_units = units_calibrated_targeted,
      corporate_units = if_else(
        as.logical(is_corporate_owned),
        replace_na(units_calibrated_targeted, 0),
        0
      )
    )

  promotion_versions <- unique(
    residential_parcels_raw$unit_model_promotion_version
  )
  if (length(promotion_versions) != 1L || is.na(promotion_versions)) {
    stop("Promoted parcel surface has inconsistent versions.", call. = FALSE)
  }
  print_progress(
    paste0(
      "Using promoted parcel unit hierarchy ",
      promotion_versions,
      "."
    )
  )
} else if (file.exists(targeted_units_file)) {
  residential_parcels_raw <- load_output(
    targeted_units_file,
    "targeted calibrated residential parcel unit counts"
  ) %>%
    mutate(
      property_units = units_calibrated_targeted,
      corporate_units = if_else(
        as.logical(is_corporate_owned),
        replace_na(units_calibrated_targeted, 0),
        0
      )
    )

  print_progress("Using explicit baseline targeted parcel units from 02e.")
} else if (file.exists(calibrated_units_file)) {
  residential_parcels_raw <- load_output(
    calibrated_units_file,
    "calibrated residential parcel unit counts"
  ) %>%
    mutate(
      property_units = units_calibrated,
      corporate_units = if_else(
        as.logical(is_corporate_owned),
        replace_na(units_calibrated, 0),
        0
      )
    )

  print_progress("Using calibrated parcel unit counts from 02d output.")
} else {
  missing_residential_files <- residential_parcel_files[!file.exists(residential_parcel_files)]

  if (length(missing_residential_files) > 0) {
    stop(
      "Missing required residential parcel universe file(s): ",
      paste(missing_residential_files, collapse = ", "),
      call. = FALSE
    )
  }

  residential_parcel_schemas <- map(
    residential_parcel_files,
    ~names(read_csv(.x, n_max = 0, col_types = cols(.default = col_character()), show_col_types = FALSE))
  )

  if (!all(map_lgl(residential_parcel_schemas, identical, residential_parcel_schemas[[1]]))) {
    stop("Residential parcel universe files do not have identical schemas.", call. = FALSE)
  }

  residential_parcels_raw <- imap_dfr(
    residential_parcel_files,
    ~read_csv(.x, col_types = cols(.default = col_character()), show_col_types = FALSE) %>%
      mutate(source_county = .y)
  )
}

duplicate_parcel_ids <- residential_parcels_raw %>%
  count(parcel_id, name = "row_count") %>%
  filter(row_count > 1)

if (nrow(duplicate_parcel_ids) > 0) {
  stop(
    "Duplicate parcel_id values found after binding county parcel files. ",
    "Example duplicate(s): ",
    paste(head(duplicate_parcel_ids$parcel_id, 10), collapse = ", "),
    call. = FALSE
  )
}

residential_parcels_clean <- residential_parcels_raw %>%
  mutate(
    lat = as.numeric(lat),
    lon = as.numeric(lon),
    parcel_count = replace_na(as.numeric(parcel_count), 0),
    property_units = replace_na(as.numeric(property_units), 0),
    improvement_sqft = replace_na(as.numeric(improvement_sqft), 0),
    land_sqft = replace_na(as.numeric(land_sqft), 0),
    corporate_parcel_count = replace_na(as.numeric(corporate_parcel_count), 0),
    corporate_units = replace_na(as.numeric(corporate_units), 0),
    corporate_improvement_sqft = replace_na(as.numeric(corporate_improvement_sqft), 0),
    is_residential = replace_na(as.logical(is_residential), FALSE),
    is_owner_occupied = replace_na(as.logical(is_owner_occupied), FALSE),
    is_corporate_owned = replace_na(as.logical(is_corporate_owned), FALSE),
    has_financialized_owner = replace_na(as.logical(has_financialized_owner), FALSE)
  )

missing_coords <- residential_parcels_clean %>%
  filter(is.na(lat) | is.na(lon))

if (nrow(missing_coords) > 0) {
  stop(
    "Residential parcel universe contains ",
    nrow(missing_coords),
    " row(s) with missing or non-numeric lat/lon coordinates.",
    call. = FALSE
  )
}

residential_parcel_county_totals <- residential_parcels_clean %>%
  group_by(source_county) %>%
  summarise(
    row_count = n(),
    parcel_count = sum(parcel_count, na.rm = TRUE),
    residential_units = sum(property_units, na.rm = TRUE),
    corporate_owned_parcels = sum(corporate_parcel_count, na.rm = TRUE),
    corporate_owned_units = sum(corporate_units, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  residential_parcel_county_totals,
  file.path(OUTPUT_DIR, "residential_parcel_universe_by_county.csv")
)

print_progress("Residential parcel universe by county:")
print(residential_parcel_county_totals)

if (nrow(residential_parcels_clean) == 0) {
  stop("Residential parcel universe is empty after binding county files.", call. = FALSE)
}

residential_parcels <- residential_parcels_clean %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(st_crs(hex_grid))

austin_boundaries <- NULL
austin_full_purpose <- NULL

if (file.exists(jurisdictions_file)) {
  austin_boundaries <- st_read(jurisdictions_file, quiet = TRUE) %>%
    st_transform(st_crs(hex_grid))

  austin_full_purpose <- austin_boundaries %>%
    filter(jurisdiction_type == "FULL")
} else {
  print_progress("WARNING: Austin jurisdiction boundaries file not found; corporate maps will omit boundary overlays.")
}

residential_hex_joined <- residential_parcels %>%
  st_join(hex_grid %>% select(hex_id), join = st_within, left = FALSE)

corporate_hex_summary <- residential_hex_joined %>%
  st_drop_geometry() %>%
  group_by(hex_id) %>%
  summarise(
    residential_parcels = sum(parcel_count, na.rm = TRUE),
    residential_units = sum(property_units, na.rm = TRUE),
    residential_improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
    residential_land_sqft = sum(land_sqft, na.rm = TRUE),
    corporate_owned_parcels = sum(corporate_parcel_count, na.rm = TRUE),
    corporate_owned_units = sum(corporate_units, na.rm = TRUE),
    corporate_owned_imprv_sqft = sum(corporate_improvement_sqft, na.rm = TRUE),
    corporate_owner_count = n_distinct(owner_names[is_corporate_owned], na.rm = TRUE),
    financialized_owner_parcels = sum(parcel_count[has_financialized_owner], na.rm = TRUE),
    geocoded_parcels = sum(coord_source != "existing_coord", na.rm = TRUE),
    .groups = "drop"
  )

citywide_corporate_units <- sum(corporate_hex_summary$corporate_owned_units, na.rm = TRUE)
citywide_corporate_parcels <- sum(corporate_hex_summary$corporate_owned_parcels, na.rm = TRUE)
citywide_residential_units <- sum(corporate_hex_summary$residential_units, na.rm = TRUE)
citywide_residential_parcels <- sum(corporate_hex_summary$residential_parcels, na.rm = TRUE)

corporate_hex_summary <- corporate_hex_summary %>%
  mutate(
    pct_corporate_parcels = if_else(
      residential_parcels > 0,
      corporate_owned_parcels / residential_parcels * 100,
      NA_real_
    ),
    pct_corporate_units = if_else(
      residential_units > 0,
      corporate_owned_units / residential_units * 100,
      NA_real_
    ),
    pct_corporate_improvement_sqft = if_else(
      residential_improvement_sqft > 0,
      corporate_owned_imprv_sqft / residential_improvement_sqft * 100,
      NA_real_
    ),
    pct_financialized_owner_parcels = if_else(
      residential_parcels > 0,
      financialized_owner_parcels / residential_parcels * 100,
      NA_real_
    ),
    corporate_unit_share_city = if (citywide_corporate_units > 0) {
      corporate_owned_units / citywide_corporate_units * 100
    } else {
      NA_real_
    },
    corporate_parcel_share_city = if (citywide_corporate_parcels > 0) {
      corporate_owned_parcels / citywide_corporate_parcels * 100
    } else {
      NA_real_
    }
  )

hex_corporate <- hex_grid %>%
  left_join(corporate_hex_summary, by = "hex_id") %>%
  mutate(
    across(
      c(
        residential_parcels,
        residential_units,
        residential_improvement_sqft,
        residential_land_sqft,
        corporate_owned_parcels,
        corporate_owned_units,
        corporate_owned_imprv_sqft,
        corporate_owner_count,
        financialized_owner_parcels,
        geocoded_parcels,
        corporate_unit_share_city,
        corporate_parcel_share_city
      ),
      ~replace_na(., 0)
    ),
    corporate_owned_units_per_km2 = corporate_owned_units / area_km2,
    corporate_owned_parcels_per_km2 = corporate_owned_parcels / area_km2,
    residential_units_per_km2 = residential_units / area_km2,
    residential_parcels_per_km2 = residential_parcels / area_km2,
    investor_owned_units = corporate_owned_units,
    pct_corporate_owned = pct_corporate_units
  )

save_output(
  residential_parcels,
  file.path(OUTPUT_DIR, "residential_parcels_for_hex_sf.rds"),
  "residential parcel universe points"
)
save_output(
  residential_parcels %>% filter(is_corporate_owned),
  file.path(OUTPUT_DIR, "corporate_owned_parcels_sf.rds"),
  "corporate-owned parcel points"
)
save_output(
  hex_corporate,
  file.path(OUTPUT_DIR, "corporate_ownership_by_hex.rds"),
  "corporate ownership hex summary"
)

hex_corporate %>%
  st_drop_geometry() %>%
  write_csv(file.path(OUTPUT_DIR, "corporate_ownership_by_hex.csv"))

print_progress("Creating corporate ownership visualizations...")

boundary_layers <- list()
if (!is.null(austin_boundaries)) {
  boundary_layers <- c(
    boundary_layers,
    list(geom_sf(data = austin_boundaries, fill = NA, color = "grey65", linewidth = 0.15))
  )
}
if (!is.null(austin_full_purpose) && nrow(austin_full_purpose) > 0) {
  boundary_layers <- c(
    boundary_layers,
    list(geom_sf(data = austin_full_purpose, fill = NA, color = "black", linewidth = 0.45))
  )
}

p_corp_parcels <- ggplot() +
  geom_sf(data = hex_corporate, aes(fill = pct_corporate_parcels), color = NA) +
  boundary_layers +
  scale_fill_viridis_c(option = "magma", labels = label_percent(scale = 1), name = "Corporate parcels") +
  ggthemes::theme_map() +
  labs(
    title = "Share of Residential Parcels with Corporate Ownership",
    subtitle = "Austin full-purpose residential parcel universe aggregated to H3 hexagons"
  ) +
  theme(plot.title = element_text(face = "bold"))

p_corp_units <- ggplot() +
  geom_sf(data = hex_corporate, aes(fill = pct_corporate_units), color = NA) +
  boundary_layers +
  scale_fill_viridis_c(option = "plasma", labels = label_percent(scale = 1), name = "Corporate units") +
  ggthemes::theme_map() +
  labs(
    title = "Share of Residential Units with Corporate Ownership",
    subtitle = "Corporate units divided by total residential units in each hex"
  ) +
  theme(plot.title = element_text(face = "bold"))

p_corp_density <- ggplot() +
  geom_sf(data = hex_corporate, aes(fill = corporate_owned_units_per_km2), color = NA) +
  boundary_layers +
  scale_fill_viridis_c(option = "inferno", trans = "sqrt", labels = comma, name = "Units/km2") +
  ggthemes::theme_map() +
  labs(title = "Density of Corporate-Owned Residential Units") +
  theme(plot.title = element_text(face = "bold"))

p_corp_points <- ggplot() +
  geom_sf(data = hex_grid, fill = NA, color = "grey85", linewidth = 0.1) +
  boundary_layers +
  geom_sf(
    data = residential_parcels %>% filter(is_corporate_owned),
    aes(size = corporate_units),
    color = "#1f78b4",
    alpha = 0.35
  ) +
  scale_size_continuous(range = c(0.2, 4), labels = comma, name = "Units") +
  ggthemes::theme_map() +
  labs(title = "Corporate-Owned Parcel Locations") +
  theme(plot.title = element_text(face = "bold"))

p_corp_improvement <- ggplot() +
  geom_sf(data = hex_corporate, aes(fill = pct_corporate_improvement_sqft), color = NA) +
  boundary_layers +
  scale_fill_viridis_c(option = "cividis", labels = label_percent(scale = 1), name = "Corporate sqft") +
  ggthemes::theme_map() +
  labs(
    title = "Share of Residential Improvement Square Footage with Corporate Ownership",
    subtitle = "Corporate-owned improvement square feet divided by total residential improvement square feet"
  ) +
  theme(plot.title = element_text(face = "bold"))

top_corporate_owners <- residential_parcels %>%
  filter(is_corporate_owned) %>%
  st_drop_geometry() %>%
  group_by(owner_names) %>%
  summarise(
    parcels = n_distinct(parcel_id),
    units = sum(corporate_units, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(units), desc(parcels)) %>%
  slice_head(n = 20) %>%
  mutate(owner_names = forcats::fct_reorder(owner_names, units))

p_top_owners <- ggplot(top_corporate_owners, aes(x = units, y = owner_names)) +
  geom_col(fill = "#2a9d8f") +
  scale_x_continuous(labels = comma) +
  labs(
    title = "Top Corporate Owners by Estimated Residential Units",
    x = "Estimated units",
    y = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(FIGURES_DIR, "02_corporate_owned_parcels_by_hex.png"), p_corp_parcels, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(FIGURES_DIR, "02_corporate_owned_units_by_hex.png"), p_corp_units, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(FIGURES_DIR, "02_corporate_owned_unit_density_by_hex.png"), p_corp_density, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(FIGURES_DIR, "02_corporate_owned_parcel_points.png"), p_corp_points, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(FIGURES_DIR, "02_corporate_owned_improvement_sqft_share_by_hex.png"), p_corp_improvement, width = 10, height = 8, dpi = 300, bg = "white")
ggsave(file.path(FIGURES_DIR, "02_top_corporate_owners.png"), p_top_owners, width = 10, height = 8, dpi = 300, bg = "white")

print_progress(paste0("Residential parcels joined to ", n_distinct(residential_hex_joined$hex_id), " hexagons"))
print_progress(paste0("Total residential parcels in joined hexes: ", comma(citywide_residential_parcels)))
print_progress(paste0("Total estimated residential units in joined hexes: ", comma(round(citywide_residential_units, 0))))
print_progress(paste0("Total corporate-owned parcels: ", comma(citywide_corporate_parcels)))
print_progress(paste0("Total estimated corporate-owned units: ", comma(round(citywide_corporate_units, 0))))

print_header("02c COMPLETE")
