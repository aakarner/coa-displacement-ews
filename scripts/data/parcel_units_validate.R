################################################################################
# Validate Calibrated Parcel Units Against ACS Housing Units
################################################################################
#
# Compares calibrated parcel-based residential unit counts to ACS 5-year total
# housing units at tract and block group geographies. Block groups are included
# to improve boundary diagnostics; ACS 1-year estimates are retained only as an
# external citywide benchmark because they are not available at small geography.
################################################################################

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(sf)
  library(tidycensus)
  library(tidyr)
})

source(here::here("R/utils.R"))

print_header("02e - VALIDATE PARCEL UNIT COUNTS AGAINST ACS")

OUTPUT_DIR <- here::here("output")
DATA_DIR <- here::here("data")
ACS_YEAR <- if (exists("ACS_YEAR")) ACS_YEAR else 2024
TARGETED_ADJUSTMENT_MIN_AUSTIN_AREA_SHARE <- 0.95
TARGETED_ADJUSTMENT_MIN_PARCEL_TO_ACS_RATIO <- 2
TARGETED_ADJUSTMENT_MIN_UNIT_OVERCOUNT <- 1000
TARGETED_ADJUSTMENT_MIN_PARCEL_UNITS <- 250
TARGETED_ADJUSTMENT_METHODS <- c(
  "costar_sqft_per_unit_estimate_strong_mf",
  "costar_sqft_per_unit_estimate_weak_mf"
)

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

calibrated_parcels_file <- file.path(OUTPUT_DIR, "residential_parcels_unit_calibrated.rds")
jurisdictions_file <- file.path(DATA_DIR, "BOUNDARIES_jurisdictions_20260429.geojson")

if (!file.exists(calibrated_parcels_file)) {
  stop(
    "Missing calibrated parcel unit file: ",
    calibrated_parcels_file,
    ". Run scripts/data/parcel_units_calibrate.R first.",
    call. = FALSE
  )
}

if (!file.exists(jurisdictions_file)) {
  stop("Missing Austin jurisdiction boundary file: ", jurisdictions_file, call. = FALSE)
}

sf_use_s2(FALSE)
options(tigris_use_cache = TRUE)

city_unit_benchmark <- tibble(
  benchmark = c("acs_2024_1yr_city_total", "mid_2026_low", "mid_2026_point", "mid_2026_high"),
  units = c(518574, 535000, 540000, 545000)
)

austin_full_purpose <- st_read(jurisdictions_file, quiet = TRUE) %>%
  filter(jurisdiction_type == "FULL") %>%
  st_make_valid() %>%
  st_transform(3857) %>%
  st_union() %>%
  st_as_sf()

parcels <- load_output(
  calibrated_parcels_file,
  "calibrated residential parcel unit counts"
) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326, remove = FALSE) %>%
  st_transform(3857)

load_full_parcel_weight_points <- function(target_crs) {
  raw_parcel_dir <- file.path(DATA_DIR, "raw_parcels")
  travis_zip <- file.path(raw_parcel_dir, "travis", "Parcel_poly.zip")
  williamson_rds <- file.path(raw_parcel_dir, "williamson", "wcad_parcels.rds")
  hays_rds <- file.path(raw_parcel_dir, "hays", "hays_parcels.rds")

  full_parcel_points <- list()

  if (file.exists(travis_zip) && requireNamespace("terra", quietly = TRUE)) {
    print_progress("Loading full Travis parcel map for ACS allocation weights...")
    travis_tmp <- tempfile("travis_parcels_")
    dir.create(travis_tmp)
    utils::unzip(travis_zip, exdir = travis_tmp)
    travis_shp <- file.path(travis_tmp, "Parcel_poly.shp")

    travis_vect <- terra::vect(travis_shp)
    travis_point_coords <- as_tibble(terra::geom(travis_vect)) %>%
      filter(!is.na(x), !is.na(y)) %>%
      group_by(geom) %>%
      summarise(
        x = (min(x) + max(x)) / 2,
        y = (min(y) + max(y)) / 2,
        .groups = "drop"
      ) %>%
      rename(parcel_row_id = geom)
    travis_point_attrs <- as_tibble(as.data.frame(travis_vect)) %>%
      mutate(parcel_row_id = row_number())
    travis_point_crs <- terra::crs(travis_vect)

    full_parcel_points$Travis <- travis_point_attrs %>%
      left_join(travis_point_coords, by = "parcel_row_id") %>%
      filter(!is.na(x), !is.na(y)) %>%
      st_as_sf(coords = c("x", "y"), crs = travis_point_crs, remove = FALSE) %>%
      transmute(
        source_county = "Travis",
        full_parcel_id = as.character(PROP_ID),
        full_parcel_count_weight = 1,
        full_residential_proxy_weight = 1,
        full_improvement_sqft_weight = NA_real_,
        geometry
      ) %>%
      st_transform(target_crs)
  } else if (file.exists(travis_zip)) {
    print_progress("WARNING: terra package not available; skipping Travis full parcel weights.")
  }

  if (file.exists(williamson_rds)) {
    print_progress("Loading full Williamson parcel map for ACS allocation weights...")
    williamson_parcels <- readRDS(williamson_rds)

    full_parcel_points$Williamson <- suppressWarnings(st_point_on_surface(williamson_parcels)) %>%
      st_transform(target_crs) %>%
      mutate(
        resflrarea_numeric = parse_number(as.character(resflrarea)),
        bldgarea_numeric = parse_number(as.character(bldgarea)),
        is_residential_proxy = usedscrp == "Residential" | coalesce(resflrarea_numeric, 0) > 0,
        improvement_weight = coalesce(resflrarea_numeric, bldgarea_numeric, 0)
      ) %>%
      transmute(
        source_county = "Williamson",
        full_parcel_id = as.character(parcelid),
        full_parcel_count_weight = 1,
        full_residential_proxy_weight = if_else(is_residential_proxy, 1, 0),
        full_improvement_sqft_weight = if_else(improvement_weight > 0, improvement_weight, NA_real_),
        geometry
      )
  }

  if (file.exists(hays_rds)) {
    print_progress("Loading full Hays parcel map for ACS allocation weights...")
    hays_parcels <- readRDS(hays_rds)

    full_parcel_points$Hays <- suppressWarnings(st_point_on_surface(hays_parcels)) %>%
      st_transform(target_crs) %>%
      transmute(
        source_county = "Hays",
        full_parcel_id = coalesce(as.character(TEXT), as.character(REFNAME)),
        full_parcel_count_weight = 1,
        full_residential_proxy_weight = 1,
        full_improvement_sqft_weight = NA_real_,
        geometry
      )
  }

  if (length(full_parcel_points) == 0) {
    print_progress("WARNING: No full parcel maps found; parcel-weighted ACS allocation will be omitted.")
    return(NULL)
  }

  bind_rows(full_parcel_points)
}

full_parcel_weight_points <- load_full_parcel_weight_points(st_crs(parcels))

summarise_parcels_by_geography <- function(parcels_sf, geography_sf) {
  parcels_sf %>%
    st_join(geography_sf %>% select(GEOID), join = st_within, left = FALSE) %>%
    st_drop_geometry() %>%
    group_by(GEOID) %>%
    summarise(
      parcel_count = n(),
      parcel_residential_units = sum(units_calibrated, na.rm = TRUE),
      parcel_residential_units_conservative = sum(units_calibrated_conservative, na.rm = TRUE),
      parcel_corporate_units = sum(corporate_units, na.rm = TRUE),
      parcel_units_retained = sum(units_calibrated[unit_estimation_method == "parcel_units_retained"], na.rm = TRUE),
      parcel_units_costar_strong_mf = sum(
        units_calibrated[
          unit_estimation_method %in% c(
            "costar_sqft_per_unit_estimate_strong_mf",
            "conservative_costar_sqft_per_unit_estimate_strong_mf",
            "direct_costar_units"
          )
        ],
        na.rm = TRUE
      ),
      parcel_units_costar_weak_mf = sum(
        units_calibrated[
          unit_estimation_method %in% c(
            "costar_sqft_per_unit_estimate_weak_mf",
            "conservative_costar_sqft_per_unit_estimate_weak_mf"
          )
        ],
        na.rm = TRUE
      ),
      parcel_units_single_family_fallback = sum(
        units_calibrated[unit_estimation_method %in% c("single_family_default_1", "single_family_fractional_fallback_1")],
        na.rm = TRUE
      ),
      .groups = "drop"
    )
}

summarise_full_parcel_weights_by_geography <- function(full_parcel_points_sf, geography_sf, austin_boundary_sf) {
  if (is.null(full_parcel_points_sf)) {
    return(tibble(GEOID = character()))
  }

  full_parcel_points_sf %>%
    st_join(geography_sf %>% select(GEOID), join = st_within, left = FALSE) %>%
    mutate(in_austin_full_purpose = lengths(st_intersects(., austin_boundary_sf)) > 0) %>%
    st_drop_geometry() %>%
    group_by(GEOID) %>%
    summarise(
      full_parcel_count_weight = sum(full_parcel_count_weight, na.rm = TRUE),
      austin_parcel_count_weight = sum(full_parcel_count_weight[in_austin_full_purpose], na.rm = TRUE),
      full_residential_proxy_weight = sum(full_residential_proxy_weight, na.rm = TRUE),
      austin_residential_proxy_weight = sum(full_residential_proxy_weight[in_austin_full_purpose], na.rm = TRUE),
      full_improvement_sqft_weight = sum(full_improvement_sqft_weight, na.rm = TRUE),
      austin_improvement_sqft_weight = sum(full_improvement_sqft_weight[in_austin_full_purpose], na.rm = TRUE),
      full_parcel_weight_count = n(),
      austin_parcel_weight_count = sum(in_austin_full_purpose, na.rm = TRUE),
      .groups = "drop"
    )
}

validate_acs_geography <- function(geography, output_prefix) {
  print_progress(
    paste0(
      "Fetching ACS ",
      geography,
      " total housing units for Travis, Hays, and Williamson Counties..."
    )
  )

  acs_units <- tryCatch(
    {
      get_acs(
        geography = geography,
        variables = c(total_housing_units = "B25001_001"),
        state = "TX",
        county = c("Travis", "Hays", "Williamson"),
        year = ACS_YEAR,
        survey = "acs5",
        geometry = TRUE,
        output = "wide"
      ) %>%
        st_transform(3857) %>%
        transmute(
          GEOID,
          NAME,
          acs_year = ACS_YEAR,
          acs_survey = "acs5",
          acs_geography = geography,
          acs_total_housing_units = total_housing_unitsE,
          acs_total_housing_units_moe = total_housing_unitsM,
          geometry
        )
    },
    error = function(e) {
      print_progress(
        paste0(
          "WARNING: Could not fetch ACS ",
          geography,
          " housing units: ",
          conditionMessage(e)
        )
      )
      NULL
    }
  )

  if (is.null(acs_units)) {
    return(
      list(
        validation = NULL,
        summary = tibble(
          metric_group = paste0("acs_", output_prefix, "_validation"),
          metric = paste0(output_prefix, "_housing_unit_validation_status"),
          value = NA_real_,
          note = "ACS housing units could not be fetched. Check Census API/network access."
        )
      )
    )
  }

  acs_with_area <- acs_units %>%
    mutate(geography_area_sqm = as.numeric(st_area(geometry)))

  austin_overlay <- suppressWarnings(
    st_intersection(
      acs_with_area %>% select(GEOID, geography_area_sqm),
      austin_full_purpose
    )
  ) %>%
    mutate(austin_intersection_area_sqm = as.numeric(st_area(geometry))) %>%
    st_drop_geometry() %>%
    group_by(GEOID) %>%
    summarise(
      austin_intersection_area_sqm = sum(austin_intersection_area_sqm, na.rm = TRUE),
      .groups = "drop"
    )

  geographies_for_validation <- acs_with_area %>%
    left_join(austin_overlay, by = "GEOID") %>%
    mutate(
      austin_intersection_area_sqm = replace_na(austin_intersection_area_sqm, 0),
      austin_area_share = pmin(1, austin_intersection_area_sqm / geography_area_sqm),
      acs_total_housing_units_area_weighted = acs_total_housing_units * austin_area_share,
      acs_total_housing_units_moe_area_weighted = acs_total_housing_units_moe * austin_area_share
    ) %>%
    filter(austin_area_share > 0)

  parcel_units <- summarise_parcels_by_geography(parcels, geographies_for_validation)
  full_parcel_weights <- summarise_full_parcel_weights_by_geography(
    full_parcel_weight_points,
    geographies_for_validation,
    austin_full_purpose
  )

  validation <- geographies_for_validation %>%
    left_join(parcel_units, by = "GEOID") %>%
    left_join(full_parcel_weights, by = "GEOID") %>%
    mutate(
      across(
        c(
          parcel_count,
          parcel_residential_units,
          parcel_residential_units_conservative,
          parcel_corporate_units,
          parcel_units_retained,
          parcel_units_costar_strong_mf,
          parcel_units_costar_weak_mf,
          parcel_units_single_family_fallback,
          full_parcel_count_weight,
          austin_parcel_count_weight,
          full_residential_proxy_weight,
          austin_residential_proxy_weight,
          full_improvement_sqft_weight,
          austin_improvement_sqft_weight,
          full_parcel_weight_count,
          austin_parcel_weight_count
        ),
        ~replace_na(., 0)
      ),
      acs_parcel_count_allocation_share = if_else(
        full_parcel_count_weight > 0,
        pmin(1, austin_parcel_count_weight / full_parcel_count_weight),
        NA_real_
      ),
      acs_residential_proxy_allocation_share = if_else(
        full_residential_proxy_weight > 0,
        pmin(1, austin_residential_proxy_weight / full_residential_proxy_weight),
        NA_real_
      ),
      acs_improvement_sqft_allocation_share = if_else(
        full_improvement_sqft_weight > 0,
        pmin(1, austin_improvement_sqft_weight / full_improvement_sqft_weight),
        NA_real_
      ),
      acs_total_housing_units_parcel_count_weighted =
        acs_total_housing_units * acs_parcel_count_allocation_share,
      acs_total_housing_units_residential_proxy_weighted =
        acs_total_housing_units * acs_residential_proxy_allocation_share,
      acs_total_housing_units_improvement_sqft_weighted =
        acs_total_housing_units * acs_improvement_sqft_allocation_share,
      acs_total_housing_units_moe_parcel_count_weighted =
        acs_total_housing_units_moe * acs_parcel_count_allocation_share,
      parcel_minus_acs_full_units = parcel_residential_units - acs_total_housing_units,
      conservative_parcel_minus_acs_full_units =
        parcel_residential_units_conservative - acs_total_housing_units,
      parcel_to_acs_full_ratio = if_else(
        acs_total_housing_units > 0,
        parcel_residential_units / acs_total_housing_units,
        NA_real_
      ),
      conservative_parcel_to_acs_full_ratio = if_else(
        acs_total_housing_units > 0,
        parcel_residential_units_conservative / acs_total_housing_units,
        NA_real_
      ),
      parcel_minus_acs_area_weighted_units =
        parcel_residential_units - acs_total_housing_units_area_weighted,
      conservative_parcel_minus_acs_area_weighted_units =
        parcel_residential_units_conservative - acs_total_housing_units_area_weighted,
      parcel_to_acs_area_weighted_ratio = if_else(
        acs_total_housing_units_area_weighted > 0,
        parcel_residential_units / acs_total_housing_units_area_weighted,
        NA_real_
      ),
      conservative_parcel_to_acs_area_weighted_ratio = if_else(
        acs_total_housing_units_area_weighted > 0,
        parcel_residential_units_conservative / acs_total_housing_units_area_weighted,
        NA_real_
      ),
      parcel_minus_acs_parcel_count_weighted_units =
        parcel_residential_units - acs_total_housing_units_parcel_count_weighted,
      conservative_parcel_minus_acs_parcel_count_weighted_units =
        parcel_residential_units_conservative - acs_total_housing_units_parcel_count_weighted,
      parcel_to_acs_parcel_count_weighted_ratio = if_else(
        acs_total_housing_units_parcel_count_weighted > 0,
        parcel_residential_units / acs_total_housing_units_parcel_count_weighted,
        NA_real_
      ),
      conservative_parcel_to_acs_parcel_count_weighted_ratio = if_else(
        acs_total_housing_units_parcel_count_weighted > 0,
        parcel_residential_units_conservative / acs_total_housing_units_parcel_count_weighted,
        NA_real_
      ),
      full_count_moe_flag = case_when(
        parcel_residential_units > acs_total_housing_units + acs_total_housing_units_moe ~ "parcel_above_acs_moe",
        parcel_residential_units < acs_total_housing_units - acs_total_housing_units_moe ~ "parcel_below_acs_moe",
        TRUE ~ "within_acs_moe"
      ),
      area_weighted_moe_flag = case_when(
        parcel_residential_units >
          acs_total_housing_units_area_weighted + acs_total_housing_units_moe_area_weighted ~
          "parcel_above_acs_moe",
        parcel_residential_units <
          acs_total_housing_units_area_weighted - acs_total_housing_units_moe_area_weighted ~
          "parcel_below_acs_moe",
        TRUE ~ "within_acs_moe"
      ),
      parcel_count_weighted_moe_flag = case_when(
        parcel_residential_units >
          acs_total_housing_units_parcel_count_weighted + acs_total_housing_units_moe_parcel_count_weighted ~
          "parcel_above_acs_moe",
        parcel_residential_units <
          acs_total_housing_units_parcel_count_weighted - acs_total_housing_units_moe_parcel_count_weighted ~
          "parcel_below_acs_moe",
        TRUE ~ "within_acs_moe"
      )
    )

  validation %>%
    st_drop_geometry() %>%
    write_csv(file.path(OUTPUT_DIR, paste0("unit_calibration_", output_prefix, "_validation.csv")))

  save_output(
    validation,
    file.path(OUTPUT_DIR, paste0("unit_calibration_", output_prefix, "_validation.rds")),
    paste0(output_prefix, "-level unit calibration validation")
  )

  summary <- bind_rows(
    tibble(
      metric_group = paste0("acs_", output_prefix, "_validation"),
      metric = c(
        "acs_year",
        paste0(output_prefix, "_count_intersecting_austin_full_purpose"),
        "acs_total_housing_units_full_geographies",
        "acs_total_housing_units_area_weighted",
        "parcel_residential_units_in_validation_geographies",
        "conservative_parcel_residential_units_in_validation_geographies",
        "parcel_corporate_units_in_validation_geographies",
        "parcel_minus_acs_full_units",
        "parcel_minus_acs_area_weighted_units",
        "parcel_minus_acs_parcel_count_weighted_units",
        "parcel_to_acs_full_ratio",
        "parcel_to_acs_area_weighted_ratio",
        "parcel_to_acs_parcel_count_weighted_ratio",
        "conservative_parcel_to_acs_full_ratio",
        "conservative_parcel_to_acs_area_weighted_ratio",
        "conservative_parcel_to_acs_parcel_count_weighted_ratio",
        "full_count_above_acs_moe_geographies",
        "full_count_below_acs_moe_geographies",
        "full_count_within_acs_moe_geographies",
        "area_weighted_above_acs_moe_geographies",
        "area_weighted_below_acs_moe_geographies",
        "area_weighted_within_acs_moe_geographies",
        "parcel_count_weighted_above_acs_moe_geographies",
        "parcel_count_weighted_below_acs_moe_geographies",
        "parcel_count_weighted_within_acs_moe_geographies",
        "acs_total_housing_units_parcel_count_weighted"
      ),
      value = c(
        ACS_YEAR,
        nrow(validation),
        sum(validation$acs_total_housing_units, na.rm = TRUE),
        sum(validation$acs_total_housing_units_area_weighted, na.rm = TRUE),
        sum(validation$parcel_residential_units, na.rm = TRUE),
        sum(validation$parcel_residential_units_conservative, na.rm = TRUE),
        sum(validation$parcel_corporate_units, na.rm = TRUE),
        sum(validation$parcel_minus_acs_full_units, na.rm = TRUE),
        sum(validation$parcel_minus_acs_area_weighted_units, na.rm = TRUE),
        sum(validation$parcel_minus_acs_parcel_count_weighted_units, na.rm = TRUE),
        sum(validation$parcel_residential_units, na.rm = TRUE) /
          sum(validation$acs_total_housing_units, na.rm = TRUE),
        sum(validation$parcel_residential_units, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_area_weighted, na.rm = TRUE),
        sum(validation$parcel_residential_units, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_parcel_count_weighted, na.rm = TRUE),
        sum(validation$parcel_residential_units_conservative, na.rm = TRUE) /
          sum(validation$acs_total_housing_units, na.rm = TRUE),
        sum(validation$parcel_residential_units_conservative, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_area_weighted, na.rm = TRUE),
        sum(validation$parcel_residential_units_conservative, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_parcel_count_weighted, na.rm = TRUE),
        sum(validation$full_count_moe_flag == "parcel_above_acs_moe", na.rm = TRUE),
        sum(validation$full_count_moe_flag == "parcel_below_acs_moe", na.rm = TRUE),
        sum(validation$full_count_moe_flag == "within_acs_moe", na.rm = TRUE),
        sum(validation$area_weighted_moe_flag == "parcel_above_acs_moe", na.rm = TRUE),
        sum(validation$area_weighted_moe_flag == "parcel_below_acs_moe", na.rm = TRUE),
        sum(validation$area_weighted_moe_flag == "within_acs_moe", na.rm = TRUE),
        sum(validation$parcel_count_weighted_moe_flag == "parcel_above_acs_moe", na.rm = TRUE),
        sum(validation$parcel_count_weighted_moe_flag == "parcel_below_acs_moe", na.rm = TRUE),
        sum(validation$parcel_count_weighted_moe_flag == "within_acs_moe", na.rm = TRUE),
        sum(validation$acs_total_housing_units_parcel_count_weighted, na.rm = TRUE)
      ),
      note = NA_character_
    )
  )

  print_progress(
    paste0(
      "Parcel units in ",
      output_prefix,
      " validation geographies: ",
      scales::comma(round(sum(validation$parcel_residential_units, na.rm = TRUE), 0))
    )
  )
  print_progress(
    paste0(
      "ACS area-weighted ",
      output_prefix,
      " units: ",
      scales::comma(round(sum(validation$acs_total_housing_units_area_weighted, na.rm = TRUE), 0))
    )
  )
  print_progress(
    paste0(
      "Parcel / ACS area-weighted ",
      output_prefix,
      " ratio: ",
      round(
        sum(validation$parcel_residential_units, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_area_weighted, na.rm = TRUE),
        3
      )
    )
  )
  print_progress(
    paste0(
      "Parcel / ACS parcel-count-weighted ",
      output_prefix,
      " ratio: ",
      round(
        sum(validation$parcel_residential_units, na.rm = TRUE) /
          sum(validation$acs_total_housing_units_parcel_count_weighted, na.rm = TRUE),
        3
      )
    )
  )

  list(validation = validation, summary = summary)
}

tract_results <- validate_acs_geography("tract", "tract")
block_group_results <- validate_acs_geography("block group", "block_group")

write_overcount_diagnostics <- function(block_group_validation, parcels_sf) {
  if (is.null(block_group_validation)) {
    print_progress("Skipping overcount diagnostics because block group validation is unavailable.")
    return(invisible(NULL))
  }

  print_progress("Writing parcel-level diagnostics for mostly/full Austin block group overcounts...")

  block_group_targets <- block_group_validation %>%
    st_drop_geometry() %>%
    filter(
      austin_area_share >= 0.95,
      parcel_minus_acs_full_units > 0
    ) %>%
    mutate(
      overcount_rank = min_rank(desc(parcel_minus_acs_full_units)),
      overcount_share_of_parcel_units = if_else(
        parcel_residential_units > 0,
        parcel_minus_acs_full_units / parcel_residential_units,
        NA_real_
      )
    )

  if (nrow(block_group_targets) == 0) {
    print_progress("No mostly/full Austin block groups have parcel counts above ACS.")
    return(invisible(NULL))
  }

  parcels_with_block_group <- parcels_sf %>%
    st_join(
      block_group_validation %>%
        select(
          GEOID,
          NAME,
          austin_area_share,
          acs_total_housing_units,
          acs_total_housing_units_moe,
          parcel_residential_units,
          parcel_residential_units_conservative,
          parcel_minus_acs_full_units,
          parcel_to_acs_full_ratio
        ),
      join = st_within,
      left = FALSE
    ) %>%
    st_drop_geometry() %>%
    filter(GEOID %in% block_group_targets$GEOID) %>%
    mutate(
      parcel_share_of_block_group_units = if_else(
        parcel_residential_units > 0,
        units_calibrated / parcel_residential_units,
        NA_real_
      ),
      parcel_share_of_block_group_conservative_units = if_else(
        parcel_residential_units_conservative > 0,
        units_calibrated_conservative / parcel_residential_units_conservative,
        NA_real_
      ),
      parcel_share_of_block_group_overcount = if_else(
        parcel_minus_acs_full_units > 0,
        units_calibrated / parcel_minus_acs_full_units,
        NA_real_
      ),
      calibrated_minus_raw_units = units_calibrated - units_raw
    )

  method_diagnostics <- parcels_with_block_group %>%
    group_by(GEOID, unit_estimation_method, unit_estimation_confidence) %>%
    summarise(
      method_parcels = n(),
      method_units = sum(units_calibrated, na.rm = TRUE),
      method_conservative_units = sum(units_calibrated_conservative, na.rm = TRUE),
      method_raw_units = sum(units_raw, na.rm = TRUE),
      method_improvement_sqft = sum(improvement_sqft, na.rm = TRUE),
      method_direct_costar_match_parcels = sum(has_direct_costar_calibration_match, na.rm = TRUE),
      method_commercial_mixed_zoning_parcels = sum(has_commercial_mixed_zoning, na.rm = TRUE),
      method_multifamily_like_parcels = sum(is_multifamily_like, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    left_join(
      block_group_targets %>%
        select(
          GEOID,
          NAME,
          overcount_rank,
          austin_area_share,
          acs_total_housing_units,
          acs_total_housing_units_moe,
          parcel_residential_units,
          parcel_minus_acs_full_units,
          parcel_to_acs_full_ratio
        ),
      by = "GEOID"
    ) %>%
    mutate(
      method_share_of_block_group_units = if_else(
        parcel_residential_units > 0,
        method_units / parcel_residential_units,
        NA_real_
      ),
      method_share_of_block_group_overcount = if_else(
        parcel_minus_acs_full_units > 0,
        method_units / parcel_minus_acs_full_units,
        NA_real_
      )
    ) %>%
    arrange(overcount_rank, desc(method_units))

  block_group_diagnostics <- block_group_targets %>%
    left_join(
      parcels_with_block_group %>%
        group_by(GEOID) %>%
        summarise(
          diagnostic_parcels = n(),
          diagnostic_units = sum(units_calibrated, na.rm = TRUE),
          diagnostic_units_conservative = sum(units_calibrated_conservative, na.rm = TRUE),
          retained_units = sum(units_calibrated[unit_estimation_method == "parcel_units_retained"], na.rm = TRUE),
          strong_mf_estimated_units = sum(
            units_calibrated[
              unit_estimation_method %in% c(
                "costar_sqft_per_unit_estimate_strong_mf",
                "conservative_costar_sqft_per_unit_estimate_strong_mf",
                "direct_costar_units"
              )
            ],
            na.rm = TRUE
          ),
          weak_mf_estimated_units = sum(
            units_calibrated[
              unit_estimation_method %in% c(
                "costar_sqft_per_unit_estimate_weak_mf",
                "conservative_costar_sqft_per_unit_estimate_weak_mf"
              )
            ],
            na.rm = TRUE
          ),
          single_family_fallback_units = sum(
            units_calibrated[
              unit_estimation_method %in% c("single_family_default_1", "single_family_fractional_fallback_1")
            ],
            na.rm = TRUE
          ),
          mixed_use_excluded_units = sum(
            units_calibrated[unit_estimation_method == "commercial_mixed_use_mf_estimate_excluded"],
            na.rm = TRUE
          ),
          direct_costar_match_units = sum(
            units_calibrated[has_direct_costar_calibration_match],
            na.rm = TRUE
          ),
          commercial_mixed_zoning_units = sum(
            units_calibrated[has_commercial_mixed_zoning],
            na.rm = TRUE
          ),
          multifamily_like_units = sum(
            units_calibrated[is_multifamily_like],
            na.rm = TRUE
          ),
          top_10_parcel_units = sum(
            head(sort(units_calibrated, decreasing = TRUE, na.last = NA), 10),
            na.rm = TRUE
          ),
          max_parcel_units = max(units_calibrated, na.rm = TRUE),
          .groups = "drop"
        ),
      by = "GEOID"
    ) %>%
    arrange(overcount_rank)

  parcel_diagnostics <- parcels_with_block_group %>%
    arrange(desc(parcel_minus_acs_full_units), desc(units_calibrated)) %>%
    select(
      GEOID,
      NAME,
      austin_area_share,
      acs_total_housing_units,
      acs_total_housing_units_moe,
      parcel_residential_units,
      parcel_minus_acs_full_units,
      parcel_to_acs_full_ratio,
      parcel_id,
      source_county,
      situs_address,
      situs_city,
      situs_zip,
      propertyChar_zoning,
      propertyProf_imprvStateCd,
      propertyProf_landStateCd,
      improvement_sqft,
      land_sqft,
      units_raw,
      units_calibrated,
      units_calibrated_conservative,
      calibrated_minus_raw_units,
      conservative_unit_delta,
      unit_estimation_method,
      unit_estimation_method_conservative,
      unit_estimation_confidence,
      unit_estimation_confidence_conservative,
      has_direct_costar_calibration_match,
      direct_costar_units,
      has_mf_zoning,
      has_commercial_mixed_zoning,
      is_multifamily_like,
      is_single_family_like,
      likely_derived_large_mf_units,
      corporate_units,
      is_corporate_owned,
      owner_names,
      parcel_share_of_block_group_units,
      parcel_share_of_block_group_conservative_units,
      parcel_share_of_block_group_overcount,
      unit_estimation_notes
    )

  write_csv(
    block_group_diagnostics,
    file.path(OUTPUT_DIR, "unit_overcount_block_group_diagnostics.csv")
  )
  write_csv(
    method_diagnostics,
    file.path(OUTPUT_DIR, "unit_overcount_method_diagnostics.csv")
  )
  write_csv(
    parcel_diagnostics,
    file.path(OUTPUT_DIR, "unit_overcount_parcel_diagnostics.csv")
  )

  print_progress(
    paste0(
      "Overcount diagnostics cover ",
      scales::comma(nrow(block_group_diagnostics)),
      " mostly/full Austin block groups and ",
      scales::comma(nrow(parcel_diagnostics)),
      " parcel records."
    )
  )

  invisible(
    list(
      block_groups = block_group_diagnostics,
      methods = method_diagnostics,
      parcels = parcel_diagnostics
    )
  )
}

overcount_diagnostics <- write_overcount_diagnostics(block_group_results$validation, parcels)

write_targeted_unit_adjustment <- function(block_group_validation, parcels_sf) {
  if (is.null(block_group_validation)) {
    print_progress("Skipping targeted unit adjustment because block group validation is unavailable.")
    return(tibble(
      metric_group = "targeted_unit_adjustment",
      metric = "targeted_adjustment_status",
      value = NA_real_,
      note = "Block group validation was unavailable."
    ))
  }

  print_progress("Writing targeted parcel unit counts for high-error block groups...")

  target_block_groups <- block_group_validation %>%
    filter(
      austin_area_share >= TARGETED_ADJUSTMENT_MIN_AUSTIN_AREA_SHARE,
      parcel_to_acs_full_ratio >= TARGETED_ADJUSTMENT_MIN_PARCEL_TO_ACS_RATIO,
      parcel_minus_acs_full_units >= TARGETED_ADJUSTMENT_MIN_UNIT_OVERCOUNT
    ) %>%
    select(
      GEOID,
      NAME,
      austin_area_share,
      acs_total_housing_units,
      parcel_residential_units,
      parcel_residential_units_conservative,
      parcel_minus_acs_full_units,
      parcel_to_acs_full_ratio
    )

  target_lookup <- parcels_sf %>%
    st_join(
      target_block_groups %>%
        select(GEOID, NAME),
      join = st_within,
      left = FALSE
    ) %>%
    filter(
      unit_estimation_method %in% TARGETED_ADJUSTMENT_METHODS,
      !has_direct_costar_calibration_match,
      units_calibrated >= TARGETED_ADJUSTMENT_MIN_PARCEL_UNITS
    ) %>%
    st_drop_geometry() %>%
    transmute(
      parcel_id,
      targeted_adjustment_block_group_geoid = GEOID,
      targeted_adjustment_block_group_name = NAME,
      targeted_unit_adjustment_applied = TRUE
    )

  targeted_parcels <- parcels_sf %>%
    st_drop_geometry() %>%
    left_join(target_lookup, by = "parcel_id") %>%
    mutate(
      targeted_unit_adjustment_applied = replace_na(targeted_unit_adjustment_applied, FALSE),
      units_calibrated_targeted = if_else(
        targeted_unit_adjustment_applied,
        units_calibrated_conservative,
        units_calibrated
      ),
      unit_estimation_method_targeted = if_else(
        targeted_unit_adjustment_applied,
        unit_estimation_method_conservative,
        unit_estimation_method
      ),
      unit_estimation_confidence_targeted = if_else(
        targeted_unit_adjustment_applied,
        unit_estimation_confidence_conservative,
        unit_estimation_confidence
      ),
      unit_estimation_notes_targeted = if_else(
        targeted_unit_adjustment_applied,
        unit_estimation_notes_conservative,
        unit_estimation_notes
      ),
      targeted_unit_delta = units_calibrated_targeted - units_calibrated,
      property_units_targeted = units_calibrated_targeted,
      corporate_units_targeted = if_else(
        is_corporate_owned,
        replace_na(units_calibrated_targeted, 0),
        0
      )
    )

  save_output(
    targeted_parcels,
    file.path(OUTPUT_DIR, "residential_parcels_unit_targeted.rds"),
    "targeted calibrated residential parcel unit counts"
  )

  write_csv(
    targeted_parcels,
    file.path(OUTPUT_DIR, "residential_parcels_unit_targeted.csv")
  )

  adjustment_diagnostics <- bind_rows(
    target_block_groups %>%
      st_drop_geometry() %>%
      transmute(
        metric_group = "targeted_adjustment_block_groups",
        metric = paste0(GEOID, " | ", NAME),
        value = parcel_residential_units - parcel_residential_units_conservative,
        note = paste0(
          "Primary units: ",
          round(parcel_residential_units, 0),
          "; conservative units: ",
          round(parcel_residential_units_conservative, 0),
          "; ACS units: ",
          round(acs_total_housing_units, 0),
          "; primary/ACS ratio: ",
          round(parcel_to_acs_full_ratio, 2)
        )
      ),
    tibble(
      metric_group = "targeted_unit_adjustment",
      metric = c(
        "targeted_block_groups",
        "targeted_parcels",
        "primary_units_total",
        "targeted_units_total",
        "targeted_minus_primary_units_total",
        "targeted_corporate_units_total"
      ),
      value = c(
        nrow(target_block_groups),
        sum(targeted_parcels$targeted_unit_adjustment_applied, na.rm = TRUE),
        sum(targeted_parcels$units_calibrated, na.rm = TRUE),
        sum(targeted_parcels$units_calibrated_targeted, na.rm = TRUE),
        sum(targeted_parcels$targeted_unit_delta, na.rm = TRUE),
        sum(targeted_parcels$corporate_units_targeted, na.rm = TRUE)
      ),
      note = c(
        paste0(
          "Criteria: Austin area share >= ",
          TARGETED_ADJUSTMENT_MIN_AUSTIN_AREA_SHARE,
          ", parcel/ACS ratio >= ",
          TARGETED_ADJUSTMENT_MIN_PARCEL_TO_ACS_RATIO,
          ", parcel minus ACS units >= ",
          TARGETED_ADJUSTMENT_MIN_UNIT_OVERCOUNT,
          ", parcel method in ",
          paste(TARGETED_ADJUSTMENT_METHODS, collapse = " / "),
          ", no direct CoStar match, parcel units >= ",
          TARGETED_ADJUSTMENT_MIN_PARCEL_UNITS
        ),
        NA_character_,
        NA_character_,
        NA_character_,
        NA_character_,
        NA_character_
      )
    )
  )

  write_csv(
    adjustment_diagnostics,
    file.path(OUTPUT_DIR, "targeted_unit_adjustment_diagnostics.csv")
  )

  print_progress(
    paste0(
      "Targeted adjustment applied to ",
      scales::comma(nrow(target_block_groups)),
      " block groups and ",
      scales::comma(sum(targeted_parcels$targeted_unit_adjustment_applied, na.rm = TRUE)),
      " parcels."
    )
  )
  print_progress(
    paste0(
      "Targeted units: ",
      scales::comma(round(sum(targeted_parcels$units_calibrated_targeted, na.rm = TRUE), 0)),
      " (",
      scales::comma(round(sum(targeted_parcels$targeted_unit_delta, na.rm = TRUE), 0)),
      " vs primary)."
    )
  )

  adjustment_diagnostics
}

targeted_adjustment_summary <- write_targeted_unit_adjustment(block_group_results$validation, parcels)

validation_summary <- bind_rows(
  tract_results$summary,
  block_group_results$summary,
  targeted_adjustment_summary,
  city_unit_benchmark %>%
    transmute(
      metric_group = "external_city_benchmark",
      metric = benchmark,
      value = units,
      note = "External citywide benchmark supplied by user; not small-geography ACS."
    )
)

write_csv(
  validation_summary,
  file.path(OUTPUT_DIR, "unit_calibration_validation_summary.csv")
)

# Preserve the original tract-summary filename for compatibility with earlier review.
write_csv(
  validation_summary,
  file.path(OUTPUT_DIR, "unit_calibration_tract_validation_summary.csv")
)

print_header("02e COMPLETE")
