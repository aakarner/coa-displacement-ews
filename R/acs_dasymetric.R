################################################################################
# Shared ACS Dasymetric Allocation Helpers
################################################################################
#
# ACS block-group counts are allocated to project hexes using 2020 decennial
# Census block population or housing counts. Blocks outside the project grid
# remain in each source-zone denominator, so edge block groups are not wholly
# pulled into the study area. ACS medians are attached from the block group
# containing the largest residential ancillary share in each hex; medians are
# never averaged across source geographies.
################################################################################

load_census_block_ancillary <- function(
  cache_dir,
  counties,
  state = "TX",
  decennial_year = 2020L
) {
  dir.create(cache_dir, recursive = TRUE, showWarnings = FALSE)

  county_blocks <- lapply(counties, function(county_name) {
    cache_slug <- gsub("[^a-z0-9]+", "_", tolower(county_name))
    cache_file <- file.path(
      cache_dir,
      paste0("decennial_", decennial_year, "_blocks_", cache_slug, ".rds")
    )

    if (file.exists(cache_file)) {
      blocks <- readRDS(cache_file)
    } else {
      print_progress(
        paste0(
          "Downloading ", decennial_year, " Census block ancillary data for ",
          county_name, " County..."
        )
      )
      blocks <- tidycensus::get_decennial(
        geography = "block",
        variables = c(
          block_population = "P1_001N",
          block_housing_units = "H1_001N"
        ),
        state = state,
        county = county_name,
        year = decennial_year,
        sumfile = "pl",
        geometry = TRUE,
        output = "wide",
        cache_table = TRUE
      )
      saveRDS(blocks, cache_file)
    }

    required_columns <- c(
      "GEOID", "block_population", "block_housing_units", "geometry"
    )
    missing_columns <- setdiff(required_columns, names(blocks))
    if (length(missing_columns) > 0) {
      stop(
        "Cached Census blocks for ", county_name,
        " are missing: ", paste(missing_columns, collapse = ", "),
        call. = FALSE
      )
    }

    blocks %>%
      transmute(
        block_geoid = GEOID,
        block_population = pmax(
          0,
          replace_na(as.numeric(block_population), 0)
        ),
        block_housing_units = pmax(
          0,
          replace_na(as.numeric(block_housing_units), 0)
        ),
        block_source_county = county_name,
        geometry
      )
  })

  blocks <- bind_rows(county_blocks)
  if (anyDuplicated(blocks$block_geoid)) {
    stop("Duplicate Census block GEOIDs found across county caches.", call. = FALSE)
  }
  blocks
}

build_census_block_hex_allocation <- function(
  hex_grid,
  census_blocks,
  residential_parcels,
  analysis_crs = 3857
) {
  required_parcel_columns <- c(
    "improvement_sqft", "property_units", "parcel_count", "geometry"
  )
  missing_parcel_columns <- setdiff(
    required_parcel_columns,
    names(residential_parcels)
  )
  if (length(missing_parcel_columns) > 0) {
    stop(
      "Residential parcel support is missing: ",
      paste(missing_parcel_columns, collapse = ", "),
      call. = FALSE
    )
  }

  hex_projected <- hex_grid %>%
    select(hex_id, geometry) %>%
    st_transform(analysis_crs)
  block_projected <- census_blocks %>%
    select(
      block_geoid,
      block_population,
      block_housing_units,
      geometry
    ) %>%
    st_transform(analysis_crs)
  parcel_points <- residential_parcels %>%
    st_transform(analysis_crs) %>%
    transmute(
      parcel_support_weight = case_when(
        as.numeric(improvement_sqft) > 0 ~ as.numeric(improvement_sqft),
        as.numeric(property_units) > 0 ~ as.numeric(property_units) * 1000,
        as.numeric(parcel_count) > 0 ~ as.numeric(parcel_count) * 1000,
        TRUE ~ 1000
      ),
      geometry
    )

  parcel_block <- suppressWarnings(
    st_join(
      parcel_points,
      block_projected %>% select(block_geoid),
      join = st_within,
      left = FALSE
    )
  )
  block_support_totals <- parcel_block %>%
    st_drop_geometry() %>%
    group_by(block_geoid) %>%
    summarise(
      block_parcel_support_weight = sum(parcel_support_weight, na.rm = TRUE),
      block_parcel_support_points = n(),
      .groups = "drop"
    )

  parcel_block_hex <- suppressWarnings(
    st_join(
      parcel_block,
      hex_projected,
      join = st_within,
      left = FALSE
    )
  )
  parcel_supported_allocation <- parcel_block_hex %>%
    st_drop_geometry() %>%
    group_by(block_geoid, hex_id) %>%
    summarise(
      project_parcel_support_weight = sum(parcel_support_weight, na.rm = TRUE),
      project_parcel_support_points = n(),
      .groups = "drop"
    ) %>%
    left_join(block_support_totals, by = "block_geoid") %>%
    left_join(
      block_projected %>%
        st_drop_geometry() %>%
        select(block_geoid, block_population, block_housing_units),
      by = "block_geoid"
    ) %>%
    mutate(
      block_to_hex_share = if_else(
        block_parcel_support_weight > 0,
        project_parcel_support_weight / block_parcel_support_weight,
        0
      ),
      block_population_contribution = block_population * block_to_hex_share,
      block_housing_units_contribution =
        block_housing_units * block_to_hex_share,
      block_hex_allocation_method = "residential_parcel_floor_area_proxy"
    )

  fallback_blocks <- block_projected %>%
    filter(!block_geoid %in% block_support_totals$block_geoid)
  fallback_allocation <- suppressWarnings(
    fallback_blocks %>%
      st_point_on_surface() %>%
      st_join(hex_projected, join = st_within, left = FALSE)
  ) %>%
    st_drop_geometry() %>%
    transmute(
      block_geoid,
      hex_id,
      project_parcel_support_weight = 0,
      project_parcel_support_points = 0L,
      block_parcel_support_weight = 0,
      block_parcel_support_points = 0L,
      block_population,
      block_housing_units,
      block_to_hex_share = 1,
      block_population_contribution = block_population,
      block_housing_units_contribution = block_housing_units,
      block_hex_allocation_method =
        "block_point_no_residential_parcel_support"
    )

  allocation <- bind_rows(
    parcel_supported_allocation,
    fallback_allocation
  ) %>%
    arrange(block_geoid, hex_id)

  qa <- tibble(
    metric = c(
      "residential_parcel_support_points_total",
      "residential_parcel_support_points_matched_to_block",
      "census_blocks_with_residential_parcel_support",
      "census_blocks_with_parcel_support_in_project",
      "census_blocks_using_point_fallback_in_project",
      "block_hex_allocation_rows",
      "block_population_allocated_with_parcel_support",
      "block_population_allocated_with_point_fallback",
      "block_housing_allocated_with_parcel_support",
      "block_housing_allocated_with_point_fallback"
    ),
    value = c(
      nrow(residential_parcels),
      nrow(parcel_block),
      nrow(block_support_totals),
      n_distinct(parcel_supported_allocation$block_geoid),
      n_distinct(fallback_allocation$block_geoid),
      nrow(allocation),
      sum(
        parcel_supported_allocation$block_population_contribution,
        na.rm = TRUE
      ),
      sum(fallback_allocation$block_population_contribution, na.rm = TRUE),
      sum(
        parcel_supported_allocation$block_housing_units_contribution,
        na.rm = TRUE
      ),
      sum(
        fallback_allocation$block_housing_units_contribution,
        na.rm = TRUE
      )
    )
  )

  list(allocation = allocation, qa = qa)
}

build_acs_hex_crosswalk <- function(
  hex_grid,
  source_geographies,
  census_blocks,
  block_hex_allocation = NULL,
  analysis_crs = 3857
) {
  required_source_columns <- c("source_geoid", "geometry")
  missing_source_columns <- setdiff(
    required_source_columns,
    names(source_geographies)
  )
  if (length(missing_source_columns) > 0) {
    stop(
      "ACS source geographies are missing: ",
      paste(missing_source_columns, collapse = ", "),
      call. = FALSE
    )
  }

  source_geographies <- source_geographies %>%
    select(source_geoid, any_of("source_name"), geometry) %>%
    distinct(source_geoid, .keep_all = TRUE) %>%
    st_transform(analysis_crs)
  hex_projected <- hex_grid %>%
    select(hex_id, geometry) %>%
    st_transform(analysis_crs)
  block_points <- suppressWarnings(
    census_blocks %>%
      st_transform(analysis_crs) %>%
      st_point_on_surface()
  )

  block_source <- suppressWarnings(
    st_join(
      block_points,
      source_geographies %>% select(source_geoid),
      join = st_within,
      left = FALSE
    )
  ) %>%
    st_drop_geometry() %>%
    select(
      block_geoid,
      block_population,
      block_housing_units,
      source_geoid
    ) %>%
    distinct(block_geoid, .keep_all = TRUE)

  if (is.null(block_hex_allocation)) {
    block_hex <- suppressWarnings(
      st_join(
        block_points %>%
          select(block_geoid, block_population, block_housing_units),
        hex_projected,
        join = st_within,
        left = FALSE
      )
    ) %>%
      st_drop_geometry() %>%
      transmute(
        block_geoid,
        hex_id,
        block_population_contribution = block_population,
        block_housing_units_contribution = block_housing_units,
        block_hex_allocation_method = "block_point_fallback"
      ) %>%
      distinct(block_geoid, .keep_all = TRUE)
  } else {
    block_hex <- block_hex_allocation %>%
      select(
        block_geoid,
        hex_id,
        block_population_contribution,
        block_housing_units_contribution,
        block_hex_allocation_method
      )
  }

  source_totals <- block_source %>%
    group_by(source_geoid) %>%
    summarise(
      source_block_count = n(),
      source_block_population = sum(block_population, na.rm = TRUE),
      source_block_housing_units = sum(block_housing_units, na.rm = TRUE),
      .groups = "drop"
    )

  crosswalk <- block_source %>%
    inner_join(block_hex, by = "block_geoid") %>%
    group_by(hex_id, source_geoid) %>%
    summarise(
      project_block_count = n_distinct(block_geoid),
      project_block_population = sum(
        block_population_contribution,
        na.rm = TRUE
      ),
      project_block_housing_units = sum(
        block_housing_units_contribution,
        na.rm = TRUE
      ),
      project_parcel_supported_block_count = n_distinct(
        block_geoid[
          block_hex_allocation_method ==
            "residential_parcel_floor_area_proxy"
        ]
      ),
      .groups = "drop"
    ) %>%
    left_join(source_totals, by = "source_geoid") %>%
    mutate(
      population_allocation_weight = case_when(
        source_block_population > 0 ~
          project_block_population / source_block_population,
        source_block_housing_units > 0 ~
          project_block_housing_units / source_block_housing_units,
        source_block_count > 0 ~ project_block_count / source_block_count,
        TRUE ~ 0
      ),
      population_allocation_basis = case_when(
        source_block_population > 0 ~ "2020_block_population",
        source_block_housing_units > 0 ~ "2020_block_housing_units_fallback",
        source_block_count > 0 ~ "2020_block_count_fallback",
        TRUE ~ "no_ancillary_support"
      ),
      housing_allocation_weight = case_when(
        source_block_housing_units > 0 ~
          project_block_housing_units / source_block_housing_units,
        source_block_population > 0 ~
          project_block_population / source_block_population,
        source_block_count > 0 ~ project_block_count / source_block_count,
        TRUE ~ 0
      ),
      housing_allocation_basis = case_when(
        source_block_housing_units > 0 ~ "2020_block_housing_units",
        source_block_population > 0 ~ "2020_block_population_fallback",
        source_block_count > 0 ~ "2020_block_count_fallback",
        TRUE ~ "no_ancillary_support"
      ),
      dominant_ancillary = case_when(
        project_block_housing_units > 0 ~ project_block_housing_units,
        project_block_population > 0 ~ project_block_population,
        TRUE ~ as.numeric(project_block_count)
      ),
      dominant_ancillary_basis = case_when(
        project_block_housing_units > 0 ~ "2020_block_housing_units",
        project_block_population > 0 ~ "2020_block_population",
        TRUE ~ "2020_block_count_fallback"
      )
    )

  dominant_source <- crosswalk %>%
    group_by(hex_id) %>%
    mutate(
      hex_ancillary_total = sum(dominant_ancillary, na.rm = TRUE),
      dominant_source_share = if_else(
        hex_ancillary_total > 0,
        dominant_ancillary / hex_ancillary_total,
        NA_real_
      )
    ) %>%
    arrange(hex_id, desc(dominant_ancillary), source_geoid) %>%
    slice_head(n = 1) %>%
    ungroup() %>%
    transmute(
      hex_id,
      dominant_source_geoid = source_geoid,
      dominant_source_share,
      dominant_source_method = dominant_ancillary_basis
    )

  missing_hex_points <- suppressWarnings(
    hex_projected %>%
      filter(!hex_id %in% dominant_source$hex_id) %>%
      st_point_on_surface()
  )
  missing_dominant_hexes <- missing_hex_points %>%
    st_join(
      source_geographies %>% select(source_geoid),
      join = st_within,
      left = TRUE
    ) %>%
    st_drop_geometry() %>%
    filter(!is.na(source_geoid)) %>%
    transmute(
      hex_id,
      dominant_source_geoid = source_geoid,
      dominant_source_share = NA_real_,
      dominant_source_method = "hex_point_fallback"
    )

  dominant_source <- bind_rows(dominant_source, missing_dominant_hexes) %>%
    distinct(hex_id, .keep_all = TRUE)

  qa <- tibble(
    metric = c(
      "census_blocks_total",
      "census_blocks_matched_to_source",
      "block_population_total",
      "block_population_matched_to_source",
      "block_housing_units_total",
      "block_housing_units_matched_to_source",
      "census_blocks_in_project_hexes",
      "project_hexes_total",
      "project_hexes_with_block_dominant_source",
      "project_hexes_with_point_fallback",
      "project_hexes_without_source",
      "source_geographies_population_fallback",
      "source_geographies_housing_fallback"
    ),
    value = c(
      nrow(census_blocks),
      nrow(block_source),
      sum(census_blocks$block_population, na.rm = TRUE),
      sum(block_source$block_population, na.rm = TRUE),
      sum(census_blocks$block_housing_units, na.rm = TRUE),
      sum(block_source$block_housing_units, na.rm = TRUE),
      n_distinct(block_hex$block_geoid),
      nrow(hex_grid),
      sum(dominant_source$dominant_source_method != "hex_point_fallback"),
      sum(dominant_source$dominant_source_method == "hex_point_fallback"),
      nrow(hex_grid) - nrow(dominant_source),
      sum(source_totals$source_block_population <= 0),
      sum(source_totals$source_block_housing_units <= 0)
    )
  )

  list(
    crosswalk = crosswalk,
    dominant_source = dominant_source,
    source_totals = source_totals,
    qa = qa
  )
}

allocate_acs_count_variables <- function(
  acs_long,
  crosswalk,
  population_variables,
  housing_variables
) {
  count_variables <- c(population_variables, housing_variables)
  variable_weight <- c(
    setNames(
      rep("population_allocation_weight", length(population_variables)),
      population_variables
    ),
    setNames(
      rep("housing_allocation_weight", length(housing_variables)),
      housing_variables
    )
  )

  source_values <- acs_long %>%
    filter(variable %in% count_variables) %>%
    st_drop_geometry() %>%
    transmute(
      source_geoid = GEOID,
      variable,
      estimate = as.numeric(estimate),
      moe = as.numeric(moe)
    )

  allocation_long <- crosswalk %>%
    select(
      hex_id,
      source_geoid,
      population_allocation_weight,
      housing_allocation_weight,
      population_allocation_basis,
      housing_allocation_basis
    ) %>%
    inner_join(
      source_values,
      by = "source_geoid",
      relationship = "many-to-many"
    ) %>%
    mutate(
      allocation_weight = if_else(
        variable_weight[variable] == "population_allocation_weight",
        population_allocation_weight,
        housing_allocation_weight
      ),
      allocation_basis = if_else(
        variable_weight[variable] == "population_allocation_weight",
        population_allocation_basis,
        housing_allocation_basis
      ),
      allocated_estimate = estimate * allocation_weight,
      allocated_moe = abs(moe * allocation_weight)
    )

  count_estimates <- allocation_long %>%
    group_by(hex_id, variable) %>%
    summarise(
      value = if (all(is.na(allocated_estimate))) {
        NA_real_
      } else {
        sum(allocated_estimate, na.rm = TRUE)
      },
      value_moe = if (all(is.na(allocated_moe))) {
        NA_real_
      } else {
        sqrt(sum(allocated_moe^2, na.rm = TRUE))
      },
      .groups = "drop"
    )

  estimate_wide <- count_estimates %>%
    select(hex_id, variable, value) %>%
    pivot_wider(names_from = variable, values_from = value)
  moe_wide <- count_estimates %>%
    select(hex_id, variable, value_moe) %>%
    mutate(variable = paste0(variable, "_moe")) %>%
    pivot_wider(names_from = variable, values_from = value_moe)

  project_source_weights <- crosswalk %>%
    group_by(source_geoid) %>%
    summarise(
      population_project_share = sum(
        population_allocation_weight,
        na.rm = TRUE
      ),
      housing_project_share = sum(
        housing_allocation_weight,
        na.rm = TRUE
      ),
      .groups = "drop"
    )

  conservation_qa <- source_values %>%
    left_join(project_source_weights, by = "source_geoid") %>%
    mutate(
      population_project_share = replace_na(population_project_share, 0),
      housing_project_share = replace_na(housing_project_share, 0),
      project_share = if_else(
        variable_weight[variable] == "population_allocation_weight",
        population_project_share,
        housing_project_share
      ),
      expected_project_estimate = estimate * project_share
    ) %>%
    group_by(variable) %>%
    summarise(
      source_zone_estimate_total = sum(estimate, na.rm = TRUE),
      expected_project_estimate = sum(
        expected_project_estimate,
        na.rm = TRUE
      ),
      allocated_project_estimate = sum(
        allocation_long$allocated_estimate[
          allocation_long$variable == first(variable)
        ],
        na.rm = TRUE
      ),
      conservation_difference =
        allocated_project_estimate - expected_project_estimate,
      .groups = "drop"
    )

  list(
    values = full_join(estimate_wide, moe_wide, by = "hex_id"),
    long = count_estimates,
    conservation_qa = conservation_qa
  )
}

assign_acs_median_variables <- function(
  acs_long,
  dominant_source,
  median_variables,
  source_geography
) {
  median_long <- dominant_source %>%
    inner_join(
      acs_long %>%
        filter(variable %in% median_variables) %>%
        st_drop_geometry() %>%
        transmute(
          dominant_source_geoid = GEOID,
          variable,
          estimate = as.numeric(estimate),
          moe = as.numeric(moe)
        ),
      by = "dominant_source_geoid",
      relationship = "many-to-many"
    )

  estimate_wide <- median_long %>%
    select(hex_id, variable, estimate) %>%
    pivot_wider(names_from = variable, values_from = estimate)
  moe_wide <- median_long %>%
    select(hex_id, variable, moe) %>%
    mutate(variable = paste0(variable, "_moe")) %>%
    pivot_wider(names_from = variable, values_from = moe)

  dominant_metadata <- dominant_source %>%
    transmute(
      hex_id,
      acs_source_geoid = dominant_source_geoid,
      acs_source_geography = source_geography,
      acs_source_residential_share = dominant_source_share,
      acs_source_assignment_method = dominant_source_method
    )

  dominant_metadata %>%
    left_join(estimate_wide, by = "hex_id") %>%
    left_join(moe_wide, by = "hex_id")
}

combine_acs_median_sources <- function(
  primary,
  fallback,
  median_variables
) {
  combine_variable <- function(variable) {
    moe_variable <- paste0(variable, "_moe")

    primary_variable <- primary %>%
      transmute(
        hex_id,
        primary_estimate = .data[[variable]],
        primary_moe = .data[[moe_variable]],
        primary_source_geoid = acs_source_geoid,
        primary_source_geography = acs_source_geography,
        primary_source_share = acs_source_residential_share,
        primary_source_method = acs_source_assignment_method
      )
    fallback_variable <- fallback %>%
      transmute(
        hex_id,
        fallback_estimate = .data[[variable]],
        fallback_moe = .data[[moe_variable]],
        fallback_source_geoid = acs_source_geoid,
        fallback_source_geography = acs_source_geography,
        fallback_source_share = acs_source_residential_share,
        fallback_source_method = acs_source_assignment_method
      )

    primary_variable %>%
      full_join(fallback_variable, by = "hex_id") %>%
      mutate(
        source_choice = case_when(
          !is.na(primary_estimate) ~ "primary",
          !is.na(fallback_estimate) ~ "fallback",
          TRUE ~ NA_character_
        )
      ) %>%
      transmute(
        hex_id,
        !!variable := coalesce(primary_estimate, fallback_estimate),
        !!moe_variable := coalesce(primary_moe, fallback_moe),
        !!paste0(variable, "_source_geoid") := case_when(
          source_choice == "primary" ~ primary_source_geoid,
          source_choice == "fallback" ~ fallback_source_geoid,
          TRUE ~ NA_character_
        ),
        !!paste0(variable, "_source_geography") := case_when(
          source_choice == "primary" ~ primary_source_geography,
          source_choice == "fallback" ~ fallback_source_geography,
          TRUE ~ NA_character_
        ),
        !!paste0(variable, "_source_residential_share") := case_when(
          source_choice == "primary" ~ primary_source_share,
          source_choice == "fallback" ~ fallback_source_share,
          TRUE ~ NA_real_
        ),
        !!paste0(variable, "_source_assignment_method") := case_when(
          source_choice == "primary" ~ primary_source_method,
          source_choice == "fallback" ~ fallback_source_method,
          TRUE ~ NA_character_
        )
      )
  }

  variable_tables <- lapply(median_variables, combine_variable)
  Reduce(
    function(left, right) full_join(left, right, by = "hex_id"),
    variable_tables
  )
}
