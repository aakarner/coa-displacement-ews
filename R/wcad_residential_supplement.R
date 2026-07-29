################################################################################
# WCAD Certified Residential Source-Gap Supplement
################################################################################

normalize_wcad_supplement_address <- function(x) {
  x %>%
    stringr::str_to_upper() %>%
    stringr::str_replace_all("\\bTEXAS\\b", "TX") %>%
    stringr::str_replace_all("[^A-Z0-9]+", " ") %>%
    stringr::str_squish() %>%
    dplyr::na_if("")
}

is_corporate_wcad_owner <- function(x) {
  owner <- stringr::str_to_upper(dplyr::coalesce(x, ""))
  entity_pattern <- paste0(
    "\\b(LLC|L L C|INC|CORP|CORPORATION|LTD|LIMITED|LP|L P|",
    "COMPANY|CO|PARTNERS|PARTNERSHIP|HOLDINGS|INVESTMENTS|",
    "PROPERTIES|REALTY|DEVELOPMENT|ASSOCIATES)\\b"
  )
  public_pattern <- paste0(
    "^(CITY OF|STATE OF|COUNTY OF|UNITED STATES|UNIVERSITY OF|",
    "SCHOOL DISTRICT|INDEPENDENT SCHOOL DISTRICT)"
  )

  stringr::str_detect(owner, entity_pattern) &
    !stringr::str_detect(owner, public_pattern)
}

build_wcad_certified_residential_supplement <- function(
    base_parcels,
    property_file,
    parcel_map_file,
    hex_grid_file,
    review_file) {
  required_base_columns <- c(
    "parcel_id",
    "situs_address",
    "source_county"
  )
  missing_base_columns <- setdiff(required_base_columns, names(base_parcels))
  if (length(missing_base_columns) > 0L) {
    stop(
      "WCAD supplement base data are missing: ",
      paste(missing_base_columns, collapse = ", "),
      call. = FALSE
    )
  }

  required_files <- c(
    property_file,
    parcel_map_file,
    hex_grid_file,
    review_file
  )
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0L) {
    stop(
      "WCAD supplement is missing: ",
      paste(missing_files, collapse = ", "),
      call. = FALSE
    )
  }

  review <- readr::read_csv(
    review_file,
    col_types = readr::cols(.default = readr::col_character()),
    show_col_types = FALSE
  )
  required_review_columns <- c(
    "review_layer",
    "source_county",
    "parcel_id",
    "review_outcome",
    "review_basis",
    "linked_parcel_id"
  )
  missing_review_columns <- setdiff(required_review_columns, names(review))
  if (length(missing_review_columns) > 0L) {
    stop(
      "Residual parcel review file is missing: ",
      paste(missing_review_columns, collapse = ", "),
      call. = FALSE
    )
  }

  hex_grid <- readRDS(hex_grid_file)
  previous_s2 <- sf::sf_use_s2()
  on.exit(sf::sf_use_s2(previous_s2), add = TRUE)
  suppressMessages(sf::sf_use_s2(FALSE))
  parcel_map <- readRDS(parcel_map_file) %>%
    dplyr::mutate(
      parcelid = as.character(parcelid),
      supplement_sort_value = dplyr::coalesce(
        readr::parse_number(as.character(cntassdval)),
        readr::parse_number(as.character(bldgarea)),
        readr::parse_number(as.character(resflrarea)),
        0
      )
    ) %>%
    dplyr::filter(!is.na(parcelid), nzchar(parcelid)) %>%
    dplyr::arrange(parcelid, dplyr::desc(supplement_sort_value)) %>%
    dplyr::distinct(parcelid, .keep_all = TRUE) %>%
    dplyr::select(-supplement_sort_value)

  spatial_crs <- 3857
  parcel_map_projected <- parcel_map %>%
    sf::st_transform(spatial_crs) %>%
    sf::st_make_valid()
  hex_grid_projected <- hex_grid %>%
    sf::st_transform(spatial_crs) %>%
    sf::st_make_valid()
  parcel_points <- suppressWarnings(
    sf::st_point_on_surface(parcel_map_projected)
  )
  in_grid <- lengths(
    sf::st_intersects(parcel_points, hex_grid_projected)
  ) > 0L
  parcel_points <- parcel_points[in_grid, ] %>%
    sf::st_transform(4326)
  coordinates <- sf::st_coordinates(parcel_points)

  parcel_point_attributes <- parcel_points %>%
    sf::st_drop_geometry() %>%
    tibble::as_tibble() %>%
    dplyr::transmute(
      geometry_source_parcel_id = as.character(parcelid),
      raw_property_id = as.character(propertyid),
      raw_site_address = as.character(siteaddress),
      raw_owner_name_1 = as.character(ownernme1),
      raw_owner_name_2 = as.character(ownernme2),
      raw_owner_address = as.character(pstladdres),
      raw_owner_city = as.character(pstlcity),
      raw_owner_state = as.character(pstlstate),
      raw_owner_zip = as.character(pstlzip5),
      raw_use_description = as.character(usedscrp),
      raw_use_code = as.character(usecd),
      raw_living_area = readr::parse_number(as.character(resflrarea)),
      raw_building_area = readr::parse_number(as.character(bldgarea)),
      raw_assessed_value = readr::parse_number(as.character(cntassdval)),
      lon = coordinates[, 1],
      lat = coordinates[, 2]
    )

  direct_geometry_links <- parcel_point_attributes %>%
    dplyr::transmute(
      certified_quick_ref_id = geometry_source_parcel_id,
      geometry_source_parcel_id,
      geometry_link_method = "matching_wcad_parcel_geometry"
    )
  reviewed_geometry_links <- review %>%
    dplyr::filter(
      review_layer == "full_parcel",
      source_county == "Williamson",
      review_outcome == "replace_with_certified_property",
      !is.na(linked_parcel_id),
      nzchar(linked_parcel_id)
    ) %>%
    dplyr::transmute(
      certified_quick_ref_id = sub(
        "^WILLIAMSON:",
        "",
        linked_parcel_id
      ),
      geometry_source_parcel_id = parcel_id,
      geometry_link_method = "reviewed_legacy_geometry_proxy"
    )
  geometry_links <- dplyr::bind_rows(
    direct_geometry_links,
    reviewed_geometry_links
  ) %>%
    dplyr::distinct(
      certified_quick_ref_id,
      geometry_source_parcel_id,
      .keep_all = TRUE
    )
  if (anyDuplicated(geometry_links$certified_quick_ref_id)) {
    stop(
      "A certified WCAD property has multiple supplement geometries.",
      call. = FALSE
    )
  }

  property_fields <- c(
    "PropertyID",
    "QuickRefID",
    "PropertyStatusDesc",
    "PropertyTypeDesc",
    "TotalSqFtLivingArea",
    "LegalDescription",
    "PropertyLegalType",
    "SitusAddress",
    "City",
    "State",
    "Zip",
    "Acres"
  )
  property_header <- names(
    data.table::fread(property_file, nrows = 0L, showProgress = FALSE)
  )
  missing_property_fields <- setdiff(property_fields, property_header)
  if (length(missing_property_fields) > 0L) {
    stop(
      "WCAD certified property file is missing: ",
      paste(missing_property_fields, collapse = ", "),
      call. = FALSE
    )
  }

  certified <- data.table::fread(
    property_file,
    select = property_fields,
    colClasses = "character",
    showProgress = FALSE
  ) %>%
    tibble::as_tibble() %>%
    dplyr::filter(QuickRefID %in% geometry_links$certified_quick_ref_id) %>%
    dplyr::mutate(
      certified_living_area = readr::parse_number(TotalSqFtLivingArea),
      certified_acres = readr::parse_number(Acres)
    )

  certified_conflicts <- certified %>%
    dplyr::group_by(QuickRefID) %>%
    dplyr::summarise(
      property_ids = dplyr::n_distinct(PropertyID),
      statuses = dplyr::n_distinct(PropertyStatusDesc),
      property_types = dplyr::n_distinct(PropertyTypeDesc),
      living_areas = dplyr::n_distinct(certified_living_area),
      addresses = dplyr::n_distinct(SitusAddress),
      .groups = "drop"
    ) %>%
    dplyr::filter(
      property_ids > 1L |
        statuses > 1L |
        property_types > 1L |
        living_areas > 1L |
        addresses > 1L
    )
  if (nrow(certified_conflicts) > 0L) {
    stop(
      "Conflicting duplicate rows occur in the WCAD certified source.",
      call. = FALSE
    )
  }

  certified <- certified %>%
    dplyr::distinct(QuickRefID, .keep_all = TRUE) %>%
    dplyr::filter(
      PropertyStatusDesc == "Active",
      PropertyTypeDesc == "Residential",
      is.finite(certified_living_area),
      certified_living_area > 0
    )

  base_williamson_ids <- base_parcels %>%
    dplyr::filter(source_county == "Williamson") %>%
    dplyr::pull(parcel_id) %>%
    sub("^WILLIAMSON:", "", .)
  supplement_audit <- geometry_links %>%
    dplyr::inner_join(
      certified,
      by = c("certified_quick_ref_id" = "QuickRefID"),
      relationship = "one-to-one"
    ) %>%
    dplyr::filter(!certified_quick_ref_id %in% base_williamson_ids) %>%
    dplyr::left_join(
      parcel_point_attributes,
      by = "geometry_source_parcel_id",
      relationship = "many-to-one"
    ) %>%
    dplyr::mutate(
      certified_address = dplyr::coalesce(
        dplyr::na_if(SitusAddress, ""),
        dplyr::na_if(raw_site_address, "")
      ),
      address_key = normalize_wcad_supplement_address(
        certified_address
      ),
      owner_names = stringr::str_c(
        dplyr::coalesce(raw_owner_name_1, ""),
        dplyr::coalesce(raw_owner_name_2, ""),
        sep = " & "
      ) %>%
        stringr::str_remove(" & $") %>%
        stringr::str_squish() %>%
        dplyr::na_if(""),
      is_corporate_owned = is_corporate_wcad_owner(owner_names),
      corporate_owned_flag = is_corporate_owned,
      repair_reason = paste(
        "Active certified WCAD Residential record with positive living area",
        "was absent from the broad Williamson parcel input."
      )
    ) %>%
    dplyr::arrange(certified_quick_ref_id)

  if (
    anyDuplicated(supplement_audit$certified_quick_ref_id) ||
      any(!is.finite(supplement_audit$lat)) ||
      any(!is.finite(supplement_audit$lon)) ||
      any(is.na(supplement_audit$address_key))
  ) {
    stop("WCAD supplement source crosswalk failed validation.", call. = FALSE)
  }

  base_address_keys <- normalize_wcad_supplement_address(
    base_parcels$situs_address
  )
  address_collisions <- supplement_audit %>%
    dplyr::filter(address_key %in% base_address_keys)
  if (nrow(address_collisions) > 0L) {
    stop(
      "WCAD supplement contains an address already in the parcel input.",
      call. = FALSE
    )
  }

  supplement <- supplement_audit %>%
    dplyr::transmute(
      parcel_id = paste0(
        "WILLIAMSON:",
        certified_quick_ref_id
      ),
      situs_address = certified_address,
      situs_city = dplyr::coalesce(
        dplyr::na_if(City, ""),
        dplyr::na_if(raw_owner_city, "")
      ),
      situs_state = dplyr::coalesce(
        dplyr::na_if(State, ""),
        "TX"
      ),
      situs_zip = dplyr::coalesce(
        dplyr::na_if(Zip, ""),
        dplyr::na_if(raw_owner_zip, "")
      ),
      propertyChar_zoning = NA_character_,
      propertyProf_imprvStateCd = NA_character_,
      propertyProf_landStateCd = NA_character_,
      propertyProf_imprvActualYearBuilt = NA_character_,
      improvement_sqft = as.character(certified_living_area),
      land_sqft = as.character(
        dplyr::coalesce(certified_acres, 0) * 43560
      ),
      property_units = "1",
      lat = as.character(lat),
      lon = as.character(lon),
      coord_source = dplyr::if_else(
        geometry_link_method == "reviewed_legacy_geometry_proxy",
        "wcad_reviewed_geometry_proxy",
        "wcad_certified_gap_repair"
      ),
      is_residential = "TRUE",
      is_owner_occupied = "FALSE",
      has_financialized_owner = "FALSE",
      is_corporate_owned = as.character(corporate_owned_flag),
      owner_names,
      n_owner_rows = "1",
      parcel_count = "1",
      corporate_parcel_count = as.character(
        as.integer(corporate_owned_flag)
      ),
      corporate_units = as.character(
        as.integer(corporate_owned_flag)
      ),
      corporate_improvement_sqft = as.character(
        dplyr::if_else(
          corporate_owned_flag,
          certified_living_area,
          0
        )
      ),
      source_county = "Williamson"
    )

  missing_output_columns <- setdiff(names(base_parcels), names(supplement))
  if (length(missing_output_columns) > 0L) {
    stop(
      "WCAD supplement cannot populate parcel schema fields: ",
      paste(missing_output_columns, collapse = ", "),
      call. = FALSE
    )
  }
  supplement <- supplement %>%
    dplyr::select(dplyr::all_of(names(base_parcels)))

  if (
    anyDuplicated(supplement$parcel_id) ||
      any(supplement$parcel_id %in% base_parcels$parcel_id)
  ) {
    stop("WCAD supplement contains duplicate parcel IDs.", call. = FALSE)
  }

  summary <- supplement_audit %>%
    dplyr::group_by(geometry_link_method) %>%
    dplyr::summarise(
      parcels = dplyr::n(),
      residential_units = dplyr::n(),
      certified_living_area = sum(certified_living_area),
      corporate_owned_parcels = sum(is_corporate_owned),
      .groups = "drop"
    )

  list(
    parcels = supplement,
    audit = supplement_audit,
    summary = summary
  )
}
