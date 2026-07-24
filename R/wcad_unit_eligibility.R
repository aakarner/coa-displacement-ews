################################################################################
# Williamson Residential Unit Eligibility
################################################################################

wcad_property_unit_fields <- function() {
  c(
    "PropertyID",
    "QuickRefID",
    "PropertyTypeDesc",
    "TotalSqFtLivingArea",
    "LegalDescription",
    "PropertyComment",
    "DBA",
    "SubUnit",
    "PropertyLegalType",
    "UnitTypeKey",
    "UnitNumber",
    "CondoBuilding",
    "CondoPercentage",
    "CondoUnit"
  )
}

wcad_parcel_unit_fields <- function() {
  c(
    "propertyid",
    "parcelid",
    "unit",
    "building",
    "usedscrp"
  )
}

wcad_unit_attribute_columns <- function() {
  c(
    "wcad_property_id",
    "wcad_property_type",
    "wcad_living_area",
    "wcad_legal_description",
    "wcad_property_comment",
    "wcad_dba",
    "wcad_subunit",
    "wcad_legal_type",
    "wcad_unit_type",
    "wcad_unit_number",
    "wcad_condo_building",
    "wcad_condo_percentage",
    "wcad_condo_unit",
    "wcad_parcel_unit",
    "wcad_parcel_building",
    "wcad_use_description"
  )
}

load_wcad_unit_attributes <- function(
    property_file,
    parcel_file,
    compact_file,
    parcel_ids,
    refresh = FALSE) {
  required_files <- c(property_file, parcel_file)
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0L) {
    stop(
      "Missing Williamson appraisal source file(s): ",
      paste(missing_files, collapse = ", "),
      call. = FALSE
    )
  }

  parcel_ids <- sort(unique(as.character(parcel_ids)))
  parcel_ids <- parcel_ids[!is.na(parcel_ids) & nzchar(parcel_ids)]
  dir.create(dirname(compact_file), showWarnings = FALSE, recursive = TRUE)

  compact_covers_request <- FALSE
  if (file.exists(compact_file) && !refresh) {
    compact_ids <- readr::read_csv(
      compact_file,
      col_select = "parcel_id",
      col_types = readr::cols(.default = readr::col_character()),
      show_col_types = FALSE
    )$parcel_id
    compact_covers_request <- all(parcel_ids %in% compact_ids)
  }

  if (!compact_covers_request) {
    if (exists("print_progress", mode = "function")) {
      print_progress("Extracting compact Williamson unit and legal fields...")
    } else {
      message("Extracting compact Williamson unit and legal fields...")
    }

    property_fields <- wcad_property_unit_fields()
    parcel_fields <- wcad_parcel_unit_fields()
    property_header <- names(
      data.table::fread(property_file, nrows = 0L, showProgress = FALSE)
    )
    parcel_header <- names(
      data.table::fread(parcel_file, nrows = 0L, showProgress = FALSE)
    )
    missing_fields <- c(
      setdiff(property_fields, property_header),
      setdiff(parcel_fields, parcel_header)
    )
    if (length(missing_fields) > 0L) {
      stop(
        "Williamson sources are missing fields: ",
        paste(unique(missing_fields), collapse = ", "),
        call. = FALSE
      )
    }

    property_extract <- data.table::fread(
      property_file,
      select = property_fields,
      colClasses = "character",
      showProgress = FALSE
    ) %>%
      tibble::as_tibble() %>%
      dplyr::distinct(PropertyID, .keep_all = TRUE)

    parcel_extract <- data.table::fread(
      parcel_file,
      select = parcel_fields,
      colClasses = "character",
      showProgress = FALSE
    ) %>%
      tibble::as_tibble() %>%
      dplyr::group_by(propertyid) %>%
      dplyr::summarise(
        wcad_parcel_id = as.character(
          first_non_missing_unit_value(parcelid)
        ),
        wcad_parcel_unit = as.character(
          first_non_missing_unit_value(unit)
        ),
        wcad_parcel_building = as.character(
          first_non_missing_unit_value(building)
        ),
        wcad_use_description = as.character(
          first_non_missing_unit_value(usedscrp)
        ),
        .groups = "drop"
      )

    compact <- property_extract %>%
      dplyr::left_join(
        parcel_extract,
        by = c("PropertyID" = "propertyid"),
        relationship = "one-to-one"
      ) %>%
      dplyr::mutate(
        wcad_match_id = dplyr::coalesce(wcad_parcel_id, QuickRefID),
        parcel_id = paste0("WILLIAMSON:", wcad_match_id)
      ) %>%
      dplyr::filter(parcel_id %in% parcel_ids) %>%
      dplyr::select(parcel_id, dplyr::everything(), -wcad_match_id)

    readr::write_csv(compact, compact_file)
    rm(property_extract, parcel_extract, compact)
    invisible(gc())
  }

  attributes <- readr::read_csv(
    compact_file,
    col_types = readr::cols(.default = readr::col_character()),
    show_col_types = FALSE
  ) %>%
    dplyr::transmute(
      parcel_id = as.character(parcel_id),
      wcad_property_id = as.character(PropertyID),
      wcad_property_type = dplyr::na_if(PropertyTypeDesc, ""),
      wcad_living_area = unit_numeric(TotalSqFtLivingArea),
      wcad_legal_description = dplyr::na_if(LegalDescription, ""),
      wcad_property_comment = dplyr::na_if(PropertyComment, ""),
      wcad_dba = dplyr::na_if(DBA, ""),
      wcad_subunit = dplyr::na_if(SubUnit, ""),
      wcad_legal_type = dplyr::na_if(PropertyLegalType, ""),
      wcad_unit_type = dplyr::na_if(UnitTypeKey, ""),
      wcad_unit_number = dplyr::na_if(UnitNumber, ""),
      wcad_condo_building = dplyr::na_if(CondoBuilding, ""),
      wcad_condo_percentage = unit_numeric(CondoPercentage),
      wcad_condo_unit = dplyr::na_if(CondoUnit, ""),
      wcad_parcel_unit = dplyr::na_if(wcad_parcel_unit, ""),
      wcad_parcel_building = dplyr::na_if(wcad_parcel_building, ""),
      wcad_use_description = dplyr::na_if(wcad_use_description, "")
    ) %>%
    dplyr::distinct(parcel_id, .keep_all = TRUE)

  missing_parcel_ids <- setdiff(parcel_ids, attributes$parcel_id)
  if (length(missing_parcel_ids) > 0L) {
    stop(
      "Williamson compact extract is missing ",
      length(missing_parcel_ids),
      " requested parcel(s). Examples: ",
      paste(utils::head(missing_parcel_ids, 10L), collapse = ", "),
      call. = FALSE
    )
  }

  attributes %>%
    dplyr::filter(parcel_id %in% parcel_ids)
}

classify_wcad_unit_eligibility <- function(data) {
  required_columns <- c(
    "source_county",
    "model_improvement_sqft",
    wcad_unit_attribute_columns()
  )
  missing_columns <- setdiff(required_columns, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      "WCAD eligibility classification is missing columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  apartment_pattern <- paste0(
    "APARTMENT|(^|[^A-Z])APTS?([^A-Z]|$)|",
    "MULTI[- ]?FAMILY"
  )

  data %>%
    dplyr::mutate(
      wcad_evidence_text = stringr::str_to_upper(
        stringr::str_c(
          dplyr::coalesce(wcad_legal_description, ""),
          dplyr::coalesce(wcad_property_comment, ""),
          dplyr::coalesce(wcad_dba, ""),
          dplyr::coalesce(wcad_use_description, ""),
          sep = " | "
        )
      ),
      wcad_primary_evidence_text = stringr::str_to_upper(
        stringr::str_c(
          dplyr::coalesce(wcad_legal_description, ""),
          dplyr::coalesce(wcad_dba, ""),
          dplyr::coalesce(wcad_use_description, ""),
          sep = " | "
        )
      ),
      wcad_apartment_primary_signal = source_county == "Williamson" &
        stringr::str_detect(
          wcad_primary_evidence_text,
          apartment_pattern
        ),
      wcad_apartment_comment_signal = source_county == "Williamson" &
        stringr::str_detect(
          stringr::str_to_upper(
            dplyr::coalesce(wcad_property_comment, "")
          ),
          apartment_pattern
        ),
      wcad_apartment_signal = wcad_apartment_primary_signal |
        (
          wcad_property_type %in% c("C3", "C5") &
            wcad_apartment_comment_signal
        ),
      wcad_small_multifamily_units = dplyr::case_when(
        source_county == "Williamson" &
          stringr::str_detect(
            wcad_evidence_text,
            "FOURPLEX|4-PLEX"
          ) ~ 4,
        source_county == "Williamson" &
          stringr::str_detect(wcad_evidence_text, "TRIPLEX") ~ 3,
        source_county == "Williamson" &
          stringr::str_detect(wcad_evidence_text, "DUPLEX") ~ 2,
        TRUE ~ NA_real_
      ),
      wcad_residential_type = source_county == "Williamson" &
        wcad_property_type %in% c(
          "Residential",
          "Manufactured Home",
          "LTRR-Land Transitional Residential"
        ),
      wcad_condo_signal = source_county == "Williamson" &
        (
          wcad_legal_type == "C" |
            !is.na(wcad_condo_unit) |
            !is.na(wcad_condo_building) |
            stringr::str_detect(
              wcad_evidence_text,
              "CONDOMINIUM|(^|[^A-Z])CONDO([^A-Z]|$)"
            )
        ),
      wcad_non_unit_reference_account = source_county == "Williamson" &
        wcad_condo_signal &
        stringr::str_detect(wcad_evidence_text, "REFERENCE ONLY"),
      wcad_explicit_residential_unit_account = wcad_residential_type &
        wcad_condo_signal &
        !wcad_non_unit_reference_account &
        model_improvement_sqft > 0,
      wcad_apartment_model_candidate = source_county == "Williamson" &
        wcad_property_type %in% c("C3", "C5") &
        wcad_apartment_signal &
        !wcad_non_unit_reference_account &
        model_improvement_sqft > 0,
      wcad_nonresidential_condo_account = source_county == "Williamson" &
        !wcad_residential_type &
        wcad_condo_signal &
        !wcad_non_unit_reference_account &
        !wcad_apartment_model_candidate,
      wcad_non_unit_amenity_parcel = source_county == "Williamson" &
        wcad_residential_type &
        (
          is.na(model_improvement_sqft) |
            model_improvement_sqft <= 0
        ) &
        (
          stringr::str_detect(
            wcad_evidence_text,
            stringr::fixed("(PARK")
          ) |
            stringr::str_detect(wcad_evidence_text, "AMENIT")
        ),
      wcad_non_unit_transitional_land = source_county == "Williamson" &
        wcad_property_type == "LTRC-Land Transitional Commercial" &
        (
          is.na(model_improvement_sqft) |
            model_improvement_sqft <= 0
        ),
      wcad_nonresidential_account = source_county == "Williamson" &
        wcad_property_type == "C6" &
        wcad_use_description == "C6" &
        !wcad_residential_type &
        !wcad_condo_signal &
        !wcad_apartment_model_candidate &
        !wcad_non_unit_transitional_land,
      wcad_single_unit_rule_units = dplyr::if_else(
        wcad_residential_type &
          model_improvement_sqft > 0 &
          !wcad_explicit_residential_unit_account &
          is.na(wcad_small_multifamily_units) &
          !wcad_apartment_signal,
        1,
        NA_real_
      ),
      wcad_model_candidate_signal = wcad_apartment_model_candidate,
      wcad_unit_exclusion_reason = dplyr::case_when(
        wcad_non_unit_reference_account ~
          "williamson_reference_only_common_interest_account",
        wcad_nonresidential_condo_account ~
          "williamson_nonresidential_condominium_account",
        wcad_non_unit_amenity_parcel ~
          "williamson_park_or_amenity_parcel_without_units",
        wcad_non_unit_transitional_land ~
          "williamson_transitional_commercial_land_without_units",
        wcad_nonresidential_account ~
          "williamson_other_nonresidential_account",
        TRUE ~ NA_character_
      ),
      wcad_unit_exclude_from_universe =
        !is.na(wcad_unit_exclusion_reason),
      wcad_unit_review_reason = dplyr::case_when(
        source_county == "Williamson" &
          !wcad_unit_exclude_from_universe &
          (
            is.na(model_improvement_sqft) |
              model_improvement_sqft <= 0
          ) ~
          "williamson_zero_or_missing_residential_floor_area",
        source_county == "Williamson" &
          !wcad_unit_exclude_from_universe &
          !wcad_residential_type &
          wcad_condo_signal &
          !wcad_apartment_model_candidate ~
          "williamson_commercial_condominium_in_residential_extract",
        source_county == "Williamson" &
          !wcad_unit_exclude_from_universe &
          !wcad_residential_type &
          !wcad_apartment_model_candidate ~
          "williamson_nonresidential_type_in_residential_extract",
        source_county == "Williamson" &
          !wcad_unit_exclude_from_universe &
          wcad_residential_type &
          wcad_apartment_signal &
          is.na(wcad_small_multifamily_units) &
          !wcad_explicit_residential_unit_account ~
          "williamson_ambiguous_residential_apartment_text",
        TRUE ~ NA_character_
      ),
      wcad_unit_evidence_class = dplyr::case_when(
        wcad_explicit_residential_unit_account ~
          "wcad_explicit_residential_unit_account",
        !is.na(wcad_small_multifamily_units) ~
          "wcad_small_multifamily_legal_description",
        wcad_apartment_model_candidate ~
          "wcad_apartment_model_candidate",
        !is.na(wcad_single_unit_rule_units) ~
          "williamson_single_unit_rule",
        wcad_non_unit_reference_account ~
          "wcad_non_unit_reference_account",
        wcad_nonresidential_condo_account ~
          "wcad_nonresidential_condominium_account",
        wcad_non_unit_amenity_parcel ~
          "wcad_non_unit_park_or_amenity_parcel",
        wcad_non_unit_transitional_land ~
          "wcad_non_unit_transitional_commercial_land",
        wcad_nonresidential_account ~
          "wcad_other_nonresidential_account",
        !is.na(wcad_unit_review_reason) ~
          "county_source_review",
        TRUE ~ NA_character_
      )
    )
}

wcad_unit_eligibility_audit <- function(data) {
  data %>%
    dplyr::filter(
      source_county == "Williamson",
      wcad_unit_exclude_from_universe |
        !is.na(wcad_unit_review_reason)
    ) %>%
    dplyr::select(
      dplyr::any_of(
        c(
          "parcel_id",
          "source_county",
          "situs_address",
          "lat",
          "lon",
          "wcad_unit_evidence_class",
          "wcad_unit_exclusion_reason",
          "wcad_unit_review_reason",
          "units_raw",
          "improvement_sqft",
          "is_corporate_owned",
          "wcad_property_id",
          "wcad_property_type",
          "wcad_use_description",
          "wcad_legal_type",
          "wcad_legal_description",
          "wcad_property_comment",
          "wcad_dba",
          "wcad_apartment_primary_signal",
          "wcad_apartment_comment_signal",
          "wcad_condo_signal"
        )
      )
    )
}
