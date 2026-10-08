# A second-stage assignment for already accepted cases. Raw ambiguity, court
# coverage and original geocodes remain authoritative audit evidence.
source("R/eviction_property_reviews.R")
eviction_property_geography_version <- function() "verified_residential_parcel_reference_v4"

read_eviction_property_geography <- function(path = "output/property_geography/eviction_address_properties.rds") {
  if (identical(Sys.getenv("EWS_EVICTION_GEOGRAPHY"), "raw")) return(NULL)
  if (!file.exists(path)) stop("Build scripts/data/build_eviction_property_geography.R first.")
  manifest_path <- file.path(dirname(path), "property_geography_manifest.json")
  manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
  if (!identical(manifest$rule, eviction_property_geography_version()))
    stop("Stale property geography rule: rebuild property geography.")
  for (i in seq_len(nrow(manifest$inputs))) {
    if (!file.exists(manifest$inputs$path[i]) || !identical(digest::digest(
      file = manifest$inputs$path[i], algo = "sha256"), manifest$inputs$sha256[i]))
      stop("Stale property geography: rebuild after input change: ", manifest$inputs$path[i])
  }
  for (i in seq_len(nrow(manifest$outputs))) {
    verify_property_review_file(manifest$outputs$path[i], manifest$outputs$sha256[i])
  }
  readRDS(path)
}

apply_eviction_property_geography <- function(resolved, evidence, geography, coverage, hex_counties, city_hexes) {
  if (is.null(geography)) return(resolved)
  keys <- c("source_county", "address_for_geocoding")
  if (anyDuplicated(geography[keys])) stop("Duplicate property-address crosswalk keys.")
  for (name in c("property_facility_type", "property_denominator_status"))
    if (!name %in% names(geography)) geography[[name]] <- NA_character_
  if (!"property_address_review_id" %in% names(geography)) geography$property_address_review_id <- NA_character_
  rows <- evidence %>% sf::st_drop_geometry() %>%
    dplyr::select(dplyr::all_of(c("case_number", keys))) %>% dplyr::distinct() %>%
    dplyr::left_join(geography %>% dplyr::select(dplyr::all_of(keys), property_id,
      property_hex_id, property_link_status, reference_distance_m,
      property_facility_type, property_denominator_status, property_address_review_id), by = keys)
  rows <- apply_property_case_reviews(rows, resolved$cases, attr(geography, "case_reviews"))
  proposals <- rows %>% dplyr::group_by(case_number) %>% dplyr::summarise(
    all_addresses_verified = all(!is.na(property_link_status) & property_link_status == "verified"),
    property_count = dplyr::n_distinct(property_id, na.rm = TRUE),
    property_hex_count = dplyr::n_distinct(property_hex_id, na.rm = TRUE),
    property_id = if (dplyr::n_distinct(property_id, na.rm = TRUE) == 1L) first(stats::na.omit(property_id)) else NA_character_,
    property_hex_key = if (dplyr::n_distinct(property_hex_id, na.rm = TRUE) == 1L) as.character(first(stats::na.omit(property_hex_id))) else NA_character_,
    property_reference_distance_m = if (any(is.finite(reference_distance_m))) max(reference_distance_m, na.rm = TRUE) else NA_real_,
    property_review_id = first(property_review_id),
    property_review_basis = first(property_review_basis),
    property_apartment_conflict = any(property_apartment_conflict),
    property_address_review_ids = paste(sort(unique(stats::na.omit(property_address_review_id))), collapse = ";"),
    property_facility_type = paste(sort(unique(stats::na.omit(property_facility_type))), collapse = " | "),
    property_denominator_status = paste(sort(unique(stats::na.omit(property_denominator_status))), collapse = " | "),
    .groups = "drop")
  cases <- resolved$cases %>% dplyr::mutate(original_assigned_hex_key = assigned_hex_key) %>%
    dplyr::left_join(proposals, by = "case_number") %>%
    dplyr::left_join(hex_counties %>% dplyr::transmute(property_hex_key = as.character(hex_id),
      property_source_county = source_county), by = "property_hex_key") %>%
    dplyr::left_join(coverage %>% dplyr::transmute(property_hex_key = as.character(hex_id), outcome_year,
      property_source_covered = source_covered, property_coverage_jp = coverage_jp_district),
      by = c("property_hex_key", "outcome_year")) %>%
    dplyr::mutate(property_geography_rule = eviction_property_geography_version(),
      property_assignment_status = dplyr::case_when(
        assignment_status != "assigned_unique_hex" ~ "original_case_exclusion_preserved",
        !dplyr::coalesce(all_addresses_verified, FALSE) ~ "unverified_property_link_original_hex_retained",
        property_count != 1L | property_hex_count != 1L ~ "conflicting_property_links_original_hex_retained",
        !property_hex_key %in% as.character(city_hexes) ~ "property_outside_city_scope_original_hex_retained",
        is.na(property_source_county) | property_source_county != source_county ~ "property_county_conflict_original_hex_retained",
        source_county == "Williamson" & (!dplyr::coalesce(property_source_covered, FALSE) |
          !dplyr::coalesce(source_jp_district == property_coverage_jp, FALSE)) ~ "property_court_conflict_original_hex_retained",
        property_hex_key == original_assigned_hex_key ~ "verified_same_hex",
        TRUE ~ "verified_reassigned_to_unit_hex"),
      assigned_hex_key = dplyr::if_else(property_assignment_status == "verified_reassigned_to_unit_hex",
        property_hex_key, original_assigned_hex_key))
  stopifnot(identical(cases$assignment_status, resolved$cases$assignment_status),
    sum(!is.na(cases$assigned_hex_key)) == sum(!is.na(resolved$cases$assigned_hex_key)))
  resolved$cases <- cases
  assigned_name <- if ("assigned_cases" %in% names(resolved)) "assigned_cases" else "assigned"
  a <- resolved[[assigned_name]]
  a$assigned_hex_key <- cases$assigned_hex_key[match(a$case_number, cases$case_number)]
  resolved[[assigned_name]] <- a
  resolved
}
