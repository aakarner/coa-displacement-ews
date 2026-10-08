# Housing evidence is independent of eviction occurrence and is shared by
# candidate recovery and downstream eligibility. C3/C5 alone are insufficient.
load_wcad_residential_evidence <- function(
    land_use_file = "data/austin_land_use_inventory_202607.csv",
    review_file = "config/wcad_residential_evidence_reviews.csv") {
  land <- readr::read_csv(land_use_file, col_types = readr::cols(.default = "c"), show_col_types = FALSE)
  city <- land %>%
    dplyr::filter(grepl("^R[0-9]+$", parcel_id_10),
      sub("^R", "", parcel_id_10) != property_id, land_use == "220") %>%
    dplyr::transmute(parcel_id = paste0("WILLIAMSON:", parcel_id_10),
      wcad_residential_evidence_source = "city_exact_parcel_apartment_condo") %>% dplyr::distinct()
  review <- readr::read_csv(review_file, col_types = readr::cols(.default = "c"), show_col_types = FALSE)
  stopifnot(!anyDuplicated(review$parcel_id), all(nzchar(review$evidence)),
    all(review$outcome %in% c("residential_multifamily", "non_unit_companion")))
  dplyr::bind_rows(city %>% dplyr::filter(!parcel_id %in% review$parcel_id),
    review %>% dplyr::transmute(parcel_id,
      wcad_residential_evidence_source = ifelse(outcome == "residential_multifamily",
        "reviewed_housing_inventory_and_county_geometry", "reviewed_non_unit_companion")))
}

wcad_corroborated_multifamily <- function(type, living_area, evidence, reference = FALSE) {
  type %in% c("C3", "C5") & is.finite(living_area) & living_area > 0 &
    !is.na(evidence) & evidence %in% c("city_exact_parcel_apartment_condo",
      "reviewed_housing_inventory_and_county_geometry") & !reference
}

wcad_nonreference_links <- function(certified, reviewed_file = "config/wcad_residential_geometry_reviews.csv") {
  refs <- certified %>% dplyr::filter(grepl("REFERENCE ONLY", LegalDescription, ignore.case = TRUE))
  matches <- stringr::str_match_all(refs$LegalDescription, "\\{(R[0-9]+)/NON-REF\\}")
  explicit <- dplyr::bind_rows(lapply(seq_along(matches), function(i) {
    if (!nrow(matches[[i]])) return(NULL)
    data.frame(geometry_source_parcel_id = refs$QuickRefID[i], certified_quick_ref_id = matches[[i]][,2],
      geometry_link_method = "certified_explicit_nonreference_account")
  }))
  reviewed <- readr::read_csv(reviewed_file, col_types = readr::cols(.default = "c"), show_col_types = FALSE) %>%
    dplyr::transmute(geometry_source_parcel_id, certified_quick_ref_id,
      geometry_link_method = "reviewed_residential_account_geometry")
  out <- dplyr::bind_rows(explicit, reviewed) %>% dplyr::distinct()
  if (anyDuplicated(out$geometry_source_parcel_id)) stop("Conflicting residential geometry-account links.")
  out
}
