# End-to-end case conservation and alignment, independent of the matcher code.
suppressPackageStartupMessages({library(dplyr); library(readr)})
source("R/eviction_property_geography.R")
source("R/pipeline.R")
source("R/reviewed_unit_properties.R")
unit_reviews <- read_reviewed_unit_properties()
validate_reviewed_unit_surface(readRDS("output/residential_parcels_unit_promoted.rds"), unit_reviews)
root <- "output/property_review_production"
before <- file.path(root, "before")
g <- read_eviction_property_geography()
r <- attr(g, "case_reviews")
expected <- r %>% distinct(case_number, property_id, property_hex_id,
  property_review_id, property_review_basis, property_apartment_conflict)
stopifnot(nrow(expected) == 144L, !anyDuplicated(expected$case_number),
  sum(expected$property_apartment_conflict) == 1L,
  all(table(expected$property_hex_id)[c("6670", "6974", "6973", "6964")] == c(53, 1, 6, 11)))
stopifnot(all(table(expected$property_hex_id)[c("3321", "3261", "6929", "596")] == c(18, 9, 3, 1)))
stopifnot(all(table(expected$property_hex_id)[c("3767", "3769")] == c(10, 14)))
a <- readRDS("output/part2/evictions/eviction_case_ledger.rds")
b <- read_csv("output/part3/eviction_property_assignment_ledger.csv",
  col_types = cols(.default = col_guess(), case_number = "c", assigned_hex_key = "c", original_assigned_hex_key = "c",
    property_review_id = "c", property_review_basis = "c"), show_col_types = FALSE)
for (x in list(a, b)) {
  v <- x[match(expected$case_number, x$case_number), ]
  stopifnot(identical(v$case_number, expected$case_number),
    all(v$assignment_status == "assigned_unique_hex"), all(v$all_addresses_verified),
    all(v$property_count == 1L), all(v$property_hex_count == 1L),
    identical(v$assigned_hex_key, as.character(expected$property_hex_id)),
    identical(v$property_id, expected$property_id),
    identical(v$property_review_id, expected$property_review_id),
    identical(v$property_apartment_conflict, expected$property_apartment_conflict))
}
shared <- inner_join(select(a, case_number, assignment_status, assigned_hex_key),
  select(b, case_number, assignment_status, assigned_hex_key), by = "case_number", suffix = c("_paired", "_annual"), relationship = "one-to-one")
stopifnot(identical(shared$assignment_status_paired, shared$assignment_status_annual),
  identical(shared$assigned_hex_key_paired, shared$assigned_hex_key_annual))
if (dir.exists(before)) {
  for (pair in list(list(new = a, path = "output/part2/evictions/eviction_case_ledger.rds"),
                    list(new = b, path = "output/part3/eviction_property_assignment_ledger.csv"))) {
    path <- file.path(before, pair$path)
    old <- if (grepl("[.]rds$", path)) readRDS(path) else read_csv(path,
      col_types = cols(.default = col_guess(), case_number = "c", assigned_hex_key = "c", original_assigned_hex_key = "c"), show_col_types = FALSE)
    x <- pair$new; old <- old[match(x$case_number, old$case_number), ]
    precise <- !x$has_rejected_imprecise_geocode
    source("tests/repaired_case_conservation.R")
    grid_change <- assert_repaired_raw_case_conservation(x, old, "original_assigned_hex_key")
    unreviewed <- !x$case_number %in% expected$case_number
    # Decision 0017 recognizes two projects within county polygon 737155.
    # Withdraw its former blanket match to the neighboring 390-unit project.
    domain_fallback <- old$property_id %in% "parcel:737158" &
      x$property_assignment_status %in% "unverified_property_link_original_hex_retained" &
      x$assigned_hex_key %in% c("3455", "3459") &
      x$assigned_hex_key == x$original_assigned_hex_key
    domain_fallback[is.na(domain_fallback)] <- FALSE
    # The longer annual panel additionally contains one affected 2020 filing.
    stopifnot(sum(domain_fallback) == if (grepl("[.]rds$", path)) 36L else 37L)
    # Decision 0019 permits verified automatic links to the two reviewed
    # boundary references in the longer panel, beyond the 31 audited cases.
    boundary_reference <- x$property_id %in% c("parcel:911866", "parcel:WILLIAMSON:R500219")
    stopifnot(all(x$assigned_hex_key[boundary_reference] %in% c("3261", "6929")))
    alias <- !is.na(x$property_address_review_ids) & nzchar(x$property_address_review_ids)
    alias_config <- jsonlite::read_json("config/eviction_property_address_reviews.json")
    for (review in alias_config$reviews) {
      matched <- grepl(review$review_id,coalesce(x$property_address_review_ids,""),fixed=TRUE) & x$assignment_status == "assigned_unique_hex"
      stopifnot(all(x$property_id[matched] == review$project_id),
        all(x$assigned_hex_key[matched] == as.character(review$expected_unit_hex_id)))
    }
    # Rebuilt county crosswalk removes Olivine's former false Elm Ridge link
    # and recognizes only the explicitly reviewed new apartment/park groups.
    residual_auto <- x$property_id %in% c("parcel:549351","parcel:942518",
      "park_phase:545554","park_phase:545754","park_phase:292158","park_phase:291921","park_phase:191248")
    olivine_withdrawal <- old$property_id %in% "parcel:549351" & is.na(x$property_id) &
      x$property_assignment_status %in% "unverified_property_link_original_hex_retained"
    stopifnot(all(x$assigned_hex_key[olivine_withdrawal] == x$original_assigned_hex_key[olivine_withdrawal]))
    # Expanded coverage can now link an existing precise filing to a
    # verified residential reference that previously lay outside the grid.
    new_grid_property <- is.na(old$property_id) & !is.na(x$property_id) &
      x$property_assignment_status %in% "verified_reassigned_to_unit_hex" &
      as.integer(coalesce(x$assigned_hex_key,"0")) > 7027L
    stopifnot(all(x$all_addresses_verified[new_grid_property]),
      all(x$property_hex_count[new_grid_property] == 1L),
      all(x$assigned_hex_key[new_grid_property] == x$property_hex_key[new_grid_property]))
    unreviewed <- unreviewed & !new_grid_property & !domain_fallback & !boundary_reference & !alias & !residual_auto & !olivine_withdrawal & precise & !grid_change
    stopifnot(identical(x$assigned_hex_key[unreviewed], old$assigned_hex_key[unreviewed]),
      sum(!is.na(x$assigned_hex_key[precise & !grid_change])) == sum(!is.na(old$assigned_hex_key[precise & !grid_change])))
  }
  protected <- jsonlite::read_json(file.path(before, "protected_inputs.json"), simplifyVector = TRUE)
  # Only the two derived surfaces intentionally superseded by decision 0017.
  protected <- protected[!protected$path %in% c("output/residential_parcels_unit_promoted.rds",
    "output/corporate_ownership_by_hex.rds"), ]
  for (i in seq_len(nrow(protected))) verify_property_review_file(protected$path[i], protected$sha256[i])
  # Mapped totals may change under the precision rule; unchanged source case
  # identities above establish conservation. Annual/paired parity and the
  # independent event-output tests reconcile counts to the new case ledgers.

}
cat("144 reviewed cases align in annual and paired products; mixed-parcel fallbacks and reviewed boundary references preserve precise-location exclusions and source files.\n")
