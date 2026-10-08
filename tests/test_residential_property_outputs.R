# Independent reconciliation of the repaired inventory and both case consumers.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
p <- readRDS("output/residential_parcels_unit_promoted.rds")
stopifnot(!anyDuplicated(p$parcel_id))
recovered <- read_csv("output/williamson_residential_geometry_links.csv", show_col_types=FALSE)
stopifnot(all(paste0("WILLIAMSON:",recovered$certified_quick_ref_id) %in% p$parcel_id),
  all(c("WILLIAMSON:R577451","WILLIAMSON:R577471","WILLIAMSON:R577682","WILLIAMSON:R605025") %in% p$parcel_id),
  !any(c("WILLIAMSON:R039681","WILLIAMSON:R065343","WILLIAMSON:R065344",
    "WILLIAMSON:R081221","WILLIAMSON:R401725","WILLIAMSON:R072533","WILLIAMSON:R081218") %in% p$parcel_id))
g <- readRDS("output/property_geography/eviction_address_properties.rds")
stopifnot(!anyDuplicated(g[c("source_county","address_for_geocoding")]),
  all(!is.na(g$property_hex_id[g$property_link_status=="verified"])))
a <- readRDS("output/part2/evictions/eviction_case_ledger.rds")
b <- read_csv("output/part3/eviction_property_assignment_ledger.csv",
  col_types=cols(.default=col_guess(),case_number="c",assigned_hex_key="c",original_assigned_hex_key="c",
    property_review_id="c",property_review_basis="c"))
shared <- inner_join(select(a,case_number,assignment_status,assigned_hex_key),
  select(b,case_number,assignment_status,assigned_hex_key),by="case_number",suffix=c("_paired","_annual"),relationship="one-to-one")
stopifnot(all(a$assignment_status[!a$case_number %in% b$case_number]=="excluded_missing_valid_case_identifier"),
  nrow(shared)>0L,
  identical(shared$assignment_status_paired,shared$assignment_status_annual),
  identical(shared$assigned_hex_key_paired,shared$assigned_hex_key_annual))
for(x in list(a,b)) {
  moved <- x$property_assignment_status=="verified_reassigned_to_unit_hex"
  accepted <- x$assignment_status=="assigned_unique_hex"
  stopifnot(any(moved),all(accepted[moved]),
    all(x$all_addresses_verified[moved]),all(x$property_count[moved]==1L),
    all(x$property_hex_count[moved]==1L),
    all(x$assigned_hex_key[moved]==x$property_hex_key[moved]),
    all(is.na(x$assigned_hex_key[!accepted])),
    identical(x$assigned_hex_key[!moved],x$original_assigned_hex_key[!moved]))
}
old_path <- "output/residential_geography_repair/before/output/part2/evictions/eviction_case_ledger.rds"
if(file.exists(old_path)) {
  old <- readRDS(old_path); old <- old[match(a$case_number,old$case_number),]
  source("tests/repaired_case_conservation.R")
  assert_repaired_raw_case_conservation(a, old, "assigned_hex_key")
}
annual <- read_csv("output/eviction_filings_complete_by_hex_year.csv",show_col_types=FALSE)
tally <- b %>% filter(assignment_status=="assigned_unique_hex") %>%
  count(hex_id=as.integer(assigned_hex_key),outcome_year,name="expected")
check <- annual %>% left_join(tally,by=c("hex_id","outcome_year"))
stopifnot(identical(is.na(check$eviction_cases_observed_to_date),!check$source_covered),
  all(check$eviction_cases_observed_to_date[check$source_covered]==coalesce(check$expected[check$source_covered],0L)))
cat("Repaired residential inventory, shared case assignments, preserved ambiguity and annual totals passed.\n")
