# Evidence-based, mutually exclusive case categories; no production mutations.
suppressPackageStartupMessages({library(dplyr);library(readr);library(sf)})
root <- "output/eviction_low_unit_audit"
e <- readRDS(file.path(root,"evidence.rds")); s <- readRDS(file.path(root,"spatial.rds"))
pa <- readRDS(file.path(root,"polygon_accounts.rds"))
w <- read_csv("data/raw_parcels/williamson/wcad_property_certified.csv",show_col_types=FALSE,
  col_types=cols_only(PropertyID="c",QuickRefID="c",PropertyTypeDesc="c",PropertyComment="c",
    TotalSqFtLivingArea="d",LegalDescription="c",SitusAddress="c",PropertyAddress="c",DBA="c",PropertyStatusDesc="c")) %>%
  filter(paste0("WILLIAMSON:",QuickRefID) %in% s$links$parcel_id)
write_csv(w,file.path(root,"williamson_certified_attributes.csv"))
w$parcel_id <- paste0("WILLIAMSON:",w$QuickRefID)
markers <- "\\b(APARTMENTS?|APTS?|MULTI[- ]?FAMILY|DUPLEX|TRIPLEX|FOURPLEX|CONDOMINIUM|CONDO|TOWNHOME|TOWNHOUSE)\\b"
w$passes_text_rule <- grepl(markers,paste(w$PropertyAddress,w$SitusAddress,w$LegalDescription,w$PropertyComment,w$DBA),ignore.case=TRUE)
w$present_in_promoted <- w$parcel_id %in% e$parcels$parcel_id
w$present_in_upstream_input <- w$parcel_id %in% read_csv("data/williamson_residential_parcels_for_hex.csv",
  col_types=cols_only(parcel_id="c"),show_col_types=FALSE)$parcel_id
stopifnot(all(w$passes_text_rule==w$present_in_upstream_input),all(w$present_in_upstream_input==w$present_in_promoted))
write_csv(w,file.path(root,"williamson_upstream_filter_check.csv"))
land <- read_csv("data/austin_land_use_inventory_202607.csv",col_types=cols(.default="c"),show_col_types=FALSE)
land_codes <- read_csv("config/austin_land_use_codes.csv",col_types=cols(.default="c"),show_col_types=FALSE)
land_evidence <- s$links %>% filter(is.na(property_units),!is.na(parcel_id)) %>% distinct(event_hex_id,parcel_id) %>%
  left_join(land,by=c("parcel_id"="property_id"),na_matches="never",relationship="many-to-many") %>%
  left_join(land_codes,by=c("land_use"="land_use_code"),na_matches="never")
write_csv(land_evidence,file.path(root,"unlinked_parcel_land_use.csv"))
l <- s$links %>% mutate(category=case_when(
  !is.na(property_units) & property_units>0 & !is.na(unit_hex_id) & unit_hex_id!=event_hex_id ~ "Units counted in another cell: same parcel ID",
  !is.na(property_units) & property_units>0 & is.na(unit_hex_id) ~ "Unit reference point outside grid",
  parcel_id=="956770" ~ "Units counted in another cell: verified related account",
  parcel_id %in% w$parcel_id[!w$present_in_promoted] ~ "Williamson residential parcel omitted",
  is.na(polygon_row) ~ "Geocode just outside parcel boundary: review",
  TRUE ~ "Other parcel/account linkage requires review"))
# Case assignments remain unchanged. Multiple accepted address points in the same
# case must not inflate counts, and mixed-category cases remain explicit.
cr <- e$rows %>% distinct(case_number,hex_id,point_key) %>%
  left_join(l %>% select(point_key,category),by="point_key",relationship="many-to-many") %>%
  group_by(case_number,hex_id) %>% summarise(category=if(n_distinct(category)==1)first(category) else "Mixed evidence",.groups="drop")
stopifnot(nrow(cr)==1219L,!anyDuplicated(cr$case_number))
count <- cr %>% count(category,name="filings")
cells <- cr %>% count(hex_id,category,name="filings")
write_csv(cr,file.path(root,"case_audit_categories.csv"))
write_csv(count,file.path(root,"category_summary.csv"))
write_csv(cells,file.path(root,"cell_category_summary.csv"))
report <- e$targets %>% left_join(cells %>% group_by(hex_id) %>% summarise(
  findings=paste(paste0(category,": ",filings),collapse="; "),.groups="drop"),by="hex_id")
write_csv(report,file.path(root,"all_41_cells.csv"))
writeLines(c("| Cell | Current units | Filings | Evidence / filing count |","|---|---:|---:|---|",
  sprintf("| %s | %s | %s | %s |",report$hex_id,report$residential_units,report$eviction_recent_observed_cases,report$findings)),
  file.path(root,"all_41_cells.md"))
saveRDS(list(locations=l,cases=cr,counts=count,cells=cells,report=report,williamson=w),file.path(root,"summary.rds"))
print(count); print(w %>% select(QuickRefID,DBA,passes_text_rule,present_in_promoted),n=22,width=150)
print(cells,n=60,width=150)
