# Rebuild the local stage and evaluate saved Census responses conservatively.
source("scripts/data/hays_evictions_geocode_local.R")
census_file <- "data/raw_hays_evictions/geocode_cache/census_response.csv"
census <- read.csv(census_file, header=FALSE, colClasses="character", fill=TRUE,
  na.strings=character(), col.names=c("address_id","input_address","match_indicator",
  "match_type","matched_address","coordinates","tigerline_id","side")) |> as_tibble()
needed_ids <- geocodes$address_id[geocodes$geocode_status!="matched_local_parcel_address"]
stopifnot(!anyDuplicated(census$address_id), all(needed_ids %in% census$address_id))
# A later local match can supersede a cached Census response. Retain that raw
# response in its cache, but evaluate only the currently unresolved addresses.
census <- census |> filter(address_id %in% needed_ids)
# Verify opaque IDs still identify the submitted address before reusing cache.
requests <- read.csv("data/raw_hays_evictions/geocode_cache/census_request.csv",
  header=FALSE,colClasses="character",col.names=c("address_id","street","city","state","zip"))
stopifnot(!anyDuplicated(requests$address_id))
checks <- requests |> filter(address_id %in% needed_ids) |>
  left_join(addresses |> select(address_id,street_key,postal_city,postal_zip),by="address_id")
stopifnot(nrow(checks)==length(needed_ids),
  all(normalize_street(checks$street)==checks$street_key),
  all(checks$city==checks$postal_city),all(checks$state=="TX"),all(checks$zip==checks$postal_zip))
parsed <- parse_address(census$matched_address)
census$matched_street_key <- parsed$street_key
census$matched_zip <- parsed$postal_zip
xy <- str_split_fixed(census$coordinates,",",2)
census$longitude <- suppressWarnings(as.numeric(xy[,1]))
census$latitude <- suppressWarnings(as.numeric(xy[,2]))
census <- census |> left_join(addresses |> select(address_id,street_key,postal_zip),by="address_id") |>
  mutate(accepted=coalesce(match_indicator=="Match" & match_type %in% c("Exact","Non_Exact") &
    matched_street_key==street_key & matched_zip==postal_zip &
    longitude > -107 & longitude < -93 & latitude > 25 & latitude < 37,FALSE),
    review_reason=case_when(match_indicator!="Match" ~ paste0("census_",tolower(match_indicator)),
      matched_street_key!=street_key ~ "street_mismatch_review",
      matched_zip!=postal_zip ~ "zip_mismatch_review",
      !accepted ~ "coordinate_or_parse_review",TRUE ~ "street_zip_match_interpolated_location"))
geocodes$geocode_method <- if_else(geocodes$geocode_status=="matched_local_parcel_address",
  "local_cad_parcel","unresolved")
geocodes$census_match_address <- ""
geocodes$census_review_reason <- ""
geocodes$point_boundary_distance_m <- NA_real_
for(i in seq_len(nrow(census))) {
  c <- census[i,]
  j <- match(c$address_id,geocodes$address_id)
  geocodes$census_match_address[j] <- c$matched_address
  geocodes$census_review_reason[j] <- c$review_reason
  if(!c$accepted) next
  point <- st_transform(st_sfc(st_point(c(c$longitude,c$latitude)),crs=4326),26914)
  inside <- lengths(st_covered_by(point,full))>0
  edge_distance <- as.numeric(st_distance(point,st_boundary(full)))
  geocodes$geocode_status[j] <- "matched_census_street_address"
  geocodes$geocode_method[j] <- "census_interpolated_street_location"
  geocodes$longitude[j] <- c$longitude
  geocodes$latitude[j] <- c$latitude
  geocodes$point_boundary_distance_m[j] <- round(edge_distance,1)
  geocodes$distance_to_austin_m[j] <- round(as.numeric(st_distance(point,full)),1)
  geocodes$candidate_boundary_status[j] <- if(edge_distance<=100) "within_100m_boundary_review" else if(inside) "inside_austin_full" else "outside_austin_full"
  geocodes$evidence[j] <- "Census matched normalized street and exact ZIP. Interpolated street location; unit and eviction premises remain unverified."
}
all_cases <- cases |> left_join(geocodes,by="candidate_premises_address") |>
  mutate(geocode_status=coalesce(geocode_status,"not_geocoded_address_review_required"),
    candidate_boundary_status=coalesce(candidate_boundary_status,"unresolved"))
stopifnot(nrow(all_cases)==nrow(cases),!anyDuplicated(all_cases$case_key))
write_csv(census,"output/hays_eviction_census_geocode_review.csv",na="")
write_csv(geocodes,"output/hays_eviction_candidate_addresses_geocoded.csv",na="")
write_csv(all_cases,"output/hays_eviction_current_window_geography_review.csv",na="")
write_csv(all_cases |> filter(candidate_boundary_status!="outside_austin_full"),
          "output/hays_eviction_current_window_geography_followup.csv",na="")
coa <- read_chars("output/hays_eviction_coa_geocode_candidates.csv")
# Austin returned no address-point/subaddress matches. Street-name and
# interpolated alternatives are retained for audit, not accepted automatically.
stopifnot(!any(coa$address_type %in% c("PointAddress","SubAddress")))
summary <- list(current_window_cases=nrow(all_cases),single_address_cases=nrow(candidate),
  unique_address_strings=nrow(geocodes),
  local_matched_case_count=sum(all_cases$geocode_status=="matched_local_parcel_address"),
  census_matched_case_count=sum(all_cases$geocode_status=="matched_census_street_address"),
  case_boundary_counts=as.list(table(all_cases$candidate_boundary_status)),
  unaccepted_coa_candidates=nrow(coa),census_accepted_address_strings=sum(census$accepted),
  census_unaccepted_address_strings=sum(!census$accepted),
  no_new_premises_verification=TRUE,no_scoring_or_coverage_changes=TRUE,
  review_log_sha256=digest::digest("data/hays_eviction_address_review.csv",algo="sha256",file=TRUE),
  census_response_sha256=digest::digest(census_file,algo="sha256",file=TRUE))
write_json(summary,"output/hays_eviction_geography_review_summary.json",pretty=TRUE,auto_unbox=TRUE)
cat(toJSON(summary,pretty=TRUE,auto_unbox=TRUE),"\n")
