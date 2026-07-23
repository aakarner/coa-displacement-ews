################################################################################
# 02m - Audit Amenity Change Sources
################################################################################
#
# Downloads and audits three public establishment histories:
#   1. Texas sales-tax locations active at any point in the prior 48 months;
#   2. Texas mixed-beverage monthly reports; and
#   3. Austin food-establishment inspections for local corroboration.
#
# The sales-tax location history is the classification backbone because it has
# statewide location coverage, NAICS, first-sale dates, and out-of-business
# dates. Mixed-beverage and inspection records are corroborating sources; they
# are never added as duplicate opening events.
#
# Optional environment variables:
#   EWS_REFRESH_AMENITIES=true  Redownload cached API extracts
#   SOCRATA_APP_TOKEN=...       Optional Socrata app token
#
# Outputs:
#   output/amenity_source_candidates.rds
#   output/amenity_source_audit.csv
#   output/amenity_naics_qa.csv
#   output/amenity_category_year_qa.csv
#   output/amenity_window_change_qa.csv
#   output/amenity_event_date_qa.csv
#   output/amenity_cross_source_match_qa.csv
################################################################################

project_path <- function(...) {
  if (requireNamespace("here", quietly = TRUE)) {
    here::here(...)
  } else {
    file.path(getwd(), ...)
  }
}

source(project_path("R", "utils.R"))
source(project_path("R", "analysis_config.R"))

suppressPackageStartupMessages({
  library(dplyr)
  library(httr)
  library(lubridate)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

print_header("02m - AMENITY SOURCE AUDIT")

OUTPUT_DIR <- project_path("output")
RAW_DIR <- project_path("data", "raw_amenities")
TAXONOMY_FILE <- project_path("config", "amenity_categories.csv")

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)

analysis_as_of <- EWS_CONFIG$analysis_as_of_date
window_months <- as.integer(EWS_CONFIG$amenity_window_months)
recent_start <- analysis_as_of %m-% months(window_months) + days(1)
previous_start <- recent_start %m-% months(window_months)

if (window_months <= 0L || previous_start >= recent_start) {
  stop("Amenity window configuration is invalid.", call. = FALSE)
}

county_lookup <- c(
  `105` = "Hays",
  `227` = "Travis",
  `246` = "Williamson"
)
county_codes <- names(county_lookup)

refresh <- tolower(Sys.getenv("EWS_REFRESH_AMENITIES", unset = "false")) %in%
  c("1", "true", "yes")
app_token <- Sys.getenv("SOCRATA_APP_TOKEN", unset = "")

taxonomy <- read_csv(
  TAXONOMY_FILE,
  col_types = cols(
    naics = col_character(),
    category = col_character(),
    evidence_tier = col_character(),
    include_in_index = col_logical(),
    name_filter_required = col_logical(),
    name_pattern = col_character(),
    notes = col_character()
  )
)

if (anyDuplicated(taxonomy$naics)) {
  stop("Amenity taxonomy contains duplicate NAICS rows.", call. = FALSE)
}

as_api_date <- function(x, end_of_day = FALSE) {
  suffix <- if (end_of_day) "T23:59:59" else "T00:00:00"
  paste0(format(as.Date(x), "%Y-%m-%d"), suffix)
}

parse_api_date <- function(x) {
  as.Date(substr(as.character(x), 1L, 10L))
}

safe_min_date <- function(x) {
  x <- as.Date(x)
  if (all(is.na(x))) as.Date(NA) else min(x, na.rm = TRUE)
}

safe_max_date <- function(x) {
  x <- as.Date(x)
  if (all(is.na(x))) as.Date(NA) else max(x, na.rm = TRUE)
}

normalize_name <- function(x) {
  x <- str_to_upper(coalesce(as.character(x), ""))
  x <- str_replace_all(x, "[[:punct:]]+", " ")
  x <- str_replace_all(
    x,
    "\\b(LLC|L L C|INC|CORP|LTD|LP|COMPANY|CO)\\b",
    " "
  )
  str_squish(x)
}

normalize_address <- function(x) {
  x <- str_to_upper(coalesce(as.character(x), ""))
  x <- str_replace(x, ",.*$", "")
  x <- str_replace_all(x, "[[:punct:]]+", " ")
  x <- str_replace(
    x,
    "\\b(SUITE|STE|UNIT|APT|APARTMENT|ROOM|RM)\\b.*$",
    ""
  )
  replacements <- c(
    " STREET\\b" = " ST",
    " ROAD\\b" = " RD",
    " AVENUE\\b" = " AVE",
    " BOULEVARD\\b" = " BLVD",
    " DRIVE\\b" = " DR",
    " HIGHWAY\\b" = " HWY",
    " LANE\\b" = " LN",
    " PARKWAY\\b" = " PKWY"
  )
  for (pattern in names(replacements)) {
    x <- str_replace_all(x, pattern, replacements[[pattern]])
  }
  str_squish(x)
}

download_socrata_csv <- function(
    endpoint,
    select,
    where,
    cache_file,
    order = NULL,
    page_size = 50000L) {
  if (file.exists(cache_file) && !refresh) {
    print_progress(paste0("Using cached source: ", cache_file))
    return(read_csv(
      cache_file,
      col_types = cols(.default = col_character()),
      show_col_types = FALSE
    ))
  }

  print_progress(paste0("Downloading source: ", endpoint))
  headers <- if (nzchar(app_token)) {
    add_headers(`X-App-Token` = app_token)
  } else {
    add_headers(Accept = "text/csv")
  }

  chunks <- list()
  offset <- 0L
  repeat {
    query <- list(select, where, page_size, offset)
    names(query) <- c("$select", "$where", "$limit", "$offset")
    if (!is.null(order)) {
      query[["$order"]] <- order
    }

    response <- RETRY(
      "GET",
      endpoint,
      query = query,
      headers,
      times = 5,
      pause_base = 1,
      pause_cap = 15,
      terminate_on = c(400, 401, 403, 404),
      timeout(120),
      quiet = TRUE
    )
    stop_for_status(response)

    response_text <- content(response, as = "text", encoding = "UTF-8")
    chunk <- read_csv(
      I(response_text),
      col_types = cols(.default = col_character()),
      show_col_types = FALSE,
      progress = FALSE
    )
    if (nrow(chunk) == 0L) break

    chunks[[length(chunks) + 1L]] <- chunk
    if (nrow(chunk) < page_size) break
    offset <- offset + page_size
  }

  result <- bind_rows(chunks)
  write_csv(result, cache_file)
  print_progress(paste0("Cached ", nrow(result), " rows at: ", cache_file))
  result
}

naics_sql <- paste0("'", taxonomy$naics, "'", collapse = ",")
county_sql <- paste0("'", county_codes, "'", collapse = ",")

sales_cache <- file.path(
  RAW_DIR,
  "texas_permitted_sales_tax_target_amenities.csv"
)
sales <- download_socrata_csv(
  endpoint = "https://data.texas.gov/resource/3kx8-uryv.csv",
  select = paste(
    "tp_number", "loc_number", "loc_name", "address_number",
    "address_text", "permit_date", "juris_city", "loc_city",
    "loc_state", "loc_zip", "loc_county", "naics",
    "first_sale_date", "out_of_business_date",
    sep = ","
  ),
  where = paste0(
    "loc_county in (", county_sql, ") and naics in (", naics_sql, ")"
  ),
  cache_file = sales_cache,
  order = "tp_number,loc_number"
)

mixed_cache <- file.path(RAW_DIR, "texas_mixed_beverage_openings.csv")
mixed_beverage <- download_socrata_csv(
  endpoint = "https://data.texas.gov/resource/naix-2893.csv",
  select = paste(
    "taxpayer_number", "location_name", "location_address",
    "location_city", "location_state", "location_zip",
    "location_county", "tabc_permit_number",
    "responsibility_begin_date_yyyymmdd",
    "responsibility_end_date_yyyymmdd",
    "obligation_end_date_yyyymmdd", "total_receipts",
    sep = ","
  ),
  where = paste0(
    "location_county in (", county_sql, ")",
    " and responsibility_begin_date_yyyymmdd >= '",
    as_api_date(previous_start), "'",
    " and obligation_end_date_yyyymmdd <= '",
    as_api_date(analysis_as_of, end_of_day = TRUE), "'"
  ),
  cache_file = mixed_cache,
  order = "tabc_permit_number,obligation_end_date_yyyymmdd"
)

food_cache <- file.path(RAW_DIR, "austin_food_inspections.csv")
food_inspections <- download_socrata_csv(
  endpoint = "https://data.austintexas.gov/resource/ecmv-9xxi.csv",
  select = paste(
    "facility_id", "restaurant_name", "address", "zip_code",
    "inspection_date", "process_description",
    sep = ","
  ),
  where = paste0(
    "inspection_date >= '", as_api_date(previous_start), "'",
    " and inspection_date <= '",
    as_api_date(analysis_as_of, end_of_day = TRUE), "'"
  ),
  cache_file = food_cache,
  order = "facility_id,inspection_date"
)

################################################################################
# Classify and audit sales-tax locations
################################################################################

sales_duplicate_rows <- nrow(sales) - n_distinct(
  paste(sales$tp_number, sales$loc_number, sep = ":")
)

institutional_pattern <- paste(
  "SODEXO", "ARAMARK", "COMPASS GROUP", "CHARTWELLS", "DELAWARE NORTH",
  "LEVY", "SCHOOL", "UNIVERSITY", "HOSPITAL", "MEDICAL CENTER",
  sep = "|"
)

sales_categorized <- sales %>%
  transmute(
    event_id = paste(tp_number, loc_number, sep = ":"),
    taxpayer_number = tp_number,
    location_number = loc_number,
    location_name = loc_name,
    street = str_squish(paste(address_number, address_text)),
    city = coalesce(juris_city, loc_city),
    state = loc_state,
    zip = str_sub(loc_zip, 1L, 5L),
    county_code = loc_county,
    county = unname(county_lookup[loc_county]),
    naics = as.character(naics),
    permit_date = parse_api_date(permit_date),
    first_sale_date = parse_api_date(first_sale_date),
    out_of_business_date = parse_api_date(out_of_business_date)
  ) %>%
  distinct(event_id, .keep_all = TRUE) %>%
  left_join(taxonomy, by = "naics") %>%
  mutate(
    opening_date = first_sale_date,
    normalized_name = normalize_name(location_name),
    street_key = normalize_address(street),
    address_key = paste(street_key, zip, sep = "|"),
    name_filter_pass = !name_filter_required |
      str_detect(normalized_name, regex(coalesce(name_pattern, "$^"))),
    home_business_flag = str_detect(
      str_to_upper(coalesce(street, "")),
      "\\b(APT|APARTMENT|TRLR|TRAILER|LOT)\\b"
    ),
    institutional_flag = category == "full_service_restaurant" &
      str_detect(normalized_name, regex(institutional_pattern)),
    category_classified = if_else(
      category == "cafe" & !name_filter_pass,
      "other_snack_non_alcoholic",
      category
    ),
    record_available_as_of = is.na(permit_date) |
      permit_date <= analysis_as_of,
    source_eligible = !is.na(opening_date) & record_available_as_of &
      opening_date >= previous_start &
      opening_date <= analysis_as_of,
    core_index_eligible = include_in_index & name_filter_pass &
      !home_business_flag & !institutional_flag & source_eligible,
    event_window = case_when(
      opening_date >= recent_start & opening_date <= analysis_as_of ~ "recent",
      opening_date >= previous_start & opening_date < recent_start ~ "previous",
      TRUE ~ "outside"
    ),
    active_as_of = is.na(out_of_business_date) |
      out_of_business_date > analysis_as_of,
    first_of_month_flag = day(opening_date) == 1L,
    january_first_flag = month(opening_date) == 1L & day(opening_date) == 1L,
    permit_lag_days = as.integer(permit_date - first_sale_date),
    permit_after_cutoff_flag = !is.na(permit_date) &
      permit_date > analysis_as_of
  )

if (any(is.na(sales_categorized$county))) {
  stop("Sales-tax rows contain an unmapped county code.", call. = FALSE)
}

################################################################################
# Collapse corroborating sources to stable establishment records
################################################################################

mixed_rows <- mixed_beverage %>%
  transmute(
    tabc_permit_number,
    location_name,
    street = location_address,
    city = location_city,
    state = location_state,
    zip = str_sub(location_zip, 1L, 5L),
    county_code = location_county,
    county = unname(county_lookup[location_county]),
    responsibility_begin_date = parse_api_date(
      responsibility_begin_date_yyyymmdd
    ),
    responsibility_end_date = parse_api_date(
      responsibility_end_date_yyyymmdd
    ),
    obligation_end_date = parse_api_date(obligation_end_date_yyyymmdd),
    total_receipts = suppressWarnings(as.numeric(total_receipts)),
    normalized_name = normalize_name(location_name),
    street_key = normalize_address(street),
    address_key = paste(street_key, zip, sep = "|")
  ) %>%
  filter(
    !is.na(tabc_permit_number),
    responsibility_begin_date <= analysis_as_of,
    obligation_end_date <= analysis_as_of
  )

mixed_latest <- mixed_rows %>%
  arrange(tabc_permit_number, obligation_end_date) %>%
  group_by(tabc_permit_number) %>%
  slice_tail(n = 1L) %>%
  ungroup() %>%
  select(
    tabc_permit_number, location_name, street, city, state, zip,
    county_code, county, normalized_name, street_key, address_key
  )

mixed_summary <- mixed_rows %>%
  group_by(tabc_permit_number) %>%
  summarise(
    responsibility_begin_date = safe_min_date(responsibility_begin_date),
    responsibility_end_date = safe_max_date(responsibility_end_date),
    last_report_date = safe_max_date(obligation_end_date),
    receipts_through_cutoff = sum(total_receipts, na.rm = TRUE),
    reporting_months = n_distinct(obligation_end_date),
    .groups = "drop"
  ) %>%
  left_join(mixed_latest, by = "tabc_permit_number") %>%
  mutate(
    event_window = case_when(
      responsibility_begin_date >= recent_start &
        responsibility_begin_date <= analysis_as_of ~ "recent",
      responsibility_begin_date >= previous_start &
        responsibility_begin_date < recent_start ~ "previous",
      TRUE ~ "outside"
    )
  )

food_summary <- food_inspections %>%
  transmute(
    facility_id,
    restaurant_name,
    street = address,
    zip = str_sub(zip_code, 1L, 5L),
    inspection_date = parse_api_date(inspection_date),
    process_description,
    normalized_name = normalize_name(restaurant_name),
    street_key = normalize_address(street),
    address_key = paste(street_key, zip, sep = "|")
  ) %>%
  filter(!is.na(facility_id), inspection_date <= analysis_as_of) %>%
  arrange(facility_id, inspection_date) %>%
  group_by(facility_id) %>%
  summarise(
    restaurant_name = dplyr::last(restaurant_name),
    street = dplyr::last(street),
    zip = dplyr::last(zip),
    normalized_name = dplyr::last(normalized_name),
    street_key = dplyr::last(street_key),
    address_key = dplyr::last(address_key),
    first_inspection_date = safe_min_date(inspection_date),
    last_inspection_date = safe_max_date(inspection_date),
    inspection_count = n(),
    .groups = "drop"
  )

mixed_addresses <- unique(mixed_summary$address_key[nzchar(mixed_summary$address_key)])
food_addresses <- unique(food_summary$address_key[nzchar(food_summary$address_key)])

sales_categorized <- sales_categorized %>%
  mutate(
    mixed_beverage_address_match = address_key %in% mixed_addresses,
    austin_food_address_match = address_key %in% food_addresses
  )

################################################################################
# QA outputs
################################################################################

source_audit <- tribble(
  ~source, ~dataset_id, ~rows_downloaded, ~establishments, ~date_min,
  ~date_max, ~geographic_coverage, ~role, ~status, ~caveat,
  "Texas permitted sales-tax locations", "3kx8-uryv", nrow(sales),
  n_distinct(sales_categorized$event_id),
  safe_min_date(sales_categorized$opening_date[
    sales_categorized$opening_date <= analysis_as_of
  ]),
  safe_max_date(sales_categorized$opening_date[
    sales_categorized$opening_date <= analysis_as_of
  ]),
  "Hays, Travis, and Williamson counties", "classification backbone",
  "usable",
  "Rolling source retains locations active in the prior 48 months; equal 18-month windows stay inside retained history",
  "Texas mixed-beverage gross receipts", "naix-2893",
  nrow(mixed_beverage), n_distinct(mixed_summary$tabc_permit_number),
  safe_min_date(mixed_summary$responsibility_begin_date),
  safe_max_date(mixed_summary$responsibility_begin_date),
  "Hays, Travis, and Williamson counties", "alcohol corroboration",
  "usable",
  "Alcohol establishments are not added as duplicate events; reporting responsibility is not a business-type classification",
  "Austin food-establishment inspections", "ecmv-9xxi",
  nrow(food_inspections), n_distinct(food_summary$facility_id),
  safe_min_date(food_summary$first_inspection_date),
  safe_max_date(food_summary$last_inspection_date),
  "Austin Public Health reporting area", "local corroboration",
  "partial coverage",
  "Only the latest three years are published and first inspection is not treated as an opening date"
) %>%
  mutate(
    analysis_as_of_date = analysis_as_of,
    previous_window_start = previous_start,
    recent_window_start = recent_start,
    window_months = window_months,
    source_download_date = Sys.Date()
  )

naics_qa <- sales_categorized %>%
  group_by(
    county, naics, category, category_classified, evidence_tier,
    include_in_index
  ) %>%
  summarise(
    locations = n(),
    opening_window_locations = sum(source_eligible, na.rm = TRUE),
    core_index_locations = sum(core_index_eligible, na.rm = TRUE),
    home_business_flags = sum(home_business_flag, na.rm = TRUE),
    institutional_flags = sum(institutional_flag, na.rm = TRUE),
    permit_after_cutoff_flags = sum(permit_after_cutoff_flag, na.rm = TRUE),
    closed_by_as_of = sum(!active_as_of, na.rm = TRUE),
    .groups = "drop"
  )

category_year_qa <- sales_categorized %>%
  filter(source_eligible) %>%
  mutate(opening_year = year(opening_date)) %>%
  group_by(
    county, category_classified, evidence_tier, core_index_eligible,
    event_window, opening_year
  ) %>%
  summarise(
    openings = n(),
    closed_by_as_of = sum(!active_as_of, na.rm = TRUE),
    mixed_beverage_address_matches = sum(
      mixed_beverage_address_match,
      na.rm = TRUE
    ),
    austin_food_address_matches = sum(austin_food_address_match, na.rm = TRUE),
    .groups = "drop"
  )

event_date_qa <- sales_categorized %>%
  filter(source_eligible) %>%
  group_by(county, category_classified, event_window) %>%
  summarise(
    openings = n(),
    first_of_month_pct = 100 * mean(first_of_month_flag, na.rm = TRUE),
    january_first_pct = 100 * mean(january_first_flag, na.rm = TRUE),
    permit_after_first_sale_pct = 100 * mean(permit_lag_days > 0, na.rm = TRUE),
    permit_after_cutoff_count = sum(permit_after_cutoff_flag, na.rm = TRUE),
    median_permit_lag_days = median(permit_lag_days, na.rm = TRUE),
    .groups = "drop"
  )

core_events <- sales_categorized %>% filter(core_index_eligible)
cross_source_qa <- core_events %>%
  group_by(county, category_classified, event_window) %>%
  summarise(
    core_openings = n(),
    mixed_beverage_address_matches = sum(
      mixed_beverage_address_match,
      na.rm = TRUE
    ),
    mixed_beverage_match_pct = 100 * mean(
      mixed_beverage_address_match,
      na.rm = TRUE
    ),
    austin_food_address_matches = sum(austin_food_address_match, na.rm = TRUE),
    austin_food_match_pct = 100 * mean(austin_food_address_match, na.rm = TRUE),
    .groups = "drop"
  )

window_coverage <- core_events %>%
  count(county, category_classified, event_window) %>%
  complete(
    county = unname(county_lookup),
    category_classified = c("cafe", "full_service_restaurant", "drinking_place"),
    event_window = c("previous", "recent"),
    fill = list(n = 0L)
  )

window_change_qa <- window_coverage %>%
  pivot_wider(
    names_from = event_window,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(
    opening_change = recent - previous,
    opening_change_pct = if_else(
      previous > 0,
      100 * opening_change / previous,
      NA_real_
    ),
    analysis_as_of_date = analysis_as_of,
    previous_window_start = previous_start,
    recent_window_start = recent_start,
    window_months = window_months
  )

if (sales_duplicate_rows > 0L) {
  print_progress(paste0(
    "Collapsed ", sales_duplicate_rows,
    " duplicate sales-tax location row(s) by taxpayer and outlet number."
  ))
}
if (any(window_coverage$n == 0L)) {
  missing_cells <- window_coverage %>%
    filter(n == 0L) %>%
    transmute(label = paste(county, category_classified, event_window)) %>%
    pull(label)
  stop(
    "Core amenity source has an empty county/category/window cell: ",
    paste(missing_cells, collapse = "; "),
    call. = FALSE
  )
}
if (!any(!sales_categorized$active_as_of & sales_categorized$source_eligible)) {
  stop(
    "Sales-tax history did not retain any closed opening-window locations; ",
    "the expected four-year history may not have downloaded correctly.",
    call. = FALSE
  )
}

source_candidates <- list(
  sales_tax_locations = sales_categorized,
  mixed_beverage_locations = mixed_summary,
  food_inspection_locations = food_summary,
  taxonomy = taxonomy,
  analysis_as_of_date = analysis_as_of,
  previous_window_start = previous_start,
  recent_window_start = recent_start,
  window_months = window_months,
  source_download_date = Sys.Date()
)

save_output(
  source_candidates,
  file.path(OUTPUT_DIR, "amenity_source_candidates.rds"),
  "amenity source candidates"
)
write_csv(source_audit, file.path(OUTPUT_DIR, "amenity_source_audit.csv"))
write_csv(naics_qa, file.path(OUTPUT_DIR, "amenity_naics_qa.csv"))
write_csv(
  category_year_qa,
  file.path(OUTPUT_DIR, "amenity_category_year_qa.csv")
)
write_csv(
  window_change_qa,
  file.path(OUTPUT_DIR, "amenity_window_change_qa.csv")
)
write_csv(event_date_qa, file.path(OUTPUT_DIR, "amenity_event_date_qa.csv"))
write_csv(
  cross_source_qa,
  file.path(OUTPUT_DIR, "amenity_cross_source_match_qa.csv")
)

cat("\nAmenity source audit:\n")
print(source_audit %>% select(source, establishments, role, status))
cat("\nCore opening-window coverage:\n")
print(window_coverage)
cat("\nAmenity source audit complete.\n")
