# Assemble the six-cell follow-up from current production; evidence review only.
suppressPackageStartupMessages({library(dplyr); library(readr); library(sf)})
root <- "output/residential_property_batch2"
dir.create(root, recursive = TRUE, showWarnings = FALSE)
paths <- c(cases = "output/part2/evictions/eviction_case_ledger.rds",
  measurement = "output/part1/measurement/current_measurement.rds",
  original_audit = "output/residential_geography_repair/audited_case_outcomes.csv",
  evidence = "output/eviction_low_unit_audit/evidence.rds")
original <- read_csv(paths[["original_audit"]], show_col_types = FALSE)
m <- st_drop_geometry(readRDS(paths[["measurement"]]))
cases <- readRDS(paths[["cases"]]) %>% semi_join(original, by = "case_number") %>%
  mutate(hex_id = as.integer(assigned_hex_key)) %>%
  left_join(m %>% select(hex_id, residential_units), by = "hex_id") %>%
  filter(residential_units < 20)
stopifnot(nrow(cases) == 55L, !anyDuplicated(cases$case_number),
  setequal(cases$hex_id, c(602L, 3261L, 3319L, 3767L, 3769L, 6929L)))
evidence <- readRDS(paths[["evidence"]])$rows %>% semi_join(cases, by = "case_number")
stopifnot(setequal(evidence$case_number, cases$case_number))
groups <- tibble(hex_id = c(3319L, 3261L, 6929L, 602L, 3767L, 3769L),
  review_order = c(1L, 2L, 3L, 4L, 5L, 5L),
  property_situation = c("9009 N FM 620: adjacent-parcel/address discrepancy",
    "Caliza, 12638 Ridgeline: omitted account and grid boundary",
    "Nexus, 2001 E Slaughter: existing unit reference outside grid",
    "8000 W Highway 290: unresolved residential-account link",
    "Mobile-home area: reconcile parent, pad and individual-home accounts",
    "Mobile-home area: reconcile parent, pad and individual-home accounts"))
queue <- cases %>% left_join(groups, by = "hex_id") %>%
  mutate(review_status = "pending_property_evidence_review") %>%
  arrange(review_order, hex_id, case_number)
summary <- queue %>% group_by(review_order, hex_id, property_situation, residential_units) %>%
  summarise(filings = n(), .groups = "drop") %>% arrange(review_order, hex_id)
locations <- evidence %>% group_by(hex_id, st_addr, longitude, latitude) %>%
  summarise(filings = n_distinct(case_number), .groups = "drop")
write_csv(queue, file.path(root, "case_review_queue.csv"))
write_csv(evidence, file.path(root, "case_address_evidence.csv"))
write_csv(summary, file.path(root, "cell_review_summary.csv"))
write_csv(locations, file.path(root, "locations.csv"))
pins <- data.frame(path = unname(paths), sha256 = vapply(unname(paths),
  function(p) digest::digest(file = p, algo = "sha256"), character(1)))
jsonlite::write_json(list(status = "evidence_queue_only_no_production_changes",
  review_date = "2026-10-07", recent_window = c("2025-04-02", "2026-04-01"),
  cases = nrow(queue), cells = nrow(summary), inputs = pins),
  file.path(root, "queue_manifest.json"), pretty = TRUE, auto_unbox = TRUE)
print(summary, width = 160)
