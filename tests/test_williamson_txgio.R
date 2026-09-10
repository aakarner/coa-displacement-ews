# Synthetic regressions; does not read private source inputs or use the network.
# Run from the repository root: Rscript tests/test_williamson_txgio.R
source("scripts/data/prepare_williamson_txgio.R")

expect_error <- function(expr, pattern) {
  result <- tryCatch(force(expr), error = identity)
  stopifnot(inherits(result, "error"), grepl(pattern, conditionMessage(result)))
}

good_members <- c("fgdb/test.gdb/", "fgdb/test.gdb/a.gdbtable", "shp/test.shp")
selected <- txgio_gdb_members(good_members)
stopifnot(identical(selected$root, "fgdb/test.gdb"), length(selected$members) == 2L)
for (unsafe in c("../outside", "/absolute", "C:/absolute", "fgdb/test.gdb/../../bad",
                 "fgdb\\test.gdb\\bad", "./fgdb/test.gdb/a")) {
  expect_error(txgio_gdb_members(c(good_members, unsafe)), "Unsafe")
}
expect_error(txgio_gdb_members(c(good_members, good_members[1])), "duplicate")
expect_error(txgio_gdb_members(c(good_members, "fgdb/other.gdb/a")), "Expected one")

rows <- as.data.frame(setNames(rep(list(rep("", 4)), length(WILLIAMSON_TXGIO_SOURCE_FIELDS)),
                               WILLIAMSON_TXGIO_SOURCE_FIELDS), stringsAsFactors = FALSE)
rows$Prop_ID <- c("R1", "R1", "R2", "R2")
rows$OBJECTID <- as.character(1:4)
rows$OWNER_NAME <- c(" Example LLC ", "EXAMPLE  LLC", "UNAVAILABLE", "UNAVAILABLE")
rows$MAIL_LINE1 <- c("PO BOX 1", "PO BOX 1", "UNAVAILABLE", "UNAVAILABLE")
rows$SITUS_ADDR <- c("10 A ST", "10 A ST", "20 B ST", "22 B ST")
rows$TAX_YEAR <- "2025"
rows$DATE_ACQ <- "20250701"
result <- txgio_annotate_evidence(rows)
stopifnot(nrow(result) == 4L,
          identical(result$OWNER_NAME, rows$OWNER_NAME),
          identical(result$MAIL_LINE1, rows$MAIL_LINE1),
          identical(result$source_evidence_is_duplicate, c(FALSE, TRUE, FALSE, FALSE)),
          identical(result$source_evidence_duplicate_count, c(2L, 2L, 1L, 1L)),
          identical(result$source_evidence_conflict, c(FALSE, FALSE, TRUE, TRUE)),
          length(unique(result$source_evidence_group)) == 3L)
# A difference in care-of or a decomposed situs field must not be discarded.
for (field in c("NAME_CARE", "SITUS_ST_2", "MAIL_LINE2")) {
  changed <- rows
  changed[[field]][2] <- "DIFFERENT"
  annotated <- txgio_annotate_evidence(changed)
  stopifnot(all(annotated$source_evidence_conflict[1:2]),
            !any(annotated$source_evidence_is_duplicate[1:2]))
}
stopifnot(identical(txgio_annotate_evidence(rows[4:1, ]), result))
bad <- rows; bad$OBJECTID[2] <- "1"
expect_error(txgio_annotate_evidence(bad), "OBJECTID")
bad <- rows; bad$SITUS_CITY <- NULL
expect_error(txgio_annotate_evidence(bad), "Missing TxGIO")

config <- list(path = "synthetic.zip", sha256 = paste(rep("a", 64), collapse = ""),
  source_url = "https://example.invalid/synthetic.zip", tax_year = 2025L,
  acquisition_date = "2025-07-01", layer = "synthetic_2025", county_rows = 4L, field_count = 37L)
txgio_validate_source_config(config, 2025L)
expect_error(txgio_validate_source_config(config, 2024L), "cross-year")
bad_config <- config; bad_config$acquisition_date <- "2024-07-01"
expect_error(txgio_validate_source_config(bad_config, 2025L), "cross-year")
bad_config <- config; bad_config$county_rows <- NULL
expect_error(txgio_validate_source_config(bad_config, 2025L), "Incomplete")
bad_config <- config; bad_config$sha256 <- "unverified"
expect_error(txgio_validate_source_config(bad_config, 2025L), "Invalid")
bad_config <- config; bad_config$layer <- "table; DROP TABLE"
expect_error(txgio_validate_source_config(bad_config, 2025L), "Invalid")
county <- data.frame(TAX_YEAR = rep("2025", 4), DATE_ACQ = "20250701",
                     COUNTY = "WILLIAMSON", FIPS = "48491")
txgio_validate_county(county, config)
bad <- county; bad$TAX_YEAR[1] <- "2026"
expect_error(txgio_validate_county(bad, config), "TAX_YEAR")
bad <- county; bad$DATE_ACQ[1] <- "20260701"
expect_error(txgio_validate_county(bad, config), "DATE_ACQ")
expect_error(txgio_validate_county(county[-1, ], config), "row count")
config_2024 <- config
config_2024$tax_year <- 2024L
config_2024$acquisition_date <- "2024-07-01"
config_2024$layer <- "synthetic_2024"
county_2024 <- county
county_2024$TAX_YEAR <- "2024"
county_2024$DATE_ACQ <- "20240701"
txgio_validate_county(county_2024, config_2024)
expect_error(txgio_validate_county(county_2024, config), "TAX_YEAR")
expect_error(prepare_williamson_txgio(tax_year = "2024;invalid"), "four-digit")

# One-pass county filtering must preserve target rows and duplicate annotation
# exactly, independently of which order the county returns the features in.
extra <- rows[1, ]
extra$Prop_ID <- "NON_TARGET"
extra$OBJECTID <- "99"
county_attributes <- rbind(extra, rows[4:1, ])
restricted <- county_attributes[county_attributes$Prop_ID %in% c("R1", "R2"), ]
stopifnot(identical(txgio_annotate_evidence(restricted), result))

message("Williamson TxGIO synthetic tests passed.")
