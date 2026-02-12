## Running Individual Scripts

If you want to run scripts individually (rather than the full `run_analysis.R`), you'll need to source the utility functions first:

```r
source("R/utils.R")
```

Then you can run individual scripts:

```r
source("R/01_create_hex_grid.R")
source("R/02_process_data.R")
# etc.
```

**Note**: The utility functions in `utils.R` (like `print_header()` and `print_progress()`) are used throughout the analysis scripts.