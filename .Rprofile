local({
  project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)

  if (file.exists(file.path(project_root, "00_requirements.R"))) {
    r_minor <- strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1]
    project_library <- file.path(
      project_root,
      ".r-library",
      paste0("R-", R.version$major, ".", r_minor)
    )

    dir.create(project_library, recursive = TRUE, showWarnings = FALSE)
    .libPaths(c(project_library, .libPaths()))
  }
})
