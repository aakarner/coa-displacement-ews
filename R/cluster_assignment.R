################################################################################
# Fixed Baseline Cluster Assignment
################################################################################

plain_feature_data <- function(x) {
  if (inherits(x, "sf")) {
    return(sf::st_drop_geometry(x))
  }
  as.data.frame(x)
}

cluster_distance_matrix <- function(x, centers) {
  distances <- vapply(
    seq_len(nrow(centers)),
    function(center_index) {
      centered <- sweep(x, 2, centers[center_index, ], FUN = "-")
      sqrt(rowSums(centered^2))
    },
    numeric(nrow(x))
  )
  colnames(distances) <- rownames(centers)
  distances
}

scale_cluster_features <- function(data, model) {
  missing_features <- setdiff(model$features, names(data))
  if (length(missing_features) > 0L) {
    stop(
      "Cluster assignment data are missing: ",
      paste(missing_features, collapse = ", "),
      call. = FALSE
    )
  }

  x <- as.matrix(data[, model$features, drop = FALSE])
  storage.mode(x) <- "double"
  eligible <- rep(TRUE, nrow(data))
  if (
    !is.null(model$eligibility_feature) &&
      model$eligibility_feature %in% names(data)
  ) {
    eligible <- !is.na(data[[model$eligibility_feature]]) &
      as.logical(data[[model$eligibility_feature]])
  }
  complete <- eligible & apply(x, 1, function(row) all(is.finite(row)))
  x <- x[complete, , drop = FALSE]
  x <- sweep(x, 2, model$preprocessing$center, FUN = "-")
  x <- sweep(x, 2, model$preprocessing$scale, FUN = "/")

  list(matrix = x, complete = complete)
}

freeze_baseline_cluster_model <- function(
  feature_file,
  cluster_results_file,
  label_file,
  output_file,
  config
) {
  features <- plain_feature_data(readRDS(feature_file))
  results <- readRDS(cluster_results_file)
  labels <- readr::read_csv(label_file, show_col_types = FALSE)
  specification <- config$baseline_cluster_specification

  selected_k <- as.integer(results$selected_k[[specification]])
  fit <- results$full_evaluations[[specification]]$models[[
    as.character(selected_k)
  ]]
  model_features <- if (identical(specification, "baseline")) {
    results$baseline_vars
  } else {
    c(results$baseline_vars, results$amenity_var)
  }

  preprocessing <- results$scaling_parameters[[specification]]
  if (is.null(preprocessing)) {
    selected_training_hexes <- results$assignments$hex_id[
      results$assignments$specification == specification &
        results$assignments$k == selected_k
    ]
    training_rows <- features$hex_id %in% selected_training_hexes
    training_matrix <- as.matrix(
      features[training_rows, model_features, drop = FALSE]
    )
    storage.mode(training_matrix) <- "double"
    complete_training <- apply(
      training_matrix,
      1,
      function(row) all(is.finite(row))
    )
    training_matrix <- training_matrix[complete_training, , drop = FALSE]
    preprocessing <- list(
      center = colMeans(training_matrix),
      scale = apply(training_matrix, 2, stats::sd)
    )
  }
  preprocessing$center <- preprocessing$center[model_features]
  preprocessing$scale <- preprocessing$scale[model_features]
  if (
    any(!is.finite(preprocessing$center)) ||
      any(!is.finite(preprocessing$scale)) ||
      any(preprocessing$scale <= 0)
  ) {
    stop("Baseline cluster scaling parameters are invalid.", call. = FALSE)
  }

  provisional_model <- list(
    features = model_features,
    preprocessing = preprocessing,
    eligibility_feature = if (
      "primary_cluster_eligible" %in% names(features)
    ) {
      "primary_cluster_eligible"
    } else {
      "sufficient_data"
    }
  )
  scaled <- scale_cluster_features(features, provisional_model)
  complete_features <- features[scaled$complete, , drop = FALSE]
  distances <- cluster_distance_matrix(scaled$matrix, fit$centers)
  assigned_index <- max.col(-distances, ties.method = "first")
  assigned_cluster <- as.integer(rownames(fit$centers)[assigned_index])

  selected_assignments <- results$assignments[
    results$assignments$specification == specification &
      results$assignments$k == selected_k,
    c("hex_id", "cluster")
  ]
  if (
    nrow(selected_assignments) != nrow(complete_features) ||
      anyDuplicated(selected_assignments$hex_id)
  ) {
    stop(
      "Selected Part 1 assignments do not match the unique training hexes.",
      call. = FALSE
    )
  }
  assignment_check <- data.frame(
    hex_id = complete_features$hex_id,
    reassigned_cluster = assigned_cluster
  )
  assignment_check <- merge(
    assignment_check,
    selected_assignments,
    by = "hex_id",
    all.x = TRUE,
    sort = FALSE
  )
  mismatch_count <- sum(
    assignment_check$reassigned_cluster != assignment_check$cluster,
    na.rm = TRUE
  )
  if (mismatch_count > 0L || anyNA(assignment_check$cluster)) {
    stop(
      "Frozen-centroid reassignment did not reproduce the Part 1 solution: ",
      mismatch_count,
      " mismatches.",
      call. = FALSE
    )
  }

  minimum_distance <- distances[
    cbind(seq_len(nrow(distances)), assigned_index)
  ]
  second_distance <- apply(distances, 1, function(row) sort(row)[[2]])
  margin_confidence <- pmax(
    0,
    pmin(1, 1 - minimum_distance / second_distance)
  )

  distance_thresholds <- data.frame(
    cluster = sort(unique(assigned_cluster)),
    distance_threshold = vapply(
      sort(unique(assigned_cluster)),
      function(cluster_id) {
        stats::quantile(
          minimum_distance[assigned_cluster == cluster_id],
          probs = config$cluster_assignment_distance_quantile,
          na.rm = TRUE,
          names = FALSE
        )
      },
      numeric(1)
    )
  )
  margin_threshold <- stats::quantile(
    margin_confidence,
    probs = config$cluster_assignment_margin_quantile,
    na.rm = TRUE,
    names = FALSE
  )

  required_label_columns <- c(
    "solution_k", "cluster", "tentative_name", "concern_level",
    "map_color", "interpretation", "profile_anchor"
  )
  missing_label_columns <- setdiff(required_label_columns, names(labels))
  if (length(missing_label_columns) > 0L) {
    stop(
      "Cluster label configuration is missing: ",
      paste(missing_label_columns, collapse = ", "),
      call. = FALSE
    )
  }
  labels <- labels[
    labels$solution_k == selected_k,
    ,
    drop = FALSE
  ]
  if (
    nrow(labels) != selected_k ||
      anyDuplicated(labels$cluster) ||
      !setequal(labels$cluster, seq_len(selected_k))
  ) {
    stop(
      "Cluster labels do not uniquely cover the selected Part 1 solution.",
      call. = FALSE
    )
  }

  model <- list(
    schema_version = 1L,
    created_at = Sys.time(),
    analysis_as_of_date = config$analysis_as_of_date,
    h3_resolution = config$h3_resolution,
    specification = specification,
    k = selected_k,
    features = model_features,
    eligibility_feature = provisional_model$eligibility_feature,
    preprocessing = preprocessing,
    centroids = fit$centers,
    distance_metric = "euclidean_on_baseline_standardized_features",
    distance_threshold_quantile =
      config$cluster_assignment_distance_quantile,
    distance_thresholds = distance_thresholds,
    margin_confidence_definition = "1 - nearest_distance / second_distance",
    margin_threshold_quantile = config$cluster_assignment_margin_quantile,
    margin_threshold = as.numeric(margin_threshold),
    labels = labels,
    training_hex_ids = complete_features$hex_id,
    training_assignment = assigned_cluster,
    training_minimum_distance = minimum_distance,
    training_margin_confidence = margin_confidence
  )

  dir.create(dirname(output_file), recursive = TRUE, showWarnings = FALSE)
  saveRDS(model, output_file)
  output_file
}

assign_fixed_clusters <- function(features, model) {
  data <- plain_feature_data(features)
  if (anyDuplicated(data$hex_id)) {
    stop("Cluster assignment data contain duplicate hex IDs.", call. = FALSE)
  }
  scaled <- scale_cluster_features(data, model)
  assigned_data <- data[scaled$complete, , drop = FALSE]
  distances <- cluster_distance_matrix(scaled$matrix, model$centroids)
  assigned_index <- max.col(-distances, ties.method = "first")
  assigned_cluster <- as.integer(
    rownames(model$centroids)[assigned_index]
  )
  minimum_distance <- distances[
    cbind(seq_len(nrow(distances)), assigned_index)
  ]
  second_distance <- apply(distances, 1, function(row) sort(row)[[2]])
  margin_confidence <- pmax(
    0,
    pmin(1, 1 - minimum_distance / second_distance)
  )

  distance_threshold <- model$distance_thresholds$distance_threshold[
    match(assigned_cluster, model$distance_thresholds$cluster)
  ]
  assigned <- data.frame(
    hex_id = assigned_data$hex_id,
    cluster = assigned_cluster,
    distance_to_centroid = minimum_distance,
    second_centroid_distance = second_distance,
    margin_confidence = margin_confidence,
    outside_baseline_distance = minimum_distance > distance_threshold,
    boundary_flag = margin_confidence < model$margin_threshold
  )

  assigned <- merge(
    assigned,
    model$labels,
    by = "cluster",
    all.x = TRUE,
    sort = FALSE
  )

  eligible <- rep(TRUE, nrow(data))
  if (
    !is.null(model$eligibility_feature) &&
      model$eligibility_feature %in% names(data)
  ) {
    eligible <- !is.na(data[[model$eligibility_feature]]) &
      as.logical(data[[model$eligibility_feature]])
  }
  feature_matrix <- as.matrix(data[, model$features, drop = FALSE])
  storage.mode(feature_matrix) <- "double"
  finite_features <- apply(
    feature_matrix,
    1,
    function(row) all(is.finite(row))
  )
  coverage <- data.frame(
    hex_id = data$hex_id,
    assignment_status = ifelse(
      !eligible,
      "not_eligible",
      ifelse(!finite_features, "missing_cluster_feature", "assigned")
    )
  )

  merge(
    coverage,
    assigned,
    by = "hex_id",
    all.x = TRUE,
    sort = FALSE
  )
}

write_baseline_assignment_audit <- function(
  feature_file,
  model_file,
  cluster_results_file,
  assignment_file,
  summary_file
) {
  features <- readRDS(feature_file)
  model <- readRDS(model_file)
  results <- readRDS(cluster_results_file)
  assignment <- assign_fixed_clusters(features, model)
  selected <- results$assignments[
    results$assignments$specification == model$specification &
      results$assignments$k == model$k,
    c("hex_id", "cluster")
  ]
  names(selected)[names(selected) == "cluster"] <- "part1_cluster"
  assignment <- merge(
    assignment,
    selected,
    by = "hex_id",
    all.x = TRUE,
    sort = FALSE
  )
  assignment$reproduces_part1 <- ifelse(
    assignment$assignment_status == "assigned",
    assignment$cluster == assignment$part1_cluster,
    NA
  )

  summary <- data.frame(
    analysis_as_of_date = as.character(model$analysis_as_of_date),
    specification = model$specification,
    k = model$k,
    total_hexes = nrow(assignment),
    assigned_hexes = sum(assignment$assignment_status == "assigned"),
    not_eligible_hexes =
      sum(assignment$assignment_status == "not_eligible"),
    missing_cluster_feature_hexes =
      sum(assignment$assignment_status == "missing_cluster_feature"),
    mismatched_hexes = sum(!assignment$reproduces_part1, na.rm = TRUE),
    boundary_hexes = sum(assignment$boundary_flag, na.rm = TRUE),
    outside_baseline_distance_hexes =
      sum(assignment$outside_baseline_distance, na.rm = TRUE),
    median_margin_confidence = stats::median(
      assignment$margin_confidence,
      na.rm = TRUE
    )
  )

  dir.create(dirname(assignment_file), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(assignment, assignment_file)
  readr::write_csv(summary, summary_file)
  c(assignment_file, summary_file)
}
