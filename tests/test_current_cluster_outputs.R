# Independent checks of the corrected current model, with no production helpers.
suppressPackageStartupMessages({library(sf); library(readr)})
eq <- function(a,b) stopifnot(isTRUE(all.equal(a,b,check.attributes=FALSE,tolerance=1e-9)))
m <- readRDS("output/part1/baseline_cluster_model.rds")
r <- readRDS("output/amenity_cluster_sensitivity.rds")
x <- st_drop_geometry(readRDS("output/hex_features.rds"))
eligible <- x$primary_cluster_eligible
stopifnot(!anyNA(eligible),m$k==7L,length(m$features)==7L,
  identical(m$measurement_version,"complete-components-v2"),
  identical(m$measurement_scope,"single_cutoff_current_sample"),
  m$analysis_as_of_date==as.Date("2026-04-01"),sum(eligible)==r$n_observations,
  identical(m$measurement_manifest_sha256,digest::digest(file="output/part1/measurement/current_measurement_manifest.json",algo="sha256")),
  identical(m$interpretation$centroids_sha256,digest::digest(m$centroids,algo="sha256")),
  identical(m$interpretation$labels_sha256,digest::digest(file="config/amenity_cluster_labels.csv",algo="sha256")))
eq(m$component_scaling,readRDS("output/part1/measurement/current_component_scaling.rds"))
eq(m$interpretation,jsonlite::fromJSON("config/part1_cluster_interpretation.json"))
a <- as.matrix(x[eligible,m$features]); stopifnot(all(is.finite(a)))
eq(m$training_hex_ids,x$hex_id[eligible]);eq(m$preprocessing$center,colMeans(a))
eq(m$preprocessing$scale,apply(a,2,sd))
z <- scale(a)
d <- sapply(seq_len(m$k),function(k) sqrt(rowSums(sweep(z,2,m$centroids[k,],"-")^2)))
assignment <- max.col(-d,ties.method="first")
eq(assignment,m$training_assignment)
eq(m$training_minimum_distance,apply(d,1,min))
eq(m$training_margin_confidence,apply(d,1,function(row){v<-sort(row);1-v[1]/v[2]}))
for(k in 1:7)eq(m$centroids[k,],colMeans(z[assignment==k,,drop=FALSE]))
selected <- r$assignments[r$assignments$specification==m$specification & r$assignments$k==m$k,]
eq(selected$cluster[match(x$hex_id[eligible],selected$hex_id)],assignment)
published <- read_csv("output/part1/baseline_cluster_assignments.csv",show_col_types=FALSE)
stopifnot(nrow(published)==sum(eligible),!anyDuplicated(published$hex_id),
  setequal(published$hex_id,x$hex_id[eligible]))
published <- published[match(x$hex_id,published$hex_id),]
eq(published$cluster[eligible],assignment);stopifnot(all(is.na(published$cluster[!eligible])))
labels <- read_csv("config/amenity_cluster_labels.csv",show_col_types=FALSE)
eq(published$tentative_name[eligible],labels$tentative_name[match(assignment,labels$cluster)])
for(i in which(!is.na(labels$profile_anchor) & nzchar(labels$profile_anchor)))
  stopifnot(which.max(m$centroids[,labels$profile_anchor[i]])==labels$cluster[i])
fixed <- read_csv("output/part2/baseline_fixed_cluster_assignments.csv",show_col_types=FALSE)
fixed <- fixed[match(x$hex_id,fixed$hex_id),]
eq(fixed$cluster[eligible],assignment);stopifnot(all(fixed$reproduces_part1[eligible]),all(is.na(fixed$cluster[!eligible])))
summary <- read_csv("output/part1/baseline_cluster_summary.csv",show_col_types=FALSE)
eq(summary$average_silhouette,mean(cluster::silhouette(assignment,dist(z))[,"sil_width"]))
stopifnot(r$gap_bootstraps==100L,r$stability_replicates==100L,r$max_k==12L)
eq(summary$smallest_cluster_hexes,min(table(assignment)))
eq(summary$largest_cluster_hexes,max(table(assignment)))
paired <- readRDS("output/part2/matrix/part2_eligibility_by_hex.rds")
stopifnot(all(paired$hex_id[paired$common_comparison_ready] %in% m$training_hex_ids))
cat("Independent current cluster audit passed:",sum(eligible),"current cells, seven reviewed profiles, current scaling, exact assignments, margins, silhouette, labels and frozen-model reproduction.\n")
