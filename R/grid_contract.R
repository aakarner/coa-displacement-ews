# Versioned computational coverage; analytical center selection remains explicit.
grid_contract <- function(path = 'output/hex_grid_manifest.json') {
  x <- jsonlite::read_json(path, simplifyVector = TRUE)
  stopifnot(identical(x$schema_version,1L),x$resolution==9L,
    identical(x$grid_sha256,digest::digest(file='output/hex_grid.rds',algo='sha256')))
  x
}
