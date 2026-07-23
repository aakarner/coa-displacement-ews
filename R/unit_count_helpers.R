################################################################################
# Residential Unit Count Helpers
################################################################################

normalize_unit_address <- function(x) {
  x %>%
    stringr::str_to_upper() %>%
    stringr::str_replace_all(
      "\\bAUSTIN\\b\\s*\\bTX\\b\\s*\\d{5}(-\\d{4})?\\s*$",
      ""
    ) %>%
    stringr::str_replace_all("\\bTEXAS\\b\\s*\\d{5}(-\\d{4})?\\s*$", "") %>%
    stringr::str_replace_all("\\bTX\\b\\s*\\d{5}(-\\d{4})?\\s*$", "") %>%
    stringr::str_replace_all("\\b(\\d+)(ST|ND|RD|TH)\\b", "\\1") %>%
    stringr::str_replace_all("\\bROAD\\b", "RD") %>%
    stringr::str_replace_all("\\bSTREET\\b", "ST") %>%
    stringr::str_replace_all("\\bAVENUE\\b", "AVE") %>%
    stringr::str_replace_all("\\bBOULEVARD\\b", "BLVD") %>%
    stringr::str_replace_all("\\bDRIVE\\b", "DR") %>%
    stringr::str_replace_all("\\bLANE\\b", "LN") %>%
    stringr::str_replace_all("\\bCOURT\\b", "CT") %>%
    stringr::str_replace_all("\\bPLACE\\b", "PL") %>%
    stringr::str_replace_all("\\bPARKWAY\\b", "PKWY") %>%
    stringr::str_replace_all("\\bHIGHWAY\\b", "HWY") %>%
    stringr::str_replace_all("\\bINTERSTATE\\b", "IH") %>%
    stringr::str_replace_all("\\bNORTH\\b", "N") %>%
    stringr::str_replace_all("\\bSOUTH\\b", "S") %>%
    stringr::str_replace_all("\\bEAST\\b", "E") %>%
    stringr::str_replace_all("\\bWEST\\b", "W") %>%
    stringr::str_replace_all("[^A-Z0-9]+", " ") %>%
    stringr::str_squish() %>%
    dplyr::na_if("")
}

unit_zip5 <- function(x) {
  stringr::str_extract(as.character(x), "\\d{5}")
}

unit_street_number <- function(x) {
  stringr::str_extract(x, "^\\d+[A-Z]?")
}

unit_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

first_non_missing_unit_value <- function(x) {
  observed <- x[!is.na(x) & as.character(x) != ""]
  if (length(observed) == 0L) {
    return(NA)
  }
  observed[[1]]
}

unit_mode <- function(x) {
  observed <- as.character(x[!is.na(x) & as.character(x) != ""])
  if (length(observed) == 0L) {
    return(NA_character_)
  }
  counts <- sort(table(observed), decreasing = TRUE)
  sort(names(counts)[counts == max(counts)])[[1]]
}

unit_weighted_mean <- function(x, w) {
  keep <- is.finite(x) & is.finite(w) & w > 0
  if (!any(keep)) {
    return(NA_real_)
  }
  stats::weighted.mean(x[keep], w[keep])
}

unit_relative_spread <- function(x) {
  observed <- x[is.finite(x) & x > 0]
  if (length(observed) <= 1L) {
    return(0)
  }
  median_value <- stats::median(observed)
  if (!is.finite(median_value) || median_value <= 0) {
    return(NA_real_)
  }
  (max(observed) - min(observed)) / median_value
}

unit_address_similarity <- function(x, y) {
  max_length <- pmax(nchar(x), nchar(y))
  distance <- mapply(
    function(a, b) as.numeric(utils::adist(a, b)[1]),
    x,
    y,
    USE.NAMES = FALSE
  )
  ifelse(max_length > 0, 1 - distance / max_length, NA_real_)
}

# Return one connected-component identifier for every parcel. The link table
# may contain several rows per source or exact-address group.
unit_connected_components <- function(parcel_ids, links) {
  parcel_ids <- unique(as.character(parcel_ids))
  parent <- seq_along(parcel_ids)
  rank <- integer(length(parcel_ids))
  names(parent) <- parcel_ids

  find_root <- function(index) {
    root <- index
    while (parent[[root]] != root) {
      root <- parent[[root]]
    }
    while (parent[[index]] != index) {
      next_index <- parent[[index]]
      parent[[index]] <<- root
      index <- next_index
    }
    root
  }

  union_roots <- function(left, right) {
    left_root <- find_root(left)
    right_root <- find_root(right)
    if (left_root == right_root) {
      return(invisible(NULL))
    }
    if (rank[[left_root]] < rank[[right_root]]) {
      parent[[left_root]] <<- right_root
    } else if (rank[[left_root]] > rank[[right_root]]) {
      parent[[right_root]] <<- left_root
    } else {
      parent[[right_root]] <<- left_root
      rank[[left_root]] <<- rank[[left_root]] + 1L
    }
    invisible(NULL)
  }

  if (nrow(links) > 0L) {
    grouped_ids <- split(as.character(links$parcel_id), links$link_group_id)
    for (members in grouped_ids) {
      member_indices <- unname(parent[unique(members)])
      member_indices <- member_indices[!is.na(member_indices)]
      if (length(member_indices) > 1L) {
        anchor <- member_indices[[1]]
        for (other in member_indices[-1]) {
          union_roots(anchor, other)
        }
      }
    }
  }

  roots <- vapply(seq_along(parcel_ids), find_root, integer(1))
  root_keys <- tapply(parcel_ids, roots, function(x) sort(x)[[1]])

  tibble::tibble(
    parcel_id = parcel_ids,
    component_index = roots,
    component_key = unname(root_keys[as.character(roots)])
  )
}
