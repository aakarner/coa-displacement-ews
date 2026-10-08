# Decision 0020 can admit an old outside-grid location or refine its exclusion
# reason. The precision rule can remove coarse candidates. Neither change
# licenses loss/reassignment of a previously precise in-grid case.
assert_repaired_raw_case_conservation <- function(x, old, old_raw_column) {
  stopifnot(identical(x$case_number, old$case_number))
  precise <- !x$has_rejected_imprecise_geocode
  outside <- old$assignment_status %in% c('excluded_reliable_geocode_outside_grid',
    'excluded_mixed_inside_outside_grid')
  grid_change <- precise & outside & x$assignment_status != old$assignment_status
  if(any(grid_change)) {
    stopifnot(file.exists('docs/decisions/0020-full-purpose-h3-grid.md'))
    accepted <- grid_change & x$assignment_status == 'assigned_unique_hex'
    stopifnot(all(as.integer(x$original_assigned_hex_key[accepted]) > 7027L),
      all(x$assignment_status[grid_change & !accepted] %in%
        c('excluded_reliable_geocode_outside_study_geography',
          'excluded_mixed_inside_outside_study_geography')),
      all(is.na(x$original_assigned_hex_key[grid_change & !accepted])))
  }
  stable <- precise & !grid_change
  stopifnot(identical(x$assignment_status[stable], old$assignment_status[stable]),
    identical(x$original_assigned_hex_key[stable], old[[old_raw_column]][stable]))
  grid_change
}
