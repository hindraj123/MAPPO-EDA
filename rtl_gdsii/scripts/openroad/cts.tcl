# Clock-tree synthesis stage.
clock_tree_synthesis
set_propagated_clock [all_clocks]
report_checks -path_delay max -format summary
