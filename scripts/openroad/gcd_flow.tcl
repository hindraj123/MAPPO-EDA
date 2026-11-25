# Complete direct OpenROAD flow for the GCD fixture.
# Technology-specific paths and floorplan geometry are supplied by the Linux
# deployment environment. This script does not invent technology data.

proc mappo_require_env {name} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "required environment variable is missing: $name"
    }
    return $::env($name)
}

set rtl [mappo_require_env MAPPO_EDA_GCD_RTL]
set lef [mappo_require_env MAPPO_EDA_LEF]
set liberty [mappo_require_env MAPPO_EDA_LIBERTY]
set sdc [mappo_require_env MAPPO_EDA_SDC]
set output_dir [mappo_require_env MAPPO_EDA_OUTPUT_DIR]
set gds [mappo_require_env MAPPO_EDA_GDS]
set gds_map [mappo_require_env MAPPO_EDA_GDS_MAP]

file mkdir $output_dir
read_lef $lef
read_liberty $liberty
read_verilog $rtl
link_design gcd
read_sdc $sdc

set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir floorplanning.tcl]
source [file join $script_dir placement.tcl]
source [file join $script_dir cts.tcl]
source [file join $script_dir routing.tcl]

write_def [file join $output_dir gcd.def]
write_db  [file join $output_dir gcd.odb]
write_gds $gds -map_file $gds_map
source [file join $script_dir extract_metrics.tcl]
