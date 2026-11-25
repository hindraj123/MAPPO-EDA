# Floorplanning stage. All geometry values are deployment inputs.
foreach required {MAPPO_EDA_DIE_AREA MAPPO_EDA_CORE_AREA} {
    if {![info exists ::env($required)] || $::env($required) eq ""} {
        error "required environment variable is missing: $required"
    }
}
initialize_floorplan \
    -die_area $::env(MAPPO_EDA_DIE_AREA) \
    -core_area $::env(MAPPO_EDA_CORE_AREA)
make_tracks
place_pins -random
