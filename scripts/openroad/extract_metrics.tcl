# Emit only measurements obtained from the current OpenROAD/OpenDB session.
# The Python integration parses lines beginning with MAPP0EDA_METRIC.

proc mappo_emit_metric {name value} {
    if {![catch {expr {double($value)}} numeric] &&
        ![regexp -nocase {^(nan|inf|infinity)} $numeric]} {
        puts "MAPP0EDA_METRIC $name $numeric"
    }
}

set block [ord::get_db_block]
if {$block eq ""} {
    error "no OpenDB block is loaded; cannot extract physical-design metrics"
}

set dbu [$block getDbUnitsPerMicron]
set bbox [$block getBBox]
mappo_emit_metric die_width_um [expr {double([$bbox xMax] - [$bbox xMin]) / $dbu}]
mappo_emit_metric die_height_um [expr {double([$bbox yMax] - [$bbox yMin]) / $dbu}]
mappo_emit_metric instance_count [llength [$block getInsts]]
mappo_emit_metric net_count [llength [$block getNets]]

# OpenSTA returns no finite value when timing has not been analyzed. In that
# case no timing metric is emitted; absence is preserved by the Python parser.
if {![catch {set wns [sta::worst_slack -max]}]} {
    mappo_emit_metric wns_ns $wns
}
if {![catch {set tns [sta::total_negative_slack -max]}]} {
    mappo_emit_metric tns_ns $tns
}
