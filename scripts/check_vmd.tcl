# Run with: vmd -dispdev text -e scripts/check_vmd.tcl -args particle.pdb
# Requires VMD. This is an external acceptance check, not a substitute for the
# Python geometry validator. No VMD installation is bundled with mastercap.
set status [catch {
    if {[llength $argv] != 1} {
        error "Usage: vmd -dispdev text -e scripts/check_vmd.tcl -args particle.pdb"
    }
    set path [lindex $argv 0]
    set input [open $path r]
    set expected {}
    set groups {}
    set segment 0
    set lastkey ""
    set group -1
    while {[gets $input line] >= 0} {
        set record [string trim [string range $line 0 5]]
        if {$record eq "TER"} {
            incr segment
            set lastkey ""
        }
        if {$record ne "ATOM"} { continue }
        set key [list $segment [string range $line 21 26] [string range $line 17 19]]
        if {$key ne $lastkey} {
            incr group
            set lastkey $key
        }
        lappend groups $group
        lappend expected [list \
            [string trim [string range $line 12 15]] \
            [string trim [string range $line 17 19]] \
            [string toupper [string trim [string range $line 76 77]]] \
            [string trim [string range $line 30 37]] \
            [string trim [string range $line 38 45]] \
            [string trim [string range $line 46 53]]]
    }
    close $input
    if {![llength $expected]} { error "No ATOM records" }
    set molecule [mol new $path type pdb waitfor all autobonds off]
    if {[molinfo $molecule get numframes] != 1} {
        error "Expected a single VMD frame"
    }
    set selection [atomselect $molecule all]
    if {[$selection num] != [llength $expected]} {
        error "Atom count mismatch: PDB [llength $expected], VMD [$selection num]"
    }
    set observed [$selection get {name resname element x y z}]
    set observed_groups [$selection get residue]
    set previous_actual ""
    set actual_group -1
    foreach source $expected actual $observed wanted_group $groups actual_residue $observed_groups {
        if {$actual_residue ne $previous_actual} {
            incr actual_group
            set previous_actual $actual_residue
        }
        if {$actual_group != $wanted_group} { error "Residue grouping differs in VMD" }
        for {set i 0} {$i < 3} {incr i} {
            if {[string toupper [lindex $source $i]] ne [string toupper [lindex $actual $i]]} {
                error "Atom name, residue name or element differs in VMD"
            }
        }
        for {set i 3} {$i < 6} {incr i} {
            if {abs([lindex $source $i]-[lindex $actual $i]) > 0.002} {
                error "Coordinate differs in VMD by more than 0.002 A"
            }
        }
    }
    puts "PASS: [$selection num] atoms, one frame, identities, residue groups and coordinates match"
    $selection delete
    mol delete $molecule
} message]
if {$status} {
    puts stderr "FAIL: $message"
    exit 1
}
exit 0
