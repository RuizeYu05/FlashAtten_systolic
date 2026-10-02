# top_asap7.sdc -- starting constraints for rtl/top.v on ASAP7 (liberty units: ps, fF).
# rtl/top.catapult.sdc is Catapult's own SDC for its nangate45 run (ns units, 1.3 ns clock,
# nangate loads); use it only for the per-port list, not for ASAP7 numbers.
#
# Single clock, posedge only; rst is synchronous, active high.
# Every top-level port except clk/rst is a banked synchronous-memory interface to the four
# host buffers (q, K^T, V^T, out): radr/re/q (read) and wadr/we/d (write), plus *_triosy_*_lz
# "array released" flags. In a chip these connect to SRAMs (or a DMA) -- treat them as
# register-to-register paths into/out of those macros.

set PERIOD 500.0                                   ;# ps -- PE/FIFO blocks close 500 ps (LVT, TT)
create_clock -name clk -period $PERIOD [get_ports clk]
set_clock_uncertainty 20 [get_clocks clk]

set ins  [remove_from_collection [all_inputs] [get_ports clk]]
set outs [all_outputs]
# host-memory read data arrives one cycle after re (registered in the SRAM): give it a
# typical macro clock-to-q; addresses/enables must reach the macro before its setup.
set_input_delay  [expr 0.3 * $PERIOD] -clock clk $ins
set_output_delay [expr 0.3 * $PERIOD] -clock clk $outs
set_false_path -to   [get_ports *_triosy_*_lz]
set_load 1.0 $outs
set_driving_cell -lib_cell BUFx2_ASAP7_75t_L $ins
