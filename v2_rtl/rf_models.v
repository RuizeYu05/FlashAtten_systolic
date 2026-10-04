// Register-file implementations for the two small RAM kinds of the tkv6b ASIC top
// (same behaviour as Catapult's ccs_ram_sync_1R1W: 1W + 1R, synchronous read register).
module pp_rf_2x19 (radr, wadr, d, we, re, clk, q);          // p ring, one per PE (512)
  input  [0:0]  radr;
  input  [0:0]  wadr;
  input  [18:0] d;
  input         we;
  input         re;
  input         clk;
  output reg [18:0] q;
  reg [18:0] mem [0:1];
  always @(posedge clk) begin
    if (we) mem[wadr] <= d;
    if (re) q <= mem[radr];
  end
endmodule

module oacc_rf_32x32 (radr, wadr, d, we, re, clk, q);       // right-border O bank, 2 per row (8)
  input  [4:0]  radr;
  input  [4:0]  wadr;
  input  [31:0] d;
  input         we;
  input         re;
  input         clk;
  output reg [31:0] q;
  reg [31:0] mem [0:31];
  always @(posedge clk) begin
    if (we) mem[wadr] <= d;
    if (re) q <= mem[radr];
  end
endmodule
