// kvbuf_256x16 -- black box for the K/V replay buffers (512 instances: 256 x 16 bit, 1R1W,
// synchronous read with 1-cycle latency: q <= mem[radr] at posedge clk when re).
// Replace with a wrapper around your SRAM / register-file macro with the same ports.
module kvbuf_256x16 (radr, wadr, d, we, re, clk, q);
  input  [7:0]  radr;
  input  [7:0]  wadr;
  input  [15:0] d;
  input         we;
  input         re;
  input         clk;
  output [15:0] q;
endmodule
