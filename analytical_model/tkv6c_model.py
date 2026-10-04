#!/usr/bin/env python3
"""Analytical model of attn_beatserial_tkv6c.py on csynth numbers.

tkv6c = fused fp32-FMA beat-serial FlashAttention array (one flattened beat loop per head, V lag 2)
with ONE shared fp32 adder per PE for the once-per-key boundary, a sequential per-tile merge
(rb_merge) + reciprocal (rb_inv) per row, and 16-deep inter-PE FIFOs. CW = 2 is the calibrated
chunk width (the kernel supports CW <= 2).

INPUT   workload : L (context), dh (head dim), B (batch), H (heads, serial in one array)
        hardware : R x C array (NROWS x NCOLS), CW, FIFO depths (inter-PE K/V and o in elements,
                   feed / Q / fill as multiples of NB or dh), freq (MHz)
OUTPUT  interval, top latency, FLOP/cycle, GFLOP/s, E, and the
        absolute LUT / FF / DSP / BRAM / URAM of ONE array.

DEFAULT = the target design: 4x128, CW=2, inter-PE K/V and o FIFOs 16 deep, feed max(8*CW*NB,
8*NCOLS) beats. RTL-confirmed at 2x128 CTX=512: depth 16 costs 0 cycles with feed 32*NB; feed
16*NB + depth 16 passes but runs 6.7% slower (feed + D_K must cover NCOLS x o-hop).

RESOURCES
    DSP     R*C*2*CW + R*(2*CW+2) + R                        (exact by construction)
    URAM    2*C*CW * ceil(NKV*NB/4096)                        (k_buf, v_buf banks)
    BRAM  FIFOs by ceil(depth/512)*ceil(bits/36), I/O memory 4*32*2*ceil(L*HIDDEN/32/1024), +24,
            +2 per rb_merge (exp table)
    LUT/FF  per-role instance costs from the csynth reports + FIFO per-type costs + glue

CALIBRATION / CONFIRMATION
    tkv6c-specific terms (PE, merge, rb_inv, XC, the +1 FIFO slot): the two 4x128 CW=2
    L=1024 reports (inter-PE depth 1024 and 16); model within 0.1% on LUT/FF, exact on cycles,
    DSP, BRAM, URAM for both.

    python tkv6c_model.py                          # target design + CTX sweep
    python tkv6c_model.py --L 4096 --R 4 --C 128
    python tkv6c_model.py --sweep-shapes
"""

from __future__ import annotations

import argparse
import collections
import math
from dataclasses import dataclass, asdict

# ---------------------------------------------------------------- cycles
CWS = (1, 2)                             # the shared boundary adder needs NB >= 25: CW <= 2 at dh = 64
IL_PE = 12                               # beat loop depth 11 (+1 init); PE = beats + 180 at dh = 64
PE_LUT, PE_FF = 2421.0, 1284.0           # interior PE at L=1024, dh=64, CW=2
# Vitis sizes every FIFO written by an interior PE one deeper than declared (1025 x 64 K/V and
# o, 17 x 32 m; the d, Q and border-written ones stay as declared). Cost of that extra slot, fitted
# on the depth-32 and depth-16 4x128 reports: LUT ~0.35/bit (shallow), ~0.19/bit (BRAM-deep),
# FF +2 (shallow) / +4 (deep).
PLUS1_LUT_PER_BIT = {"shallow": 0.349, "deep": 0.192}
PLUS1_FF = {"shallow": 2, "deep": 4}
MERGE = (2210, 843)                      # + 2 BRAM each (exp table), 1 DSP
INV = (47, 26)
RB_FIXED = 183                           # right border: bt loop fill, q0 drain, epilogue, inits
IL_LOADQ = 7                             # NB + 7 per row per q-tile (NB + 9 from L = 4096: wider index)
IO_ELEMS = 16                            # fp16 elements read per cycle
IO_OVH = 75
YQ_TOP = 1257                            # top-latency offset when load_q is the longest process
# XC (top-latency start offset of the array) as a function of the shape; fitted below
# shape law from two tkv6b shapes (4x128, 8x32; independent of NB and L), scaled to the one tkv6c
# measurement: 790 at 4x128 (tkv6b 3,451)
XC_SCALE = 790.0 / 3451.0
XC_PER_COL = 22.57 * XC_SCALE
XC_PER_ROW = 140.6 * XC_SCALE
XC_FIXED = 0.0

# ---------------------------------------------------------------- resources (CW=2)
# per-instance (LUT, FF) at L=1024, dh=64
ROLE = dict(top=(94, 20), bottom=(92, 19), left=(203, 52), right=(1542.2, 2105.2))
L_STEP_LUT = 4.0                         # per doubling of L above 1024 (counter widths)
L_STEP_FF = 2.0
L512_LUT_DROP = {"pe": 37, "top": 35, "bottom": 35, "left": 35, "right": 38, "merge": 3}
CW_LUT_SCALE = {1: 0.816, 2: 1.0}        # CW = 1: tkv6b plain-flow smoke ratio, not measured on tkv6c
CW_FF_SCALE = {1: 0.78, 2: 1.0}
PE_LUT_PER_DH, PE_FF_PER_DH = 1.89, 0.59       # q_loc LUTRAM + wider beat counters (dh = 64 -> 128)
RB_LUT_PER_DH, RB_FF_PER_DH = 2.09, 0.19       # oacc[2][CW][NB]
# feeders at R=4, C=128 (LUT, FF) and their scaling
LOADQ = (2968, 349)
LOADQ_PER_ROW = (496, 21.5)              # from 8x32
STORE = (3457, 111)
STORE_PER_ROW = (575, 17)
LOADBUF = (392, 313)                     # x3
STORE_RES = (629, 628)
FILL = (2411, 2216)                      # at C = 128; (153, 64) at C = 32 -> linear in C
FILL_PER_COL = (23.52, 22.42)
REPLAY = (2263, 8630)                    # at C*CW = 256; (738, 2203) at 64 -> linear in C*CW
REPLAY_PER_LANE = (7.94, 33.47)
GLUE = (5456, 6328)
INSTANCE_BRAM = 24
# FIFO per-instance costs
FIFO_FF = {4: 7, 8: 9, 16: 11, 17: 13, 32: 15, 33: 15, 64: 61, 65: 65, 128: 65, 129: 37,
           256: 41, 257: 41, 512: 41, 513: 45, 1024: 45, 1025: 49, 2048: 49, 2049: 53}
FIFO_LUT_DEEP = 33.8                     # depth >= 2048: csynth reports NO BRAM for these (estimator
                                         # artefact; physically ~ceil(depth/512)*ceil(bits/36) BRAM)
FIFO_LUT_BRAM512 = 45.0
FIFO_LUT_BRAM_STEP = 8.2                 # per additional 512 of depth
FIFO_LUT_Q = 127.0                       # 64 x 32 SRL
FIFO_LUT_MD = 27.4                       # 16 x 32 SRL
FIFO_LUT_FILL = 60.0                     # 129 x 32 LUTRAM
FIFO_LUT_SMALL = 15.0
FIFO_LUT_MISC = 100.0


@dataclass
class Cfg:
    L: int = 1024
    dh: int = 64
    B: int = 1
    H: int = 1
    R: int = 4
    C: int = 128
    CW: int = 2
    freq: float = 300.0
    mul_q: int = 2
    mul_feed: int = 0          # feed FIFO depth in NB; 0 -> kernel default max(8*CW, 8*NCOLS/NB)
    mul_fill: int = 2
    mul_rb: int = 4            # right-border o FIFO depth in NB
    d_k: int = 16              # inter-PE K/V FIFO depth in elements (D_K_ABS)
    d_o: int = 16              # inter-PE o FIFO depth in elements (D_O_ABS)

    def __post_init__(self):
        if not self.mul_feed:                      # feed + inter-PE K/V depth must cover NCOLS x hop
            NBh = self.dh // self.CW
            self.mul_feed = max(8 * self.CW, -(-8 * self.C // NBh))


def derive(c: Cfg):
    why = []
    if c.CW not in CWS or c.dh % c.CW:
        why.append("CW")
    if c.L % c.R:
        why.append("L % NROWS")
    if c.L % c.C:
        why.append("L % NCOLS")
    NB = c.dh // c.CW
    NKV = c.L // c.C if c.L % c.C == 0 else 0
    NQ = c.L // c.R if c.L % c.R == 0 else 0
    if NQ < 2:
        why.append("NUM_Q_TILES < 2")
    if NKV < 2:
        why.append("NUM_KV_TILES < 2")
    units = c.B * c.H
    return dict(NB=NB, NKV=NKV, NQ=NQ, GT=NQ * NKV + 2, units=units, HIDDEN=c.H * c.dh,
                legal=not why, why="; ".join(why))


def xc(c: Cfg) -> float:
    return XC_PER_COL * c.C + XC_PER_ROW * c.R + XC_FIXED


def cycles(c: Cfg) -> dict:
    """Interval (cycles per head pass, steady state) and top-level latency of the array."""
    d = derive(c)
    NB, NKV, NQ, GT, U = d["NB"], d["NKV"], d["NQ"], d["GT"], d["units"]
    beats = GT * NB
    array = U * (beats + IL_PE + 2 + NB + 2) + 2 * c.dh + 2 + 2
    loader = U * NQ * c.R * (NB + IL_LOADQ + (2 if c.L >= 4096 else 0)) + 1
    interval = max(array,
                   U * (beats + RB_FIXED - 130) + 130,
                   U * (beats + NKV * c.dh + 3) + 5,
                   loader) + 1
    io_in = c.B * c.L * d["HIDDEN"] // IO_ELEMS + IO_OVH
    top = io_in + max(array + xc(c), loader + YQ_TOP)
    ms = 1e3 / (c.freq * 1e6)
    return dict(d=d, interval=interval, top_latency=int(round(top)),
                ms_interval=interval * ms, ms_top=top * ms, useful_beats=U * NQ * NKV * NB)


def dsp(c: Cfg) -> dict:
    pe = c.R * c.C * 2 * c.CW
    rb = c.R * (2 * c.CW + 2)
    mg = c.R
    return dict(pe=pe, right_border=rb, rb_merge=mg, total=pe + rb + mg)


def throughput(c: Cfg, cy=None) -> dict:
    cy = cy or cycles(c)
    d = cy["d"]
    flops = 4.0 * c.L * c.L * c.dh * d["units"]
    n = dsp(c)["total"]
    fpc = flops / cy["interval"]
    return dict(flops=flops, flop_per_cycle=fpc, peak_flop_per_cycle=2.0 * n, E=fpc / (2.0 * n),
                U=cy["useful_beats"] / cy["interval"], F=dsp(c)["pe"] / n,
                gflops=fpc * c.freq * 1e6 / 1e9, tokens_per_s=c.B * c.L * d["units"] / (cy["interval"] / (c.freq * 1e6)),
                gflops_top=flops / cy["top_latency"] * c.freq * 1e6 / 1e9)


def _fifo(depth: int, bits: int):
    """(bram as csynth reports it, lut, ff, physical bram) of one FIFO."""
    eff = depth - 1 if depth % 2 == 1 and depth > 16 else depth
    ff = FIFO_FF.get(depth, 45 if eff >= 256 else 37)
    phys = math.ceil(eff / 512) * math.ceil(bits / 36) if (eff >= 256 and eff * bits > 4096) else 0
    if eff >= 2048:
        return 0, FIFO_LUT_DEEP, ff, phys
    if eff >= 256 and eff * bits > 4096:
        k = math.ceil(eff / 512)
        return phys, FIFO_LUT_BRAM512 + FIFO_LUT_BRAM_STEP * (k - 1), ff, phys
    if eff * bits >= 4096 and depth % 2 == 1:          # 129 x 32 fill FIFO: LUTRAM
        return 0, FIFO_LUT_FILL * bits / 32.0, ff, 0
    if eff >= 32:                                      # SRL: 64x32 -> 127, 128x16 -> 155 (tkv5 fit)
        return 0, -9 + 1.6875 * bits * math.ceil(eff / 32) + 0.4375 * eff, ff, 0
    if eff >= 16:
        return 0, FIFO_LUT_MD * bits / 32.0, ff, 0
    return 0, FIFO_LUT_SMALL, ff, 0


def resources(c: Cfg) -> dict:
    d = derive(c)
    R, C, CW, NB = c.R, c.C, c.CW, d["NB"]
    l2 = math.log2(c.L / 1024.0) if c.L >= 1024 else 0.0
    low = c.L < 1024
    rows = collections.OrderedDict()

    def add(name, n, lut, ff, dsp_=0, bram=0, uram=0):
        rows[name] = [n, n * lut, n * ff, n * dsp_, n * bram, n * uram]

    def lstep(role):
        return L_STEP_LUT * l2 - (L512_LUT_DROP[role] if low else 0)

    fstep = L_STEP_FF * (math.log2(c.L / 1024.0))
    add("interior PE", R * C, PE_LUT * CW_LUT_SCALE[CW] + lstep("pe") + PE_LUT_PER_DH * (c.dh - 64),
        PE_FF * CW_FF_SCALE[CW] + fstep + PE_FF_PER_DH * (c.dh - 64), 2 * CW)
    add("right border", R, ROLE["right"][0] * (0.5 + 0.25 * CW) + lstep("right") + RB_LUT_PER_DH * (c.dh - 64),
        ROLE["right"][1] * (0.5 + 0.25 * CW) + fstep + RB_FF_PER_DH * (c.dh - 64), 2 * CW + 2)
    add("rb_merge", R, MERGE[0] + lstep("merge"), MERGE[1] + fstep, 1, bram=2)
    add("rb_inv", R, INV[0], INV[1])
    add("border top", C, ROLE["top"][0] + lstep("top"), ROLE["top"][1] + fstep)
    add("border bottom", C, ROLE["bottom"][0] + lstep("bottom"), ROLE["bottom"][1] + fstep)
    add("border left", R, ROLE["left"][0] + lstep("left"), ROLE["left"][1] + fstep)
    add("load_q", 1, LOADQ[0] + LOADQ_PER_ROW[0] * (R - 4), LOADQ[1] + LOADQ_PER_ROW[1] * (R - 4))
    add("store", 1, STORE[0] + STORE_PER_ROW[0] * (R - 4), STORE[1] + STORE_PER_ROW[1] * (R - 4))
    add("load_buf (x3)", 3, LOADBUF[0], LOADBUF[1])
    add("store_res", 1, STORE_RES[0], STORE_RES[1])
    add("load_kv_fill", 1, max(153.0, FILL[0] + FILL_PER_COL[0] * (C - 128)), max(64.0, FILL[1] + FILL_PER_COL[1] * (C - 128)))
    banks = 2 * C * CW
    add("load_kv_replay", 1, max(300.0, REPLAY[0] + REPLAY_PER_LANE[0] * (C * CW - 256)),
        max(300.0, REPLAY[1] + REPLAY_PER_LANE[1] * (C * CW - 256)), uram=banks * math.ceil(d["NKV"] * NB / 4096))
    swords = c.B * c.L * d["HIDDEN"] // 32
    smux = 0.0 if swords <= 2048 else (5.0 if swords <= 4096 else 32.0 * swords / 8192.0)   # read mux per bank
    add("I/O memory q/k/v/o", 4 * 32, smux, 2.0 if swords > 4096 else 0.0, bram=2 * math.ceil(swords / 1024))
    add("glue (interface, ctrl)", 1, GLUE[0], GLUE[1], bram=INSTANCE_BRAM)

    DK, DO, DQ = c.d_k, c.d_o, c.mul_q * NB
    DF, DFILL = c.mul_feed * NB, c.mul_fill * c.dh
    fam = [
        ("fifo_KV", (R + 1) * C, DK, 32 * CW),
        ("fifo_o", (C - 1) * R, DO, 32 * CW), ("fifo_o (left)", R, DO + 1, 32 * CW),
        ("fifo_Q", C * R, DQ, 16 * CW), ("fifo_Q (left)", R, DQ + 1, 16 * CW),
        ("fifo_m/d", 2 * C * R, 16, 32), ("fifo_m/d (left)", 2 * R, 17, 32),
        ("fifo_in_KV", C, DF + 1, 32 * CW),
        ("fifo_in_Q + fifo_out", 2 * R, DF, 16 * CW),
        ("fifo_fill_KV", C, DFILL + 1, 32),
        ("fifo_ab", R, 8, 64), ("fifo_inv", R, 4, 32),
        ("fifo_orb (right border)", R, c.mul_rb * NB, 32 * CW), ("fifo_D", R, 4, 32),
    ]
    plus1 = {"fifo_KV": R * C, "fifo_o": (C - 1) * R, "fifo_m/d": R * C}
    phys_bram = 0
    for name, n, depth, bits in fam:
        b, l, f, ph = _fifo(depth, bits)
        rows[name] = [n, n * l, n * f, 0, n * b, 0]
        if name in plus1:                                  # the +1 slot on interior-written FIFOs
            kind = "deep" if depth >= 256 else "shallow"
            rows[name][1] += plus1[name] * PLUS1_LUT_PER_BIT[kind] * bits
            rows[name][2] += plus1[name] * PLUS1_FF[kind]
        phys_bram += n * ph
    rows["fifo (misc)"] = [1, FIFO_LUT_MISC, 37 if C + 11 > 128 else 93, 0, 2 if C + 11 > 128 else 0, 0]

    tot = dict(lut=sum(r[1] for r in rows.values()), ff=sum(r[2] for r in rows.values()),
               dsp=sum(r[3] for r in rows.values()), bram=sum(r[4] for r in rows.values()),
               uram=sum(r[5] for r in rows.values()))
    fifo_csyn = sum(r[4] for k, r in rows.items() if k.startswith("fifo_"))
    return dict(rows=rows, total=tot, calibrated=(CW == 2), bram_physical=tot["bram"] - fifo_csyn + phys_bram)


# ---------------------------------------------------------------- reporting
def fi(x):
    return f"{int(round(x)):,}"


def report(c: Cfg) -> None:
    d = derive(c)
    print("=" * 84)
    print(f"tkv6c model L={c.L} dh={c.dh} B={c.B} H={c.H} | {c.R}x{c.C} CW={c.CW} | "
          f"inter-PE FIFOs K/V {c.d_k} o {c.d_o}, feed {c.mul_feed * (c.dh // c.CW)} | {c.freq:.0f} MHz")
    print("=" * 84)
    print(f"  NB={d['NB']} NUM_KV_TILES={d['NKV']} NUM_Q_TILES={d['NQ']} tiles/head={d['GT']} "
          f"(b,h) passes={d['units']}  legal={'yes' if d['legal'] else 'NO: ' + d['why']}")
    if not d["legal"]:
        return
    cy, th, rs = cycles(c), throughput(c), resources(c)
    print("\nLATENCY")
    print(f"  interval        {fi(cy['interval']):>12} cyc  {cy['ms_interval']:9.4f} ms")
    print(f"  top latency     {fi(cy['top_latency']):>12} cyc  {cy['ms_top']:9.4f} ms")
    print("\nTHROUGHPUT (FLOPs = 4*L^2*dh per head)")
    print(f"  {th['flop_per_cycle']:10.1f} FLOP/cycle on the interval  (peak {th['peak_flop_per_cycle']:.0f} = 2 x DSP; "
          f"E = {th['E']:.3f} = U {th['U']:.3f} x F {th['F']:.3f})")
    print(f"  {th['gflops']:10.1f} GFLOP/s at {c.freq:.0f} MHz   {th['tokens_per_s']:,.0f} tokens/s   "
          f"({th['gflops_top']:.1f} GFLOP/s on the top latency)")

def sweep_ctx(base: Cfg, Ls) -> None:
    print(f"\nCTX sweep  {base.R}x{base.C} CW={base.CW} dh={base.dh} @ {base.freq:.0f} MHz")
    hdr = (f"{'L':>6} {'interval':>11} {'int ms':>8} {'top latency':>12} {'top ms':>8} "
           f"{'FLOP/cyc':>9} {'E':>6} {'GFLOP/s':>8} | {'LUT':>10} {'FF':>10} {'DSP':>6} {'BRAM':>7} {'URAM':>5}")
    print(hdr)
    print("-" * len(hdr))
    for L in Ls:
        c = Cfg(**{**asdict(base), "L": L})
        d = derive(c)
        if not d["legal"]:
            print(f"{L:>6}  illegal: {d['why']}")
            continue
        cy, th, t = cycles(c), throughput(c), resources(c)["total"]
        print(f"{L:>6} {fi(cy['interval']):>11} {cy['ms_interval']:8.3f} {fi(cy['top_latency']):>12} {cy['ms_top']:8.3f} "
              f"{th['flop_per_cycle']:9.1f} {th['E']:6.3f} {th['gflops']:8.1f} | "
              f"{fi(t['lut']):>10} {fi(t['ff']):>10} {fi(t['dsp']):>6} {fi(t['bram']):>7} {fi(t['uram']):>5}")


def sweep_shapes(base: Cfg) -> None:
    print(f"\nSHAPE sweep  L={base.L} dh={base.dh} @ {base.freq:.0f} MHz; feed FIFO = max(512, 8*NCOLS) beats; "
          f"inter-PE FIFOs {base.d_k} / {base.d_o}")
    hdr = (f"{'RxC':>8} {'CW':>2} {'PEs':>5} {'interval':>10} {'top latency':>11} {'FLOP/cyc':>9} {'E':>6} | "
           f"{'LUT':>10} {'FF':>10} {'DSP':>6} {'BRAM':>7} {'URAM':>5}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for R in (2, 4, 8, 16, 32):
        for C in (16, 32, 64, 128, 256):
            for CW in CWS:
                c = Cfg(**{**asdict(base), "R": R, "C": C, "CW": CW, "mul_feed": 0})     # feed re-derived per shape
                if not derive(c)["legal"]:
                    continue
                out.append((throughput(c)["E"], c))
    for E, c in sorted(out, key=lambda x: -x[0])[:24]:
        cy, th, t = cycles(c), throughput(c), resources(c)["total"]
        print(f"{c.R:>4}x{c.C:<3} {c.CW:>2} {c.R*c.C:>5} {fi(cy['interval']):>10} {fi(cy['top_latency']):>11} "
              f"{th['flop_per_cycle']:9.1f} {E:6.3f} | {fi(t['lut']):>10} {fi(t['ff']):>10} {fi(t['dsp']):>6} {fi(t['bram']):>7} {fi(t['uram']):>5}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--dh", type=int, default=64)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=1, help="heads processed serially by the array")
    ap.add_argument("--R", type=int, default=4)
    ap.add_argument("--C", type=int, default=128)
    ap.add_argument("--CW", type=int, default=2)
    ap.add_argument("--freq", type=float, default=300.0)
    ap.add_argument("--d-k", type=int, default=16, help="inter-PE K/V FIFO depth in elements (D_K_ABS)")
    ap.add_argument("--d-o", type=int, default=16, help="inter-PE o FIFO depth in elements (D_O_ABS)")
    ap.add_argument("--mul-feed", type=int, default=0, help="feed FIFO depth in NB (0 = max(8*CW, 8*NCOLS/NB))")
    ap.add_argument("--sweep-ctx", type=str, default="")
    ap.add_argument("--sweep-shapes", action="store_true")
    a = ap.parse_args(argv)
    c = Cfg(L=a.L, dh=a.dh, B=a.B, H=a.H, R=a.R, C=a.C, CW=a.CW, freq=a.freq, d_k=a.d_k, d_o=a.d_o,
            mul_feed=a.mul_feed)
    report(c)
    if a.sweep_shapes:
        sweep_shapes(c)
    else:
        sweep_ctx(c, [int(x) for x in a.sweep_ctx.split(",")] if a.sweep_ctx else [256, 512, 1024, 2048, 4096, 8192, 16384])


if __name__ == "__main__":
    main()
