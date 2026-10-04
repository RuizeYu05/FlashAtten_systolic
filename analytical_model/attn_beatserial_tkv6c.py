# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
tkv6c -- tkv6b (fused fp32-FMA PE, one flattened beat loop per head, VLAG=2) with the
once-per-key boundary arithmetic of the interior PE time-multiplexed on ONE fp32 adder.

tkv6b spends six fabric fp32 add/sub per PE (lane sum, x = sum - previous sum,
t = x - x_ref, two inside exp2's floor/fraction, d + p): 2,208 of the 4,570 LUT of a PE,
each used one beat in NB. Vitis never shares operators inside an II=1 loop, so the
sharing is written out:

  * the boundary of key g runs DURING key g+1 (p is only needed by key g+2), one
    operation every BSP beats through a single adder whose operands are selected by
    the beat index:
        beat 0              s  = acc[0] + acc[1]      -> rr[0]   (acc of key g, copied one beat earlier)
        beat B1 = BSP       x  = s - s_prev          -> rr[1]
        beat B2 = 2*BSP     t  = x - x_ref           -> rr[0]   (m stream get/put here)
        beat B3 = 3*BSP     p  = 2^t  (no adder)     -> rr[1], pp ring
        beat B4 = 4*BSP     d  = p + d_in            (d stream get/put here)
    subtraction is an fp32 sign flip on the second operand (free).
  * exp2's floor and fraction come from the fp32 bits of t (one 24-bit right shift and
    a sticky bit) instead of fptosi + sitofp + fcmp + two fp32 subtracts. This is the
    exact floor(t * 1024); tkv6b's fp32 `t - floor(t)` rounds, which moved the table
    index by one in ~2e-5 of draws near t = 0 and returned p = 0.5 for -2^-25 < t < 0.

The m/d token a PE handles during key g therefore belongs to key g-1: token 0 of a
head is a dummy, rb_merge skips it. rb_merge is a sequential (unpipelined) loop with one
fsub, one 2^t and one FMA per tile (one of a, b is always 1): a pipelined merge releases
(a, b) of a tile only when later tiles push it through (deadlock in RTL cosim at 4x4). The right border reads its o from a dedicated
stream (fifo_orb, depth MUL_RB*NB) because (a, b) of a key now arrive ~NB beats later
and that lag must not be paid for in every inter-PE fifo_o.

The accumulators are untouched (read only), the beat loop stays II=1, DSP/PE = 2*CW.
Needs NB >= 4*BSP + 1 (CW <= 2 at dh = 64); the carried distances (BSP on the boundary
ring rr, NB - B3 on the pp ring) are injected as dependence pragmas.

Defaults: inter-PE K/V and o FIFOs 16 deep (RTL-confirmed), feed FIFO max(8*CW*NB, 8*NCOLS).
Env knobs: as tkv6b, plus BSP (beats per shared-adder op, default 6 = fabric fadd
latency 5 + 1) and MUL_RB (right-border o FIFO depth in NB, default 4).
Cosim: COSIM_ALLOW_EMPTY=1; COSIM_OP_IMPL=fadd:fabric,fsub:fabric to simulate the
fabric adders (cosim.tcl re-synthesizes and does not see OP_IMPL).
"""

import os

import numpy as np
import allo
import allo.dataflow as df
import allo.backend.hls as hls
from allo.ir.types import float16, float32, int32, uint16, uint32, Stream
from allo import Memory

from attn_beatserial_tkv5 import (
    EXP_NTAB, EXP_LGTAB, EXP_SHIFT_BASE, EXP_E_MIN, _inject_op_impl, _golden_inputs,
)

EXP_POS_NP = np.exp2(np.arange(EXP_NTAB, dtype=np.float64) / EXP_NTAB).astype(np.float16).view(np.uint16)
assert EXP_NTAB == 1024


def get_tkv6c_flash_attention(
    BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC=1
):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    NB = HEAD_DIM // CW
    KV_TILE = NCOLS
    NUM_KV_TILES = CONTEXT_LENGTH // KV_TILE
    NUM_Q_TILES = CONTEXT_LENGTH // NROWS
    VLAG = 2                                     # V (and p) lag in keys: p of key g is used two keys later
    GT = NUM_Q_TILES * NUM_KV_TILES + VLAG       # tiles per head incl. the trailing dummies
    NKV = NUM_KV_TILES

    BSP = int(os.environ.get("BSP", "6"))        # beats between two ops on the shared adder
    B1 = BSP                                      # boundary beats inside the NEXT key (beat 0: lane sum)
    B2 = 2 * BSP
    B3 = 3 * BSP
    B4 = 4 * BSP
    assert HEAD_DIM % CW == 0 and CW in (1, 2), "shared boundary adder: lane sum is one op (CW <= 2)"
    assert NB >= 4 * BSP + 1, f"boundary needs 4*BSP+1 <= NB beats (NB={NB}, BSP={BSP})"
    assert CONTEXT_LENGTH % KV_TILE == 0 and CONTEXT_LENGTH % NROWS == 0
    assert NUM_Q_TILES >= 2 and NUM_KV_TILES >= 2

    P0 = NROWS + 2
    P1 = NCOLS + 2
    D_SQRT = float(HEAD_DIM**0.5)
    D_SCALE = 1.4426950408889634 / D_SQRT
    QKV_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * HIDDEN_SIZE
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM
    KVBUF_RESOURCE = os.environ.get("KVBUF_RESOURCE", "URAM")

    D_Q = int(os.environ.get("MUL_Q", "2")) * NB
    # Inter-PE K/V and o FIFOs: 16 elements (RTL-confirmed 2026-09-30 at 2x128 CTX=256 and CTX=512,
    # 0 cycles slower than 1024 deep; 4,608 BRAM18 less at 4x128). With VLAG=2 the o hop is a few
    # beats; the column skew NCOLS x hop is absorbed once per column by fifo_in_KV (D_FEED), which
    # must stay >= 8*NCOLS beats (16*NB = 512 at 128 columns passed but ran 6.7% slower).
    # MUL_K / MUL_O (in NB) or D_K_ABS / D_O_ABS (elements) override.
    D_K = int(os.environ["MUL_K"]) * NB if os.environ.get("MUL_K") else int(os.environ.get("D_K_ABS", "16"))
    D_O = int(os.environ["MUL_O"]) * NB if os.environ.get("MUL_O") else int(os.environ.get("D_O_ABS", "16"))
    # per-column feed FIFO: together with D_K it must cover the column skew NCOLS x o-hop
    # (2x128 CTX=512: 1024 + 16 runs at full rate, 512 + 16 is 6.7% slower, 512 + 1024 full rate).
    D_FEED = int(os.environ["MUL_FEED"]) * NB if os.environ.get("MUL_FEED") else max(8 * CW * NB, 8 * NCOLS)
    D_FILL = int(os.environ.get("MUL_FILL", "2")) * HEAD_DIM
    # right-border o FIFO: (a, b) of key g reach the right border ~B4 + merge latency beats
    # after key g ends, i.e. ~1.2 NB after its o started arriving.
    D_RB = int(os.environ.get("MUL_RB", "4")) * NB

    @df.region()
    def top(
        q_mem: float16[QKV_ELEMS],
        k_mem: float16[QKV_ELEMS],
        v_mem: float16[QKV_ELEMS],
        output_mem: float16[OUT_ELEMS],
    ):
        fifo_Q: Stream[float16[CW], D_Q][P0, P1]
        fifo_KV: Stream[float16[2 * CW], D_K][P0, P1]
        fifo_m: Stream[float32, 16][P0, P1]
        fifo_d: Stream[float32, 16][P0, P1]
        fifo_o: Stream[float32[CW], D_O][P0, P1]
        fifo_orb: Stream[float32[CW], D_RB][NROWS]

        fifo_in_Q: Stream[float16[CW], D_FEED][P0 - 2]
        fifo_in_KV: Stream[float16[2 * CW], D_FEED][P1 - 2]
        fifo_fill_KV: Stream[float16[2], D_FILL][NCOLS]
        fifo_out: Stream[float16[CW], D_FEED][NROWS]
        fifo_ab: Stream[float32[2], 8][NROWS]
        fifo_inv: Stream[float32, 4][NROWS]
        fifo_D: Stream[float32, 4][NROWS]

        # ---- load_q: beat-serial, CW per beat, scale folded in -------------
        @df.kernel(mapping=[1], args=[q_mem])
        def load_q(q_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as r:
                            for jj in range(NB):
                                qv: float16[CW] = 0
                                with allo.meta_for(CW) as w:
                                    q_idx = (
                                        b * (CONTEXT_LENGTH * HIDDEN_SIZE)
                                        + (tr + r) * HIDDEN_SIZE
                                        + h * HEAD_DIM
                                        + jj * CW + w
                                    )
                                    sc: float16 = D_SCALE
                                    qv[w] = q_d[q_idx] * sc
                                fifo_in_Q[r].put(qv)

        # ---- load_kv_fill: TKV staging [dh][L], all NCOLS lanes per cycle ---
        @df.kernel(mapping=[1], args=[k_mem, v_mem])
        def load_kv_fill(k_d: float16[QKV_ELEMS], v_d: float16[QKV_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_f in range(NUM_KV_TILES):
                        for jjf in range(HEAD_DIM):
                            with allo.meta_for(NCOLS) as c:
                                t = tile_f * KV_TILE + c
                                base = (
                                    b * (CONTEXT_LENGTH * HIDDEN_SIZE)
                                    + h * (CONTEXT_LENGTH * HEAD_DIM)
                                    + jjf * CONTEXT_LENGTH
                                    + t
                                )
                                e2: float16[2] = 0
                                e2[0] = k_d[base]
                                e2[1] = v_d[base]
                                fifo_fill_KV[c].put(e2)

        # ---- load_kv_replay: capture once, then ONE flattened replay per head with the
        #      V lag wrapping across q-tiles (tile g: K(g % NKV), V((g-1) % NKV)) --------
        @df.kernel(mapping=[1], args=[])
        def load_kv_replay():
            k_buf: float16[NUM_KV_TILES, NCOLS * CW, NB] @ Memory(
                resource=KVBUF_RESOURCE, storage_type="RAM_2P"
            )
            v_buf: float16[NUM_KV_TILES, NCOLS * CW, NB] @ Memory(
                resource=KVBUF_RESOURCE, storage_type="RAM_2P"
            )
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tile_r in range(NUM_KV_TILES):
                        for jjr in range(HEAD_DIM):
                            jb: int32 = jjr // CW
                            jw: int32 = jjr % CW
                            with allo.meta_for(NCOLS) as c:
                                e2r: float16[2] = fifo_fill_KV[c].get()
                                k_buf[tile_r, c * CW + jw, jb] = e2r[0]
                                v_buf[tile_r, c * CW + jw, jb] = e2r[1]
                    for gb in range(GT * NB):
                        g: int32 = gb // NB
                        jjp: int32 = gb % NB
                        kidx: int32 = g % NKV
                        if g >= GT - VLAG:
                            kidx = NKV - 1                          # trailing dummy K tiles
                        vidx: int32 = (g + NKV - VLAG) % NKV
                        if g < VLAG:
                            vidx = 0                                # leading dummy V tiles
                        with allo.meta_for(NCOLS) as c2:
                            e: float16[2 * CW] = 0
                            with allo.meta_for(CW) as w:
                                e[w] = k_buf[kidx, c2 * CW + w, jjp]
                                e[CW + w] = v_buf[vidx, c2 * CW + w, jjp]
                            fifo_in_KV[c2].put(e)

        # ---- the PE array ----------------------------------------------------
        @df.kernel(mapping=[P0, P1], args=[])
        def pe():
            i, j = df.get_pid()
            q_loc: float16[CW, 2 * NB] @ Memory(
                resource="LUTRAM", storage_type="RAM_2P"
            ) = 0
            exp_tab: uint16[EXP_NTAB] @ Memory(
                resource="LUTRAM", storage_type="ROM_1P"
            ) = EXP_POS_NP
            acc: float32[CW] = 0
            pp: float32[VLAG] = 0         # ring: p of key g lives in slot g % VLAG until key g+VLAG uses it
            asnap: float32[CW] = 0        # acc after the previous beat (unconditional consumer of the fmacc)
            rr: float32[2] = 0            # boundary ring: sum -> x -> t -> p, each read BSP beats after its write
            sprev: float32[1] = 0         # lane sum of the previous key
            areg: float32[1] = 0
            breg: float32[1] = 0
            invD: float32[1] = 0
            oacc: float32[2, CW, NB] = 0

            with allo.meta_if(i in {0, P0 - 1} and j in {0, P1 - 1}):
                pass

            # ---- top border: forward K+V, one flattened loop per head ----------
            with allo.meta_elif(i == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for dkv in range(GT * NB):
                            ekv: float16[2 * CW] = fifo_in_KV[j - 1].get()
                            fifo_KV[i + 1, j].put(ekv)

            # ---- left border ------------------------------------------------------
            with allo.meta_elif(j == 0):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for dq0 in range(NB):
                            q0: float16[CW] = fifo_in_Q[i - 1].get()
                            fifo_Q[i, j + 1].put(q0)
                        for bl in range(GT * NB):
                            gl: int32 = bl // NB
                            dbl: int32 = bl % NB
                            qtl: int32 = gl // NKV
                            ktl: int32 = gl % NKV
                            if ktl == 0:
                                if qtl < NUM_Q_TILES - 1:
                                    qn: float16[CW] = fifo_in_Q[i - 1].get()
                                    fifo_Q[i, j + 1].put(qn)
                            if dbl == 0:
                                m_init: float32 = 0.0
                                d_init: float32 = 0.0
                                fifo_m[i, j + 1].put(m_init)
                                fifo_d[i, j + 1].put(d_init)
                            o_init: float32[CW] = 0
                            fifo_o[i, j + 1].put(o_init)

            # ---- bottom border ----------------------------------------------------
            with allo.meta_elif(i == P0 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for dkv2 in range(GT * NB):
                            _kv: float16[2 * CW] = fifo_KV[i, j].get()

            # ---- right border: O accumulate; emit q-tile qt during tile 1 of qt+1 -----
            with allo.meta_elif(j == P1 - 1):
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for dq2 in range(NB):
                            _q0: float16[CW] = fifo_Q[i, j].get()
                        for bt in range(GT * NB):
                            g: int32 = bt // NB
                            db3: int32 = bt % NB
                            qtg: int32 = g // NKV
                            ktg: int32 = g % NKV
                            if ktg == 0:
                                if qtg < NUM_Q_TILES - 1:
                                    _qn: float16[CW] = fifo_Q[i, j].get()
                            ov: float32[CW] = fifo_orb[i - 1].get()
                            if g >= VLAG:
                                # o of tile g-VLAG (q-tile (g-VLAG)//NKV, parity selects the buffer)
                                tp: int32 = g - VLAG
                                qtp: int32 = tp // NKV
                                par: int32 = qtp % 2
                                if db3 == 0:
                                    ab: float32[2] = fifo_ab[i - 1].get()
                                    areg[0] = ab[0]
                                    breg[0] = ab[1]
                                with allo.meta_for(CW) as w:
                                    oacc[par, w, db3] = areg[0] * oacc[par, w, db3] + breg[0] * ov[w]
                                # emit q-tile qtp-1 during the SECOND tile of q-tile qtp
                                if tp % NKV == 1:
                                    if qtp > 0:
                                        if db3 == 0:
                                            invD[0] = fifo_inv[i - 1].get()
                                        pe_: int32 = 1 - par
                                        out16: float16[CW] = 0
                                        with allo.meta_for(CW) as w:
                                            ow: float32 = oacc[pe_, w, db3] * invD[0]
                                            out16[w] = ow
                                        fifo_out[i - 1].put(out16)
                        lastb: int32 = (NUM_Q_TILES - 1) % 2
                        invD[0] = fifo_inv[i - 1].get()
                        for dq4 in range(NB):
                            outl: float16[CW] = 0
                            with allo.meta_for(CW) as w:
                                owl: float32 = oacc[lastb, w, dq4] * invD[0]
                                outl[w] = owl
                            fifo_out[i - 1].put(outl)

            # ---- compute PE: ONE flattened beat loop per head ------------------------
            with allo.meta_else():
                for b in range(BATCH_SIZE):
                    for h in range(NUM_HEADS):
                        for dq0i in range(NB):
                            qv0: float16[CW] = fifo_Q[i, j].get()
                            with allo.meta_for(CW) as w:
                                q_loc[w, dq0i] = qv0[w]
                            fifo_Q[i, j + 1].put(qv0)
                        with allo.meta_for(VLAG) as v:
                            pp[v] = 0.0
                        with allo.meta_for(CW) as w:
                            acc[w] = 0.0
                        with allo.meta_for(2) as v:
                            rr[v] = 0.0
                        with allo.meta_for(CW) as w:
                            asnap[w] = 0.0
                        sprev[0] = 0.0
                        for gb in range(GT * NB):
                            g: int32 = gb // NB
                            dm: int32 = gb % NB
                            pslot: int32 = g % VLAG
                            pnext: int32 = (g + 1) % VLAG
                            qt: int32 = g // NKV
                            kt: int32 = g % NKV
                            cur: int32 = qt % 2
                            cbase: int32 = cur * NB
                            nbase: int32 = (1 - cur) * NB
                            if kt == 0:
                                if qt < NUM_Q_TILES - 1:
                                    qn2: float16[CW] = fifo_Q[i, j].get()
                                    with allo.meta_for(CW) as w:
                                        q_loc[w, nbase + dm] = qn2[w]
                                    fifo_Q[i, j + 1].put(qn2)

                            kv: float16[2 * CW] = fifo_KV[i, j].get()
                            fifo_KV[i + 1, j].put(kv)

                            # phase A: pure FMA-accumulate, NEVER reset inside the loop (any mux on
                            # the accumulator breaks the DSPFP32 fmacc inference: II=7 measured).
                            # acc runs across the whole head; x = sum - previous sum stays exact to
                            # ~6e-8 * |acc| (|acc| <= keys/head * |x|: ~2e-4 on a score at L=1024).
                            # The boundary below only READS acc.
                            with allo.meta_for(CW) as w:
                                qw: float32 = q_loc[w, cbase + dm]
                                kw: float32 = kv[w]
                                acc[w] = acc[w] + qw * kw

                            # phase B of key g-VLAG: its p was written VLAG keys ago into slot g % VLAG
                            oi: float32[CW] = fifo_o[i, j].get()
                            oo: float32[CW] = 0
                            pw: float32 = pp[pslot]
                            with allo.meta_for(CW) as w:
                                vw: float32 = kv[CW + w]
                                oo[w] = oi[w] + pw * vw
                            with allo.meta_if(j == P1 - 2):
                                fifo_orb[i - 1].put(oo)
                            with allo.meta_else():
                                fifo_o[i, j + 1].put(oo)

                            # ---- boundary on ONE shared fp32 adder. During key g this finishes
                            # key g-1: the lane sum at beat 0, the rest BSP beats apart. Every intermediate goes
                            # through the 2-entry ring rr and is read exactly BSP beats after it
                            # was written (indices are computed, not constant: Vitis drops a
                            # dependence pragma on scalars). a - b is a + (-b).
                            # The accumulators are copied to asnap on EVERY beat and the lane sum
                            # uses the copy one beat later. Feeding acc straight into the operand
                            # mux makes that mux the only consumer of the fmacc, and Vitis then
                            # predicates the fmacc's in_valid with the mux condition: RTL
                            # accumulates one product per key while C simulation stays correct
                            # (C/RTL mismatch, found in cosim; check_fma_predicate.py guards it).
                            sn0: float32 = asnap[0]
                            sn1: float32 = 0.0
                            with allo.meta_if(CW == 2):
                                sn1 = asnap[1]
                            with allo.meta_for(CW) as w:
                                asnap[w] = acc[w]
                            ridx: int32 = 0
                            widx: int32 = 0
                            wen: int32 = 0
                            if dm == 0:
                                wen = 1
                            if dm == B1:
                                widx = 1
                                wen = 1
                            if dm == B2:
                                ridx = 1
                                wen = 1
                            if dm == B3:
                                widx = 1
                                wen = 1
                            if dm == B4:
                                ridx = 1
                            prev: float32 = rr[ridx]
                            opa: float32 = prev
                            opb: float32 = 0.0
                            if dm == 0:                            # op0: lane sum of key g-1
                                opa = sn0
                                opb = sn1
                            if dm == B1:                           # op1: x = sum - previous sum
                                opb = -sprev[0]
                                sprev[0] = prev
                            if dm == B2:                           # op2: t = x - x_ref
                                m_in: float32 = fifo_m[i, j].get()
                                m_ref: float32 = m_in
                                with allo.meta_if(j == 1):
                                    m_ref = prev
                                fifo_m[i, j + 1].put(m_ref)
                                opb = -m_ref
                            if dm == B4:                           # op3: d_out = p + d_in
                                d_in: float32 = fifo_d[i, j].get()
                                opb = d_in
                            rs: float32 = opa + opb                # the only fp32 adder of the PE
                            if dm == B4:
                                fifo_d[i, j + 1].put(rs)
                            wv: float32 = rs

                            # p = 2^t from the bits of t: F = floor(t * 1024) exactly,
                            # exponent = F >> 10 (clamped to +-126), table index = F & 1023.
                            if dm == B3:
                                tb_p: uint32 = prev.bitcast()
                                sg_p: uint32 = (tb_p >> 31) & 1
                                te_p: uint32 = (tb_p >> 23) & 255
                                tn_p: uint32 = tb_p & 8388607
                                tm_p: uint32 = tn_p | 8388608
                                au_p: uint32 = 0               # floor(|t| * 1024), < 2^18
                                st_p: uint32 = 0               # sticky: |t| * 1024 has a fraction
                                big_p: uint32 = 0              # |t| >= 256: clamp
                                if te_p >= 135:
                                    big_p = 1
                                else:
                                    if te_p >= 117:
                                        sh_p: uint32 = 140 - te_p
                                        au_p = tm_p >> sh_p
                                        bk_p: uint32 = au_p << sh_p
                                        if bk_p != tm_p:
                                            st_p = 1
                                    else:
                                        if te_p + tn_p > 0:
                                            st_p = 1
                                fi_p: int32 = au_p >> 10
                                ixi_p: int32 = au_p & 1023
                                if sg_p == 1:                  # floor(-a) = -ceil(a)
                                    an_p: uint32 = au_p + st_p
                                    hn_p: int32 = an_p >> 10
                                    ln_p: int32 = an_p & 1023
                                    fi_p = 0 - hn_p
                                    ixi_p = 0
                                    if ln_p > 0:
                                        fi_p = fi_p - 1
                                        ixi_p = 1024 - ln_p
                                if big_p == 1:
                                    ixi_p = 0
                                    fi_p = 126
                                    if sg_p == 1:
                                        fi_p = -126
                                if fi_p > 126:
                                    fi_p = 126
                                if fi_p < -126:
                                    fi_p = -126
                                mt_p: uint16 = exp_tab[ixi_p]
                                mw_p: uint32 = mt_p & 1023
                                eb_p: int32 = fi_p + 127
                                ebu_p: uint32 = eb_p
                                pb_p: uint32 = (ebu_p << 23) | (mw_p << 13)
                                pv: float32 = pb_p.bitcast()
                                wv = pv
                                if g > 0:                      # key -1 does not exist: its p stays 0
                                    pp[pnext] = pv             # p of key g-1, first read by key g+1
                            if wen == 1:
                                rr[widx] = wv

        # ---- rb_merge: per-tile LSE state per row. A plain sequential loop (pipeline off):
        #      a pipelined loop only advances when the NEXT token arrives, so (a, b) of tile g
        #      would leave it depth/II tiles late and the right border would need that many
        #      tiles of o buffered (deadlock at D_RB = 4 tiles with the 43-deep tkv6b merge).
        #      One of a, b is always 1, so a tile is one fsub, one 2^t and one FMA:
        #          m_t >  M :  a = 2^(M - m_t), b = 1      D = d_t + a * D
        #          m_t <= M :  a = 1, b = 2^(m_t - M)      D = D + b * d_t
        @df.kernel(mapping=[NROWS], args=[])
        def rb_merge():
            r = df.get_pid()
            exp_tab: uint16[EXP_NTAB] @ Memory(
                resource="LUTRAM", storage_type="ROM_1P"
            ) = EXP_POS_NP
            Mreg: float32[1] = 0
            Dreg: float32[1] = 0
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for g in range(GT):
                        m_t: float32 = fifo_m[r + 1, P1 - 1].get()
                        d_t: float32 = fifo_d[r + 1, P1 - 1].get()
                        # token g carries key g-1 (the PE finishes key g-1 during key g);
                        # token 0 is a dummy, keys >= NQ*NKV are the trailing dummies.
                        if g >= 1:
                            if g < GT - 1:
                                kt: int32 = (g - 1) % NKV
                                ab: float32[2] = 0
                                if kt == 0:
                                    Mreg[0] = m_t
                                    Dreg[0] = d_t
                                    ab[0] = 0.0
                                    ab[1] = 1.0
                                else:
                                    Mold: float32 = Mreg[0]
                                    Dold: float32 = Dreg[0]
                                    up: int32 = 0
                                    hi: float32 = Mold
                                    lo: float32 = m_t
                                    if m_t > Mold:
                                        up = 1
                                        hi = m_t
                                        lo = Mold
                                    tq: float32 = lo - hi              # <= 0
                                    tb_a: uint32 = tq.bitcast()
                                    sg_a: uint32 = (tb_a >> 31) & 1
                                    te_a: uint32 = (tb_a >> 23) & 255
                                    tn_a: uint32 = tb_a & 8388607
                                    tm_a: uint32 = tn_a | 8388608
                                    au_a: uint32 = 0
                                    st_a: uint32 = 0
                                    big_a: uint32 = 0
                                    if te_a >= 135:
                                        big_a = 1
                                    else:
                                        if te_a >= 117:
                                            sh_a: uint32 = 140 - te_a
                                            au_a = tm_a >> sh_a
                                            bk_a: uint32 = au_a << sh_a
                                            if bk_a != tm_a:
                                                st_a = 1
                                        else:
                                            if te_a + tn_a > 0:
                                                st_a = 1
                                    fi_a: int32 = au_a >> 10
                                    ixi_a: int32 = au_a & 1023
                                    if sg_a == 1:
                                        an_a: uint32 = au_a + st_a
                                        hn_a: int32 = an_a >> 10
                                        ln_a: int32 = an_a & 1023
                                        fi_a = 0 - hn_a
                                        ixi_a = 0
                                        if ln_a > 0:
                                            fi_a = fi_a - 1
                                            ixi_a = 1024 - ln_a
                                    if big_a == 1:
                                        ixi_a = 0
                                        fi_a = 126
                                        if sg_a == 1:
                                            fi_a = -126
                                    if fi_a > 126:
                                        fi_a = 126
                                    if fi_a < -126:
                                        fi_a = -126
                                    mt_a: uint16 = exp_tab[ixi_a]
                                    mw_a: uint32 = mt_a & 1023
                                    eb_a: int32 = fi_a + 127
                                    ebu_a: uint32 = eb_a
                                    pb_a: uint32 = (ebu_a << 23) | (mw_a << 13)
                                    ev: float32 = pb_a.bitcast()
                                    xs: float32 = d_t
                                    ys: float32 = Dold
                                    ab[0] = 1.0
                                    ab[1] = ev
                                    if up == 1:
                                        xs = Dold
                                        ys = d_t
                                        ab[0] = ev
                                        ab[1] = 1.0
                                    Dreg[0] = ys + ev * xs
                                    Mreg[0] = hi
                                fifo_ab[r].put(ab)
                                if kt == NKV - 1:
                                    dfin: float32 = Dreg[0]
                                    fifo_D[r].put(dfin)

        # ---- rb_inv: 1/D once per q-tile, off the per-tile path (with the divide inside
        #      rb_merge every tile costs 39 cycles > NB and the merge throttles the array).
        #      Sequential for the same reason as rb_merge.
        @df.kernel(mapping=[NROWS], args=[])
        def rb_inv():
            r = df.get_pid()
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for qi in range(NUM_Q_TILES):
                        dq: float32 = fifo_D[r].get()
                        inv: float32 = 1.0 / dq
                        fifo_inv[r].put(inv)

        # ---- store: beat-serial drain, CW per beat ----------------------------
        @df.kernel(mapping=[1], args=[output_mem])
        def store(global_mem: float16[OUT_ELEMS]):
            for b in range(BATCH_SIZE):
                for h in range(NUM_HEADS):
                    for tr in range(0, CONTEXT_LENGTH, NROWS):
                        with allo.meta_for(NROWS) as rr:
                            for jj2 in range(NB):
                                ov2: float16[CW] = fifo_out[rr].get()
                                with allo.meta_for(CW) as w:
                                    idx = (
                                        (b * CONTEXT_LENGTH + (tr + rr)) * NUM_HEADS + h
                                    ) * HEAD_DIM + jj2 * CW + w
                                    global_mem[idx] = ov2[w]

    return top


def _inject_dependence(prj, NB, VLAG, BSP):
    """Carried distances Vitis cannot see (it would assume 1 and lose II=1):
      pp   written at beat B3 of key g (slot (g+1) % VLAG), first read at beat 0 of key g+1:
           NB - B3 iterations.
      rr   every boundary value is read exactly BSP iterations after it is written (that is
           the whole schedule of the shared adder: result at S+5, next operand at S+BSP).
    Both must stay dynamically indexed arrays: on a scalar (constant index) Vitis ignores the
    pragma and assumes distance 1 (II=4 measured)."""
    import re as _re
    path = os.path.join(prj, "kernel.cpp")
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    B3 = 3 * BSP
    rules = [(_re.compile(r"^(\s*)float (pp\d*)\[%d\];" % VLAG), NB - B3),
             (_re.compile(r"^(\s*)float (rr\d*)\[2\];"), BSP)]
    out, n = [], [0, 0]
    for ln in lines:
        out.append(ln)
        for k, (decl, dist) in enumerate(rules):
            m = decl.match(ln)
            if m:
                out.append(f"{m.group(1)}#pragma HLS dependence variable={m.group(2)} inter distance={dist} true")
                n[k] += 1
    if n[0] == 0 or n[1] != n[0]:
        raise RuntimeError(f"dependence injection: {n[0]} pp rings, {n[1]} rr rings")
    # rb_merge's tile loop and rb_inv's q-tile loop must NOT be pipelined (Vitis would
    # auto-pipeline them): see the kernel.
    lab = _re.compile(r"^(\s*)l_S_(?:g|qi)_\d+_(?:g|qi)\d*: for ")
    out2, nm = [], 0
    for ln in out:
        out2.append(ln)
        m = lab.match(ln)
        if m:
            out2.append(f"{m.group(1)}#pragma HLS pipeline off")
            nm += 1
    if nm == 0:
        raise RuntimeError("no rb_merge / rb_inv loop matched")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out2))
    print(f"  kernel.cpp: dependence distance {NB - B3} on {n[0]} pp rings, {BSP} on {n[1]} rr rings; "
          f"pipeline off on {nm} rb_merge / rb_inv loops")


def get_scheduled_tkv6c(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC=1):
    P0 = NROWS + 2
    P1 = NCOLS + 2
    s = df.customize(get_tkv6c_flash_attention(
        BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC))

    def pipe_all(band, base, maxn=40):
        for n in range(maxn):
            nm = base if n == 0 else f"{base}_{n}"
            try:
                s.pipeline(getattr(band, nm))
            except Exception:
                pass

    def part(name, dim):
        try:
            s.partition(name, dim=dim)
        except Exception as e:  # noqa: BLE001
            print(f"[partition {name} dim={dim}] FAILED: {e}")

    def loops(name):
        try:
            return s.get_loops(name).S_b_0
        except Exception:  # noqa: BLE001
            return None

    lp = loops("load_kv_fill_0")
    if lp is not None:
        pipe_all(lp, "jjf")
    part("load_kv_replay_0:k_buf", 2)
    part("load_kv_replay_0:v_buf", 2)
    lp = loops("load_kv_replay_0")
    if lp is not None:
        for base in ["jjr", "gb"]:
            pipe_all(lp, base)
    lp = loops("load_q_0")
    if lp is not None:
        pipe_all(lp, "jj")
    for pi in range(P0):
        for pj in range(P1):
            pname = f"pe_{pi}_{pj}"
            lp = loops(pname)
            if lp is None:
                continue
            interior = 1 <= pi <= P0 - 2 and 1 <= pj <= P1 - 2
            if interior:
                for arr in ("acc", "asnap", "rr", "sprev", "pp", "q_loc"):
                    part(f"{pname}:{arr}", 1)
                pipe_all(lp, "gb")
                pipe_all(lp, "dq0i")
            else:
                if pj == P1 - 1 and 1 <= pi <= P0 - 2:
                    for arr in ("areg", "breg", "invD"):
                        part(f"{pname}:{arr}", 1)
                    part(f"{pname}:oacc", 1)
                    part(f"{pname}:oacc", 2)
                for base in ["dkv", "dkv2", "dq0", "bl", "dq2", "bt", "dq4"]:
                    pipe_all(lp, base)
    for r in range(NROWS):
        for arr in ("Mreg", "Dreg"):
            part(f"rb_merge_{r}:{arr}", 1)
        # rb_merge's g loop is deliberately left unpipelined (pipeline off is injected)
    lp = loops("store_0")
    if lp is not None:
        pipe_all(lp, "jj2")
    return s


def preflight(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC=1):
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    NB = HEAD_DIM // CW
    NKV = CONTEXT_LENGTH // NCOLS
    NQ = CONTEXT_LENGTH // NROWS
    GT = NQ * NKV + 2
    print("-" * 64)
    BSP = int(os.environ.get("BSP", "6"))
    assert CW in (1, 2) and NB >= 4 * BSP + 1
    print(f"PRE-FLIGHT tkv6c (VLAG=2, shared boundary adder, BSP={BSP})  dh={HEAD_DIM} CW={CW} NB={NB} {NROWS}x{NCOLS} NKV={NKV} NQ={NQ} tiles/head={GT}")
    assert HEAD_DIM % CW == 0 and NQ >= 2 and NKV >= 2
    assert CONTEXT_LENGTH % NROWS == 0 and CONTEXT_LENGTH % NCOLS == 0
    per = GT * NB
    print(f"  KV beats/col: replay={per} top={per} interior={per} bottom={per}  OK")
    print(f"  o beats/row: left={per} right={per}  OK")
    print(f"  m,d per row per head: PE puts {GT}, merge gets {GT} (token 0 and the last are dummies)  OK")
    print(f"  boundary beats (in the next key): lane sum 0, x {BSP}, t {2 * BSP}, p {3 * BSP}, d {4 * BSP}; "
          f"p ready {NB - 3 * BSP} beats before its first use  OK")
    print(f"  (a,b) per row per head: merge puts {GT - 2}, right border gets {GT - 2}; 1/D: {NQ} each  OK")
    print(f"  out beats/row: emitted {(NQ - 1) * NB} in-loop + {NB} epilogue = store gets {NQ * NB}  OK")
    print(f"  rb_merge must finish a tile in < NB = {NB} cycles on average (sequential loop)")
    print(f"  DSP/PE = {2 * CW}; beats/head = {per} (useful {NQ * NKV * NB}, {100 * NQ * NKV * NB / per:.2f}%)")
    print("PRE-FLIGHT PASSED")
    print("-" * 64)


def run_test_with_params(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC=1, mode="csyn"):
    preflight(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC)
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
    OUT_ELEMS = BATCH_SIZE * CONTEXT_LENGTH * NUM_HEADS * HEAD_DIM
    print("=" * 64)
    print(f"tkv6c: B={BATCH_SIZE} L={CONTEXT_LENGTH} HIDDEN={HIDDEN_SIZE} H={NUM_HEADS} dh={HEAD_DIM} "
          f"NROWS={NROWS} NCOLS={NCOLS} CW={CW} mode={mode}")
    print("=" * 64)
    if mode == "csyn":
        Q = K = V = B_out = B_golden = None
    else:
        Q, K, V, B_out, B_golden = _golden_inputs(
            BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, HEAD_DIM, OUT_ELEMS)
    if not hls.is_available("vitis_hls"):
        print("Vitis HLS not available, skipping.")
        return
    s = get_scheduled_tkv6c(BATCH_SIZE, CONTEXT_LENGTH, HIDDEN_SIZE, NUM_HEADS, NROWS, NCOLS, CW, RPC)
    prj = os.environ.get("PRJ", f"/data/ry375/scratch/tkv6c_dh{HEAD_DIM}_L{CONTEXT_LENGTH}_{mode}.prj")
    target = os.environ.get("TARGET", "vitis_hls")     # vitis_hls | catapult
    dev = os.environ.get("DEVICE", "")
    cfg = {"device": dev} if (dev and target == "vitis_hls") else {}
    if os.environ.get("FREQ"):
        cfg["frequency"] = int(os.environ["FREQ"])
    if target == "catapult" and "frequency" not in cfg:
        cfg["frequency"] = 500                          # Allo's Catapult tcl needs one
    hls_mod = (s.build(target=target, mode=mode, project=prj, configs=cfg)
               if cfg else s.build(target=target, mode=mode, project=prj))
    if target == "vitis_hls":
        os.environ["FLATTEN_QT"] = "1"    # no q-tile loop exists any more; skip the flatten-off patch
        _inject_op_impl(prj)
        _inject_dependence(prj, HEAD_DIM // CW, 2, int(os.environ.get("BSP", "6")))
    else:
        print(f"  TARGET={target}: no Vitis kernel.cpp/run.tcl patches applied. Catapult equivalents "
              f"(see HANDOFF_TKV6B.md): carried distances pp = NB - B3, rr = BSP, "
              f"the shared boundary fp32 adder as a plain adder, exp table as ROM.")
    if target == "catapult" and os.environ.get("EMIT_ONLY", "1") == "1":
        print(f"emitted Catapult project -> {prj} (EMIT_ONLY=1: not running `catapult`; "
              f"run `make` in the project or set EMIT_ONLY=0)")
        return
    if mode == "csyn" and os.environ.get("GEN_ONLY", "0") == "1":
        print(f"GEN_ONLY -> {prj}")
        return
    if mode != "csyn":
        hls_mod(Q, K, V, B_out)
        np.testing.assert_allclose(B_out, B_golden, rtol=0.02, atol=1e-2)
        print(f"{mode} PASSED -> {prj}")
    else:
        hls_mod()
        print(f"csyn done -> {prj}")
        if target == "vitis_hls":
            import check_fma_predicate
            check_fma_predicate.check(os.path.join(prj, "out.prj", "solution1", "syn", "verilog"))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "128")
    CW = int(os.environ.get("CW", "2"))
    if int(os.environ.get("FREE", "0")):
        fdh = int(os.environ.get("DH", "64"))
        farr = int(os.environ.get("ARR", "8"))
        run_test_with_params(
            BATCH_SIZE=1, CONTEXT_LENGTH=int(os.environ.get("CTX", "1024")), HIDDEN_SIZE=fdh, NUM_HEADS=1,
            NROWS=int(os.environ.get("NROWS", farr)), NCOLS=int(os.environ.get("NCOLS", farr)),
            CW=CW, mode=os.environ.get("MODE", "csyn"),
        )
        raise SystemExit(0)
    run_test_with_params(BATCH_SIZE=1, CONTEXT_LENGTH=64, HIDDEN_SIZE=64, NUM_HEADS=1,
                         NROWS=4, NCOLS=4, CW=CW, mode="csyn")
