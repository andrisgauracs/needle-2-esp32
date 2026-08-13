#!/usr/bin/env python3
"""Validate the C engine against a NumPy reference decode of the .cact blob.

The reference here re-implements the format from needle/model/export.py
(_cq_unpack) rather than calling it, because the shipped needle2.cact is v3
(tag 0x05E12A83) and the Python package only reads v2.

  python3 tools/check_engine.py model/needle2.cact build/nd_dump
"""
import struct
import subprocess
import sys

import numpy as np

TAG = 0x05E12A83
HDR_WORDS = 30
REC = 44


def read_header(raw):
    w = struct.unpack_from("<%dI" % HDR_WORDS, raw, 0)
    assert w[0] == TAG, "bad tag %#x" % w[0]
    return {
        "num_tensors": w[1], "codebook_len": w[2], "kv_window": w[3],
        "kv_bits": w[4], "vocab_size": w[5], "d_model": w[6],
        "num_heads": w[7], "num_kv_heads": w[8], "num_layers": w[9],
        "head_dim": w[10], "max_seq_len": w[11], "attn_dim": w[12],
        "mhc_lanes": w[13], "engram_slots": w[14], "engram_sub_dim": w[15],
        "engram_conv_taps": w[16], "engram_tables": w[17],
        "engram_dilation": w[18], "num_orders": w[19],
        "orders": w[20:20 + w[19]], "num_sites": w[24],
        "sites": w[25:25 + w[24]],
        "rope_theta": struct.unpack_from("<f", raw, 29 * 4)[0],
    }


def read_directory(raw, h):
    off = HDR_WORDS * 4
    codebook = np.frombuffer(raw[off:off + h["codebook_len"] * 4], np.float32)
    off += h["codebook_len"] * 4
    tensors = []
    for _ in range(h["num_tensors"]):
        rec = struct.unpack_from("<BBHIIIIQQII", raw, off)
        off += REC
        tensors.append({
            "dtype": rec[0], "ndim": rec[1], "shape": rec[3:3 + rec[1]],
            "offset": rec[7], "nbytes": rec[8], "group": rec[9], "bits": rec[10],
        })
    return codebook, tensors


def codebook_for(codebook, bits):
    return {2: codebook[0:4], 3: codebook[4:12], 4: codebook[12:28]}[bits]


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def unpack_lsb(packed, bits, count):
    """Inverse of export._pack_lsb: one LSB-first bitstream per row."""
    bitpos = np.arange(count) * bits
    byt = bitpos >> 3
    sh = bitpos & 7
    wide = packed.astype(np.uint32)
    lo = wide[:, byt]
    hi = np.where(byt + 1 < packed.shape[1], wide[:, np.minimum(byt + 1, packed.shape[1] - 1)], 0)
    val = lo | (hi << 8)
    return ((val >> sh) & ((1 << bits) - 1)).astype(np.uint8)


def dequant(raw, t, codebook):
    out, in_dim = t["shape"]
    g = t["group"]
    bits = t["bits"]
    in_pad = (in_dim + g - 1) // g * g
    rowbytes = in_pad * bits // 8
    blob = raw[t["offset"]:t["offset"] + t["nbytes"]]
    packed = np.frombuffer(blob[:out * rowbytes], np.uint8).reshape(out, rowbytes)
    norms = np.frombuffer(blob[out * rowbytes:], np.float16).reshape(out, in_pad // g)
    idx = unpack_lsb(packed, bits, in_pad)
    cb = codebook_for(codebook, bits)
    v = cb[idx].reshape(out, in_pad // g, g) * norms[:, :, None].astype(np.float32)
    w = v @ hadamard(g)
    return w.reshape(out, in_pad)[:, :in_dim]


def seeded_vec(n, seed):
    """Mirror of seeded_vec() in host/nd_dump.c."""
    s = (seed * 1103515245 + 12345) & 0xFFFFFFFF
    out = np.empty(n, np.float32)
    for i in range(n):
        s = (s * 1103515245 + 12345) & 0xFFFFFFFF
        v = (s >> 8) % 2000 - 1000
        out[i] = v / 1000.0
    return out


def run(binary, blob_path, *args):
    r = subprocess.run([binary, blob_path, *map(str, args)],
                       capture_output=True, text=True, check=True)
    return np.array([float(x) for x in r.stdout.split()], np.float32)


def main():
    blob_path, binary = sys.argv[1], sys.argv[2]
    raw = open(blob_path, "rb").read()
    h = read_header(raw)
    codebook, tensors = read_directory(raw, h)

    print("header:", {k: h[k] for k in ("num_tensors", "d_model", "num_layers",
                                        "vocab_size", "kv_window", "kv_bits")})

    # Compare the C header dump field by field.
    dumped = dict(
        line.split(None, 1) for line in
        subprocess.run([binary, blob_path, "header"], capture_output=True,
                       text=True, check=True).stdout.strip().splitlines())
    for key in ("num_tensors", "d_model", "num_layers", "vocab_size",
                "num_heads", "num_kv_heads", "head_dim", "attn_dim",
                "mhc_lanes", "kv_window", "kv_bits", "engram_slots",
                "engram_sub_dim", "engram_tables"):
        assert int(dumped[key]) == h[key], (key, dumped[key], h[key])
    print("header fields match C reader: OK")

    # Pick a representative CQ tensor of each width.
    picks = []
    for width in (4, 2):
        for i, t in enumerate(tensors):
            if t["dtype"] == 3 and t["bits"] == width:
                picks.append(i)
                break
    picks += [2, 3, 7]  # q_proj, k_proj, gate_proj of layer 0

    worst_row = worst_gemv = 0.0
    for i in sorted(set(picks)):
        t = tensors[i]
        ref = dequant(raw, t, codebook)

        for row in (0, t["shape"][0] // 2, t["shape"][0] - 1):
            got = run(binary, blob_path, "row", i, row)
            err = np.max(np.abs(got - ref[row])) / max(1e-9, np.max(np.abs(ref[row])))
            worst_row = max(worst_row, err)

        x = seeded_vec(t["shape"][1], 7)
        got = run(binary, blob_path, "gemv", i, 7)
        exp = ref @ x
        err = np.max(np.abs(got - exp)) / max(1e-9, np.max(np.abs(exp)))
        worst_gemv = max(worst_gemv, err)
        print(f"tensor {i:3d} shape={t['shape']} bits={t['bits']}  "
              f"row_relerr={worst_row:.2e}  gemv_relerr={err:.2e}")

    print(f"\nworst dequant rel err: {worst_row:.3e}")
    print(f"worst gemv    rel err: {worst_gemv:.3e}")
    ok = worst_row < 1e-5 and worst_gemv < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
