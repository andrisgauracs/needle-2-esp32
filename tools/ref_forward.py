#!/usr/bin/env python3
"""NumPy reference forward pass for Needle 2, read straight from the .cact blob.

This mirrors needle/model/architecture.py (SimpleAttentionNetwork) exactly,
but without JAX and without the checkpoint - it decodes the deployment blob,
so it validates the same bytes the C engine reads. Full-sequence and
non-incremental on purpose: comparing it against the C engine's incremental
decode also validates the KV cache and ring-buffer logic.

  python3 tools/ref_forward.py model/needle2.cact --tokens 2 100 200 300
"""
import argparse
import sys

import numpy as np

from check_engine import (dequant, read_directory, read_header)

EPS = 1e-6
EG_SEED = 0x9E3779B9
EG_PRIME = 0x01000193


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def rms_unit(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS)


def zcrms(x, scale):
    return (1.0 + scale) * x / np.sqrt((x ** 2).mean(-1, keepdims=True) + EPS)


def hadamard(n):
    H = np.array([[1.0]], np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def sinkhorn(logits, iters=20):
    log_K = logits.astype(np.float64)
    for _ in range(iters):
        log_K = log_K - _logsumexp(log_K, -1)
        log_K = log_K - _logsumexp(log_K, -2)
    return np.exp(log_K).astype(np.float32)


def _logsumexp(a, axis):
    mx = a.max(axis=axis, keepdims=True)
    return mx + np.log(np.exp(a - mx).sum(axis=axis, keepdims=True))


class Ref:
    def __init__(self, path, int8_kv=False):
        self.int8_kv = int8_kv
        raw = open(path, "rb").read()
        self.raw = raw
        self.h = read_header(raw)
        self.codebook, self.dir = read_directory(raw, self.h)
        self.cache = {}

    def t(self, i):
        """Decode tensor i to float32 (cached)."""
        if i in self.cache:
            return self.cache[i]
        rec = self.dir[i]
        if rec["dtype"] == 3:
            v = dequant(self.raw, rec, self.codebook)
        elif rec["dtype"] == 1:
            v = np.frombuffer(self.raw[rec["offset"]:rec["offset"] + rec["nbytes"]],
                              np.float16).astype(np.float32)
            shape = rec["shape"]
            if rec["ndim"] == 3:      # mhc_b_res is (L, n, n)
                n = shape[1]
                v = v.reshape(shape[0], n, n)
            else:
                v = v.reshape(shape)
        else:
            v = None
        self.cache[i] = v
        return v

    def layer(self, i, k):
        return self.t(1 + i * 14 + k)

    def engram_kv(self, tokens):
        """k, v per engram site for the whole sequence."""
        h = self.h
        T = len(tokens)
        orders = list(h["orders"])
        tables_n = h["engram_tables"]
        heads = tables_n // len(orders)
        slots = h["engram_slots"]
        sub = h["engram_sub_dim"]
        dil = h["engram_dilation"]
        taps_n = h["engram_conv_taps"]
        base = 1 + h["num_layers"] * 14 + 9

        u = np.asarray(tokens, np.uint32)
        idx = np.zeros((T, tables_n), np.int64)
        ok = np.zeros((T, tables_n), np.float32)
        col = 0
        for oi, order in enumerate(orders):
            for hh in range(heads):
                seed = np.uint32((EG_SEED * (oi * heads + hh + 1)) & 0xFFFFFFFF)
                acc = np.full(T, seed, np.uint32)
                for j in range(order):
                    shifted = np.zeros(T, np.uint32)
                    if j < T:
                        shifted[j:] = u[:T - j]
                    acc = (acc ^ shifted) * np.uint32(EG_PRIME)
                acc = acc ^ (acc >> np.uint32(15))
                idx[:, col] = acc % np.uint32(slots)
                ok[:, col] = (np.arange(T) >= order - 1).astype(np.float32)
                col += 1

        out = []
        for s in range(h["num_sites"]):
            tables = self.t(base + s * 4 + 0).reshape(tables_n, slots, sub)
            key_proj = self.t(base + s * 4 + 1)
            value_proj = self.t(base + s * 4 + 2)
            taps = self.t(base + s * 4 + 3)

            fetched = tables[np.arange(tables_n)[None, :], idx] * ok[:, :, None]
            e = fetched.reshape(T, tables_n * sub)
            k = e @ key_proj.T
            v = e @ value_proj.T

            acc = np.zeros_like(v)
            for j in range(taps_n):
                shift = j * dil
                tap_ok = (np.arange(T) >= shift).astype(np.float32)
                sv = np.zeros_like(v)
                if shift < T:
                    sv[shift:] = v[:T - shift]
                acc += taps[j] * sv * tap_ok[:, None]
            out.append((k, acc))
        return out

    def attention(self, i, x, T):
        h = self.h
        nh, nkv, hd = h["num_heads"], h["num_kv_heads"], h["head_dim"]
        q = x @ self.layer(i, 1).T
        k = x @ self.layer(i, 2).T
        v = x @ self.layer(i, 3).T

        q = q.reshape(T, nh, hd).transpose(1, 0, 2)
        k = k.reshape(T, nkv, hd).transpose(1, 0, 2)
        v = v.reshape(T, nkv, hd).transpose(1, 0, 2)

        q = zcrms(q, self.layer(i, 4))
        k = zcrms(k, self.layer(i, 5))

        # Half-split rotary.
        half = hd // 2
        inv = 1.0 / (h["rope_theta"] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
        ang = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
        cos, sin = np.cos(ang), np.sin(ang)

        def rope(z):
            z1, z2 = z[..., :half], z[..., half:]
            return np.concatenate([z1 * cos - z2 * sin, z2 * cos + z1 * sin], -1)

        q, k = rope(q), rope(k)

        if self.int8_kv:
            # Match the C engine's cache: symmetric int8, one scale per
            # (position, kv head), applied after rope.
            def q8(z):
                s = np.abs(z).max(-1, keepdims=True) / 127.0
                s = np.where(s > 0, s, 1.0)
                return np.clip(np.rint(z / s), -127, 127) * s
            k, v = q8(k), q8(v)

        rep = nh // nkv
        k = np.repeat(k, rep, axis=0)
        v = np.repeat(v, rep, axis=0)

        scale = np.sqrt(np.float32(hd))
        scores = (q @ k.transpose(0, 2, 1)) / scale
        window = h["kv_window"] or T
        pos = np.arange(T)
        allowed = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < window)
        scores = np.where(allowed[None], scores, -np.inf)
        w = np.exp(scores - scores.max(-1, keepdims=True))
        w /= w.sum(-1, keepdims=True)
        out = (w @ v).transpose(1, 0, 2).reshape(T, nh * hd)

        out = out * sigmoid(x @ self.layer(i, 6).T)
        return out @ self.layer(i, 7).T

    def block(self, i, u, T, eg):
        h = self.h
        d = h["d_model"]
        for s in range(h["num_sites"]):
            if h["sites"][s] != i:
                continue
            ek, ev = eg[s]
            alpha = sigmoid((rms_unit(u) * rms_unit(ek)).sum(-1) / np.sqrt(d))
            u = u + alpha[:, None] * ev

        skip = u
        t = zcrms(u, self.layer(i, 0))
        t = self.attention(i, t, T)
        t = zcrms(t, self.layer(i, 8))
        u = skip + sigmoid(self.layer(i, 9)[0]) * t

        skip = u
        t = zcrms(u, self.layer(i, 10))
        # Hadamard MLP
        n = 1 << (d - 1).bit_length()
        H = hadamard(n)
        d1, d2, d3 = self.layer(i, 11), self.layer(i, 12), self.layer(i, 13)
        z = np.pad(t, ((0, 0), (0, n - d))) if n > d else t
        z = (d1 * z) @ H
        z = d2 * z
        z = z * sigmoid(z)
        z = z @ H
        z = (d3 * z)[:, :d]
        return skip + z

    def forward(self, tokens):
        h = self.h
        T = len(tokens)
        d = h["d_model"]
        n = h["mhc_lanes"]
        L = h["num_layers"]
        base = 1 + L * 14

        emb = self.t(0)
        x = emb[list(tokens)] * np.sqrt(np.float32(d))
        eg = self.engram_kv(tokens)

        a_pre, a_post, a_res = self.t(base), self.t(base + 1), self.t(base + 2)
        b_pre, b_post, b_res = self.t(base + 3), self.t(base + 4), self.t(base + 5)
        phi_pre = self.t(base + 6).reshape(L, n, n * d).transpose(0, 2, 1)
        phi_post = self.t(base + 7).reshape(L, n, n * d).transpose(0, 2, 1)
        phi_res = self.t(base + 8).reshape(L, n * n, n * d).transpose(0, 2, 1)

        lane = np.eye(n, dtype=np.float32)[np.arange(L) % n]
        pre_off = 8 * lane - 4
        post_off = -4 * (1 - lane)

        xl = np.broadcast_to(x[:, None, :], (T, n, d)).copy()

        for i in range(L):
            nx = rms_unit(xl.reshape(T, n * d))
            hpre = sigmoid(a_pre[i] * (nx @ phi_pre[i]) + b_pre[i] + pre_off[i])
            u = np.einsum("tn,tnc->tc", hpre, xl)
            y = self.block(i, u, T, eg) - u
            hpost = 2 * sigmoid(a_post[i] * (nx @ phi_post[i]) + b_post[i] + post_off[i])
            res = nx @ phi_res[i]
            hres = sinkhorn(a_res[i] * res.reshape(T, n, n) + b_res[i])
            xl = (np.einsum("tij,tjc->tic", hres, xl)
                  + hpost[:, :, None] * y[:, None, :])

        x = xl.mean(1)
        x = zcrms(x, self.t(base + 9 + h["num_sites"] * 4))
        return x @ emb.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blob")
    ap.add_argument("--tokens", type=int, nargs="+", default=[2, 100, 200, 300])
    ap.add_argument("--out", default=None)
    ap.add_argument("--int8-kv", action="store_true")
    args = ap.parse_args()

    ref = Ref(args.blob, int8_kv=args.int8_kv)
    logits = ref.forward(args.tokens)
    print(f"tokens: {args.tokens}", file=sys.stderr)
    for t in range(len(args.tokens)):
        top = np.argsort(-logits[t])[:5]
        print(f"pos {t}: argmax={top[0]} top5={top.tolist()} "
              f"max={logits[t].max():.4f}", file=sys.stderr)
    if args.out:
        np.save(args.out, logits)
    else:
        for t in range(len(args.tokens)):
            for v in logits[t]:
                print("%.6f" % v)


if __name__ == "__main__":
    sys.exit(main())
