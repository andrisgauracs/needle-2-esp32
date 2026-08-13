#include "nd_quant.h"

#include <math.h>
#include <string.h>

float nd_f16_slow(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (man == 0) {
            bits = sign; /* +-0 */
        } else {
            /* Subnormal: renormalise into a float32 exponent. */
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) {
                man <<= 1;
                exp--;
            }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13); /* inf / nan */
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }

    {
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
}

ND_HOT void nd_fwht(float *x, uint32_t n)
{
    uint32_t len;

    for (len = 1; len < n; len <<= 1) {
        uint32_t i;
        for (i = 0; i < n; i += len << 1) {
            uint32_t j;
            for (j = i; j < i + len; j++) {
                float a = x[j];
                float b = x[j + len];
                x[j]       = a + b;
                x[j + len] = a - b;
            }
        }
    }
}

uint32_t nd_cq_scratch(const nd_tensor *t)
{
    return nd_cq_in_pad(t);
}

void nd_cq_prepare(const nd_tensor *t, const float *x, float *xh)
{
    uint32_t in_pad = nd_cq_in_pad(t);
    uint32_t g      = t->group;
    uint32_t ngroup = in_pad / g;
    float    scale  = 1.0f / sqrtf((float)g);
    uint32_t gi;

    memcpy(xh, x, t->shape[1] * sizeof(float));
    if (in_pad > t->shape[1])
        memset(xh + t->shape[1], 0, (in_pad - t->shape[1]) * sizeof(float));

    for (gi = 0; gi < ngroup; gi++) {
        float   *blk = xh + (size_t)gi * g;
        uint32_t j;
        nd_fwht(blk, g);
        for (j = 0; j < g; j++)
            blk[j] *= scale;
    }
}

typedef struct {
    const uint8_t  *packed;
    const uint16_t *norms;
    const float    *xh, *cb;
    float          *y;
    uint32_t        ngroup, g, bits, rowbytes, base;
} gemv_ctx;

/* Sum over one group of cb[idx] * xh[j], for the common packings.
 * Indices are an LSB-first bitstream per row: index k occupies bits
 * [k*bits, (k+1)*bits). */
static ND_HOT float dot_group(const uint8_t *p, uint32_t bit_off, uint32_t bits,
                       uint32_t g, const float *cb, const float *xh)
{
    float    s = 0.0f;
    uint32_t j;

    /* Four independent accumulators: a single running sum serialises on the
     * FPU's add latency, which dominates this loop on an LX7. */
    if (bits == 2 && (bit_off & 7u) == 0) {
        const uint8_t *q = p + (bit_off >> 3);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (j = 0; j < g; j += 8) {
            uint8_t b0 = q[0], b1 = q[1];
            q += 2;
            s0 += cb[b0 & 3u]        * xh[j];
            s1 += cb[(b0 >> 2) & 3u] * xh[j + 1];
            s2 += cb[(b0 >> 4) & 3u] * xh[j + 2];
            s3 += cb[(b0 >> 6) & 3u] * xh[j + 3];
            s0 += cb[b1 & 3u]        * xh[j + 4];
            s1 += cb[(b1 >> 2) & 3u] * xh[j + 5];
            s2 += cb[(b1 >> 4) & 3u] * xh[j + 6];
            s3 += cb[(b1 >> 6) & 3u] * xh[j + 7];
        }
        return (s0 + s1) + (s2 + s3);
    }
    if (bits == 4 && (bit_off & 7u) == 0) {
        const uint8_t *q = p + (bit_off >> 3);
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (j = 0; j < g; j += 4) {
            uint8_t b0 = q[0], b1 = q[1];
            q += 2;
            s0 += cb[b0 & 15u] * xh[j];
            s1 += cb[b0 >> 4]  * xh[j + 1];
            s2 += cb[b1 & 15u] * xh[j + 2];
            s3 += cb[b1 >> 4]  * xh[j + 3];
        }
        return (s0 + s1) + (s2 + s3);
    }

    /* Generic bit reader (covers bits == 3, and any unaligned start). Only
     * fetches the second byte when the field actually straddles, so the last
     * index of the last row never reads past the blob. */
    for (j = 0; j < g; j++) {
        uint32_t b   = bit_off + j * bits;
        uint32_t byt = b >> 3;
        uint32_t sh  = b & 7u;
        uint32_t w   = p[byt];
        if (sh + bits > 8)
            w |= (uint32_t)p[byt + 1] << 8;
        s += cb[(w >> sh) & ((1u << bits) - 1u)] * xh[j];
    }
    return s;
}

/* Row-range variant. `base` shifts the tensor row that output i maps to, so
 * the mHC phi tensors (all 27 layers stacked into one tensor) can hand out a
 * per-layer slice and still be split across cores. */
static ND_HOT void gemv_rows_offset(void *vc, uint32_t i0, uint32_t i1)
{
    const gemv_ctx *c = (const gemv_ctx *)vc;
    uint32_t        i;

    for (i = i0; i < i1; i++) {
        uint32_t        r   = c->base + i;
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * c->g * c->bits, c->bits, c->g, c->cb,
                             c->xh + (size_t)gi * c->g);
        c->y[i] = acc;
    }
}

ND_HOT void nd_cq_gemv_rows(const nd_cact *c, const nd_tensor *t, const void *blob,
                     const float *xh, uint32_t r0, uint32_t nrows, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    gemv_ctx       ctx;

    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.xh       = xh;
    ctx.cb       = nd_cact_codebook(c, t->bits);
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = t->group;
    ctx.bits     = t->bits;
    ctx.rowbytes = rowbytes;
    ctx.base     = r0;

    nd_parallel_rows(gemv_rows_offset, &ctx, nrows);
}



static ND_HOT void gemv_rows_generic(void *vc, uint32_t r0, uint32_t r1)
{
    const gemv_ctx *c = (const gemv_ctx *)vc;
    uint32_t        r;

    for (r = r0; r < r1; r++) {
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * c->g * c->bits, c->bits, c->g, c->cb,
                             c->xh + (size_t)gi * c->g);
        c->y[r] = acc;
    }
}

ND_HOT void nd_cq_gemv_prepared(const nd_cact *c, const nd_tensor *t, const void *blob,
                         const float *xh, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    gemv_ctx       ctx;

    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.xh       = xh;
    ctx.cb       = nd_cact_codebook(c, t->bits);
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = t->group;
    ctx.bits     = t->bits;
    ctx.rowbytes = rowbytes;

    nd_parallel_rows(gemv_rows_generic, &ctx, out);
}

ND_HOT void nd_cq_lut_build(const nd_cact *c, const float *xh, uint32_t in_pad,
                            float *lut)
{
    const float *cb = nd_cact_codebook(c, 2);
    uint32_t     p;

    /* T[i0 | (i1 << 2)] = cb[i0]*xh[2p] + cb[i1]*xh[2p+1], matching the
     * LSB-first packing (the low 2 bits of a nibble are the earlier weight). */
    for (p = 0; p < in_pad / 2; p++) {
        float  x0 = xh[2 * p];
        float  x1 = xh[2 * p + 1];
        float  a0 = cb[0] * x0, a1 = cb[1] * x0, a2 = cb[2] * x0, a3 = cb[3] * x0;
        float  b0 = cb[0] * x1, b1 = cb[1] * x1, b2 = cb[2] * x1, b3 = cb[3] * x1;
        float *T  = lut + (size_t)p * 16;

        T[0]  = a0 + b0; T[1]  = a1 + b0; T[2]  = a2 + b0; T[3]  = a3 + b0;
        T[4]  = a0 + b1; T[5]  = a1 + b1; T[6]  = a2 + b1; T[7]  = a3 + b1;
        T[8]  = a0 + b2; T[9]  = a1 + b2; T[10] = a2 + b2; T[11] = a3 + b2;
        T[12] = a0 + b3; T[13] = a1 + b3; T[14] = a2 + b3; T[15] = a3 + b3;
    }
}

/* One group's contribution: 8 weights per iteration, four table lookups. */
static ND_HOT float dot_group_lut2(const uint8_t *q, const float *T, uint32_t g)
{
    float    s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    uint32_t j;

    for (j = 0; j < g; j += 8) {
        uint8_t b0 = q[0], b1 = q[1];
        q += 2;
        s0 += T[b0 & 15u];
        s1 += T[16 + (b0 >> 4)];
        s2 += T[32 + (b1 & 15u)];
        s3 += T[48 + (b1 >> 4)];
        T += 64;                      /* 4 pairs consumed */
    }
    return (s0 + s1) + (s2 + s3);
}

/* Default splitter: run everything on the calling thread. */
static void rows_serial(nd_row_fn fn, void *ctx, uint32_t nrows)
{
    fn(ctx, 0, nrows);
}

void (*nd_parallel_rows)(nd_row_fn fn, void *ctx, uint32_t nrows) = rows_serial;

typedef struct {
    const nd_tensor *t;
    const uint8_t   *packed;
    const uint16_t  *norms;
    const float     *lut;
    float           *y;
    uint32_t         ngroup, gbytes, gpairs, g, rowbytes;
} lut2_ctx;

static ND_HOT void lut2_rows(void *vc, uint32_t r0, uint32_t r1)
{
    const lut2_ctx *c = (const lut2_ctx *)vc;
    uint32_t        r;

    for (r = r0; r < r1; r++) {
        const uint8_t  *row = c->packed + (size_t)r * c->rowbytes;
        const uint16_t *nrm = c->norms + (size_t)r * c->ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < c->ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group_lut2(row + (size_t)gi * c->gbytes,
                                  c->lut + (size_t)gi * c->gpairs * 16, c->g);
        c->y[r] = acc;
    }
}

ND_HOT void nd_cq_gemv_lut2(const nd_tensor *t, const void *blob,
                            const float *lut, float *y)
{
    const uint8_t *packed   = (const uint8_t *)blob;
    uint32_t       out      = t->shape[0];
    uint32_t       g        = t->group;
    uint32_t       rowbytes = nd_cq_row_bytes(t);
    lut2_ctx       ctx;

    ctx.t        = t;
    ctx.packed   = packed;
    ctx.norms    = (const uint16_t *)(const void *)(packed +
                       (size_t)out * rowbytes);
    ctx.lut      = lut;
    ctx.y        = y;
    ctx.ngroup   = nd_cq_groups(t);
    ctx.g        = g;
    ctx.gbytes   = g / 4;              /* 2 bits per weight */
    ctx.gpairs   = g / 2;
    ctx.rowbytes = rowbytes;

    nd_parallel_rows(lut2_rows, &ctx, out);
}

ND_HOT void nd_cq_gemv_gather(const nd_cact *c, const nd_tensor *t, const void *blob,
                              const float *xh, const uint32_t *ids, uint32_t n,
                              float *y)
{
    const uint8_t  *packed   = (const uint8_t *)blob;
    uint32_t        out      = t->shape[0];
    uint32_t        g        = t->group;
    uint32_t        ngroup   = nd_cq_groups(t);
    uint32_t        rowbytes = nd_cq_row_bytes(t);
    const uint16_t *norms    = (const uint16_t *)(const void *)(packed +
                                (size_t)out * rowbytes);
    const float    *cb       = nd_cact_codebook(c, t->bits);
    uint32_t        i;

    for (i = 0; i < n; i++) {
        const uint8_t  *row = packed + (size_t)ids[i] * rowbytes;
        const uint16_t *nrm = norms + (size_t)ids[i] * ngroup;
        float           acc = 0.0f;
        uint32_t        gi;

        for (gi = 0; gi < ngroup; gi++)
            acc += nd_f16(nrm[gi]) *
                   dot_group(row, gi * g * t->bits, t->bits, g, cb,
                             xh + (size_t)gi * g);
        y[i] = acc;
    }
}

void nd_cq_gemv(const nd_cact *c, const nd_tensor *t, const void *blob,
                const float *x, float *scratch, float *y)
{
    nd_cq_prepare(t, x, scratch);
    nd_cq_gemv_prepared(c, t, blob, scratch, y);
}

void nd_cq_dequant_row(const nd_cact *c, const nd_tensor *t, const void *blob,
                       uint32_t row, float *scratch, float *w)
{
    const uint8_t  *packed   = (const uint8_t *)blob;
    uint32_t        out      = t->shape[0];
    uint32_t        g        = t->group;
    uint32_t        ngroup   = nd_cq_groups(t);
    uint32_t        rowbytes = nd_cq_row_bytes(t);
    const uint16_t *norms    = (const uint16_t *)(const void *)(packed +
                                (size_t)out * rowbytes);
    const float    *cb       = nd_cact_codebook(c, t->bits);
    const uint8_t  *p        = packed + (size_t)row * rowbytes;
    const uint16_t *nrm      = norms + (size_t)row * ngroup;
    float           scale    = 1.0f / sqrtf((float)g);
    uint32_t        gi;

    for (gi = 0; gi < ngroup; gi++) {
        float   *blk    = scratch + (size_t)gi * g;
        float    norm   = nd_f16(nrm[gi]);
        uint32_t bit_off = gi * g * t->bits;
        uint32_t j;

        for (j = 0; j < g; j++) {
            uint32_t b   = bit_off + j * t->bits;
            uint32_t byt = b >> 3;
            uint32_t sh  = b & 7u;
            uint32_t v   = p[byt];
            if (sh + t->bits > 8)
                v |= (uint32_t)p[byt + 1] << 8;
            blk[j] = cb[(v >> sh) & ((1u << t->bits) - 1u)] * norm;
        }
        nd_fwht(blk, g);
        for (j = 0; j < g; j++)
            blk[j] *= scale;
    }
    memcpy(w, scratch, t->shape[1] * sizeof(float));
}
