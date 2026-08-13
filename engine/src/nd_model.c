#include "nd_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "nd_quant.h"

uint64_t nd_prof[ND_P_COUNT];

#ifdef ND_PROFILE
#ifdef ESP_PLATFORM
#include "esp_timer.h"
#define ND_NOW_US() ((uint64_t)esp_timer_get_time())
#else
#include <time.h>
#define ND_NOW_US() ((uint64_t)(clock() * (1000000.0 / CLOCKS_PER_SEC)))
#endif
#define ND_T0(v) uint64_t v = ND_NOW_US()
#define ND_T1(v, slot) nd_prof[slot] += ND_NOW_US() - (v)
#else
#define ND_T0(v) ((void)0)
#define ND_T1(v, slot) ((void)0)
#endif

#define ND_EPS      1e-6f
#define ND_EG_HIST  16      /* >= (taps-1)*dilation + 1 = 10, power of two */
#define ND_SINKHORN 20

/* Engram hash constants (architecture.py). */
#define ND_EG_SEED  0x9E3779B9u
#define ND_EG_PRIME 0x01000193u

/* ------------------------------------------------------------------ utils */

static float sigmoidf_(float x)
{
    if (x >= 0.0f) {
        float e = nd_expf(-x);
        return 1.0f / (1.0f + e);
    }
    {
        float e = nd_expf(x);
        return e / (1.0f + e);
    }
}

/* FP16 tensors are read element-wise; they are small (norm scales, gates). */
static float fp16_get(const nd_model *m, const nd_tensor *t, size_t i)
{
    const uint16_t *p = (const uint16_t *)nd_cact_data(&m->c, t);
    return nd_f16(p[i]);
}

/* x * rsqrt(mean(x^2) + eps) */
static void rms_unit(const float *x, uint32_t n, float *out)
{
    float    ss = 0.0f;
    uint32_t i;
    for (i = 0; i < n; i++)
        ss += x[i] * x[i];
    {
        float inv = 1.0f / sqrtf(ss / (float)n + ND_EPS);
        for (i = 0; i < n; i++)
            out[i] = x[i] * inv;
    }
}

/* ZCRMSNorm: (1 + scale) * x / sqrt(mean(x^2) + eps) */
static void zcrms(const nd_model *m, const nd_tensor *scale, const float *x,
                  uint32_t n, float *out)
{
    const uint16_t *s  = (const uint16_t *)nd_cact_data(&m->c, scale);
    float           ss = 0.0f;
    uint32_t        i;

    for (i = 0; i < n; i++)
        ss += x[i] * x[i];
    {
        float inv = 1.0f / sqrtf(ss / (float)n + ND_EPS);
        for (i = 0; i < n; i++)
            out[i] = (1.0f + nd_f16(s[i])) * x[i] * inv;
    }
}

/* In-place per-head ZCRMSNorm over head_dim, shared scale across heads. */
static void zcrms_heads(const nd_model *m, const nd_tensor *scale, float *x,
                        uint32_t nheads, uint32_t dim)
{
    const uint16_t *s = (const uint16_t *)nd_cact_data(&m->c, scale);
    uint32_t        h, i;

    for (h = 0; h < nheads; h++) {
        float *v  = x + (size_t)h * dim;
        float  ss = 0.0f;
        for (i = 0; i < dim; i++)
            ss += v[i] * v[i];
        {
            float inv = 1.0f / sqrtf(ss / (float)dim + ND_EPS);
            for (i = 0; i < dim; i++)
                v[i] = (1.0f + nd_f16(s[i])) * v[i] * inv;
        }
    }
}

/* GPT-NeoX style half-split rotary, against cos/sin precomputed once per
 * token (the position is the same for all 27 layers). */
static void apply_rope(const nd_model *m, float *x, uint32_t nheads, uint32_t dim)
{
    uint32_t half = dim / 2;
    uint32_t h, i;

    for (h = 0; h < nheads; h++) {
        float *v = x + (size_t)h * dim;
        for (i = 0; i < half; i++) {
            float c  = m->rope_cos[i];
            float s  = m->rope_sin[i];
            float x1 = v[i];
            float x2 = v[i + half];
            v[i]        = x1 * c - x2 * s;
            v[i + half] = x2 * c + x1 * s;
        }
    }
}

/* Doubly-stochastic normalisation of a lanes x lanes matrix, in log space. */
static void sinkhorn(float *a, uint32_t n)
{
    uint32_t it, i, j;

    for (it = 0; it < ND_SINKHORN; it++) {
        for (i = 0; i < n; i++) {           /* rows */
            float mx = a[i * n];
            float sum = 0.0f;
            for (j = 1; j < n; j++)
                if (a[i * n + j] > mx) mx = a[i * n + j];
            for (j = 0; j < n; j++)
                sum += nd_expf(a[i * n + j] - mx);
            {
                float lse = mx + logf(sum);
                for (j = 0; j < n; j++)
                    a[i * n + j] -= lse;
            }
        }
        for (j = 0; j < n; j++) {           /* columns */
            float mx = a[j];
            float sum = 0.0f;
            for (i = 1; i < n; i++)
                if (a[i * n + j] > mx) mx = a[i * n + j];
            for (i = 0; i < n; i++)
                sum += nd_expf(a[i * n + j] - mx);
            {
                float lse = mx + logf(sum);
                for (i = 0; i < n; i++)
                    a[i * n + j] -= lse;
            }
        }
    }
    for (i = 0; i < n * n; i++)
        a[i] = nd_expf(a[i]);
}

/* ------------------------------------------------------------------- open */

static void bind_layer(nd_model *m, uint32_t i)
{
    uint32_t   b = 1 + i * 14;
    nd_layer  *L = &m->layer[i];
    nd_tensor *slots[14] = {
        &L->norm_in, &L->q_proj, &L->k_proj, &L->v_proj, &L->q_norm,
        &L->k_norm, &L->gate_proj, &L->out_proj, &L->post_norm,
        &L->attn_gate, &L->pre_hada, &L->d1, &L->d2, &L->d3
    };
    uint32_t k;
    for (k = 0; k < 14; k++)
        nd_cact_tensor(&m->c, b + k, slots[k]);
}

int nd_model_open(nd_model *m, const void *blob, size_t size)
{
    uint32_t base, s, i;
    int      rc;

    memset(m, 0, sizeof(*m));
    rc = nd_cact_open(&m->c, blob, size);
    if (rc != 0)
        return rc;

    m->d_model    = m->c.h.d_model;
    m->n_layers   = m->c.h.num_layers;
    m->n_heads    = m->c.h.num_heads;
    m->n_kv_heads = m->c.h.num_kv_heads;
    m->head_dim   = m->c.h.head_dim;
    m->attn_dim   = m->c.h.attn_dim;
    m->kv_dim     = m->n_kv_heads * m->head_dim;
    m->lanes      = m->c.h.mhc_lanes;
    m->vocab      = m->c.h.vocab_size;
    m->window     = m->c.h.kv_window ? m->c.h.kv_window : m->c.h.max_seq_len;
    m->n_sites    = m->c.h.num_sites;

    if (m->lanes > ND_MAX_LANES || m->n_sites > ND_MAX_SITES)
        return -10;

    /* Canonical positional layout, verified against the shipped blob. */
    nd_cact_tensor(&m->c, 0, &m->embedding);

    m->layer = (nd_layer *)ND_ALLOC(sizeof(nd_layer) * m->n_layers);
    if (!m->layer)
        return -11;
    for (i = 0; i < m->n_layers; i++)
        bind_layer(m, i);

    base = 1 + m->n_layers * 14;
    nd_cact_tensor(&m->c, base + 0, &m->mhc_a_pre);
    nd_cact_tensor(&m->c, base + 1, &m->mhc_a_post);
    nd_cact_tensor(&m->c, base + 2, &m->mhc_a_res);
    nd_cact_tensor(&m->c, base + 3, &m->mhc_b_pre);
    nd_cact_tensor(&m->c, base + 4, &m->mhc_b_post);
    nd_cact_tensor(&m->c, base + 5, &m->mhc_b_res);
    nd_cact_tensor(&m->c, base + 6, &m->mhc_phi_pre);
    nd_cact_tensor(&m->c, base + 7, &m->mhc_phi_post);
    nd_cact_tensor(&m->c, base + 8, &m->mhc_phi_res);
    base += 9;

    for (s = 0; s < m->n_sites; s++) {
        nd_cact_tensor(&m->c, base + s * 4 + 0, &m->engram[s].tables);
        nd_cact_tensor(&m->c, base + s * 4 + 1, &m->engram[s].key_proj);
        nd_cact_tensor(&m->c, base + s * 4 + 2, &m->engram[s].value_proj);
        nd_cact_tensor(&m->c, base + s * 4 + 3, &m->engram[s].taps);
    }
    base += m->n_sites * 4;
    nd_cact_tensor(&m->c, base, &m->final_norm);

    /* Probe heads, if this blob carries them. Layout after final_norm:
     * a manifest of H head codes (1 contrastive, 2 confidence), then H fixed
     * triples [probes, proj, bias]; the tokenizer is the final tensor. */
    {
        uint32_t after = base + 1;
        uint32_t extra = (m->c.n > after + 1) ? m->c.n - after - 1 : 0;

        if (extra >= 4) {
            nd_tensor manifest;
            uint32_t  heads, k;

            nd_cact_tensor(&m->c, after, &manifest);
            heads = (extra - 1) / 3;
            for (k = 0; k < heads && k < manifest.shape[0]; k++) {
                float code = fp16_get(m, &manifest, k);
                if (code > 1.5f && code < 2.5f) {   /* confidence */
                    nd_cact_tensor(&m->c, after + 1 + k * 3 + 0, &m->conf_probes);
                    nd_cact_tensor(&m->c, after + 1 + k * 3 + 1, &m->conf_proj);
                    nd_cact_tensor(&m->c, after + 1 + k * 3 + 2, &m->conf_bias);
                    /* proj is (1, n_probes*d_model); probes is (P, d_model) */
                    if (m->conf_probes.shape[1] == m->d_model &&
                        m->conf_proj.shape[1] ==
                            m->conf_probes.shape[0] * m->d_model) {
                        m->n_probes = m->conf_probes.shape[0];
                        m->has_conf = 1;
                    }
                }
            }
        }
    }

    /* The tokenizer is the single RAW tensor, last in canon order. */
    for (i = m->c.n; i-- > 0;) {
        nd_tensor t;
        nd_cact_tensor(&m->c, i, &t);
        if (t.dtype == ND_DT_RAW) {
            if (nd_tok_init(&m->tok, nd_cact_data(&m->c, &t), (size_t)t.nbytes) == 0)
                m->tok_ready = 1;
            break;
        }
    }

    /* ---- buffers ---- */
    {
        uint32_t dm  = m->d_model;
        uint32_t nl  = m->lanes * dm;
        size_t   kvn = (size_t)m->n_layers * m->window * m->kv_dim;
        size_t   scn = (size_t)m->n_layers * m->window * m->n_kv_heads;

        m->lane      = (float *)ND_ALLOC_FAST(sizeof(float) * nl);
        m->lane_next = (float *)ND_ALLOC_FAST(sizeof(float) * nl);
        m->nx        = (float *)ND_ALLOC_FAST(sizeof(float) * nl);
        m->xh        = (float *)ND_ALLOC_FAST(sizeof(float) * (nl > dm ? nl : dm));
        m->lut       = (float *)ND_ALLOC_FAST(sizeof(float) * nd_cq_lut_floats(dm));

        m->u         = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->ublk      = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        uint32_t dpow = 1;
        while (dpow < dm)
            dpow <<= 1;

        m->n1        = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->n2        = (float *)ND_ALLOC_FAST(sizeof(float) * dpow);
        m->y         = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->tmp       = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->tmp2      = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->q         = (float *)ND_ALLOC_FAST(sizeof(float) * m->attn_dim);
        m->kbuf      = (float *)ND_ALLOC_FAST(sizeof(float) * m->kv_dim);
        m->vbuf      = (float *)ND_ALLOC_FAST(sizeof(float) * m->kv_dim);
        m->gate      = (float *)ND_ALLOC_FAST(sizeof(float) * m->attn_dim);
        m->attn      = (float *)ND_ALLOC_FAST(sizeof(float) * m->attn_dim);
        m->aout      = (float *)ND_ALLOC_FAST(sizeof(float) * dm);
        m->rope_inv  = (float *)ND_ALLOC_FAST(sizeof(float) * (m->head_dim / 2));
        m->rope_cos  = (float *)ND_ALLOC_FAST(sizeof(float) * (m->head_dim / 2));
        m->rope_sin  = (float *)ND_ALLOC_FAST(sizeof(float) * (m->head_dim / 2));
        m->eg_k      = (float *)ND_ALLOC(sizeof(float) * m->n_sites * dm);
        m->eg_v      = (float *)ND_ALLOC(sizeof(float) * m->n_sites * dm);
        m->eg_hist   = (float *)ND_ALLOC(sizeof(float) * m->n_sites * ND_EG_HIST * dm);
        m->logits    = (float *)ND_ALLOC(sizeof(float) * m->vocab);
        m->row       = (float *)ND_ALLOC_FAST(sizeof(float) * (dm > 128 ? dm : 128));
        m->k_cache   = (int8_t *)ND_ALLOC(kvn);
        m->v_cache   = (int8_t *)ND_ALLOC(kvn);
        m->k_scale   = (float *)ND_ALLOC(sizeof(float) * scn);
        m->v_scale   = (float *)ND_ALLOC(sizeof(float) * scn);

        if (m->has_conf) {
            m->probes_f = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes * dm);
            m->pool_acc = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes * dm);
            m->pool_max = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes);
            m->pool_sum = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes);
            if (!m->probes_f || !m->pool_acc || !m->pool_max || !m->pool_sum) {
                nd_model_close(m);
                return -13;
            }
        }

        if (!m->lane || !m->lane_next || !m->nx || !m->xh || !m->lut || !m->u || !m->ublk ||
            !m->n1 || !m->n2 || !m->y || !m->tmp || !m->tmp2 || !m->q ||
            !m->kbuf || !m->vbuf || !m->gate || !m->attn || !m->aout ||
            !m->rope_inv || !m->rope_cos || !m->rope_sin ||
            !m->eg_k || !m->eg_v || !m->eg_hist || !m->logits ||
            !m->row || !m->k_cache || !m->v_cache || !m->k_scale || !m->v_scale) {
            nd_model_close(m);
            return -12;
        }
    }

    /* Expand the confidence probes to float32 once; pool_cell would otherwise
     * convert 8*512 halves per hidden cell, ~114K conversions per token. */
    if (m->has_conf) {
        const uint16_t *pr = (const uint16_t *)nd_cact_data(&m->c, &m->conf_probes);
        uint32_t        k;
        for (k = 0; k < m->n_probes * m->d_model; k++)
            m->probes_f[k] = nd_f16(pr[k]);
    }

    /* Rotary inverse frequencies: 1/theta^(2i/head_dim). */
    for (i = 0; i < m->head_dim / 2; i++)
        m->rope_inv[i] = 1.0f / powf(m->c.h.rope_theta,
                                     (float)(2 * i) / (float)m->head_dim);

    nd_model_reset(m);
    return 0;
}

void nd_model_close(nd_model *m)
{
    if (!m)
        return;
    if (m->tok_ready)
        nd_tok_free(&m->tok);
    ND_FREE(m->layer);
    ND_FREE(m->lane); ND_FREE(m->lane_next); ND_FREE(m->nx); ND_FREE(m->xh);
    ND_FREE(m->u); ND_FREE(m->ublk); ND_FREE(m->n1); ND_FREE(m->n2);
    ND_FREE(m->y); ND_FREE(m->tmp); ND_FREE(m->tmp2);
    ND_FREE(m->q); ND_FREE(m->kbuf); ND_FREE(m->vbuf); ND_FREE(m->gate);
    ND_FREE(m->attn); ND_FREE(m->aout);
    ND_FREE(m->rope_inv); ND_FREE(m->rope_cos); ND_FREE(m->rope_sin);
    ND_FREE(m->eg_k); ND_FREE(m->eg_v); ND_FREE(m->eg_hist);
    ND_FREE(m->logits); ND_FREE(m->row);
    ND_FREE(m->k_cache); ND_FREE(m->v_cache);
    ND_FREE(m->k_scale); ND_FREE(m->v_scale);
    ND_FREE(m->snap_eg_hist);
    ND_FREE(m->snap_pool_acc); ND_FREE(m->snap_pool_max); ND_FREE(m->snap_pool_sum);
    ND_FREE(m->probes_f);
    ND_FREE(m->pool_acc); ND_FREE(m->pool_max); ND_FREE(m->pool_sum);
    memset(m, 0, sizeof(*m));
}

/* Cache slot for an absolute position. Sinks occupy the first n_sink slots
 * permanently; everything after rings through the remainder. */
static uint32_t kv_slot(const nd_model *m, uint32_t p)
{
    if (p < m->n_sink)
        return p;
    return m->n_sink + (p - m->n_sink) % (m->window - m->n_sink);
}

/* Recent-context slots kept free no matter how long the pinned prefix is. */
#define ND_MIN_RECENT 64

void nd_model_set_sink(nd_model *m, uint32_t n)
{
    /* The prefix must be pinned *entirely*: a partially pinned prefix leaves
     * its tail in the ring, where a long generation wraps around and silently
     * overwrites part of the tool schema. */
    uint32_t cap = (m->window > ND_MIN_RECENT) ? m->window - ND_MIN_RECENT : 0;
    m->n_sink = (n > cap) ? cap : n;
}

/* Fold one hidden cell into the running softmax pool.
 *
 * probe_pool() softmaxes probe·cell/sqrt(d) over every cell of every token,
 * then takes the weighted mean. Streaming it with a running max keeps the
 * result identical to the batch computation while holding only the
 * accumulator, which is what makes this affordable on the ESP32. */
static void pool_cell(nd_model *m, const float *cell)
{
    const float    *probes = m->probes_f;
    uint32_t        dm     = m->d_model;
    float           inv_s  = 1.0f / sqrtf((float)dm);
    uint32_t        k, i;

    for (k = 0; k < m->n_probes; k++) {
        const float    *pr = probes + (size_t)k * dm;
        float          *ac = m->pool_acc + (size_t)k * dm;
        float           z = 0.0f, w;

        for (i = 0; i < dm; i++)
            z += pr[i] * cell[i];
        z *= inv_s;

        if (z > m->pool_max[k]) {
            float rescale = expf(m->pool_max[k] - z);
            for (i = 0; i < dm; i++)
                ac[i] *= rescale;
            m->pool_sum[k] *= rescale;
            m->pool_max[k] = z;
        }
        w = expf(z - m->pool_max[k]);
        m->pool_sum[k] += w;
        for (i = 0; i < dm; i++)
            ac[i] += w * cell[i];
    }
}

float nd_model_confidence(nd_model *m)
{
    const uint16_t *proj;
    float           logit;
    uint32_t        k, i;

    if (!m->has_conf)
        return -1.0f;

    proj  = (const uint16_t *)nd_cact_data(&m->c, &m->conf_proj);
    logit = nd_f16(*(const uint16_t *)nd_cact_data(&m->c, &m->conf_bias));

    for (k = 0; k < m->n_probes; k++) {
        const float *ac  = m->pool_acc + (size_t)k * m->d_model;
        float        inv = (m->pool_sum[k] > 0.0f) ? 1.0f / m->pool_sum[k] : 0.0f;
        for (i = 0; i < m->d_model; i++)
            logit += nd_f16(proj[(size_t)k * m->d_model + i]) * ac[i] * inv;
    }
    return sigmoidf_(logit);
}

int nd_model_snapshot(nd_model *m)
{
    size_t egn = (size_t)m->n_sites * ND_EG_HIST * m->d_model;

    if (!m->snap_eg_hist) {
        m->snap_eg_hist = (float *)ND_ALLOC(sizeof(float) * egn);
        if (!m->snap_eg_hist)
            return -1;
        if (m->has_conf) {
            m->snap_pool_acc = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes * m->d_model);
            m->snap_pool_max = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes);
            m->snap_pool_sum = (float *)ND_ALLOC_FAST(sizeof(float) * m->n_probes);
            if (!m->snap_pool_acc || !m->snap_pool_max || !m->snap_pool_sum)
                return -1;
        }
    }

    nd_model_set_sink(m, m->pos);
    m->snap_pos    = m->pos;
    m->snap_eg_pos = m->eg_pos;
    memcpy(m->snap_hist, m->hist, sizeof(m->hist));
    memcpy(m->snap_eg_hist, m->eg_hist, sizeof(float) * egn);
    if (m->has_conf) {
        memcpy(m->snap_pool_acc, m->pool_acc,
               sizeof(float) * m->n_probes * m->d_model);
        memcpy(m->snap_pool_max, m->pool_max, sizeof(float) * m->n_probes);
        memcpy(m->snap_pool_sum, m->pool_sum, sizeof(float) * m->n_probes);
    }
    m->has_snap = 1;
    return 0;
}

void nd_model_rewind(nd_model *m)
{
    size_t egn = (size_t)m->n_sites * ND_EG_HIST * m->d_model;

    if (!m->has_snap) {
        nd_model_reset(m);
        return;
    }
    m->pos    = m->snap_pos;
    m->eg_pos = m->snap_eg_pos;
    memcpy(m->hist, m->snap_hist, sizeof(m->hist));
    memcpy(m->eg_hist, m->snap_eg_hist, sizeof(float) * egn);
    if (m->has_conf) {
        memcpy(m->pool_acc, m->snap_pool_acc,
               sizeof(float) * m->n_probes * m->d_model);
        memcpy(m->pool_max, m->snap_pool_max, sizeof(float) * m->n_probes);
        memcpy(m->pool_sum, m->snap_pool_sum, sizeof(float) * m->n_probes);
    }
}

void nd_model_reset(nd_model *m)
{
    m->pos    = 0;
    m->n_sink = 0;
    m->has_snap = 0;
    m->eg_pos = 0;
    if (m->has_conf) {
        uint32_t k;
        memset(m->pool_acc, 0, sizeof(float) * m->n_probes * m->d_model);
        for (k = 0; k < m->n_probes; k++) {
            m->pool_max[k] = -INFINITY;
            m->pool_sum[k] = 0.0f;
        }
    }
    memset(m->hist, 0, sizeof(m->hist));
    if (m->eg_hist)
        memset(m->eg_hist, 0,
               sizeof(float) * m->n_sites * ND_EG_HIST * m->d_model);
}

/* ----------------------------------------------------------------- engram */

/* k/v for the current token at every engram site. */
static void engram_step(nd_model *m, uint32_t token)
{
    uint32_t orders = m->c.h.num_orders;
    uint32_t heads  = m->c.h.engram_tables / (orders ? orders : 1);
    uint32_t slots  = m->c.h.engram_slots;
    uint32_t sub    = m->c.h.engram_sub_dim;
    uint32_t dil    = m->c.h.engram_dilation;
    uint32_t taps   = m->c.h.engram_conv_taps;
    uint32_t dm     = m->d_model;
    uint32_t s, oi, h, j;

    /* Shift the token history: hist[0] is the current token. */
    for (j = 7; j > 0; j--)
        m->hist[j] = m->hist[j - 1];
    m->hist[0] = token;

    for (s = 0; s < m->n_sites; s++) {
        float   *e = m->xh; /* reused: engram e is d_model long, xh is >= that */
        uint32_t table = 0;

        for (oi = 0; oi < orders; oi++) {
            uint32_t order = m->c.h.orders[oi];
            for (h = 0; h < heads; h++, table++) {
                uint32_t seed = ND_EG_SEED * (oi * heads + h + 1);
                uint32_t acc  = seed;
                uint32_t idx;
                int      ok   = (m->pos + 1 >= order); /* enough history */

                for (j = 0; j < order; j++) {
                    uint32_t tk = (j <= m->pos) ? m->hist[j] : 0u;
                    acc = (acc ^ tk) * ND_EG_PRIME;
                }
                acc ^= acc >> 15;
                idx = acc % slots;

                if (ok) {
                    nd_tensor *tt = &m->engram[s].tables;
                    nd_cq_dequant_row(&m->c, tt, nd_cact_data(&m->c, tt),
                                      table * slots + idx, m->row,
                                      e + (size_t)table * sub);
                } else {
                    memset(e + (size_t)table * sub, 0, sizeof(float) * sub);
                }
            }
        }

        /* k = key_proj @ e, raw v = value_proj @ e */
        {
            nd_tensor *kp = &m->engram[s].key_proj;
            nd_tensor *vp = &m->engram[s].value_proj;
            float     *vraw = m->eg_hist + ((size_t)s * ND_EG_HIST +
                                            (m->eg_pos % ND_EG_HIST)) * dm;
            /* e currently lives in xh; prepare needs its own output, so use
             * tmp2 as the transformed activation buffer. */
            nd_cq_prepare(kp, e, m->tmp2);
            nd_cq_lut_build(&m->c, m->tmp2, nd_cq_in_pad(kp), m->lut);
            nd_cq_gemv_lut2(kp, nd_cact_data(&m->c, kp), m->lut,
                            m->eg_k + (size_t)s * dm);
            nd_cq_gemv_lut2(vp, nd_cact_data(&m->c, vp), m->lut, vraw);
        }

        /* Dilated causal tap convolution over the raw v history. */
        {
            const uint16_t *tp = (const uint16_t *)nd_cact_data(&m->c,
                                                               &m->engram[s].taps);
            float *out = m->eg_v + (size_t)s * dm;
            uint32_t d;

            memset(out, 0, sizeof(float) * dm);
            for (j = 0; j < taps; j++) {
                uint32_t back = j * dil;
                const float *src;
                if (back > m->pos)
                    continue;  /* tap_ok */
                src = m->eg_hist + ((size_t)s * ND_EG_HIST +
                                    ((m->eg_pos - back) % ND_EG_HIST)) * dm;
                for (d = 0; d < dm; d++)
                    out[d] += nd_f16(tp[j * dm + d]) * src[d];
            }
        }
    }
    m->eg_pos++;
}

/* ------------------------------------------------------------- attention */

/* Online softmax over one range of heads.
 *
 * A two-pass max-then-accumulate computes every q.k twice, and with a pinned
 * 158-token prefix those dot products dominate the whole forward pass. Keeping
 * a running max and rescaling the accumulator gives the identical result from
 * a single pass over K and V. Heads are independent, so the range also splits
 * across cores exactly like GEMV rows. */
typedef struct {
    nd_model *m;
    uint32_t  li, nkv, rep, hd, sinks, rfirst, rcount;
    float     scale;
} attn_ctx;

static ND_HOT void attn_heads(void *vc, uint32_t h0, uint32_t h1)
{
    const attn_ctx *c   = (const attn_ctx *)vc;
    nd_model       *m   = c->m;
    uint32_t        hd  = c->hd;
    uint32_t        nkv = c->nkv;
    uint32_t        li  = c->li;
    uint32_t        h, i;

    for (h = h0; h < h1; h++) {
        const float *qh  = m->q + (size_t)h * hd;
        float       *oh  = m->attn + (size_t)h * hd;
        uint32_t     kvh = h / c->rep;
        float        mx = -INFINITY, denom = 0.0f;
        uint32_t     run, p;

        memset(oh, 0, sizeof(float) * hd);

        for (run = 0; run < 2; run++) {
            uint32_t base  = run ? c->rfirst : 0;
            uint32_t count = run ? c->rcount : c->sinks;

            for (p = 0; p < count; p++) {
                uint32_t      sl = kv_slot(m, base + p);
                size_t        kb = ((size_t)li * m->window + sl) * m->kv_dim
                                   + (size_t)kvh * hd;
                size_t        sb = ((size_t)li * m->window + sl) * nkv + kvh;
                const int8_t *kp = m->k_cache + kb;
                const int8_t *vp = m->v_cache + kb;
                float         dot = 0.0f, w;

                for (i = 0; i < hd; i++)
                    dot += qh[i] * (float)kp[i];
                dot *= m->k_scale[sb] * c->scale;

                if (dot > mx) {
                    if (denom > 0.0f) {
                        float rescale = nd_expf(mx - dot);
                        for (i = 0; i < hd; i++)
                            oh[i] *= rescale;
                        denom *= rescale;
                    }
                    mx = dot;
                }
                w = nd_expf(dot - mx);
                denom += w;
                {
                    float wv = w * m->v_scale[sb];
                    for (i = 0; i < hd; i++)
                        oh[i] += wv * (float)vp[i];
                }
            }
        }
        {
            float inv = 1.0f / denom;
            for (i = 0; i < hd; i++)
                oh[i] *= inv;
        }
    }
}

static void attention(nd_model *m, uint32_t li, const float *xin, float *out)
{
    const nd_layer *L    = &m->layer[li];
    uint32_t        hd   = m->head_dim;
    uint32_t        nh   = m->n_heads;
    uint32_t        nkv  = m->n_kv_heads;
    uint32_t        slot = kv_slot(m, m->pos);
    size_t          kvbase = ((size_t)li * m->window + slot) * m->kv_dim;
    size_t          scbase = ((size_t)li * m->window + slot) * nkv;
    uint32_t        i, kh;

    /* q, k, v and the gate all reduce over the same activation, so one
     * prepare and one pair table serve all four. */
    /* Pair table, not the quad table: the quad kernel needs group-outer
     * ordering, which reads 32 bytes out of every 64-byte cache line and
     * measured 38% SLOWER on device despite issuing fewer instructions. */
    { ND_T0(tp);
      nd_cq_prepare(&L->q_proj, xin, m->xh);
      nd_cq_lut_build(&m->c, m->xh, nd_cq_in_pad(&L->q_proj), m->lut);
      ND_T1(tp, ND_P_PREP); }
    { ND_T0(tg);
      nd_cq_gemv_lut2(&L->q_proj,    nd_cact_data(&m->c, &L->q_proj),    m->lut, m->q);
      nd_cq_gemv_lut2(&L->k_proj,    nd_cact_data(&m->c, &L->k_proj),    m->lut, m->kbuf);
      nd_cq_gemv_lut2(&L->v_proj,    nd_cact_data(&m->c, &L->v_proj),    m->lut, m->vbuf);
      nd_cq_gemv_lut2(&L->gate_proj, nd_cact_data(&m->c, &L->gate_proj), m->lut, m->gate);
      ND_T1(tg, ND_P_PROJ); }

    zcrms_heads(m, &L->q_norm, m->q, nh, hd);
    zcrms_heads(m, &L->k_norm, m->kbuf, nkv, hd);

    apply_rope(m, m->q, nh, hd);
    apply_rope(m, m->kbuf, nkv, hd);

    /* Store this position's k/v as symmetric int8, one scale per head. */
    for (kh = 0; kh < nkv; kh++) {
        float mk = 0.0f, mv = 0.0f;
        for (i = 0; i < hd; i++) {
            float a = fabsf(m->kbuf[kh * hd + i]);
            float b = fabsf(m->vbuf[kh * hd + i]);
            if (a > mk) mk = a;
            if (b > mv) mv = b;
        }
        {
            float ks = (mk > 0.0f) ? mk / 127.0f : 1.0f;
            float vs = (mv > 0.0f) ? mv / 127.0f : 1.0f;
            m->k_scale[scbase + kh] = ks;
            m->v_scale[scbase + kh] = vs;
            for (i = 0; i < hd; i++) {
                float kq = m->kbuf[kh * hd + i] / ks;
                float vq = m->vbuf[kh * hd + i] / vs;
                if (kq > 127.0f)  kq = 127.0f;
                if (kq < -127.0f) kq = -127.0f;
                if (vq > 127.0f)  vq = 127.0f;
                if (vq < -127.0f) vq = -127.0f;
                m->k_cache[kvbase + kh * hd + i] = (int8_t)lrintf(kq);
                m->v_cache[kvbase + kh * hd + i] = (int8_t)lrintf(vq);
            }
        }
    }

    /* Attend to the pinned sinks [0, n_sink) plus the most recent
     * (window - n_sink) positions. With n_sink == 0 this is a plain sliding
     * window; the two runs are visited without materialising a score buffer. */
    { ND_T0(ta);
    {
        attn_ctx actx;
        uint32_t rcap = m->window - m->n_sink;

        actx.m      = m;
        actx.li     = li;
        actx.nkv    = nkv;
        actx.rep    = nh / nkv;
        actx.hd     = hd;
        actx.scale  = 1.0f / sqrtf((float)hd);
        actx.sinks  = (m->n_sink < m->pos + 1) ? m->n_sink : m->pos + 1;

        if (m->pos + 1 <= m->n_sink) {
            actx.rfirst = m->pos + 1;
            actx.rcount = 0;
        } else {
            actx.rcount = m->pos + 1 - m->n_sink;
            if (actx.rcount > rcap)
                actx.rcount = rcap;
            actx.rfirst = m->pos + 1 - actx.rcount;
        }
        nd_parallel_rows(attn_heads, &actx, nh);
    }
    ND_T1(ta, ND_P_ATTN); }

    /* Gate, then project back to d_model. */
    for (i = 0; i < m->attn_dim; i++)
        m->attn[i] *= sigmoidf_(m->gate[i]);

    { ND_T0(tp2);
      nd_cq_prepare(&L->out_proj, m->attn, m->xh);
      nd_cq_lut_build(&m->c, m->xh, nd_cq_in_pad(&L->out_proj), m->lut);
      ND_T1(tp2, ND_P_PREP); }
    { ND_T0(tg2);
      nd_cq_gemv_lut2(&L->out_proj, nd_cact_data(&m->c, &L->out_proj), m->lut, out);
      ND_T1(tg2, ND_P_PROJ); }
}

/* -------------------------------------------------------- Hadamard MLP */

static void hadamard_mlp(nd_model *m, uint32_t li, const float *x, float *out)
{
    const nd_layer *L  = &m->layer[li];
    uint32_t        dm = m->d_model;
    uint32_t        n  = 1;
    const uint16_t *d1, *d2, *d3;
    float           inv;
    uint32_t        i;

    while (n < dm)
        n <<= 1;
    inv = 1.0f / sqrtf((float)n);

    d1 = (const uint16_t *)nd_cact_data(&m->c, &L->d1);
    d2 = (const uint16_t *)nd_cact_data(&m->c, &L->d2);
    d3 = (const uint16_t *)nd_cact_data(&m->c, &L->d3);

    for (i = 0; i < dm; i++)
        out[i] = nd_f16(d1[i]) * x[i];
    for (i = dm; i < n; i++)
        out[i] = 0.0f;

    nd_fwht(out, n);
    for (i = 0; i < n; i++) {
        float z = out[i] * inv * nd_f16(d2[i]);
        out[i] = z * sigmoidf_(z);            /* silu */
    }
    nd_fwht(out, n);
    for (i = 0; i < n; i++)
        out[i] *= inv * nd_f16(d3[i]);
}

/* ------------------------------------------------------------------ block */

/* u <- block(u); the engram injection happens first when this layer is a site. */
static void block(nd_model *m, uint32_t li, float *u)
{
    const nd_layer *L  = &m->layer[li];
    uint32_t        dm = m->d_model;
    uint32_t        s, i;

    for (s = 0; s < m->n_sites; s++) {
        if (m->c.h.sites[s] != li)
            continue;
        {
            const float *ek = m->eg_k + (size_t)s * dm;
            const float *ev = m->eg_v + (size_t)s * dm;
            float        dot = 0.0f, alpha;

            rms_unit(u, dm, m->n1);
            rms_unit(ek, dm, m->n2);
            for (i = 0; i < dm; i++)
                dot += m->n1[i] * m->n2[i];
            alpha = sigmoidf_(dot / sqrtf((float)dm));
            for (i = 0; i < dm; i++)
                u[i] += alpha * ev[i];
        }
    }

    /* attention sub-block */
    zcrms(m, &L->norm_in, u, dm, m->n1);
    attention(m, li, m->n1, m->aout);
    zcrms(m, &L->post_norm, m->aout, dm, m->n2);
    {
        float g = sigmoidf_(fp16_get(m, &L->attn_gate, 0));
        for (i = 0; i < dm; i++)
            u[i] += g * m->n2[i];
    }

    /* Hadamard MLP sub-block. The MLP writes into a padded buffer, so n2 must
     * hold next_pow2(d_model) floats; for d_model=512 that is exact. */
    { ND_T0(tm);
    zcrms(m, &L->pre_hada, u, dm, m->n1);
    hadamard_mlp(m, li, m->n1, m->n2);
    ND_T1(tm, ND_P_MLP); }
    for (i = 0; i < dm; i++)
        u[i] += m->n2[i];
}

/* ------------------------------------------------------------------- step */

const float *nd_model_step(nd_model *m, uint32_t token)
{
    const float *hidden = nd_model_step_hidden(m, token);
    return nd_model_logits_all(m, hidden);
}

const float *nd_model_logits_all(nd_model *m, const float *hidden)
{
    ND_T0(tl);
    nd_cq_gemv(&m->c, &m->embedding, nd_cact_data(&m->c, &m->embedding),
               hidden, m->xh, m->logits);
    ND_T1(tl, ND_P_LOGITS);
    return m->logits;
}

void nd_model_logits_subset(nd_model *m, const float *hidden,
                            const uint32_t *ids, uint32_t n, float *out)
{
    ND_T0(tl);
    nd_cq_prepare(&m->embedding, hidden, m->xh);
    nd_cq_gemv_gather(&m->c, &m->embedding, nd_cact_data(&m->c, &m->embedding),
                      m->xh, ids, n, out);
    ND_T1(tl, ND_P_LOGITS);
}

const float *nd_model_step_hidden(nd_model *m, uint32_t token)
{
    uint32_t dm    = m->d_model;
    uint32_t n     = m->lanes;
    uint32_t nl    = n * dm;
    float    escale = sqrtf((float)dm);
    uint32_t i, j, k, li;

    /* Rotary tables for this position, shared by every layer. */
    for (i = 0; i < m->head_dim / 2; i++) {
        float angle = (float)m->pos * m->rope_inv[i];
        m->rope_cos[i] = cosf(angle);
        m->rope_sin[i] = sinf(angle);
    }

    /* Embedding (tied), scaled. */
    nd_cq_dequant_row(&m->c, &m->embedding, nd_cact_data(&m->c, &m->embedding),
                      token, m->xh, m->tmp);
    for (i = 0; i < dm; i++)
        m->tmp[i] *= escale;
    if (m->has_conf)
        pool_cell(m, m->tmp);       /* cells[0] = x0 */
    for (j = 0; j < n; j++)
        for (i = 0; i < dm; i++)
            m->lane[j * dm + i] = m->tmp[i];

    { ND_T0(te); engram_step(m, token); ND_T1(te, ND_P_ENGRAM); }

    for (li = 0; li < m->n_layers; li++) {
        float hpre[ND_MAX_LANES], hpost[ND_MAX_LANES];
        float hres[ND_MAX_LANES * ND_MAX_LANES];
        float a_pre  = fp16_get(m, &m->mhc_a_pre, li);
        float a_post = fp16_get(m, &m->mhc_a_post, li);
        float a_res  = fp16_get(m, &m->mhc_a_res, li);
        uint32_t lane_id = li % n;

        rms_unit(m->lane, nl, m->nx);

        /* The phi tensors stack all layers; this layer owns a row slice. */
        ND_T0(tphi);
        nd_cq_prepare(&m->mhc_phi_pre, m->nx, m->xh);
        nd_cq_gemv_rows(&m->c, &m->mhc_phi_pre,
                        nd_cact_data(&m->c, &m->mhc_phi_pre), m->xh,
                        li * n, n, hpre);
        nd_cq_gemv_rows(&m->c, &m->mhc_phi_post,
                        nd_cact_data(&m->c, &m->mhc_phi_post), m->xh,
                        li * n, n, hpost);
        nd_cq_gemv_rows(&m->c, &m->mhc_phi_res,
                        nd_cact_data(&m->c, &m->mhc_phi_res), m->xh,
                        li * n * n, n * n, hres);
        ND_T1(tphi, ND_P_PHI);

        for (j = 0; j < n; j++) {
            float pre_off  = 8.0f * (j == lane_id ? 1.0f : 0.0f) - 4.0f;
            float post_off = -4.0f * (j == lane_id ? 0.0f : 1.0f);
            hpre[j]  = sigmoidf_(a_pre * hpre[j] +
                                 fp16_get(m, &m->mhc_b_pre, li * n + j) + pre_off);
            hpost[j] = 2.0f * sigmoidf_(a_post * hpost[j] +
                                 fp16_get(m, &m->mhc_b_post, li * n + j) + post_off);
        }
        for (i = 0; i < n * n; i++)
            hres[i] = a_res * hres[i] + fp16_get(m, &m->mhc_b_res, li * n * n + i);
        sinkhorn(hres, n);

        /* u = sum_j hpre[j] * lane[j] */
        for (i = 0; i < dm; i++) {
            float acc = 0.0f;
            for (j = 0; j < n; j++)
                acc += hpre[j] * m->lane[j * dm + i];
            m->u[i] = acc;
        }

        /* y = block(u) - u */
        memcpy(m->ublk, m->u, sizeof(float) * dm);
        block(m, li, m->u);
        for (i = 0; i < dm; i++)
            m->u[i] -= m->ublk[i];

        /* lane' = hres @ lane + hpost * y */
        for (j = 0; j < n; j++) {
            float *dst = m->lane_next + (size_t)j * dm;
            for (i = 0; i < dm; i++) {
                float acc = 0.0f;
                for (k = 0; k < n; k++)
                    acc += hres[j * n + k] * m->lane[k * dm + i];
                dst[i] = acc + hpost[j] * m->u[i];
            }
        }
        {
            float *swap = m->lane;
            m->lane      = m->lane_next;
            m->lane_next = swap;
        }

        if (m->has_conf) {
            /* collect_hidden yields the mean over lanes for each layer. */
            for (i = 0; i < dm; i++) {
                float acc = 0.0f;
                for (j = 0; j < n; j++)
                    acc += m->lane[j * dm + i];
                m->n1[i] = acc / (float)n;
            }
            { ND_T0(tc); pool_cell(m, m->n1); ND_T1(tc, ND_P_CONF); }
        }
    }

    /* Mean over lanes, final norm, tied-embedding logits. */
    for (i = 0; i < dm; i++) {
        float acc = 0.0f;
        for (j = 0; j < n; j++)
            acc += m->lane[j * dm + i];
        m->tmp[i] = acc / (float)n;
    }
    zcrms(m, &m->final_norm, m->tmp, dm, m->y);

    m->pos++;
    return m->y;
}
