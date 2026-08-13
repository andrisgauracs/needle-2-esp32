/* nd_model.h - Simple Attention Network forward pass, one token at a time.
 *
 * Architecture (needle/model/architecture.py), per layer, over 4 hyper-
 * connection lanes:
 *
 *   nx        = rms_unit(flatten(lanes))                 (4*d_model)
 *   h_pre     = sigmoid(a_pre * nx@phi_pre + b_pre + pre_off)
 *   u         = sum_j h_pre[j] * lane[j]
 *   y         = block(u) - u
 *   h_post    = 2*sigmoid(a_post * nx@phi_post + b_post + post_off)
 *   h_res     = sinkhorn(a_res * (nx@phi_res) + b_res)
 *   lane'[j]  = sum_k h_res[j][k]*lane[k] + h_post[j]*y
 *
 * and block() is engram-inject -> ZCRMSNorm -> gated GQA attention ->
 * ZCRMSNorm -> gated residual -> ZCRMSNorm -> Hadamard MLP -> residual.
 *
 * The KV cache is a per-layer ring buffer of the last kv_window positions,
 * stored int8 with a per-vector scale. At kv_bits=8 the model was not
 * post-trained against a codebook (configure_deploy zeroes KV_BITS), so plain
 * symmetric int8 is faithful and costs 3.5 MB instead of 14 MB of float.
 */
#ifndef ND_MODEL_H
#define ND_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "nd_cact.h"
#include "nd_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Route large buffers to PSRAM on the ESP32; plain malloc elsewhere.
 * ND_ALLOC/ND_FREE are defined by the ESP-IDF component build. */
#ifdef ESP_PLATFORM
/* The weights are mmap'd from flash, but activations, the KV cache and the
 * confidence pool must live in PSRAM - internal SRAM is only ~380 KB. */
#include "esp_heap_caps.h"
#ifndef ND_ALLOC
#define ND_ALLOC(n) heap_caps_malloc((n), MALLOC_CAP_SPIRAM)
#endif
#ifndef ND_FREE
#define ND_FREE(p) heap_caps_free(p)
#endif
/* Small per-token scratch is touched on every weight; keeping it in internal
 * SRAM instead of PSRAM matters far more than its size does. */
#ifndef ND_ALLOC_FAST
#define ND_ALLOC_FAST(n) heap_caps_malloc((n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
#endif
#else
#include <stdlib.h>
#ifndef ND_ALLOC
#define ND_ALLOC(n) malloc(n)
#endif
#ifndef ND_FREE
#define ND_FREE(p) free(p)
#endif
#ifndef ND_ALLOC_FAST
#define ND_ALLOC_FAST(n) malloc(n)
#endif
#endif

#define ND_MAX_LANES 8
#define ND_MAX_SITES 4

typedef struct {
    nd_tensor norm_in, q_proj, k_proj, v_proj, q_norm, k_norm;
    nd_tensor gate_proj, out_proj, post_norm, attn_gate, pre_hada;
    nd_tensor d1, d2, d3;
} nd_layer;

typedef struct {
    nd_tensor tables, key_proj, value_proj, taps;
} nd_engram;

typedef struct {
    nd_cact       c;
    nd_tokenizer  tok;
    int           tok_ready;

    uint32_t      d_model, n_layers, n_heads, n_kv_heads, head_dim;
    uint32_t      attn_dim, kv_dim, lanes, vocab, window;

    nd_layer     *layer;
    nd_tensor     mhc_a_pre, mhc_a_post, mhc_a_res;
    nd_tensor     mhc_b_pre, mhc_b_post, mhc_b_res;
    nd_tensor     mhc_phi_pre, mhc_phi_post, mhc_phi_res;
    nd_engram     engram[ND_MAX_SITES];
    uint32_t      n_sites;
    nd_tensor     embedding, final_norm;

    /* Confidence head (optional; present in needle2.cact). Pooling is a
     * softmax over every hidden cell of every token, which materialised
     * naively would be ~12 MB. It is accumulated online instead - running max
     * and running weighted sum per probe - so it costs P*d_model floats. */
    int           has_conf;
    nd_tensor     conf_probes, conf_proj, conf_bias;
    uint32_t      n_probes;
    float        *probes_f;     /* [n_probes][d_model], fp16 expanded once */
    float        *pool_acc;     /* [n_probes][d_model] */
    float        *pool_max;     /* [n_probes] */
    float        *pool_sum;     /* [n_probes] */

    /* ---- session state ---- */
    uint32_t      pos;          /* absolute position of the next token */
    uint32_t      n_sink;       /* leading positions pinned in the KV cache */
    int8_t       *k_cache;      /* [layer][window][kv_dim] */
    int8_t       *v_cache;
    float        *k_scale;      /* [layer][window][n_kv_heads] */
    float        *v_scale;

    /* Snapshot of session state after a fixed prompt prefix, so the prefix is
     * prefilled once and every later request resumes from it. The prefix's KV
     * lives in the pinned sink slots, which the ring never overwrites, so only
     * the non-KV state has to be copied. */
    int           has_snap;
    uint32_t      snap_pos, snap_eg_pos;
    uint32_t      snap_hist[8];
    float        *snap_eg_hist;
    float        *snap_pool_acc, *snap_pool_max, *snap_pool_sum;

    uint32_t      hist[8];      /* recent token ids, for engram n-grams */
    float        *eg_hist;      /* [site][ND_EG_HIST][d_model] raw engram v */
    uint32_t      eg_pos;

    /* ---- scratch ---- */
    float        *lane;         /* [lanes][d_model] */
    float        *lane_next;
    float        *nx;           /* [lanes*d_model] */
    float        *xh;           /* gemv activation scratch, max in_pad */
    float        *lut;          /* 2-bit pair table over xh */
    float        *lut4;         /* 2-bit quad table, one group live */
    float        *gacc;         /* per-row accumulator for lut4 */
    float        *u, *ublk;     /* lane mix, and its pre-block copy */
    float        *n1, *n2;      /* block-local norm / sub-block output */
    float        *y, *tmp, *tmp2;
    float        *q, *kbuf, *vbuf, *gate, *attn, *aout;
    float        *rope_inv;     /* [head_dim/2] inverse frequencies */
    float        *rope_cos;     /* [head_dim/2] for the current position */
    float        *rope_sin;
    float        *eg_k, *eg_v;  /* [site][d_model] for the current token */
    float        *logits;       /* [vocab] */
    float        *row;          /* dequant scratch for engram table rows */
} nd_model;

/* Open a model over a `.cact` blob used in place (may be mmap'd flash).
 * Returns 0 on success. */
int  nd_model_open(nd_model *m, const void *blob, size_t size);
void nd_model_close(nd_model *m);

/* Clear the KV cache and conversation position. */
void nd_model_reset(nd_model *m);

/* Pin the first `n` positions in the KV cache so the sliding window can never
 * evict them. Needle renders the tool schemas at the head of the prompt and
 * relies on them staying visible for the whole turn; without pinning, a long
 * prompt plus a long generation scrolls the <tools> block out and the model
 * starts inventing tool names. Call after prefilling the pinned prefix.
 * `n` is clamped to leave at least half the window for recent context. */
void nd_model_set_sink(nd_model *m, uint32_t n);

/* Freeze the current position as a reusable prefix: pins it as the KV sink and
 * snapshots the engram/confidence state. Returns 0 on success. */
int  nd_model_snapshot(nd_model *m);

/* Resume from the snapshot, discarding everything decoded since. */
void nd_model_rewind(nd_model *m);

/* Feed one token at the current position and return logits[vocab].
 * The pointer stays valid until the next call. */
const float *nd_model_step(nd_model *m, uint32_t token);

/* Same, but stops before the vocabulary projection and returns the final
 * normalised hidden state. Prefill never looks at logits, and constrained
 * decoding only needs a handful of rows, so computing all 8192 is waste. */
const float *nd_model_step_hidden(nd_model *m, uint32_t token);

/* Full logits from a hidden state returned by nd_model_step_hidden. */
const float *nd_model_logits_all(nd_model *m, const float *hidden);

/* Logits for `n` specific tokens only, written to `out`. */
void nd_model_logits_subset(nd_model *m, const float *hidden,
                            const uint32_t *ids, uint32_t n, float *out);

/* Per-phase microsecond counters, filled when ND_PROFILE is compiled in.
 * Indices: 0 attn projections (2-bit LUT), 1 attention itself, 2 Hadamard MLP,
 * 3 mHC phi (4-bit), 4 engram, 5 logits (4-bit), 6 prepare+LUT build,
 * 7 confidence pool. */
enum { ND_P_PROJ, ND_P_ATTN, ND_P_MLP, ND_P_PHI, ND_P_ENGRAM,
       ND_P_LOGITS, ND_P_PREP, ND_P_CONF, ND_P_COUNT };
extern uint64_t nd_prof[ND_P_COUNT];

/* Calibrated confidence over everything fed so far, in [0,1].
 * Returns -1 if the blob carries no confidence head. */
float nd_model_confidence(nd_model *m);

#ifdef __cplusplus
}
#endif
#endif /* ND_MODEL_H */
