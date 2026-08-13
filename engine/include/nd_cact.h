/* nd_cact.h - reader for the Needle 2 `.cact` v3 deployment blob.
 *
 * The v3 blob (tag 0x05E12A83) is self-describing: unlike the v2 format
 * documented in the Python package, the architecture geometry rides in the
 * header, so the runtime does not bake in a canon config.
 *
 * Layout (little-endian):
 *   0x00  header      (see nd_cact_header)
 *   0x78  codebook    codebook_len * f32 = cb2[4] | cb3[8] | cb4[16]
 *   0xE8  directory   num_tensors * 44-byte records
 *   ...   tensor blobs, each 64-byte aligned
 *
 * Nothing here copies or allocates: the blob is used in place, so on the
 * ESP32-S3 it can point straight at an mmap'd flash partition.
 */
#ifndef ND_CACT_H
#define ND_CACT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ND_CACT_TAG      0x05E12A83u
#define ND_CACT_ALIGN    64
#define ND_CACT_REC_SIZE 44

/* dtype codes in a directory record */
enum {
    ND_DT_FP16 = 1,
    ND_DT_FP32 = 2,
    ND_DT_CQ   = 3, /* Cactus-Quants; `bits` gives the width (2, 3 or 4) */
    ND_DT_RAW  = 4
};

/* Geometry as stored in the v3 header. Field order is the on-disk order. */
typedef struct {
    uint32_t tag;
    uint32_t num_tensors;
    uint32_t codebook_len;
    uint32_t kv_window;   /* sliding-window width the model was trained with */
    uint32_t kv_bits;     /* KV-cache width it was post-trained for */
    uint32_t vocab_size;
    uint32_t d_model;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t num_layers;
    uint32_t head_dim;
    uint32_t max_seq_len;
    uint32_t attn_dim;
    uint32_t mhc_lanes;
    uint32_t engram_slots;
    uint32_t engram_sub_dim;
    uint32_t engram_conv_taps;
    uint32_t engram_tables;    /* len(orders) * heads */
    uint32_t engram_dilation;
    uint32_t num_orders;
    uint32_t orders[4];        /* n-gram orders, zero-padded */
    uint32_t num_sites;
    uint32_t sites[4];         /* layer indices the engram fires at, zero-padded */
    float    rope_theta;
} nd_cact_header;

/* One directory record, decoded. */
typedef struct {
    uint8_t  dtype;
    uint8_t  ndim;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t nbytes;
    uint32_t group;
    uint32_t bits;
} nd_tensor;

typedef struct {
    const uint8_t   *base;      /* start of the blob */
    size_t           size;
    nd_cact_header   h;
    const float     *codebook;  /* cb2 | cb3 | cb4, already scaled by 1/sqrt(group) */
    const uint8_t   *directory; /* raw record array */
    uint32_t         n;         /* == h.num_tensors */
} nd_cact;

/* Codebook slice for a given width. `bits` must be 2, 3 or 4. */
const float *nd_cact_codebook(const nd_cact *c, uint32_t bits);

/* Parse a blob in place. Returns 0 on success, negative on a bad blob. */
int nd_cact_open(nd_cact *c, const void *blob, size_t size);

/* Decode directory record `i`. Returns 0 on success. */
int nd_cact_tensor(const nd_cact *c, uint32_t i, nd_tensor *t);

/* Pointer to tensor `i`'s payload, or NULL if the record runs past the blob. */
const void *nd_cact_data(const nd_cact *c, const nd_tensor *t);

/* Padded reduction length for a CQ tensor: shape[1] rounded up to `group`. */
static inline uint32_t nd_cq_in_pad(const nd_tensor *t)
{
    return (t->shape[1] + t->group - 1) / t->group * t->group;
}

/* Bytes of packed indices per output row. */
static inline uint32_t nd_cq_row_bytes(const nd_tensor *t)
{
    return nd_cq_in_pad(t) * t->bits / 8;
}

/* Number of quantisation groups per output row. */
static inline uint32_t nd_cq_groups(const nd_tensor *t)
{
    return nd_cq_in_pad(t) / t->group;
}

#ifdef __cplusplus
}
#endif
#endif /* ND_CACT_H */
