#include "nd_cact.h"

#include <string.h>

/* The header is a flat run of 30 little-endian u32-sized fields and
 * nd_cact_header mirrors that run exactly, so one bounds-checked copy decodes
 * it. (Every field is 4 bytes, so the struct has no interior padding.) */
#define ND_HDR_WORDS 30
#define ND_HDR_BYTES (ND_HDR_WORDS * 4)

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t rd_u64(const uint8_t *p)
{
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}

int nd_cact_open(nd_cact *c, const void *blob, size_t size)
{
    const uint8_t *b = (const uint8_t *)blob;

    if (!c || !b || size < ND_HDR_BYTES)
        return -1;

    memset(c, 0, sizeof(*c));
    c->base = b;
    c->size = size;

    /* Decode the header word by word rather than casting: the blob is only
     * 64-byte aligned at tensor boundaries and may be an mmap'd flash window,
     * and this keeps the reader endian-explicit. */
    {
        uint32_t words[ND_HDR_WORDS];
        uint32_t i;
        for (i = 0; i < ND_HDR_WORDS; i++)
            words[i] = rd_u32(b + i * 4);
        memcpy(&c->h, words, sizeof(c->h));
    }

    if (c->h.tag != ND_CACT_TAG)
        return -2;

    /* Only the geometry this engine actually implements. */
    if (c->h.d_model == 0 || c->h.num_layers == 0 || c->h.num_heads == 0 ||
        c->h.num_kv_heads == 0 || c->h.head_dim == 0 ||
        c->h.num_heads % c->h.num_kv_heads != 0 ||
        c->h.num_heads * c->h.head_dim != c->h.attn_dim ||
        c->h.num_orders > 4 || c->h.num_sites > 4)
        return -3;

    {
        size_t cb_off  = ND_HDR_BYTES;
        size_t cb_len  = (size_t)c->h.codebook_len * 4;
        size_t dir_off = cb_off + cb_len;
        size_t dir_len = (size_t)c->h.num_tensors * ND_CACT_REC_SIZE;

        if (cb_off + cb_len < cb_off || dir_off + dir_len < dir_off ||
            dir_off + dir_len > size)
            return -4;

        /* cb2[4] | cb3[8] | cb4[16] */
        if (c->h.codebook_len < 28)
            return -5;

        c->codebook  = (const float *)(const void *)(b + cb_off);
        c->directory = b + dir_off;
        c->n         = c->h.num_tensors;
    }
    return 0;
}

const float *nd_cact_codebook(const nd_cact *c, uint32_t bits)
{
    switch (bits) {
    case 2: return c->codebook;          /* 4 entries  */
    case 3: return c->codebook + 4;      /* 8 entries  */
    case 4: return c->codebook + 12;     /* 16 entries */
    default: return NULL;
    }
}

int nd_cact_tensor(const nd_cact *c, uint32_t i, nd_tensor *t)
{
    const uint8_t *r;

    if (!c || !t || i >= c->n)
        return -1;

    r = c->directory + (size_t)i * ND_CACT_REC_SIZE;

    t->dtype    = r[0];
    t->ndim     = r[1];
    /* r[2..3] is padding */
    t->shape[0] = rd_u32(r + 4);
    t->shape[1] = rd_u32(r + 8);
    t->shape[2] = rd_u32(r + 12);
    t->shape[3] = rd_u32(r + 16);
    t->offset   = rd_u64(r + 20);
    t->nbytes   = rd_u64(r + 28);
    t->group    = rd_u32(r + 36);
    t->bits     = rd_u32(r + 40);

    /* Shape words past ndim are written as zero; normalise them so callers
     * can treat shape[] uniformly. */
    {
        uint32_t d;
        for (d = t->ndim; d < 4; d++)
            t->shape[d] = 0;
    }
    return 0;
}

const void *nd_cact_data(const nd_cact *c, const nd_tensor *t)
{
    if (t->offset > c->size || t->nbytes > c->size - t->offset)
        return NULL;
    return c->base + t->offset;
}
