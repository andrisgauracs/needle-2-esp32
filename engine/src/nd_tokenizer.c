#include "nd_tokenizer.h"

#include <stdlib.h>
#include <string.h>

/* U+2581 LOWER ONE EIGHTH BLOCK, SentencePiece's space marker. */
static const char SP_SPACE[3] = { (char)0xE2, (char)0x96, (char)0x81 };
#define SP_SPACE_LEN 3

static uint32_t rd_u32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static float rd_f32(const uint8_t *p)
{
    uint32_t b = rd_u32(p);
    float    f;
    memcpy(&f, &b, 4);
    return f;
}

static uint32_t fnv1a(const char *s, size_t n)
{
    uint32_t h = 2166136261u;
    size_t   i;
    for (i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static const char *surface(const nd_tokenizer *t, uint32_t id, uint16_t *len)
{
    *len = t->pieces[id].len;
    return (const char *)(t->blob + t->pieces[id].off);
}

/* Look up a surface string; returns the id or -1. */
static int32_t lookup(const nd_tokenizer *t, const char *s, size_t n)
{
    uint32_t slot = fnv1a(s, n) & t->mask;

    for (;;) {
        uint16_t entry = t->bucket[slot];
        uint16_t plen;
        const char *p;

        if (entry == 0)
            return -1;
        p = surface(t, (uint32_t)(entry - 1), &plen);
        if (plen == n && memcmp(p, s, n) == 0)
            return (int32_t)(entry - 1);
        slot = (slot + 1) & t->mask;
    }
}

static void insert(nd_tokenizer *t, uint32_t id)
{
    uint16_t    len;
    const char *s    = surface(t, id, &len);
    uint32_t    slot = fnv1a(s, len) & t->mask;

    while (t->bucket[slot] != 0)
        slot = (slot + 1) & t->mask;
    t->bucket[slot] = (uint16_t)(id + 1);
}

static int hex_nibble(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* "<0xAB>" -> 0xAB, or -1. */
static int byte_piece_value(const char *s, uint16_t len)
{
    int hi, lo;
    if (len != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>')
        return -1;
    hi = hex_nibble(s[3]);
    lo = hex_nibble(s[4]);
    if (hi < 0 || lo < 0)
        return -1;
    return (hi << 4) | lo;
}

int nd_tok_init(nd_tokenizer *t, const void *blob, size_t size)
{
    const uint8_t *b = (const uint8_t *)blob;
    size_t         off;
    uint32_t       i, nbucket;

    if (!t || !b || size < 24)
        return -1;

    memset(t, 0, sizeof(*t));
    t->blob             = b;
    t->n_pieces         = rd_u32(b);
    t->pad_id           = rd_u32(b + 4);
    t->eos_id           = rd_u32(b + 8);
    t->bos_id           = rd_u32(b + 12);
    t->unk_id           = rd_u32(b + 16);
    t->add_dummy_prefix = b[20];
    t->byte_fallback    = b[21];

    if (t->n_pieces == 0 || t->n_pieces > 65534)
        return -2;

    t->pieces = (nd_piece *)calloc(t->n_pieces, sizeof(nd_piece));
    if (!t->pieces)
        return -3;

    off = 24;
    for (i = 0; i < t->n_pieces; i++) {
        uint16_t slen;
        if (off + 7 > size)
            goto bad;
        t->pieces[i].score = rd_f32(b + off);
        t->pieces[i].type  = b[off + 4];
        slen               = (uint16_t)(b[off + 5] | (b[off + 6] << 8));
        off += 7;
        if (off + slen > size)
            goto bad;
        t->pieces[i].off = (uint32_t)off;
        t->pieces[i].len = slen;
        off += slen;
    }

    /* Power-of-two table with load factor < 0.6. */
    nbucket = 1;
    while (nbucket < t->n_pieces * 2)
        nbucket <<= 1;
    t->mask   = nbucket - 1;
    t->bucket = (uint16_t *)calloc(nbucket, sizeof(uint16_t));
    if (!t->bucket)
        goto bad;

    for (i = 0; i < 256; i++)
        t->byte_id[i] = -1;

    t->markers = (uint32_t *)calloc(t->n_pieces, sizeof(uint32_t));
    if (!t->markers)
        goto bad;

    for (i = 0; i < t->n_pieces; i++) {
        uint16_t    len;
        const char *s = surface(t, i, &len);

        /* Later duplicates must not shadow the canonical (lower) id. */
        if (lookup(t, s, len) < 0)
            insert(t, i);

        if (t->pieces[i].type == ND_TK_BYTE) {
            int v = byte_piece_value(s, len);
            if (v >= 0)
                t->byte_id[v] = (int32_t)i;
        } else if (t->pieces[i].type == ND_TK_USER_DEFINED) {
            t->markers[t->n_markers++] = i;
        }
    }

    /* Longest surface first, so a marker is never split by a shorter prefix. */
    for (i = 1; i < t->n_markers; i++) {
        uint32_t key = t->markers[i];
        uint32_t j   = i;
        while (j > 0 && t->pieces[t->markers[j - 1]].len < t->pieces[key].len) {
            t->markers[j] = t->markers[j - 1];
            j--;
        }
        t->markers[j] = key;
    }
    return 0;

bad:
    nd_tok_free(t);
    return -4;
}

void nd_tok_free(nd_tokenizer *t)
{
    if (!t)
        return;
    free(t->pieces);
    free(t->bucket);
    free(t->markers);
    t->pieces  = NULL;
    t->bucket  = NULL;
    t->markers = NULL;
}

const char *nd_tok_piece(const nd_tokenizer *t, uint32_t id, uint16_t *len)
{
    if (id >= t->n_pieces) {
        *len = 0;
        return NULL;
    }
    return surface(t, id, len);
}

/* One BPE symbol: a slice of the escaped buffer, in a doubly-linked list so
 * merges are O(1) once the best pair is known. */
typedef struct {
    uint32_t off;
    uint32_t len;
    int32_t  prev, next;
} sym_t;

static uint32_t utf8_len(uint8_t c)
{
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; /* stray continuation byte: treat as its own symbol */
}

/* Merge one segment and append its ids. Returns ids written, or -1 on
 * overflow. */
static int bpe_segment(const nd_tokenizer *t, const char *buf, uint32_t off,
                       uint32_t len, uint32_t *ids, uint32_t cap, uint32_t n)
{
    sym_t   *syms;
    uint32_t nsym = 0, i;
    int32_t  head = 0;

    if (len == 0)
        return (int)n;

    syms = (sym_t *)malloc(sizeof(sym_t) * len);
    if (!syms)
        return -1;

    for (i = 0; i < len;) {
        uint32_t cl = utf8_len((uint8_t)buf[off + i]);
        if (i + cl > len)
            cl = len - i;
        syms[nsym].off  = off + i;
        syms[nsym].len  = cl;
        syms[nsym].prev = (int32_t)nsym - 1;
        syms[nsym].next = (int32_t)nsym + 1;
        nsym++;
        i += cl;
    }
    syms[nsym - 1].next = -1;

    /* Repeatedly merge the highest-scoring adjacent pair present in the
     * vocabulary, exactly as RefTokenizer._bpe does. */
    for (;;) {
        int32_t best = -1;
        float   best_score = 0.0f;
        int32_t a;

        for (a = head; a >= 0 && syms[a].next >= 0; a = syms[a].next) {
            int32_t  bnode = syms[a].next;
            uint32_t total = syms[a].len + syms[bnode].len;
            int32_t  id    = lookup(t, buf + syms[a].off, total);
            if (id < 0)
                continue;
            if (best < 0 || t->pieces[id].score > best_score) {
                best       = a;
                best_score = t->pieces[id].score;
            }
        }
        if (best < 0)
            break;
        {
            int32_t bnode = syms[best].next;
            syms[best].len += syms[bnode].len;
            syms[best].next = syms[bnode].next;
            if (syms[bnode].next >= 0)
                syms[syms[bnode].next].prev = best;
        }
    }

    for (i = (uint32_t)head; ; ) {
        int32_t  id = lookup(t, buf + syms[i].off, syms[i].len);
        int32_t  nxt;

        if (id >= 0) {
            if (n >= cap) goto overflow;
            ids[n++] = (uint32_t)id;
        } else if (t->byte_fallback) {
            uint32_t k;
            for (k = 0; k < syms[i].len; k++) {
                int32_t bid = t->byte_id[(uint8_t)buf[syms[i].off + k]];
                if (n >= cap) goto overflow;
                ids[n++] = (bid >= 0) ? (uint32_t)bid : t->unk_id;
            }
        } else {
            if (n >= cap) goto overflow;
            ids[n++] = t->unk_id;
        }

        nxt = syms[i].next;
        if (nxt < 0)
            break;
        i = (uint32_t)nxt;
    }

    free(syms);
    return (int)n;

overflow:
    free(syms);
    return -1;
}

int nd_tok_encode(const nd_tokenizer *t, const char *text, size_t len,
                  uint32_t *ids, uint32_t cap)
{
    return nd_tok_encode_ex(t, text, len, ids, cap, t->add_dummy_prefix);
}

int nd_tok_encode_ex(const nd_tokenizer *t, const char *text, size_t len,
                     uint32_t *ids, uint32_t cap, int add_dummy)
{
    char    *esc;
    size_t   elen = 0, i;
    uint32_t n = 0, seg_start;
    int      rc;

    if (len == 0)
        return 0;

    /* Escape: every space becomes a 3-byte marker, plus the dummy prefix. */
    esc = (char *)malloc(len * SP_SPACE_LEN + SP_SPACE_LEN);
    if (!esc)
        return -1;

    if (add_dummy) {
        memcpy(esc, SP_SPACE, SP_SPACE_LEN);
        elen = SP_SPACE_LEN;
    }
    for (i = 0; i < len; i++) {
        if (text[i] == ' ') {
            memcpy(esc + elen, SP_SPACE, SP_SPACE_LEN);
            elen += SP_SPACE_LEN;
        } else {
            esc[elen++] = text[i];
        }
    }

    seg_start = 0;
    i         = 0;
    while (i < elen) {
        uint32_t m, hit = 0;

        for (m = 0; m < t->n_markers; m++) {
            uint32_t    id  = t->markers[m];
            uint16_t    mlen;
            const char *ms  = surface(t, id, &mlen);

            if (mlen == 0 || i + mlen > elen)
                continue;
            if (memcmp(esc + i, ms, mlen) != 0)
                continue;

            rc = bpe_segment(t, esc, seg_start, (uint32_t)i - seg_start, ids, cap, n);
            if (rc < 0) goto overflow;
            n = (uint32_t)rc;

            if (n >= cap) goto overflow;
            ids[n++] = id;

            i += mlen;
            seg_start = (uint32_t)i;
            hit = 1;
            break;
        }
        if (!hit)
            i += utf8_len((uint8_t)esc[i]);
    }

    rc = bpe_segment(t, esc, seg_start, (uint32_t)elen - seg_start, ids, cap, n);
    if (rc < 0) goto overflow;
    n = (uint32_t)rc;

    free(esc);
    return (int)n;

overflow:
    free(esc);
    return -1;
}

int nd_tok_decode(const nd_tokenizer *t, const uint32_t *ids, uint32_t n,
                  char *out, size_t cap)
{
    return nd_tok_decode_ex(t, ids, n, out, cap, 1);
}

int nd_tok_decode_ex(const nd_tokenizer *t, const uint32_t *ids, uint32_t n,
                     char *out, size_t cap, int strip_dummy)
{
    size_t   w = 0;
    uint32_t i;

    for (i = 0; i < n; i++) {
        uint16_t    len;
        const char *s;

        if (ids[i] >= t->n_pieces)
            continue;
        s = surface(t, ids[i], &len);

        if (t->pieces[ids[i]].type == ND_TK_BYTE) {
            int v = byte_piece_value(s, len);
            if (v < 0)
                continue;
            if (w + 1 >= cap)
                return -1;
            out[w++] = (char)v;
            continue;
        }
        if (t->pieces[ids[i]].type == ND_TK_CONTROL ||
            t->pieces[ids[i]].type == ND_TK_UNKNOWN)
            continue;

        if (w + len >= cap)
            return -1;
        memcpy(out + w, s, len);
        w += len;
    }
    out[w] = '\0';

    /* Unescape U+2581 back to a space, in place. */
    {
        size_t r = 0, k = 0;
        while (r < w) {
            if (r + SP_SPACE_LEN <= w && memcmp(out + r, SP_SPACE, SP_SPACE_LEN) == 0) {
                out[k++] = ' ';
                r += SP_SPACE_LEN;
            } else {
                out[k++] = out[r++];
            }
        }
        w = k;
        out[w] = '\0';
    }

    /* Drop the dummy prefix the encoder added. */
    if (strip_dummy && t->add_dummy_prefix && w > 0 && out[0] == ' ') {
        memmove(out, out + 1, w - 1);
        w--;
        out[w] = '\0';
    }
    return (int)w;
}
