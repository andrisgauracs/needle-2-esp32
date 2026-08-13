/* nd_tokenizer.h - SentencePiece BPE reader for the RAW tokenizer tensor.
 *
 * Blob layout (from export.py):
 *   header  u32 n_pieces, u32 pad, u32 eos, u32 bos, u32 unk,
 *           u8 add_dummy_prefix, u8 byte_fallback, u16 _pad
 *   then n_pieces records in id order:
 *           f32 score, u8 type, u16 surface_len, surface_len UTF-8 bytes
 *
 * Encoding follows RefTokenizer: spaces become U+2581, an optional dummy
 * prefix is prepended, USER_DEFINED pieces (the chat markers) are matched
 * longest-first as atomic units, and everything between them is merged by
 * highest-score adjacent pair until no pair is in the vocabulary.
 */
#ifndef ND_TOKENIZER_H
#define ND_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Piece types. */
enum {
    ND_TK_NORMAL = 0,
    ND_TK_UNKNOWN = 1,
    ND_TK_CONTROL = 2,
    ND_TK_USER_DEFINED = 3,
    ND_TK_BYTE = 4
};

/* Special ids, fixed by the training recipe (needle/model/tokenizer.py). */
enum {
    ND_PAD_ID = 0, ND_EOS_ID = 1, ND_BOS_ID = 2, ND_UNK_ID = 3,
    ND_IM_START_ID = 4, ND_IM_END_ID = 5,
    ND_THINK_START_ID = 6, ND_THINK_END_ID = 7,
    ND_TOOLS_START_ID = 8, ND_TOOLS_END_ID = 9,
    ND_TOOL_CALL_START_ID = 10, ND_TOOL_CALL_END_ID = 11,
    ND_TOOL_RESULT_START_ID = 12, ND_TOOL_RESULT_END_ID = 13
};

typedef struct {
    uint32_t off;   /* offset of the surface bytes within the blob */
    uint16_t len;
    uint8_t  type;
    float    score;
} nd_piece;

typedef struct {
    const uint8_t *blob;
    uint32_t       n_pieces;
    uint32_t       pad_id, eos_id, bos_id, unk_id;
    uint8_t        add_dummy_prefix;
    uint8_t        byte_fallback;

    nd_piece      *pieces;
    uint16_t      *bucket;     /* open-addressed id+1 table, 0 = empty */
    uint32_t       mask;       /* bucket count - 1 */
    int32_t        byte_id[256];

    uint32_t      *markers;    /* USER_DEFINED ids, longest surface first */
    uint32_t       n_markers;
} nd_tokenizer;

/* Build the lookup tables over a RAW tokenizer blob used in place.
 * Returns 0 on success. Call nd_tok_free to release the tables. */
int  nd_tok_init(nd_tokenizer *t, const void *blob, size_t size);
void nd_tok_free(nd_tokenizer *t);

/* Encode `text` (UTF-8, not NUL-terminated) into `ids`.
 * Returns the number of ids written, or -1 if `cap` is too small. */
int nd_tok_encode(const nd_tokenizer *t, const char *text, size_t len,
                  uint32_t *ids, uint32_t cap);

/* Same, but `add_dummy` overrides the tokenizer's add_dummy_prefix flag.
 * Needed when encoding a continuation: the prefix already consumed the dummy
 * space, and re-adding it would shift every following token. Splitting only at
 * a USER_DEFINED marker (e.g. </tools>) also guarantees no BPE merge spans the
 * boundary, so prefix+suffix encodes identically to the whole string. */
int nd_tok_encode_ex(const nd_tokenizer *t, const char *text, size_t len,
                     uint32_t *ids, uint32_t cap, int add_dummy);

/* Decode ids back to UTF-8 in `out`. Returns bytes written (NUL-terminated),
 * or -1 if `cap` is too small. Control and unknown pieces are skipped. */
int nd_tok_decode(const nd_tokenizer *t, const uint32_t *ids, uint32_t n,
                  char *out, size_t cap);

/* Same, but `strip_dummy` controls whether a leading space is removed. Pass 0
 * when streaming token by token: the dummy-prefix rule applies once to a whole
 * sequence, and applying it per token eats real spaces between words. */
int nd_tok_decode_ex(const nd_tokenizer *t, const uint32_t *ids, uint32_t n,
                     char *out, size_t cap, int strip_dummy);

/* Surface bytes of one piece (not NUL-terminated). */
const char *nd_tok_piece(const nd_tokenizer *t, uint32_t id, uint16_t *len);

#ifdef __cplusplus
}
#endif
#endif /* ND_TOKENIZER_H */
