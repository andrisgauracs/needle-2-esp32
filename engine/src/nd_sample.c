#include "nd_sample.h"

#include <string.h>

void nd_sampler_init(nd_sampler *s, const nd_tokenizer *tok,
                     const nd_grammar *g)
{
    memset(s, 0, sizeof(*s));
    s->tok = tok;
    nd_gstate_init(&s->st, g);
}

/* Can the grammar accept every byte of this token's surface? */
static int token_ok(const nd_sampler *s, uint32_t id, nd_gstate *out)
{
    nd_gstate    trial = s->st;
    uint16_t     len;
    const char  *surf;
    uint16_t     i;

    surf = nd_tok_piece(s->tok, id, &len);
    if (!surf)
        return 0;

    for (i = 0; i < len; i++) {
        /* SentencePiece's space marker is U+2581; inside a JSON call the
         * model emits real bytes, so a marker byte here is not legal. */
        if (!nd_gstate_byte(&trial, surf[i]))
            return 0;
    }
    if (out)
        *out = trial;
    return 1;
}

uint32_t nd_sample(nd_sampler *s, const float *logits, uint32_t vocab)
{
    uint32_t best = (uint32_t)-1;
    float    bv = -1e30f;
    uint32_t j;

    if (!s->engaged) {
        /* Unconstrained: plain argmax over the whole vocabulary. */
        for (j = 0; j < vocab; j++)
            if (logits[j] > bv) { bv = logits[j]; best = j; }
        return best;
    }

    /* Once the call is complete only </tool_call> may follow. */
    if (nd_gstate_complete(&s->st))
        return ND_TOOL_CALL_END_ID;

    for (j = 0; j < vocab; j++) {
        if (logits[j] <= bv)
            continue;
        /* Control pieces carry no bytes and would slip through the byte
         * check, so exclude them explicitly while constrained. */
        if (j < 16)
            continue;
        if (!token_ok(s, j, NULL))
            continue;
        bv = logits[j];
        best = j;
    }
    return best;
}

uint32_t nd_sample_hidden(nd_model *m, nd_sampler *s, const float *hidden)
{
    static uint32_t cand[ND_SAMPLE_MAX_CAND];
    static float    score[ND_SAMPLE_MAX_CAND];
    uint32_t        n = 0, j, best = (uint32_t)-1;
    float           bv = -1e30f;

    /* Unconstrained (the <think> block): plain argmax over everything. */
    if (!s->engaged)
        return nd_sample(s, nd_model_logits_all(m, hidden), m->vocab);

    if (nd_gstate_complete(&s->st))
        return ND_TOOL_CALL_END_ID;

    /* Enumerate what the grammar allows. Byte-checking the vocabulary costs
     * ~25K byte steps, far less than 4.2M multiply-adds. */
    for (j = 16; j < m->vocab; j++) {
        if (!token_ok(s, j, NULL))
            continue;
        if (n < ND_SAMPLE_MAX_CAND)
            cand[n] = j;
        n++;
        if (n > ND_SAMPLE_MAX_CAND)
            break;
    }

    if (n == 0)
        return (uint32_t)-1;
    if (n > ND_SAMPLE_MAX_CAND)                 /* too many: full projection */
        return nd_sample(s, nd_model_logits_all(m, hidden), m->vocab);

    nd_model_logits_subset(m, hidden, cand, n, score);
    for (j = 0; j < n; j++)
        if (score[j] > bv) { bv = score[j]; best = cand[j]; }
    return best;
}

void nd_sample_accept(nd_sampler *s, uint32_t id)
{
    if (id == ND_TOOL_CALL_START_ID) {
        s->engaged = 1;
        nd_gstate_open(&s->st);
        return;
    }
    if (id == ND_TOOL_CALL_END_ID) {
        s->engaged = 0;
        s->finished = 1;
        return;
    }
    if (s->engaged) {
        nd_gstate trial;
        if (token_ok(s, id, &trial))
            s->st = trial;
    }
}
