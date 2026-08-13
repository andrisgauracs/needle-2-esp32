/* nd_sample.h - grammar-constrained greedy sampling.
 *
 * Picks the highest-logit token whose bytes the grammar can accept from the
 * current state. Candidates are trialled on a copy of the state, so a
 * rejected token costs nothing.
 *
 * The grammar is disengaged while the model writes its <think> block and
 * engages on <tool_call>, which matches how the model was trained: only the
 * call is constrained, the reasoning stays legible.
 */
#ifndef ND_SAMPLE_H
#define ND_SAMPLE_H

#include <stdint.h>

#include "nd_grammar.h"
#include "nd_model.h"
#include "nd_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const nd_tokenizer *tok;
    nd_gstate           st;
    int                 engaged;   /* inside <tool_call> */
    int                 finished;  /* </tool_call> emitted */
} nd_sampler;

void nd_sampler_init(nd_sampler *s, const nd_tokenizer *tok,
                     const nd_grammar *g);

/* Choose the next token from `logits`, honouring the grammar.
 * Returns the token id, or (uint32_t)-1 if nothing is legal. */
uint32_t nd_sample(nd_sampler *s, const float *logits, uint32_t vocab);

/* Maximum candidates the constrained fast path will score directly. Beyond
 * this it is cheaper to project the whole vocabulary. */
#define ND_SAMPLE_MAX_CAND 512

/* Pick the next token from a hidden state, scoring only the tokens the grammar
 * currently permits. Inside a tool call that is typically a few dozen of 8192,
 * so the vocabulary projection - otherwise the second-largest cost per token -
 * nearly vanishes. Falls back to full logits when unconstrained or when too
 * many tokens are legal. */
uint32_t nd_sample_hidden(nd_model *m, nd_sampler *s, const float *hidden);

/* Advance sampler state by a chosen token (call for every accepted token). */
void nd_sample_accept(nd_sampler *s, uint32_t id);

#ifdef __cplusplus
}
#endif
#endif /* ND_SAMPLE_H */
