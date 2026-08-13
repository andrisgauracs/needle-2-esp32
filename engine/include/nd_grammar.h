/* nd_grammar.h - byte-level grammar compiled from tool schemas.
 *
 * Needle emits `<think>...</think>\n<tool_call>[...]</tool_call>`. The
 * reasoning is generated free-form; only the call is constrained, so the
 * grammar stays disengaged until the model emits <tool_call> and disengages
 * again at </tool_call>.
 *
 * Inside the call the machine accepts exactly:
 *
 *   [] | [ obj (, obj)* ]
 *   obj  = {"name":"<one of the declared tools>","arguments":{ pairs }}
 *   pair = "<a declared property>":<value matching its type and constraints>
 *
 * A candidate token is legal iff every byte of its surface advances the
 * machine. State is a small POD struct so the sampler can copy it, trial-run
 * a token's bytes, and throw the copy away - no allocation per candidate.
 *
 * Deliberate scope: string (with enum), integer, number and boolean values,
 * which is what device-control schemas use. Nested objects and arrays are
 * rejected at compile time rather than silently unconstrained, so we never
 * believe we are enforcing a schema we are not.
 */
#ifndef ND_GRAMMAR_H
#define ND_GRAMMAR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ND_GR_MAX_TOOLS  8
#define ND_GR_MAX_PROPS  12
#define ND_GR_MAX_ENUM   16
#define ND_GR_STRLEN     40

typedef enum {
    ND_T_STRING = 0,
    ND_T_INTEGER,
    ND_T_NUMBER,
    ND_T_BOOLEAN
} nd_vtype;

typedef struct {
    char     name[ND_GR_STRLEN];
    uint8_t  type;                 /* nd_vtype */
    uint8_t  n_enum;
    char     enums[ND_GR_MAX_ENUM][ND_GR_STRLEN];
    uint8_t  has_min, has_max;
    double   min, max;
} nd_prop;

typedef struct {
    char     name[ND_GR_STRLEN];
    uint8_t  n_props;
    nd_prop  props[ND_GR_MAX_PROPS];
    uint16_t required;             /* bitmask over props */
} nd_tool;

typedef struct {
    uint8_t  n_tools;
    nd_tool  tools[ND_GR_MAX_TOOLS];
} nd_grammar;

/* Compact a JSON document: strip whitespace that sits outside string
 * literals, leaving string contents untouched. Returns the written length, or
 * -1 if `cap` is too small.
 *
 * This matters more than it looks. The model was trained on schemas rendered
 * with json.dumps(separators=(",",":")), and a pretty-printed schema is far
 * enough off-distribution that it starts inventing tools that were never
 * declared - with no error anywhere. Compacting on the way in makes the schema
 * file's formatting irrelevant. */
int nd_json_compact(const char *src, size_t len, char *dst, size_t cap);

/* Parse a tools JSON array into a grammar. Returns 0 on success, negative on
 * malformed input or a construct outside the supported scope. On error,
 * `err` (if non-NULL) receives a short human-readable reason. */
int nd_grammar_compile(nd_grammar *g, const char *tools_json, size_t len,
                       const char **err);

/* Parser states. Exposed only so nd_gstate can be a POD the caller copies. */
typedef enum {
    ND_G_OFF = 0,      /* not yet inside <tool_call> */
    ND_G_ARRAY_OPEN,
    ND_G_ARRAY_FIRST,
    ND_G_OBJ_OPEN,
    ND_G_NAME_KEY,
    ND_G_NAME_VAL,
    ND_G_ARGS_KEY,
    ND_G_ARGS_FIRST,
    ND_G_NEXT_PROP,    /* after a ',' inside arguments: only a key may follow */
    ND_G_PROP_NAME,
    ND_G_PROP_COLON,
    ND_G_VAL_START,
    ND_G_VAL_STR,
    ND_G_VAL_ENUM,
    ND_G_VAL_NUM,
    ND_G_VAL_BOOL,
    ND_G_AFTER_VAL,
    ND_G_AFTER_OBJ,    /* arguments object closed; expect the call's '}' */
    ND_G_CLOSE_CALL,   /* call object closed; expect ',' or ']' */
    ND_G_DONE
} nd_gphase;

typedef struct {
    const nd_grammar *g;
    uint8_t  phase;
    uint8_t  tool;                 /* selected tool index */
    uint8_t  prop;                 /* selected property index */
    uint16_t seen;                 /* properties already emitted */
    uint16_t cand;                 /* viable alternatives bitmask */
    uint8_t  lit_pos;              /* cursor within the literal being matched */
    uint8_t  num_frac;             /* seen a '.' */
    uint8_t  num_any;              /* seen at least one digit */
    uint8_t  num_neg;
    double   num_val;
    double   num_scale;            /* fractional place value */
    const char *lit;               /* literal being matched, when fixed */
} nd_gstate;

/* Begin in the disengaged state. */
void nd_gstate_init(nd_gstate *s, const nd_grammar *g);

/* Engage the machine (call when <tool_call> has been emitted). */
void nd_gstate_open(nd_gstate *s);

/* Feed one byte. Returns 1 if accepted (state advanced), 0 if rejected
 * (state is then undefined - callers work on a copy). */
int nd_gstate_byte(nd_gstate *s, char c);

/* True when the call is complete and only </tool_call> may follow. */
int nd_gstate_complete(const nd_gstate *s);

#ifdef __cplusplus
}
#endif
#endif /* ND_GRAMMAR_H */
