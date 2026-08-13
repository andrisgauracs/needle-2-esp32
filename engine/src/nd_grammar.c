#include "nd_grammar.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------ JSON reader */

typedef struct {
    const char *p, *end;
} jr;

static void skip_ws(jr *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\t' ||
                             *j->p == '\n' || *j->p == '\r'))
        j->p++;
}

static int jeat(jr *j, char c)
{
    skip_ws(j);
    if (j->p < j->end && *j->p == c) {
        j->p++;
        return 1;
    }
    return 0;
}

/* Copy a JSON string into `out`. Handles the escapes a schema realistically
 * contains; \u is rejected rather than mangled. */
static int jstring(jr *j, char *out, size_t cap)
{
    size_t w = 0;

    skip_ws(j);
    if (j->p >= j->end || *j->p != '"')
        return 0;
    j->p++;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end)
                return 0;
            c = *j->p++;
            switch (c) {
            case 'n': c = '\n'; break;
            case 't': c = '\t'; break;
            case 'r': c = '\r'; break;
            case '"': case '\\': case '/': break;
            default: return 0;
            }
        }
        if (out) {
            if (w + 1 >= cap)
                return 0;
            out[w++] = c;
        }
    }
    if (j->p >= j->end)
        return 0;
    j->p++;
    if (out)
        out[w] = '\0';
    return 1;
}

static int jnumber(jr *j, double *out)
{
    char  *endp;
    double v;

    skip_ws(j);
    v = strtod(j->p, &endp);
    if (endp == j->p)
        return 0;
    j->p = endp;
    if (out)
        *out = v;
    return 1;
}

/* Skip any value, so unknown schema keys cost nothing. */
static int jskip(jr *j)
{
    skip_ws(j);
    if (j->p >= j->end)
        return 0;
    if (*j->p == '"')
        return jstring(j, NULL, 0);
    if (*j->p == '{' || *j->p == '[') {
        char open = *j->p, close = (open == '{') ? '}' : ']';
        int  depth = 0;
        while (j->p < j->end) {
            if (*j->p == '"') {
                if (!jstring(j, NULL, 0))
                    return 0;
                continue;
            }
            if (*j->p == open) depth++;
            else if (*j->p == close) {
                depth--;
                if (depth == 0) { j->p++; return 1; }
            }
            j->p++;
        }
        return 0;
    }
    while (j->p < j->end && *j->p != ',' && *j->p != '}' && *j->p != ']')
        j->p++;
    return 1;
}

int nd_json_compact(const char *src, size_t len, char *dst, size_t cap)
{
    size_t i, w = 0;
    int    in_str = 0, esc = 0;

    for (i = 0; i < len; i++) {
        char c = src[i];

        if (in_str) {
            if (w + 1 >= cap) return -1;
            dst[w++] = c;
            if (esc)             esc = 0;
            else if (c == '\\') esc = 1;
            else if (c == '"')   in_str = 0;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            continue;
        if (w + 1 >= cap) return -1;
        dst[w++] = c;
        if (c == '"')
            in_str = 1;
    }
    if (w >= cap) return -1;
    dst[w] = '\0';
    return (int)w;
}

/* ------------------------------------------------------------- compiling */

static int parse_type(const char *s, uint8_t *out)
{
    if (!strcmp(s, "string"))  { *out = ND_T_STRING;  return 1; }
    if (!strcmp(s, "integer")) { *out = ND_T_INTEGER; return 1; }
    if (!strcmp(s, "number"))  { *out = ND_T_NUMBER;  return 1; }
    if (!strcmp(s, "boolean")) { *out = ND_T_BOOLEAN; return 1; }
    return 0;
}

static int parse_prop(jr *j, nd_prop *p, const char **err)
{
    char key[64];

    p->type = ND_T_STRING;
    if (!jeat(j, '{')) { *err = "property is not an object"; return -1; }
    if (jeat(j, '}'))
        return 0;

    for (;;) {
        if (!jstring(j, key, sizeof(key))) { *err = "bad property key"; return -1; }
        if (!jeat(j, ':')) { *err = "expected ':'"; return -1; }

        if (!strcmp(key, "type")) {
            char t[32];
            if (!jstring(j, t, sizeof(t))) { *err = "bad type"; return -1; }
            if (!parse_type(t, &p->type)) {
                *err = "unsupported type (object/array/null not supported)";
                return -1;
            }
        } else if (!strcmp(key, "enum")) {
            if (!jeat(j, '[')) { *err = "enum is not an array"; return -1; }
            if (!jeat(j, ']')) {
                for (;;) {
                    if (p->n_enum >= ND_GR_MAX_ENUM) { *err = "too many enum values"; return -1; }
                    if (!jstring(j, p->enums[p->n_enum], ND_GR_STRLEN)) {
                        *err = "non-string enum value"; return -1;
                    }
                    p->n_enum++;
                    if (jeat(j, ',')) continue;
                    if (jeat(j, ']')) break;
                    *err = "malformed enum"; return -1;
                }
            }
        } else if (!strcmp(key, "minimum")) {
            if (!jnumber(j, &p->min)) { *err = "bad minimum"; return -1; }
            p->has_min = 1;
        } else if (!strcmp(key, "maximum")) {
            if (!jnumber(j, &p->max)) { *err = "bad maximum"; return -1; }
            p->has_max = 1;
        } else {
            if (!jskip(j)) { *err = "bad property value"; return -1; }
        }

        if (jeat(j, ',')) continue;
        if (jeat(j, '}')) break;
        *err = "malformed property"; return -1;
    }
    return 0;
}

static int find_prop(const nd_tool *t, const char *name)
{
    uint8_t i;
    for (i = 0; i < t->n_props; i++)
        if (!strcmp(t->props[i].name, name))
            return i;
    return -1;
}

static int parse_parameters(jr *j, nd_tool *t, const char **err)
{
    char key[64];

    if (!jeat(j, '{')) { *err = "parameters is not an object"; return -1; }
    if (jeat(j, '}'))
        return 0;

    for (;;) {
        if (!jstring(j, key, sizeof(key))) { *err = "bad parameters key"; return -1; }
        if (!jeat(j, ':')) { *err = "expected ':'"; return -1; }

        if (!strcmp(key, "properties")) {
            if (!jeat(j, '{')) { *err = "properties is not an object"; return -1; }
            if (!jeat(j, '}')) {
                for (;;) {
                    nd_prop *p;
                    if (t->n_props >= ND_GR_MAX_PROPS) { *err = "too many properties"; return -1; }
                    p = &t->props[t->n_props];
                    memset(p, 0, sizeof(*p));
                    if (!jstring(j, p->name, ND_GR_STRLEN)) { *err = "bad property name"; return -1; }
                    if (!jeat(j, ':')) { *err = "expected ':'"; return -1; }
                    if (parse_prop(j, p, err) != 0) return -1;
                    t->n_props++;
                    if (jeat(j, ',')) continue;
                    if (jeat(j, '}')) break;
                    *err = "malformed properties"; return -1;
                }
            }
        } else if (!strcmp(key, "required")) {
            /* Resolved after properties are known; stash and re-scan below. */
            const char *save = j->p;
            if (!jskip(j)) { *err = "bad required"; return -1; }
            {
                jr r = { save, j->end };
                if (jeat(&r, '[') && !jeat(&r, ']')) {
                    for (;;) {
                        char nm[ND_GR_STRLEN];
                        if (!jstring(&r, nm, sizeof(nm))) break;
                        {
                            int idx = find_prop(t, nm);
                            if (idx >= 0)
                                t->required |= (uint16_t)(1u << idx);
                        }
                        if (jeat(&r, ',')) continue;
                        break;
                    }
                }
            }
        } else {
            if (!jskip(j)) { *err = "bad parameters value"; return -1; }
        }

        if (jeat(j, ',')) continue;
        if (jeat(j, '}')) break;
        *err = "malformed parameters"; return -1;
    }
    return 0;
}

int nd_grammar_compile(nd_grammar *g, const char *tools_json, size_t len,
                       const char **err)
{
    jr          j = { tools_json, tools_json + len };
    const char *dummy = NULL;

    if (!err)
        err = &dummy;
    *err = NULL;
    memset(g, 0, sizeof(*g));

    if (!jeat(&j, '[')) { *err = "tools is not an array"; return -1; }
    if (jeat(&j, ']'))
        return 0;

    for (;;) {
        nd_tool *t;
        char     key[64];

        if (g->n_tools >= ND_GR_MAX_TOOLS) { *err = "too many tools"; return -1; }
        t = &g->tools[g->n_tools];
        memset(t, 0, sizeof(*t));

        if (!jeat(&j, '{')) { *err = "tool is not an object"; return -1; }
        for (;;) {
            if (!jstring(&j, key, sizeof(key))) { *err = "bad tool key"; return -1; }
            if (!jeat(&j, ':')) { *err = "expected ':'"; return -1; }
            if (!strcmp(key, "name")) {
                if (!jstring(&j, t->name, ND_GR_STRLEN)) { *err = "bad tool name"; return -1; }
            } else if (!strcmp(key, "parameters")) {
                if (parse_parameters(&j, t, err) != 0) return -1;
            } else {
                if (!jskip(&j)) { *err = "bad tool value"; return -1; }
            }
            if (jeat(&j, ',')) continue;
            if (jeat(&j, '}')) break;
            *err = "malformed tool"; return -1;
        }
        if (t->name[0] == '\0') { *err = "tool has no name"; return -1; }
        g->n_tools++;

        if (jeat(&j, ',')) continue;
        if (jeat(&j, ']')) break;
        *err = "malformed tools array"; return -1;
    }
    return 0;
}

/* -------------------------------------------------------------- matching */

void nd_gstate_init(nd_gstate *s, const nd_grammar *g)
{
    memset(s, 0, sizeof(*s));
    s->g = g;
    s->phase = ND_G_OFF;
}

void nd_gstate_open(nd_gstate *s)
{
    s->phase = ND_G_ARRAY_OPEN;
}

int nd_gstate_complete(const nd_gstate *s)
{
    return s->phase == ND_G_DONE;
}

/* Match against a fixed literal; completes when the whole literal is read. */
static int lit_byte(nd_gstate *s, char c, const char *lit, uint8_t next_phase)
{
    if (lit[s->lit_pos] != c)
        return 0;
    s->lit_pos++;
    if (lit[s->lit_pos] == '\0') {
        s->lit_pos = 0;
        s->phase = next_phase;
    }
    return 1;
}

/* Narrow a set of alternatives by one byte. Returns 1 if any survive. */
static int alt_byte(uint16_t *cand, uint8_t pos, char c,
                    const char (*opts)[ND_GR_STRLEN], uint8_t n)
{
    uint16_t next = 0;
    uint8_t  i;

    for (i = 0; i < n; i++) {
        if (!(*cand & (uint16_t)(1u << i)))
            continue;
        if (opts[i][pos] == c)
            next |= (uint16_t)(1u << i);
    }
    *cand = next;
    return next != 0;
}

/* Which alternative is uniquely finished at `pos`? -1 if none. */
static int alt_finished(uint16_t cand, uint8_t pos,
                        const char (*opts)[ND_GR_STRLEN], uint8_t n)
{
    uint8_t i;
    for (i = 0; i < n; i++)
        if ((cand & (uint16_t)(1u << i)) && opts[i][pos] == '\0')
            return i;
    return -1;
}

static int all_required_seen(const nd_tool *t, uint16_t seen)
{
    return (t->required & seen) == t->required;
}

/* Can the number built so far still reach a valid value? */
static int num_in_range(const nd_gstate *s, const nd_prop *p, int final)
{
    double v = s->num_neg ? -s->num_val : s->num_val;

    /* While digits are still arriving, only the upper bound can be violated
     * irrecoverably (appending digits only grows magnitude). */
    if (p->has_max && !s->num_neg && v > p->max)
        return 0;
    if (p->has_min && s->num_neg && v < p->min)
        return 0;
    if (final) {
        if (p->has_min && v < p->min) return 0;
        if (p->has_max && v > p->max) return 0;
    }
    return 1;
}

int nd_gstate_byte(nd_gstate *s, char c)
{
    const nd_grammar *g = s->g;
    const nd_tool    *t = (s->phase >= ND_G_ARGS_KEY) ? &g->tools[s->tool] : NULL;

    switch (s->phase) {
    case ND_G_OFF:
        return 1;                            /* reasoning is unconstrained */

    case ND_G_ARRAY_OPEN:
        if (c != '[') return 0;
        s->phase = ND_G_ARRAY_FIRST;
        return 1;

    case ND_G_ARRAY_FIRST:
        if (c == ']') { s->phase = ND_G_DONE; return 1; }   /* the empty call */
        if (c != '{') return 0;
        s->cand = (uint16_t)((1u << g->n_tools) - 1u);
        s->lit  = "\"name\":\"";
        s->lit_pos = 0;
        s->phase = ND_G_NAME_KEY;
        return 1;

    case ND_G_OBJ_OPEN:
        if (c != '{') return 0;
        s->cand = (uint16_t)((1u << g->n_tools) - 1u);
        s->lit  = "\"name\":\"";
        s->lit_pos = 0;
        s->phase = ND_G_NAME_KEY;
        return 1;

    case ND_G_NAME_KEY:
        if (!lit_byte(s, c, s->lit, ND_G_NAME_VAL))
            return 0;
        if (s->phase == ND_G_NAME_VAL)
            s->lit_pos = 0;
        return 1;

    case ND_G_NAME_VAL:
        if (c == '"') {
            int done = -1;
            uint8_t i;
            for (i = 0; i < g->n_tools; i++)
                if ((s->cand & (uint16_t)(1u << i)) &&
                    g->tools[i].name[s->lit_pos] == '\0') { done = i; break; }
            if (done < 0) return 0;
            s->tool = (uint8_t)done;
            s->seen = 0;
            s->lit = ",\"arguments\":{";
            s->lit_pos = 0;
            s->phase = ND_G_ARGS_KEY;
            return 1;
        }
        {
            uint16_t next = 0;
            uint8_t  i;
            for (i = 0; i < g->n_tools; i++) {
                if (!(s->cand & (uint16_t)(1u << i)))
                    continue;
                if (g->tools[i].name[s->lit_pos] == c)
                    next |= (uint16_t)(1u << i);
            }
            if (!next) return 0;
            s->cand = next;
            s->lit_pos++;
            return 1;
        }

    case ND_G_ARGS_KEY:
        if (!lit_byte(s, c, s->lit, ND_G_ARGS_FIRST))
            return 0;
        return 1;

    case ND_G_ARGS_FIRST:
    case ND_G_NEXT_PROP:
        /* ARGS_FIRST may close an empty argument object; NEXT_PROP follows a
         * comma, so a key is mandatory and a trailing comma is rejected. */
        if (c == '}') {
            if (s->phase != ND_G_ARGS_FIRST) return 0;
            if (!all_required_seen(t, s->seen)) return 0;
            s->phase = ND_G_AFTER_OBJ;
            return 1;
        }
        if (c != '"') return 0;
        s->cand = (uint16_t)(((1u << t->n_props) - 1u) & ~s->seen);
        if (!s->cand) return 0;
        s->lit_pos = 0;
        s->phase = ND_G_PROP_NAME;
        return 1;

    case ND_G_PROP_NAME:
        if (c == '"') {
            int idx = -1;
            uint8_t i;
            for (i = 0; i < t->n_props; i++)
                if ((s->cand & (uint16_t)(1u << i)) &&
                    t->props[i].name[s->lit_pos] == '\0') { idx = i; break; }
            if (idx < 0) return 0;
            s->prop = (uint8_t)idx;
            s->lit  = ":";
            s->lit_pos = 0;
            s->phase = ND_G_PROP_COLON;
            return 1;
        }
        {
            uint16_t next = 0;
            uint8_t  i;
            for (i = 0; i < t->n_props; i++) {
                if (!(s->cand & (uint16_t)(1u << i)))
                    continue;
                if (t->props[i].name[s->lit_pos] == c)
                    next |= (uint16_t)(1u << i);
            }
            if (!next) return 0;
            s->cand = next;
            s->lit_pos++;
            return 1;
        }

    case ND_G_PROP_COLON:
        if (c != ':') return 0;
        s->phase = ND_G_VAL_START;
        return 1;

    case ND_G_VAL_START: {
        const nd_prop *p = &t->props[s->prop];
        switch (p->type) {
        case ND_T_STRING:
            if (c != '"') return 0;
            s->lit_pos = 0;
            if (p->n_enum) {
                s->cand = (uint16_t)((1u << p->n_enum) - 1u);
                s->phase = ND_G_VAL_ENUM;
            } else {
                s->phase = ND_G_VAL_STR;
            }
            return 1;
        case ND_T_BOOLEAN:
            if (c == 't') { s->lit = "true";  s->lit_pos = 1; s->phase = ND_G_VAL_BOOL; return 1; }
            if (c == 'f') { s->lit = "false"; s->lit_pos = 1; s->phase = ND_G_VAL_BOOL; return 1; }
            return 0;
        case ND_T_INTEGER:
        case ND_T_NUMBER:
            s->num_val = 0;
            s->num_frac = 0;
            s->num_any = 0;
            s->num_neg = 0;
            s->num_scale = 0.1;
            if (c == '-') {
                if (p->has_min && p->min >= 0) return 0;
                s->num_neg = 1;
                s->phase = ND_G_VAL_NUM;
                return 1;
            }
            if (c < '0' || c > '9') return 0;
            s->num_val = c - '0';
            s->num_any = 1;
            if (!num_in_range(s, p, 0)) return 0;
            s->phase = ND_G_VAL_NUM;
            return 1;
        default:
            return 0;
        }
    }

    case ND_G_VAL_ENUM: {
        const nd_prop *p = &t->props[s->prop];
        if (c == '"') {
            if (alt_finished(s->cand, s->lit_pos, p->enums, p->n_enum) < 0)
                return 0;
            s->seen |= (uint16_t)(1u << s->prop);
            s->phase = ND_G_AFTER_VAL;
            return 1;
        }
        if (!alt_byte(&s->cand, s->lit_pos, c, p->enums, p->n_enum))
            return 0;
        s->lit_pos++;
        return 1;
    }

    case ND_G_VAL_STR:
        /* Free-form string: everything but a raw control char or a quote,
         * which terminates. Backslash escapes are not emitted by the model
         * for these fields and are rejected to keep the machine total. */
        if (c == '"') {
            s->seen |= (uint16_t)(1u << s->prop);
            s->phase = ND_G_AFTER_VAL;
            return 1;
        }
        if ((unsigned char)c < 0x20 || c == '\\')
            return 0;
        if (s->lit_pos < 255)
            s->lit_pos++;
        return 1;

    case ND_G_VAL_BOOL:
        if (s->lit[s->lit_pos] != c) return 0;
        s->lit_pos++;
        if (s->lit[s->lit_pos] == '\0') {
            s->seen |= (uint16_t)(1u << s->prop);
            s->phase = ND_G_AFTER_VAL;
        }
        return 1;

    case ND_G_VAL_NUM: {
        const nd_prop *p = &t->props[s->prop];
        if (c >= '0' && c <= '9') {
            if (s->num_frac) {
                s->num_val += (c - '0') * s->num_scale;
                s->num_scale *= 0.1;
            } else {
                s->num_val = s->num_val * 10.0 + (c - '0');
            }
            s->num_any = 1;
            if (!num_in_range(s, p, 0)) return 0;
            return 1;
        }
        if (c == '.') {
            if (p->type == ND_T_INTEGER || s->num_frac || !s->num_any) return 0;
            s->num_frac = 1;
            return 1;
        }
        /* Value ends: only a separator may follow, and the number must be
         * complete and in range. */
        if ((c == ',' || c == '}') && s->num_any && num_in_range(s, p, 1)) {
            s->seen |= (uint16_t)(1u << s->prop);
            if (c == ',') {
                if (!((((1u << t->n_props) - 1u) & ~s->seen))) return 0;
                s->phase = ND_G_NEXT_PROP;
                return 1;
            }
            if (!all_required_seen(t, s->seen)) return 0;
            s->phase = ND_G_AFTER_OBJ;
            return 1;
        }
        return 0;
    }

    case ND_G_AFTER_VAL:
        if (c == ',') {
            if (!((((1u << t->n_props) - 1u) & ~s->seen))) return 0;
            s->phase = ND_G_NEXT_PROP;
            return 1;
        }
        if (c == '}') {
            if (!all_required_seen(t, s->seen)) return 0;
            s->phase = ND_G_AFTER_OBJ;
            return 1;
        }
        return 0;

    case ND_G_AFTER_OBJ:
        /* The arguments object is closed; this '}' closes the call object. */
        if (c == '}') { s->phase = ND_G_CLOSE_CALL; return 1; }
        return 0;

    case ND_G_CLOSE_CALL:
        if (c == ',') { s->phase = ND_G_OBJ_OPEN; return 1; }
        if (c == ']') { s->phase = ND_G_DONE; return 1; }
        return 0;

    case ND_G_DONE:
        return 0;

    default:
        return 0;
    }
}
