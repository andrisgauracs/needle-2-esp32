/* nd_gtest - unit tests for the grammar machine.
 *
 * Feeds whole strings byte by byte and checks accept/reject, so the schema
 * enforcement is verified independently of the model.
 */
#include <stdio.h>
#include <string.h>

#include "nd_grammar.h"

static const char *SCHEMA =
"[{\"name\":\"set_led\",\"description\":\"Control the onboard RGB LED\","
"\"parameters\":{\"type\":\"object\",\"properties\":{"
"\"color\":{\"type\":\"string\",\"enum\":[\"red\",\"green\",\"blue\",\"yellow\",\"purple\",\"white\"]},"
"\"mode\":{\"type\":\"string\",\"enum\":[\"solid\",\"flash\",\"off\"]},"
"\"duration_seconds\":{\"type\":\"number\",\"minimum\":0.1,\"maximum\":60}},"
"\"required\":[\"color\",\"mode\"]}}]";

static int fails;

/* Returns 1 if the whole string is accepted and the call is complete. */
static int feed(const nd_grammar *g, const char *s)
{
    nd_gstate st;
    size_t    i;

    nd_gstate_init(&st, g);
    nd_gstate_open(&st);
    for (i = 0; s[i]; i++)
        if (!nd_gstate_byte(&st, s[i]))
            return 0;
    return nd_gstate_complete(&st);
}

static void expect(const nd_grammar *g, const char *s, int want, const char *why)
{
    int got = feed(g, s);
    if (got != want) {
        fails++;
        printf("  FAIL  %-13s %s\n        %s\n",
               want ? "(want ok)" : "(want rej)", why, s);
    } else {
        printf("  ok    %-13s %s\n", want ? "accepted" : "rejected", why);
    }
}

int main(void)
{
    nd_grammar  g;
    const char *err = NULL;

    if (nd_grammar_compile(&g, SCHEMA, strlen(SCHEMA), &err) != 0) {
        printf("compile failed: %s\n", err ? err : "?");
        return 1;
    }
    printf("compiled %u tool(s), %u props, required mask 0x%x\n\n",
           g.n_tools, g.tools[0].n_props, g.tools[0].required);

    printf("valid calls:\n");
    expect(&g, "[]", 1, "empty call (refusal)");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\"}}]",
           1, "both required props");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\","
           "\"duration_seconds\":2}}]", 1, "with optional number");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"mode\":\"off\",\"color\":\"white\"}}]",
           1, "props in any order");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"blue\",\"mode\":\"solid\","
           "\"duration_seconds\":0.5}}]", 1, "fractional number");

    printf("\ninvalid calls:\n");
    expect(&g, "[{\"name\":\"set_lights\",\"arguments\":{}}]", 0, "unknown tool name");
    expect(&g, "[{\"name\":\"set_onboard_lighting\",\"arguments\":{}}]", 0,
           "hallucinated tool (the bug we hit)");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"cyan\",\"mode\":\"flash\"}}]",
           0, "colour outside enum");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"blink\"}}]",
           0, "mode outside enum");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\"}}]",
           0, "missing required 'mode'");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{}}]",
           0, "missing both required");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\","
           "\"duration_seconds\":99}}]", 0, "number above maximum");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\","
           "\"brightness\":5}}]", 0, "undeclared property");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"color\":\"blue\","
           "\"mode\":\"off\"}}]", 0, "duplicate property");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\",}}]",
           0, "trailing comma");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\"}}",
           0, "unterminated array");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"mode\":\"flash\",\"color\":\"gre\"}}]",
           0, "enum prefix only");

    printf("\nmulti-call:\n");
    expect(&g, "[{\"name\":\"set_led\",\"arguments\":{\"color\":\"red\",\"mode\":\"flash\"}},"
           "{\"name\":\"set_led\",\"arguments\":{\"color\":\"blue\",\"mode\":\"off\"}}]",
           1, "two calls");

    printf("\n%s (%d failure%s)\n", fails ? "FAIL" : "PASS", fails,
           fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
