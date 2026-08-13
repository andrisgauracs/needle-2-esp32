/* nd_dump - host-side inspector used to validate the C reader against the
 * Python reference implementation.
 *
 *   nd_dump <model.cact> header
 *   nd_dump <model.cact> dir [n]
 *   nd_dump <model.cact> row <tensor> <row>     dequantised row, one float/line
 *   nd_dump <model.cact> gemv <tensor> <seed>   W @ x for a seeded x
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "nd_cact.h"
#include "nd_quant.h"
#include "nd_tokenizer.h"
#include "nd_model.h"
#include "nd_grammar.h"
#include "nd_sample.h"

static void *slurp(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    void *buf;
    long  n;

    if (!f) {
        perror(path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    n = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "read failed\n");
        exit(1);
    }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

/* Same generator as tools/check_engine.py so both sides see identical input. */
static void seeded_vec(float *x, uint32_t n, uint32_t seed)
{
    uint32_t s = seed * 1103515245u + 12345u;
    uint32_t i;
    for (i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = (float)((int32_t)(s >> 8) % 2000 - 1000) / 1000.0f;
    }
}

int main(int argc, char **argv)
{
    size_t   size;
    void    *blob;
    nd_cact  c;
    int      rc;

    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.cact> <header|dir|row|gemv> ...\n", argv[0]);
        return 2;
    }

    blob = slurp(argv[1], &size);
    rc   = nd_cact_open(&c, blob, size);
    if (rc != 0) {
        fprintf(stderr, "nd_cact_open failed: %d\n", rc);
        return 1;
    }

    if (!strcmp(argv[2], "header")) {
        const nd_cact_header *h = &c.h;
        uint32_t i;
        printf("tag 0x%08X\n", h->tag);
        printf("num_tensors %u\n", h->num_tensors);
        printf("codebook_len %u\n", h->codebook_len);
        printf("kv_window %u\n", h->kv_window);
        printf("kv_bits %u\n", h->kv_bits);
        printf("vocab_size %u\n", h->vocab_size);
        printf("d_model %u\n", h->d_model);
        printf("num_heads %u\n", h->num_heads);
        printf("num_kv_heads %u\n", h->num_kv_heads);
        printf("num_layers %u\n", h->num_layers);
        printf("head_dim %u\n", h->head_dim);
        printf("max_seq_len %u\n", h->max_seq_len);
        printf("attn_dim %u\n", h->attn_dim);
        printf("mhc_lanes %u\n", h->mhc_lanes);
        printf("engram_slots %u\n", h->engram_slots);
        printf("engram_sub_dim %u\n", h->engram_sub_dim);
        printf("engram_conv_taps %u\n", h->engram_conv_taps);
        printf("engram_tables %u\n", h->engram_tables);
        printf("engram_dilation %u\n", h->engram_dilation);
        printf("num_orders %u\n", h->num_orders);
        for (i = 0; i < h->num_orders; i++)
            printf("order[%u] %u\n", i, h->orders[i]);
        printf("num_sites %u\n", h->num_sites);
        for (i = 0; i < h->num_sites; i++)
            printf("site[%u] %u\n", i, h->sites[i]);
        printf("rope_theta %g\n", (double)h->rope_theta);
        return 0;
    }

    if (!strcmp(argv[2], "dir")) {
        uint32_t n = (argc > 3) ? (uint32_t)atoi(argv[3]) : c.n;
        uint32_t i;
        if (n > c.n)
            n = c.n;
        for (i = 0; i < n; i++) {
            nd_tensor t;
            nd_cact_tensor(&c, i, &t);
            printf("%u dt=%u ndim=%u shape=%u,%u off=%llu nbytes=%llu group=%u bits=%u\n",
                   i, t.dtype, t.ndim, t.shape[0], t.shape[1],
                   (unsigned long long)t.offset, (unsigned long long)t.nbytes,
                   t.group, t.bits);
        }
        return 0;
    }

    if (!strcmp(argv[2], "row") && argc >= 5) {
        nd_tensor t;
        uint32_t  idx = (uint32_t)atoi(argv[3]);
        uint32_t  row = (uint32_t)atoi(argv[4]);
        float    *scratch, *w;
        uint32_t  j;

        nd_cact_tensor(&c, idx, &t);
        if (t.dtype != ND_DT_CQ) {
            fprintf(stderr, "tensor %u is not CQ\n", idx);
            return 1;
        }
        scratch = malloc(nd_cq_scratch(&t) * sizeof(float));
        w       = malloc(t.shape[1] * sizeof(float));
        nd_cq_dequant_row(&c, &t, nd_cact_data(&c, &t), row, scratch, w);
        for (j = 0; j < t.shape[1]; j++)
            printf("%.8g\n", (double)w[j]);
        return 0;
    }

    if (!strcmp(argv[2], "gemv") && argc >= 5) {
        nd_tensor t;
        uint32_t  idx  = (uint32_t)atoi(argv[3]);
        uint32_t  seed = (uint32_t)atoi(argv[4]);
        float    *x, *scratch, *y;
        uint32_t  j;

        nd_cact_tensor(&c, idx, &t);
        if (t.dtype != ND_DT_CQ) {
            fprintf(stderr, "tensor %u is not CQ\n", idx);
            return 1;
        }
        x       = malloc(t.shape[1] * sizeof(float));
        scratch = malloc(nd_cq_scratch(&t) * sizeof(float));
        y       = malloc(t.shape[0] * sizeof(float));
        seeded_vec(x, t.shape[1], seed);
        nd_cq_gemv(&c, &t, nd_cact_data(&c, &t), x, scratch, y);
        for (j = 0; j < t.shape[0]; j++)
            printf("%.8g\n", (double)y[j]);
        return 0;
    }

    if (!strcmp(argv[2], "genp") && argc >= 5) {
        /* genp <tools.json> <query> [max_new] - same as `gen`, but emits the
         * line protocol the TUI speaks, so the host engine and the ESP32
         * firmware are interchangeable behind it. */
        nd_model  m;
        char     *tools;
        size_t    tools_len;
        char     *prompt;
        uint32_t *ids;
        int       n;
        uint32_t  max_new = (argc > 5) ? (uint32_t)atoi(argv[5]) : 96;
        uint32_t  i, produced = 0;
        clock_t   t0;
        double    pre_s;
        nd_grammar gram;
        nd_sampler smp;
        static char gen[2048];
        size_t      gw = 0;

        tools = (char *)slurp(argv[3], &tools_len);
        {   /* Formatting of the schema file must not reach the model. */
            char *packed = (char *)malloc(tools_len + 1);
            int   pn = nd_json_compact(tools, tools_len, packed, tools_len + 1);
            if (pn < 0) { printf("ERR schema_too_large\n"); return 1; }
            tools = packed;
            tools_len = (size_t)pn;
        }

        prompt = (char *)malloc(tools_len + strlen(argv[4]) + 256);
        sprintf(prompt,
                "<|im_start|>user\n<tools>%.*s</tools>\n%s<|im_end|>\n"
                "<|im_start|>assistant\n",
                (int)tools_len, tools, argv[4]);

        if (nd_model_open(&m, blob, size) != 0) {
            printf("ERR open\n");
            return 1;
        }
        {
            const char *gerr = NULL;
            if (nd_grammar_compile(&gram, tools, tools_len, &gerr) != 0) {
                printf("ERR grammar %s\n", gerr ? gerr : "?");
                return 1;
            }
        }
        nd_sampler_init(&smp, &m.tok, &gram);
        printf("EVT ready model=needle2 layers=%u d_model=%u tools=%u\n",
               m.n_layers, m.d_model, gram.n_tools);
        fflush(stdout);

        /* Split at the </tools> marker: the prefix is identical for every
         * request, so it is prefilled once and reused. */
        {
            char pre[4096], suf[1024];
            uint32_t npre, nsuf;
            int rc2;

            sprintf(pre, "<|im_start|>user\n<tools>%.*s</tools>", (int)tools_len, tools);
            sprintf(suf, "\n%s<|im_end|>\n<|im_start|>assistant\n", argv[4]);

            ids = (uint32_t *)malloc(sizeof(uint32_t) * 4096);
            ids[0] = ND_BOS_ID;
            rc2 = nd_tok_encode(&m.tok, pre, strlen(pre), ids + 1, 4095);
            if (rc2 < 0) { printf("ERR prompt_too_long\n"); return 1; }
            npre = (uint32_t)rc2 + 1;
            rc2 = nd_tok_encode_ex(&m.tok, suf, strlen(suf), ids + npre,
                                   4096 - npre, 0);
            if (rc2 < 0) { printf("ERR prompt_too_long\n"); return 1; }
            nsuf = (uint32_t)rc2;
            n = (int)(npre + nsuf);

            t0 = clock();
            {
                const float *lg = NULL;
                for (i = 0; i < npre; i++)
                    lg = nd_model_step_hidden(&m, ids[i]);
                nd_model_snapshot(&m);
                printf("EVT prefix tokens=%u ms=%.0f (cached)\n", npre,
                       (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0);
                for (i = npre; i < (uint32_t)n; i++)
                    lg = nd_model_step_hidden(&m, ids[i]);
            pre_s = (double)(clock() - t0) / CLOCKS_PER_SEC;

            printf("EVT prefill tokens=%d ms=%.0f tps=%.2f sink=%u\n",
                   n, pre_s * 1000.0, n / pre_s, m.n_sink);
            fflush(stdout);

            t0 = clock();

            /* Optional: skip the reasoning block by force-feeding an empty
             * <think></think> and opening the call directly. The reasoning is
             * ~2/3 of the generated tokens, which at ~1 tok/s is most of the
             * wall clock. */
            if (argc > 6 && !strcmp(argv[6], "nothink")) {
                static const char *nl = "\n";
                uint32_t forced[8];
                uint32_t nf = 0, fi;
                forced[nf++] = ND_THINK_START_ID;
                forced[nf++] = ND_THINK_END_ID;
                {
                    uint32_t tmp[4];
                    int tn = nd_tok_encode(&m.tok, nl, 1, tmp, 4);
                    for (fi = 0; fi < (uint32_t)tn; fi++)
                        forced[nf++] = tmp[fi];
                }
                forced[nf++] = ND_TOOL_CALL_START_ID;
                for (fi = 0; fi < nf; fi++) {
                    char piece[256], esc[600];
                    uint32_t k, w = 0;
                    nd_sample_accept(&smp, forced[fi]);
                    nd_tok_decode_ex(&m.tok, &forced[fi], 1, piece, sizeof(piece), 0);
                    for (k = 0; piece[k] && w < sizeof(esc) - 3; k++) {
                        if (piece[k] == '\n') { esc[w++] = '\\'; esc[w++] = 'n'; }
                        else if (piece[k] != '\r') esc[w++] = piece[k];
                    }
                    esc[w] = '\0';
                    printf("TOK %s\n", esc);
                    if (gw + strlen(piece) < sizeof(gen) - 1) {
                        strcpy(gen + gw, piece);
                        gw += strlen(piece);
                    }
                    produced++;
                    lg = nd_model_step_hidden(&m, forced[fi]);
                }
            }

            for (i = 0; i < max_new; i++) {
                uint32_t best;
                char     piece[512], esc[1100];
                uint32_t k, w = 0;

                best = nd_sample_hidden(&m, &smp, lg);
                if (best == (uint32_t)-1) {
                    printf("ERR no_legal_token\n");
                    break;
                }
                if (best == ND_EOS_ID || best == ND_IM_END_ID)
                    break;
                nd_sample_accept(&smp, best);

                nd_tok_decode_ex(&m.tok, &best, 1, piece, sizeof(piece), 0);
                /* Escape newlines so one token is always one line. */
                for (k = 0; piece[k] && w < sizeof(esc) - 3; k++) {
                    if (piece[k] == '\n') { esc[w++] = '\\'; esc[w++] = 'n'; }
                    else if (piece[k] == '\r') { continue; }
                    else esc[w++] = piece[k];
                }
                esc[w] = '\0';
                printf("TOK %s\n", esc);
                fflush(stdout);
                if (gw + strlen(piece) < sizeof(gen) - 1) {
                    strcpy(gen + gw, piece);
                    gw += strlen(piece);
                }
                produced++;
                lg = nd_model_step_hidden(&m, best);
            }
        }
        }
        {
            double dec_s = (double)(clock() - t0) / CLOCKS_PER_SEC;
            float  conf = nd_model_confidence(&m);
            if (conf >= 0.0f)
                printf("CONF %.4f\n", (double)conf);
            printf("EVT done tokens=%u ms=%.0f tps=%.2f\n",
                   produced, dec_s * 1000.0,
                   produced ? produced / dec_s : 0.0);

            /* Mirror the firmware's ACT line so both backends speak the exact
             * same protocol and the TUI needs no special cases. */
            {
                const char *call = strstr(gen, "<tool_call>");
                char        color[32] = "", mode[16] = "";
                const char *pc, *pm, *pd;

                pc = call ? strstr(call, "\"color\"") : NULL;
                pm = call ? strstr(call, "\"mode\"") : NULL;
                if (pc && pm) {
                    sscanf(pc, "\"color\":\"%31[^\"]", color);
                    sscanf(pm, "\"mode\":\"%15[^\"]", mode);
                }
                pd = call ? strstr(call, "\"duration_seconds\"") : NULL;
                if (color[0] && mode[0])
                    printf("ACT led color=%s mode=%s duration=%.1f\n", color, mode,
                           pd ? atof(strchr(pd, ':') + 1) : 2.0);
                else
                    printf("ACT none\n");
            }
        }
        fflush(stdout);
        nd_model_close(&m);
        return 0;
    }

    if (!strcmp(argv[2], "gen") && argc >= 5) {
        /* gen <tools.json> <query> [max_new] - render the chat template,
         * prefill, then greedy decode. */
        nd_model  m;
        char     *tools;
        size_t    tools_len;
        char     *prompt;
        uint32_t *ids;
        int       n;
        uint32_t  max_new = (argc > 5) ? (uint32_t)atoi(argv[5]) : 64;
        uint32_t  i;
        clock_t   t0;

        tools = (char *)slurp(argv[3], &tools_len);
        {   /* Formatting of the schema file must not reach the model. */
            char *packed = (char *)malloc(tools_len + 1);
            int   pn = nd_json_compact(tools, tools_len, packed, tools_len + 1);
            if (pn < 0) { printf("ERR schema_too_large\n"); return 1; }
            tools = packed;
            tools_len = (size_t)pn;
        }

        prompt = (char *)malloc(tools_len + strlen(argv[4]) + 256);
        sprintf(prompt,
                "<|im_start|>user\n<tools>%.*s</tools>\n%s<|im_end|>\n"
                "<|im_start|>assistant\n",
                (int)tools_len, tools, argv[4]);

        if (nd_model_open(&m, blob, size) != 0) {
            fprintf(stderr, "nd_model_open failed\n");
            return 1;
        }

        ids = (uint32_t *)malloc(sizeof(uint32_t) * 4096);
        ids[0] = ND_BOS_ID;
        n = nd_tok_encode(&m.tok, prompt, strlen(prompt), ids + 1, 4095);
        if (n < 0) {
            fprintf(stderr, "prompt too long\n");
            return 1;
        }
        n += 1;
        fprintf(stderr, "prompt: %d tokens\n", n);

        t0 = clock();
        {
            const float *lg = NULL;
            for (i = 0; i < (uint32_t)n; i++)
                lg = nd_model_step(&m, ids[i]);
            fprintf(stderr, "prefill: %.2f s (%.1f tok/s)\n",
                    (double)(clock() - t0) / CLOCKS_PER_SEC,
                    n / ((double)(clock() - t0) / CLOCKS_PER_SEC));

            t0 = clock();
            for (i = 0; i < max_new; i++) {
                uint32_t best = 0, j;
                float    bv = -1e30f;
                char     piece[512];
                uint16_t plen;
                const char *sp;

                for (j = 0; j < m.vocab; j++)
                    if (lg[j] > bv) { bv = lg[j]; best = j; }

                if (best == ND_EOS_ID || best == ND_IM_END_ID)
                    break;

                sp = nd_tok_piece(&m.tok, best, &plen);
                if (sp && plen < sizeof(piece)) {
                    uint32_t one = best;
                    nd_tok_decode_ex(&m.tok, &one, 1, piece, sizeof(piece), 0);
                    fputs(piece, stdout);
                    fflush(stdout);
                }
                lg = nd_model_step_hidden(&m, best);
            }
        }
        printf("\n");
        fprintf(stderr, "decode: %.2f s\n",
                (double)(clock() - t0) / CLOCKS_PER_SEC);
        nd_model_close(&m);
        return 0;
    }

    if (!strcmp(argv[2], "split") && argc >= 5) {
        /* Verify prefix+suffix encodes identically to the whole prompt. */
        nd_model m;
        char *tools; size_t tl;
        char full[2048], pre[2048], suf[512];
        uint32_t a[1024], b[1024];
        int na, nb, nb2, i, ok = 1;

        tools = (char *)slurp(argv[3], &tl);
        while (tl && (tools[tl-1]=='\n'||tools[tl-1]==' ')) tl--;
        if (nd_model_open(&m, blob, size) != 0) return 1;

        sprintf(pre, "<|im_start|>user\n<tools>%.*s</tools>", (int)tl, tools);
        sprintf(suf, "\n%s<|im_end|>\n<|im_start|>assistant\n", argv[4]);
        sprintf(full, "%s%s", pre, suf);

        na  = nd_tok_encode(&m.tok, full, strlen(full), a, 1024);
        nb  = nd_tok_encode(&m.tok, pre, strlen(pre), b, 1024);
        nb2 = nd_tok_encode_ex(&m.tok, suf, strlen(suf), b + nb, 1024 - nb, 0);
        printf("joint=%d  prefix=%d + suffix=%d = %d\n", na, nb, nb2, nb + nb2);
        if (na != nb + nb2) ok = 0;
        else for (i = 0; i < na; i++) if (a[i] != b[i]) { ok = 0; printf("mismatch at %d: %u vs %u\n", i, a[i], b[i]); break; }
        printf("%s\n", ok ? "IDENTICAL" : "DIFFERENT");
        return ok ? 0 : 1;
    }

    if (!strcmp(argv[2], "logits")) {
        nd_model m;
        uint32_t i, j;

        if (nd_model_open(&m, blob, size) != 0) {
            fprintf(stderr, "nd_model_open failed\n");
            return 1;
        }
        for (i = 3; i < (uint32_t)argc; i++) {
            const float *lg = nd_model_step(&m, (uint32_t)atoi(argv[i]));
            for (j = 0; j < m.vocab; j++)
                printf("%.6f\n", (double)lg[j]);
        }
        nd_model_close(&m);
        return 0;
    }

    if (!strcmp(argv[2], "tok") || !strcmp(argv[2], "detok")) {
        nd_tokenizer  tk;
        nd_tensor     t;
        uint32_t      i, found = 0;

        /* The tokenizer is the single RAW tensor, last in canon order. */
        for (i = c.n; i-- > 0;) {
            nd_cact_tensor(&c, i, &t);
            if (t.dtype == ND_DT_RAW) { found = 1; break; }
        }
        if (!found) {
            fprintf(stderr, "no tokenizer tensor\n");
            return 1;
        }
        if (nd_tok_init(&tk, nd_cact_data(&c, &t), (size_t)t.nbytes) != 0) {
            fprintf(stderr, "nd_tok_init failed\n");
            return 1;
        }

        if (!strcmp(argv[2], "tok")) {
            uint32_t ids[4096];
            int      n;
            if (argc < 4) { fprintf(stderr, "tok needs text\n"); return 2; }
            n = nd_tok_encode(&tk, argv[3], strlen(argv[3]), ids, 4096);
            if (n < 0) { fprintf(stderr, "encode overflow\n"); return 1; }
            for (i = 0; i < (uint32_t)n; i++)
                printf("%u\n", ids[i]);
        } else {
            uint32_t ids[4096];
            uint32_t n = 0;
            char     out[16384];
            for (i = 3; i < (uint32_t)argc && n < 4096; i++)
                ids[n++] = (uint32_t)atoi(argv[i]);
            if (nd_tok_decode(&tk, ids, n, out, sizeof(out)) < 0) {
                fprintf(stderr, "decode overflow\n");
                return 1;
            }
            printf("%s", out);
        }
        nd_tok_free(&tk);
        return 0;
    }

    fprintf(stderr, "unknown command\n");
    return 2;
}
