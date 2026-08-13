#!/usr/bin/env python3
"""Validate the C tokenizer against real SentencePiece.

The C side reads the RAW blob embedded in needle2.cact; the reference side
loads tokenizer.model from the same HF repo. They must agree id-for-id,
including on the chat markers used by the prompt template.

  python3 tools/check_tokenizer.py model/needle2.cact build/nd_dump
"""
import subprocess
import sys

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOLS_START, TOOLS_END = "<tools>", "</tools>"
TOOL_CALL_START, TOOL_CALL_END = "<tool_call>", "</tool_call>"
THINK_START, THINK_END = "<think>", "</think>"

TOOLS_JSON = ('[{"name":"set_lights","description":"Turn a room\'s lights on or off",'
              '"parameters":{"type":"object","properties":{"room":{"type":"string"},'
              '"on":{"type":"boolean"},"brightness":{"type":"integer"}},'
              '"required":["room","on"]}}]')

CASES = [
    "hello world",
    "dim the living room to 30",
    "what's it like in Lagos right now?",
    "  leading and trailing  ",
    "Invoice from Acme Corp, $1,200.00, due 2026-09-01",
    "unicode: café — 東京 — 🎉",
    "",
    "a",
    " ",
    "\n",
    "tabs\tand\nnewlines",
    "digits 0123456789 and symbols !@#$%^&*()_+-=[]{}|;':\",./<>?",
    THINK_START + "'living room' -> room" + THINK_END,
    TOOL_CALL_START + '[{"name":"set_lights","arguments":{"room":"living room"}}]'
    + TOOL_CALL_END,
    # The full rendered prompt, which is what actually matters.
    IM_START + "user\n" + TOOLS_START + TOOLS_JSON + TOOLS_END
    + "\ndim the living room to 30" + IM_END + "\n" + IM_START + "assistant\n",
]


def c_encode(binary, blob, text):
    r = subprocess.run([binary, blob, "tok", text], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return [int(x) for x in r.stdout.split()]


def c_decode(binary, blob, ids):
    r = subprocess.run([binary, blob, "detok", *map(str, ids)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def main():
    blob, binary = sys.argv[1], sys.argv[2]

    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    model = hf_hub_download("Cactus-Compute/needle2", "tokenizer/tokenizer.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(model)
    print(f"sentencepiece vocab: {sp.GetPieceSize()}")

    fails = 0
    for text in CASES:
        want = sp.Encode(text, out_type=int)
        got = c_encode(binary, blob, text)
        label = repr(text if len(text) < 48 else text[:45] + "...")
        if want != got:
            fails += 1
            print(f"MISMATCH {label}\n  sp: {want}\n   c: {got}")
        else:
            # Round-trip through the C decoder too.
            back = c_decode(binary, blob, got)
            note = "" if back == sp.Decode(want) else f"  [decode differs: {back!r}]"
            print(f"ok  {len(got):4d} ids  {label}{note}")

    print("\nPASS" if fails == 0 else f"\nFAIL ({fails} mismatches)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
