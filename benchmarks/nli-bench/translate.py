#!/usr/bin/env python3
"""Produce cached English renderings (content_en) for bench pairs via Haiku.

Mirrors the proposed prod design: translate only non-English prose, preserve
CML prefixes/symbols/identifiers, keep line structure (line-count loss guard,
one retry, fall back to original on repeated failure). Cache keyed by sha256
so re-runs are free.

Usage: /root/venvs/ai/bin/python translate.py [pairs.jsonl synth_pairs.jsonl ...]
Deps: requests; ANTHROPIC_API_KEY in /root/.secrets/anthropic.env
"""

import hashlib
import json
import os
import re
import sys
import time

import requests

API_URL = "https://api.anthropic.com/v1/chat/completions"
MODEL = "claude-haiku-4-5"
CACHE = "translations.json"

SWEDISH_HINT = re.compile(
    r"[åäöÅÄÖ]|\b(och|inte|som|för|med|det|den|har|ska|kan|hon|han|hos|när|"
    r"efter|innan|redan|behöver|kommer|gjort|vill|bara|också|mycket)\b"
)

PROMPT = """Translate this memory record to English.

Rules:
- Preserve the line structure exactly: same number of lines, same order.
- Do NOT translate or alter: CML prefixes (F:, D:, C:, L:, P:, W:, R:), symbols (→ ↔ ∴ ⚠ ✓ ✗), names of people/products/places, file paths, code, commands, numbers, dates, version strings, technical identifiers.
- Translate ONLY the natural-language prose. Text already in English stays untouched.
- Output ONLY the translated record, nothing else.

Record:
{text}"""


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:24]


def needs_translation(t):
    return bool(SWEDISH_HINT.search(t))


def load_key():
    for line in open("/root/.secrets/anthropic.env"):
        if line.startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ANTHROPIC_API_KEY not found")


def translate(key, text):
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    files = sys.argv[1:] or ["pairs.jsonl", "synth_pairs.jsonl"]
    cache = {}
    if os.path.exists(CACHE):
        cache = json.load(open(CACHE))
    key = load_key()

    texts = {}
    for fn in files:
        if not os.path.exists(fn):
            continue
        for line in open(fn, encoding="utf-8"):
            p = json.loads(line)
            for side in ("a_content", "b_content"):
                t = p[side]
                texts[sha(t)] = t

    todo = {h: t for h, t in texts.items()
            if h not in cache and needs_translation(t)}
    print(f"{len(texts)} unique texts, {len(todo)} need translation, "
          f"{len(cache)} cached", file=sys.stderr)

    for i, (h, t) in enumerate(todo.items()):
        ok = False
        for attempt in (1, 2):
            try:
                out = translate(key, t)
                if len(out.split("\n")) == len(t.split("\n")):
                    cache[h] = out
                    ok = True
                    break
                print(f"  {h}: line-count mismatch attempt {attempt}", file=sys.stderr)
            except Exception as e:
                print(f"  {h}: {e} (attempt {attempt})", file=sys.stderr)
                time.sleep(2)
        if not ok:
            cache[h] = t  # loss-guard fallback: keep original
            print(f"  {h}: FALLBACK to original", file=sys.stderr)
        if (i + 1) % 10 == 0:
            json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
            print(f"  progress {i + 1}/{len(todo)}", file=sys.stderr)

    json.dump(cache, open(CACHE, "w"), ensure_ascii=False)
    print(f"done; cache size {len(cache)}", file=sys.stderr)


if __name__ == "__main__":
    main()
