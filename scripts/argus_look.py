"""
Argus LOOK — ask a question about one image (or a set) using either backend.

Two-tier on purpose, because the benchmark says so:
  local (ornith:9b) : FREE and private, ~20 s/image on this Mac. Best local quality,
                      caught details the other local models missed and invented nothing.
                      Right for bulk triage across hundreds of frames.
  api (gemini)      : faster per image and better on fine judgement. Costs money and
                      quota. Right for the call that actually matters.

Measured on a 704 px previz crop: ornith:9b named head, legs, tail and shadow correctly;
qwen2.5vl:7b MISSED the tail; minicpm-v claimed "no shadow cast" when there was one. That is
why local is not a straight replacement for the API — a weaker eye that asserts something
false is worse than a slower one that says less.

Usage:
  python argus_look.py <image|dir> "<question>" [--backend local|api|both] [--model NAME]
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

LOCAL_URL = "http://localhost:11434/api/chat"
LOCAL_DEFAULT = "ornith:9b"
BASE = "https://generativelanguage.googleapis.com"


def keys():
    out = []
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY_BACKUP"):
        v = os.environ.get(env)
        if v:
            out.append(v.strip())
    for path, var in ((os.path.expanduser("~/.hermes/.env"), "GOOGLE_API_KEY"),
                      (os.path.expanduser("~/.hermes/secrets/gemini-backup.env"),
                       "GEMINI_API_KEY_BACKUP")):
        if os.path.exists(path):
            for line in open(path):
                if line.startswith(var + "="):
                    v = line.split("=", 1)[1].strip().strip('"')
                    if v and v not in out:
                        out.append(v)
    return out


def ask_local(path, question, model):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    payload = {"model": model, "stream": False,
               "messages": [{"role": "user", "content": question, "images": [b64]}]}
    req = urllib.request.Request(LOCAL_URL, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    return (d.get("message") or {}).get("content", "")


def ask_api(path, question, model):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    parts = [{"text": question},
             {"inline_data": {"mime_type": mime, "data": b64}}]
    payload = {"contents": [{"role": "user", "parts": parts}]}
    last = None
    for k in keys():
        for attempt in range(3):
            req = urllib.request.Request(
                "%s/v1beta/models/%s:generateContent?key=%s" % (BASE, model, k),
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    d = json.loads(r.read().decode())
                txt = ""
                for c in d.get("candidates", []):
                    for p in c.get("content", {}).get("parts", []):
                        txt += p.get("text", "")
                return txt
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last = "HTTP %d: %s" % (e.code, body[:200])
                if e.code in (500, 502, 503, 504) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                if e.code in (429, 401, 403):
                    break
                return "API error: " + last
            except Exception as e:
                return "look failed: %s: %s" % (type(e).__name__, e)
    return "API error: " + str(last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("question")
    ap.add_argument("--backend", default="local", choices=["local", "api", "both"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=1, help="max images if given a directory")
    a = ap.parse_args()

    imgs = ([os.path.join(a.image, f) for f in sorted(os.listdir(a.image))
             if f.lower().endswith((".png", ".jpg", ".jpeg"))][:a.limit]
            if os.path.isdir(a.image) else [a.image])

    for p in imgs:
        print("\n" + "=" * 72)
        print("IMAGE: %s" % os.path.basename(p))
        print("=" * 72)
        for backend in (("local", "api") if a.backend == "both" else (a.backend,)):
            t0 = time.time()
            try:
                if backend == "local":
                    txt = ask_local(p, a.question, a.model or LOCAL_DEFAULT)
                else:
                    txt = ask_api(p, a.question, a.model or "gemini-3.5-flash")
            except Exception as e:
                txt = "FAILED: %s: %s" % (type(e).__name__, e)
            print("\n--- %s (%.1fs) ---" % (backend, time.time() - t0))
            print(txt[:1400] if txt else "(empty)")


if __name__ == "__main__":
    main()
