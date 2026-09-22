"""
Argus NATIVE VIDEO path — send the clip itself to Gemini at a chosen frame rate.

Why this exists. Argus currently extracts stills and sends JPEGs. Measured consequence: on a
3-shot previz where shot 2's camera demonstrably sweeps 3.8 m sideways, stills-based analysis
reported "the camera remains completely static" — because a vision model cannot infer motion
from sparse stills and it guesses instead. Gemini can ingest the video file itself at a
requested frame rate (`video_metadata.fps`), so the sampling decision stops being ours.

Flow (all REST, no SDK dependency):
  1. resumable upload to /upload/v1beta/files   -> x-goog-upload-url header
  2. upload+finalize the bytes                  -> file.uri
  3. poll the file until state == ACTIVE
  4. POST /v1beta/models/<model>:generateContent with the video part + video_metadata.fps

Usage:
  python argus_native_video.py <video> "<question>" [--fps 24] [--model gemini-3.5-flash]
                               [--start 0] [--end 4] [--agentic]
Prints the model's answer, plus token usage when the API reports it.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://generativelanguage.googleapis.com"
KEY_FILES = [os.path.expanduser("~/.hermes/.env"),
             os.path.expanduser("~/.hermes/secrets/gemini-backup.env")]


def keys():
    """Primary first, then the backup: a 429 is a per-PROJECT cap, so a second key serves."""
    out = []
    for env in ("GOOGLE_API_KEY", "GEMINI_API_KEY_BACKUP"):
        v = os.environ.get(env)
        if v:
            out.append(v.strip())
    for path, var in ((KEY_FILES[0], "GOOGLE_API_KEY"), (KEY_FILES[1], "GEMINI_API_KEY_BACKUP")):
        if os.path.exists(path):
            for line in open(path):
                if line.startswith(var + "="):
                    v = line.split("=", 1)[1].strip()
                    if v and v not in out:
                        out.append(v)
    return out


def upload(path, key):
    size = os.path.getsize(path)
    mime = "video/mp4" if path.lower().endswith(".mp4") else "video/quicktime"
    req = urllib.request.Request(
        "%s/upload/v1beta/files?key=%s" % (BASE, key), data=b'{"file":{"display_name":"argus"}}',
        headers={"X-Goog-Upload-Protocol": "resumable", "X-Goog-Upload-Command": "start",
                 "X-Goog-Upload-Header-Content-Length": str(size),
                 "X-Goog-Upload-Header-Content-Type": mime,
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        up_url = r.headers.get("x-goog-upload-url")
    if not up_url:
        raise SystemExit("no upload url returned")
    body = open(path, "rb").read()
    req = urllib.request.Request(
        up_url, data=body,
        headers={"Content-Length": str(len(body)), "X-Goog-Upload-Offset": "0",
                 "X-Goog-Upload-Command": "upload, finalize"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        info = json.loads(r.read().decode())
    f = info.get("file", {})
    name, uri = f.get("name"), f.get("uri")
    for _ in range(60):
        req = urllib.request.Request("%s/v1beta/%s?key=%s" % (BASE, name, key))
        with urllib.request.urlopen(req, timeout=60) as r:
            st = json.loads(r.read().decode()).get("state")
        if st == "ACTIVE":
            return uri, name
        if st == "FAILED":
            raise SystemExit("file processing FAILED")
        time.sleep(3)
    raise SystemExit("file never became ACTIVE")


def ask(uri, mime, question, key, model, fps=None, start=None, end=None, agentic=False):
    part = {"file_data": {"mime_type": mime, "file_uri": uri}}
    vm = {}
    if fps is not None:
        vm["fps"] = fps
    if start is not None:
        vm["start_offset"] = "%ss" % start
    if end is not None:
        vm["end_offset"] = "%ss" % end
    if vm:
        part["video_metadata"] = vm
    if agentic:
        part["processing"] = "agentic"
    payload = {"contents": [{"role": "user", "parts": [{"text": question}, part]}]}
    # 5xx here is TRANSIENT upstream demand (measured: a 503 on an identical request that
    # succeeded moments earlier). Rotating keys cannot fix it; waiting can. Without this
    # retry a 503 is indistinguishable from "the API rejected fps=30", which is a completely
    # different conclusion — that ambiguity is why it matters.
    last = None
    for attempt in range(3):
        req = urllib.request.Request(
            "%s/v1beta/models/%s:generateContent?key=%s" % (BASE, model, key),
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.loads(r.read().decode()), None
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            last = "HTTP %d: %s" % (e.code, body[:700])
            if e.code in (500, 502, 503, 504) and attempt < 2:
                wait = 6 * (attempt + 1)
                print("  %d (transient); retry %d/3 in %ds" % (e.code, attempt + 2, wait),
                      file=sys.stderr)
                time.sleep(wait)
                continue
            return None, last
        except Exception as e:
            return None, "%s: %s" % (type(e).__name__, e)
    return None, last


def ask_interactions(uri, mime, question, key, model, agentic=True):
    """The newer Interactions API surface. `processing: "agentic"` was REJECTED on
    v1beta generateContent ("Unknown name processing"), so that mode is tried here instead,
    where parts are typed objects {type: video|text}."""
    part = {"type": "video", "uri": uri, "mime_type": mime}
    if agentic:
        part["processing"] = "agentic"
    payload = {"model": model, "input": [{"type": "text", "text": question}, part]}
    req = urllib.request.Request(
        "%s/v1beta/interactions?key=%s" % (BASE, key),
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read().decode()), None
    except urllib.error.HTTPError as e:
        return None, "HTTP %d: %s" % (e.code, e.read().decode()[:700])
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, e)


def harvest_text(obj, acc=None):
    """Pull every text field out of an unknown response shape."""
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "text" and isinstance(v, str):
                acc.append(v)
            else:
                harvest_text(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            harvest_text(v, acc)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("question")
    ap.add_argument("--fps", type=float, default=24.0)
    # Default changed after measuring this: gemini-3.5-flash returns 429 (quota exceeded)
    # while 3.7-flash / 3.8-flash / 3.1-flash-lite all accept native video at fps=24 with the
    # same 19,008-token accounting (verified = 24.0 fps). Quota is per-MODEL, so the oldest
    # model in a family is often the one that is spent while its siblings are not.
    ap.add_argument("--model", default="gemini-3.7-flash")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--agentic", action="store_true")
    ap.add_argument("--interactions", action="store_true",
                    help="use the Interactions API (where agentic processing lives)")
    ap.add_argument("--no-fps", action="store_true", help="omit video_metadata entirely")
    ap.add_argument("--output")
    a = ap.parse_args()

    ks = keys()
    if not ks:
        raise SystemExit("no Gemini key found")
    print("keys available: %d (primary first)" % len(ks), file=sys.stderr)

    chosen, uri, name = None, None, None
    for i, k in enumerate(ks):
        try:
            uri, name = upload(a.video, k)
            chosen = k
            break
        except urllib.error.HTTPError as e:
            print("key #%d upload refused (%s)" % (i + 1, e.code), file=sys.stderr)
    if not uri:
        raise SystemExit("upload failed on every key")

    mime = "video/mp4" if a.video.lower().endswith(".mp4") else "video/quicktime"
    fps = None if a.no_fps else a.fps
    print("uploaded -> %s   requesting fps=%s%s" % (name, fps, " (agentic)" if a.agentic else ""),
          file=sys.stderr)
    if a.interactions or a.agentic:
        out, err = ask_interactions(uri, mime, a.question, chosen,
                                    a.model if a.interactions else "gemini-3.8-flash",
                                    agentic=True)
        if not err:
            txt = "\n".join(harvest_text(out))
            print("=" * 74)
            print(txt or json.dumps(out)[:1500])
            print("=" * 74)
            print("usage: %s" % json.dumps(out.get("usageMetadata", out.get("usage", {}))))
            if a.output:
                json.dump({"request": "interactions", "analysis": txt, "raw_keys": list(out)},
                          open(a.output, "w"), indent=1)
            return
        print("interactions path failed: %s" % err[:400], file=sys.stderr)
    effective = fps
    out, err = ask(uri, mime, a.question, chosen, a.model, fps, a.start, a.end, a.agentic)
    if err and fps is not None:
        print("fps=%s REJECTED by the API, retrying without video_metadata:\n  %s"
              % (fps, err[:300]), file=sys.stderr)
        out, err = ask(uri, mime, a.question, chosen, a.model, None, a.start, a.end, a.agentic)
        effective = None       # the answer came from the API default, NOT from `fps`
    if err:
        print("FAILED: %s" % err)
        sys.exit(1)

    # Cross-check what sampling the API ACTUALLY used, from its own token accounting.
    # Recording the REQUESTED fps as if it were the effective one is an estimate in
    # measurement's clothing: a rejected fps=30 was logged as fps:30 while the answer had
    # actually been produced at the default 1 fps (792 video tokens for a 12 s clip).
    vt = 0
    for d in out.get("usageMetadata", {}).get("promptTokensDetails", []):
        if d.get("modality") == "VIDEO":
            vt = int(d.get("tokenCount", 0))
    dur = None
    try:
        import subprocess
        dur = float(subprocess.run(
            "ffprobe -v error -show_entries format=duration -of csv=p=0 \"%s\"" % a.video,
            shell=True, capture_output=True, text=True).stdout.strip())
    except Exception:
        dur = None
    est = round((vt / 66.0) / dur, 2) if (vt and dur) else None
    print("requested fps=%s -> effective fps=%s%s"
          % (fps, effective if effective is not None else "API default (1)",
             "  [FELL BACK]" if effective is None else ""), file=sys.stderr)
    if est is not None:
        print("token accounting implies ~%.2f fps actually sampled (%d video tokens / %.1fs)"
              % (est, vt, dur), file=sys.stderr)

    text = ""
    for c in out.get("candidates", []):
        for p in c.get("content", {}).get("parts", []):
            text += p.get("text", "")
    print("=" * 74)
    print(text)
    print("=" * 74)
    print("usage: %s" % json.dumps(out.get("usageMetadata", {})))
    if a.output:
        json.dump({"question": a.question,
                   "fps_requested": fps,
                   "fps_effective": effective if effective is not None else 1.0,
                   "fell_back_to_default": effective is None,
                   "fps_implied_by_tokens": est,
                   "video_tokens": vt, "agentic": a.agentic,
                   "analysis": text, "usage": out.get("usageMetadata", {}),
                   "model": a.model, "video": a.video},
                  open(a.output, "w"), indent=1)
        print("written: %s" % a.output)


if __name__ == "__main__":
    main()
