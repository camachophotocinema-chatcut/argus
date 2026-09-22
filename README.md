---
name: video-analysis
version: "0.4.0"
description: "ARGUS -- Analyze any video from any source. YouTube, X, Vimeo, direct MP4/WebM URLs, or local video files. Samples frames by density in frames-per-second (frame-rate aware: 24/30/60fps), attaches the audio track for sound-design analysis, and sends everything to Gemini 3.5 Flash for timestamped analysis."
argument-hint: "<video-url-or-path> [optional: question]"
allowed-tools: Terminal, Read, VisionAnalyze
related_skills: [youtube-content]
---

# Argus -- Video Analysis (Hermes / Gemini-powered)

Named for the hundred-eyed giant of Greek myth -- Argus Panoptes ("all-seeing"). Each extracted frame is another eye on the video.

This skill gives your agent video vision for **any video, from any source**. It can:
- **Download** from YouTube, Twitter/X, Vimeo, TikTok, Instagram, Reddit, or any site yt-dlp supports
- **Direct download** raw MP4/WebM/MOV/etc. URLs (`cdn.example.com/video.mp4`)
- **Accept local files** you already have on disk (`/path/to/video.mp4`)

Then it samples frames on a density grid measured in **frames per second** (frame-rate aware, so a 24fps film and a 30fps clip get the same temporal coverage), pulls captions (native yt-dlp subs, sidecar files, or Gemini transcription as fallback), attaches the **audio track itself**, and sends everything to **Gemini 3.5 Flash** for a structured, timestamped analysis.

## Prerequisites

- **ffmpeg** — installed and on `PATH`
- **yt-dlp** — installed via brew (`brew install yt-dlp`)
- **GOOGLE_API_KEY** -- set in `~/.hermes/.env`
- **GEMINI_API_KEY_BACKUP** -- optional backup in `~/.hermes/secrets/gemini-backup.env` (chmod 600)
- **Groq key** -- NOT configured; audio transcription falls back to Gemini 3.5 Flash

## Key failover: a 429 is a per-PROJECT cap, so a second key keeps working

The primary key hit `429 RESOURCE_EXHAUSTED` ("your project has exceeded its monthly
spending cap") and vision analysis stopped dead. The cap is enforced per **project**, not
per key or per account — so a key issued under a *different* project keeps serving. Raise
the cap at https://ai.studio/spend; you cannot top it up.

`_post_gemini()` now owns every Gemini request and handles two distinct failure classes,
which are NOT interchangeable:

- **429 / 401 / 403** → credential or quota problem. Rotate to the next key in
  `get_api_keys()` (primary, then `GEMINI_API_KEY_BACKUP`) and retry the *same* request.
- **500 / 502 / 503 / 504** → transient, server-side (e.g. `503 UNAVAILABLE`, "model is
  currently experiencing high demand"). Retry the *same key* up to 3 times with 5 s then
  10 s backoff. Rotating keys here is useless — the fault is upstream, and burning a
  backup key on it just wastes it.

Verified end-to-end with the primary spent: the log shows `Gemini key #1 refused this
request (429); rotating to backup key` for both the transcription and the frame call, and
the analysis completes. Do not conclude a model is unavailable from one 503 — retest with
a minimal single-image payload before changing the model name.

## Native video at a real frame rate — stills cannot see motion

**The measured failure that forced this.** On a 12 s previz, stills-based analysis at 4 fps
reported *"the camera remains completely static"* for shot 2 — a shot where the camera
provably sweeps 3.8 m sideways. A vision model cannot infer motion from sparse stills; it
guesses. Give it the video instead.

`scripts/argus_native_video.py` uploads the file, sends it as a video part, and sets the
sampling rate explicitly:

```bash
python3 scripts/argus_native_video.py clip.mp4 "what does the camera do?" --fps 24
python3 scripts/argus_native_video.py clip.mp4 "per-shot camera move" --interactions --model gemini-3.7-flash
```

Measured behaviour, all verified on this machine:

- **`fps` is accepted up to 24, and 30 is NOT.** `--fps 30` returns `INVALID_ARGUMENT`. A
  30 fps source must be analysed at 24 or via agentic mode; there is no higher-rate path.
- **24 fps cost, measured:** 19,008 video tokens for a 12 s clip (~66 tokens per frame at the
  default media resolution).
- **It fixed the verdict.** At 24 fps the same clip returns *shot 1 dollying in, shot 2
  tracking right, shot 3 craning up* — with the extra catch that the cat moves at the same
  speed as the tracking camera, keeping it centred. Stills analysis had got 1 of 3 shots
  right; native video got 3 of 3.
- **Agentic processing works on `gemini-3.7-flash`, not on `gemini-3.5-flash`** (which returns
  *"Agentic video processing is not enabled for models/gemini-3.5-flash"*). It lives on the
  Interactions API (`/v1beta/interactions`) with `processing: "agentic"` — NOT on
  `models:generateContent`, which rejects it with *Unknown name "processing"*. When it runs,
  the token accounting proves the model fetches frames itself: `tool_use_tokens` shows video
  AND image pulls over three model invocations, and it was both correct and cheaper
  (~7.7 k tokens vs 19.6 k for the fixed-rate path).

### Rule: record the EFFECTIVE sampling, never the requested one

A rejected `fps=30` was logged as `fps: 30` while the answer had actually been produced at the
API's default **1 fps** — 792 video tokens for a 12 s clip, which is ~12 frames, not 360. That
is an estimate in measurement's clothing, in the tool's own output file. The script now
records `fps_requested`, `fps_effective`, `fell_back_to_default` and `fps_implied_by_tokens`,
and prints the implied rate next to the requested one. **Cross-check any claimed sampling
against the API's own token accounting**; tokens per frame at default resolution is ~66, so
`fps_implied = (video_tokens / 66) / duration`.

## Making motion and audio VISIBLE: the evidence bundle

`scripts/argus_evidence.py` converts a clip into images and numbers a model can read — useful
alongside the native path, and essential when comparing versions of the same shot:

- **kymograph** (space-time slices at left/centre/right): a static camera gives vertical
  stripes, a lateral move gives parallel diagonals, a zoom gives converging ones. One image,
  motion made unambiguous.
- **optical-flow contact sheet** (hue = direction, brightness = speed), **frame-difference
  sheet** (what changed, where), **spectrogram**, **waveform**, and a motion+audio time plot.
- **per-shot motion in px/frame with a verdict**, cut detection, and loudness per 0.5 s.

Calibration lesson: cut detection used a fixed 0.22 frame-difference floor, which sat ABOVE
both real cuts (measured 0.134 and 0.201) while the largest in-shot motion was 0.0797 — so it
found no cuts and collapsed the whole clip into one segment, destroying the per-shot readout.
Use an **adaptive** threshold (baseline mean + 2.5 sd, with a small absolute floor so a static
clip cannot invent cuts).

## Pitfall: 1 fps was a hard floor — and a raw frame cap cannot raise it

The old sampler derived `interval = max(1, int(duration / max_frames))`. That `max(...,1)`
clamped the interval to **one second**, so sampling could never exceed **1 fps on any
video in any mode**, and the 30/100/200 caps were inert for anything under ~30s. It also
never read the source frame rate, so a 24fps film and a 30fps clip were sampled
identically by wall clock — `--mode detailed --max-frames 200` on a 30s short still
yielded exactly 30 frames.

Why this matters: at 1 fps every claim the model makes about *motion* is fabricated. It
reported four "AI artifacts" on a polished 30s short — a "gelatinous" sauce flip, morphing
garlic, a rigid oil stream — all of which were single-frame guesses about action it could
not see. Re-run at 8 fps on the same file with the audio attached, the verdict reversed to
"exceptionally polished… no rendering artifacts visible in any of the frames".

If a report ever claims defects, **check `frame_sampling` first**: if `effective_fps` is
near 1, or `coverage_pct` is single digits, the report is describing a slideshow, not a film.

## Pitfall: the report used to lie about its own method — do not trust a bare label

`frame_extraction` was hardcoded to `"scene-aware (ffmpeg scene detection)"` **even when the
scene detector found nothing and the code had fallen back to blind uniform sampling**. It now
reports the method that actually ran (`uniform 8fps grid (no cuts detected)`). Same family of
bug in `caption_source`, which reported `gemini-transcribe` even when the transcription call had
failed and returned empty, and in the audio part, which used a shape the endpoint rejects — so
**audio transcription had never worked**, and every run implied it had. Labels now distinguish
`gemini-transcribe` / `no-speech` / `transcribe-error` / `none`.

General rule for this script: when a field describes *how* something was produced, it must be
derived from what actually happened, not asserted. An absent signal is not a pass.

## Pitfall: an empty analysis used to be written out as a successful report

Gemini intermittently returns `completion_tokens: 0` with empty content on a large request
(observed once at 240 frames / 270K prompt tokens). The script logged
`Gemini response in 76.4s`, `Analysis complete!`, and wrote `"analysis": "\n"` — a report
that looks like it ran but contains nothing. Downstream that reads as "the model found
nothing," which is a different claim from "the call failed."

It now exits non-zero (`SystemExit(2)`) instead of writing the file. If a run produces no
output file, check the log for the usage line before blaming your invocation:

```bash
grep -E 'Usage:|EMPTY analysis' run.log
```

Recovery: simply lower `--max-frames` and re-run. The same segment that returned 0 tokens at
240 frames returned 4,272 tokens at 150 frames on the next attempt — treat this as a transient
large-request failure, not a property of the video. Budget for one retry.

**A zero-token completion is not a negative finding.** Never report an empty analysis as
"no issues found."

## Pitfall: scene detection silently reported 0 cuts on a video full of cuts

`scene_cuts` / `method` read `(no cuts detected)` on a 5-minute short with ~133 hard
cuts. Cause: the detector runs `ffmpeg ... showinfo` through `run()`, which used
`capture_output=True` and returned only `result.stdout` — but ffmpeg writes filter
output on **stderr**, so the `pts_time:` regex always scanned an empty string. The
field then asserted "no cuts" for a property it had never measured, which is the same
class of bug as the hardcoded `frame_extraction` label above.

Fix in place: `run()` takes `stderr_to_stdout=True` (uses
`stdout=PIPE, stderr=STDOUT` — do NOT combine it with `capture_output=True`, which
raises `ValueError`) and the scene-detection call passes it.

**Check it whenever `scene_cuts` is 0 on real footage.** Verify independently:

```bash
# cut counts at descending thresholds -- a genuine fast-cut piece is not 0 at 0.3
ffmpeg -v error -i in.mp4 \
  -vf "select='gt(scene,0.3)',metadata=print:file=/tmp/sc.txt" -an -f null -
grep -c pts_time /tmp/sc.txt
```

Measured on a 300s short: 82 cuts at scene>0.4, 133 at >0.3, 216 at >0.2 — i.e. mean
shot length ~2.2s. State the threshold when quoting a cut count; the number is not a
property of the video alone. Do not infer shot boundaries from a coarse sampling grid
instead — at 3fps every "shot" lands on a 1/3s multiple, which is the grid talking.

## Pitfall: Gemini's OpenAI-compatible endpoint rejects `{"type":"audio"}`

Audio content parts use OpenAI's shape. This is rejected with HTTP 400
`Invalid content part type: audio`:

```json
{"type": "audio", "audio_url": "data:audio/mpeg;base64,..."}   // ✗ 400
```

These work (verified against `/v1beta/openai/chat/completions`):

```json
{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "mp3"}}   // ✓
```

Audio likely works via the native path too (`/v1beta/models/<model>:generateContent` with
`inline_data`), but the OpenAI-compatible endpoint is what this script uses. Note `format`
is a bare extension (`mp3`, `wav`), not a MIME type.

## Cost of a dense run (measured)

242 frames + a 30s audio track against `gemini-3.5-flash` = **271,631 prompt tokens**,
1,424 completion tokens, ~64s wall time. Density is the cost driver, so reach for
`--fps 8` on shorts and use `--start/--end` (or a pre-cut file) on long sources rather
than raising the cap.

## Pitfall: `--start/--end` may be ignored — verify `duration_sec` before trusting a scoped run

`--start/--end` may be ignored; if `duration_sec` equals the full video, every citation is
full-video and the report must say so. If you need a genuine window, cut it first with ffmpeg and
pass the local file path.

## Pitfall: YouTube returns HTTP 403 to yt-dlp's default player clients

Symptom: download "succeeds" but then fails at `ffprobe` with `moov atom not found` /
non-zero exit, because yt-dlp fetched an error page instead of media. The failure surfaces
in `get_video_duration`, which makes it look like an ffprobe or file-format problem rather
than an extraction problem -- that misdirection is the expensive part.

Fix: force a player client that still serves media. The script passes
`--extractor-args "youtube:player_client=android"` on every yt-dlp call (download, title,
and both caption paths). As of this writing `android` and `mweb` work; `ios`, `tv`,
`web_safari`, and `tv_embedded` all 403.

If YouTube breaks again, swap the client name rather than debugging the video pipeline --
and apply it to *all* yt-dlp calls in the script, not just the download, or captions still fail.

## How it works

```
User: "analyze this video" or provides a URL / file path
  ↓
Hermes loads this skill
  ↓
Script: analyze_video.py
  1. Gets the video:
     - Streaming URLs (YouTube, X, Vimeo...): yt-dlp downloads
     - Direct video URLs (.mp4, .webm, .mov...): curl downloads
     - Local files (/path/to/video.mp4): used in-place
  2. Extracts frames on a density grid measured in frames-per-second
     - minimal:  1 fps  — talking-head/tutorial summaries only
     - balanced: 4 fps  — default (cap 300)
     - detailed: 8 fps  — craft review (cap 600)
     - `--fps N` overrides; clamped to the source frame rate
     - scene cuts are merged into the grid so shot boundaries are never skipped
  3. Gets captions:
     ① yt-dlp (any platform with subs)
     ② sidecar file (.vtt/.srt/.ass next to local video)
     ③ Gemini 3.5 Flash transcription (fallback -- extracts audio, transcribes)
  4. Sends ALL frames + captions + the audio track to Gemini 3.5 Flash in a single
     multimodal request
  5. Returns structured JSON: analysis text + frame paths + captions
  ↓
Agent: reads the analysis, presents to user with timestamps
```

## Usage

### Basic -- analyze a YouTube video

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://youtube.com/watch?v=..."
```

### Ask a specific question (strongly preferred for diagnosis)

Pass a question as the second positional argument. It becomes the primary task and the
model is told to cite a timestamp per claim and to flag defects rather than infer:

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py /path/to/clip.mp4 \
  "Does the subject's anatomy stay consistent as the camera pulls back? Describe any distortion, deformation or artifact, with timestamps." \
  --mode detailed --output /tmp/review.json
```

Use this whenever you are reviewing your own output rather than summarising someone
else's video.

### Analyze a section (faster for long videos)

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://youtu.be/..." --start 2:30 --end 5:00
```

### Local file

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py /path/to/video.mp4
```

### Direct video URL (raw mp4/webm)

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://cdn.example.com/video.mp4"
```

### X/Twitter attached video

```bash
# 1fps summary of what happens
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://x.com/User/status/<tweet_id>/video/1" --mode minimal

# craft review of a short -- use this for anything you are judging the quality of
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://x.com/User/status/<tweet_id>/video/1" --fps 8
```

No auth needed — yt-dlp resolves X video URLs. For tweet text, media metadata, and login-gated X-article handling, see `references/x-twitter-extraction.md` (syndication API + twitter-cli articleText path).

### Adjust sampling density (use `--fps`)

Sampling is expressed in **frames per second**, not a raw frame count. Presets:
`minimal` 1 fps · `balanced` 4 fps · `detailed` 8 fps. Override the preset with `--fps`
(clamped to the source rate — you cannot sample more frames than exist).

```bash
# 8 fps -- craft review: animation quality, physical continuity, motion
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --mode detailed

# explicit rate (a 30s short at 8fps = ~240 frames)
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --fps 8

# 1 fps -- summaries of talking-head/tutorial content ONLY
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --mode minimal
```

**Choose the mode by what you are asking.** `minimal` answers "what happens in this
video". It cannot answer "is this well made" — at 1 fps you get 30 frames of a 30s
short and **no motion information at all**. Judge craft at `detailed`/`--fps 6-8`.

Every run reports `frame_sampling` (source fps, source frames, target/effective fps,
scene cuts, % of source frames covered) and the true `frame_extraction` method, so
the report states its own sampling density rather than implying it saw everything.

### Audio is attached to the analysis, not just transcribed

The audio track is sent to the model **as audio** alongside the frames. A transcript
carries only *speech* — on a music/ASMR/no-dialogue piece it is empty, which leaves
sound design unanalysable. With the audio attached the model reports specific,
timestamped sound events (foley, score swells, silence beats) against the picture.
Use `--no-audio` only when you deliberately want frames + transcript only.

### Extract only (no vision analysis)

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --no-vision --keep-frames
```

### Save analysis to a file instead of stdout

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --output /tmp/video-analysis.json
```

### Cost discipline (parent-context bloat)

A full Argus run returns multi-thousand-token JSON. Do NOT dump it inline into the parent
thread — the parent re-reads its whole context on every subsequent tool call, so a few inline
Argus dumps balloon the session's token cost (measured: ~331M cache-read tokens ≈ $7 in one
long session). Two ways to keep the parent lean:

- Always pass `--output <file>`, then read only the `analysis` field: `jq -r '.analysis' <file>`.
- When the run is one step in a larger task, delegate it to a subagent (isolated context) so
  only a short summary returns to the parent.

## Workflow for the Hermes agent

1. **Identify the source** -- user provides a URL or local file path
2. **Run the script** -- use `terminal()` to call the script with appropriate flags
3. **Read the JSON output** -- the analysis text is in the `analysis` field
4. **Present to user** -- structure includes timestamped frame descriptions, transcript highlights, and an overall summary

## Reviewing STILLS and storyboards (frames that were never a video)

Argus is the right instrument for checking a set of stills against each other — storyboard frames, plate
sets, generated images that must be consistent. **Do not review them one at a time.** A per-image vision
pass sees a single frame with no shared context: it cannot see what happens *between* frames, and it
describes small objects inconsistently from frame to frame. Observed twice on the same set: two isolated
passes returned **opposite** verdicts about a telephone's keypad, and eight isolated passes missed a
day/night break, a colour change in a prop, and two frames that were supposed to be the same physical
object and were not.

The method is to turn the stills into a sequence and hand the whole thing over in one request:

```sh
# concat the frames in intended order, ~2.5s each, then analyse the result
# (build a concat list of 'file <path>' + 'duration 2.5' per frame, last file repeated)
ffmpeg -y -f concat -safe 0 -i list.txt -vf scale=1600:-2,setsar=1 -r 4 \
  -pix_fmt yuv420p -c:v libx264 -crf 20 sequence.mp4

python3 scripts/analyze_video.py sequence.mp4 "<continuity brief>" --mode detailed --output out.json
```

**Put a frame manifest in the question** — number, filename and a one-line description of each frame — so
the model can cite frames by name rather than by timestamp. Ask it explicitly to check:

- **object continuity** (is the phone/chair/document the same physical object in every frame?)
- **character continuity** (wardrobe, hair, eyewear, badges, and props they are holding)
- **room and lighting continuity** (time of day, architecture, window content)
- **artifact continuity** (is chart B the same sheet of paper as chart A — same corner curl, grid, line style?)

and to **flag defects rather than infer**, and to answer "cannot be determined" when a check is not
possible instead of guessing.

Two quirks when the input is a slideshow rather than footage:

- `Scene detection found no cuts, falling back to uniform sampling` is **expected** — hard cuts between
  stills often fall below the scene threshold. Uniform sampling is fine; you get ~2-3 frames per slide.
- There is no audio, so captions come back `none`. That is not an error.

**Settle object continuity by construction, not by analysis.** When two instruments disagree about a small
object, do not arbitrate between them: lock one canonical instance as an **image reference** and regenerate
every frame containing that object against it. Attaching the canonical frame as a reference image makes the
objects the same by construction, which is more reliable than any reviewer's verdict.

## Interpreting a review pass — what the model reliably does and does not do

**Never ask for defects on a clip you are judging the quality of.** A question containing "flag any
defect / artifact / glitch" manufactures a defect list — the model answers the question you asked.
Measured on one polished 30s short: a defect-hunting prompt returned four confident "AI artifacts";
the same file re-run on a neutral question described it as having no visible rendering artifacts.
Ask what happens, neutrally, then ask targeted follow-ups.

**Treat causal claims as hypotheses, not findings.** A review attributed a jump cut to a specific
edit ("a direct result of the removed shot"). Independent measurement showed the identical
discontinuity already existed in the untouched source at the same cut point. The model can correctly
see *that* something is wrong and be completely wrong about *why*.

**Timestamps and framing labels are approximate.** The same shot was called a "close-up" in one
pass and a "medium-wide" in the next, and beat timings shift by up to a second between runs. Use a
review to locate *what* is wrong; use a pixel measurement to place, size or verify it.

**Conversely, trust it on things a meter cannot see.** It reliably caught audio content, physics
plausibility, whether a specific story beat was present at all, and that a character had become
frozen mid-shot. Those are exactly the judgements ffmpeg cannot make — treat them as findings, while
still confirming anything structural yourself.

## Frame extraction details

- Samples on a uniform grid at the target rate (default 4 fps; `--fps N` to override),
  clamped to the source frame rate
- Runs ffmpeg's `select='gt(scene,0.3)'` scene detection and **merges any cuts into the
  grid**, so shot boundaries are never skipped even when they fall between grid points
- Frames are JPEG at quality 3 (~120-200KB each)
- Filenames carry millisecond timestamps (`frame_00123_00051250ms.jpg`)
- Extracted frames + captions are deleted after analysis unless `--keep-frames`

`frame_sampling` in the output records source fps, source frame count, target and
effective fps, scene cuts found, and the percentage of source frames actually covered.

## Download pipeline

See `references/download-pipeline.md` for the three download strategies (curl, yt-dlp, local), caption resolution order, and audio edge cases.

## Dependencies

| Tool | Status | Purpose |
|------|--------|---------|
| ffmpeg 8.1.2 | ✓ installed | Frame extraction, audio extraction, scene detection |
| yt-dlp | ✓ installed (brew) | Video download, caption extraction |
| Gemini 3.5 Flash | ✓ API key set | Vision + optional audio transcription |
| Groq Whisper | ✗ (no key) | Unavailable -- audio transcription uses Gemini instead |

## Token Budget (Critical)

| Limit | Value |
|-------|-------|
| **Input context** | **1,048,576 tokens** (1M) |
| **Output limit** | **65,536 tokens** (64K) |
| Script's `max_tokens` (analysis) | 32,768 (bumped from 8,192 on 2026-08-04 after truncation) |
| Script's `max_tokens` (transcription) | 4,096 |

See `references/gemini-token-limits.md` for full budget details.

### Practical guidance

If you hit input context limits:
1. Use `--start MM:SS --end MM:SS` to trim the source video before extraction
2. Drop to `--mode minimal` to reduce frame count
3. Use `--no-vision` for just frames + captions, then analyze in chunks
4. For very long videos (>30 min), consider splitting into segments

## Caveats

- **Script-imposed 32K output cap** -- the model can output 64K, but the script limits itself to 32,768 (was 8,192 until 2026-08-04, when a real run truncated). If analysis text is STILL truncated, bump `max_tokens` in `analyze_frames_with_gemini()` further.
- **Audio transcription token cost** -- 32 tok/sec is invisible but can dominate the budget on long videos. Prefer captions (yt-dlp) over Gemini transcription when available.
- **No Groq Whisper** -- if yt-dlp can't find captions, the script uses Gemini to transcribe audio, which is slower and more expensive.
- **Private videos** -- yt-dlp can't download private/age-restricted videos without cookies. Pass the local file path instead.
- **System temp-dir cleanup** -- frames/captions land in a `hermes-video-*` dir under `/var/folders` that macOS can clean mid-session (observed: the .vtt vanished before it could be read). If you need captions after a run, re-extract to a stable path: `yt-dlp --skip-download --write-subs --sub-langs en --sub-format vtt -o /tmp/caps/video "URL"` (or pass `--keep-frames` and copy the .vtt out immediately).
- **Direct URLs** -- raw .mp4/.webm URLs download via curl. Typically faster than yt-dlp but may lack metadata. Captions only if embedded.

## Using this as a Hermes skill

Hermes loads a skill from `SKILL.md`; GitHub renders `README.md`. Rather than keep two
copies of the same document in sync, `README.md` is the single source of truth and
`SKILL.md` is a **symlink** to it. Edit either name and you are editing the same file.

```bash
git clone https://github.com/camachophotocinema-chatcut/argus \
  ~/.hermes/skills/video-analysis
```

The repo is then a live working copy of the skill — commit and push your local
improvements instead of letting them drift unshared.

## Recommended limits

| Duration | Mode | Sampling | Input tokens est. | Notes |
|----------|------|----------|-------------------|-------|
| < 1 min | `--fps 8` | ~8 fps | ~200-280K (measured 271K for 30s/242 frames) | Correct choice for judging craft |
| < 1 min | minimal | 1 fps (~30 frames) | ~10K | Summary only — misreads motion |
| 1-3 min | balanced | 4 fps (cap 300) | ~40-90K | Good quality-cost tradeoff |
| 3-10 min | balanced | 4 fps, cap binds | ~40-90K | Cap reports when it binds |
| > 10 min | balanced + `--start/--end` | varies | varies | Pre-cut the file — `--start/--end` may be ignored |
