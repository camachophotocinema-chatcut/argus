---
name: video-analysis
version: "0.2.0"
description: ARGUS — Analyze any video from a URL (YouTube, X, Vimeo, etc.) or local file — extracts scene-aware frames, gets captions, and sends everything to Gemini 3.5 Flash for timestamped analysis. Voice agent version of bradautomates/claude-video, powered by Gemini vision.
argument-hint: "<video-url-or-path> [optional: question]"
allowed-tools: Terminal, Read, VisionAnalyze
related_skills: [youtube-content]
---

# Argus — Video Analysis (Hermes / Gemini-powered)

Named for the hundred-eyed giant of Greek myth — Argus Panoptes ("all-seeing"). Each extracted frame is another eye on the video.

This skill gives your agent video vision. It downloads video (if URL), extracts frames at scene-aware intervals via ffmpeg, pulls captions (native yt-dlp subs, or Gemini-transcribed audio as fallback), and sends frames + captions to **Gemini 3.5 Flash** for a structured, timestamped analysis.

## Prerequisites

- **ffmpeg** — installed (`/Users/javiercamacho/.local/bin/ffmpeg`)
- **yt-dlp** — installed via brew (`/opt/homebrew/bin/yt-dlp`)
- **GOOGLE_API_KEY** — set in `~/.hermes/.env` (already configured)
- **Groq key** — NOT configured; audio transcription falls back to Gemini 3.5 Flash

All verified working on macOS 26.5 (Apple Silicon).

## How it works

```
User: "analyze this video" or provides a URL
  ↓
Hermes loads this skill
  ↓
Script: analyze_video.py
  1. Downloads video via yt-dlp (or uses local file)
  2. Extracts scene-aware frames via ffmpeg scene detection
     - minimal:  30 frames max, scene-change triggered
     - balanced: 100 frames max, scene-aware + sparse fill (default)
     - detailed: 200 frames max, denser sampling
  3. Gets captions via yt-dlp (auto-subs), or transcribes audio via Gemini 3.5 Flash
  4. Sends ALL frames + captions to Gemini 3.5 Flash in a single multimodal request
  5. Returns structured JSON: analysis text + frame paths + captions
  ↓
Agent: reads the analysis, presents to user with timestamps
```

## Usage

### Basic — analyze a YouTube video

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://youtube.com/watch?v=..."
```

### Analyze a section (faster for long videos)

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "https://youtu.be/..." --start 2:30 --end 5:00
```

### Local file

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py /path/to/video.mp4
```

### Adjust frame density

```bash
# Minimal (30 frames max) — fast, good for 2-3 min videos
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --mode minimal

# Detailed (200 frames max) — max fidelity, high Gemini token cost
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --mode detailed
```

### Extract only (no vision analysis)

Use this when you just want frames + captions as files:

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --no-vision --keep-frames
```

### Save analysis to a file instead of stdout

```bash
python3 ~/.hermes/skills/video-analysis/scripts/analyze_video.py "URL" --output /tmp/video-analysis.json
```

## Workflow for the Hermes agent

1. **Identify the source** — user provides a URL or local file path
2. **Run the script** — use `terminal()` to call the script with appropriate flags
3. **Read the JSON output** — the analysis text is in the `analysis` field
4. **Present to user** — structure includes timestamped frame descriptions, transcript highlights, and an overall summary

## Frame extraction details

- Uses ffmpeg's `select='gt(scene,0.3)'` filter for shot-boundary detection
- Falls back to uniform sampling if scene detection finds < 2 cuts
- Frames are JPEG at quality 3 (~120-200KB each)
- Extracted frames + captions are deleted after analysis unless `--keep-frames`

## Dependencies

| Tool | Status | Purpose |
|------|--------|---------|
| ffmpeg 8.1.2 | ✓ installed | Frame extraction, audio extraction, scene detection |
| yt-dlp | ✓ installed (brew) | Video download, caption extraction |
| Gemini 3.5 Flash | ✓ API key set | Vision + optional audio transcription |
| Groq Whisper | ✗ (no key) | Unavailable — audio transcription uses Gemini instead |

## Token Budget (Critical)

Confirmed from Google's model metadata API — **Gemini 3.5 Flash**:

| Limit | Value |
|-------|-------|
| **Input context** | **1,048,576 tokens** (1M) |
| **Output limit** | **65,536 tokens** (64K) |
| Script's `max_tokens` (analysis) | 8,192 ← **self-imposed bottleneck** |
| Script's `max_tokens` (transcription) | 4,096 ← **can truncate long transcripts** |

The model can output 64K, but the script caps at 8K for analysis and 4K for transcription. Bump these in `analyze_video.py` lines 439 and 340 if you need the full output.

### What consumes the 1M input

| Component | Token cost per unit |
|-----------|-------------------|
| JPEG frame (@640p, ~120-200KB) | ~258 tokens per image |
| Audio (Gemini transcription) | **32 tokens per second** |
| Caption text | ~1 token per 4 chars (clipped to 20K chars = ~5K tokens) |
| System prompt + context | ~500 tokens |

### Headroom calculator (10-minute video, balanced mode)

| Component | Cost |
|-----------|------|
| 100 frames @ 258 tok each | ~25,800 |
| 10 min audio @ 32 tok/s | ~19,200 |
| Caption text (20K chars) | ~5,000 |
| System prompt + context | ~500 |
| **Total** | **~50,500** (4.8% of 1M) |

You can fit about **20× this before filling the input window** — so ~3+ hours of video in balanced mode, or ~90 min in detailed mode (200 frames). Full token budget reference: `references/gemini-token-limits.md`

### Practical guidance

If you hit input context limits:
1. Use `--start MM:SS --end MM:SS` to trim the source video before extraction
2. Drop to `--mode minimal` to reduce frame count
3. Use `--no-vision` for just frames + captions, then analyze in chunks
4. For very long videos (>30 min), consider splitting into segments

## Caveats

- **Script-imposed 8K output cap** — the model can output 64K, but the script limits itself. If analysis text is truncated, bump `max_tokens` in `analyze_frames_with_gemini()`.
- **Audio transcription token cost** — 32 tok/sec is invisible but can dominate the budget on long videos. Prefer captions (yt-dlp) over Gemini transcription when available.
- **No Groq Whisper** — if yt-dlp can't find captions, the script uses Gemini to transcribe audio, which is slower and more expensive. Set a Groq key for fast transcription with no context cost (Groq Whisper is external, not token-billed).
- **Private videos** — yt-dlp can't download private/age-restricted videos without cookies. Pass the local file path instead.

## Recommended limits

| Duration | Mode | Frames | Input tokens est. | Notes |
|----------|------|--------|-------------------|-------|
| < 1 min | minimal | ~15-30 | ~8K-15K | Overkill to use detailed |
| 1-3 min | balanced | ~40-60 | ~20K-30K | Good quality-cost tradeoff |
| 3-10 min | balanced | ~60-100 | ~30K-50K | Covers most tutorial content |
| > 10 min | balanced + start/end | varies | varies | Focus on the relevant section |
