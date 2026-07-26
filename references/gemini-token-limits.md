# Gemini 3.5 Flash — Token Limits Reference

> Queried from Google's model metadata API (`v1beta/models`) on 2026-07-26.
> Endpoint used: `https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`

## Model Identity

The script sends `model: "gemini-3.5-flash"` to the OpenAI-compatible endpoint. This resolves to Google's **Gemini 3.5 Flash** — a stable, fast multimodal model.

## Limits

| Property | Value |
|----------|-------|
| Input context (total prompt) | **1,048,576 tokens** |
| Output limit (model capability) | **65,536 tokens** |
| Script max_tokens (analysis) | **8,192** ← self-imposed cap |
| Script max_tokens (transcription) | **4,096** ← self-imposed cap |

## Token Cost by Content Type

| Content | Token cost | Notes |
|---------|-----------|-------|
| JPEG frame (~120-200KB, 640p) | ~258 tokens per image | Base64 inline in the content array |
| Audio (via Gemini transcription) | **32 tokens/second** of audio | 16kHz mono, any language |
| English text | ~1 token per 4 chars | Roughly 250 words = ~333 tokens |
| Caption text clipping | Script clips to 20,000 chars | That's ~5,000 tokens max |

## Headroom Formula

```
total_tokens = 500 (system) + (frames × 258) + (audio_sec × 32) + (caption_chars / 4)

Example: 10 min video, balanced mode (100 frames), with captions:
  500 + (100 × 258) + (0) + (5000) = 500 + 25,800 + 0 + 5,000 = 31,300 tokens
  → 2.9% of 1M input window

Same video, no captions (Gemini transcribes audio):
  500 + (100 × 258) + (600 × 32) + (0) = 500 + 25,800 + 19,200 + 0 = 45,500 tokens
  → 4.3% of 1M input window
```

## Audio Transcription vs Captions

| Source | Token cost | Quality | Speed |
|--------|-----------|---------|-------|
| yt-dlp captions | ~5K for transcript | Variable (auto-gen may be poor) | Instant |
| Gemini transcribe | 32 tok/sec audio | High accuracy | ~30-60s per 10 min |
| Groq Whisper (not configured) | 0 (external API) | Best | Fastest |

## When to Bump Script max_tokens

The `analyze_video.py` script has two hardcoded caps:

1. **Line 439**: `"max_tokens": 8192` in `analyze_frames_with_gemini()` — caps analysis output
2. **Line 340**: `"max_tokens": 4096` in `transcribe_with_gemini()` — caps transcription output

These are safety limits, not model limits. Bump to `65536` and `16384` respectively if analysis/transcription is being truncated.

## All Gemini Flash Models Available on This API Key

From the full model list (2026-07-26):

| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| gemini-3.5-flash | 1,048,576 | 65,536 | **Current** — stable, fast |
| gemini-3.6-flash | 1,048,576 | 65,536 | Newer, same limits |
| gemini-3.1-flash | 1,048,576 | 65,536 | Previous gen |
| gemini-3.1-flash-lite | 1,048,576 | 65,536 | Cheaper, same input |
| gemini-3-flash-preview | 1,048,576 | 65,536 | Preview |

All Flash models share the 1M input / 64K output spec through this API key.
