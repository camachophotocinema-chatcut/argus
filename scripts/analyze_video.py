#!/usr/bin/env python3
"""
Video Analyzer for Hermes Agent
Downloads video (if URL), extracts scene-aware frames, gets captions, and
sends everything to Gemini 3.5 Flash for analysis.

Usage:
    python3 analyze_video.py <url_or_path> [--mode balanced|minimal|detailed]
                             [--start MM:SS] [--end MM:SS] [--output FILE]

Returns JSON with:
  - mode: analyzed mode used
  - title: video title (if available)
  - duration_sec: video duration
  - frame_count: number of frames analyzed
  - caption_source: yt-dlp / gemini-transcribe / local
  - analysis: the full Gemini analysis text
  - frames_dir: path to extracted frames
  - caption_file: path to caption text
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def log(msg):
    print(f"[analyze-video] {msg}", file=sys.stderr)


def run(cmd, timeout=120, check=True, stderr_to_stdout=False):
    """Run a command, return stdout.

    Set stderr_to_stdout=True for ffmpeg filters that report on stderr
    (showinfo/metadata=print) -- with the default capture_output=True those
    lines never reach stdout and any regex over them silently finds nothing.
    """
    if stderr_to_stdout:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout
        )
    else:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    if check and result.returncode != 0:
        log(f"Command failed: {' '.join(cmd)}")
        log(f"STDERR: {(result.stderr or '')[:500]}")
        result.check_returncode()
    return (result.stdout or "").strip()


def format_timestamp(seconds):
    """Convert seconds to SRT/SSRT timestamp format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_timestamp(ts):
    """Parse MM:SS or HH:MM:SS to seconds."""
    parts = list(map(int, ts.split(":")))
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def get_video_duration(path):
    """Get video duration in seconds via ffprobe."""
    out = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", path
    ], timeout=30)
    return float(out)


def probe_video_stream(path):
    """Read duration, native frame rate and frame count.

    Argus must know the source frame rate to sample *relative to it*. Sampling by
    wall-clock seconds alone is frame-rate blind: a 24fps film and a 30fps clip
    both got 1 frame per second regardless of how much motion sits between them.
    """
    out = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height:format=duration",
        "-of", "default=noprint_wrappers=1", path,
    ], timeout=30)

    vals = {}
    for line in (out or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip()

    def _rate(s):
        try:
            if "/" in s:
                num, den = s.split("/")
                return float(num) / float(den) if float(den) else 0.0
            return float(s)
        except Exception:
            return 0.0

    def _int(s):
        try:
            return int(s)
        except Exception:
            return 0

    fps = _rate(vals.get("r_frame_rate", "")) or _rate(vals.get("avg_frame_rate", ""))
    try:
        duration = float(vals.get("duration") or 0)
    except Exception:
        duration = 0.0
    if not duration:
        try:
            duration = get_video_duration(path)
        except Exception:
            duration = 0.0

    return {
        "duration": duration,
        "fps": round(fps, 3),
        "nb_frames": _int(vals.get("nb_frames")),
        "width": _int(vals.get("width")),
        "height": _int(vals.get("height")),
    }


def extract_captions_ytdlp(url, workdir):
    """Extract captions via yt-dlp. Returns path to caption file or None."""
    log("Extracting captions via yt-dlp...")
    # Same 403 workaround as download_video: default clients are blocked.
    YT_ARGS = ["--extractor-args", "youtube:player_client=android"]
    try:
        out = run([
            "yt-dlp", *YT_ARGS, "--write-auto-subs", "--sub-langs", "en,en-US,en-GB",
            "--skip-download", "--sub-format", "vtt",
            "-o", f"{workdir}/%(id)s.%(ext)s",
            url
        ], timeout=60, check=False)
        log(f"yt-dlp caption output: {out[:200] if out else '(empty)'}")

        # Find the caption file
        caption_files = list(Path(workdir).glob("*.vtt")) + list(Path(workdir).glob("*.srt")) + list(Path(workdir).glob("*.json"))
        if caption_files:
            log(f"Found captions: {caption_files[0]}")
            return str(caption_files[0])

        # Try alternative: write-subs (not auto)
        out = run([
            "yt-dlp", *YT_ARGS, "--write-subs", "--sub-langs", "en,en-US,en-GB",
            "--skip-download", "--sub-format", "vtt",
            "-o", f"{workdir}/%(id)s.%(ext)s",
            url
        ], timeout=60, check=False)

        caption_files = list(Path(workdir).glob("*.vtt")) + list(Path(workdir).glob("*.srt"))
        if caption_files:
            log(f"Found captions (subs): {caption_files[0]}")
            return str(caption_files[0])

        log("No captions found via yt-dlp")
        return None
    except Exception as e:
        log(f"Caption extraction failed: {e}")
        return None


def is_direct_video_url(url):
    """Check if a URL points directly to a video file (not a streaming site)."""
    direct_exts = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".gif", ".mpg", ".mpeg")
    path_part = url.split("?")[0].split("#")[0].lower()
    return path_part.endswith(direct_exts)


def download_direct_url(url, workdir):
    """Download a direct video URL via curl. Returns path to video file."""
    log("Downloading direct video URL via curl...")
    ext = ".mp4"
    for known_ext in (".mp4", ".webm", ".mkv", ".mov", ".m4v"):
        if url.lower().endswith(known_ext):
            ext = known_ext
            break
    out_path = f"{workdir}/video{ext}"
    result = subprocess.run(
        ["curl", "-L", "-o", out_path, url],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode == 0 and os.path.exists(out_path):
        log(f"Downloaded to {out_path}")
        return out_path
    log(f"Direct download failed: {result.stderr[:200]}")
    return None


def download_video(url, workdir):
    """Download video via yt-dlp (or fall back to curl for direct URLs). Returns path to video file and metadata."""
    log("Downloading video...")

    # YouTube blocks yt-dlp's default player clients with HTTP 403. The `android`
    # client still serves media as of this writing; `mweb` also works. Without this
    # every YouTube URL fails at ffprobe with "moov atom not found" because the
    # download silently produced an error page instead of a video.
    YT_ARGS = ["--extractor-args", "youtube:player_client=android"]

    # Try direct download first for obvious video-file URLs (fast, no yt-dlp overhead)
    video_path = None
    title = "Unknown"

    if is_direct_video_url(url):
        video_path = download_direct_url(url, workdir)
        if video_path:
            return video_path, url.rsplit("/", 1)[-1].split("?")[0]

    # Otherwise try yt-dlp
    try:
        # Get title first (metadata only, instant)
        title_out = run([
            "yt-dlp", *YT_ARGS, "--print", "title", "--skip-download", url
        ], timeout=30)
        title = title_out.strip() or "Unknown"

        # Then download the video
        run([
            "yt-dlp", *YT_ARGS,
            "-f", "bestvideo*[height<=1080]+bestaudio/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", f"{workdir}/video.%(ext)s",
            url
        ], timeout=300)
    except Exception as e:
        log(f"yt-dlp download failed: {e}")
        # Fallback: try direct curl download if we haven't already
        if video_path is None:
            video_path = download_direct_url(url, workdir)
            if video_path:
                return video_path, url.rsplit("/", 1)[-1].split("?")[0]
        return None, "Unknown"

    # Find the actual video file
    candidates = sorted(Path(workdir).glob("*.mp4"))
    if not candidates:
        candidates = sorted(Path(workdir).glob("video.*"))
    video_path = str(candidates[0]) if candidates else None
    return video_path, title


def download_video_limited(url, workdir, start_sec=0, end_sec=None):
    """Download a segment of the video (much faster for long videos)."""
    log(f"Downloading video segment ({start_sec}s to {end_sec or 'end'})...")
    try:
        # First try with download-sections
        cmd = [
            "yt-dlp", "-f", "bestvideo*[height<=1080]+bestaudio/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", f"{workdir}/video_segment.%(ext)s",
            url
        ]
        if end_sec:
            cmd.extend(["--download-sections", f"*{format_timestamp(start_sec)}-{format_timestamp(end_sec)}"])
        elif start_sec > 0:
            cmd.extend(["--download-sections", f"*{format_timestamp(start_sec)}-"])

        out = run(cmd, timeout=300, check=False)
        candidates = list(Path(workdir).glob("video_segment.*"))
        if candidates:
            return str(candidates[0])
        return None
    except Exception as e:
        log(f"Segment download failed: {e}")
        return None


def extract_frames_ffmpeg(video_path, workdir, mode="balanced", max_frames=None,
                          target_fps=None):
    """
    Extract frames on a density grid measured in frames-per-second.

    Modes set the target rate, not a raw count:
      - minimal:  1 fps  — summaries of talking-head/tutorial content
      - balanced: 4 fps  — default; enough to read motion
      - detailed: 8 fps  — craft review: animation quality, physical continuity

    Why fps and not a frame cap: the previous version derived
    `interval = max(1, int(duration / max))`, whose floor of one second silently
    capped sampling at 1 fps for EVERY video and made the 30/100/200 caps inert
    for anything under ~30s. It also never read the source frame rate, so a 24fps
    film and a 30fps clip were sampled identically by wall clock. Sampling density
    is now expressed relative to the source rate and clamped to it.

    Returns (frames, duration, meta) where frames is a list of (timestamp_sec, path).
    """
    log(f"Extracting frames ({mode} mode)...")

    frames_dir = Path(workdir) / "frames"
    frames_dir.mkdir(exist_ok=True)

    FPS_PRESETS = {"minimal": 1.0, "balanced": 4.0, "detailed": 8.0}

    src = probe_video_stream(video_path)
    duration = src["duration"]
    native_fps = src["fps"] or 0.0
    log(f"Source: {duration:.2f}s @ {native_fps or '?'} fps "
        f"({src['nb_frames'] or '?'} frames, {src['width']}x{src['height']})")

    want_fps = float(target_fps) if target_fps else FPS_PRESETS.get(mode, 4.0)
    if native_fps and want_fps > native_fps:
        log(f"Requested {want_fps:g} fps exceeds source {native_fps:g} fps — clamping")
        want_fps = native_fps

    # Dense uniform grid at the requested rate (sub-second resolution)
    step = 1.0 / want_fps if want_fps else 1.0
    n = int(duration * want_fps) if duration and want_fps else 0
    times = [round(i * step, 3) for i in range(n)]

    # Union in scene cuts so shot boundaries can never be skipped
    log("Running scene detection...")
    try:
        scene_out = run([
            "ffmpeg", "-i", video_path,
            "-filter:v", "select='gt(scene,0.3)',showinfo",
            "-f", "null", "-",
        ], timeout=120, check=False, stderr_to_stdout=True)
    except Exception as e:
        log(f"Scene detection failed: {e}")
        scene_out = ""

    cuts = [round(float(m), 3) for m in re.findall(r"pts_time:([\d.]+)", scene_out or "")]
    cuts = sorted({t for t in cuts if 0 < t < duration})
    if cuts:
        times = sorted(set(times) | set(cuts))
        log(f"Scene detection: {len(cuts)} cuts merged into the grid")
    else:
        log("Scene detection found no cuts — uniform grid only")

    # Cap, preserving even coverage
    capped = False
    if max_frames and len(times) > max_frames:
        idx = [int(i * len(times) / max_frames) for i in range(max_frames)]
        times = [times[i] for i in idx]
        capped = True

    effective_fps = (len(times) / duration) if duration else 0.0
    method = f"uniform {want_fps:g}fps grid"
    method += f" + {len(cuts)} scene cuts" if cuts else " (no cuts detected)"
    if capped:
        method += f", capped at {max_frames}"

    log(f"Extracting {len(times)} frames ({method})")

    # Step 2: Extract each frame
    frames = []
    for i, ts in enumerate(times):
        if ts >= duration:
            continue
        # millisecond precision — int(ts) collided for sub-second timestamps
        frame_file = frames_dir / f"frame_{i:05d}_{int(round(ts * 1000)):08d}ms.jpg"
        if not frame_file.exists():
            try:
                run([
                    "ffmpeg", "-ss", f"{ts:.3f}", "-i", video_path,
                    "-vframes", "1", "-q:v", "3",
                    "-y", str(frame_file)
                ], timeout=30, check=False)
            except Exception:
                continue
        frames.append((ts, str(frame_file)))

    coverage = (100.0 * len(frames) / src["nb_frames"]) if src["nb_frames"] else None
    meta = {
        "source_fps": native_fps,
        "source_frames": src["nb_frames"],
        "source_resolution": f"{src['width']}x{src['height']}",
        "target_fps": want_fps,
        "effective_fps": round(effective_fps, 2),
        "scene_cuts": len(cuts),
        "coverage_pct": round(coverage, 1) if coverage is not None else None,
        "method": method,
    }
    log(f"Extracted {len(frames)} frames "
        f"({meta['effective_fps']} fps, {coverage and round(coverage,1)}% of source frames)")
    return frames, duration, meta


def parse_captions_vtt(vtt_path):
    """Parse VTT/SRT captions into a list of (start_sec, end_sec, text) tuples."""
    text = Path(vtt_path).read_text()
    segments = []

    # Strip VTT header
    if text.startswith("WEBVTT"):
        text = re.sub(r"^WEBVTT.*?\n\n", "", text, flags=re.DOTALL)

    # Parse SRT/VTT-style blocks
    blocks = re.split(r"\n\n+", text.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        # Find the timestamp line
        ts_line = None
        text_lines = []
        for line in lines:
            if "-->" in line:
                ts_line = line
            elif line.strip() and not line.strip().isdigit():
                text_lines.append(line.strip())

        if ts_line and text_lines:
            # Parse timestamps
            parts = re.split(r"\s+-->\s+", ts_line)
            if len(parts) == 2:
                start = _vtt_to_seconds(parts[0])
                end = _vtt_to_seconds(parts[1])
                text = " ".join(text_lines)
                segments.append((start, end, text))

    return segments


def _vtt_to_seconds(ts):
    """Convert VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def caption_segments_to_text(segments):
    """Convert caption segments to a single text with timestamps."""
    lines = []
    for start, end, text in segments:
        lines.append(f"[{format_timestamp(start)} - {format_timestamp(end)}] {text}")
    return "\n".join(lines)


def extract_audio(video_path, workdir):
    """Extract audio for transcription. Returns path or None on failure."""
    audio_file = Path(workdir) / "audio.mp3"
    try:
        run([
            "ffmpeg", "-i", video_path, "-vn",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1",
            "-y", str(audio_file)
        ], timeout=120)
    except Exception as e:
        log(f"Audio extraction failed (video may have no audio track): {e}")
        return None
    return str(audio_file) if audio_file.exists() else None


def transcribe_with_gemini(audio_path, workdir, api_key):
    """Transcribe audio via Gemini (sends audio file + transcription prompt)."""
    log("Transcribing audio via Gemini 3.5 Flash...")

    import base64

    audio_data = Path(audio_path).read_bytes()
    audio_b64 = base64.b64encode(audio_data).decode()

    prompt = {
        "model": "gemini-3.5-flash",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio verbatim. Include timestamps in [HH:MM:SS] format at each sentence. Output only the transcript, no preamble."},
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}}
            ]
        }],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    result, err = _post_gemini(gemini_url, prompt, timeout=120)
    if err or result is None:
        return None, ""

    try:
        transcript = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        log(f"Transcription length: {len(transcript)} chars")

        # Save transcript
        transcript_file = Path(workdir) / "transcript.txt"
        transcript_file.write_text(transcript)
        return str(transcript_file), transcript
    except Exception as e:
        log(f"Gemini transcription parse failed: {type(e).__name__}: {e}")
        return None, ""


def analyze_frames_with_gemini(frames, captions_text, title, duration, api_key, mode="balanced", question=None, audio_path=None):
    """
    Send all extracted frames + captions + (when present) the audio track to
    Gemini 3.5 Flash for analysis.

    The audio track matters: a text transcript carries only *speech*. On a
    no-dialogue ASMR / score-driven piece the transcript is empty, which
    previously left sound design — often the whole point of the short —
    structurally invisible to the model. Gemini ingests audio natively, so the
    waveform is attached directly.

    Uses the OpenAI-compatible endpoint of Gemini API.
    """
    log("Sending frames to Gemini 3.5 Flash for analysis...")

    import base64

    # Build the multi-part content
    content_parts = []

    # System-like instruction as first text
    instruction = (
        "You are analyzing a video from timestamped frames"
        + (", its audio track," if audio_path else "")
        + " and caption text.\n\n"
    )
    if question:
        instruction += (
            "**Answer this question as your primary task, citing a timestamp for every "
            "claim:**\n" + question + "\n\n"
            "Ground every statement in what the frames actually show. If the frames do not "
            "support a claim, say so instead of inferring it. Call out defects, artefacts "
            "and anything that looks wrong plainly and specifically. Only then summarise "
            "what happens in the video.\n\n"
        )
    instruction += (
        "For each major section or timestamp cluster, provide:\n"
        "1. **Timestamp** — when this section occurs\n"
        "2. **What's on screen** — describe the visual content (UI, code, people, etc.)\n"
        "3. **What's being said** — key points from the captions at this time\n"
        "4. **Key takeaway** — one-line substance, stripped of hype\n\n"
        "End with an overall summary (3-5 bullet points) that a busy person can read "
        "in 30 seconds. Be concise and specific."
    )
    content_parts.append({"type": "text", "text": instruction})

    # Add context
    ctx = f"Title: {title}\nDuration: {duration:.1f}s\nAnalysis mode: {mode}\n\n"
    content_parts.append({"type": "text", "text": ctx})

    # Add frames (base64 inline JPEG) — batch in groups to avoid token limits
    frame_count = 0
    for ts, frame_path in frames:
        try:
            img_data = Path(frame_path).read_bytes()
            img_b64 = base64.b64encode(img_data).decode()
            content_parts.append({
                "type": "text",
                "text": f"\n--- Frame at {format_timestamp(ts)} ---"
            })
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            frame_count += 1
        except Exception as e:
            log(f"Failed to read frame {frame_path}: {e}")

    # Add captions
    if captions_text:
        content_parts.append({
            "type": "text",
            "text": f"\n\n--- Captions/Transcript ---\n{captions_text[:20000]}"
        })
    else:
        content_parts.append({
            "type": "text",
            "text": ("\n\nNo speech transcript available. "
                     + ("Describe the sound design from the attached audio track."
                        if audio_path else "Rely on visual frame analysis only."))
        })

    # Attach the audio track itself, not just a transcript. A transcript carries
    # only speech — on a no-dialogue ASMR/score piece it is empty, which left the
    # sound design invisible to the model.
    if audio_path and os.path.exists(audio_path):
        try:
            audio_b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()
            # The Gemini OpenAI-compatible endpoint takes OpenAI's `input_audio`
            # shape. The previously used {"type":"audio","audio_url":...} is
            # rejected with 400 "Invalid content part type: audio".
            fmt = Path(audio_path).suffix.lstrip(".").lower() or "mp3"
            content_parts.append({"type": "text", "text": "\n\n--- Audio track (full) ---"})
            content_parts.append({
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": fmt},
            })
            log(f"Attached the audio track to the analysis request ({fmt})")
        except Exception as e:
            log(f"Failed to attach audio track: {e}")

    # Send to Gemini
    payload = {
        "model": "gemini-3.5-flash",
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 32768,
        "temperature": 0.3,
    }

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    log(f"Sending {frame_count} frames to Gemini...")
    start_time = time.time()

    result, err = _post_gemini(gemini_url, payload, timeout=300)
    if err or result is None:
        return err

    elapsed = time.time() - start_time
    log(f"Gemini response in {elapsed:.1f}s")

    analysis = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = result.get("usage", {})
    log(f"Usage: {json.dumps(usage)}")
    return analysis


def get_api_key():
    """Get Google/Gemini API key (first available)."""
    return (get_api_keys() or [None])[0]


def get_api_keys():
    """Ordered list of Gemini API keys: primary first, then any backups.

    Failover exists because the primary key hit its monthly spend cap (HTTP 429
    RESOURCE_EXHAUSTED), which killed vision analysis outright. A 429 is a
    per-PROJECT limit, so a key from a different project keeps working.
    Backup source: ~/.hermes/secrets/gemini-backup.env (chmod 600) or env.
    """
    keys = []

    def _push(v):
        v = (v or "").strip()
        if v and "your_" not in v and v not in keys:
            keys.append(v)

    _push(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if not keys:
        env_path = os.path.expanduser("~/.hermes/.env")
        if os.path.exists(env_path):
            for line in Path(env_path).read_text().split("\n"):
                if line.startswith("GOOGLE_API_KEY=") and "your_" not in line:
                    _push(line.split("=", 1)[1])

    _push(os.environ.get("GEMINI_API_KEY_BACKUP"))
    backup_path = os.path.expanduser("~/.hermes/secrets/gemini-backup.env")
    if os.path.exists(backup_path):
        for line in Path(backup_path).read_text().split("\n"):
            if line.startswith("GEMINI_API_KEY_BACKUP=") and "your_" not in line:
                _push(line.split("=", 1)[1])

    return keys


def _post_gemini(url, payload, timeout=300):
    """POST to the Gemini OpenAI-compatible endpoint, rotating keys on quota errors.

    Returns (parsed_json_or_None, error_string_or_None). A 429 is a per-PROJECT
    cap, so when the primary is spent we retry the SAME request with the backup key
    rather than failing the whole analysis.
    """
    import urllib.request
    import urllib.error

    keys = get_api_keys()
    if not keys:
        return None, "No GOOGLE_API_KEY found. Set it in .env or environment."

    last = None
    transient = (500, 502, 503, 504)
    for i, k in enumerate(keys):
        for attempt in range(3):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {k}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode()), None
            except urllib.error.HTTPError as e:
                body = e.read().decode()
                last = f"Gemini API error {e.code}: {body[:500]}"
                if e.code in transient and attempt < 2:
                    # 503 = server-side demand spike, not a credential problem.
                    # Rotating keys would not help; wait it out.
                    wait = 5 * (attempt + 1)
                    log(f"Gemini {e.code} (transient); retry {attempt + 2}/3 in {wait}s")
                    time.sleep(wait)
                    continue
                if e.code in (429, 401, 403) and i + 1 < len(keys):
                    log(f"Gemini key #{i + 1} refused this request ({e.code}); rotating to backup key")
                    break
                log(last)
                return None, last
            except Exception as e:
                last = f"Analysis failed: {type(e).__name__}: {e}"
                log(last)
                return None, last
    return None, last


def main():
    parser = argparse.ArgumentParser(description="Analyze a video via Gemini 3.5 Flash")
    parser.add_argument("source", help="YouTube URL or local video file path")
    parser.add_argument("question", nargs="?", default=None,
                        help="Optional question to focus the analysis on")
    parser.add_argument("--mode", choices=["minimal", "balanced", "detailed"],
                        default="balanced", help="Frame extraction density")
    parser.add_argument("--start", help="Start timestamp (MM:SS or HH:MM:SS)")
    parser.add_argument("--end", help="End timestamp (MM:SS or HH:MM:SS)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep extracted frames after analysis")
    parser.add_argument("--max-frames", type=int, help="Cap on frames sent to the model")
    parser.add_argument("--fps", type=float, dest="fps",
                        help="Sample density in frames-per-second (overrides the mode preset; "
                             "clamped to the source frame rate). Use 4-8 for craft review.")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip vision analysis, only extract frames and captions")
    parser.add_argument("--no-audio", action="store_true",
                        help="Do not extract or attach the audio track (frames + transcript only). "
                             "Leave this off for ASMR/music-driven pieces.")
    args = parser.parse_args()

    # Get API key
    api_key = get_api_key()
    if not api_key and not args.no_vision:
        log("ERROR: No GOOGLE_API_KEY found. Set it in .env or environment.")
        sys.exit(1)

    # Create working directory
    workdir = tempfile.mkdtemp(prefix="hermes-video-")
    log(f"Working directory: {workdir}")

    # Determine if URL or local file
    is_url = False
    if args.source.startswith(("http://", "https://", "ftp://")):
        is_url = True
    elif "://" not in args.source and not os.path.exists(args.source):
        # Not a local file and no protocol — might be a scheme-less URL or a missing file
        if "." in args.source.replace("/", ""):
            is_url = True
            if not args.source.startswith("http"):
                args.source = "https://" + args.source
        else:
            log(f"ERROR: '{args.source}' is neither a valid URL nor an existing file path")
            sys.exit(1)

    if is_url and not args.source.startswith(("http://", "https://", "ftp://")):
        args.source = "https://" + args.source

    video_path = None
    title = "Unknown"

    try:
        # Get the video
        if is_url:
            # Parse start/end for segment download if provided
            start_sec = parse_timestamp(args.start) if args.start else 0
            end_sec = parse_timestamp(args.end) if args.end else None

            if args.start or args.end:
                video_path = download_video_limited(args.source, workdir, start_sec, end_sec)

            if not video_path:
                video_path, title = download_video(args.source, workdir)
                if not video_path or not os.path.exists(video_path):
                    log("ERROR: Failed to download video")
                    sys.exit(1)
        else:
            # Local file
            video_path = args.source
            if not os.path.exists(video_path):
                log(f"ERROR: File not found: {video_path}")
                sys.exit(1)

        # Get title from local file if not set
        if not is_url or title == "Unknown":
            title = Path(video_path).name

        # Extract captions
        captions_file = None
        captions_text = ""
        # Try yt-dlp captions for both URLs and local files (yt-dlp can extract embedded subs from local files too)
        try:
            captions_file = extract_captions_ytdlp(args.source if is_url else video_path, workdir)
        except Exception:
            pass

        if not captions_file and not is_url:
            # Check for sidecar caption files alongside the local video file
            local_captions = list(Path(video_path).parent.glob(f"{Path(video_path).stem}.*"))
            for f in local_captions:
                if f.suffix in (".vtt", ".srt", ".txt", ".scc", ".dfxp", ".ass", ".ssa"):
                    captions_file = str(f)
                    break

        # Extract the audio track once. It serves two purposes: transcription when
        # there are no captions, and direct attachment to the analysis request so
        # sound design is analysable (a transcript carries only speech).
        audio_file = None if args.no_audio else extract_audio(video_path, workdir)
        if not audio_file:
            log("No audio track available (or --no-audio)")

        if captions_file:
            segments = parse_captions_vtt(captions_file)
            captions_text = caption_segments_to_text(segments)
            caption_source = "yt-dlp"
            log(f"Got {len(segments)} caption segments from {captions_file}")
        else:
            log("No captions available, transcribing audio via Gemini...")
            if audio_file:
                captions_file, captions_text = transcribe_with_gemini(audio_file, workdir, api_key)
                # Be honest about which of three things happened — the old code
                # reported "gemini-transcribe" even when the call had failed.
                if captions_file is None:
                    caption_source = "transcribe-error"
                    log("Transcription FAILED (API error) — no transcript available")
                elif captions_text.strip():
                    caption_source = "gemini-transcribe"
                else:
                    caption_source = "no-speech"
                    log("Transcription succeeded but found no speech (music/ASMR-only audio)")
            else:
                caption_source = "none"

        # Extract frames. Caps are a token guard for long sources — for shorts the
        # density preset governs, and meta reports whether the cap actually bound.
        if args.mode == "minimal":
            max_frames = args.max_frames or 60
        elif args.mode == "detailed":
            max_frames = args.max_frames or 600
        else:
            max_frames = args.max_frames or 300

        frames, duration, frame_meta = extract_frames_ffmpeg(
            video_path, workdir, mode=args.mode, max_frames=max_frames,
            target_fps=getattr(args, "fps", None),
        )

        if not frames:
            log("ERROR: No frames extracted")
            sys.exit(1)

        # If --no-vision, just output metadata
        if args.no_vision:
            result = {
                "mode": args.mode,
                "title": title,
                "duration_sec": duration,
                "frame_count": len(frames),
                "frame_extraction": frame_meta["method"],
                "frame_sampling": frame_meta,
                "caption_source": caption_source,
                "frames_dir": str(Path(workdir) / "frames"),
                "caption_file": captions_file,
                "frames": [{"timestamp": format_timestamp(ts), "path": fp} for ts, fp in frames],
            }
            output_json(result, args.output)
            log("Extraction complete (no-vision mode)")
            return

        # Send to Gemini for analysis
        analysis = analyze_frames_with_gemini(
            frames, captions_text, title, duration, api_key, args.mode, args.question,
            audio_path=audio_file,
        )

        # A zero-token response is a failure, not an empty finding. Writing it out
        # as a normal report (with "Analysis complete!") is the same lie as the old
        # hardcoded frame_extraction label: it asserts success without evidence.
        if not (analysis or "").strip():
            log("Gemini returned an EMPTY analysis (completion_tokens=0).")
            log("Not writing a report. Re-run, or lower --max-frames.")
            raise SystemExit(2)

        # Build output
        result = {
            "mode": args.mode,
            "title": title,
            "duration_sec": duration,
            "frame_count": len(frames),
            "frame_extraction": frame_meta["method"],
            "frame_sampling": frame_meta,
            "caption_source": caption_source,
            "audio_attached": bool(audio_file),
            "analysis": analysis,
            "frames_dir": str(Path(workdir) / "frames"),
            "caption_file": captions_file,
        }

        output_json(result, args.output)
        log("Analysis complete!")

    finally:
        if not args.keep_frames and not args.output:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def output_json(result, output_path=None):
    """Print or save JSON result."""
    output = json.dumps(result, indent=2, default=str)
    if output_path:
        Path(output_path).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
