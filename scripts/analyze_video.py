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


def run(cmd, timeout=120, check=True):
    """Run a command, return stdout."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode != 0:
        log(f"Command failed: {' '.join(cmd)}")
        log(f"STDERR: {result.stderr[:500]}")
        result.check_returncode()
    return result.stdout.strip()


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


def extract_captions_ytdlp(url, workdir):
    """Extract captions via yt-dlp. Returns path to caption file or None."""
    log("Extracting captions via yt-dlp...")
    try:
        out = run([
            "yt-dlp", "--write-auto-subs", "--sub-langs", "en,en-US,en-GB",
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
            "yt-dlp", "--write-subs", "--sub-langs", "en,en-US,en-GB",
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
            "yt-dlp", "--print", "title", "--skip-download", url
        ], timeout=30)
        title = title_out.strip() or "Unknown"

        # Then download the video
        run([
            "yt-dlp", "-f", "bestvideo*[height<=1080]+bestaudio/best[height<=1080]",
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


def extract_frames_ffmpeg(video_path, workdir, mode="balanced", max_frames=None):
    """
    Extract frames at scene-aware intervals using ffmpeg scene detection.

    Modes:
      - minimal: 1 frame per scene change (cap 30)
      - balanced: scene-aware, sparse within long scenes (cap 100, default)
      - detailed: scene-aware, denser sampling (cap 200)

    Returns list of (timestamp_sec, frame_path) tuples.
    """
    log(f"Extracting frames ({mode} mode)...")

    frames_dir = Path(workdir) / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Mode config
    configs = {
        "minimal": {"scene_threshold": 0.3, "max": 30, "min_interval": 5},
        "balanced": {"scene_threshold": 0.3, "max": 100, "min_interval": 2},
        "detailed": {"scene_threshold": 0.4, "max": 200, "min_interval": 1},
    }
    cfg = configs.get(mode, configs["balanced"])
    if max_frames:
        cfg["max"] = max_frames

    duration = get_video_duration(video_path)
    log(f"Video duration: {duration:.1f}s")

    # Step 1: Run ffmpeg scene detection to find shot boundaries
    log("Running scene detection...")
    try:
        scene_out = run([
            "ffmpeg", "-i", video_path,
            "-filter:v", f"select='gt(scene,{cfg['scene_threshold']})',showinfo",
            "-f", "null", "-",
        ], timeout=120, check=False)
    except Exception as e:
        log(f"Scene detection failed: {e}")
        scene_out = ""

    # Parse scene detection output for timestamps
    scene_times = [0.0]  # Always include first frame
    for match in re.finditer(r"pts_time:([\d.]+)", scene_out):
        ts = float(match.group(1))
        if ts > 0 and ts < duration:
            scene_times.append(ts)

    # If scene detection found nothing useful, fall back to uniform sampling
    if len(scene_times) <= 1:
        log("Scene detection found no cuts, falling back to uniform sampling")
        interval = max(1, int(duration / cfg["max"]))
        scene_times = list(range(0, int(duration), max(interval, 1)))
    else:
        # Filter scene times by min_interval to avoid too-dense frames
        filtered = [0.0]
        for t in sorted(scene_times):
            if t - filtered[-1] >= cfg["min_interval"]:
                filtered.append(t)
        scene_times = filtered

    # Cap at max_frames
    if len(scene_times) > cfg["max"]:
        # Sample evenly from the scene list
        indices = [int(i * len(scene_times) / cfg["max"]) for i in range(cfg["max"])]
        scene_times = [scene_times[i] for i in indices]

    log(f"Extracting {len(scene_times)} frames...")

    # Step 2: Extract each frame
    frames = []
    for i, ts in enumerate(scene_times):
        if ts >= duration:
            continue
        frame_file = frames_dir / f"frame_{i:04d}_{int(ts)}.jpg"
        if not frame_file.exists():
            try:
                run([
                    "ffmpeg", "-ss", str(ts), "-i", video_path,
                    "-vframes", "1", "-q:v", "3",
                    "-y", str(frame_file)
                ], timeout=30, check=False)
            except Exception:
                continue
        frames.append((ts, str(frame_file)))

    log(f"Extracted {len(frames)} frames")
    return frames, duration


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
                {"type": "audio", "audio_url": f"data:audio/mpeg;base64,{audio_b64}"}
            ]
        }],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    import urllib.request

    req = urllib.request.Request(
        gemini_url,
        data=json.dumps(prompt).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        transcript = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        log(f"Transcription length: {len(transcript)} chars")

        # Save transcript
        transcript_file = Path(workdir) / "transcript.txt"
        transcript_file.write_text(transcript)
        return str(transcript_file), transcript
    except Exception as e:
        log(f"Gemini transcription failed: {e}")
        return None, ""


def analyze_frames_with_gemini(frames, captions_text, title, duration, api_key, mode="balanced"):
    """
    Send all extracted frames + captions to Gemini 3.5 Flash for analysis.

    Uses the OpenAI-compatible endpoint of Gemini API.
    """
    log("Sending frames to Gemini 3.5 Flash for analysis...")

    import base64

    # Build the multi-part content
    content_parts = []

    # System-like instruction as first text
    content_parts.append({
        "type": "text",
        "text": (
            "You are analyzing a video frame-by-frame. Below are timestamped frames "
            "extracted at scene-aware intervals, plus caption text.\n\n"
            "For each major section or timestamp cluster, provide:\n"
            "1. **Timestamp** — when this section occurs\n"
            "2. **What's on screen** — describe the visual content (UI, code, people, etc.)\n"
            "3. **What's being said** — key points from the captions at this time\n"
            "4. **Key takeaway** — one-line substance, stripped of hype\n\n"
            "End with an overall summary (3-5 bullet points) that a busy person can read "
            "in 30 seconds. Be concise and specific."
        )
    })

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
            "text": "\n\nNo captions available. Rely on visual frame analysis only."
        })

    # Send to Gemini
    payload = {
        "model": "gemini-3.5-flash",
        "messages": [{"role": "user", "content": content_parts}],
        "max_tokens": 8192,
        "temperature": 0.3,
    }

    gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    import urllib.request

    req = urllib.request.Request(
        gemini_url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    log(f"Sending {frame_count} frames to Gemini...")
    start_time = time.time()

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode())
        elapsed = time.time() - start_time
        log(f"Gemini response in {elapsed:.1f}s")

        analysis = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        log(f"Usage: {json.dumps(usage)}")
        return analysis
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        log(f"Gemini API error {e.code}: {body[:500]}")
        return f"Gemini API error {e.code}: {body[:500]}"
    except Exception as e:
        log(f"Gemini API call failed: {e}")
        return f"Analysis failed: {e}"


def get_api_key():
    """Get Google/Gemini API key."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key and key != "your_google_ai_studio_key_here":
        return key
    # Try hermes .env
    env_path = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_path):
        for line in Path(env_path).read_text().split("\n"):
            if line.startswith("GOOGLE_API_KEY=") and "your_" not in line:
                return line.split("=", 1)[1].strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze a video via Gemini 3.5 Flash")
    parser.add_argument("source", help="YouTube URL or local video file path")
    parser.add_argument("--mode", choices=["minimal", "balanced", "detailed"],
                        default="balanced", help="Frame extraction density")
    parser.add_argument("--start", help="Start timestamp (MM:SS or HH:MM:SS)")
    parser.add_argument("--end", help="End timestamp (MM:SS or HH:MM:SS)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--keep-frames", action="store_true",
                        help="Keep extracted frames after analysis")
    parser.add_argument("--max-frames", type=int, help="Override max frame count")
    parser.add_argument("--no-vision", action="store_true",
                        help="Skip vision analysis, only extract frames and captions")
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

        if captions_file:
            segments = parse_captions_vtt(captions_file)
            captions_text = caption_segments_to_text(segments)
            caption_source = "yt-dlp"
            log(f"Got {len(segments)} caption segments from {captions_file}")
        else:
            log("No captions available, transcribing audio via Gemini...")
            audio_file = extract_audio(video_path, workdir)
            if audio_file:
                captions_file, captions_text = transcribe_with_gemini(audio_file, workdir, api_key)
                caption_source = "gemini-transcribe"
            else:
                caption_source = "none"

        # Extract frames
        if args.mode == "minimal":
            max_frames = args.max_frames or 30
        elif args.mode == "detailed":
            max_frames = args.max_frames or 200
        else:
            max_frames = args.max_frames or 100

        frames, duration = extract_frames_ffmpeg(
            video_path, workdir, mode=args.mode, max_frames=max_frames
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
                "frame_extraction": "scene-aware",
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
            frames, captions_text, title, duration, api_key, args.mode
        )

        # Build output
        result = {
            "mode": args.mode,
            "title": title,
            "duration_sec": duration,
            "frame_count": len(frames),
            "frame_extraction": "scene-aware (ffmpeg scene detection)",
            "caption_source": caption_source,
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
