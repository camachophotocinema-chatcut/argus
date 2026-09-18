# Argus Download Pipeline

Three strategies, tried in order of preference.

## Strategy 1: Direct URL → curl

**Trigger**: URL ends in `.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`, `.m4v`, `.gif`, `.mpg`, `.mpeg`
  (checked in `is_direct_video_url()` after stripping query/fragment)

**Behavior**:
1. `curl -L -o <workdir>/video<ext> <url>` with 300s timeout
2. Returns immediately with the file path (no yt-dlp overhead)

**Pros**: Fast, no yt-dlp dependency, no HTTP headers sent by a scraper tool
**Cons**: No metadata (title defaults to URL filename), caption extraction skipped unless embedded

**Edge cases**:
- URL has no recognizable extension → falls through to strategy 2 (yt-dlp)
- curl fails (404, timeout, SSL error) → falls through to strategy 2
- Server redirects to an HTML page → curl downloads the HTML as "video.mp4" → ffmpeg fails later → script errors on analysis. Mitigation: ffmpeg reading failures propagate up naturally.

## Strategy 2: Streaming URL → yt-dlp

**Trigger**: Any URL that doesn't match strategy 1
  (or strategy 1 fails)

**Behavior**:
1. `yt-dlp --print title --skip-download <url>` (instant metadata)
2. `yt-dlp -f "bestvideo*[height<=1080]+bestaudio/best[height<=1080]" --merge-output-format mp4 -o <workdir>/video.%(ext)s <url>`
3. Searches `workdir/video.*` or `workdir/*.mp4` for the output

**Pros**: Handles hundreds of sites (YouTube, X/Twitter, Vimeo, TikTok, Instagram, Reddit, etc.), extracts captions natively
**Cons**: Slower (site scraping + format negotiation), heavier dependency

**Edge cases**:
- yt-dlp times out or errors (private video, geo-blocked, deleted) → falls to strategy 3 (curl, then fail)
- No bestvideo+bestaudio format available (e.g., live streams) → error, no fallback
- YouTube age-restricted content → needs cookies file

## Strategy 3: Local file → in-place

**Trigger**: Source doesn't start with `http://`, `https://`, or `ftp://` AND `os.path.exists()` is true

**Behavior**:
1. Copies nothing — works directly from the given path
2. Title = file basename

**Pros**: Zero download time, works for any format ffmpeg can read
**Cons**: Must already exist on disk, no yt-dlp caption extraction (though yt-dlp is now tried on the local file too for embedded subs)

## Caption resolution order

```
① yt-dlp --write-subs on the source (works for URLs AND local files)
② Sidecar file: <videoname>.vtt, .srt, .txt, .scc, .dfxp, .ass, .ssa in same directory
③ Gemini 3.5 Flash audio transcription (extracts audio via ffmpeg → Gemini API)
```

Caption step ③ is the most fragile:

- **No audio track** → ffmpeg `libmp3lame` extraction fails with exit code 234
  → `extract_audio()` catches the exception and returns `None`
  → `analyze_frames_with_gemini()` proceeds with `caption_text=""` — **vision-only analysis**
  → This is correct behavior — don't make audio extraction a hard failure

- **Short audio** (< 3s) → Gemini's OpenAI-compatible endpoint can return 400 Bad Request
  → Transcription result is empty string
  → Analysis still proceeds without captions

- **Long audio** (> ~15 min) → ffmpeg + Gemini may hit the 120s extraction timeout
  → Bump the timeout in `extract_audio()` or use `--start MM:SS --end MM:SS`

## Test commands

```bash
# Direct URL (curl path)
python3 scripts/analyze_video.py "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4" --mode minimal --no-vision

# Local file
python3 scripts/analyze_video.py /tmp/test_video.mp4 --mode minimal --no-vision

# YouTube (yt-dlp path)
python3 scripts/analyze_video.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --mode minimal --no-vision
```

Behaviors to verify after code changes:
- Direct URL → curl: check `Downloading direct video URL via curl` in log
- No-audio video → crash: check `Audio extraction failed` is logged, but script continues and exits 0
- Bad URL → error: check `is neither a valid URL nor an existing file path` exits 1
- Scheme-less URL → auto-prefixed: `"example.com/video.mp4"` → `"https://example.com/video.mp4"` → curl path
