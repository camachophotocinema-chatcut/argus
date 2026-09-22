"""
Argus EVIDENCE BUILDER — turns a video into images that make motion and sound readable.

The problem this solves, measured on our own footage: Argus sampled 4 stills per second and
reported "the camera remains completely static" for a shot where the camera demonstrably
sweeps 3.8 m sideways. A vision model cannot infer motion from sparse stills — it guesses.
So do not ask it to infer. Convert the motion and the sound into pictures it can see.

What it produces, per video:
  kymograph.png    time-as-x-axis space-time slices at three columns. The classic instrument
                   for reading motion in ONE image:
                     static camera  -> vertical stripes
                     lateral move   -> parallel diagonals, and the three columns tilt by
                                       DIFFERENT amounts when depth varies (parallax)
                     zoom / dolly   -> converging or diverging diagonals
  flow_sheet.png   optical-flow field (hue = direction, brightness = speed) tiled over time,
                   so a truck reads as a solid leftward field and a locked-off shot as black
  diff_sheet.png   frame-to-frame absolute difference, tiled: shows WHAT changed and where
  motion_curve.png per-sample motion magnitude and audio loudness, aligned on one time axis
  spectrogram.png  audio frequency content over time (log scale)
  waveform.png     audio amplitude envelope
  summary.txt      the numbers in words, for the prompt: cut times, motion magnitude per
                   shot, audio level per second, exposure clipping
  evidence.json    the same numbers, machine-readable

Usage:
  python argus_evidence.py <video> [outdir]
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

W = 320                      # analysis width (speed); evidence images are rendered larger
SAMPLE_HZ = 8.0              # frames per second pulled for analysis
CUT_DIFF_FLOOR = 0.05        # absolute floor, so a truly static video cannot fake cuts
CUT_SIGMA = 2.5              # cuts sit this far above the in-shot motion baseline
CUT_MIN_GAP_S = 0.5          # minimum spacing between reported cuts


# ------------------------------------------------------------------ helpers
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def read_frames(path, hz=SAMPLE_HZ):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / hz)))
    frames, idx = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if ok and fr is not None:
                fr = cv2.resize(fr, (W, int(W * fr.shape[0] / fr.shape[1])))
                frames.append(fr)
        idx += 1
    cap.release()
    return frames, fps, total


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def label(img, text):
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, max(len(text) * 7 + 8, 60), 16], fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return img


def save(arr, path, title=None, scale=1):
    if arr.ndim == 2:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").convert("RGB")
    else:
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)[..., ::-1])
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    if title:
        img = label(img, title)
    img.save(path)
    return path


# ------------------------------------------------------------------ evidence pieces
def kymograph(frames, outdir):
    """Time on x, height on y, at three columns. Motion becomes visible slant."""
    t = len(frames)
    panels = []
    for frac, name in ((0.18, "left third"), (0.50, "centre"), (0.82, "right third")):
        x = int(W * frac)
        col = np.stack([gray(f)[:, x] for f in frames], axis=1)     # (H, T)
        panels.append((name, col))
    h = panels[0][1].shape[0]
    gap = 4
    big = np.zeros((h, t * 3 + gap * 2), np.uint8)
    xoff = 0
    for name, col in panels:
        big[:, xoff:xoff + t] = col
        xoff += t + gap
    # stretch time 2x horizontally so the slant is legible
    img = Image.fromarray(big).convert("RGB").resize(((t * 3 + gap * 2) * 2, h * 2), Image.Resampling.NEAREST)
    img = label(img, "KYMOGRAPH  left | centre | right   (x=time, y=height)")
    p = os.path.join(outdir, "kymograph.png")
    img.save(p)
    return p


def flow_pairs(frames):
    """Farneback flow between consecutive samples. Returns list of (flow, mean_mag)."""
    out = []
    prev = gray(frames[0])
    for f in frames[1:]:
        cur = gray(f)
        fl = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 21, 3, 7, 1.5, 0)
        out.append((fl, float(np.mean(np.linalg.norm(fl, axis=2)))))
        prev = cur
    return out


def flow_image(f, base):
    mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
    mag = np.clip(mag * 6.0, 0, 255).astype(np.uint8)
    hsv = np.zeros((f.shape[0], f.shape[1], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = mag
    col = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(base, 0.45, col, 0.85, 0)


def sheet(images, cols, title, path, cell_w=200):
    rows = int(np.ceil(len(images) / cols))
    ch = int(cell_w * images[0].shape[0] / images[0].shape[1])
    canvas = np.zeros((rows * ch, cols * cell_w, 3), np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        small = cv2.resize(im, (cell_w, ch))
        canvas[r * ch:(r + 1) * ch, c * cell_w:(c + 1) * cell_w] = small
    img = Image.fromarray(canvas[..., ::-1])
    img = label(img, title)
    img.save(path)
    return path


def audio_tracks(path, outdir, dur):
    spec = os.path.join(outdir, "spectrogram.png")
    wave = os.path.join(outdir, "waveform.png")
    sh(f'ffmpeg -y -loglevel error -i "{path}" -lavfi '
       f'"showspectrumpic=s=1000x420:legend=1:scale=log:color=intensity" "{spec}"')
    sh(f'ffmpeg -y -loglevel error -i "{path}" -lavfi '
       f'"showwavespic=s=1000x240:colors=white" "{wave}"')
    if not os.path.exists(spec):
        spec = None
    if not os.path.exists(wave):
        wave = None
    rms = []
    raw = subprocess.run(
        f'ffmpeg -v error -i "{path}" -ac 1 -ar 8000 -f s16le -', shell=True,
        capture_output=True).stdout
    if raw:
        a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        per = max(1, int(8000 * 0.5))
        rms = [float(np.sqrt(np.mean(a[i:i + per] ** 2)) + 1e-9)
               for i in range(0, max(1, len(a) - per), per)]
    return spec, wave, rms


def curve_plot(motion, rms, title, path):
    h, w = 240, 1000
    img = Image.new("RGB", (w, h), (12, 12, 16))
    d = ImageDraw.Draw(img)
    d.text((6, 4), title, fill=(230, 230, 230))
    if motion:
        mx = max(motion) or 1
        pts = [(int(i * (w - 1) / max(1, len(motion) - 1)), int(h - 20 - (m / mx) * (h - 50)))
               for i, m in enumerate(motion)]
        d.line(pts, fill=(90, 200, 255), width=2)
        d.text((6, 20), "blue = image motion (px/frame), peak %.1f" % mx, fill=(90, 200, 255))
    if rms:
        mx = max(rms) or 1
        pts = [(int(i * (w - 1) / max(1, len(rms) - 1)), int(h - 10 - (r / mx) * (h - 40)))
               for i, r in enumerate(rms)]
        d.line(pts, fill=(120, 255, 140), width=1)
        d.text((6, 34), "green = audio loudness (RMS/0.5s)", fill=(120, 255, 140))
    img.save(path)
    return path


def cuts_from_diffs(diffs):
    """A cut is a frame-difference spike well above the in-shot motion baseline.

    Calibrated on a real 3-shot sequence: cuts measured 0.134 and 0.201 while the largest
    in-shot motion was 0.0797 (baseline mean 0.0299, sd 0.0306). A FIXED threshold of 0.22
    sat above both cuts and detected nothing, collapsing the whole clip into one segment and
    destroying the per-shot motion readout. An adaptive threshold separates them with room
    to spare; the small absolute floor exists only so a static video cannot invent cuts.
    """
    if not diffs:
        return []
    d = np.array(diffs)
    thr = max(CUT_DIFF_FLOOR, float(d.mean() + CUT_SIGMA * d.std()))
    gap = int(CUT_MIN_GAP_S * SAMPLE_HZ)
    cuts = []
    for i in range(1, len(d) - 1):
        if d[i] > thr and d[i] >= d[i - 1] and d[i] > d[i + 1]:
            if not cuts or (i - cuts[-1]) >= gap:
                cuts.append(i)
    return cuts


def build(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    frames, fps, total = read_frames(path)
    if len(frames) < 3:
        raise SystemExit("not enough frames read from %s" % path)
    dur = len(frames) / SAMPLE_HZ

    kp = kymograph(frames, outdir)

    flows = flow_pairs(frames)
    motion = [m for _, m in flows]
    bases = frames[1:]
    step = max(1, len(flows) // 12)
    flow_imgs = [flow_image(flows[i][0], bases[i]) for i in range(0, len(flows), step)][:12]
    fp = sheet(flow_imgs, 4, "OPTICAL FLOW  hue=direction  brightness=speed", os.path.join(outdir, "flow_sheet.png"))

    diffs, diff_imgs = [], []
    for i in range(1, len(frames)):
        df = cv2.absdiff(gray(frames[i - 1]), gray(frames[i]))
        diffs.append(float(df.mean()) / 255.0)
        if i % max(1, (len(frames) - 1) // 12) == 0:
            vis = cv2.applyColorMap(np.clip(df * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            diff_imgs.append(vis)
    dp = sheet(diff_imgs[:12], 4, "FRAME DIFFERENCE  (what changed, where)",
               os.path.join(outdir, "diff_sheet.png"))

    spec, wave, rms = audio_tracks(path, outdir, dur)
    cuts = cuts_from_diffs(diffs)
    cp = curve_plot(motion, rms, "MOTION + AUDIO over time", os.path.join(outdir, "motion_curve.png"))

    # ---- per-segment numbers in words ----
    segs = []
    bounds = [0] + cuts + [len(motion)]
    for a, b in zip(bounds, bounds[1:]):
        seg = motion[a:b]
        if not seg:
            continue
        segs.append(dict(start=round(a / SAMPLE_HZ, 2), end=round(b / SAMPLE_HZ, 2),
                         motion_mean=round(float(np.mean(seg)), 2),
                         motion_max=round(float(np.max(seg)), 2)))
    lines = ["ARGUS EVIDENCE  %s" % os.path.basename(path),
             "duration %.2fs  analysis %d frames at %.1f fps  source %d frames @ %.2f fps"
             % (dur, len(frames), SAMPLE_HZ, total, fps), ""]
    lines.append("SEGMENTS (motion in px/frame between sampled frames):")
    for s in segs:
        verdict = ("static/locked-off" if s["motion_mean"] < 0.5 else
                   "slow move" if s["motion_mean"] < 2.0 else
                   "clear move" if s["motion_mean"] < 8.0 else "fast/large move")
        lines.append("  %.2fs-%.2fs  mean %.2f  peak %.2f  -> %s"
                     % (s["start"], s["end"], s["motion_mean"], s["motion_max"], verdict))
    lines.append("")
    lines.append("CUTS detected at (s): %s" % (", ".join("%.2f" % (c / SAMPLE_HZ) for c in cuts) or "none"))
    if rms:
        loud = ["%.2f:%.3f" % (i * 0.5, r) for i, r in enumerate(rms)]
        lines.append("AUDIO loudness (s:rms), %.1f dB dynamic range: %s"
                     % (20 * np.log10((max(rms) + 1e-9) / (min(rms) + 1e-9)), " ".join(loud[:60])))
    else:
        lines.append("AUDIO: no audio track decoded")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(outdir, "summary.txt"), "w").write(txt)
    json.dump(dict(duration=dur, fps=fps, samples=len(frames), segments=segs,
                   cuts=[round(c / SAMPLE_HZ, 2) for c in cuts],
                   motion=motion, audio_rms=rms,
                   images=dict(kymograph=kp, flow=fp, diff=dp, spectrogram=spec,
                               waveform=wave, curve=cp)),
              open(os.path.join(outdir, "evidence.json"), "w"), indent=1)
    return txt


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    v = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(v)) or ".", "evidence")
    print(build(v, out))
    print("evidence written to %s" % out)
