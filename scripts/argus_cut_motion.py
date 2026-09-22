"""
CUT MOTION CONTINUITY — does the motion the eye is tracking survive the cut?

A cut is smooth when the motion the viewer is already following carries across it. It feels
jarring when the incoming shot's motion contradicts the outgoing shot's motion in a way that
reads as an accident rather than a choice.

The mechanism this measures:

  PUSH-IN / dolly toward the subject  ->  optical flow FANS OUTWARD radially from the frame
                                          centre. Flow directions are INCOHERENT (they point
                                          every which way), because each part of the image
                                          moves away from the vanishing point.
  TRUCK / pan / lateral move          ->  optical flow is UNIFORM: nearly every vector points
                                          the same way. Flow is highly COHERENT.

So a push-in cutting to a track hands the eye a completely different motion field at the cut,
even if both shots are individually beautiful. That is quantifiable, and this script reports it.

Per cut it reports, for a window of frames either side:
  speed      mean flow magnitude, in pixels per frame
  direction  circular mean of the flow direction, in degrees (0 = right, 90 = down on screen)
  coherence  |mean vector| / mean magnitude.  ~1.0 = uniform lateral move.  low = fanning
             radial move. This single number separates a dolly from a truck.
  radial     signed radial component: positive = flowing outward (push in), negative = inward
  type       static / radial (push/pull) / lateral (truck/pan) / mixed, from the above

Then the verdict at the cut, comparing outgoing to incoming:
  MATCHED     direction within ANGLE_TOL and coherence class the same
  CONTRAST    both moves strong and directions intentionally opposed (a deliberate reversal)
  MISMATCH    direction differs by CUT_ANGLE_WARN..180-... or a radial<->lateral type change
              -> this is the "jarring" class: unmatched motion vector, accidental-looking

Thresholds are craft heuristics, not published perceptual limits. See the honest note in
notes-11-perception-and-thresholds.md: treat ANGLE_TOL as a flag for human review, not a
scientific pass/fail.

Usage:
  python3 argus_cut_motion.py VIDEO [--window 8] [--cuts 3.88,7.88] [--json out.json]
"""
import argparse, json, math, subprocess, sys
from collections import Counter

import cv2
import numpy as np


def probe(video):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", video],
        capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    st = d["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    return dict(w=int(st["width"]), h=int(st["height"]), fps=float(num) / float(den),
                duration=float(d["format"]["duration"]))


def frames(video, scale_w=480):
    """Yield (index, gray image) downscaled, so flow is fast and scale-consistent."""
    info = probe(video)
    w = scale_w
    h = int(round(info["h"] * scale_w / info["w"]))
    cmd = ["ffmpeg", "-v", "error", "-i", video,
           "-vf", "scale=%d:%d,format=gray" % (w, h), "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h
    i = 0
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        yield i, np.frombuffer(buf, np.uint8).reshape(h, w).copy(), info
        i += 1
    p.stdout.close()
    p.wait()


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)


def describe(fl, cx, cy):
    """Reduce a flow field to speed / direction / coherence / radial character."""
    mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1], angleInDegrees=True)
    # ignore near-zero flow, which is noise
    m = mag > 0.15
    if m.sum() < 50:
        return dict(speed=0.0, direction=0.0, coherence=0.0, radial=0.0, type="static", px=int(m.sum()))
    vy, vx = fl[..., 1][m], fl[..., 0][m]
    speed = float(np.mean(mag[m]))
    mean_vx, mean_vy = float(np.mean(vx)), float(np.mean(vy))
    coherence = math.hypot(mean_vx, mean_vy) / max(speed, 1e-6)
    direction = math.degrees(math.atan2(mean_vy, mean_vx)) % 360

    # radial component: project each vector onto the direction away from frame centre
    ys, xs = np.nonzero(m)
    ux, uy = xs - cx, ys - cy
    norm = np.hypot(ux, uy)
    keep = norm > 1.0
    radial = 0.0
    if keep.sum() > 20:
        ux, uy, n2 = ux[keep] / norm[keep], uy[keep] / norm[keep], norm[keep]
        radial = float(np.mean((vx[keep] * ux + vy[keep] * uy)))

    # near = bottom third (foreground, where parallax shows), far = top third (background,
    # dominated by rotation and perspective). A push that pushes through depth shows a strong
    # near band; a move that merely slides shows the two bands agreeing.
    near = band_stats(fl, 0.62, 1.0, cx, cy)
    far = band_stats(fl, 0.0, 0.38, cx, cy)
    parallax = abs(angdiff_full(near["dir"], far["dir"]))

    if speed < 0.35:
        t = "static"
    elif coherence >= 0.55:
        t = "lateral"
    elif abs(radial) > 0.5 * speed:
        t = "radial"
    else:
        t = "mixed"
    return dict(speed=speed, direction=direction, coherence=coherence,
                radial=radial, type=t, px=int(m.sum()),
                near_speed=near["speed"], near_dir=near["dir"], near_coh=near["coh"],
                far_speed=far["speed"], far_dir=far["dir"], far_coh=far["coh"],
                parallax=parallax)


def band_stats(fl, y0f, y1f, cx, cy):
    """Flow statistics for a horizontal band of the frame, given as fractions of height."""
    h = fl.shape[0]
    y0, y1 = int(h * y0f), int(h * y1f)
    sub = fl[y0:y1, :, :]
    mag, _ = cv2.cartToPolar(sub[..., 0], sub[..., 1], angleInDegrees=True)
    m = mag > 0.15
    if m.sum() < 30:
        return dict(speed=0.0, dir=0.0, coh=0.0)
    vx, vy = sub[..., 0][m], sub[..., 1][m]
    speed = float(np.mean(mag[m]))
    rad = np.radians(np.degrees(np.arctan2(vy, vx)))
    direction = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360.0)
    return dict(speed=speed, dir=direction,
                coh=math.hypot(vx.mean(), vy.mean()) / max(speed, 1e-6))


def angdiff_full(a, b):
    d = abs((a - b) % 360.0)
    return 360.0 - d if d > 180 else d


angdiff = angdiff_full


def analyse(video, window=8, cuts=None, angle_tol=25.0, motion_min=0.6):
    W = window
    flows = []          # (frame_index_of_pair, flow, describe)
    allf = []
    cx = cy = None
    for i, g, info in frames(video):
        allf.append(g)
        if len(allf) > 2:
            allf.pop(0)
        if len(allf) == 2:
            fl = flow(allf[0], allf[1])
            cy, cx = fl.shape[0] / 2.0, fl.shape[1] / 2.0
            flows.append((i, fl, describe(fl, cx, cy)))
    if not flows:
        raise SystemExit("no frames decoded")

    # cut detection: between consecutive flows, a large frame difference
    if cuts is None:
        diffs = []
        for i, g, _ in frames(video):
            allf.append(g)
            if len(allf) > 2:
                allf.pop(0)
            if len(allf) == 2:
                diffs.append((i, float(np.mean(np.abs(allf[1].astype(np.int16) -
                                                     allf[0].astype(np.int16))))))
        if diffs:
            vals = np.array([d for _, d in diffs])
            thr = max(vals.mean() + 2.5 * vals.std(), vals.mean() * 2.2)
            info = probe(video)
            cuts = [i / info["fps"] for i, d in diffs if d > thr and i > 2]
    if not cuts:
        print("no cuts supplied or detected")
        return None

    fps = probe(video)["fps"]
    report = []
    print("=" * 78)
    print("CUT MOTION CONTINUITY   %s" % video.split("/")[-1])
    print("=" * 78)
    for c in cuts:
        cf = int(round(c * fps))
        before = [d for i, _, d in flows if cf - W <= i < cf]
        after = [d for i, _, d in flows if cf <= i < cf + W]
        if len(before) < 2 or len(after) < 2:
            print("  cut %.2fs: not enough frames either side" % c)
            continue
        # Two traps here. (1) "type" is a string, so a naive mean over every key raises.
        # (2) Direction is an ANGLE: a linear mean of 359 and 1 gives 180, exactly backwards.
        # Average angles as unit vectors, and take the modal type as the window's character.
        def summarise(win):
            out = {k: float(np.mean([d[k] for d in win]))
                   for k in ("speed", "coherence", "radial", "px", "parallax",
                             "near_speed", "near_coh", "far_speed", "far_coh")}
            for key in ("near_dir", "far_dir"):
                rad = np.radians([d[key] for d in win])
                out[key] = float(np.degrees(np.arctan2(np.sin(rad).mean(),
                                                       np.cos(rad).mean())) % 360.0)
            rad = np.radians([d["direction"] for d in win])
            out["direction"] = float(np.degrees(np.arctan2(np.sin(rad).mean(),
                                                           np.cos(rad).mean())) % 360.0)
            out["type"] = Counter(d["type"] for d in win).most_common(1)[0][0]
            return out

        B = summarise(before)
        A = summarise(after)
        ang = angdiff(B["direction"], A["direction"])
        ratio = A["speed"] / max(B["speed"], 1e-6)
        typ = (B["type"], A["type"])

        if B["speed"] < motion_min or A["speed"] < motion_min:
            verdict = "STATIC SIDE"
        elif typ[0] != typ[1] and set(typ) & {"radial", "lateral"}:
            verdict = "MISMATCH (type change)"
        elif ang <= angle_tol:
            verdict = "MATCHED"
        elif ang >= 180 - angle_tol:
            verdict = "CONTRAST (reversal)"
        elif ang > 90:
            verdict = "MISMATCH (opposed)"
        else:
            verdict = "MISMATCH (angle)"

        print("  cut at %.2fs  (frame %d)" % (c, cf))
        print("    outgoing  %-7s speed %5.2f px/f  dir %6.1f deg  coherence %.2f  radial %+5.2f"
              % (B["type"], B["speed"], B["direction"], B["coherence"], B["radial"]))
        print("    incoming  %-7s speed %5.2f px/f  dir %6.1f deg  coherence %.2f  radial %+5.2f"
              % (A["type"], A["speed"], A["direction"], A["coherence"], A["radial"]))
        print("    near band (foreground)  speed %5.2f px/f  dir %6.1f  coherence %.2f"
              % (B["near_speed"], B["near_dir"], B["near_coh"]))
        print("    far  band (background)  speed %5.2f px/f  dir %6.1f  coherence %.2f"
              % (B["far_speed"], B["far_dir"], B["far_coh"]))
        print("    OUTGOING parallax (near vs far direction split) %.0f deg" % B["parallax"])
        print("    INCOMING parallax %.0f deg" % A["parallax"])
        print("    angle between %.0f deg   speed ratio %.2fx   type %s -> %s"
              % (ang, ratio, typ[0], typ[1]))
        print("    VERDICT: %s" % verdict)
        if verdict.startswith("MISMATCH"):
            print("      Why it jars: the eye is tracking one motion field and the cut hands it")
            print("      another. Either match the vector, or make the change big enough to read")
            print("      as a deliberate change of coverage - the slight-difference middle is what")
            print("      reads as a mistake.")
        report.append(dict(cut=c, outgoing=B, incoming=A, angle=ang, speed_ratio=ratio,
                           types=typ, verdict=verdict))
        print()
    print("  NOTE: angle_tol %.0f deg and the coherence cut at 0.55 are craft heuristics, not" % angle_tol)
    print("  published perceptual thresholds (see notes-11). Treat MISMATCH as 'review this',")
    print("  not as a measured failure of human perception.")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--window", type=int, default=8, help="frames either side of the cut")
    ap.add_argument("--cuts", default=None, help="comma-separated cut times in seconds")
    ap.add_argument("--json", default=None)
    ap.add_argument("--angle-tol", type=float, default=25.0)
    a = ap.parse_args()
    cuts = [float(x) for x in a.cuts.split(",")] if a.cuts else None
    rep = analyse(a.video, a.window, cuts, a.angle_tol)
    if a.json and rep:
        json.dump(rep, open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)


if __name__ == "__main__":
    main()
