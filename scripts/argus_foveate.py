"""
Argus FOVEATION — full frame for context, tight high-resolution crops for detail.

Human vision is not uniform: a small fovea carries fine detail while the periphery carries
layout. A vision model fed one downscaled full frame gets neither. This produces both:

  full_<frame>.png            the whole frame (peripheral: layout, framing, where things are)
  fovea_<frame>_<subject>.png a crop around one subject, upscaled to FOVEA_PX (foveal:
                              fine detail — silhouette, proportions, defects)

Boxes come from Blender (`dump_boxes.py`) for anything I built, so they are exact rather
than detected. For external footage a detector would be needed; nothing here fakes that.

Usage:
  python argus_foveate.py <frames_dir|video> <boxes.json> <outdir> [--full-every N]
"""
import argparse
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

FOVEA_PX = 704          # target width of a crop after upscaling
PAD = 0.22              # fraction of box size added as margin, so nothing is clipped


def frames_from_dir(d):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    return [(f, os.path.join(d, f)) for f in fs]


def frames_from_video(path, outdir, every):
    os.makedirs(outdir, exist_ok=True)
    subprocess.run('ffmpeg -y -loglevel error -i "%s" -vf fps=1/%d "%s/src_%%04d.png"'
                   % (path, every, outdir), shell=True)
    return frames_from_dir(outdir)


def frame_number_from_name(name):
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else None


def stamp(img, text):
    """Burn a citable ID into the image.

    Chain-of-Frames (CVPR 2026) embeds temporal grounding in the model's own reasoning by
    making it reference frames by identifier. An ID that exists only in my index is invisible
    to the model; an ID printed IN the pixels is not, and it makes an unsourced claim
    checkable — if the model says "the cat drifts out of frame", it must name the frame.
    """
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, max(78, len(text) * 8 + 10), 18], fill=(0, 0, 0))
    d.text((4, 4), text, fill=(255, 255, 0))
    return img


def crop_box(box, w, h, pad=PAD):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    px0, py0 = max(0, int(x0 * w)), max(0, int(y0 * h))
    px1, py1 = min(w, int(x1 * w)), min(h, int(y1 * h))
    if px1 - px0 < 8 or py1 - py0 < 8:
        return None
    return (px0, py0, px1, py1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("boxes")
    ap.add_argument("outdir")
    ap.add_argument("--full-every", type=int, default=1,
                    help="write a full frame every N sampled frames (context is cheap to skip)")
    a = ap.parse_args()

    boxes = json.load(open(a.boxes))
    by_frame = {r["frame"]: r["subjects"] for r in boxes.get("frames", [])}
    step = boxes.get("every", 1)
    sampled = sorted(by_frame)

    if os.path.isdir(a.source):
        pairs = frames_from_dir(a.source)
    else:
        pairs = frames_from_video(a.source, os.path.join(a.outdir, "src"), step)

    os.makedirs(a.outdir, exist_ok=True)
    index, wrote_full = [], 0

    for i, (name, path) in enumerate(pairs):
        n = frame_number_from_name(name)
        if n is None:
            continue
        # map a rendered frame number onto the sampled box entry at or before it
        cands = [f for f in sampled if f <= n]
        key = cands[-1] if cands else (sampled[0] if sampled else None)
        subs = by_frame.get(key, {})
        try:
            im = Image.open(path).convert("RGB")
        except Exception as e:
            print("  skip %s (%s)" % (name, e))
            continue
        w, h = im.size
        entry = {"frame": n, "source": path, "size": [w, h], "crops": [], "full": None}

        if i % a.full_every == 0:
            fp = os.path.join(a.outdir, "full_%04d.png" % n)
            stamp(im.copy(), "FRAME-%04d FULL" % n).save(fp)
            entry["full"] = fp
            wrote_full += 1

        for sname, box in subs.items():
            pb = crop_box(box, w, h)
            if not pb:
                continue
            c = im.crop(pb)
            if c.width < FOVEA_PX:
                f = FOVEA_PX / c.width
                c = c.resize((int(c.width * f), int(c.height * f)), Image.Resampling.LANCZOS)
            cp = os.path.join(a.outdir, "fovea_%04d_%s.png" % (n, sname.replace("SUBJ_", "")))
            stamp(c.copy(), "FRAME-%04d  %s" % (n, sname.replace("SUBJ_", "").upper())).save(cp)
            entry["crops"].append({"subject": sname, "path": cp, "box_px": pb,
                                   "box_norm": [round(v, 4) for v in box]})
        index.append(entry)

    meta = os.path.join(a.outdir, "fovea_index.json")
    json.dump({"source": a.source, "frames": index, "fovea_px": FOVEA_PX, "pad": PAD},
              open(meta, "w"), indent=1)
    ncrops = sum(len(e["crops"]) for e in index)
    print("foveation: %d frames -> %d full frames + %d foveal crops" % (len(index), wrote_full, ncrops))
    print("index: %s" % meta)
    for e in index[:2]:
        for c in e["crops"]:
            print("  frame %d  %-10s crop %dx%d px  from box %s"
                  % (e["frame"], c["subject"].replace("SUBJ_", ""),
                     Image.open(c["path"]).size[0], Image.open(c["path"]).size[1], c["box_px"]))


if __name__ == "__main__":
    main()
