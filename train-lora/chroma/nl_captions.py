#!/usr/bin/env python3
"""Guided natural-language captions for Chroma LoRA training.

Chroma reads captions with a T5 text encoder: it wants full sentences, not
tag soup. This helper walks you through your dataset image by image and
writes one rich caption per photo (a .txt next to each image) in the format
that worked in our training runs:

    A photo of {trigger}, {subject} {what varies}. {framing}. photorealistic.

GOLDEN RULE — describe the PHOTO, not the character. Whatever you do NOT
describe, the trigger word absorbs as "the character". So describe only what
CHANGES between photos (outfit, place, light, expression) — and any photo
where your character's look DEVIATES from canon (e.g. an accent hair color
spreading wider than usual), say so in that photo's caption to protect the
canonical look.

Run it from inside your dataset folder (double-click nl_captions.bat).
Takes ~5 minutes for 15-25 images. 100% local, nothing to download.
"""
import sys
from pathlib import Path

EXTS = (".png", ".jpg", ".jpeg")


def ask(prompt, default):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        val = ""
    return val if val else default


def main():
    folder = Path.cwd()
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)
    if not images:
        print("No images (.png/.jpg) found in this folder.")
        print("Put nl_captions.bat + nl_captions.py INSIDE your dataset folder and run it there.")
        return 1

    print()
    print("=" * 68)
    print("  Guided captions for Chroma LoRA training")
    print("=" * 68)
    print()
    print("  GOLDEN RULE: describe the PHOTO, not the character.")
    print("  Describe only what VARIES (outfit, place, light, expression).")
    print("  What you don't describe, your trigger word learns as the character.")
    print()
    print(f"  Found {len(images)} image(s). Press Enter to accept [defaults].")
    print()

    trigger = ask("Your trigger word (a made-up word)", "cmfychar")
    subject = ask("What is it - a woman / a man / a robot / a dog...", "a woman")

    existing = [p for p in images if p.with_suffix(".txt").exists()]
    redo_existing = True
    if existing:
        print()
        print(f"NOTE: {len(existing)} of these images already have a .txt caption.")
        choice = ask("(s)kip the ones that have captions, or (r)edo all?", "s").lower()
        redo_existing = choice.startswith("r")

    print()
    print("Now one line per photo. Look at the image, type what varies in it.")
    print("Example:  in a denim jacket, in a cozy cafe, warm light, laughing")
    print("Empty Enter = same answer as the previous photo (fast for similar shots).")
    print()

    prev_varies = "in a plain t-shirt, neutral background, soft light, neutral expression"
    prev_framing = "close-up portrait"
    planned = []
    for i, img in enumerate(images, 1):
        if img.with_suffix(".txt").exists() and not redo_existing:
            print(f"[{i}/{len(images)}] {img.name} - already captioned, skipping.")
            continue
        print(f"[{i}/{len(images)}] {img.name}")
        varies = ask("  what varies in this photo? (outfit, place, light, expression)", prev_varies)
        framing = ask("  framing? (close-up portrait / upper body / full body)", prev_framing)
        prev_varies, prev_framing = varies, framing
        caption = f"A photo of {trigger}, {subject} {varies}. {framing}. photorealistic."
        planned.append((img, caption))

    if not planned:
        print("\nNothing to write (everything already captioned). Done.")
        return 0

    print()
    print("-" * 68)
    print("Review:")
    for img, cap in planned:
        print(f"  {img.name}")
        print(f"    {cap}")
    print("-" * 68)
    ok = ask(f"Write {len(planned)} caption file(s)?", "Y").lower()
    if ok.startswith("n"):
        print("Cancelled - nothing was written.")
        return 0

    for img, cap in planned:
        img.with_suffix(".txt").write_text(cap + "\n", encoding="utf-8")
    print()
    print(f"Done: {len(planned)} caption file(s) written.")
    print("Tip: open a couple of .txt files and sanity-check them. If a photo shows")
    print("your character off-canon (accent color spread, different hair tint), add")
    print("that to ITS caption - it keeps the trigger's canonical look clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
