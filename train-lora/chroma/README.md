# Train a Character LoRA (Chroma, OneTrainer) — step by step

📺 **Tutorial video:** coming with the premiere

> You already ran **train-lora-chroma.bat**, so the models, the work folders, OneTrainer and the
> training preset are in place. If you followed our **SDXL LoRA video**, your existing OneTrainer
> was reused — nothing was reinstalled — and your SDXL files were not touched.

**The idea:** teach Chroma1-HD your own original character with a small LoRA, 100% local and free.
Same pipeline as the SDXL episode — generate → curate → caption → train → use — with two Chroma
twists: captions are **full sentences** (Chroma reads them with a T5 encoder, not tags), and the
recipe uses RAM offload so it fits a **12 GB** card.

1. **Hardware.** 12 GB VRAM is enough (the preset offloads to system RAM). ⚠️ 12 GB card:
   training peaks at **11.96 / 12 GB** — close everything else (browser included) before you
   hit Start. ⚠️ The bf16 recipe peaks around **53 GB of system RAM** — 64 GB recommended.
   Only 32 GB? Open the preset in OneTrainer and switch the **transformer weight dtype to
   FLOAT_8** (small quality trade-off, still no GGUF needed).
2. **Disk.** The installer downloaded ~24 GB of models for ComfyUI. Your **first training run**
   downloads the training copy of the base model (~18 GB) into the Hugging Face cache — leave
   ~45 GB free in total.
3. **ComfyUI not in the default place?** Everything assumes `C:\ComfyUI_windows_portable`. If
   yours is elsewhere, re-point the red fields in OneTrainer with the `[...]` button.

Your work folder (created by the installer): **`Downloads\train-lora-chroma\`**
(`raw_gen` / `dataset` / `workspace` / `output`).

## Step A — Generate the images → 📁 `Downloads\train-lora-chroma\raw_gen\`
👉 *Already have a dataset from the SDXL video? You can reuse those exact images — that's what we
did in the video. Copy them into `dataset\` and jump to Step B (still read the curation rule there).*

1. Open ComfyUI → **Workflows menu → Comfy Workflow → LoRA → Chroma → "1 - Generate images"**.
2. Chroma wants **plain sentences**, not tag lists. Describe YOUR character in the Positive prompt
   and keep the identity block identical between runs; vary only outfit / place / light /
   expression / framing. No trigger word here — that comes later, in the captions. Queue a few
   runs of each prompt (batch 4) until you have ~60 raw images. Examples (swap the identity block
   for your character):

```
A photo of a young woman with a silver-grey undercut bob with a subtle teal front streak, light freckles and a small beauty mark under her left eye, wearing a rust-orange utility jacket, standing on a city street at golden hour, neutral expression. close-up portrait. photorealistic, detailed skin texture.
```
```
A photo of a young woman with a silver-grey undercut bob with a subtle teal front streak, light freckles and a small beauty mark under her left eye, wearing a denim jacket, laughing in a cozy cafe, warm light. three-quarter portrait. photorealistic, detailed skin texture.
```
```
A photo of a young woman with a silver-grey undercut bob with a subtle teal front streak, light freckles and a small beauty mark under her left eye, wearing a grey knit sweater, indoors by a bright window, gentle smile. upper body. photorealistic, detailed skin texture.
```
```
A photo of a young woman with a silver-grey undercut bob with a subtle teal front streak, light freckles and a small beauty mark under her left eye, in a plain white t-shirt and jeans, standing in a grey studio, smiling. full body. photorealistic, detailed skin texture.
```

3. Copy the images from ComfyUI's output into **`Downloads\train-lora-chroma\raw_gen\`**.

## Step B — Curate down to 15 → 📁 `Downloads\train-lora-chroma\dataset\`
👤 Look at the raw images at full size. Keep the **15 best** (same character; a mix of close-up /
upper body / full body; clean hands and eyes; no watermarks) and **copy them into
`Downloads\train-lora-chroma\dataset\`**.

🔑 **New rule that matters MORE on Chroma: keep your accent color CONSISTENT across the dataset.**
If your character has an accent feature (like our Juno's teal front streak), keep only images
where it looks the **same** — same place, same size, same tone. Drop the outliers where it
spreads, shifts or tints other parts of the hair. SDXL tolerated that inconsistency; Chroma
diffuses it all over the character. We had to drop 4 of our own 15 for exactly this.

## Step C — Caption (guided, ~5 minutes)
👤 The installer put **`nl_captions.bat`** inside `dataset\`. Double-click it. It asks your
**trigger word** and **subject** once, then for **each photo** one line: *what varies in this
photo?* (outfit, place, light, expression) plus the framing. Enter = repeat the previous answer.
It shows a review and writes one caption `.txt` per image, in the exact format we trained with:

```
A photo of cmfychar, a woman in a denim jacket, in a cozy cafe, warm light, laughing. three-quarter portrait. photorealistic.
```

🔑 **Golden rule: describe the PHOTO, not the character.** Whatever you do NOT describe, the
trigger word absorbs as "the character" — that's the point. Never describe your character's
canonical features (hair cut/color, freckles, eye color). DO describe anything that deviates in
that specific photo (e.g. "with teal-tinted hair ends" on a photo where the accent spread) — it
protects the canonical look.
- *Power-user note:* a local VLM captioner (e.g. JoyCaption) can write these for you, but it's a
  ~8 GB extra download for something 5 minutes of typing does better on 15 images. Unsupported here.

## Step D — Train (OneTrainer) → 📁 `Downloads\train-lora-chroma\output\`
1. Open `Downloads\OneTrainer\start-ui.bat`. In the blue dropdown (top-left), load
   **`train-lora-chroma`**.
2. The preset carries the whole recipe: Chroma1 · LoRA rank 16 · LR 2e-4 · AdamW · attn-mlp
   layers · bf16 · CPU-offloaded checkpointing (fits 12 GB) · 1024 px with aspect-ratio
   bucketing · **100 epochs** (that's where our character locked in — more tends to overcook).
3. **Your images are already linked.** The installer wrote a Chroma-specific concepts file, so
   the **concepts** tab shows your `dataset` folder — nothing to add by hand. Just glance that
   it's there and **enabled**. *(Safety net: if it's ever empty, concepts tab → Add Concept →
   Path = your `dataset` → Prompt Source = `From text file per sample` → enabled.)*
4. **Start Training.** First run downloads the base model (~18 GB) — the progress bar sits at 0
   during that download; that's normal. ✅ Sanity check: caching should report your image count
   (15/15, not 0), and the step bar should move for real. It saves checkpoints every 25 epochs
   and finishes with `juno_chroma.safetensors` in `output\`. (The red `fatal: not a git
   repository` text is normal — ignore it.)

## Step E — Use the LoRA (ComfyUI) → the result
1. Copy `Downloads\train-lora-chroma\output\juno_chroma.safetensors` into
   `C:\ComfyUI_windows_portable\ComfyUI\models\loras\`.
2. Open **Workflows menu → Comfy Workflow → LoRA → Chroma → "2 - Use LoRA"** → check the Load
   LoRA (Model Only) node = `juno_chroma.safetensors`, strength 1.0 (drop to 0.8 if it looks
   overcooked).
3. Prompt in plain sentences **with your trigger word**:

```
A photo of cmfychar, a woman in steel plate armor, on a snowy mountain, dramatic light. upper body. photorealistic.
```
```
A photo of cmfychar, a woman in a denim jacket, sitting in a cozy cafe, warm afternoon light, soft smile. close-up portrait. photorealistic.
```

Honest note from our own runs: Chroma locks the face and hair well, but **accent details are
finicky** — if your character's accent color wanders between generations, tighten the dataset
(Step B) before touching any training number.

Licenses are clean (commercial use OK): OneTrainer (MIT), Chroma1-HD (Apache 2.0),
FLAN-T5 encoder (Apache 2.0).
