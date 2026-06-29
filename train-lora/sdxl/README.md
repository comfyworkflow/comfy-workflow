# CW20 — Train a Character LoRA (SDXL, OneTrainer) — step by step

> You already ran **INSTALL_CW20.bat**, so the model, the work folders, OneTrainer and the training preset
> are ready. This guide covers the part you do by hand — it's the actual tutorial.
>
> Legend: 🤖 = automatic · 👤 = you do it · 🟦 = adapt to your own character.

---

## 🟦 WHAT TO ADAPT TO YOUR OWN CHARACTER (only this)
1. **Your character.** This demo uses **"Juno"** (silver undercut bob + teal streak + freckles), trigger **`cmfychar`**. For yours: swap the description for your character and pick a **unique trigger** (a made-up word). In the prompts, only the **identity block** changes.
2. **ComfyUI not in the default place?** Everything assumes `C:\ComfyUI_windows_portable`. If yours is elsewhere, the "Base Model" in OneTrainer will show red — just re-point it with the `[...]` button.

Your work folder (created by the installer): **`Downloads\CW20_SDXL_LoRA\`** (`raw_gen` / `dataset` / `workspace` / `output`).

---

## Step A — Generate the images → 📁 `Downloads\CW20_SDXL_LoRA\raw_gen\`
1. Open ComfyUI → **Workflows menu → Comfy Workflow → LoRA → SDXL → "1 - Generate images"**.
2. Paste each of the **16 prompts** below into the **Positive** box and click **Queue** (4 per queue, seed on randomize) = **~64 images**.
   - 🟦 *For your character:* change only the **identity block** (`silver-grey undercut bob ... light freckles`).
3. Copy the ~64 images from ComfyUI's output into **`Downloads\CW20_SDXL_LoRA\raw_gen\`**.

**CLOSE-UPS (1-6):**
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, neutral expression, close-up portrait, front view, wearing a rust-orange utility jacket, golden hour light, city street, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, soft smile, close-up portrait, three-quarter view, wearing a white tee, soft overcast light, cafe interior, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, serious, close-up portrait, profile view, wearing a black hoodie, warm window light, bedroom, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, laughing, close-up portrait, looking up, wearing a cable-knit sweater, studio softbox light, plain grey studio, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, surprised, close-up portrait, three-quarter view, wearing a denim jacket, neon city night, rooftop, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, serious, close-up portrait, front view, wearing a formal blazer, studio softbox light, forest path, photorealistic, detailed skin texture, sharp focus, 85mm
```

**BUST / WAIST (7-12):**
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, soft smile, three-quarter bust, front view, wearing a rust-orange utility jacket, golden hour light, rooftop, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, neutral expression, waist-up mid-shot, three-quarter view, wearing a white tee, soft overcast light, city street, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, serious, three-quarter bust, over-the-shoulder, wearing a black hoodie, warm window light, cafe interior, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, soft smile, waist-up mid-shot, sitting, wearing a cable-knit sweater, warm window light, bedroom, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, laughing, three-quarter bust, front view, wearing a denim jacket, studio softbox light, plain grey studio, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, neutral expression, waist-up mid-shot, profile view, wearing a formal blazer, golden hour light, forest path, photorealistic, detailed skin texture, sharp focus, 85mm
```

**FULL-BODY (13-16):**
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, neutral expression, full body shot, full figure head to toe, feet visible, full length portrait, wide shot, standing, wearing a rust-orange utility jacket and denim, golden hour light, city street, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, serious, full body shot, full figure head to toe, feet visible, full length portrait, wide shot, standing with one hand in pocket, wearing a formal blazer, soft overcast light, park path, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, soft smile, full body shot, full figure head to toe, feet visible, full length portrait, wide shot, sitting, wearing a black hoodie and denim, warm window light, cafe interior, photorealistic, detailed skin texture, sharp focus, 85mm
```
```
cmfychar, silver-grey undercut bob with a subtle teal front streak, a small beauty mark under the left eye, light freckles, neutral expression, full body shot, full figure head to toe, feet visible, full length portrait, wide shot, standing, wearing a cable-knit sweater, studio softbox light, plain grey studio, photorealistic, detailed skin texture, sharp focus, 85mm
```

## Step B — Curate down to 15 → 📁 `Downloads\CW20_SDXL_LoRA\dataset\`
👤 Look at the ~64 in `raw_gen\` at full size. Keep the **15 best** (same character; a mix of close-up / bust / full-body; clean hands and eyes; no watermarks) and **copy them into `Downloads\CW20_SDXL_LoRA\dataset\`** (the .png files only).

## Step C — Caption (🤖 1 click)
👤 The installer already put **`simple_captions.bat`** inside `dataset\`. 🤖 Double-click it → it asks for your **trigger word** and your **subject class**, shows you the caption and asks you to **confirm**, then writes a matching `.txt` for every image. (If you've already hand-edited captions, it asks before overwriting them.)
- 🟦 *For your character:* type your own trigger (a made-up word, e.g. `myhero`) and class (`1boy`, `robot`, `a sword`...) when it asks. Press Enter on each to use the demo defaults (`cmfychar` / `1girl, solo`).
- This is the **simple** method — the same caption on every image, which is the beginner default and what this tutorial uses. For more flexibility (changing outfits and scenes freely), caption what *varies* per image by hand, or use OneTrainer's built-in WD14 tagger.

## Step D — Train (OneTrainer) → 📁 `Downloads\CW20_SDXL_LoRA\output\`
1. Open `Downloads\OneTrainer\start-ui.bat`. In the blue dropdown (top-left), load **`CW20_SDXL_config`**.
2. The preset fills the settings: on the **model** tab, **Workspace / Output / Base Model** are already set — just confirm none is red. 🟦 If Base Model is red, your ComfyUI isn't in the default place → re-point it with `[...]`.
3. **Your images are already linked.** The installer pointed the training at your `dataset` folder for you, so the **concepts** tab already shows your character — nothing to add by hand. Just glance that it's there and **enabled**. *(Safety net: if it's ever empty, add it — concepts tab → Add Concept → Path = your `dataset` → Prompt Source = `From text file per sample` → enabled.)*
4. **Start Training**. ✅ Sanity check: the **step** bar should move for real (~15 steps per epoch) and the run takes **minutes**. If the 150 epochs fly by in seconds showing `0it`, OneTrainer found no images → your concept Path is wrong/empty or the captions are missing (redo step 3 / Step C). It saves `cw20_juno.safetensors` in `output\`. (The red `fatal: not a git repository` text is normal — ignore it.)

## Step E — Use the LoRA (ComfyUI) → the result
1. Copy `Downloads\CW20_SDXL_LoRA\output\cw20_juno.safetensors` into `C:\ComfyUI_windows_portable\ComfyUI\models\loras\`.
2. Open **Workflows menu → Comfy Workflow → LoRA → SDXL → "2 - Use LoRA"** → check the Load LoRA node = `cw20_juno.safetensors`, strength 1.0.
3. Test (if it looks overcooked, drop strength to 0.8):
```
cmfychar, 1girl, solo, grey hair, silver bob, steel plate armor, snowy mountain, upper body, bokeh
```
```
cmfychar, 1girl, solo, grey hair, silver bob, winter coat, snowy pine forest, upper body, bokeh
```
```
cmfychar, 1girl, solo, soft smile, city street, blurred background, bokeh, upper body, 85mm
```
```
cmfychar, 1girl, solo, soft smile, cafe interior, blurred background, bokeh, upper body, 85mm
```
> 🔑 A **focused prompt + upper-body + a reminder of the identity** (`grey hair, silver bob`) keeps the character locked. A busy, distant full-body prompt can drift — if it does, shorten the prompt, restate the hair, or nudge strength to 1.1.

---
Licenses are clean (commercial use OK): OneTrainer (MIT), Juggernaut XL v9 (OpenRAIL-M), WD14 (Apache).
