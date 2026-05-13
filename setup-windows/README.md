# Comfy Workflow — Windows Setup

Manual setup script for the **Comfy Workflow** YouTube channel.

This installs (or updates) a clean ComfyUI Portable environment on
Windows 10 / Windows 11, ready to run channel workflows.

> **TODO (project-internal note for Rafael — remove before publishing):**
> The base install uses ComfyUI Portable with **Python 3.13 + CUDA
> 13.0** (the default Comfy-Org build as of May 2026, replacing the
> older Python 3.12 + CUDA 12.8 build). Update `audit.py` and
> `04-schema-audit-v3.md` accordingly: `embedded_python_version`
> should now expect `3.13.x`, not `3.12.x`.

## Requirements

- **Windows 10** (version 1809 or later) **or Windows 11**
- **NVIDIA GPU** (RTX 20-series, 30-series, 40-series, or 50-series)
  with recent drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- **~5 GB free disk space** for the install (more for models later)
- **Internet connection** (the ComfyUI Portable archive is ~3 GB)

If you have an older NVIDIA GPU (10-series or earlier), this script
will not work — you would need the Python 3.12 + CUDA 12.6 build of
ComfyUI Portable instead.

## Files

There are three flavors of script in this directory:

1. **`01-install-base.bat`** — the base installer (ComfyUI Portable +
   3 essential custom nodes).
2. **`install-<model>.bat`** — per-Pillar / per-model installers that
   download a specific model family + ship the matching workflow.
3. **`setup-pillar1.bat`** — a master orchestrator that runs the base
   install + a Pillar #1 model installer in one shot.

### `01-install-base.bat`

The base script works in two modes depending on what's already on your
machine.

| If your machine has... | The script will... |
|---|---|
| **No previous ComfyUI install** | Install everything from scratch |
| **An existing ComfyUI install** | Ask whether you want to **Update** (recommended — keeps your models and workflows) or do a **Fresh** install (backs up the old one and reinstalls) |

Most users want **Update**. **Fresh** is for people whose existing
install is broken and they want to start clean — your old folder is
renamed to `C:\ComfyUI_windows_portable.OLD-<timestamp>\` so nothing
is lost.

### Per-model installers

Each `install-<model>.bat` assumes the base install is already present
at `C:\ComfyUI_windows_portable\`. It downloads the model files with
byte-exact size verification, clones any required custom nodes, and
ships the corresponding workflow JSON + sidecar README into
`ComfyUI\user\default\workflows\`. All file operations are
**idempotent**: re-running a script skips files already present with
matching size.

| Script | Pillar(s) | Display name | Hardware mínimo | Custom nodes |
|---|---|---|---|---|
| `install-sdxl.bat` | #1 | SDXL Base 1.0 | 16 GB RAM · 8 GB VRAM | — |
| `install-flux1.bat` | #4, #1 | FLUX.1 dev (fp8 + fp16) | 64 GB RAM · 24 GB VRAM | — |
| `install-flux2.bat` | #2, #4 | FLUX.2 dev GGUF (Q4_K_M) | 48 GB RAM · 12 GB VRAM | ComfyUI-GGUF |
| `install-qwen-image.bat` | #1 | Qwen-Image fp8 | 64 GB RAM · 12 GB VRAM | — |
| `install-qwen-2512.bat` | #2 | Qwen-Image 2512 fp8 | 64 GB RAM · 12 GB VRAM | — |
| `install-hunyuan-21.bat` | #2, #3 | Hunyuan-Image 2.1 bf16 | 96 GB RAM · 16 GB VRAM | — |
| `install-wan22.bat` | #5 | WAN 2.2 i2v fp8 dual-expert | 96 GB RAM · 16 GB VRAM | — |

These scripts are **generated** from
`installer/benchmark/models_manifest.yaml` via
`installer/benchmark/generate_install_scripts.py` — do not edit them
by hand. To refresh after a manifest change, run:

```cmd
python -m installer.benchmark.generate_install_scripts
```

### `setup-pillar1.bat`

Master orchestrator for the Pillar #1 video (SDXL). Runs
`01-install-base.bat` then `install-sdxl.bat` in sequence. Pass
`--unattended` (or `-u`, or set the env var `COMFY_NONINTERACTIVE=1`)
to bypass the U/F/C prompt and the FIRST LAUNCH manual step in the
base installer — useful for SSH / remote dispatch where there is no
interactive console.

## Quick start

1. Download `01-install-base.bat`.
2. **Double-click to run it.** Do NOT use "Run as administrator"
   by default — winget installs per-user and works without
   elevation in the vast majority of cases.
3. If any `winget install` step fails asking for elevation, close
   the window and re-run via right-click → **Run as administrator**.
4. Follow the on-screen instructions. If you already have ComfyUI,
   the script will ask whether to Update or do a Fresh install.
5. Total time: 5–10 minutes (Update) or 15–30 minutes (Fresh,
   depending on internet speed).

> **Remote desktop / Tailscale RDP users:** always try non-admin
> first. "Run as administrator" triggers UAC prompts on the Windows
> secure desktop, which sometimes do not render correctly through
> remote tunnels (you see a frozen screen instead of the prompt).
> Non-admin avoids that entire class of issue.

## What gets installed

### Via winget (system-wide tools)

- **7-Zip** (`7zip.7zip`)
- **Git** (`Git.Git`)
- **FFmpeg** (`Gyan.FFmpeg`)
- **Python 3.13** (`Python.Python.3.13`)

These tools may already be useful for other projects on your machine.
The script does **not** remove them.

### Via direct download (Fresh mode only)

- **ComfyUI Portable** (Python 3.13 + CUDA 13.0) at
  `C:\ComfyUI_windows_portable\`

### Via git clone or git pull (always)

- **ComfyUI-Manager** — install/manage custom nodes from inside
  ComfyUI
- **rgthree-comfy** — workflow QoL nodes (pipes, switches, fast
  groups, etc.)
- **ComfyUI-Crystools** — system monitor (CPU/GPU/VRAM) and image
  metadata tools

In Update mode these are refreshed via `git pull --ff-only`. If you
have local modifications in any of these folders, the pull is
skipped with a warning and your local copy is preserved.

### Via pip (always)

- `requests` and `pillow` into global Python 3.13
- `requirements.txt` of each of the 3 custom nodes into the embedded
  ComfyUI Python

## Manual step: Enable Node ID Badge

After install, run `C:\ComfyUI_windows_portable\run_nvidia_gpu.bat`,
open Settings (gear icon, top-right of the ComfyUI browser tab),
search for **"Node ID Badge Mode"**, and set it to **"Show All"**.

This is required for the workflows shared on the Comfy Workflow
channel (workflow nodes are referenced by ID in tutorials).

## Troubleshooting

### "NVIDIA GPU not detected"

You need an NVIDIA GPU with recent drivers. Get them from
[nvidia.com](https://www.nvidia.com/Download/index.aspx).

### "winget not found"

Install **App Installer** from the Microsoft Store. winget is built
into Windows 10 (1809+) and Windows 11 by default — if it's missing,
the App Installer package will restore it.

### "'python' resolves to a ComfyUI embedded Python"

You have a previous ComfyUI install whose `python_embeded\` folder
got added to your PATH (some old tutorials ask for this — don't do
it). Open **System Properties → Environment Variables**, find any
PATH entry containing `python_embeded`, remove it, then re-run the
script.

### Download fails

The ComfyUI Portable archive is ~3 GB. The script uses `curl` with
automatic retry (3 attempts, 5s delay) and validates that the
downloaded file is at least 1 GB before proceeding — if it's smaller,
the script aborts and deletes the partial file so you can retry.

### Could not rename folder (Fresh mode)

Some files inside `C:\ComfyUI_windows_portable\` are still locked by
a running program. Close ComfyUI, close any cmd window inside that
folder, close any text editor with files open from it, then retry.

### `git pull --ff-only` warning during Update

You have local edits in one of the 3 essential custom node folders
(`ComfyUI-Manager`, `rgthree-comfy`, or `ComfyUI-Crystools`). The
script preserves your local copy and continues. If you want the
update, commit/stash your edits first, or just delete the folder
and re-run — the script will then clone fresh.

## Note on backups

The Fresh install mode of `01-install-base.bat` creates timestamped
backup folders like
`C:\ComfyUI_windows_portable.OLD-2026-05-03-143022\`.

These are **never deleted automatically**. Each one can be 5–50 GB
(or more if you had models). When you're sure you don't need an old
install anymore:

```cmd
rmdir /S /Q "C:\ComfyUI_windows_portable.OLD-2026-05-03-143022"
```

## License

MIT. See top-level `LICENSE` of this repository.