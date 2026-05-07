# Comfy Workflow Benchmarks

Benchmark suite for ComfyUI workflows across consumer GPUs.

> **Status:** Alpha — under active development. The base setup script is functional; benchmarking tooling is in development.

## Overview

Public benchmark tooling for ComfyUI workflows running on consumer GPUs. The project provides:

- A reproducible installer that verifies your ComfyUI Portable setup
- Benchmark profiles (YAML) defining which models and parameters to test
- ComfyUI workflows (JSON) used in benchmarks

## Prerequisites

This repository assumes you have followed the base setup tutorial, which installs:

- Python 3.13
- Git, FFmpeg, 7-Zip
- ComfyUI Portable in `C:\ComfyUI_windows_portable\`
- Three essential custom nodes: ComfyUI-Manager, rgthree-comfy, ComfyUI-Crystools

The base setup script is at [`setup-windows/01-install-base.bat`](setup-windows/01-install-base.bat).

## Installation

`installer.py` is coming soon. Once published:

```cmd
git clone https://github.com/comfyworkflow/comfy-workflow.git
cd comfy-workflow
python -m pip install -e .
python installer.py --check
```

## License

MIT — see [LICENSE](LICENSE).
