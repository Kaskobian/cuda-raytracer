# Environment Notes

## Overview

This project supports both CPU rendering and full GPU rendering.

- **Option 1**: fast low-resolution preview
- **Option 2**: CPU rendering
- **Option 3**: full GPU CUDA rendering
- **Option 4**: 4K GPU CUDA rendering

The renderer depends on:

- `numpy`
- `pillow`
- `numba`
- `numba-cuda`

The Python script uses CUDA through Numba for the GPU path, so GPU rendering requires a working NVIDIA driver and a CUDA-capable Python environment. :contentReference[oaicite:1]{index=1}

---

## Recommended Setup

The most reliable setup for GPU rendering is a dedicated Conda environment.

### Create the environment

```bash
conda create -n raygpu python=3.12 -y
