# CUDA Ray Tracer with Multi-View Rendering

## Overview

This project is a Python-based ray tracer that supports both CPU rendering and full GPU rendering with CUDA through Numba. The renderer generates a reflective and refractive sphere scene over a checkerboard plane and can export the scene from multiple camera viewpoints.

The project was built to demonstrate the core ideas of ray tracing, recursive light transport, reflections, refractions, Fresnel effects, plane and sphere intersections, and multi-view scene rendering.

The repository is designed so that you can:

- render a quick preview
- render with a CPU fallback path
- render with a full CUDA GPU path
- generate multiple viewpoints automatically
- optionally render a 4K version on the GPU

---

## Features

### Rendering Features
- Sphere intersection
- Plane intersection
- Recursive ray tracing
- Reflection
- Refraction
- Fresnel blending using Schlick approximation
- Checkerboard plane shading
- Multi-light illumination
- Multiple predefined viewpoints
- CPU fallback rendering
- Full GPU CUDA rendering
- Optional 4K GPU render mode

### Outputs
The renderer currently generates the following camera views:

#### View 1 - Front
![View 1 Front](output/view_1_front.png)

#### View 2 - Right
![View 2 Right](output/view_2_right.png)

#### View 3 - Left
![View 3 Left](output/view_3_left.png)

#### View 4 - Wide
![View 4 Wide](output/view_4_wide.png)

#### View 5 - Closeup
![View 5 Closeup](output/view_5_closeup.png)

### Render Modes
The program supports four runtime presets:

1. **Fast low-res preview**
2. **CPU multiprocessing / CPU fallback**
3. **Full GPU CUDA render**
4. **4K GPU CUDA render**

---

## Repository Structure

```text
.
├── README.md
├── raytracer.py
├── requirements.txt
├── environment_notes.md
├── outputs/
│   ├── view_1_front.png
│   ├── view_2_right.png
│   ├── view_3_left.png
│   ├── view_4_wide.png
│   └── view_5_closeup.png
└── screenshots/
    ├── terminal_progress.png
    └── sample_render.png
