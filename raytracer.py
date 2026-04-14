
import math
import os
import time
import argparse
import numpy as np

try:
    from numba import cuda
    CUDA_AVAILABLE = True
except Exception:
    cuda = None
    CUDA_AVAILABLE = False

EPSILON = 1e-4
MAX_SPHERES = 8


def normalize_np(v):
    n = np.linalg.norm(v)
    if n < EPSILON:
        return v
    return v / n

def build_scene_arrays():
    # spheres: center xyz, radius, color rgb, metallic, roughness, specular, transmission, ior
    spheres = np.array([
        [-1.8, -0.15, -6.8, 1.05, 0.85, 0.14, 0.14, 0.0, 0.18, 0.55, 0.0, 1.50],
        [ 0.35,  0.00, -5.35, 1.05, 0.92, 0.92, 0.94, 1.0, 0.06, 1.00, 0.0, 1.50],
        [ 2.05,  0.10, -7.20, 1.15, 0.70, 0.98, 0.78, 0.0, 0.03, 0.85, 0.98, 1.52],
    ], dtype=np.float32)

    # plane: point xyz, normal xyz, light checker rgb, dark checker rgb, scale, metallic, roughness, specular
    plane = np.array([
        0.0, -1.15, 0.0,
        0.0, 1.0, 0.0,
        0.88, 0.89, 0.93,
        0.16, 0.17, 0.20,
        1.25,
        0.0, 0.30, 0.18
    ], dtype=np.float32)

    # lights: position xyz, color rgb, intensity, radius
    lights = np.array([
        [ 5.5, 7.0,  1.0, 1.00, 0.98, 0.95, 22.0, 1.8],
        [-7.5, 5.5, -4.0, 0.55, 0.65, 1.00,  1.5, 0.8],
    ], dtype=np.float32)

    # Old lights: previous version of the light ray direction
#    lights = np.array([
#        [ 5.5, 7.0,  1.0, 1.00, 0.98, 0.95, 22.0, 1.8],
#        [-6.0, 4.0, -2.5, 0.55, 0.65, 1.00,  2.5, 0.8],
#    ], dtype=np.float32)

    return spheres, plane, lights

def get_preset(option):
    presets = {
        "1": {"label": "Fast low-res preview", "width": 640,  "height": 420,  "max_depth": 3, "spp": 1},
        "2": {"label": "CPU multiprocessing",  "width": 1200, "height": 800,  "max_depth": 6, "spp": 4},
        "3": {"label": "Full GPU CUDA render", "width": 1400, "height": 900,  "max_depth": 6, "spp": 1},
        "4": {"label": "4K GPU CUDA render",   "width": 3840, "height": 2160, "max_depth": 6, "spp": 1},
    }
    return presets[option]

def choose_option_interactively():
    print("\nSelect a render mode:")
    print("  1. Fast low-res preview")
    print("  2. CPU multiprocessing")
    print("  3. Full GPU CUDA render")
    print("  4. 4K GPU CUDA render")
    while True:
        choice = input("Enter 1, 2, 3, or 4: ").strip()
        if choice in ("1", "2", "3", "4"):
            return choice
        print("Invalid choice. Please enter 1, 2, 3, or 4.")

# ---------------- CPU fallback path ----------------

def reflect_np(I, N):
    return I - 2.0 * np.dot(I, N) * N


def refract_np(I, N, eta_t, eta_i=1.0):
    cosi = np.clip(np.dot(I, N), -1.0, 1.0)
    n = N.copy()
    etai = eta_i
    etat = eta_t
    if cosi < 0:
        cosi = -cosi
    else:
        n = -N
        etai, etat = etat, etai
    eta = etai / etat
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    if k < 0:
        return None
    return normalize_np(eta * I + (eta * cosi - math.sqrt(k)) * n)


def schlick_np(cos_theta, f0):
    return f0 + (1.0 - f0) * ((1.0 - cos_theta) ** 5)


def aces_np(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def sky_np(direction):
    t = 0.5 * (direction[1] + 1.0)
    sky_top = np.array([0.25, 0.36, 0.52], dtype=np.float32)
    sky_bottom = np.array([0.72, 0.76, 0.82], dtype=np.float32)
    sky = (1.0 - t) * sky_bottom + t * sky_top
    sun_dir = normalize_np(np.array([0.7, 0.55, -0.35], dtype=np.float32))
    sun = max(np.dot(direction, sun_dir), 0.0) ** 256
    return sky + np.array([1.0, 0.85, 0.6], dtype=np.float32) * (2.0 * sun)


def plane_color_np(hit_point, plane):
    scale = plane[12]
    x = math.floor(hit_point[0] * scale)
    z = math.floor(hit_point[2] * scale)
    if (x + z) % 2 == 0:
        return plane[6:9].copy()
    return plane[9:12].copy()


def intersect_scene_np(ro, rd, spheres, plane):
    hit_t = 1e20
    hit_type = -1
    hit_idx = -1
    hit_n = None
    hit_p = None

    for i in range(spheres.shape[0]):
        cx, cy, cz, r = spheres[i, 0], spheres[i, 1], spheres[i, 2], spheres[i, 3]
        oc = ro - np.array([cx, cy, cz], dtype=np.float32)
        a = np.dot(rd, rd)
        b = 2.0 * np.dot(oc, rd)
        c = np.dot(oc, oc) - r * r
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            s = math.sqrt(disc)
            t1 = (-b - s) / (2.0 * a)
            t2 = (-b + s) / (2.0 * a)
            t = None
            if t1 > EPSILON:
                t = t1
            elif t2 > EPSILON:
                t = t2
            if t is not None and t < hit_t:
                hit_t = t
                hit_type = 1
                hit_idx = i
                hit_p = ro + rd * t
                hit_n = normalize_np(hit_p - np.array([cx, cy, cz], dtype=np.float32))

    plane_p = plane[0:3]
    plane_n = plane[3:6]
    denom = np.dot(plane_n, rd)
    if abs(denom) > EPSILON:
        t = np.dot(plane_p - ro, plane_n) / denom
        if t > EPSILON and t < hit_t:
            hit_t = t
            hit_type = 2
            hit_idx = 0
            hit_p = ro + rd * t
            hit_n = plane_n if denom < 0.0 else -plane_n

    return hit_type, hit_idx, hit_t, hit_p, hit_n


def shade_cpu_pixel(ro, rd, spheres, plane, lights, max_depth):
    color = np.zeros(3, dtype=np.float32)
    throughput = np.ones(3, dtype=np.float32)

    for _ in range(max_depth):
        hit_type, hit_idx, hit_t, hit_p, hit_n = intersect_scene_np(ro, rd, spheres, plane)
        if hit_type < 0:
            color += throughput * sky_np(rd)
            break

        if hit_type == 1:
            s = spheres[hit_idx]
            base = s[4:7].copy()
            metallic = s[7]
            roughness = s[8]
            specular = s[9]
            transmission = s[10]
            ior = s[11]
        else:
            base = plane_color_np(hit_p, plane)
            metallic = plane[13]
            roughness = plane[14]
            specular = plane[15]
            transmission = 0.0
            ior = 1.5

        view_dir = normalize_np(-rd)
        local = 0.02 * base

        for li in range(lights.shape[0]):
            lp = lights[li, 0:3]
            lc = lights[li, 3:6]
            intensity = lights[li, 6]

            to_light = lp - hit_p
            dist = np.linalg.norm(to_light)
            ldir = to_light / max(dist, EPSILON)

            # hard shadow
            s_type, _, s_t, _, _ = intersect_scene_np(hit_p + hit_n * EPSILON, ldir, spheres, plane)
            if s_type >= 0 and s_t < dist:
                continue

            ndotl = max(np.dot(hit_n, ldir), 0.0)
            if ndotl <= 0.0:
                continue

            h = normalize_np(ldir + view_dir)
            ndoth = max(np.dot(hit_n, h), 0.0)
            vdoth = max(np.dot(view_dir, h), 0.0)
            attenuation = intensity / (1.0 + 0.09 * dist + 0.032 * dist * dist)

            f0_scalar = 0.04 * (1.0 - metallic) + specular * metallic
            f0 = np.array([f0_scalar, f0_scalar, f0_scalar], dtype=np.float32)
            f0 = f0 * (1.0 - metallic) + base * metallic
            F = schlick_np(vdoth, f0)

            diffuse = (1.0 - F) * (1.0 - metallic) * base * ndotl
            shininess = max(2.0, 2.0 / max(roughness * roughness, 1e-4))
            spec = F * (ndoth ** shininess)

            local += lc * attenuation * (diffuse + spec)

        color += throughput * local

        cos_theta = max(np.dot(hit_n, view_dir), 0.0)
        dielectric_f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
        f0 = np.array([dielectric_f0, dielectric_f0, dielectric_f0], dtype=np.float32)
        f0 = f0 * (1.0 - metallic) + base * metallic
        F = schlick_np(cos_theta, f0)

        if transmission > 0.0:
            refr = refract_np(rd, hit_n, ior)
            if refr is not None:
                ro = hit_p - hit_n * EPSILON
                rd = refr
                throughput *= (1.0 - F) * transmission * base
                continue

        rd = normalize_np(reflect_np(rd, hit_n))
        ro = hit_p + hit_n * EPSILON
        throughput *= (F * (0.25 + 0.75 * max(specular, metallic))).astype(np.float32)

        if np.max(throughput) < 0.01:
            break

    color = aces_np(color * 1.15)
    color = np.power(np.clip(color, 0.0, 1.0), 1.0 / 2.2)
    return np.clip(color * 255.0, 0.0, 255.0).astype(np.uint8)


def render_cpu_single(scene_spheres, plane, lights, width, height, eye, look_at, fov, max_depth):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye = np.array(eye, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)

    w = normalize_np(eye - look_at)
    u = normalize_np(np.cross(up, w))
    v = normalize_np(np.cross(w, u))

    aspect = width / height
    scale = math.tan(math.radians(fov) / 2.0)

    start = time.time()
    for j in range(height):
        for i in range(width):
            px = (2.0 * ((i + 0.5) / width) - 1.0) * aspect * scale
            py = (1.0 - 2.0 * ((j + 0.5) / height)) * scale
            rd = normalize_np(px * u + py * v - w)
            img[j, i] = shade_cpu_pixel(eye.copy(), rd, scene_spheres, plane, lights, max_depth)

        progress = (j + 1) / height
        elapsed = time.time() - start
        eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
        print(f"[CPU fallback] Row {j+1}/{height} | {progress*100:6.2f}% | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", flush=True)

    return img


# ---------------- Full GPU path ----------------

if CUDA_AVAILABLE:
    @cuda.jit(device=True)
    def dot3(ax, ay, az, bx, by, bz):
        return ax * bx + ay * by + az * bz

    @cuda.jit(device=True)
    def norm3(x, y, z):
        return math.sqrt(x * x + y * y + z * z)

    @cuda.jit(device=True)
    def normalize3(x, y, z):
        n = math.sqrt(x * x + y * y + z * z)
        if n < EPSILON:
            return x, y, z
        return x / n, y / n, z / n

    @cuda.jit(device=True)
    def reflect3(ix, iy, iz, nx, ny, nz):
        d = dot3(ix, iy, iz, nx, ny, nz)
        return ix - 2.0 * d * nx, iy - 2.0 * d * ny, iz - 2.0 * d * nz

    @cuda.jit(device=True)
    def refract3(ix, iy, iz, nx, ny, nz, eta_t, out_arr):
        cosi = dot3(ix, iy, iz, nx, ny, nz)
        if cosi < -1.0:
            cosi = -1.0
        if cosi > 1.0:
            cosi = 1.0

        etai = 1.0
        etat = eta_t
        nnx, nny, nnz = nx, ny, nz

        if cosi < 0.0:
            cosi = -cosi
        else:
            nnx, nny, nnz = -nx, -ny, -nz
            etai, etat = etat, etai

        eta = etai / etat
        k = 1.0 - eta * eta * (1.0 - cosi * cosi)
        if k < 0.0:
            return 0

        rx = eta * ix + (eta * cosi - math.sqrt(k)) * nnx
        ry = eta * iy + (eta * cosi - math.sqrt(k)) * nny
        rz = eta * iz + (eta * cosi - math.sqrt(k)) * nnz
        rx, ry, rz = normalize3(rx, ry, rz)
        out_arr[0] = rx
        out_arr[1] = ry
        out_arr[2] = rz
        return 1

    @cuda.jit(device=True)
    def schlick_scalar(cos_theta, f0):
        return f0 + (1.0 - f0) * ((1.0 - cos_theta) ** 5)

    @cuda.jit(device=True)
    def sky_color(dx, dy, dz, out):
        t = 0.5 * (dy + 1.0)
        sky_top_x, sky_top_y, sky_top_z = 0.25, 0.36, 0.52
        sky_bot_x, sky_bot_y, sky_bot_z = 0.72, 0.76, 0.82
        sx = (1.0 - t) * sky_bot_x + t * sky_top_x
        sy = (1.0 - t) * sky_bot_y + t * sky_top_y
        sz = (1.0 - t) * sky_bot_z + t * sky_top_z

        sunx, suny, sunz = normalize3(0.7, 0.55, -0.35)
        sun = dot3(dx, dy, dz, sunx, suny, sunz)
        if sun < 0.0:
            sun = 0.0
        sun = sun ** 256
        out[0] = sx + 1.2 * sun * 1.0
        out[1] = sy + 1.2 * sun * 0.85
        out[2] = sz + 1.2 * sun * 0.6

    @cuda.jit(device=True)
    def plane_checker(px, pz, plane, out):
        scale = plane[12]
        x = math.floor(px * scale)
        z = math.floor(pz * scale)
        if (x + z) % 2 == 0:
            out[0] = plane[6]
            out[1] = plane[7]
            out[2] = plane[8]
        else:
            out[0] = plane[9]
            out[1] = plane[10]
            out[2] = plane[11]

    @cuda.jit(device=True)
    def intersect_scene(ro_x, ro_y, ro_z, rd_x, rd_y, rd_z, spheres, plane, hit_info, hit_n, hit_base):
        hit_t = 1e20
        hit_type = -1
        hit_idx = -1
        hp_x = 0.0
        hp_y = 0.0
        hp_z = 0.0
        hn_x = 0.0
        hn_y = 0.0
        hn_z = 0.0

        for i in range(spheres.shape[0]):
            cx = spheres[i, 0]
            cy = spheres[i, 1]
            cz = spheres[i, 2]
            r = spheres[i, 3]

            oc_x = ro_x - cx
            oc_y = ro_y - cy
            oc_z = ro_z - cz
            a = dot3(rd_x, rd_y, rd_z, rd_x, rd_y, rd_z)
            b = 2.0 * dot3(oc_x, oc_y, oc_z, rd_x, rd_y, rd_z)
            c = dot3(oc_x, oc_y, oc_z, oc_x, oc_y, oc_z) - r * r
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                s = math.sqrt(disc)
                t1 = (-b - s) / (2.0 * a)
                t2 = (-b + s) / (2.0 * a)
                t = -1.0
                if t1 > EPSILON:
                    t = t1
                elif t2 > EPSILON:
                    t = t2
                if t > EPSILON and t < hit_t:
                    hit_t = t
                    hit_type = 1
                    hit_idx = i
                    hp_x = ro_x + rd_x * t
                    hp_y = ro_y + rd_y * t
                    hp_z = ro_z + rd_z * t
                    hn_x, hn_y, hn_z = normalize3(hp_x - cx, hp_y - cy, hp_z - cz)

        pp_x, pp_y, pp_z = plane[0], plane[1], plane[2]
        pn_x, pn_y, pn_z = plane[3], plane[4], plane[5]
        denom = dot3(pn_x, pn_y, pn_z, rd_x, rd_y, rd_z)
        if abs(denom) > EPSILON:
            t = dot3(pp_x - ro_x, pp_y - ro_y, pp_z - ro_z, pn_x, pn_y, pn_z) / denom
            if t > EPSILON and t < hit_t:
                hit_t = t
                hit_type = 2
                hit_idx = 0
                hp_x = ro_x + rd_x * t
                hp_y = ro_y + rd_y * t
                hp_z = ro_z + rd_z * t
                if denom < 0.0:
                    hn_x, hn_y, hn_z = pn_x, pn_y, pn_z
                else:
                    hn_x, hn_y, hn_z = -pn_x, -pn_y, -pn_z

        hit_info[0] = hit_type
        hit_info[1] = hit_idx
        hit_info[2] = hit_t
        hit_info[3] = hp_x
        hit_info[4] = hp_y
        hit_info[5] = hp_z
        hit_n[0] = hn_x
        hit_n[1] = hn_y
        hit_n[2] = hn_z

        if hit_type == 1:
            hit_base[0] = spheres[hit_idx, 4]
            hit_base[1] = spheres[hit_idx, 5]
            hit_base[2] = spheres[hit_idx, 6]
        elif hit_type == 2:
            plane_checker(hp_x, hp_z, plane, hit_base)

    @cuda.jit
    def render_kernel(out_img, spheres, plane, lights, eye, basis, width, height, fov, max_depth, progress):
        x, y = cuda.grid(2)
        if x >= width or y >= height:
            return

        aspect = width / height
        scale = math.tan(fov * 0.5 * math.pi / 180.0)

        px = (2.0 * ((x + 0.5) / width) - 1.0) * aspect * scale
        py = (1.0 - 2.0 * ((y + 0.5) / height)) * scale

        ux, uy, uz = basis[0], basis[1], basis[2]
        vx, vy, vz = basis[3], basis[4], basis[5]
        wx, wy, wz = basis[6], basis[7], basis[8]

        rd_x, rd_y, rd_z = normalize3(px * ux + py * vx - wx, px * uy + py * vy - wy, px * uz + py * vz - wz)
        ro_x, ro_y, ro_z = eye[0], eye[1], eye[2]

        col_x, col_y, col_z = 0.0, 0.0, 0.0
        thr_x, thr_y, thr_z = 1.0, 1.0, 1.0

        hit_info = cuda.local.array(6, dtype=np.float32)
        hit_n = cuda.local.array(3, dtype=np.float32)
        hit_base = cuda.local.array(3, dtype=np.float32)
        sky = cuda.local.array(3, dtype=np.float32)
        refr = cuda.local.array(3, dtype=np.float32)

        for _ in range(max_depth):
            intersect_scene(ro_x, ro_y, ro_z, rd_x, rd_y, rd_z, spheres, plane, hit_info, hit_n, hit_base)
            hit_type = int(hit_info[0])

            if hit_type < 0:
                sky_color(rd_x, rd_y, rd_z, sky)
                col_x += thr_x * sky[0]
                col_y += thr_y * sky[1]
                col_z += thr_z * sky[2]
                break

            hp_x, hp_y, hp_z = hit_info[3], hit_info[4], hit_info[5]
            nx, ny, nz = hit_n[0], hit_n[1], hit_n[2]
            base_x, base_y, base_z = hit_base[0], hit_base[1], hit_base[2]

            if hit_type == 1:
                idx = int(hit_info[1])
                metallic = spheres[idx, 7]
                roughness = spheres[idx, 8]
                specular = spheres[idx, 9]
                transmission = spheres[idx, 10]
                ior = spheres[idx, 11]
            else:
                metallic = plane[13]
                roughness = plane[14]
                specular = plane[15]
                transmission = 0.0
                ior = 1.5

            vdx, vdy, vdz = normalize3(-rd_x, -rd_y, -rd_z)
            local_x, local_y, local_z = 0.02 * base_x, 0.02 * base_y, 0.02 * base_z

            for li in range(lights.shape[0]):
                lx = lights[li, 0] - hp_x
                ly = lights[li, 1] - hp_y
                lz = lights[li, 2] - hp_z
                dist = norm3(lx, ly, lz)
                ldx, ldy, ldz = normalize3(lx, ly, lz)

                intersect_scene(hp_x + nx * EPSILON, hp_y + ny * EPSILON, hp_z + nz * EPSILON,
                                ldx, ldy, ldz, spheres, plane, hit_info, hit_n, hit_base)
                if int(hit_info[0]) >= 0 and hit_info[2] < dist:
                    continue

                ndotl = dot3(nx, ny, nz, ldx, ldy, ldz)
                if ndotl <= 0.0:
                    continue

                hx, hy, hz = normalize3(ldx + vdx, ldy + vdy, ldz + vdz)
                ndoth = dot3(nx, ny, nz, hx, hy, hz)
                if ndoth < 0.0:
                    ndoth = 0.0
                vdoth = dot3(vdx, vdy, vdz, hx, hy, hz)
                if vdoth < 0.0:
                    vdoth = 0.0

                attenuation = lights[li, 6] / (1.0 + 0.09 * dist + 0.032 * dist * dist)

                f0s = 0.04 * (1.0 - metallic) + specular * metallic
                f0r = f0s * (1.0 - metallic) + base_x * metallic
                f0g = f0s * (1.0 - metallic) + base_y * metallic
                f0b = f0s * (1.0 - metallic) + base_z * metallic
                Fr = schlick_scalar(vdoth, f0r)
                Fg = schlick_scalar(vdoth, f0g)
                Fb = schlick_scalar(vdoth, f0b)

                diffuse_r = (1.0 - Fr) * (1.0 - metallic) * base_x * ndotl
                diffuse_g = (1.0 - Fg) * (1.0 - metallic) * base_y * ndotl
                diffuse_b = (1.0 - Fb) * (1.0 - metallic) * base_z * ndotl

                shininess = 2.0 / max(roughness * roughness, 1e-4)
                if shininess < 2.0:
                    shininess = 2.0
                spec = ndoth ** shininess

                local_x += lights[li, 3] * attenuation * (diffuse_r + Fr * spec)
                local_y += lights[li, 4] * attenuation * (diffuse_g + Fg * spec)
                local_z += lights[li, 5] * attenuation * (diffuse_b + Fb * spec)

            col_x += thr_x * local_x
            col_y += thr_y * local_y
            col_z += thr_z * local_z

            cos_theta = dot3(nx, ny, nz, vdx, vdy, vdz)
            if cos_theta < 0.0:
                cos_theta = 0.0
            dielectric_f0 = ((ior - 1.0) / (ior + 1.0)) ** 2
            f0r = dielectric_f0 * (1.0 - metallic) + base_x * metallic
            f0g = dielectric_f0 * (1.0 - metallic) + base_y * metallic
            f0b = dielectric_f0 * (1.0 - metallic) + base_z * metallic
            Fr = schlick_scalar(cos_theta, f0r)
            Fg = schlick_scalar(cos_theta, f0g)
            Fb = schlick_scalar(cos_theta, f0b)

            if transmission > 0.0:
                ok = refract3(rd_x, rd_y, rd_z, nx, ny, nz, ior, refr)
                if ok == 1:
                    ro_x = hp_x - nx * EPSILON
                    ro_y = hp_y - ny * EPSILON
                    ro_z = hp_z - nz * EPSILON
                    rd_x, rd_y, rd_z = refr[0], refr[1], refr[2]
                    thr_x *= (1.0 - Fr) * transmission * base_x
                    thr_y *= (1.0 - Fg) * transmission * base_y
                    thr_z *= (1.0 - Fb) * transmission * base_z
                    continue

            rd_x, rd_y, rd_z = reflect3(rd_x, rd_y, rd_z, nx, ny, nz)
            rd_x, rd_y, rd_z = normalize3(rd_x, rd_y, rd_z)
            ro_x = hp_x + nx * EPSILON
            ro_y = hp_y + ny * EPSILON
            ro_z = hp_z + nz * EPSILON
            scale_thr = 0.25 + 0.75 * max(specular, metallic)
            thr_x *= Fr * scale_thr
            thr_y *= Fg * scale_thr
            thr_z *= Fb * scale_thr

            if max(thr_x, thr_y, thr_z) < 0.01:
                break

        # ACES + gamma
        col_x *= 1.15
        col_y *= 1.15
        col_z *= 1.15

        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        col_x = min(max((col_x * (a * col_x + b)) / (col_x * (c * col_x + d) + e), 0.0), 1.0)
        col_y = min(max((col_y * (a * col_y + b)) / (col_y * (c * col_y + d) + e), 0.0), 1.0)
        col_z = min(max((col_z * (a * col_z + b)) / (col_z * (c * col_z + d) + e), 0.0), 1.0)

        col_x = col_x ** (1.0 / 2.2)
        col_y = col_y ** (1.0 / 2.2)
        col_z = col_z ** (1.0 / 2.2)

        out_img[y, x, 0] = int(min(max(col_x * 255.0, 0.0), 255.0))
        out_img[y, x, 1] = int(min(max(col_y * 255.0, 0.0), 255.0))
        out_img[y, x, 2] = int(min(max(col_z * 255.0, 0.0), 255.0))

        if x == 0 and y < progress.shape[0]:
            progress[y] = 1


def build_camera_basis(eye, look_at):
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye = np.array(eye, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    w = normalize_np(eye - look_at)
    u = normalize_np(np.cross(up, w))
    v = normalize_np(np.cross(w, u))
    basis = np.array([u[0], u[1], u[2], v[0], v[1], v[2], w[0], w[1], w[2]], dtype=np.float32)
    return eye.astype(np.float32), basis


def build_views():
    return [
        ("view_1_front.png",   (0.20, 1.00,  2.80), (0.40, -0.10, -6.40), 42.0),
        ("view_2_right.png",   (3.70, 1.20,  1.40), (0.40, -0.10, -6.10), 44.0),
        ("view_3_left.png",    (-3.80, 1.30, 1.00), (0.30, -0.05, -6.30), 44.0),
        ("view_4_wide.png",    (0.00, 1.60,  5.20), (0.35, -0.05, -6.20), 36.0),
        ("view_5_closeup.png", (1.30, 0.85,  0.70), (0.55,  0.00, -5.90), 50.0),
    ]


def render_gpu_image(filename, eye, look_at, fov, width, height, spheres, plane, lights, max_depth):
    from PIL import Image

    eye_np, basis_np = build_camera_basis(eye, look_at)
    d_spheres = cuda.to_device(spheres)
    d_plane = cuda.to_device(plane)
    d_lights = cuda.to_device(lights)
    d_eye = cuda.to_device(eye_np)
    d_basis = cuda.to_device(basis_np)
    d_img = cuda.device_array((height, width, 3), dtype=np.uint8)
    d_progress = cuda.to_device(np.zeros(height, dtype=np.uint8))

    threads = (16, 16)
    blocks = ((width + threads[0] - 1) // threads[0], (height + threads[1] - 1) // threads[1])

    start = time.time()
    render_kernel[blocks, threads](d_img, d_spheres, d_plane, d_lights, d_eye, d_basis, width, height, fov, max_depth, d_progress)

    last_done = -1
    while True:
        cuda.synchronize()
        prog = d_progress.copy_to_host()
        done = int(prog.sum())
        if done != last_done:
            progress = done / height
            elapsed = time.time() - start
            eta = (elapsed / progress - elapsed) if progress > 0 else 0.0
            print(f"[GPU] Rows done {done}/{height} | {progress*100:6.2f}% | elapsed {elapsed:6.1f}s | ETA {eta:6.1f}s", flush=True)
            last_done = done
        if done >= height:
            break
        time.sleep(0.2)

    img = d_img.copy_to_host()
    Image.fromarray(img).save(filename)
    total = time.time() - start
    print(f"[GPU] Saved {filename} in {total:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Ray tracer with real full-GPU option 3")
    parser.add_argument("--option", choices=["1", "2", "3", "4"], default=None)
    args = parser.parse_args()

    option = args.option if args.option is not None else choose_option_interactively()
    preset = get_preset(option)
    print(f'Using option {option}: {preset["label"]}')

    spheres, plane, lights = build_scene_arrays()
    width, height, max_depth = preset["width"], preset["height"], preset["max_depth"]
    views = build_views()

    if option in ("3", "4"):
        if not CUDA_AVAILABLE:
            print("CUDA/Numba is not available in this Python environment, so full GPU rendering cannot start.")
            print("Install a CUDA-capable Numba stack, then rerun option 3.")
            return

        print("Render method: full GPU CUDA ray tracing")
        try:
            for idx, (filename, eye, look_at, fov) in enumerate(views, start=1):
                print(f"\nRendering view {idx}/{len(views)} -> {filename}")
                render_gpu_image(filename, eye, look_at, fov, width, height, spheres, plane, lights, max_depth)
        except KeyboardInterrupt:
            print("\n[GPU] Render interrupted by user.", flush=True)
    else:
        print("Render method: CPU fallback")
        try:
            from PIL import Image
            prefix = f"option_{option}_"
            for idx, (filename, eye, look_at, fov) in enumerate(views, start=1):
                out_name = prefix + filename
                print(f"\nRendering view {idx}/{len(views)} -> {out_name}")
                img = render_cpu_single(spheres, plane, lights, width, height, eye, look_at, fov, max_depth)
                Image.fromarray(img).save(out_name)
                print(f"Saved {out_name}")
        except KeyboardInterrupt:
            print("\n[CPU fallback] Render interrupted by user.", flush=True)


if __name__ == "__main__":
    main()
