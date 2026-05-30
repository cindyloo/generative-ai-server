import os
import sys
import json
import struct
import numpy as np
import trimesh
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('Filtering verts and centroids...')

glb_path        = sys.argv[1]
classify_json   = sys.argv[2]
tire_verts_path = sys.argv[3]
texture_path    = sys.argv[4]

with open(classify_json) as f:
    classify_data = json.load(f)

joint_hints = classify_data.get('joint_hints', [])
wheel_hints = [h for h in joint_hints if h.get('body_part') in ['wheel', 'gear']]

mesh   = trimesh.load(glb_path, force='mesh')
verts  = np.array(mesh.vertices)
bmin   = verts.min(axis=0)
bmax   = verts.max(axis=0)
brange = bmax - bmin
brange[brange == 0] = 1.0
n_verts = len(verts)

print(f"Mesh bounds: X {bmin[0]:.3f}..{bmax[0]:.3f}, "
      f"Y {bmin[1]:.3f}..{bmax[1]:.3f}, Z {bmin[2]:.3f}..{bmax[2]:.3f}")

# ── Axis assignment ───────────────────────────────────────────────────────────
# Vehicles (side view): fixed by Meshy convention
#   mesh X = front-to-rear, mesh Y = height, mesh Z = left-right/depth
# Mechanical (front view): detect from geometry
is_vehicle = any(h.get('body_part') == 'wheel' for h in wheel_hints)

if is_vehicle:
    wide_axis = 0   # mesh X = front-to-rear
    tall_axis = 1   # mesh Y = height
    thin_axis = 2   # mesh Z = left-right/depth
    print(f"Axis assignment: vehicle (fixed) wide=0 tall=1 thin=2")
else:
    thin_axis = int(np.argmin(brange))
    remaining = sorted([i for i in range(3) if i != thin_axis],
                       key=lambda i: brange[i], reverse=True)
    wide_axis = remaining[0]
    tall_axis = remaining[1]
    print(f"Axis detection: mechanical thin={thin_axis} wide={wide_axis} tall={tall_axis}")

print(f"  ranges: thin={brange[thin_axis]:.3f} wide={brange[wide_axis]:.3f} tall={brange[tall_axis]:.3f}")

is_two_wheel = len([h for h in wheel_hints if h.get('body_part') == 'wheel']) == 2

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL COLOR FILTER
# Uses wheel_colors_rgb from classify JSON (merged from joints data).
# Skipped gracefully if not available.
# ══════════════════════════════════════════════════════════════════════════════
candidate_mask  = np.ones(n_verts, dtype=bool)
color_filter_on = False

print(f"\n[Filters]  total verts: {n_verts}")

try:
    wheel_colors_data = classify_data.get('wheel_colors_rgb', [])
    if wheel_colors_data and os.path.exists(texture_path):
        with open(glb_path, 'rb') as f:
            f.read(12)
            json_len = struct.unpack('<I', f.read(4))[0]; f.read(4)
            j        = json.loads(f.read(json_len))
            bin_len  = struct.unpack('<I', f.read(4))[0]; f.read(4)
            binary   = f.read(bin_len)

        def read_accessor(acc_idx):
            acc   = j['accessors'][acc_idx]
            bv    = j['bufferViews'][acc['bufferView']]
            start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
            count = acc['count']
            nc    = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[acc['type']]
            fmt   = {5126:'f', 5123:'H', 5125:'I'}[acc['componentType']]
            data  = struct.unpack_from(f'<{count*nc}{fmt}', binary, start)
            return np.array(data).reshape(count, nc) if nc > 1 else np.array(data)

        prim   = j['meshes'][0]['primitives'][0]
        uv_idx = prim['attributes'].get('TEXCOORD_0')

        if uv_idx is not None:
            uvs     = read_accessor(uv_idx)
            img_arr = np.array(Image.open(texture_path).convert('RGB'))
            ih, iw  = img_arr.shape[:2]
            px_u    = (uvs[:, 0] * iw).astype(int) % iw
            px_v    = ((1.0 - uvs[:, 1]) * ih).astype(int) % ih  # flip V: GLB v=0=bottom, image y=0=top
            vert_colors = img_arr[px_v, px_u]

            if isinstance(wheel_colors_data[0], dict):
                # dict format: {color: [r,g,b]} where values are 0-1 floats
                wheel_colors = np.array([[int(c['color'][i]*255) for i in range(3)]
                                          for c in wheel_colors_data])
            elif isinstance(wheel_colors_data[0][0], float) and max(wheel_colors_data[0]) <= 1.0:
                # list of [r,g,b] floats 0-1
                wheel_colors = np.array([[int(c[i]*255) for i in range(3)]
                                          for c in wheel_colors_data])
            else:
                # list of [r,g,b] integers 0-255 — use directly
                wheel_colors = np.array(wheel_colors_data, dtype=int)

            # Debug: sample actual texture colors
            sample_idx = np.random.choice(len(vert_colors), min(20, len(vert_colors)), replace=False)
            print(f"  Sample vert colors (random): {vert_colors[sample_idx].tolist()}")
            print(f"  Looking for wheel colors: {wheel_colors.tolist()}")
            print(f"  UV range: u={uvs[:,0].min():.3f}..{uvs[:,0].max():.3f} v={uvs[:,1].min():.3f}..{uvs[:,1].max():.3f}")
            print(f"  Texture size: {iw}x{ih}")

            color_mask = np.zeros(n_verts, dtype=bool)
            for wc in wheel_colors:
                dists      = np.sqrt(np.sum((vert_colors.astype(int) - wc)**2, axis=1))
                color_mask |= (dists < 80)

            if color_mask.sum() > 100:
                candidate_mask  = color_mask
                color_filter_on = True
                log.info(f"\nMesh bounds:")
                print(f"  Color filter: {color_mask.sum()} verts "
                      f"({color_mask.sum()/n_verts*100:.1f}%) "
                      f"← colors: {wheel_colors.tolist()}")
            else:
                print(f"  Color filter too aggressive ({color_mask.sum()} verts) — skipped")
        else:
            print(f"  Color filter: skipped (no UVs)")
    else:
        reason = "no wheel_colors_rgb" if not wheel_colors_data else "texture not found"
        print(f"  Color filter: skipped ({reason})")
except Exception as e:
    print(f"  Color filter: failed ({e}) — using all verts")

# ══════════════════════════════════════════════════════════════════════════════
# FIND CENTROIDS
# ══════════════════════════════════════════════════════════════════════════════
centroids = {}

for hint in wheel_hints:
    name  = hint['name']
    p     = hint.get('position_normalized', {})
    norm_x = np.clip(p.get('x', 0.5), 0.0, 1.0)
    norm_y = np.clip(p.get('y', 0.5), 0.0, 1.0)
    norm_z = np.clip(p.get('z', 0.5), 0.0, 1.0)

    # Claude uses y=0=top, mesh uses y=0=bottom — invert
    norm_y_mesh = np.clip(1.0 - p.get('y', 0.5), 0.0, 1.0)

    center = np.zeros(3)
    if is_two_wheel and hint.get('body_part') == 'wheel':
        # Bike: front/rear position is in x, left/right forced to center
        center[wide_axis] = bmin[wide_axis] + norm_x * brange[wide_axis]
        center[thin_axis] = (bmin[thin_axis] + bmax[thin_axis]) / 2.0
    else:
        # Car (4-wheel): Claude x = left/right → thin_axis (mesh Z)
        #                Claude z = front/rear  → wide_axis (mesh X)
        center[wide_axis] = bmin[wide_axis] + norm_z * brange[wide_axis]
        center[thin_axis] = bmin[thin_axis] + norm_x * brange[thin_axis]
    center[tall_axis] = bmin[tall_axis] + norm_y_mesh * brange[tall_axis]

    r_norm = hint.get('wheel_radius_normalized', 0.12)
    radius = r_norm * brange[tall_axis]   # use r_norm directly, no multiplier
    dists  = np.linalg.norm(verts - center, axis=1)

    # Apply color filter to sphere
    in_sphere = np.where(dists <= radius)[0]
    if color_filter_on:
        filtered = in_sphere[candidate_mask[in_sphere]]
        pct = len(filtered) / len(in_sphere) * 100 if len(in_sphere) > 0 else 0
        print(f"\n[{name}] sphere={len(in_sphere)} → color filter={len(filtered)} ({pct:.1f}% kept)")
        if len(filtered) > 20:
            nearby = verts[filtered]
            dists_filtered = dists[filtered]
        else:
            print(f"  Color filter too aggressive — using full sphere")
            nearby = verts[in_sphere]
            dists_filtered = dists[in_sphere]
    else:
        nearby = verts[in_sphere]
        dists_filtered = dists[in_sphere]
        print(f"\n[{name}] sphere={len(in_sphere)} verts")

    if len(nearby) == 0:
        print(f"  ERROR: no vertices found, using seed position")
        centroids[name] = {'centroid': center.tolist(), 'radius': float(radius)}
        continue

    centroid = nearby.mean(axis=0)
    print(f"  '{name}': {len(nearby)} verts, centroid={centroid.round(3).tolist()}, radius={radius:.3f}")

    centroids[name] = {
        'centroid': centroid.tolist(),
        'radius':   float(radius),
    }


with open(tire_verts_path, 'w') as f:
    json.dump({'wheel_hints': wheel_hints}, f, indent=2)

centroids_path = tire_verts_path.replace('.json', '_centroids.json')
with open(centroids_path, 'w') as f:
    json.dump(centroids, f, indent=2)

print(f"\nCentroids written: {centroids_path}")
