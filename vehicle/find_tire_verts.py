import sys
import json
import numpy as np
import trimesh


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

print(f"Mesh bounds: X {bmin[0]:.3f}..{bmax[0]:.3f}, Y {bmin[1]:.3f}..{bmax[1]:.3f}, Z {bmin[2]:.3f}..{bmax[2]:.3f}")

# ── Axis assignment ───────────────────────────────────────────────────────────
# Vehicles (side view): axes are fixed by Meshy convention
#   mesh X = front-to-rear, mesh Y = height, mesh Z = left-right/depth
# Mechanical (front view): detect from geometry — thin axis = depth into screen
is_vehicle = any(h.get('body_part') == 'wheel' for h in wheel_hints)

if is_vehicle:
    wide_axis = 0   # mesh X = front-to-rear
    tall_axis = 1   # mesh Y = height
    thin_axis = 2   # mesh Z = left-right/depth
    print(f"Axis assignment: vehicle (fixed) wide(FR)=0 tall(UP)=1 thin(LR)=2")
else:
    thin_axis = int(np.argmin(brange))
    remaining = sorted([i for i in range(3) if i != thin_axis],
                       key=lambda i: brange[i], reverse=True)
    wide_axis = remaining[0]
    tall_axis = remaining[1]
    print(f"Axis detection: mechanical (geometry) thin={thin_axis} wide={wide_axis} tall={tall_axis}")

print(f"  ranges: thin={brange[thin_axis]:.3f} wide={brange[wide_axis]:.3f} tall={brange[tall_axis]:.3f}")

centroids = {}
is_two_wheel = len([h for h in wheel_hints if h.get('body_part') == 'wheel']) == 2

for hint in wheel_hints:
    p      = hint.get('position_normalized', {})
    norm_x = np.clip(p.get('x', 0.5), 0.0, 1.0)
    norm_y = np.clip(p.get('y', 0.5), 0.0, 1.0)
    norm_z = np.clip(p.get('z', 0.5), 0.0, 1.0)

    center = np.zeros(3)
    center[wide_axis] = bmin[wide_axis] + norm_x * brange[wide_axis]
    center[tall_axis] = bmin[tall_axis] + norm_y * brange[tall_axis]
    center[thin_axis] = (bmin[thin_axis] + bmax[thin_axis]) / 2.0

    r_norm = hint.get('wheel_radius_normalized', 0.12)
    radius = r_norm * brange[tall_axis] * 1.5
    dists  = np.linalg.norm(verts - center, axis=1)
    nearby = verts[dists <= radius]  # initial capture with r_norm estimate

    if len(nearby) == 0:
        print(f"  ERROR: '{hint['name']}' no vertices found, using joint position")
        centroids[hint['name']] = {'centroid': center.tolist(), 'radius': float(radius)}
        continue

    if len(nearby) > 10:
        y_spread = nearby[:, tall_axis].max() - nearby[:, tall_axis].min()
        z_spread = nearby[:, thin_axis].max() - nearby[:, thin_axis].min()
        true_radius = min(y_spread, z_spread) / 1.5
        print(f"    true_radius from spread: {true_radius:.3f} (vs r_norm radius: {radius:.3f})")
        nearby = verts[dists <= true_radius]
        radius = true_radius

    centroid = nearby.mean(axis=0)
    z_spread = nearby[:, thin_axis].max() - nearby[:, thin_axis].min()
    y_spread = nearby[:, tall_axis].max() - nearby[:, tall_axis].min()
    print(f"  '{hint['name']}': {len(nearby)} verts, centroid={centroid.round(3).tolist()}, spread Y={y_spread:.3f} Z={z_spread:.3f}")

    centroids[hint['name']] = {
        'centroid': centroid.tolist(),
        'radius':   float(radius),  # now saves true_radius
    }


with open(tire_verts_path, 'w') as f:
    json.dump({'wheel_hints': wheel_hints}, f, indent=2)

centroids_path = tire_verts_path.replace('.json', '_centroids.json')
with open(centroids_path, 'w') as f:
    json.dump(centroids, f, indent=2)

print(f"Centroids written: {centroids_path}")
