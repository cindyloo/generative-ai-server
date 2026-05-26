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

FR_IDX = 0  # mesh X = front-to-rear
UP_IDX = 1  # mesh Y = height
LR_IDX = 2  # mesh Z = left-right

print(f"Mesh bounds: X {bmin[0]:.3f}..{bmax[0]:.3f}, Y {bmin[1]:.3f}..{bmax[1]:.3f}, Z {bmin[2]:.3f}..{bmax[2]:.3f}")

centroids = {}
is_two_wheel = len([h for h in wheel_hints if h.get('body_part') == 'wheel']) == 2

for hint in wheel_hints:
    p      = hint.get('position_normalized', {})
    norm_x = np.clip(p.get('x', 0.5), 0.0, 1.0)
    norm_y = np.clip(p.get('y', 0.5), 0.0, 1.0)
    norm_z = np.clip(p.get('z', 0.5), 0.0, 1.0)

    center = np.zeros(3)
    if is_two_wheel:
        # Bike: image-x = front-to-rear → mesh X, Z forced to center
        center[FR_IDX] = bmin[FR_IDX] + norm_x * brange[FR_IDX]
        center[LR_IDX] = (bmin[LR_IDX] + bmax[LR_IDX]) / 2.0
    else:
        # Car (4+ wheels): image-z = front-to-rear → mesh X
        #                  image-x = left-right → mesh Z
        center[FR_IDX] = bmin[FR_IDX] + norm_z * brange[FR_IDX]
        center[LR_IDX] = bmin[LR_IDX] + norm_x * brange[LR_IDX]

    center[UP_IDX] = bmin[UP_IDX] + norm_y * brange[UP_IDX]

    r_norm = hint.get('wheel_radius_normalized', 0.12)
    radius = r_norm * brange[UP_IDX] * 1.5
    dists  = np.linalg.norm(verts - center, axis=1)
    nearby = verts[dists <= radius]  # initial capture with r_norm estimate

    if len(nearby) == 0:
        print(f"  ERROR: '{hint['name']}' no vertices found, using joint position")
        centroids[hint['name']] = {'centroid': center.tolist(), 'radius': float(radius)}
        continue

    if len(nearby) > 10:
        y_spread = nearby[:, UP_IDX].max() - nearby[:, UP_IDX].min()
        z_spread = nearby[:, LR_IDX].max() - nearby[:, LR_IDX].min()
        true_radius = min(y_spread, z_spread) / 2.0
        print(f"    true_radius from spread: {true_radius:.3f} (vs r_norm radius: {radius:.3f})")
        nearby = verts[dists <= true_radius]
        radius = true_radius

    centroid = nearby.mean(axis=0)
    z_spread = nearby[:, LR_IDX].max() - nearby[:, LR_IDX].min()
    y_spread = nearby[:, UP_IDX].max() - nearby[:, UP_IDX].min()
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


