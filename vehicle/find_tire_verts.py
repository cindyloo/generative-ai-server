import struct, json, sys
import numpy as np
from PIL import Image

glb_path        = sys.argv[1]
classify_json   = sys.argv[2]
tire_verts_path = sys.argv[3]
texture_path    = sys.argv[4]

with open(classify_json) as f:
    classify_data = json.load(f)

wheel_colors_data = classify_data.get('wheel_colors_rgb', [[0.05, 0.05, 0.05]])
if isinstance(wheel_colors_data[0], dict):
    wheel_colors = np.array([
        [int(c['color'][0]*255), int(c['color'][1]*255), int(c['color'][2]*255)]
        for c in wheel_colors_data
    ])
else:
    wheel_colors = np.array([
        [int(c[0]*255), int(c[1]*255), int(c[2]*255)]
        for c in wheel_colors_data
    ])

print(f"Wheel colors from Gemini: {wheel_colors}")

with open(glb_path, 'rb') as f:
    f.read(12)
    json_len = struct.unpack('<I', f.read(4))[0]
    f.read(4)
    j       = json.loads(f.read(json_len))
    bin_len = struct.unpack('<I', f.read(4))[0]
    f.read(4)
    binary  = f.read(bin_len)

def read_accessor(acc_idx):
    acc   = j['accessors'][acc_idx]
    bv    = j['bufferViews'][acc['bufferView']]
    start = bv.get('byteOffset', 0) + acc.get('byteOffset', 0)
    count = acc['count']
    n_components = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[acc['type']]
    fmt   = {5126:'f', 5123:'H', 5125:'I'}[acc['componentType']]
    data  = struct.unpack_from(f'<{count*n_components}{fmt}', binary, start)
    return np.array(data).reshape(count, n_components) if n_components > 1 else np.array(data)

prim      = j['meshes'][0]['primitives'][0]
positions = read_accessor(prim['attributes']['POSITION'])
uvs       = read_accessor(prim['attributes']['TEXCOORD_0'])

img     = Image.open(texture_path).convert('RGB')
img_arr = np.array(img)
h, w    = img_arr.shape[:2]

colors = np.array([
    [int(x) for x in img_arr[int(uv[1]*h)%h, int(uv[0]*w)%w]]
    for uv in uvs
])

# Match vertex against ANY color in wheel palette
color_threshold = 60
tire_color_mask = np.zeros(len(colors), dtype=bool)
for wc in wheel_colors:
    dists = np.sqrt(np.sum((colors - wc)**2, axis=1))
    tire_color_mask |= (dists < color_threshold)

print(f"Wheel-colored vertices: {tire_color_mask.sum()}")

# Read normals
normal_idx = prim['attributes'].get('NORMAL')
if normal_idx is not None:
    normals = read_accessor(normal_idx)
    wheel_normal_mask        = np.abs(normals[:, 0]) > 0.4
    tire_color_mask_centroid = tire_color_mask & wheel_normal_mask
    print(f"Vertices with outward X normal: {wheel_normal_mask.sum()}")
    print(f"After normal constraint: {tire_color_mask_centroid.sum()}")
else:
    normals                  = None
    tire_color_mask_centroid = tire_color_mask

mesh_size = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))

# Get wheel joints from Gemini
wheel_joints = [
    jh for jh in classify_data.get('joint_hints', [])
    if jh.get('body_part') == 'wheel'
]

# Compute tire radius from Gemini
if wheel_joints:
    avg_radius  = np.mean([wj.get('wheel_radius_normalized', 0.18) for wj in wheel_joints])
    tire_radius = mesh_size * avg_radius
else:
    tire_radius = mesh_size * 0.20

print(f"Tire radius: {tire_radius:.3f}")

if wheel_joints:
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    wheel_world_positions = np.array([
        [
            x_min + jh['position_normalized']['x'] * (x_max - x_min),
            y_min + jh['position_normalized']['z'] * (y_max - y_min),
            z_min + jh['position_normalized']['y'] * (z_max - z_min),
        ]
        for jh in wheel_joints
        if jh.get('position_normalized')
    ])

    print(f"Wheel positions from Gemini: {len(wheel_world_positions)}")
    for i, wp in enumerate(wheel_world_positions):
        print(f"  [{i}]: ({wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f})")

    # Per-wheel tight centroid search
    centroid_mask     = np.zeros(len(positions), dtype=bool)
    wheel_vert_counts = {}

    for wp, wj in zip(wheel_world_positions, wheel_joints):
        r_normalized  = wj.get('wheel_radius_normalized', 0.12)
        search_radius = mesh_size * r_normalized * 0.5
        dists         = np.linalg.norm(positions - wp, axis=1)
        near_wheel    = dists < search_radius
        added         = tire_color_mask_centroid & near_wheel
        centroid_mask |= added
        wheel_vert_counts[wj['name']] = added.sum()
        print(f"  Wheel {wj['name']} r={search_radius:.3f}: {added.sum()} verts")

    # Mirror any wheels that got 0 verts from their opposite side
    def mirror_wheel_name(name):
        replacements = [
            ('front_right', 'front_left'), ('front_left',  'front_right'),
            ('rear_right',  'rear_left'),  ('rear_left',   'rear_right'),
            ('wheel_fr',    'wheel_fl'),   ('wheel_fl',    'wheel_fr'),
            ('wheel_rr',    'wheel_rl'),   ('wheel_rl',    'wheel_rr'),
        ]
        for src, dst in replacements:
            if src in name:
                return name.replace(src, dst)
        return None

    for wj, wp in zip(wheel_joints, wheel_world_positions):
        if wheel_vert_counts.get(wj['name'], 0) == 0:
            mirror_name  = mirror_wheel_name(wj['name'])
            mirror_count = wheel_vert_counts.get(mirror_name, 0)
            if mirror_name and mirror_count > 0:
                for wj2, wp2 in zip(wheel_joints, wheel_world_positions):
                    if wj2['name'] == mirror_name:
                        mirrored_wp   = np.array([-wp2[0], wp2[1], wp2[2]])
                        search_radius = mesh_size * wj.get('wheel_radius_normalized', 0.12) * 0.5
                        dists         = np.linalg.norm(positions - mirrored_wp, axis=1)
                        near_wheel    = dists < search_radius
                        added         = tire_color_mask_centroid & near_wheel
                        centroid_mask |= added
                        print(f"  Mirrored position: ({mirrored_wp[0]:.3f},{mirrored_wp[1]:.3f},{mirrored_wp[2]:.3f})")
                        print(f"  Mirrored {wj['name']} from {mirror_name}: {added.sum()} verts")
                        break

    # Outside both loops
    wheel_positions = positions[centroid_mask]
    print(f"Centroid vertices total: {centroid_mask.sum()}")

    if len(wheel_positions) == 0:
        print("No wheel vertices found — using fallback")
        wheel_positions = positions[tire_color_mask_centroid]

else:
    x_range  = positions[:, 0].max() - positions[:, 0].min()
    y_range  = positions[:, 1].max() - positions[:, 1].min()
    outer_x  = (
        (positions[:, 0] < positions[:, 0].min() + x_range * 0.30) |
        (positions[:, 0] > positions[:, 0].max() - x_range * 0.30)
    )
    bottom_y      = positions[:, 1] < (positions[:, 1].min() + y_range * 0.40)
    centroid_mask = tire_color_mask_centroid & outer_x & bottom_y
    wheel_positions = positions[centroid_mask]

# Compute 4 centroids from wheel_positions
left_verts  = wheel_positions[wheel_positions[:, 0] < 0]
right_verts = wheel_positions[wheel_positions[:, 0] >= 0]

def split_front_rear(verts):
    if len(verts) == 0:
        return np.zeros((0,3)), np.zeros((0,3))
    median = np.median(verts[:, 2])
    return verts[verts[:, 2] < median], verts[verts[:, 2] >= median]

lf, lr = split_front_rear(left_verts)
rf, rr = split_front_rear(right_verts)

valid_centroids = [c.mean(axis=0) for c in [lf, lr, rf, rr] if len(c) > 0]
if not valid_centroids:
    print("Could not compute wheel centroids")
    sys.exit(1)

centroids = np.array(valid_centroids)
print(f"\nWheel centroids ({len(centroids)}):")
for i, c in enumerate(centroids):
    print(f"  [{i}]: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

# Per-wheel X slab + YZ radius selection
if normals is not None:
    left_facing  = normals[:, 0] < -0.4
    right_facing = normals[:, 0] >  0.4
else:
    left_facing  = positions[:, 0] < 0
    right_facing = positions[:, 0] >= 0

 
# Replace the entire X slab section with this simpler approach:
all_tire_mask = np.zeros(len(positions), dtype=bool)

for ci, centroid in enumerate(centroids):
    # Simple spherical selection from centroid
    dists2      = np.linalg.norm(positions - centroid, axis=1)
    this_wheel  = tire_color_mask & (dists2 < tire_radius)
    print(f"  Wheel {ci}: centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}) → {this_wheel.sum()} verts")
    all_tire_mask |= this_wheel

    all_tire_mask |= this_wheel

tire_vertices = np.where(all_tire_mask)[0].tolist()
print(f"\nFinal tire vertices: {len(tire_vertices)}")

with open(tire_verts_path, 'w') as f:
    json.dump(tire_vertices, f)

print(f"Saved {len(tire_vertices)} tire vertices to {tire_verts_path}")
