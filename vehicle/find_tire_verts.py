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

print(f"\nDEBUG: Wheel joints loaded from classify_data:")
for wj in wheel_joints:
    p = wj.get('position_normalized', {})
    print(f"  {wj['name']}: x={p.get('x')}, y={p.get('y')}, z={p.get('z')}")

# Compute tire radius from Gemini
if wheel_joints:
    avg_radius  = np.mean([wj.get('wheel_radius_normalized', 0.18) for wj in wheel_joints])
    tire_radius = mesh_size * avg_radius
else:
    tire_radius = mesh_size * 0.20

print(f"Tire radius: {tire_radius:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE CONVENTION (NO DETECTION)
# ══════════════════════════════════════════════════════════════════════════════
# Claude ALWAYS uses:
#   x = left-right (0=left, 1=right)
#   y = height (0=bottom, 1=top)
#   z = front-rear (0=front, 1=rear)

lr_idx = 0   # left-right axis
h_idx = 1    # height axis
fr_idx = 2   # front-rear axis

if wheel_joints:
    # Get mesh bounds
    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    bounds_range = bounds_max - bounds_min
    
    def claude_to_world(p):
        """Convert Claude normalized position to world position.
        Claude convention: x=left-right, y=height, z=front-rear
        """
        cx = p.get('x', 0.5)
        cy = p.get('y', 0.5)
        cz = p.get('z', 0.5)
        
        world = np.zeros(3)
        world[0] = bounds_min[0] + cx * bounds_range[0]
        world[1] = bounds_min[1] + cy * bounds_range[1]
        world[2] = bounds_min[2] + cz * bounds_range[2]
        
        return world
    
    wheel_world_positions = np.array([
        claude_to_world(jh['position_normalized'])
        for jh in wheel_joints
        if jh.get('position_normalized')
    ])

    print(f"Wheel positions from Gemini: {len(wheel_world_positions)}")
    for i, wp in enumerate(wheel_world_positions):
        print(f"  [{i}]: ({wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f})")

    # Per-wheel tight centroid search
    centroid_mask     = np.zeros(len(positions), dtype=bool)
    wheel_vert_counts = {}


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

    for wp, wj in zip(wheel_world_positions, wheel_joints):
        r_normalized  = wj.get('wheel_radius_normalized', 0.12)
        
        # Search radius should be based on actual wheel radius, not a fixed multiplier
        # Use 1.5x the normalized radius to capture full tire including edges
        search_radius = mesh_size * r_normalized * 1.5  # ← Single scale factor
        
        dists         = np.linalg.norm(positions - wp, axis=1)
        near_wheel    = dists < search_radius
        added         = tire_color_mask_centroid & near_wheel
        centroid_mask |= added
        wheel_vert_counts[wj['name']] = added.sum()
        print(f"  Wheel {wj['name']} r_norm={r_normalized:.3f}, search_r={search_radius:.3f}: {added.sum()} verts")

    wheel_positions = positions[centroid_mask]
    print(f"Centroid vertices total: {centroid_mask.sum()}")

    if len(wheel_positions) == 0:
        print("No wheel vertices found — using fallback")
        wheel_positions = positions[tire_color_mask_centroid]

else:
    # Fallback if no wheel joints
    x_range  = positions[:, 0].max() - positions[:, 0].min()
    y_range  = positions[:, 1].max() - positions[:, 1].min()
    outer_x  = (
        (positions[:, 0] < positions[:, 0].min() + x_range * 0.30) |
        (positions[:, 0] > positions[:, 0].max() - x_range * 0.30)
    )
    bottom_y      = positions[:, 1] < (positions[:, 1].min() + y_range * 0.40)
    centroid_mask = tire_color_mask_centroid & outer_x & bottom_y
    wheel_positions = positions[centroid_mask]

# ══════════════════════════════════════════════════════════════════════════════
# SPLIT WHEELS BY LEFT/RIGHT, THEN FRONT/REAR
# ══════════════════════════════════════════════════════════════════════════════

left_verts  = wheel_positions[wheel_positions[:, lr_idx] < 0]
right_verts = wheel_positions[wheel_positions[:, lr_idx] >= 0]

print(f"Left tire verts: {len(left_verts)}")
print(f"Right tire verts: {len(right_verts)}")

def split_front_rear(verts):
    """Split by front-rear axis (axis 2)"""
    if len(verts) == 0:
        return np.zeros((0,3)), np.zeros((0,3))
    median = np.median(verts[:, fr_idx])
    front = verts[verts[:, fr_idx] < median]
    rear = verts[verts[:, fr_idx] >= median]
    return front, rear

lf, lr = split_front_rear(left_verts)
rf, rr = split_front_rear(right_verts)

print(f"Front-left: {len(lf)}, Front-right: {len(rf)}")
print(f"Rear-left: {len(lr)}, Rear-right: {len(rr)}")

# ══════════════════════════════════════════════════════════════════════════════
# ASSIGN TIRE VERTICES TO WHEELS AND SAVE
# ══════════════════════════════════════════════════════════════════════════════

all_tire_mask = np.zeros(len(positions), dtype=bool)

centroids = [
    (lf.mean(axis=0), 'wheel_fl'),
    (rf.mean(axis=0), 'wheel_fr'),
    (lr.mean(axis=0), 'wheel_rl'),
    (rr.mean(axis=0), 'wheel_rr'),
]

for centroid, name in centroids:
    if len(centroid) == 0:
        continue
    dists2     = np.linalg.norm(positions - centroid, axis=1)
    this_wheel = tire_color_mask & (dists2 < tire_radius * 0.7)
    print(f"  {name}: centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}) → {this_wheel.sum()} verts")
    all_tire_mask |= this_wheel

tire_vertices = np.where(all_tire_mask)[0].tolist()
print(f"\nFinal tire vertices: {len(tire_vertices)}")

with open(tire_verts_path, 'w') as f:
    json.dump(tire_vertices, f)

print(f"Saved {len(tire_vertices)} tire vertices to {tire_verts_path}")


print(f"\nTIRE DETECTION EFFICIENCY:")
print(f"  Total tire-colored verts: {tire_color_mask.sum()}")
print(f"  After normal constraint: {tire_color_mask_centroid.sum()}")
print(f"  Assigned to wheels: {len(tire_vertices)}")
print(f"  Detection efficiency: {len(tire_vertices) / tire_color_mask_centroid.sum() * 100:.1f}%")
print(f"  Unassigned tire verts: {tire_color_mask_centroid.sum() - len(tire_vertices)}")

# Find which vertices were NOT assigned
assigned_set = set(tire_vertices)
unassigned = [i for i in np.where(tire_color_mask_centroid)[0] if i not in assigned_set]
if len(unassigned) > 100:
    unassigned_verts = positions[unassigned]
    print(f"  Unassigned vert distribution:")
    print(f"    X range: {unassigned_verts[:, 0].min():.3f} to {unassigned_verts[:, 0].max():.3f}")
    print(f"    Y range: {unassigned_verts[:, 1].min():.3f} to {unassigned_verts[:, 1].max():.3f}")
    print(f"    Z range: {unassigned_verts[:, 2].min():.3f} to {unassigned_verts[:, 2].max():.3f}")

# Save centroids
wheel_centroids_out = {
    'wheel_fl': lf.mean(axis=0).tolist() if len(lf) > 0 else None,
    'wheel_fr': rf.mean(axis=0).tolist() if len(rf) > 0 else None,
    'wheel_rl': lr.mean(axis=0).tolist() if len(lr) > 0 else None,
    'wheel_rr': rr.mean(axis=0).tolist() if len(rr) > 0 else None,
}
centroids_path = tire_verts_path.replace('.json', '_centroids.json')
with open(centroids_path, 'w') as f:
    json.dump(wheel_centroids_out, f, indent=2)
print(f"Saved wheel centroids to {centroids_path}")
for name, c in wheel_centroids_out.items():
    if c:
        print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")
