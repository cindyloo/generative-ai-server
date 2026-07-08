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
    raw = [[c['color'][0], c['color'][1], c['color'][2]] for c in wheel_colors_data]
else:
    raw = [[c[0], c[1], c[2]] for c in wheel_colors_data]

# Normalize to 0-255: if values are already >1 they're integers, otherwise scale
arr = np.array(raw, dtype=float)
if arr.max() <= 1.0:
    wheel_colors = (arr * 255).astype(int)
else:
    wheel_colors = arr.astype(int)

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

# Read normals — kept for reference but NOT used to filter candidate verts.
# |normal_x| > 0.4 was excluding tyre tread verts (which face up/down/front/rear)
# and keeping only the flat side faces, causing tread geometry to be missed.
normal_idx = prim['attributes'].get('NORMAL')
if normal_idx is not None:
    normals = read_accessor(normal_idx)
    print(f"Normals loaded: {len(normals)} (not used for pre-filter)")
else:
    normals = None

# Use colour mask only — spatial + Taubin cylindrical filter handles the rest
tire_color_mask_centroid = tire_color_mask
print(f"Colour-filtered vertices: {tire_color_mask_centroid.sum()}")

mesh_size = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))

# Get wheel joints from Gemini
wheel_joints = [
    jh for jh in classify_data.get('joint_hints', [])
    if jh.get('body_part') in ('wheel', 'gear')
]

# Compute tire radius from Gemini
if wheel_joints:
    avg_radius  = np.mean([wj.get('wheel_radius_normalized', 0.18) for wj in wheel_joints])
    tire_radius = mesh_size * avg_radius
else:
    tire_radius = mesh_size * 0.20

print(f"Tire radius: {tire_radius:.3f}")


def taubin_pivot(verts_3d, hint_x=None):
    """
    Fit a circle to a wheel point cloud in the X/Y plane
    (trimesh: X=front/rear, Y=height).

    The wheel faces the user who looks along trimesh Z (axle direction),
    so the wheel circle lies in the X/Y plane. Taubin fits directly in X/Y,
    giving accurate center_x (front/rear) and center_y (height) from geometry.

    pivot_z = cluster Z midpoint (axle position, ± half_thick).
    hint_x: unused, kept for API compatibility.
    """
    if len(verts_3d) < 10:
        mean = verts_3d.mean(axis=0)
        return mean, np.array([0.0, 0.0, 1.0]), 0.0, 0.0

    X = verts_3d[:, 0]
    Y = verts_3d[:, 1]

    # Taubin circle fit in X/Y plane
    try:
        x_mean, y_mean = X.mean(), Y.mean()
        Xc, Yc = X - x_mean, Y - y_mean
        Z_t  = Xc**2 + Yc**2
        M_z  = np.column_stack((Z_t, Xc, Yc, np.ones(len(Xc))))
        M    = np.dot(M_z.T, M_z) / len(Xc)
        w, v = np.linalg.eig(M)
        imin  = np.argmin(np.abs(w))
        A, B, C, D = v[:, imin]
        center_x = -B / (2 * A) + x_mean
        center_y = -C / (2 * A) + y_mean
    except Exception:
        center_x, center_y = X.mean(), Y.mean()

    rad_dists  = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    radius     = float(np.percentile(rad_dists, 99))

    tight_mask = rad_dists <= radius * 1.05
    if tight_mask.sum() >= 10:
        verts_3d   = verts_3d[tight_mask]
        X, Y       = verts_3d[:, 0], verts_3d[:, 1]
        rad_dists  = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        radius     = float(np.percentile(rad_dists, 99))

    # Pivot Z = midpoint of cluster Z extent (axle position)
    pivot_z    = float((np.percentile(verts_3d[:, 2], 2) +
                        np.percentile(verts_3d[:, 2], 98)) / 2)
    ax_dists   = np.abs(verts_3d[:, 2] - pivot_z)
    half_thick = float(np.percentile(ax_dists, 95))
    if half_thick < radius * 0.05:
        half_thick = radius * 0.3
    half_thick = min(half_thick, radius * 0.6)

    pivot = np.array([center_x, center_y, pivot_z])
    return pivot, np.array([0.0, 0.0, 1.0]), radius, half_thick


output_centroids = {}

if wheel_joints:
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    # Coordinate mapping:
    # trimesh X = front/rear (truck length, widest axis)
    # trimesh Y = height
    # trimesh Z = axle left/right (thin axis)
    #
    # Claude convention (image space, truck facing left):
    #   x = left/right in image  → trimesh Z (axle)
    #   y = top/bottom in image  → trimesh Y (height, inverted)
    #   z = front/rear in image  → trimesh X (truck length)
    wheel_world_positions = np.array([
        [
            x_min + jh['position_normalized']['z'] * (x_max - x_min),          # Claude z → trimesh X (front/rear)
            y_min + (1.0 - jh['position_normalized']['y']) * (y_max - y_min),   # Claude y → trimesh Y (height, flipped)
            z_min + jh['position_normalized']['x'] * (z_max - z_min),           # Claude x → trimesh Z (axle)
        ]
        for jh in wheel_joints
        if jh.get('position_normalized')
    ])

    print(f"Wheel positions from Gemini: {len(wheel_world_positions)}")
    for i, wp in enumerate(wheel_world_positions):
        print(f"  [{i}]: ({wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f})")

    wheel_vert_counts = {}
    wheel_vert_clusters = {}   # name → position array for Taubin

    def build_clusters(search_radius_override=None):
        counts  = {}
        clusters = {}
        for wp, wj in zip(wheel_world_positions, wheel_joints):
            r_normalized  = wj.get('wheel_radius_normalized', 0.12)
            sr = search_radius_override if search_radius_override else mesh_size * r_normalized * 0.5
            dists      = np.linalg.norm(positions - wp, axis=1)
            added_mask = tire_color_mask_centroid & (dists < sr)
            counts[wj['name']]   = added_mask.sum()
            clusters[wj['name']] = positions[added_mask]
            print(f"  Wheel {wj['name']} r={sr:.3f}: {added_mask.sum()} verts")
        return counts, clusters

    # ── Pass 1: initial clusters with Gemini-derived search radius ────────────
    print("Pass 1 (initial search radius):")
    wheel_vert_counts, wheel_vert_clusters = build_clusters()

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

    # ── Detect vehicle type for mirroring strategy ────────────────────────────
    # Car/truck: wheels come in left/right pairs (fl+fr, rl+rr) — Taubin on
    #   right wheels only, mirror Z to left wheels.
    # Everything else (bike, mechanical, gears): run Taubin on all joints
    #   independently. If any joint has too few verts, borrow geometry from
    #   the cleanest one (most verts). No mirroring.

    is_mechanical_obj = classify_data.get('category', '') == 'mechanical' or \
                        classify_data.get('rig_type', '') == 'mechanical'
    wheel_names = [wj['name'] for wj in wheel_joints]
    has_right   = any('fr' in n or 'right' in n or 'rr' in n for n in wheel_names)
    has_left    = any('fl' in n or 'left'  in n or 'rl' in n for n in wheel_names)
    is_paired   = has_right and has_left and not is_mechanical_obj

    if is_paired:
        print(f"  Paired vehicle (car/truck) — Taubin on right wheels, mirror Z")
    else:
        print(f"  Independent joints (bike/mechanical) — Taubin on all, no mirroring")

    # ── Pass 1 Taubin ─────────────────────────────────────────────────────────
    pass1_results = {}   # name → (pivot, axis, radius, half_thick, n_verts)
    pass1_capture = {}   # name → capture radius (max radial extent incl. teeth)

    for wj, wp in zip(wheel_joints, wheel_world_positions):
        name = wj['name']
        p    = wj.get('position_normalized', {})

        if is_paired and not ('fr' in name or 'right' in name or 'rr' in name):
            continue   # car/truck: right wheels only

        cluster = wheel_vert_clusters.get(name, np.empty((0, 3)))
        if len(cluster) < 10:
            continue

        # Z band filter: keep only verts near this gear's Z position.
        # For gears/bikes at same X/Y but different Z, this separates clusters
        # before Taubin runs so each gets its own clean set of verts.
        r_normalized       = wj.get('wheel_radius_normalized', 0.12)
        y_range_mesh       = positions[:, 1].max() - positions[:, 1].min()
        wheel_radius_world = r_normalized * y_range_mesh

        z_dist = np.abs(cluster[:, 2] - wp[2])
        z_cluster = cluster[z_dist <= wheel_radius_world]
        if len(z_cluster) >= 10:
            cluster = z_cluster
            print(f"  {name}: Z band filter kept {len(cluster)} verts "
                  f"(Z in [{wp[2]-wheel_radius_world:.3f}, {wp[2]+wheel_radius_world:.3f}], "
                  f"gear center Z={wp[2]:.3f})")
        else:
            print(f"  {name}: Z band filter too aggressive — skipping")
        x_forward_limit = wp[0] - wheel_radius_world
        x_rear_limit    = wp[0] + wheel_radius_world
        x_mask          = (cluster[:, 0] >= x_forward_limit) & \
                          (cluster[:, 0] <= x_rear_limit)
        x_cluster       = cluster[x_mask]
        if len(x_cluster) >= 10:
            cluster = x_cluster
            print(f"  {name}: X band filter kept {len(cluster)} verts "
                  f"(X in [{x_forward_limit:.3f}, {x_rear_limit:.3f}], "
                  f"wheel center X={wp[0]:.3f})")
        else:
            print(f"  {name}: X band filter too aggressive — skipping")

        pivot, axis, radius, half_thick = taubin_pivot(cluster, hint_x=wp[0])
        pivot = np.array([pivot[0], pivot[1], wp[2]])  # use Claude's Z directly

        # Axis-aligned box filter — X/Y radius, Z from Claude hint
        x_mask     = np.abs(cluster[:, 0] - pivot[0]) <= radius * 1.1
        y_mask     = np.abs(cluster[:, 1] - pivot[1]) <= radius * 1.1
        clean_mask = x_mask & y_mask
        clean_cluster = cluster[clean_mask]

        if len(clean_cluster) >= 10:
            cluster = clean_cluster
            pivot, axis, radius, half_thick = taubin_pivot(cluster, hint_x=wp[0])
            pivot = np.array([pivot[0], pivot[1], wp[2]])  # use Claude's Z directly

        pass1_results[name] = (pivot, axis, radius, half_thick, len(cluster))

        # Capture radius: max radial extent of the cluster (tooth tips sit
        # beyond the Taubin pitch-circle fit, which is a 99th-pct estimate).
        # Used downstream by classify_wheels as the vertex-capture gate;
        # the Taubin radius stays authoritative for gear speed ratios.
        rad_xy = np.sqrt((cluster[:, 0] - pivot[0])**2 +
                         (cluster[:, 1] - pivot[1])**2)
        pass1_capture[name] = float(rad_xy.max() * 1.02)

        print(f"  Pass 1 {name}: pivot={np.round(pivot,4).tolist()} "
              f"radius={radius:.4f}  half_thick={half_thick:.4f}  "
              f"capture={pass1_capture[name]:.4f}  verts={len(cluster)}")

    # ── Output centroids ──────────────────────────────────────────────────────
    print("\nTaubin fits:")

    # For non-paired: find cleanest joint as fallback geometry source
    if not is_paired and pass1_results:
        cleanest_name = max(pass1_results.keys(), key=lambda n: pass1_results[n][4])

    for wj, wp in zip(wheel_joints, wheel_world_positions):
        name    = wj['name']

        if is_paired:
            is_right = 'fr' in name or 'right' in name or 'rr' in name
            if is_right:
                # Car/truck right wheel — use own Taubin result
                if name in pass1_results:
                    pivot, axis, radius, half_thick, _ = pass1_results[name]
                    print(f"  {name}: pivot={np.round(pivot,4).tolist()} radius={radius:.4f}")
                    output_centroids[name] = {
                        'centroid':       [float(x) for x in pivot],
                        'radius':         float(radius),
                        'half_thick':     float(half_thick),
                        'capture_radius': pass1_capture.get(name, radius * 1.15),
                        'name':           name,
                        'axis':           [1.0, 0.0, 0.0],
                    }
            else:
                # Car/truck left wheel — mirror Z from right counterpart
                mirror_name = mirror_wheel_name(name)
                if mirror_name and mirror_name in pass1_results:
                    pivot, axis, radius, half_thick, _ = pass1_results[mirror_name]
                    mirrored_pivot = np.array([pivot[0], pivot[1], -pivot[2]])
                    print(f"  {name}: mirrored from {mirror_name} "
                          f"pivot={np.round(mirrored_pivot,4).tolist()} radius={radius:.4f}")
                    output_centroids[name] = {
                        'centroid':       [float(x) for x in mirrored_pivot],
                        'radius':         float(radius),
                        'half_thick':     float(half_thick),
                        'capture_radius': pass1_capture.get(mirror_name, radius * 1.15),
                        'name':           name,
                        'axis':           [1.0, 0.0, 0.0],
                    }
        else:
            # Bike/mechanical: each joint uses its own Taubin result.
            # Z position always comes from Claude's hint (wp[2]).
            if name in pass1_results:
                pivot, axis, radius, half_thick, _ = pass1_results[name]
            elif pass1_results:
                pivot, axis, radius, half_thick, _ = pass1_results[cleanest_name]
                pivot = np.array([wp[0], pivot[1], wp[2]])
                print(f"  {name}: geometry from {cleanest_name}, X/Z from hint")
            else:
                continue

            # Ensure pivot Z = Claude's hint Z
            # Keep the MEASURED half_thick: stacked/coaxial gears (e.g. watch
            # movements) are distinguished only by their axial band. Setting
            # half_thick = radius made every gear's capture volume swallow
            # its coaxial neighbours, so the first gear separated stole most
            # of the others' vertices.
            final_pivot = np.array([pivot[0], pivot[1], wp[2]])
            print(f"  {name}: pivot={np.round(final_pivot,4).tolist()} "
                  f"radius={radius:.4f} half_thick={half_thick:.4f}")
            output_centroids[name] = {
                'centroid':       [float(x) for x in final_pivot],
                'radius':         float(radius),
                'half_thick':     float(half_thick),
                'capture_radius': pass1_capture.get(name, radius * 1.15),
                'name':           name,
                'axis':           [1.0, 0.0, 0.0],
            }

else:
    # No wheel joints — fall back to quadrant split + Taubin
    x_range  = positions[:, 0].max() - positions[:, 0].min()
    y_range  = positions[:, 1].max() - positions[:, 1].min()
    outer_x  = (
        (positions[:, 0] < positions[:, 0].min() + x_range * 0.30) |
        (positions[:, 0] > positions[:, 0].max() - x_range * 0.30)
    )
    bottom_y      = positions[:, 1] < (positions[:, 1].min() + y_range * 0.40)
    centroid_mask = tire_color_mask_centroid & outer_x & bottom_y
    wheel_positions = positions[centroid_mask]

    left_verts  = wheel_positions[wheel_positions[:, 0] < 0]
    right_verts = wheel_positions[wheel_positions[:, 0] >= 0]

    def split_front_rear(verts):
        if len(verts) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        median = np.median(verts[:, 2])
        return verts[verts[:, 2] < median], verts[verts[:, 2] >= median]

    lf, lr = split_front_rear(left_verts)
    rf, rr = split_front_rear(right_verts)

    for name, cluster in [('wheel_fl', lf), ('wheel_rl', lr),
                           ('wheel_fr', rf), ('wheel_rr', rr)]:
        if len(cluster) < 10:
            continue
        pivot, axis, radius, half_thick = taubin_pivot(cluster)
        print(f"  {name}: pivot={np.round(pivot,4).tolist()}  radius={radius:.4f}")
        output_centroids[name] = {
            'centroid': [float(x) for x in pivot],
            'radius':   float(radius),
            'name':     name,
            'axis':     [float(x) for x in axis] if axis is not None else [1.0, 0.0, 0.0],
        }

# ── Also write the flat tire_verts list for backward compat ──────────────────
# (classify_wheels.py loads centroids_path; tire_verts_path is legacy)
all_tire_mask = np.zeros(len(positions), dtype=bool)
for name, d in output_centroids.items():
    pivot  = np.array(d['centroid'])
    radius = d['radius']
    dists  = np.linalg.norm(positions - pivot, axis=1)
    all_tire_mask |= (tire_color_mask & (dists < radius * 0.7))

tire_vertices = np.where(all_tire_mask)[0].tolist()
print(f"\nFinal tire vertices: {len(tire_vertices)}")
with open(tire_verts_path, 'w') as f:
    json.dump(tire_vertices, f)

# ── Export centroids ──────────────────────────────────────────────────────────
classify_data['wheel_centroids'] = output_centroids
with open(classify_json, 'w') as f:
    json.dump(classify_data, f, indent=4)

centroids_path = (tire_verts_path.replace('.json', '_centroids.json')
                  if tire_verts_path.endswith('.json')
                  else tire_verts_path + '_centroids.json')
with open(centroids_path, 'w') as f:
    json.dump(output_centroids, f, indent=4)

print(f"INFO:seg_server:High-Precision Centroids Exported to {centroids_path}")
print(f"INFO:seg_server:Wheel centroids injected from find_tire_verts: {json.dumps(output_centroids)}")
