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
    if jh.get('body_part') == 'wheel'
]

# Compute tire radius from Gemini
if wheel_joints:
    avg_radius  = np.mean([wj.get('wheel_radius_normalized', 0.18) for wj in wheel_joints])
    tire_radius = mesh_size * avg_radius
else:
    tire_radius = mesh_size * 0.20

print(f"Tire radius: {tire_radius:.3f}")


def taubin_pivot(verts_3d):
    """
    Given a 3D point cloud believed to be one wheel, return
    (precise_pivot, rotation_axis, radius) using PCA + Taubin circle fit.

    PCA finds the wheel plane (smallest-eigenvalue axis = axle direction).
    Taubin fits a circle in the projected 2D plane.
    Falls back to (mean, None, percentile_radius) on failure.
    """
    if len(verts_3d) < 10:
        return verts_3d.mean(axis=0), None, 0.0, 0.0

    mean     = verts_3d.mean(axis=0)
    centered = verts_3d - mean
    cov      = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Smallest eigenvalue → axle direction (normal to wheel plane)
    rotation_axis = eigenvectors[:, np.argmin(eigenvalues)]
    u_axis        = eigenvectors[:, np.argmax(eigenvalues)]
    v_axis        = np.cross(rotation_axis, u_axis)

    # Axial thickness cut — removes undercarriage / axle bleed
    depths    = np.dot(centered, rotation_axis)
    depth_std = np.std(depths)
    axial_mask = np.abs(depths) < (depth_std * 2.0)
    verts_3d  = verts_3d[axial_mask]
    mean      = verts_3d.mean(axis=0)
    centered  = verts_3d - mean

    pts_2d = np.column_stack((np.dot(centered, u_axis),
                               np.dot(centered, v_axis)))

    try:
        X, Y = pts_2d[:, 0], pts_2d[:, 1]
        Z_t  = X**2 + Y**2
        M_z  = np.column_stack((Z_t, X, Y, np.ones(len(X))))
        M    = np.dot(M_z.T, M_z) / len(X)
        w, v = np.linalg.eig(M)
        imin  = np.argmin(np.abs(w))
        A, B, C, D = v[:, imin]

        center_u = -B / (2 * A)
        center_v = -C / (2 * A)
        pivot    = mean + (center_u * u_axis) + (center_v * v_axis)

        # Cylindrical re-filter
        rel       = verts_3d - pivot
        ax_d      = np.dot(rel, rotation_axis)
        rad_dists = np.linalg.norm(rel - np.outer(ax_d, rotation_axis), axis=1)
        radius    = float(np.percentile(rad_dists, 99))

        # Tight cylinder pass
        tight = ((rad_dists <= radius * 1.05) &
                 (np.abs(ax_d) <= radius * 0.6))
        verts_3d = verts_3d[tight]
        rel       = verts_3d - pivot
        ax_d      = np.dot(rel, rotation_axis)
        rad_dists = np.linalg.norm(rel - np.outer(ax_d, rotation_axis), axis=1)
        radius    = float(np.percentile(rad_dists, 99))

        # Compute half_thick: 95th percentile of axial depths on final clean verts
        rel        = verts_3d - pivot
        ax_d       = np.dot(rel, rotation_axis)
        half_thick = float(np.percentile(np.abs(ax_d), 95))
        if half_thick < radius * 0.05:   # unreliable — fall back
            half_thick = radius * 0.5
        # Cap: tyre width should not exceed 60% of radius (physical constraint)
        half_thick = min(half_thick, radius * 0.6)

        return pivot, rotation_axis, radius, half_thick

    except Exception as e:
        print(f"    Taubin fit failed ({e}) — using mean")
        rel        = verts_3d - mean
        ax_d       = np.dot(rel, rotation_axis)
        rad_dists  = np.linalg.norm(rel - np.outer(ax_d, rotation_axis), axis=1)
        half_thick = float(np.percentile(np.abs(ax_d), 95))
        return mean, rotation_axis, float(np.percentile(rad_dists, 99)), half_thick


output_centroids = {}

if wheel_joints:
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

    # Gemini convention: x=left/right, y=height (0=top), z=depth (0=front)
    # trimesh convention: X=left/right, Y=height, Z=depth
    # Gemini y is image-space (0=top), so height maps as 1-y
    wheel_world_positions = np.array([
        [
            x_min + jh['position_normalized']['x'] * (x_max - x_min),          # lr
            y_min + (1.0 - jh['position_normalized']['y']) * (y_max - y_min),   # height (flipped)
            z_min + jh['position_normalized']['z'] * (z_max - z_min),           # depth
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

    # ── Pass 1 Taubin on left wheels ─────────────────────────────────────────
    # Fit each left wheel with the initial (possibly contaminated) clusters.
    # The cleanest wheel is the one with the fewest verts — least chassis bleed.
    # Use its Taubin result (pivot, axis, radius, half_thick) as the template
    # for Pass 2 cylindrical filtering on ALL left wheels.
    pass1_results = {}   # name → (pivot, axis, radius, half_thick)
    for wj in wheel_joints:
        name = wj['name']
        p    = wj.get('position_normalized', {})
        if p.get('x', 0.5) >= 0.5:
            continue   # left wheels only
        cluster = wheel_vert_clusters.get(name, np.empty((0, 3)))
        if len(cluster) < 10:
            continue
        pivot, axis, radius, half_thick = taubin_pivot(cluster)
        pass1_results[name] = (pivot, axis, radius, half_thick, len(cluster))
        print(f"  Pass 1 {name}: radius={radius:.4f}  half_thick={half_thick:.4f}  verts={len(cluster)}")

    # ── Identify the cleanest left wheel (fewest verts = least contamination) ─
    if pass1_results:
        cleanest = min(pass1_results.keys(), key=lambda n: pass1_results[n][4])
        clean_pivot, clean_axis, clean_radius, clean_half_thick, _ = pass1_results[cleanest]
        # Cap clean_half_thick: if contamination inflated it, use radius*0.3 as ceiling
        # (a tyre's axial half-thickness is typically 25-35% of its rolling radius)
        clean_half_thick = min(clean_half_thick, clean_radius * 0.3)
        print(f"  Cleanest wheel: {cleanest} (radius={clean_radius:.4f} half_thick={clean_half_thick:.4f} capped)")

        # ── Pass 2: cylindrical filter per left wheel using clean template ────
        # For each left wheel, start from the Pass 1 cluster and remove any
        # vert that lies outside the cylinder defined by:
        #   - axial depth along thin axis: |depth| <= clean_half_thick * 1.1
        #   - radial distance from pivot:  rad    <= clean_radius * 1.1
        # This removes chassis verts that don't lie on the wheel's circular ring.
        print("Pass 2 (cylindrical filter using cleanest wheel geometry):")
        for wj in wheel_joints:
            name = wj['name']
            p    = wj.get('position_normalized', {})
            if p.get('x', 0.5) >= 0.5:
                continue
            if name not in pass1_results:
                continue

            pivot, axis, radius, half_thick, _ = pass1_results[name]
            cluster = wheel_vert_clusters[name]

            # Project onto thin axis — keep only verts within clean_half_thick
            rel        = cluster - pivot
            ax_depths  = np.dot(rel, clean_axis)
            rad_vecs   = rel - np.outer(ax_depths, clean_axis)
            rad_dists  = np.linalg.norm(rad_vecs, axis=1)

            axial_mask = np.abs(ax_depths) <= clean_half_thick * 1.1
            radial_mask = rad_dists <= clean_radius * 1.1
            clean_mask  = axial_mask & radial_mask

            wheel_vert_clusters[name] = cluster[clean_mask]
            wheel_vert_counts[name]   = clean_mask.sum()
            print(f"  {name}: {len(cluster)} → {clean_mask.sum()} verts "
                  f"(axial kept={axial_mask.sum()} radial kept={radial_mask.sum()})")
    else:
        print("Pass 2 skipped — no Pass 1 results available")

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
                        added_mask    = tire_color_mask_centroid & near_wheel
                        wheel_vert_clusters[wj['name']] = positions[added_mask]
                        wheel_vert_counts[wj['name']]   = added_mask.sum()
                        print(f"  Mirrored {wj['name']} from {mirror_name}: {added_mask.sum()} verts")
                        break

    # ── Taubin fit per wheel ──────────────────────────────────────────────────
    # Strategy:
    #   1. Run Taubin on left wheels (x < 0) — these are visible and have clean geometry.
    #   2. Mirror the result to the corresponding right wheel (flip X of pivot).
    #      The axis and radius are the same; only the pivot X sign changes.
    #   3. Use the PCA thin axis (rotation_axis from taubin_pivot) to compute
    #      the axial half-thickness for the chassis slice in classify_wheels.
    #      If the thin axis is unreliable, fall back to radius * 0.5.

    print(f"\nTaubin fits:")

    left_results  = {}   # name → (pivot, axis, radius) for left wheels
    right_results = {}   # name → corresponding left wheel name

    # Separate left and right wheel names
    for wj in wheel_joints:
        name = wj['name']
        p    = wj.get('position_normalized', {})
        if p.get('x', 0.5) < 0.5:
            left_results[name] = None   # to be filled by Taubin
        else:
            # Map right → left mirror name
            mirror = mirror_wheel_name(name)
            right_results[name] = mirror

    # Run Taubin on left wheels
    for wj in wheel_joints:
        name    = wj['name']
        if name not in left_results:
            continue
        cluster = wheel_vert_clusters.get(name, np.empty((0, 3)))
        if len(cluster) < 10:
            print(f"  {name}: too few verts ({len(cluster)}) — skipping")
            continue

        pivot, axis, radius, half_thick = taubin_pivot(cluster)

        # Compute axial half-thickness from thin axis projection
        rel        = cluster - pivot
        ax_depths  = np.dot(rel, axis)
        half_thick = float(np.percentile(np.abs(ax_depths), 95))
        if half_thick < radius * 0.1:   # unreliable — fall back
            half_thick = radius * 0.5

        print(f"  {name}: pivot={np.round(pivot,4).tolist()}  "
              f"radius={radius:.4f}  half_thick={half_thick:.4f}  verts={len(cluster)}")

        # Recenter pivot X to the midpoint of the cluster's X extent.
        # Taubin fits a circle in the Y-Z plane (axle runs along Y), so its
        # X coordinate drifts based on vert density asymmetry.
        # The true axle center in X is simply the midpoint of the tyre's X span.
        cluster_x = wheel_vert_clusters[name][:, 0]
        pivot_x   = float((np.percentile(cluster_x, 2) + np.percentile(cluster_x, 98)) / 2)
        pivot     = np.array([pivot_x, pivot[1], pivot[2]])

        left_results[name] = (pivot, axis, radius, half_thick)
        output_centroids[name] = {
            'centroid':   [float(x) for x in pivot],
            'radius':     float(radius),
            'half_thick': float(half_thick),
            'name':       name,
            'axis':       [float(x) for x in axis] if axis is not None else [1.0, 0.0, 0.0],
        }

    # Mirror left results onto right wheels
    for wj in wheel_joints:
        name        = wj['name']
        mirror_name = right_results.get(name)
        if mirror_name is None:
            continue   # this is a left wheel, already handled

        src = left_results.get(mirror_name)
        if src is None:
            print(f"  {name}: mirror source {mirror_name} not available — skipping")
            continue

        pivot, axis, radius, half_thick = src

        # Mirror: flip X of pivot, flip X component of axis
        mirrored_pivot = np.array([-pivot[0],  pivot[1],  pivot[2]])
        mirrored_axis  = np.array([-axis[0],   axis[1],   axis[2]])

        print(f"  {name}: mirrored from {mirror_name}  "
              f"pivot={np.round(mirrored_pivot,4).tolist()}  radius={radius:.4f}")

        output_centroids[name] = {
            'centroid':   [float(x) for x in mirrored_pivot],
            'radius':     float(radius),
            'half_thick': float(half_thick),
            'name':       name,
            'axis':       [float(x) for x in mirrored_axis],
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
