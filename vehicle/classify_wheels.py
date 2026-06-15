"""
classify_wheels.py  (Blender script)
=====================================
Step 2 of the vehicle pipeline.

Runs inside Blender (--background --python).  Loads the raw GLB, reads the
Taubin centroids written by find_tire_verts.py, assigns every tire vertex
to its nearest centroid via Voronoi (nearest-neighbour), separates each
wheel into its own mesh object, and exports a separated GLB.

Coordinate systems
------------------
  trimesh (find_tire_verts): X=left/right  Y=height  Z=depth
  Blender world:             X=left/right  Y=depth   Z=height   (Y↔Z swapped)

The Y/Z swap is applied once, at load time, when reading centroids from the
*_centroids.json file (line tagged # ← Y/Z SWAP).

What changed vs the previous version
--------------------------------------
1.  Voronoi vertex assignment replaces the old radius-sphere + manual X/Y/Z
    band cuts.  Every vertex in the tire-vertex pool goes to its nearest
    Taubin centroid.  No more hand-tuned y_hw / z_hw multipliers.

2.  The "outer-face Y correction" step (percentile-snap of centroid[1]) is
    kept but now runs on the Voronoi-assigned verts, not on the sphere verts.

3.  split_front_rear geometric fallback is unchanged — only used when no
    centroids file exists.

4.  Color filter (step 3 of old code) is still applied as a pre-filter to
    build the tire-vertex pool before Voronoi, not per-centroid.
"""

import os, sys, re, struct
sys.path.insert(0, '/tmp/blender_packages')

import bpy, json
import numpy as np

glb_path        = os.path.abspath(sys.argv[sys.argv.index('--') + 1])
output_path     = os.path.abspath(sys.argv[sys.argv.index('--') + 2])
classify_json   = sys.argv[sys.argv.index('--') + 3]
tire_verts_path = sys.argv[sys.argv.index('--') + 4]
mask_dir        = sys.argv[sys.argv.index('--') + 5] \
                  if len(sys.argv) > sys.argv.index('--') + 5 else None

print("glb_path:", glb_path)
print("output_path:", output_path)
print("mask_dir:", mask_dir)

with open(classify_json) as f:
    classify_data = json.load(f)

# Load mesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
mesh_obj     = mesh_objects[0]

verts = np.array([list(mesh_obj.matrix_world @ v.co)
                  for v in mesh_obj.data.vertices])
bmin   = verts.min(axis=0)
bmax   = verts.max(axis=0)
brange = bmax - bmin
brange[brange == 0] = 1.0

print(f"\nMesh bounds (Blender):")
print(f"  X: {bmin[0]:.3f} to {bmax[0]:.3f}")
print(f"  Y: {bmin[1]:.3f} to {bmax[1]:.3f}")
print(f"  Z: {bmin[2]:.3f} to {bmax[2]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD TAUBIN CENTROIDS  (trimesh → Blender: swap Y and Z)
# ══════════════════════════════════════════════════════════════════════════════
centroids_path = tire_verts_path.replace('.json', '_centroids.json')
true_centroids = {}   # name → {'centroid': np.array([x,y,z]), 'radius': float}

if os.path.exists(centroids_path):
    with open(centroids_path) as f:
        saved_centroids = json.load(f)

    for name, v in saved_centroids.items():
        if v is None:
            continue
        pos    = v['centroid'] if isinstance(v, dict) else v
        radius = v.get('radius', 0.2) if isinstance(v, dict) else 0.2
        axis   = v.get('axis', [0, 1, 0]) if isinstance(v, dict) else [0, 1, 0]

        # trimesh: [X, Y, Z] = [lr, height, depth]
        # Blender: [X, Y, Z] = [lr, depth,  height]  ← Y/Z SWAP
        true_centroids[name] = {
            'centroid':   np.array([pos[0], pos[2], pos[1]]),   # ← Y/Z SWAP
            'radius':     radius,
            'axis':       axis,
            'half_thick': v.get('half_thick', 0) if isinstance(v, dict) else 0,
        }

    print(f"\nLoaded {len(true_centroids)} Taubin centroids: {list(true_centroids.keys())}")
else:
    print(f"WARNING: centroids file not found at {centroids_path}")

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC SPLIT FALLBACK (only when no centroids loaded)
# ══════════════════════════════════════════════════════════════════════════════
if not true_centroids:
    print("WARNING: no centroids — falling back to geometric split")
    raw_tv = json.load(open(tire_verts_path))
    tire_vert_indices = [i for i in raw_tv if isinstance(i, int)] if isinstance(raw_tv, list) else []
    tire_verts_np = verts[tire_vert_indices] if tire_vert_indices else np.empty((0, 3))
    num_wheels = len([k for k in classify_data.get('wheel_centroids', {}).keys()
                      if k.startswith('wheel_')])
    print(f"Vehicle has {num_wheels} wheels")

    def split_front_rear(side_verts):
        if len(side_verts) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        median = np.median(side_verts[:, 1])   # Y = depth in Blender
        return side_verts[side_verts[:, 1] < median], side_verts[side_verts[:, 1] >= median]

    if num_wheels == 4:
        left_verts  = tire_verts_np[tire_verts_np[:, 0] < 0]
        right_verts = tire_verts_np[tire_verts_np[:, 0] >= 0]
        lf, lr = split_front_rear(left_verts)
        rf, rr = split_front_rear(right_verts)
        for wname, cluster in [('wheel_fl', lf), ('wheel_fr', rf),
                                ('wheel_rl', lr), ('wheel_rr', rr)]:
            true_centroids[wname] = {
                'centroid': cluster.mean(axis=0) if len(cluster) > 0 else None,
                'radius':   0.2,
            }
    elif num_wheels == 2:
        median = np.median(tire_verts_np[:, 1])
        lf = tire_verts_np[tire_verts_np[:, 1] < median]
        lr = tire_verts_np[tire_verts_np[:, 1] >= median]
        for wname, cluster in [('wheel_fl', lf), ('wheel_rl', lr)]:
            true_centroids[wname] = {
                'centroid': cluster.mean(axis=0) if len(cluster) > 0 else None,
                'radius':   0.2,
            }
    else:
        raise RuntimeError(f"Unexpected wheel count: {num_wheels} and no centroids loaded")

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL COLOR FILTER  (builds tire-vertex pool for Voronoi)
# ══════════════════════════════════════════════════════════════════════════════
color_mask      = None
color_filter_on = False

try:
    wheel_colors_data = classify_data.get('wheel_colors_rgb', [])
    texture_path      = glb_path.replace('_mesh.glb', '_texture.png')
    cid_match = re.search(r'([a-f0-9]{8})_mesh\.glb', glb_path)
    if cid_match and not os.path.exists(texture_path):
        cid          = cid_match.group(1)
        texture_path = os.path.join(os.path.dirname(glb_path), f"{cid}_texture.png")

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
            nc    = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}[acc['type']]
            fmt   = {5126: 'f', 5123: 'H', 5125: 'I'}[acc['componentType']]
            data  = struct.unpack_from(f'<{count*nc}{fmt}', binary, start)
            return np.array(data).reshape(count, nc) if nc > 1 else np.array(data)

        prim   = j['meshes'][0]['primitives'][0]
        uv_idx = prim['attributes'].get('TEXCOORD_0')

        if uv_idx is not None:
            uvs     = read_accessor(uv_idx)
            bpy_img = bpy.data.images.load(os.path.abspath(texture_path))
            iw, ih  = bpy_img.size
            px_flat = np.array(bpy_img.pixels[:]).reshape(ih, iw, 4)
            img_arr = (px_flat[::-1, :, :3] * 255).astype(np.uint8)
            bpy.data.images.remove(bpy_img)
            px_u = (uvs[:, 0] * iw).astype(int) % iw
            px_v = (uvs[:, 1] * ih).astype(int) % ih
            vert_colors = img_arr[px_v, px_u]

            def _to_uint8(data):
                arr = np.array(data)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return (arr * 255).astype(int) if arr.max() <= 1.0 else arr.astype(int)

            if isinstance(wheel_colors_data[0], dict):
                wheel_colors = np.array([[int(c['color'][i]*255) for i in range(3)]
                                         for c in wheel_colors_data])
            else:
                wheel_colors = _to_uint8(wheel_colors_data)

            color_mask = np.zeros(len(verts), dtype=bool)
            for wc in wheel_colors:
                dists_c     = np.sqrt(np.sum((vert_colors.astype(int) - wc)**2, axis=1))
                color_mask |= (dists_c < 80)

            if color_mask.sum() > 100:
                color_filter_on = True
                print(f"\nColor filter: {color_mask.sum()} / {len(verts)} verts "
                      f"({color_mask.sum()/len(verts)*100:.1f}%)")
            else:
                print(f"\nColor filter: too aggressive ({color_mask.sum()} verts) — skipped")
                color_mask = None

except Exception as e:
    print(f"\nColor filter: failed ({e}) — skipped")

# ══════════════════════════════════════════════════════════════════════════════
# VORONOI VERTEX ASSIGNMENT
# Every vertex in the tire-vertex pool is assigned to its nearest Taubin centre.
# The "tire-vertex pool" is: color-filtered verts (if available) OR all verts.
# ══════════════════════════════════════════════════════════════════════════════
centroid_list = [(name, d['centroid'], d['radius'], d.get('half_thick', 0))
                 for name, d in true_centroids.items()
                 if d['centroid'] is not None]

# Precompute centroid positions as a (K, 3) array for vectorised distances
k            = len(centroid_list)
c_names      = [t[0] for t in centroid_list]
c_pos        = np.array([t[1] for t in centroid_list])    # (K, 3)
c_radii      = np.array([t[2] for t in centroid_list])    # (K,)
c_half_thick = np.array([t[3] for t in centroid_list])    # (K,)

print(f"\nVoronoi assignment over {k} centroids:")
for nm, pos, r, ht in centroid_list:
    print(f"  {nm}: ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) r={r:.3f} half_thick={ht:.3f}")

# Build the tire-vertex candidate pool
if color_filter_on and color_mask is not None:
    pool_indices = np.where(color_mask)[0]
else:
    pool_indices = np.arange(len(verts))

pool_verts = verts[pool_indices]   # (P, 3)

# Nearest-centroid for every vertex in the pool  →  assignment array (P,)
# Using broadcasting to avoid an explicit loop
dists_to_centroids = np.linalg.norm(
    pool_verts[:, np.newaxis, :] - c_pos[np.newaxis, :, :], axis=2
)   # (P, K)
nearest = np.argmin(dists_to_centroids, axis=1)   # (P,)


# ── Tyre cylinder gate ────────────────────────────────────────────────────────
# The tyre is a cylinder whose axis runs along X (left/right, the axle direction).
# Blender: X=left/right, Y=depth(front/rear), Z=height
#
# The outer tyre face is a disc at X = centroid_x ± half_thick.
# The full tyre circle spans radius in both Y and Z from the centroid.
#
# So the correct gate is:
#   X ± half_thick  — tyre axial width (thin dimension, excludes chassis inward)
#   Y ± radius      — full circular extent front/rear
#   Z ± radius      — full circular extent height
#
# Previously Y used half_thick which was clipping the outer tyre face disc
# since that disc spans the full radius in Y, not just half_thick.
assigned_r  = c_radii[nearest]
assigned_ht = c_half_thick[nearest]
assigned_c  = c_pos[nearest]   # (P, 3)

# Blender convention: car faces Y, so tyre face normal points along X (left/right).
# X is the thin axis (half_thick), Y and Z span the full tyre circle (radius).
# Voronoi gate: simple sphere of radius around each centroid.
# The tread wraps the full circumference — its axial projection spans the full
# radius in the thin direction, NOT just half_thick. Using half_thick here
# cuts the front/rear tread portions. Use radius for both axial and radial.
# Post-separation slice handles chassis exclusion using half_thick.
in_gate = np.linalg.norm(pool_verts - assigned_c, axis=1) <= assigned_r * 1.1
print(f"  Tyre sphere gate (radius): {in_gate.sum()} / {len(pool_indices)} verts")


# Build per-wheel vertex index lists
wheel_vert_groups_raw = {nm: [] for nm in c_names}
for pi, (ki, gate) in enumerate(zip(nearest, in_gate)):
    if gate:
        wheel_vert_groups_raw[c_names[ki]].append(int(pool_indices[pi]))

for nm, idxs in wheel_vert_groups_raw.items():
    print(f"  {nm}: {len(idxs)} verts assigned")

# Outer-face Y correction removed — animatesam snaps pivot X to outer face
# from the clean separated mesh verts, which is more accurate.

# ══════════════════════════════════════════════════════════════════════════════
# SAM2 MASKS  (kept for future use — loaded but not used in Voronoi path)
# ══════════════════════════════════════════════════════════════════════════════
sam2_masks   = {}
obj_top = obj_left = 0
obj_h   = obj_w    = 1

if mask_dir and os.path.isdir(mask_dir):
    centers_path = os.path.join(mask_dir, 'wheel_centers.json')
    if os.path.exists(centers_path):
        with open(centers_path) as f:
            wheel_centers = json.load(f)
        img_bounds = wheel_centers.get('_image_bounds', {})
        obj_top    = img_bounds.get('obj_top',    0)
        obj_bottom = img_bounds.get('obj_bottom', 1)
        obj_left   = img_bounds.get('obj_left',   0)
        obj_right  = img_bounds.get('obj_right',  1)
        obj_h      = max(obj_bottom - obj_top,  1)
        obj_w      = max(obj_right  - obj_left, 1)

    for name in true_centroids:
        mask_path = os.path.join(mask_dir, f"{name}.png")
        if os.path.exists(mask_path):
            bpy_img  = bpy.data.images.load(os.path.abspath(mask_path))
            px       = np.array(bpy_img.pixels[:])
            mask_arr = (px[::4] > 0.5).reshape(bpy_img.size[1], bpy_img.size[0])
            mask_arr = mask_arr[::-1, :]
            bpy.data.images.remove(bpy_img)
            sam2_masks[name] = mask_arr
            print(f"  Loaded SAM2 mask '{name}': {mask_arr.shape} ({mask_arr.sum()} px)")

# ══════════════════════════════════════════════════════════════════════════════
# VERTEX GROUP CREATION + MESH SEPARATION
# ══════════════════════════════════════════════════════════════════════════════
wheel_vert_groups = [(nm, idxs) for nm, idxs in wheel_vert_groups_raw.items()]

for name, vert_indices in wheel_vert_groups:
    if not vert_indices:
        print(f"  {name}: empty, skipping")
        continue
    vg = mesh_obj.vertex_groups.new(name=name)
    vg.add([int(v) for v in vert_indices], 1.0, 'REPLACE')

for name, vert_indices in wheel_vert_groups:
    if not vert_indices:
        continue

    existing_names = {o.name for o in bpy.data.objects if o.type == 'MESH'}
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    if name not in [vg.name for vg in mesh_obj.vertex_groups]:
        print(f"  {name}: vertex group missing — skipping")
        continue

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.vertex_group_set_active(group=name)
    bpy.ops.object.vertex_group_select()

    bpy.ops.object.mode_set(mode='OBJECT')
    selected_verts = [v for v in mesh_obj.data.vertices if v.select]
    if not selected_verts:
        print(f"  {name}: no verts selected — skipping")
        continue

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')
    new_obj = next(o for o in bpy.data.objects
                   if o.type == 'MESH' and o.name not in existing_names)
    new_obj.name = name
    print(f"  {name}: separated {len(selected_verts)} verts")

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
print(f"\nAfter separation: {len(mesh_objects)} objects")
for o in mesh_objects:
    print(f"  {o.name}: {len(o.data.vertices)} verts")

# ── Clean loose verts + X-axis inner slice ────────────────────────────────────
# After separation each wheel object still contains inner-face verts that are
# chassis geometry (axle housings, suspension etc.) which share colour/position
# with the tyre but face inward toward X=0.
# Slice: left wheels keep verts on the outer (negative) X side of the centroid;
#        right wheels keep verts on the outer (positive) X side.
# Slice boundary = centroid_x + radius * 0.45 (the tyre half-width gate).
all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
body       = max(all_meshes, key=lambda o: len(o.data.vertices))

for obj in all_meshes:
    if obj == body:
        continue

    # Loose vert cleanup first
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Y-axis slice — remove chassis verts outside the tyre's axial thickness.
    # Blender Y = depth (front/rear) = thin axis for a side-facing vehicle.
    # Keep only verts within ± half_thick of the wheel centroid Y.
    # Fallback: ± radius * 0.45 if half_thick not available.
    centroid_data = true_centroids.get(obj.name)
    if centroid_data:
        cx         = float(centroid_data['centroid'][0])
        cy         = float(centroid_data['centroid'][1])
        cz         = float(centroid_data['centroid'][2])
        half_thick = float(centroid_data.get('half_thick', 0))
        radius     = float(centroid_data.get('radius', 0.4))
    else:
        wx_all = np.array([float((obj.matrix_world @ v.co).x) for v in obj.data.vertices])
        cx     = float(wx_all.mean())
        cy     = 0.0; cz = 0.0; half_thick = 0; radius = 0.4

    # Slice on thin axis only (Y for a car facing left in Blender).
    # Uses Taubin axis so it works regardless of car orientation in the GLB.
    # Only removes verts outside ±half_thick along the axle direction.
    # Voronoi already handled radial assignment so no radial cut needed here.
    axis_raw = centroid_data.get('axis', [0, 1, 0]) if centroid_data else [0, 1, 0]
    import mathutils
    thin_axis   = mathutils.Vector(axis_raw).normalized()
    centroid_pt = mathutils.Vector([cx, cy, cz])
    axial_margin = (half_thick * 1.1) if half_thick > 0 else (radius * 0.3)
    print(f"  {obj.name}: thin-axis slice ±{axial_margin:.3f} (axis {[round(float(x),3) for x in thin_axis]})")

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Cut inward chassis verts using world X.
    # Compute the inner face X from the wheel's own vert distribution:
    # the 98th/2nd percentile on the inward side gives the inner tyre face.
    wx_all = np.array([float((obj.matrix_world @ v.co).x) for v in obj.data.vertices])
    if cx < 0:
        # left wheel: inner face = most-positive X (closest to chassis)
        x_inner = float(np.percentile(wx_all, 95))   # 85th pct = inner face
    else:
        # right wheel: inner face = most-negative X (closest to chassis)
        x_inner = float(np.percentile(wx_all, 05))

    print(f"  {obj.name}: X inner face at {x_inner:.3f} (cx={cx:.3f})")

    for v in obj.data.vertices:
        co = obj.matrix_world @ v.co
        wx = float(co.x)
        if cx < 0:
            v.select = bool(wx > x_inner)
        else:
            v.select = bool(wx < x_inner)

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"  Cleaned {obj.name}: {len(obj.data.vertices)} verts remaining")

# ══════════════════════════════════════════════════════════════════════════════
# WRITE BLENDER-SPACE CENTROIDS BACK TO CLASSIFY JSON
# (used by animatesam.py to set pivot origins)
# ══════════════════════════════════════════════════════════════════════════════
blender_centroids = {}
for name, d in true_centroids.items():
    c = d['centroid']
    blender_centroids[name] = {
        'centroid': c.tolist() if hasattr(c, 'tolist') else list(c),
        'radius':   d.get('radius', 0.2),
    }
    print(f"  {name}: Blender centroid {np.array(c).round(3).tolist()}")


with open(classify_json, 'r') as f:
    cdata = json.load(f)
cdata['blender_wheel_centroids'] = blender_centroids
with open(classify_json, 'w') as f:
    json.dump(cdata, f, indent=2)

print(f"\nBlender centroids saved to classify_json.")
bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
print(f"Exported: {output_path}")
