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

        # trimesh: X=front/rear, Y=height, Z=axle(left/right)
        # Blender: X=front/rear, Y=axle(left/right), Z=height
        # Blender X = trimesh X, Blender Y = trimesh Z, Blender Z = trimesh Y
        true_centroids[name] = {
            'centroid':       np.array([pos[0], pos[2], pos[1]]),   # X unchanged, Y↔Z swap
            'radius':         radius,
            'axis':           axis,
            'half_thick':     v.get('half_thick', 0) if isinstance(v, dict) else 0,
            'capture_radius': (v.get('capture_radius', radius * 1.15)
                               if isinstance(v, dict) else radius * 1.15),
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
# ══════════════════════════════════════════════════════════════════════════════
# PER-WHEEL VERTEX ASSIGNMENT
# Right-side wheels (positive centroid X): use color filter — synthesized texture
#   gives clean color matches on the right/non-photo side of the mesh.
# Left-side wheels (negative centroid X): use ALL mesh verts — the visible/photo
#   side has noisy texture that doesn't match the color palette reliably.
# Both sides use the asymmetric X gate:
#   outward: full radius (tread extent)
#   inward:  half_thick (chassis exclusion)
#   radial:  radius in Y/Z plane
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPer-wheel vertex assignment:")

centroid_list = [(name, d['centroid'], d['radius'], d.get('half_thick', 0),
                  d.get('capture_radius', d['radius'] * 1.15))
                 for name, d in true_centroids.items()
                 if d['centroid'] is not None]

wheel_vert_groups_raw = {}

for name, pos, radius, half_thick, capture_r in centroid_list:
    pivot    = np.array(pos)
    is_left  = pivot[1] < 0   # Blender Y < 0 = left side (toward user)

    # Choose vertex pool: ALL verts for both left and right wheels.
    # Color filter applied within the box gate for all wheels.
    pool_idx   = np.arange(len(verts))
    pool_v     = verts
    pool_label = "all verts"

    # Gate selection:
    # Paired vehicle (car/truck): asymmetric Y gate — outward full radius,
    #   inward half_thick only (excludes chassis geometry)
    # Everything else (bike/gear/mechanical): simple radial gate — full
    #   sphere around pivot, no inward/outward distinction
    is_paired_vehicle = classify_data.get('category', '') == 'vehicle' and \
                        any('fr' in n or 'rr' in n or 'right' in n
                            for n in true_centroids.keys())

    xz_dist = np.sqrt((pool_v[:, 0] - pivot[0])**2 +
                      (pool_v[:, 2] - pivot[2])**2)

    if is_paired_vehicle:
        y_rel     = pool_v[:, 1] - pivot[1]
        outward_y = np.where(pivot[1] < 0, -y_rel, y_rel)
        box_gate = (
            (xz_dist   <= radius     * 1.1) &
            (outward_y >= -(half_thick * 1.0)) &
            (outward_y <=  (radius    * 1.1))
        )
    else:
        # Unpaired gear / bike wheel: full CYLINDER around the axle
        # (Blender Y), not a sphere. A sphere of the Taubin radius clips
        # exactly the rim exterior: a tooth tip at radial distance ≈ r,
        # axially offset by half the gear thickness, sits at Euclidean
        # distance sqrt(r² + t²) > r.
        #   radial: capture_radius (max cluster extent incl. tooth tips —
        #           the Taubin fit is a 99th-pct pitch-circle estimate)
        #   axial:  measured half-thickness both sides → whole gear,
        #           both faces, exterior included
        ht       = half_thick if half_thick > 1e-6 else radius
        axial    = np.abs(pool_v[:, 1] - pivot[1])
        box_gate = (xz_dist <= max(capture_r, radius * 1.1)) & \
                   (axial   <= ht * 1.25)

    # Color filter within box — excludes non-wheel colored geometry
    # (blue cab, body panels) that falls inside the spatial box.
    # Fallback to box-only if color filter removes too many verts.
    if color_filter_on and color_mask is not None:
        color_in_box   = box_gate & color_mask
        box_count      = int(box_gate.sum())
        color_count    = int(color_in_box.sum())
        if color_count >= box_count * 0.1 and color_count > 20:
            gate = color_in_box
            print(f"    box={box_count} color_in_box={color_count}")
        else:
            gate = box_gate
            print(f"    color too aggressive ({color_count}/{box_count}) — box only")
    else:
        gate = box_gate
        print(f"    box={int(box_gate.sum())} verts (no color filter)")

    indices = pool_idx[gate].tolist()
    wheel_vert_groups_raw[name] = indices
    print(f"  {name}: {len(indices)} verts ({pool_label}, "
          f"cx={pivot[0]:.3f} r={radius:.3f} ht={half_thick:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# ENFORCE DISJOINT ASSIGNMENT (true Voronoi)
# Overlapping gates (e.g. stacked/coaxial gears) previously put the same
# vertex into multiple groups; mesh.separate then ran sequentially, so the
# first wheel separated stole shared verts from every later one (gear_yellow:
# 7707 assigned → 365 separated). Assign each vertex exclusively to the wheel
# whose normalized cylinder it sits deepest inside.
# ══════════════════════════════════════════════════════════════════════════════
pivot_params = {}
for name, pos, radius, half_thick, capture_r in centroid_list:
    ht = half_thick if half_thick > 1e-6 else radius
    pivot_params[name] = (np.array(pos),
                          max(capture_r, radius * 1.1, 1e-6),
                          max(ht, 1e-6))

vert_owner = {}   # vert index → (wheel name, score)
for name, idxs in wheel_vert_groups_raw.items():
    if not idxs or name not in pivot_params:
        continue
    P, R, HT = pivot_params[name]
    vv     = verts[np.array(idxs)]
    radial = np.hypot(vv[:, 0] - P[0], vv[:, 2] - P[2]) / R
    ax     = np.abs(vv[:, 1] - P[1]) / HT
    scores = radial**2 + ax**2
    for vi, sc in zip(idxs, scores):
        prev = vert_owner.get(vi)
        if prev is None or sc < prev[1]:
            vert_owner[vi] = (name, sc)

disjoint_groups = {n: [] for n in wheel_vert_groups_raw}
for vi, (n, _) in vert_owner.items():
    disjoint_groups[n].append(vi)

for nm in wheel_vert_groups_raw:
    before, after = len(wheel_vert_groups_raw[nm]), len(disjoint_groups[nm])
    if before != after:
        print(f"  {nm}: {before} → {after} verts after overlap resolution")
wheel_vert_groups_raw = disjoint_groups

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
