import os, sys
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

print(f"\nMesh bounds:")
print(f"  X: {bmin[0]:.3f} to {bmax[0]:.3f}")
print(f"  Y: {bmin[1]:.3f} to {bmax[1]:.3f}")
print(f"  Z: {bmin[2]:.3f} to {bmax[2]:.3f}")

# Load tire vertices (kept for backward compat)
with open(tire_verts_path) as f:
    raw = json.load(f)
if isinstance(raw, list):
    tire_vert_indices = [i for i in raw if isinstance(i, int)]
else:
    tire_vert_indices = []
print(f"Tire vertices from palette: {len(tire_vert_indices)}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD CENTROIDS FROM find_tire_verts OUTPUT
# Swap Y and Z because trimesh and Blender have opposite Y/Z conventions
# ══════════════════════════════════════════════════════════════════════════════
centroids_path = tire_verts_path.replace('.json', '_centroids.json')
true_centroids = {}

if os.path.exists(centroids_path):
    with open(centroids_path) as f:
        saved_centroids = json.load(f)

    for name, v in saved_centroids.items():
        if v is None:
            continue
        if isinstance(v, dict):
            pos               = v['centroid']
            radius            = v.get('radius', 0.2)
            candidate_indices = v.get('candidate_indices', None)
        else:
            pos               = v
            radius            = 0.2
            candidate_indices = None
        # trimesh: Y=height, Z=left-right
        # Blender: Y=left-right, Z=height  →  swap Y and Z
        true_centroids[name] = {
            'centroid':          np.array([pos[0], pos[2], pos[1]]),
            'radius':            radius,
            'candidate_indices': set(candidate_indices) if candidate_indices else None,
        }

    print(f"\nLoaded centroids from find_tire_verts: {list(true_centroids.keys())}")
    for name, d in true_centroids.items():
        c     = d['centroid']
        n_cand = len(d['candidate_indices']) if d['candidate_indices'] else 0
        print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) radius={d['radius']:.3f} "
              f"candidates={n_cand}")
else:
    print(f"WARNING: centroids file not found at {centroids_path}")

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC SPLIT FALLBACK — only if centroids not loaded
# ══════════════════════════════════════════════════════════════════════════════
if not true_centroids:
    print("WARNING: no centroids loaded, falling back to geometric split")
    tire_verts_np = verts[tire_vert_indices] if tire_vert_indices else np.empty((0, 3))
    num_wheels = len([k for k in classify_data.get('wheel_centroids', {}).keys()
                      if k.startswith('wheel_')])
    print(f"Vehicle has {num_wheels} wheels")
    if num_wheels == 4:
        left_verts  = tire_verts_np[tire_verts_np[:, 0] < 0]
        right_verts = tire_verts_np[tire_verts_np[:, 0] >= 0]
        def split_front_rear(side_verts):
            if len(side_verts) == 0:
                return np.zeros((0, 3)), np.zeros((0, 3))
            median = np.median(side_verts[:, 1])
            return side_verts[side_verts[:, 1] < median], side_verts[side_verts[:, 1] >= median]
        lf, lr = split_front_rear(left_verts)
        rf, rr = split_front_rear(right_verts)
        for name, cluster in [('wheel_fl', lf), ('wheel_fr', rf),
                               ('wheel_rl', lr), ('wheel_rr', rr)]:
            true_centroids[name] = {
                'centroid': cluster.mean(axis=0) if len(cluster) > 0 else None,
                'radius':   0.2,
            }
    elif num_wheels == 2:
        median = np.median(tire_verts_np[:, 1])
        lf = tire_verts_np[tire_verts_np[:, 1] < median]
        lr = tire_verts_np[tire_verts_np[:, 1] >= median]
        for name, cluster in [('wheel_fl', lf), ('wheel_rl', lr)]:
            true_centroids[name] = {
                'centroid': cluster.mean(axis=0) if len(cluster) > 0 else None,
                'radius':   0.2,
            }
    else:
        raise RuntimeError(f"Unexpected wheel count: {num_wheels} and no centroids loaded")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SAM2 MASKS AND IMAGE BOUNDS
# ══════════════════════════════════════════════════════════════════════════════
sam2_masks   = {}   # name → binary mask array (H x W bool)
img_bounds   = {}   # from wheel_centers.json _image_bounds
obj_top = obj_left = 0
obj_h = obj_w = 1

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
        print(f"\nImage bounds: obj_top={obj_top} obj_bottom={obj_bottom} "
              f"obj_left={obj_left} obj_right={obj_right} "
              f"obj_h={obj_h} obj_w={obj_w}")

    for name in true_centroids:
        mask_path = os.path.join(mask_dir, f"{name}.png")
        if os.path.exists(mask_path):
            bpy_img  = bpy.data.images.load(os.path.abspath(mask_path))
            px       = np.array(bpy_img.pixels[:])
            mask_arr = (px[::4] > 0.5).reshape(bpy_img.size[1], bpy_img.size[0])
            mask_arr = mask_arr[::-1, :]   # flip: bpy loads bottom-up (OpenGL), image is top-down
            bpy.data.images.remove(bpy_img)
            sam2_masks[name] = mask_arr
            print(f"  Loaded SAM2 mask '{name}': {mask_arr.shape} ({mask_arr.sum()} pixels)")
            print(f"  {name}: mask white pixel Y range: "
                  f"{np.where(mask_arr)[0].min()}..{np.where(mask_arr)[0].max()}")
        else:
            print(f"  No SAM2 mask found for '{name}' at {mask_path}")

# ══════════════════════════════════════════════════════════════════════════════
# ASSIGN VERTICES TO WHEELS
# Primary: SAM2 mask projection (excludes fork/frame pixels exactly)
# Fallback: radius sphere from centroid
# ══════════════════════════════════════════════════════════════════════════════
centroid_list = [(name, d['centroid'], d['radius'], d.get('candidate_indices'))
                 for name, d in true_centroids.items()
                 if d['centroid'] is not None]

print("\nCentroids from tire vertices:")
for name, centroid, radius, cand_idx in centroid_list:
    n_cand = len(cand_idx) if cand_idx else 0
    print(f"  {name}: ({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}) "
          f"radius={radius:.3f} candidates={n_cand}")

# ── Axis assignment ───────────────────────────────────────────────────────────
# Vehicles (side view): fixed by Meshy convention
#   mesh X = front-to-rear, mesh Y = height, mesh Z = left-right/depth
# Mechanical (front view): detect from geometry
all_parts   = classify_data.get('joint_hints', [])
is_vehicle  = any(j.get('body_part') == 'wheel' for j in all_parts)

if is_vehicle:
    # Blender imports GLB with Y/Z swapped vs trimesh:
    #   Blender X = front-to-rear (same as trimesh X)
    #   Blender Z = height        (trimesh Y, swapped on import)
    #   Blender Y = left-right    (trimesh Z, swapped on import)
    wide_axis = 0   # Blender X = front-to-rear
    tall_axis = 2   # Blender Z = height
    thin_axis = 1   # Blender Y = left-right/depth
    print(f"\nAxis assignment: vehicle (fixed Blender) wide(FR)=0 tall(UP)=2 thin(LR)=1")
else:
    thin_axis = int(np.argmin(brange))
    remaining = sorted([i for i in range(3) if i != thin_axis],
                       key=lambda i: brange[i], reverse=True)
    wide_axis = remaining[0]
    tall_axis = remaining[1]
    print(f"\nAxis detection: mechanical (geometry) thin={thin_axis} wide={wide_axis} tall={tall_axis}")

print(f"  ranges: thin={brange[thin_axis]:.3f} wide={brange[wide_axis]:.3f} tall={brange[tall_axis]:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# OPTIONAL COLOR FILTER
# Build a boolean mask of wheel-colored verts using texture + wheel_colors_rgb.
# Applied to each radius sphere to remove frame/body geometry.
# Skipped gracefully if colors unavailable.
# ══════════════════════════════════════════════════════════════════════════════
color_mask      = None
color_filter_on = False

try:
    import struct
    wheel_colors_data = classify_data.get('wheel_colors_rgb', [])
    texture_path      = glb_path.replace('_mesh.glb', '_texture.png')
    # Also try results dir pattern
    import re
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
            nc    = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[acc['type']]
            fmt   = {5126:'f', 5123:'H', 5125:'I'}[acc['componentType']]
            data  = struct.unpack_from(f'<{count*nc}{fmt}', binary, start)
            return np.array(data).reshape(count, nc) if nc > 1 else np.array(data)

        prim   = j['meshes'][0]['primitives'][0]
        uv_idx = prim['attributes'].get('TEXCOORD_0')

        if uv_idx is not None:
            uvs     = read_accessor(uv_idx)
            # Load texture via bpy (PIL not available in Blender Python)
            bpy_img = bpy.data.images.load(os.path.abspath(texture_path))
            iw, ih  = bpy_img.size
            # bpy pixels: flat RGBA, bottom-up
            px_flat  = np.array(bpy_img.pixels[:]).reshape(ih, iw, 4)
            img_arr  = (px_flat[::-1, :, :3] * 255).astype(np.uint8)  # flip Y, RGB uint8
            bpy.data.images.remove(bpy_img)
            px_u    = (uvs[:, 0] * iw).astype(int) % iw
            px_v    = (uvs[:, 1] * ih).astype(int) % ih
            vert_colors = img_arr[px_v, px_u]

            if isinstance(wheel_colors_data[0], dict):
                wheel_colors = np.array([[int(c['color'][i]*255) for i in range(3)]
                                          for c in wheel_colors_data])
            else:
                wheel_colors = np.array([[int(c[i]*255) for i in range(3)]
                                          for c in wheel_colors_data])

            color_mask = np.zeros(len(verts), dtype=bool)
            for wc in wheel_colors:
                dists_c     = np.sqrt(np.sum((vert_colors.astype(int) - wc)**2, axis=1))
                color_mask |= (dists_c < 60)

            if color_mask.sum() > 100:
                color_filter_on = True
                print(f"\nColor filter: {color_mask.sum()} / {len(verts)} verts "
                      f"({color_mask.sum()/len(verts)*100:.1f}%) "
                      f"← colors: {wheel_colors.tolist()}")
            else:
                print(f"\nColor filter: too aggressive ({color_mask.sum()} verts) — skipped")
                color_mask = None
    else:
        reason = "no wheel_colors_rgb" if not wheel_colors_data else "texture not found"
        print(f"\nColor filter: skipped ({reason})")

except Exception as e:
    print(f"\nColor filter: failed ({e}) — skipped")

# ── Assign vertices ──────────────────────────────────────────────────────────
# Geometry (radius sphere) + optional color filter.

print(f"\nAssignment strategy: radius sphere" +
      (" + color filter" if color_filter_on else " (no color filter)"))

wheel_vert_groups = []
for name, centroid, radius, cand_idx in centroid_list:
    print(f"\n[{name}]")

    # ── Step 1: Geometric radius sphere ───────────────────────────────────────
    dists     = np.linalg.norm(verts - centroid, axis=1)
    in_sphere = np.where(dists <= radius)[0]
    print(f"  Radius sphere:        {len(in_sphere):6d} verts  (r={radius:.3f})")

    # ── Step 2: Color filter ──────────────────────────────────────────────────
    if color_filter_on and color_mask is not None:
        filtered = in_sphere[color_mask[in_sphere]]
        pct      = len(filtered) / len(in_sphere) * 100 if len(in_sphere) > 0 else 0
        print(f"  After color filter:   {len(filtered):6d} verts  ({pct:.1f}% kept)")
        if len(filtered) > 20:
            in_sphere = filtered
        else:
            print(f"  Color filter too aggressive — reverting to full sphere")
    else:
        print(f"  No color filter — using full sphere")

    wheel_verts = list(in_sphere)
    print(f"  Final:                {len(wheel_verts):6d} verts assigned to {name}")
    wheel_vert_groups.append((name, wheel_verts))

# ══════════════════════════════════════════════════════════════════════════════
# SAVE BLENDER-SPACE CENTROIDS EARLY
# Write before separation so animatesam.py always has them even if
# separation partially fails on one of the parts.
# ══════════════════════════════════════════════════════════════════════════════
blender_centroids = {
    name: centroid.tolist()
    for name, centroid, radius, cand_idx in centroid_list
}
with open(classify_json, 'r') as f:
    cdata = json.load(f)
cdata['blender_wheel_centroids'] = blender_centroids
with open(classify_json, 'w') as f:
    json.dump(cdata, f, indent=2)

print(f"\nBlender-space centroids saved to classify_json:")
for name, c in blender_centroids.items():
    print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# SEPARATE MESH OBJECTS
# ══════════════════════════════════════════════════════════════════════════════
bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)

for name, vert_indices in wheel_vert_groups:
    if not vert_indices:
        print(f"  {name}: empty, skipping")
        continue
    vg = mesh_obj.vertex_groups.new(name=name)
    vg.add([int(v) for v in vert_indices], 1.0, 'REPLACE')

for name, vert_indices in wheel_vert_groups:
    if not vert_indices:
        print(f"  {name}: empty, skipping separation")
        continue

    # Re-establish mesh_obj as active and selected before each separation.
    # After mesh.separate(), Blender makes the new object active — we must
    # explicitly reset context or the next vertex_group_select() has no target.
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)

    # Verify the vertex group still exists on mesh_obj (it may have been
    # separated out in a prior iteration if verts overlap groups)
    if name not in [vg.name for vg in mesh_obj.vertex_groups]:
        print(f"  {name}: vertex group missing from mesh_obj, skipping")
        continue

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.vertex_group_set_active(group=name)
    bpy.ops.object.vertex_group_select()

    # Check something is actually selected before separating
    bpy.ops.object.mode_set(mode='OBJECT')
    selected_verts = [v for v in mesh_obj.data.vertices if v.select]
    if not selected_verts:
        print(f"  {name}: no verts selected after group select, skipping")
        continue

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  {name}: separated {len(selected_verts)} verts")

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
print(f"\nAfter separation: {len(mesh_objects)} objects")
for o in mesh_objects:
    print(f"  {o.name}: {len(o.data.vertices)} verts")

# ══════════════════════════════════════════════════════════════════════════════
# CLEAN STRAY DISCONNECTED VERTICES
# ══════════════════════════════════════════════════════════════════════════════
all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
body       = max(all_meshes, key=lambda o: len(o.data.vertices))
for obj in all_meshes:
    if obj == body:
        continue
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_loose()
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  Cleaned {obj.name}: {len(obj.data.vertices)} verts remaining")

bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
print(f"\nExported: {output_path}")
