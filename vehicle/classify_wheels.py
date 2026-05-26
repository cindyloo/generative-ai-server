import os, sys
sys.path.insert(0, '/tmp/blender_packages')

import bpy, json
import numpy as np

glb_path        = os.path.abspath(sys.argv[sys.argv.index('--') + 1])
output_path     = os.path.abspath(sys.argv[sys.argv.index('--') + 2])
classify_json   = sys.argv[sys.argv.index('--') + 3]
tire_verts_path = sys.argv[sys.argv.index('--') + 4]

print("glb_path:", glb_path)
print("output_path:", output_path)

with open(classify_json) as f:
    classify_data = json.load(f)

# Load mesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
mesh_obj     = mesh_objects[0]

verts = np.array([list(mesh_obj.matrix_world @ v.co)
                  for v in mesh_obj.data.vertices])
bmin      = verts.min(axis=0)
bmax      = verts.max(axis=0)
brange    = bmax - bmin
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
            pos    = v['centroid']
            radius = v.get('radius', 0.2)
        else:
            pos    = v
            radius = 0.2
        # trimesh: Y=height, Z=left-right
        # Blender: Y=left-right, Z=height  →  swap Y and Z
        true_centroids[name] = {
            'centroid': np.array([pos[0], pos[2], pos[1]]),
            'radius':   radius,
        }

    print(f"\nLoaded centroids from find_tire_verts: {list(true_centroids.keys())}")
    for name, d in true_centroids.items():
        c = d['centroid']
        print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) radius={d['radius']:.3f}")
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
# ASSIGN VERTICES TO NEAREST CENTROID BY RADIUS
# ══════════════════════════════════════════════════════════════════════════════
centroid_list = [(name, d['centroid'], d['radius'])
                 for name, d in true_centroids.items()
                 if d['centroid'] is not None]

print("\nCentroids from tire vertices:")
for name, centroid, radius in centroid_list:
    print(f"  {name}: ({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f})")

wheel_vert_groups = []
for name, centroid, radius in centroid_list:
    dists       = np.linalg.norm(verts - centroid, axis=1)
    wheel_verts = list(np.where(dists <= radius)[0])
    print(f"  {name}: radius={radius:.3f}, {len(wheel_verts)} verts")
    wheel_vert_groups.append((name, wheel_verts))

print(f"Total assigned: {sum(len(w[1]) for w in wheel_vert_groups)} of {len(verts)}")

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
        continue
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.vertex_group_set_active(group=name)
    bpy.ops.object.vertex_group_select()
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

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

# ══════════════════════════════════════════════════════════════════════════════
# SAVE BLENDER-SPACE CENTROIDS
# ══════════════════════════════════════════════════════════════════════════════
blender_centroids = {
    name: centroid.tolist()
    for name, centroid, radius in centroid_list
}
with open(classify_json, 'r') as f:
    cdata = json.load(f)
cdata['blender_wheel_centroids'] = blender_centroids
with open(classify_json, 'w') as f:
    json.dump(cdata, f, indent=2)

print(f"\nBlender-space centroids saved to classify_json:")
for name, c in blender_centroids.items():
    print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
print(f"\nExported: {output_path}")
