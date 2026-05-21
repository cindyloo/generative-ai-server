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
size      = bmax - bmin
mesh_size = np.linalg.norm(size)

print(f"\nMesh bounds:")
print(f"  X: {bmin[0]:.3f} to {bmax[0]:.3f}")
print(f"  Y: {bmin[1]:.3f} to {bmax[1]:.3f}")
print(f"  Z: {bmin[2]:.3f} to {bmax[2]:.3f}")

# Load tire vertices
with open(tire_verts_path) as f:
    tire_vert_indices = list(json.load(f))

print(f"Tire vertices from palette: {len(tire_vert_indices)}")
tire_verts_np = verts[tire_vert_indices]

# Split by X (left/right) then by Z (front/rear)
left_verts  = tire_verts_np[tire_verts_np[:, 0] < 0]
right_verts = tire_verts_np[tire_verts_np[:, 0] >= 0]

print(f"Left tire verts: {len(left_verts)}")
print(f"Right tire verts: {len(right_verts)}")

def split_front_rear(side_verts):
    """Split by Y axis (front/back depth in Blender world space)."""
    if len(side_verts) == 0:
        return np.zeros((0,3)), np.zeros((0,3))
    y_median = np.median(side_verts[:, 1])  # ← index 1 (Y), not 2 (Z)
    front    = side_verts[side_verts[:, 1] < y_median]
    rear     = side_verts[side_verts[:, 1] >= y_median]
    return front, rear

lf, lr = split_front_rear(left_verts)
rf, rr = split_front_rear(right_verts)

true_centroids = {
    'wheel_rl': lf.mean(axis=0) if len(lf) > 0 else None,
    'wheel_rr': lr.mean(axis=0) if len(lr) > 0 else None,
    'wheel_fl': rf.mean(axis=0) if len(rf) > 0 else None,
    'wheel_fr': rr.mean(axis=0) if len(rr) > 0 else None,
}

print("\nCentroids from tire vertices:")
for name, c in true_centroids.items():
    if c is not None:
        print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

# Mirror any missing centroids
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

for name in list(true_centroids.keys()):
    if true_centroids.get(name) is None:
        mirror_name = mirror_wheel_name(name)
        if mirror_name and true_centroids.get(mirror_name) is not None:
            src = true_centroids[mirror_name]
            true_centroids[name] = np.array([-src[0], src[1], src[2]])
            print(f"  Mirrored {name} from {mirror_name}")

# Assign all tire vertices to nearest centroid
centroid_list  = [(name, c) for name, c in true_centroids.items() if c is not None]
centroid_array = np.array([c for _, c in centroid_list])

dists   = np.linalg.norm(
    tire_verts_np[:, np.newaxis, :] - centroid_array[np.newaxis, :, :],
    axis=2
)
nearest = np.argmin(dists, axis=1)

wheel_vert_groups = []
for ci, (name, centroid) in enumerate(centroid_list):
    mask        = np.where(nearest == ci)[0]
    wheel_verts = [tire_vert_indices[i] for i in mask]
    print(f"  {name}: {len(wheel_verts)} tire verts")
    wheel_vert_groups.append((name, wheel_verts))

print(f"Total assigned: {sum(len(w[1]) for w in wheel_vert_groups)} of {len(tire_vert_indices)}")

# Separate and export
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
    
# After separation, clean stray disconnected vertices from each wheel object.
# Stray black chassis fragments get assigned to wheel groups by proximity
# but are disconnected from the main tire geometry — remove them.
all_meshes = [o for o in bpy.data.objects if o.type == 'MESH']
body = max(all_meshes, key=lambda o: len(o.data.vertices))

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
# Save Blender-space centroids to classify_json so animatesam.py can use
# them directly as pivot points — no coordinate conversion needed since
# these were computed from matrix_world @ v.co (already Blender world space)
blender_centroids = {
    name: c.tolist() if c is not None else None
    for name, c in true_centroids.items()
}
with open(classify_json, 'r') as f:
    cdata = json.load(f)
cdata['blender_wheel_centroids'] = blender_centroids
with open(classify_json, 'w') as f:
    json.dump(cdata, f, indent=2)
print(f"\nBlender-space centroids saved to classify_json:")
for name, c in blender_centroids.items():
    if c:
        print(f"  {name}: ({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")

bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
print(f"\nExported: {output_path}")
