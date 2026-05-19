import sys
sys.path.insert(0, '/tmp/blender_packages')

import bpy, json, math
import numpy as np
from pathlib import Path

glb_path     = sys.argv[sys.argv.index('--') + 1]
segments_dir = sys.argv[sys.argv.index('--') + 2]
output_path  = sys.argv[sys.argv.index('--') + 3]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
mesh_obj     = mesh_objects[0]

with open(f'{segments_dir}/segments.json') as f:
    segments = json.load(f)

verts = np.array([
    list(mesh_obj.matrix_world @ v.co)
    for v in mesh_obj.data.vertices
])

bmin = verts.min(axis=0)
bmax = verts.max(axis=0)
size = bmax - bmin

wheel_vert_indices = set()

def project_front(v):
    u  = (v[0] - bmin[0]) / size[0]
    vv = 1.0 - (v[2] - bmin[2]) / size[2]
    return u, vv

def project_right(v):
    u  = (v[1] - bmin[1]) / size[1]
    vv = 1.0 - (v[2] - bmin[2]) / size[2]
    return u, vv

def project_left(v):
    u  = 1.0 - (v[1] - bmin[1]) / size[1]
    vv = 1.0 - (v[2] - bmin[2]) / size[2]
    return u, vv

def project_top(v):
    u  = (v[0] - bmin[0]) / size[0]
    vv = (v[1] - bmin[1]) / size[1]
    return u, vv

projections = {
    'front': project_front,
    'right': project_right,
    'left':  project_left,
    'top':   project_top,
}

for view, project in projections.items():
    if view not in segments:
        continue

    wheel_masks = [s for s in segments[view] if s['label'] == 'wheel']
    print(f"{view}: {len(wheel_masks)} wheel masks")

    for seg in wheel_masks:
        bbox   = seg['bbox']
        radius = seg.get('radius', 0)
        
        # Use circle center and radius directly — no expansion
        cx_bb = (bbox[0] + bbox[2] / 2) / 512
        cy_bb = (bbox[1] + bbox[3] / 2) / 512
        r_bb  = (radius * 1.1) / 512  # just 10% expansion

        for vi, v in enumerate(verts):
            u, vv = project(v)
            # Use circular test instead of bbox
            dist = ((u - cx_bb)**2 + (vv - cy_bb)**2) ** 0.5
            if dist < r_bb:
                wheel_vert_indices.add(vi)

print(f"\nTotal wheel vertices identified: {len(wheel_vert_indices)}")
print(f"Total vertices: {len(verts)}")

# Filter to outermost X vertices only — removes wheel well geometry
wheel_idx_list  = list(wheel_vert_indices)
wheel_verts_all = verts[wheel_idx_list]

x_threshold_right = bmin[0] + size[0] * 0.65  # rightmost 35% only
x_threshold_left  = bmin[0] + size[0] * 0.35  # leftmost 35% only

filtered_indices = [
    wheel_idx_list[i] for i, v in enumerate(wheel_verts_all)
    if v[0] > x_threshold_right or v[0] < x_threshold_left
]

print(f"After X filtering: {len(filtered_indices)} wheel vertices (was {len(wheel_idx_list)})")

wheel_vert_indices = set(filtered_indices)
wheel_verts        = verts[list(wheel_vert_indices)]

if len(wheel_verts) < 4:
    print("Not enough wheel vertices found")
    sys.exit(1)


def numpy_kmeans(points, k=4, iterations=100):
    idx = np.array([
        np.argmin(points[:, 0]),
        np.argmax(points[:, 0]),
        np.argmin(points[:, 1]),
        np.argmax(points[:, 1]),
    ])
    centroids = points[idx].astype(float)
    for _ in range(iterations):
        diffs  = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists  = np.linalg.norm(diffs, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([
            points[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels


def assign_labels(points, centroids):
    diffs = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.argmin(dists, axis=1)


centroids, labels = numpy_kmeans(wheel_verts[:, [0, 1]], k=4)

print("\nWheel clusters:")
for i in range(4):
    cluster_verts = wheel_verts[labels == i]
    if len(cluster_verts) == 0:
        print(f"  Wheel {i}: empty cluster")
        continue
    center = cluster_verts.mean(axis=0)
    print(f"  Wheel {i}: {len(cluster_verts)} verts, "
          f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")

all_vert_indices = list(wheel_vert_indices)
wheel_vert_array = verts[all_vert_indices]
cluster_labels   = assign_labels(wheel_vert_array[:, [0, 1]], centroids)

bpy.context.view_layer.objects.active = mesh_obj
mesh_obj.select_set(True)

for i in range(4):
    vg      = mesh_obj.vertex_groups.new(name=f'wheel_{i}')
    mask    = np.where(cluster_labels == i)[0]
    vg_idxs = [all_vert_indices[j] for j in mask]
    vg.add(vg_idxs, 1.0, 'REPLACE')

for i in range(4):
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.vertex_group_set_active(group=f'wheel_{i}')
    bpy.ops.object.vertex_group_select()
    bpy.ops.mesh.separate(type='SELECTED')
    bpy.ops.object.mode_set(mode='OBJECT')

mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
print(f"\nAfter separation: {len(mesh_objects)} objects")
for o in mesh_objects:
    print(f"  {o.name}: {len(o.data.vertices)} verts at {o.location}")

bpy.ops.export_scene.gltf(filepath=output_path, export_format='GLB')
print(f"\nExported: {output_path}")
