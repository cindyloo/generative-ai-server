"""
rig.py

Local test script: GLB → infer skeleton → rigged GLB

Pipeline:
  1. (Optional) Identify object from photo using Gemini
  2. Load mesh (GLB/OBJ/FBX)
  3. Infer joints using geometric method
  4. Visualize skeleton for inspection
  5. Place bones in Blender at detected joints with named bones
  6. Auto-weight skin mesh to bones
  7. Export rigged GLB

Usage:
  # Step 1 — identify + infer, inspect skeleton viz:
  python rig.py --input model.glb --output rigged.glb --photo photo.jpg --viz-only

  # With user tag:
  python rig.py --input sofa.glb --output rigged.glb --photo sofa.jpg --tag "flying sofa" --viz-only

  # Without photo (geometric auto-detect only):
  python rig.py --input model.glb --output rigged.glb --viz-only

  # Step 2 — rig in Blender using saved skeleton JSON:
  blender --background --python rig.py -- \\
    --from-json rigged_skeleton.json \\
    --input model.glb --output rigged.glb

Requirements:
  pip install trimesh numpy scipy pillow google-genai
  export GEMINI_API_KEY=your_key  (free at aistudio.google.com)
"""

import sys

# Only use blender_packages when running inside Blender (Python 3.11)
# When running as regular Python (3.14), use system numpy
if sys.version_info[:2] == (3, 11):
    sys.path.insert(0, '/tmp/blender_packages')

import numpy as np
import os
import argparse
import json
from pathlib import Path


def infer_from_photo(photo_path: str, tag: str = None) -> dict | None:
    try:
        from seg_server import classify_with_gemini
        ext = Path(photo_path).suffix.lower()
        mime_type = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
        with open(photo_path, 'rb') as f:
            img_bytes = f.read()
        return classify_with_gemini(img_bytes, mime_type, tag)
    except Exception as e:
        print(f"  Classification unavailable: {e}")
        return None


# ── Skeleton inference ────────────────────────────────────────────────────────

def infer_skeleton_geometric(mesh_path: str, n_joints: int = None) -> tuple:
    """
    Geometric skeleton via medial axis approximation + minimum spanning tree.
    No ML required. Works best on meshes with clear articulated segments.
    """
    import trimesh
    from scipy.spatial import cKDTree
    from scipy.cluster.vq import kmeans
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path, force='mesh')

    if mesh.is_empty:
        raise ValueError("Mesh is empty or could not be loaded")

    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    n_samples         = min(10000, len(mesh.vertices) * 3)
    surface_points, _ = trimesh.sample.sample_surface(mesh, n_samples)

    tree      = cKDTree(surface_points)
    k         = min(20, len(surface_points) - 1)
    distances, _ = tree.query(surface_points, k=k)
    mean_dist = distances.mean(axis=1)
    skeleton_candidates = surface_points[mean_dist > np.percentile(mean_dist, 75)]
    print(f"Skeleton candidates: {len(skeleton_candidates)}")

    if n_joints is None:
        bounds   = mesh.bounds
        dims     = bounds[1] - bounds[0]
        aspect   = max(dims) / min(dims) if min(dims) > 0 else 1
        n_joints = max(2, min(12, int(aspect * 1.5)))
        print(f"Auto joint count: {n_joints} (aspect {aspect:.2f})")

    n_joints  = min(n_joints, len(skeleton_candidates))
    centroids, _ = kmeans(skeleton_candidates.astype(np.float64), n_joints)
    joints    = [tuple(c) for c in centroids]

    n = len(joints)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(
                    np.array(joints[i]) - np.array(joints[j])
                )

    mst       = minimum_spanning_tree(csr_matrix(dist_matrix)).toarray()
    hierarchy = [(i, j) for i in range(n) for j in range(n) if mst[i, j] > 0]

    print(f"Geometric: {len(joints)} joints, {len(hierarchy)} bones")
    return joints, hierarchy, float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))


# ── Skeleton visualization ────────────────────────────────────────────────────

def visualize_skeleton(mesh_path, joints, hierarchy, output_path,
                       labels=None, labels_raw=None):
    """
    Save skeleton JSON + GLB visualization (red spheres at joints, green bones).
    """
    import trimesh

    def joint_name(i):
        return labels[i] if labels and i < len(labels) else f"joint_{i}"

    skeleton_data = {
        'joints': [
            {
                'id': i,
                'name': joint_name(i),
                'position': list(j),
                # carry through full hint object if available
                'hint': (labels_raw[i] if labels_raw and i < len(labels_raw)
                         and isinstance(labels_raw[i], dict) else None)
            }
            for i, j in enumerate(joints)
        ],
        'bones':  [{'parent': p, 'child': c,
                    'name': f"{joint_name(p)}_to_{joint_name(c)}"}
                   for p, c in hierarchy],
    }

    json_path = output_path.replace('.glb', '_skeleton.json')
    with open(json_path, 'w') as f:
        json.dump(skeleton_data, f, indent=2)
    print(f"Skeleton JSON: {json_path}")

    mesh  = trimesh.load(mesh_path, force='mesh')
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name='mesh')

    mesh_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    sphere_r  = mesh_size * 0.02
    bone_r    = mesh_size * 0.005

    for i, joint in enumerate(joints):
        sphere = trimesh.creation.icosphere(radius=sphere_r)
        sphere.apply_translation(joint)
        sphere.visual.face_colors = [255, 50, 50, 220]
        scene.add_geometry(sphere, node_name=f'joint_{i}_{joint_name(i)}')

    for parent_idx, child_idx in hierarchy:
        p      = np.array(joints[parent_idx])
        c      = np.array(joints[child_idx])
        length = np.linalg.norm(c - p)
        if length < 1e-4:
            continue

        direction = (c - p) / length
        midpoint  = (p + c) / 2
        cylinder  = trimesh.creation.cylinder(radius=bone_r, height=length)

        z_axis   = np.array([0, 0, 1])
        rot_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rot_axis) > 1e-4:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            angle    = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(angle, rot_axis)
            )
        cylinder.apply_translation(midpoint)
        cylinder.visual.face_colors = [50, 220, 50, 180]
        scene.add_geometry(
            cylinder,
            node_name=f'bone_{joint_name(parent_idx)}_to_{joint_name(child_idx)}'
        )

    viz_path = output_path.replace('.glb', '_skeleton_viz.glb')
    scene.export(viz_path)
    print(f"Skeleton viz:  {viz_path}")
    print(f"Open in https://gltf.report or Blender to inspect placement")
    return json_path

def extract_joint_names(joint_hints: list) -> list[str]:
    """Handle both old format (list of strings) and new format (list of objects)."""
    if not joint_hints:
        return []
    if isinstance(joint_hints[0], str):
        return joint_hints  # backwards compat
    return [j['name'] for j in joint_hints]

def create_animations_from_hints(armature_obj, joint_hints: list):
    if not joint_hints:
        return

    try:
        import bpy
    except ImportError:
        raise RuntimeError("create_animations_from_hints must run inside Blender")

    # ✓ SET ACTIVE FIRST
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    
    # Now safe to access pose.bones
    pose_bones = {b.name: b for b in armature_obj.pose.bones}
    
    clips = {}
    for hint in joint_hints:
        if not isinstance(hint, dict):
            continue
        for anim in hint.get('animations', []):
            clips.setdefault(anim['clip'], []).append((hint['name'], anim))

    if not clips:
        print("  No animations found in hints")
        return

    axis_map   = {'x': 0, 'y': 1, 'z': 2}
    
    armature_obj.animation_data_create()
    bpy.ops.object.mode_set(mode='POSE')

    for bone in armature_obj.pose.bones:
        bone.rotation_mode = 'XYZ'


    # Build set of bone names that have explicit location animations
    # so we can decide which location fcurves to keep
    bones_with_location_anim = set()
    for hint in joint_hints:
        if not isinstance(hint, dict):
            continue
        for anim in hint.get('animations', []):
            if anim.get('property') in ('location', 'position'):
                bones_with_location_anim.add(hint['name'])

    created_actions = []
    for clip_name, entries in clips.items():

        # Skip if this clip already exists — prevents duplicate walk clips
        if clip_name in [a.name for a in bpy.data.actions]:
            print(f"  Clip '{clip_name}' already exists, skipping")
            continue

        action = bpy.data.actions.new(name=clip_name)
        armature_obj.animation_data.action = action

        for bone_name, anim in entries:
            bone = pose_bones.get(bone_name)
            if not bone:
                print(f"  Bone '{bone_name}' not found, skipping")
                continue

            prop = anim['property']
            if prop == 'position':
                prop = 'location'

            if prop == 'rotation_quaternion':
                print(f"  Skipping quaternion keyframe on '{bone_name}', euler only")
                continue

            if not hasattr(bone, prop):
                print(f"  Unknown property '{prop}' on '{bone_name}', skipping")
                continue

            coord_remap = {
                'x': (0,  1.0),  # X → X, unchanged
                'y': (2,  1.0),  # Y → Z (Gemini up = Blender Z)
                'z': (1, -1.0),  # Z → -Y (Gemini depth = Blender -Y)
            }
            axis_str = str(anim['axis']).lower()
            blender_axis, sign = coord_remap.get(axis_str, (0, 1.0))
            for frame, value in anim['keyframes']:
                getattr(bone, prop)[blender_axis] = value * sign
                bone.keyframe_insert(prop, index=blender_axis, frame=frame)

        # ── FCurve cleanup ────────────────────────────────────────
        # Remove scale curves entirely — never needed
        # Remove location curves unless Gemini explicitly animated them
        # Keep only rotation curves by default
        for fc in list(action.fcurves):
            if 'scale' in fc.data_path:
                action.fcurves.remove(fc)
            elif 'location' in fc.data_path:
                # Extract bone name from data_path e.g. 'pose.bones["root"].location'
                bone_name_in_path = ''
                if '"' in fc.data_path:
                    bone_name_in_path = fc.data_path.split('"')[1]
                if bone_name_in_path not in bones_with_location_anim:
                    action.fcurves.remove(fc)

        print(f"  Created clip '{clip_name}' with {len(entries)} animated bones")
        created_actions.append((clip_name, action))

    armature_obj.animation_data.action = None  # unlink before pushing to NLA
    for clip_name, action in created_actions:
        track      = armature_obj.animation_data.nla_tracks.new()
        track.name = clip_name
        track.strips.new(clip_name, 1, action)

    bpy.ops.object.mode_set(mode='OBJECT')
# ── Skinning helpers ──────────────────────────────────────────────────────────

def point_to_segment_distance(points, seg_start, seg_end):
    """Distance from each point to nearest point on a line segment."""
    import numpy as np
    seg      = seg_end - seg_start
    seg_len2 = np.dot(seg, seg)
    if seg_len2 < 1e-10:
        return np.linalg.norm(points - seg_start, axis=1)
    t        = np.clip(
        np.einsum('ij,j->i', points - seg_start, seg) / seg_len2,
        0.0, 1.0
    )
    closest  = seg_start + t[:, np.newaxis] * seg
    return np.linalg.norm(points - closest, axis=1)

def build_segment_weights(mesh_obj, armature_obj, skeleton_joints_data):
    """
    Assign vertices with smooth, distance-based weighting.
    Avoids the hard edges of single-bone assignment.
    """
    import numpy as np
    import bpy

    bone_segments   = []
    bone_names_list = []
    for b in armature_obj.data.bones:
        head = np.array(armature_obj.matrix_world @ b.head_local)
        tail = np.array(armature_obj.matrix_world @ b.tail_local)
        bone_segments.append((head, tail))
        bone_names_list.append(b.name)

    for bname in bone_names_list:
        mesh_obj.vertex_groups.new(name=bname)

    mat         = np.array(mesh_obj.matrix_world)
    verts_local = np.array([v.co for v in mesh_obj.data.vertices])
    ones        = np.ones((len(verts_local), 1))
    verts       = (mat @ np.hstack([verts_local, ones]).T).T[:, :3]

    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    mesh_size = np.linalg.norm(bmax - bmin)

    seg_dists = np.column_stack([
        point_to_segment_distance(verts, head, tail)
        for head, tail in bone_segments
    ])

    # ── Build set of non-deforming bone indices from skeleton_joints_data ────
    # deforms_mesh=false means the bone moves rigidly — no smooth blending.
    # This is read directly from the hint Claude outputs per joint.
    non_deforming_indices = set()
    if skeleton_joints_data:
        for joint in skeleton_joints_data:
            hint = joint.get('hint') or {}
            if not hint.get('deforms_mesh', True):
                name = joint.get('name', '')
                idx  = next((i for i, n in enumerate(bone_names_list)
                             if n == name), None)
                if idx is not None:
                    non_deforming_indices.add(idx)
                    print(f"  Non-deforming bone: '{name}' (deforms_mesh=false)")

    # ── Find terminal bones ───────────────────────────────────────────────────
    all_parent_names = set()
    for b in armature_obj.data.bones:
        if b.parent:
            all_parent_names.add(b.parent.name)

    terminal_bone_indices = [
        i for i, name in enumerate(bone_names_list)
        if name not in all_parent_names
    ]

    # After computing seg_dists, before terminal locks:

    # ── Pre-partition vertices into spine vs limb regions ──────────────────────
    # Spine bones occupy the vertical center column of the mesh.
    # Limb bones occupy the outer regions.
    # We separate them by X distance from center — limb vertices are far from X=0,
    # spine vertices are near X=0.

    spine_bone_indices = set()
    limb_bone_indices  = set()

    for bi, bname in enumerate(bone_names_list):
        bname_lower = bname.lower()
        if any(p in bname_lower for p in
               ['root', 'pelvis', 'spine', 'neck', 'head']):
            spine_bone_indices.add(bi)
        elif any(p in bname_lower for p in
                 ['elbow', 'hand', 'hip', 'knee', 'foot',
                'wing_mid', 'wing_tip']):
            limb_bone_indices.add(bi)

    # For each vertex, determine if it's in a limb region by finding
    # its nearest limb bone and nearest spine bone, then comparing distances
    if spine_bone_indices and limb_bone_indices:
        spine_dists = seg_dists[:, list(spine_bone_indices)].min(axis=1)
        limb_dists  = seg_dists[:, list(limb_bone_indices)].min(axis=1)

        # Vertices closer to a limb bone → zero out spine bone influences
        # Vertices closer to a spine bone → zero out limb bone influences
        is_limb_vert  = limb_dists < spine_dists
        is_spine_vert = ~is_limb_vert

        # Zero out cross-region influences
        for bi in spine_bone_indices:
            seg_dists[is_limb_vert, bi] = np.inf
        for bi in limb_bone_indices:
            seg_dists[is_spine_vert, bi] = np.inf

        print(f"  Partitioned: {is_limb_vert.sum()} limb verts, "
              f"{is_spine_vert.sum()} spine verts")

    # ── Hard-lock vertices near terminal bones (feet, hands) ─────────────────
    # Skip non-deforming bones here — they're handled by the rigid lock below.
    for ti in terminal_bone_indices:
        if ti in non_deforming_indices:
            continue  # handled by rigid lock below

        head, tail      = bone_segments[ti]
        bone_center     = (head + tail) / 2
        bone_length     = np.linalg.norm(tail - head)
        terminal_radius = bone_length * 0.6
        terminal_radius = min(terminal_radius, mesh_size * 0.10)
        dists_to_bone   = np.linalg.norm(verts - bone_center, axis=1)
        terminal_mask   = dists_to_bone < terminal_radius
        if terminal_mask.any():
            seg_dists[terminal_mask, :]  = np.inf  # Clear all other influences
            seg_dists[terminal_mask, ti] = 0.0     # Hard-assign only to this terminal
            print(f"  Locked {terminal_mask.sum()} vertices to terminal bone: {bone_names_list[ti]}")

    # ✅ Build smooth Gaussian-based weights

    # ── Rigid lock for non-deforming bones ────────────────────────────────────
    # All vertices whose nearest bone is a non-deforming bone get locked to it
    # with 100% weight — no blending. This makes the head, pelvis, etc. move
    # as solid rigid units rather than deforming.
    if non_deforming_indices:
        # Compute nearest bone for all vertices using current seg_dists
        nearest_bone = np.argmin(seg_dists, axis=1)
        for bi in non_deforming_indices:
            rigid_mask = nearest_bone == bi
            if rigid_mask.any():
                seg_dists[rigid_mask, :]  = np.inf
                seg_dists[rigid_mask, bi] = 0.0
                print(f"  Rigid lock: {rigid_mask.sum()} vertices → "
                      f"'{bone_names_list[bi]}' (deforms_mesh=false)")

    # ── Smooth Gaussian weights for remaining vertices ────────────────────────
    smooth_weights = np.zeros((len(verts), len(bone_names_list)))

    for bi in range(len(bone_names_list)):
        head, tail = bone_segments[bi]
        bone_len   = np.linalg.norm(tail - head)
        sigma      = max(bone_len * 0.5, mesh_size * 0.02)
        distances  = seg_dists[:, bi]
        smooth_weights[:, bi] = np.exp(-(distances ** 2) / (2 * sigma ** 2))


# ── Island coherence lock ─────────────────────────────────────────────────
    # Find disconnected mesh islands and lock small ones to their dominant bone.
    # Meshy meshes have 600-800 separate islands (non-manifold surface patches).
    # Without this, stray feather/detail fragments get independent weights and
    # fly apart during animation. Small islands must move as a coherent unit.
    import bmesh as _bmesh
    bm = _bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()

    island_map  = {}   # vertex_index → island_id
    island_id   = 0
    visited     = set()
    for v in bm.verts:
        if v.index not in visited:
            stack = [v]
            while stack:
                curr = stack.pop()
                if curr.index in visited:
                    continue
                visited.add(curr.index)
                island_map[curr.index] = island_id
                for e in curr.link_edges:
                    other = e.other_vert(curr)
                    if other.index not in visited:
                        stack.append(other)
            island_id += 1
    bm.free()

    # Count island sizes
    island_sizes = {}
    for iid in island_map.values():
        island_sizes[iid] = island_sizes.get(iid, 0) + 1

    # Threshold: islands smaller than 0.5% of total verts are "small"
    small_threshold = max(10, len(verts) * 0.05)
    small_locked    = 0

    for iid in range(island_id):
        if island_sizes.get(iid, 0) >= small_threshold:
            continue  # large island — leave Gaussian weights as-is

        # Small island — find dominant bone by average weight across its verts
        island_verts = [vi for vi, i2 in island_map.items() if i2 == iid]
        if not island_verts:
            continue
        avg_weights = smooth_weights[island_verts].mean(axis=0)
        dominant    = int(np.argmax(avg_weights))

        # Lock every vertex in this island 100% to the dominant bone
        smooth_weights[island_verts, :]         = 0.0
        smooth_weights[island_verts, dominant]  = 1.0
        small_locked += len(island_verts)

    if small_locked:
        print(f"  Island lock: {small_locked} verts in small islands "
              f"→ locked to dominant bone (threshold={int(small_threshold)} verts)")
    # ── End island coherence lock ─────────────────────────────────────────────

    # Normalize weights per vertex
    weight_sums = smooth_weights.sum(axis=1, keepdims=True)
    weight_sums[weight_sums == 0] = 1.0
    smooth_weights /= weight_sums

    # Assign to vertex groups
    for bi, bname in enumerate(bone_names_list):
        for vi, weight in enumerate(smooth_weights[:, bi]):
            if weight > 0.01:  # Skip negligible weights
                mesh_obj.vertex_groups[bname].add([vi], weight, 'REPLACE')

    mod        = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armature_obj
    print(f"  Smooth segment weights assigned: {mesh_obj.name}")
    
    
    
def validate_bone_mesh_fit(mesh_obj, armature_obj):
    """
    Check if bones are actually ON/IN the mesh using Blender only.
    """
    import numpy as np
    
    print(f"\n{'='*60}")
    print("BONE-MESH VALIDATION")
    print(f"{'='*60}")
    
    # Get mesh vertices in world space
    verts_local = np.array([v.co for v in mesh_obj.data.vertices])
    mat = np.array(mesh_obj.matrix_world)
    ones = np.ones((len(verts_local), 1))
    verts_world = (mat @ np.hstack([verts_local, ones]).T).T[:, :3]
    
    mesh_min = verts_world.min(axis=0)
    mesh_max = verts_world.max(axis=0)
    mesh_center = (mesh_min + mesh_max) / 2
    mesh_size = np.linalg.norm(mesh_max - mesh_min)
    
    report = {}
    bones_inside = 0
    bones_outside = 0
    bones_near = 0
    
    print(f"\nMesh info:")
    print(f"  Center: ({mesh_center[0]:.3f}, {mesh_center[1]:.3f}, {mesh_center[2]:.3f})")
    print(f"  Size: {mesh_size:.3f}")
    print(f"  Bounds: X[{mesh_min[0]:.3f}, {mesh_max[0]:.3f}] "
          f"Y[{mesh_min[1]:.3f}, {mesh_max[1]:.3f}] "
          f"Z[{mesh_min[2]:.3f}, {mesh_max[2]:.3f}]")
    
    print(f"\nBone positions:")
    for bone in armature_obj.data.bones:
        head_world = armature_obj.matrix_world @ bone.head_local
        head_arr = np.array(head_world[:3])
        
        # Check if head is inside bounding box
        inside_bbox = all([
            mesh_min[i] <= head_arr[i] <= mesh_max[i]
            for i in range(3)
        ])
        
        # Distance to mesh center
        dist_to_center = np.linalg.norm(head_arr - mesh_center)
        
        # Distance to nearest vertex
        distances_to_verts = np.linalg.norm(verts_world - head_arr, axis=1)
        dist_to_nearest_vert = distances_to_verts.min()
        
        # Classify
        if inside_bbox:
            status = "✓ INSIDE"
            bones_inside += 1
        elif dist_to_nearest_vert < mesh_size * 0.1:  # Within 10% of mesh size
            status = "~ NEAR"
            bones_near += 1
        else:
            status = "✗ FAR"
            bones_outside += 1
        
        report[bone.name] = {
            'position': tuple(head_arr),
            'inside_bbox': inside_bbox,
            'dist_to_nearest_vert': float(dist_to_nearest_vert),
            'status': status
        }
        
        print(f"  {bone.name:20s}: {status:10s} pos=({head_arr[0]:7.3f}, {head_arr[1]:7.3f}, {head_arr[2]:7.3f}) "
              f"vert_dist={dist_to_nearest_vert:.4f}")
    
    print(f"\nSummary:")
    print(f"  Inside bbox: {bones_inside}")
    print(f"  Near surface: {bones_near}")
    print(f"  Far outside: {bones_outside}")
    print(f"{'='*60}\n")
    
    if bones_outside > 0:
        print(f"⚠️  WARNING: {bones_outside} bones are far outside mesh!")
        print(f"   Denormalization may have failed.\n")
    
    return report
    
def skin_mesh(mesh_objects, armature_obj, skeleton_joints_data):
    """
    Try heat weighting first, fall back to segment weighting.
    """
    import bpy

    # ── Attempt heat weighting ────────────────────────────────────
    bpy.ops.object.select_all(action='DESELECT')
    for mesh_obj in mesh_objects:
        mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    heat_succeeded = all(
        any(
            vg.name in [b.name for b in armature_obj.data.bones] and
            any(v.groups for v in mesh_obj.data.vertices)
            for vg in mesh_obj.vertex_groups
        )
        for mesh_obj in mesh_objects
    )

    if heat_succeeded:
        print("  Heat weighting succeeded")
        return True

    # ── Fall back to segment weighting ───────────────────────────
    print("  Heat weighting failed, falling back to segment weighting...")
    for mesh_obj in mesh_objects:
        mesh_obj.vertex_groups.clear()
        for mod in list(mesh_obj.modifiers):
            if mod.type == 'ARMATURE':
                mesh_obj.modifiers.remove(mod)

    for mesh_obj in mesh_objects:
        build_segment_weights(mesh_obj, armature_obj, skeleton_joints_data)

    bpy.ops.object.select_all(action='DESELECT')
    for mesh_obj in mesh_objects:
        mesh_obj.select_set(True)
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.parent_set(type='ARMATURE_NAME')

    return False


def build_armature(arm_data, joints, hierarchy, bone_names):
    import bpy, numpy as np

    for b in arm_data.edit_bones:
        arm_data.edit_bones.remove(b)

    # Deduplicate hierarchy
    seen_children = set()
    unique_hierarchy = []
    for p, c in hierarchy:
        if c not in seen_children:
            seen_children.add(c)
            unique_hierarchy.append((p, c))
    hierarchy = unique_hierarchy

    children_map = {}
    for p, c in hierarchy:
        children_map.setdefault(p, []).append(c)

    # One bone per joint
    for i, joint in enumerate(joints):
        name = bone_names[i] if bone_names and i < len(bone_names) else f'joint_{i}'
        bone = arm_data.edit_bones.new(name)
        bone.head = joints[i]

        child_ids = children_map.get(i, [])
        if child_ids:
            bone.tail = joints[child_ids[0]]
        else:
            # Leaf bone
            parent_idx = next((p for p, c in hierarchy if c == i), None)
            if parent_idx is not None:
                d = np.array(joints[i]) - np.array(joints[parent_idx])
                n = np.linalg.norm(d)
                d = d / n if n > 0 else np.array([0, 0.1, 0])
                bone.tail = tuple(np.array(joints[i]) + d * 0.05)
            else:
                bone.tail = (joints[i][0], joints[i][1] + 0.1, joints[i][2])

        _align_bone_roll(bone)
        print(f"  Bone: {name}")

    # Parent bones with use_connect = True
    for parent_idx, child_idx in hierarchy:
        pname = bone_names[parent_idx] if bone_names else f'joint_{parent_idx}'
        cname = bone_names[child_idx]  if bone_names else f'joint_{child_idx}'
        pb = arm_data.edit_bones.get(pname)
        cb = arm_data.edit_bones.get(cname)
        if pb and cb:
            cb.parent = pb
            # use_connect=True snaps child head to parent tail.
            # Only use it for straight chains (spine, leg, arm).
            # Branching joints (hip, shoulder) must be False or they
            # get snapped to the wrong position.
            branch_parts = {'hip', 'shoulder', 'wing_base'}
            is_branch = any(p in cname.lower() for p in branch_parts)
            cb.use_connect = not is_branch

    n_bones = len(arm_data.edit_bones)
    print(f"Armature: {n_bones} bones")
    return {}, hierarchy

def _align_bone_roll(bone):
    """
    Align bone roll so local X ≈ world X for all bone orientations.
    This ensures Euler X rotations produce the expected world-space movement:
      - Leg bones (pointing down): X rotation = forward/backward swing ✓
      - Spine bones (pointing up): X rotation = side lean ✓
      - Arm bones (pointing sideways): X rotation = forward/backward swing ✓
    Without this, local X can point in any direction (in this model it pointed
    straight DOWN, causing legs to spin instead of swing).
    """
    import mathutils
    from mathutils import Vector
    bone_dir = (bone.tail - bone.head).normalized()

    # Primary alignment: make local Z point toward world Y (forward/depth axis).
    # For a downward leg bone this gives local X = world X (left-right) = swing axis.
    align_vec = mathutils.Vector((0, 1, 0))

    # If bone is nearly parallel to world Y (e.g. a horizontal arm bone),
    # fall back to world Z (up) to avoid degenerate alignment.
    if abs(bone_dir.dot(align_vec)) > 0.9:
        align_vec = mathutils.Vector((0, 0, 1))

    bone.align_roll(align_vec)

#not used yet
def create_facial_shape_keys(mesh_objects, classify_data):
    """
    Create blink and mouth_open shape keys on the head mesh region.
    Only runs if object has a recognizable face (dog, human, creature, etc.)
    """
    import bpy
    import bmesh

    category    = (classify_data or {}).get('category', '')
    object_type = (classify_data or {}).get('object_type', '').lower()

    has_face = any(w in object_type for w in
                   ['dog', 'cat', 'human', 'creature', 'monster', 'robot', 'alien'])
    if not has_face:
        return

    for mesh_obj in mesh_objects:
        # Basis shape key required first
        if not mesh_obj.data.shape_keys:
            mesh_obj.shape_key_add(name='Basis', from_mix=False)

        # Add named shape keys — actual deformation authored separately
        # For now just register them so they appear in the GLB morph targets
        mesh_obj.shape_key_add(name='blink_left',  from_mix=False)
        mesh_obj.shape_key_add(name='blink_right', from_mix=False)
        mesh_obj.shape_key_add(name='mouth_open',  from_mix=False)

        print(f"  Shape keys added to {mesh_obj.name}")
# ── Blender rigging ───────────────────────────────────────────────────────────

def rig_in_blender(mesh_path: str, joints: list, hierarchy: list,
                   output_path: str, bone_names: list = None, skeleton_joints_data=None):
    try:
        import bpy
    except ImportError:
        raise RuntimeError(
            "Must be run inside Blender:\n"
            "  blender --background --python rig.py -- "
            "--from-json skeleton.json --input model.glb --output rigged.glb"
        )

    import traceback
    try:
        mesh_path   = os.path.abspath(mesh_path)
        output_path = os.path.abspath(output_path)

        print(f"Setting up Blender scene... {mesh_path}")
        bpy.ops.wm.read_factory_settings(use_empty=True)

        ext = Path(mesh_path).suffix.lower()
        if ext in ['.glb', '.gltf']:
            bpy.ops.import_scene.gltf(filepath=mesh_path)
        elif ext == '.obj':
            bpy.ops.wm.obj_import(filepath=mesh_path)
        elif ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=mesh_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        mesh_objects = [o for o in bpy.data.objects if o.type == 'MESH']
        if not mesh_objects:
            raise ValueError("No mesh in imported file")

        for mesh_obj in mesh_objects:
            print(f"Mesh location: {mesh_obj.location}")
            print(f"Mesh rotation: {mesh_obj.rotation_euler}")
            print(f"Mesh scale:    {mesh_obj.scale}")
        print(f"Imported {len(mesh_objects)} mesh object(s)")

        # ── Build armature ────────────────────────────────────────
        bpy.ops.object.armature_add(enter_editmode=False)
        armature_obj                = bpy.context.object
        armature_obj.name           = 'InferredArmature'
        armature_obj.location       = (0, 0, 0)
        armature_obj.rotation_euler = (0, 0, 0)
        armature_obj.scale          = (1, 1, 1)
        bpy.ops.object.mode_set(mode='EDIT')

        print(f"[DEBUG] About to call validate_bone_mesh_fit")
        print(f"[DEBUG] mesh_objects count: {len(mesh_objects)}")
        print(f"[DEBUG] armature_obj: {armature_obj.name}")
        bone_map, hierarchy = build_armature(
            armature_obj.data, joints, hierarchy, bone_names
        )

        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"Armature: {len(bone_map)} bones")
        print(f"[VALIDATE_START]")
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        try:
            print(f"[VALIDATE_START]")
            import sys
            sys.stdout.flush()
            validate_bone_mesh_fit(mesh_objects[0], armature_obj)
            print(f"[VALIDATE_COMPLETE]")
            sys.stdout.flush()
        except Exception as e:
            print(f"[VALIDATE_ERROR] {e}")
            import traceback
            traceback.print_exc()

        # ── Apply transforms ──────────────────────────────────────

        # ── Apply transforms ──────────────────────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        for mesh_obj in mesh_objects:
            mesh_obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        print("Applied transforms to all mesh objects")


        # ── Diagnostic: check mesh health before skinning ─────────────────────────
        import bmesh
        for obj in mesh_objects:
            bpy.context.view_layer.objects.active = obj
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
            loose_verts        = [v for v in bm.verts if not v.link_edges]

            islands = 0
            visited = set()
            for v in bm.verts:
                if v.index not in visited:
                    islands += 1
                    stack = [v]
                    while stack:
                        curr = stack.pop()
                        if curr.index in visited:
                            continue
                        visited.add(curr.index)
                        for e in curr.link_edges:
                            other = e.other_vert(curr)
                            if other.index not in visited:
                                stack.append(other)

            print(f"Mesh health: {len(bm.verts)} verts, {len(bm.faces)} faces")
            print(f"  Non-manifold edges: {len(non_manifold_edges)}")
            print(f"  Loose vertices:     {len(loose_verts)}")
            print(f"  Separate islands:   {islands}")
            bm.free()
        # ── End diagnostic ────────────────────────────────────────────────────────
        
        for obj in mesh_objects:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.002)
            bpy.ops.object.mode_set(mode='OBJECT')
            
            # Check improvement
            bm2 = bmesh.new()
            bm2.from_mesh(obj.data)
            nm2 = [e for e in bm2.edges if not e.is_manifold]
            print(f"  After weld: {len(nm2)} non-manifold edges "
                  f"(was {len(non_manifold_edges)})")
            bm2.free()

        # ── Skin mesh to armature ─────────────────────────────────
        skin_mesh(mesh_objects, armature_obj, skeleton_joints_data)

        # ── Animations ────────────────────────────────────────────
        joint_hints = [j.get('hint') for j in skeleton_joints_data] if skeleton_joints_data else []
        create_animations_from_hints(armature_obj, joint_hints)


        for action in bpy.data.actions:
            print(f"  Action: {action.name}")
            print(f"    FCurves: {len(action.fcurves)}")
            for fc in action.fcurves:
                print(f"      {fc.data_path} [{fc.array_index}]: {len(fc.keyframe_points)} keyframes")

        # ── Export ────────────────────────────────────────────────
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            export_skins=True,
            export_animations=True,
            export_nla_strips=True,
            export_current_frame=False,
            export_bake_animation=False,
            export_optimize_animation_size=True,
            export_optimize_animation_keep_anim_armature=True,
            export_image_format='JPEG',
            export_jpeg_quality=75,
        )
        print(f"Exported: {output_path}")

    except Exception as e:
        print(f"\nrig_in_blender FAILED: {e}")
        traceback.print_exc()
        raise
# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = argv[1:]

    parser = argparse.ArgumentParser(description='Infer skeleton and rig a 3D model')
    parser.add_argument('--input',     required=True,  help='Input mesh (.glb/.obj/.fbx)')
    parser.add_argument('--output',    required=True,  help='Output rigged mesh (.glb)')
    parser.add_argument('--photo',     default=None,   help='Original photo for classification')
    parser.add_argument('--tag',       default=None,   help='User label e.g. "flying sofa"')
    parser.add_argument('--joints',    type=int, default=None,
                        help='Override joint count (default: auto or from classification)')
    parser.add_argument('--viz-only',  action='store_true',
                        help='Output skeleton viz only, skip Blender rigging')
    parser.add_argument('--from-json', default=None,
                        help='Skip inference, load skeleton from existing JSON')
    args = parser.parse_args(argv)

    args.input  = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)
    if args.from_json:
        args.from_json = os.path.abspath(args.from_json)

    print(f"\n{'='*55}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    if args.tag:
        print(f"Tag:    {args.tag}")
    print(f"{'='*55}\n")

    object_info = None
    bone_labels = None

    if args.from_json:
        with open(args.from_json) as f:
            data = json.load(f)
        joints_raw   = [tuple(j['position']) for j in data['joints']]
        
        # Convert GLB Y-up → Blender Z-up: x stays, Y→Z, Z→-Y
        joints       = [(x, -z, y) for x, y, z in joints_raw]
        
        hierarchy    = [(b['parent'], b['child']) for b in data['bones']]
        bone_labels  = [j.get('name', f"joint_{j['id']}") for j in data['joints']]
        skeleton_joints_data = data['joints']
        
        rig_in_blender(args.input, joints, hierarchy, args.output,
           bone_names=bone_labels,
           skeleton_joints_data=skeleton_joints_data)
           
        print(f"Loaded {len(joints)} joints, {len(hierarchy)} bones")
        print(f"Bone names: {bone_labels}")
        return

    else:
        n_joints = args.joints

        if args.photo:
            object_info = infer_from_photo(args.photo, tag=args.tag)
            if object_info:
                bone_labels = extract_joint_names(object_info.get('joint_hints', []))
                if n_joints is None:
                    n_joints = object_info.get('suggested_joints')
            else:
                print("Skipping classification — using geometric auto-detection")

        joints, hierarchy, bounds_size = infer_skeleton_geometric(args.input, n_joints)

        print(f"\nSkeleton:")
        for i, j in enumerate(joints):
            name = bone_labels[i] if bone_labels and i < len(bone_labels) else f"joint_{i}"
            print(f"  {name}: ({j[0]:.3f}, {j[1]:.3f}, {j[2]:.3f})")

        json_path = visualize_skeleton(
            args.input, joints, hierarchy, args.output,
            labels=bone_labels, labels_raw=object_info.get('joint_hints', []) if object_info else None
        )

        if object_info:
            with open(json_path) as f:
                skeleton_data = json.load(f)
            skeleton_data['object_info'] = object_info
            with open(json_path, 'w') as f:
                json.dump(skeleton_data, f, indent=2)

    if args.viz_only:
        json_stem = args.output.replace('.glb', '_skeleton.json')
        viz_stem  = args.output.replace('.glb', '_skeleton_viz.glb')
        print(f"\nViz only — done.")
        print(f"  Inspect: {viz_stem}")
        print(f"  JSON:    {json_stem}")
        print(f"\nWhen happy with joint placement, rig in Blender:")
        print(f"  blender --background --python rig.py -- \\")
        print(f"    --from-json {json_stem} \\")
        print(f"    --input {args.input} \\")
        print(f"    --output {args.output}")
        return

    # ── Step 2: Rig in Blender ───────────────────────────────────
    try:
        skeleton_joints_data = object_info.get('joint_hints', []) if object_info else None
        rig_in_blender(args.input, joints, hierarchy, args.output,
                       bone_names=bone_labels,
                       skeleton_joints_data=skeleton_joints_data)
        print(f"\n✓ Done! Open {args.output} in Blender or https://gltf.report")
    except RuntimeError as e:
        print(f"\n{e}")


if __name__ == '__main__':
    main()
