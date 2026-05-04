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

    clips = {}
    for hint in joint_hints:
        if not isinstance(hint, dict):
            continue
        for anim in hint.get('animations', []):
            clips.setdefault(anim['clip'], []).append((hint['name'], anim))

    if not clips:
        return

    axis_map   = {'x': 0, 'y': 1, 'z': 2}
    pose_bones = {b.name: b for b in armature_obj.pose.bones}

    armature_obj.animation_data_create()
    bpy.context.view_layer.objects.active = armature_obj
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
    Assign each vertex to its nearest bone segment.
    Uses deforms_mesh hint from Gemini to constrain rigid body parts.
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
    mesh_size = np.linalg.norm(bmax - bmin)  # ← add this

    seg_dists = np.column_stack([
        point_to_segment_distance(verts, head, tail)
        for head, tail in bone_segments
    ])

# Find terminal bones — bones that are never a parent of another bone
    all_parent_names = set()
    for b in armature_obj.data.bones:
        if b.parent:
            all_parent_names.add(b.parent.name)

        terminal_bone_indices = [
            i for i, name in enumerate(bone_names_list)
            if name not in all_parent_names
        ]

    # Hard-lock vertices near terminal bones to those bones exclusively
    for ti in terminal_bone_indices:
        head, tail      = bone_segments[ti]
        bone_center     = (head + tail) / 2
        terminal_radius = mesh_size * 0.18
        dists_to_bone   = np.linalg.norm(verts - bone_center, axis=1)
        terminal_mask   = dists_to_bone < terminal_radius
        if terminal_mask.any():
            seg_dists[terminal_mask, :]  = np.inf
            seg_dists[terminal_mask, ti] = 0.0
            
            print(f"  Locked {terminal_mask.sum()} vertices to terminal bone: {bone_names_list[ti]}")
    if skeleton_joints_data:
        rigid_bones   = set()
        deform_bones  = set()
        for joint in skeleton_joints_data:
            hint = joint.get('hint') or {}
            name = joint.get('name', '')
            if not hint.get('deforms_mesh', True):
                rigid_bones.add(name)
            else:
                deform_bones.add(name)

        rigid_indices  = [i for i, n in enumerate(bone_names_list) if n in rigid_bones]
        deform_indices = [i for i, n in enumerate(bone_names_list) if n in deform_bones]

        if rigid_indices and deform_indices:
            mesh_size = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
            for ri in rigid_indices:
                head, tail       = bone_segments[ri]
                bone_center_y    = (head[1] + tail[1]) / 2
                y_range          = abs(tail[1] - head[1]) * 0.3 + mesh_size * 0.05  # ← REDUCED from 0.5 + 0.1
                near_rigid_mask  = np.abs(verts[:, 1] - bone_center_y) < y_range
                seg_dists[np.ix_(near_rigid_mask, deform_indices)] = np.inf

    bmin     = np.percentile(verts, 2,  axis=0)
    bmax     = np.percentile(verts, 98, axis=0)
    center_y = (bmin[2] + bmax[2]) / 2

    upper_bone_indices = [
        i for i, name in enumerate(bone_names_list)
        if any(k in name.lower() for k in ['head', 'neck', 'spine', 'torso'])
    ]
    lower_bone_indices = [
        i for i, name in enumerate(bone_names_list)
        if any(k in name.lower() for k in ['leg', 'foot', 'root', 'pelvis', 'hip'])
    ]
    upper_mask = verts[:, 2] > center_y
    lower_mask = verts[:, 2] <= center_y

    if upper_bone_indices and lower_bone_indices:
        seg_dists[np.ix_(upper_mask, lower_bone_indices)] *= 10.0
        seg_dists[np.ix_(lower_mask, upper_bone_indices)] *= 10.0

    nearest = np.argmin(seg_dists, axis=1)
    for bi, bname in enumerate(bone_names_list):
        mask = np.where(nearest == bi)[0]
        if len(mask):
            mesh_obj.vertex_groups[bname].add(
                mask.tolist(), 1.0, 'REPLACE'
            )

    mod        = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = armature_obj
    print(f"  Segment weights assigned: {mesh_obj.name}")
    
    
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
    """
    Create bones in edit mode. Returns bone_map.
    """
    import bpy

    for bone in arm_data.edit_bones:
        arm_data.edit_bones.remove(bone)

    # Deduplicate hierarchy
    seen_children    = set()
    unique_hierarchy = []
    for parent_idx, child_idx in hierarchy:
        if child_idx not in seen_children:
            seen_children.add(child_idx)
            unique_hierarchy.append((parent_idx, child_idx))
        else:
            print(f"  Skipping duplicate child bone {child_idx}")
    hierarchy = unique_hierarchy

    # Find root
    child_indices = {c for p, c in hierarchy}
    root_idx      = next((i for i in range(len(joints))
                          if i not in child_indices), 0)
    root_name     = bone_names[root_idx] if bone_names and root_idx < len(bone_names) else f'joint_{root_idx}'
    print(f"  Root joint: {root_name}")

    # Create root bone
    root_bone       = arm_data.edit_bones.new(root_name)
    root_bone.head  = joints[root_idx]
    first_child_idx = next((c for p, c in hierarchy if p == root_idx), None)
    root_bone.tail  = joints[first_child_idx] if first_child_idx is not None else (
        joints[root_idx][0], joints[root_idx][1], joints[root_idx][2] + 0.1
    )
    bone_map = {(root_idx, root_idx): root_bone}
    print(f"  Bone: {root_name} (root)")

    def bone_name(parent_idx, child_idx):
        if bone_names and child_idx < len(bone_names):
            return bone_names[child_idx]
        return f"bone_{child_idx}"

    for parent_idx, child_idx in hierarchy:
        name       = bone_name(parent_idx, child_idx)
        bone       = arm_data.edit_bones.new(name)
        bone.head  = joints[parent_idx]
        bone.tail  = joints[child_idx]
        bone_map[(parent_idx, child_idx)] = bone
        print(f"  Bone: {name}")

    for (p1, c1), b1 in bone_map.items():
        for (p2, c2), b2 in bone_map.items():
            if c1 == p2 and b1 != b2:
                b2.parent      = b1
                b2.use_connect = True

    return bone_map, hierarchy


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

        bone_map, hierarchy = build_armature(
            armature_obj.data, joints, hierarchy, bone_names
        )

        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"Armature: {len(bone_map)} bones")

        # ── Apply transforms ──────────────────────────────────────
        bpy.ops.object.select_all(action='DESELECT')
        for mesh_obj in mesh_objects:
            mesh_obj.select_set(True)
            bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        print("Applied transforms to all mesh objects")

        # ── Skin mesh to armature ─────────────────────────────────
        skin_mesh(mesh_objects, armature_obj, skeleton_joints_data)

        # ── Animations ────────────────────────────────────────────
        joint_hints = [j.get('hint') for j in skeleton_joints_data] if skeleton_joints_data else []
        create_animations_from_hints(armature_obj, joint_hints)

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
