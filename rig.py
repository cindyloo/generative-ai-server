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

            #coord_remap = {
            #    'x': (2,  1.0),  # image-x → Blender Z (forward/rear)
             #   'y': (1,  1.0),  # image-y → Blender Y (up, no inversion needed for rotation)
             #   'z': (0,  1.0),  # image-z → Blender X (left/right)
            #}
            coord_remap = {
                'x': (0, 1.0),   # x → X (leg swing axis for Z-up bones)
                'y': (2, 1.0),   # y → Z
                'z': (1, 1.0),   # z → Y
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
    parent_head_by_index  = {}
    parent_index_by_index = {}
    for i, b in enumerate(armature_obj.data.bones):
        head = np.array(armature_obj.matrix_world @ b.head_local)
        tail = np.array(armature_obj.matrix_world @ b.tail_local)
        bone_segments.append((head, tail))
        bone_names_list.append(b.name)
        if b.parent is not None:
            parent_head_by_index[i] = np.array(
                armature_obj.matrix_world @ b.parent.head_local)

    name_to_index = {n: i for i, n in enumerate(bone_names_list)}
    for i, b in enumerate(armature_obj.data.bones):
        if b.parent is not None:
            parent_index_by_index[i] = name_to_index[b.parent.name]
    child_indices_by_index = {}
    for i, pi in parent_index_by_index.items():
        child_indices_by_index.setdefault(pi, []).append(i)

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

    # ── Simplest rule that works: each vertex moves with whichever bone is
    # nearest, weighted by a Gaussian that decays to zero beyond a per-bone
    # cutoff distance. No spine-vs-limb grouping, no left/right chain
    # partitioning, no separate rigid-lock pass -- all of that was layered
    # on over several iterations to patch specific symptoms (a wide body
    # picking an off-center hip over the central spine, a foot bone's tiny
    # synthetic tail excluding real foot vertices, left/right sides
    # blending into each other) and each patch kept surfacing a new
    # failure elsewhere: internal tearing at every joint boundary, whole
    # chains of vertices defaulting to a frozen root bone, opposite-side
    # contamination. A tight enough per-bone cutoff prevents cross-talk by
    # construction -- hip_left simply can't reach hip_right or a shoulder
    # if its own reach is kept close to its real extent -- and a smooth
    # Gaussian within that cutoff means two mesh-adjacent vertices always
    # have similar bone weights, so there's no seam for the mesh to tear
    # along in the first place.
    #
    # A single bone's own head-to-tail length is not a reliable cutoff
    # scale for a TERMINAL bone (foot, hand -- no real child, so Blender
    # gives it a short, arbitrary synthetic tail): on a compact/stubby limb
    # like the tomato's, that can leave the foot with an almost-zero scale
    # even though the leg it's attached to reaches much further, starving
    # real foot-sole vertices of any valid cutoff. Non-terminal bones don't
    # have this problem -- their own segment (or the one coming in from
    # their parent) is a real, meaningful joint-to-joint distance -- and
    # giving them the same whole-chain fallback was too generous, letting
    # a bone like the hip reach nearly the full leg's length into the
    # torso. So the whole-chain fallback applies to terminal bones only.
    limb_bone_indices = set()
    for bi, bname in enumerate(bone_names_list):
        bname_lower = bname.lower()
        if any(p in bname_lower for p in
               ['shoulder', 'elbow', 'hand', 'hip', 'knee', 'foot',
                'wing_base', 'wing_mid', 'wing_tip']):
            limb_bone_indices.add(bi)

    all_parent_names = {b.parent.name for b in armature_obj.data.bones if b.parent}
    terminal_bone_indices = {i for i, name in enumerate(bone_names_list)
                              if name not in all_parent_names}

    chain_length_by_bone = {}
    chain_root_by_bone   = {}
    for bi in limb_bone_indices:
        # Chain root: walk up while the parent is ALSO a limb bone --
        # stops at the first limb bone whose parent attaches to the main
        # body (spine, or nothing).
        root_idx = bi
        while (root_idx in parent_index_by_index
               and parent_index_by_index[root_idx] in limb_bone_indices):
            root_idx = parent_index_by_index[root_idx]
        chain_root_by_bone[bi] = root_idx
        # Walk down from the root to the terminal tip, summing lengths.
        total, cur, seen = 0.0, root_idx, set()
        while cur not in seen:
            seen.add(cur)
            h, t = bone_segments[cur]
            total += float(np.linalg.norm(t - h))
            children = [c for c in child_indices_by_index.get(cur, [])
                        if c in limb_bone_indices]
            if not children:
                break
            cur = children[0]
        chain_length_by_bone[bi] = total

    # A limb chain attaches to the body at its root (hip for a leg,
    # shoulder for an arm) and only ever extends outward/downward from
    # there -- "rig the leg from where the hip is down." Cap each chain's
    # candidacy at its own root's attachment height, so a leg bone simply
    # cannot compete for a vertex sitting above where the leg attaches to
    # the body, no matter how close it is in raw 3D distance. This is a
    # hard, literal rule tied directly to the joint's own (vision-
    # verified) position -- no proportional padding to calibrate. A small
    # fixed margin allows for surface thickness right at the joint (e.g.
    # a T-pose arm's own rounded thickness sitting slightly above the
    # exact shoulder height), not a body-shape-dependent one.
    UP_AXIS = 2
    chain_ceiling_by_bone = {}
    for bi in limb_bone_indices:
        root_idx  = chain_root_by_bone[bi]
        root_head, root_tail = bone_segments[root_idx]
        # Margin scaled to the chain's OWN root segment, not mesh_size --
        # a mesh_size-based margin is an absolute reference that breaks
        # down exactly like it did for the old Z-band padding (a giant
        # head dominating mesh_size while the actual limb is tiny).
        margin = np.linalg.norm(root_tail - root_head) * 0.3
        chain_ceiling_by_bone[bi] = root_head[UP_AXIS] + margin

    # Same idea, applied laterally: a chain rooted clearly on one side of
    # the body (shoulder_left, hip_left) only ever rigs that side. Without
    # this, left and right instances of the same bone -- at very similar
    # heights by construction -- have nothing else to separate them by
    # raw distance, and can blend into each other for a near-central
    # vertex. Only exclude the OPPOSITE side, with a small crossover
    # margin at the midline for smooth blending there; a same-side vertex
    # is never excluded no matter how far laterally it sits from this
    # specific bone (a wide/round body's leg-adjacent vertices can
    # legitimately sit far in X from the leg bone's own narrow position).
    # A per-bone margin (sized off each chain's own segment) produced two
    # DIFFERENT crossover widths that didn't meet at the same line -- e.g.
    # shoulder_left's own margin excluded it only past X=+0.05, while
    # shoulder_right's margin let it reach as far as X=-0.14, leaving a
    # wide dead zone in between where BOTH remained eligible. A margin
    # shared by both sides of a pair fixes that -- but sizing it off the
    # OVERALL mesh width breaks just like the old mesh_size floor did: a
    # human in a T-pose has arms stretched out sideways, which balloons
    # the mesh's overall X-extent and made the shared margin far too
    # generous at the midline. Size each pair's margin off the actual
    # distance BETWEEN that pair's own two joints (shoulder_left to
    # shoulder_right, hip_left to hip_right) instead -- a real, local
    # measurement of how wide the body is at that specific joint level.
    SIDE_AXIS = 0
    chain_side_by_bone = {}
    for bi in limb_bone_indices:
        root_idx = chain_root_by_bone[bi]
        root_name = bone_names_list[root_idx].lower()
        root_head, root_tail = bone_segments[root_idx]
        center = (root_head[SIDE_AXIS] + root_tail[SIDE_AXIS]) / 2

        if 'left' in root_name:
            partner_name = root_name.replace('left', 'right')
        elif 'right' in root_name:
            partner_name = root_name.replace('right', 'left')
        else:
            continue  # root sits centrally -- no side to restrict to

        partner_idx = next((i for i, n in enumerate(bone_names_list)
                             if n.lower() == partner_name), None)
        if partner_idx is None:
            continue
        p_head, p_tail = bone_segments[partner_idx]
        partner_center = (p_head[SIDE_AXIS] + p_tail[SIDE_AXIS]) / 2
        pair_margin = abs(center - partner_center) * 0.02

        if center > pair_margin:
            chain_side_by_bone[bi] = (1, pair_margin)
        elif center < -pair_margin:
            chain_side_by_bone[bi] = (-1, pair_margin)
        # else: this chain's root sits within its own pair's crossover
        # margin of the midline -- no side to restrict to.

    effective_len_by_bone = {}
    for bi in range(len(bone_names_list)):
        head, tail = bone_segments[bi]
        bone_len = np.linalg.norm(tail - head)
        parent_head = parent_head_by_index.get(bi)
        incoming_len = (np.linalg.norm(head - parent_head)
                        if parent_head is not None else 0.0)
        chain_fallback = (chain_length_by_bone.get(bi, 0.0)
                          if bi in terminal_bone_indices else 0.0)
        effective_len_by_bone[bi] = max(
            bone_len, incoming_len, chain_fallback, 1e-4)

    # An ABSOLUTE cutoff (zero weight beyond a fixed distance) forces a
    # trade-off that can't be won with one constant: tight enough to keep
    # a hip from reaching into the torso, and a real chunk of the mesh
    # ends up farther than EVERY bone's cutoff (all-zero rows, falling
    # back to a Blender/exporter default "neutral_bone" with no real
    # weight at all); loose enough to give every vertex a real candidate,
    # and the same over-reach that let a hip contaminate the torso or one
    # shoulder blend into the other comes right back.
    #
    # What actually matters is RELATIVE, not absolute: how much farther is
    # this bone than the closest one *for this vertex*. Weight each bone
    # by the gap between its distance and the nearest bone's distance. The
    # nearest bone always has gap=0, so it always gets full weight before
    # normalization -- every vertex has a well-defined nearest bone, so
    # there is no coverage gap to patch. A bone on the wrong side of the
    # body (the opposite shoulder, the opposite leg) or in the wrong
    # region (a hip reaching into the torso) has a large gap versus
    # whatever bone is actually closest there, so it decays away on its
    # own without needing a separate cross-side or cross-region rule --
    # PROVIDED the decay is scaled to that competing bone's OWN local
    # size, not the nearest bone's. Scaling every bone's decay by the
    # nearest bone's scale was a real regression: on a body dominated by
    # a few large spine bones (broccoli's chest/spine, which each span a
    # big fraction of total height), that let a small bone like hip stay
    # falsely competitive over a wide swath of the torso, since a modest
    # gap looked small relative to the spine bone's large sigma even
    # though it was large relative to hip's own actual reach. Each
    # candidate bone's contribution should decay relative to ITS OWN
    # scale -- how much farther than its best case is this, for THIS
    # bone specifically.
    SIGMA_FACTOR = 0.4

    # Add the excess-above-ceiling (or excess-past-the-midline) as EXTRA
    # distance, rather than hard-excluding with infinity. A hard cutoff
    # creates a cliff exactly AT the boundary: a vertex a fraction of a
    # millimeter below the ceiling can still blend across several leg
    # bones (each near-equidistant, ~25% apiece), while its neighbor a
    # fraction above drops to 0% leg weight outright, since every leg
    # bone in the chain is excluded at once, all at the same threshold.
    # That's a harder, more abrupt cliff than the one this whole approach
    # was built to avoid -- confirmed visually as a jagged, serrated tear
    # right at the hip/body boundary. Treating "how far past the
    # boundary" as additional distance lets the Gaussian decay it away
    # smoothly instead, the same way ordinary distance already decays
    # everything else.
    # The excess is added at a multiple of its own value, not 1:1 -- a
    # flat 1x addition was too gentle to suppress anything on a character
    # whose bones have a larger sigma relative to how far past the
    # boundary a typical nearby vertex actually sits (the tomato): the
    # excess ended up small relative to sigma, barely denting the
    # Gaussian, so contamination came right back (confirmed: a whole
    # neighborhood collapsed to a near-uniform blend across pelvis, hip,
    # knee, and foot at once). Multiplying it up first makes the same
    # physical distance past the boundary count for more in the decay,
    # without reintroducing a hard, infinite-at-the-line cliff.
    EXCESS_PENALTY = 5.0
    for bi, ceiling in chain_ceiling_by_bone.items():
        excess = np.maximum(0.0, verts[:, UP_AXIS] - ceiling)
        seg_dists[:, bi] += excess * EXCESS_PENALTY
    for bi, (side, side_margin) in chain_side_by_bone.items():
        if side > 0:
            excess = np.maximum(0.0, -side_margin - verts[:, SIDE_AXIS])
        else:
            excess = np.maximum(0.0, verts[:, SIDE_AXIS] - side_margin)
        seg_dists[:, bi] += excess * EXCESS_PENALTY

    nearest_dist = seg_dists.min(axis=1, keepdims=True)
    gap          = seg_dists - nearest_dist
    sigma_by_bone = np.array(
        [effective_len_by_bone[bi] for bi in range(len(bone_names_list))]
    ) * SIGMA_FACTOR
    smooth_weights = np.exp(-(gap ** 2) / (2 * sigma_by_bone[None, :] ** 2))


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
    
def _validate_heat_weights(mesh_objects, armature_obj,
                            envelope_factor=2.5, size_floor_factor=0.06,
                            leaked_weight_threshold=0.10, leak_fraction=0.15):
    """
    Check whether Blender's heat-weighting (Automatic Weights) produced
    geometrically local weights, rather than just checking that vertex
    groups exist at all. Heat diffusion can badly leak a bone's influence
    across the whole mesh when that bone is very short relative to the
    mesh (e.g. stubby legs on a round character) — the diffusion source
    is too small/close to neighboring geometry to localize properly, so
    far-away vertices (chest, head) can end up with substantial weight
    from a leg bone. The previous check here only asked "did any vertex
    group get any weight at all", which this kind of leak still passes.

    Each bone gets its own influence envelope — envelope_factor times that
    bone's own length, with a floor of size_floor_factor times the mesh's
    bounding-box diagonal (so a very short/near-zero-length bone still
    gets a sane minimum envelope instead of ~0). A vertex whose distance
    to a bone exceeds that bone's envelope is "far" from it. Comparing
    against each bone's OWN envelope, rather than a multiple of the
    vertex's nearest-bone distance, matters specifically for a compact
    body: when every bone sits within a similar distance range of a given
    vertex (e.g. a round character with a short overall skeleton), a
    relative "3x the nearest bone" comparison never triggers even for a
    clearly wrong assignment, because nothing is proportionally much
    farther than anything else.

    For each weighted vertex, sum the weight assigned to far bones —
    checking only the single heaviest-weighted bone per vertex is not
    enough, since a vertex can have a perfectly reasonable primary bone
    (e.g. chest) but still carry real, anatomically-implausible secondary
    weight from a distant bone (e.g. a leg) that a primary-only check
    would miss. Flags a vertex as leaked when that far-bone weight
    exceeds leaked_weight_threshold of its total weight. Returns False
    (meaning "fall back to segment weighting") when too large a fraction
    of vertices are leaked.
    """
    import numpy as np

    bone_names = [b.name for b in armature_obj.data.bones]
    bone_segments = []
    for b in armature_obj.data.bones:
        head = np.array(armature_obj.matrix_world @ b.head_local)
        tail = np.array(armature_obj.matrix_world @ b.tail_local)
        bone_segments.append((head, tail))
    if not bone_segments:
        return True

    all_verts_for_size = []
    for mesh_obj in mesh_objects:
        verts_local = np.array([v.co for v in mesh_obj.data.vertices])
        if len(verts_local) == 0:
            continue
        mat  = np.array(mesh_obj.matrix_world)
        ones = np.ones((len(verts_local), 1))
        all_verts_for_size.append((mat @ np.hstack([verts_local, ones]).T).T[:, :3])
    if not all_verts_for_size:
        return True
    all_verts = np.vstack(all_verts_for_size)
    mesh_diagonal = float(np.linalg.norm(all_verts.max(axis=0) - all_verts.min(axis=0)))
    size_floor = size_floor_factor * mesh_diagonal

    bone_envelopes = np.array([
        envelope_factor * max(np.linalg.norm(tail - head), size_floor)
        for head, tail in bone_segments
    ])

    bad_count   = 0
    total_count = 0
    for mesh_obj in mesh_objects:
        verts_local = np.array([v.co for v in mesh_obj.data.vertices])
        if len(verts_local) == 0:
            continue
        mat  = np.array(mesh_obj.matrix_world)
        ones = np.ones((len(verts_local), 1))
        verts = (mat @ np.hstack([verts_local, ones]).T).T[:, :3]

        dists = np.column_stack([
            point_to_segment_distance(verts, head, tail)
            for head, tail in bone_segments
        ])

        group_to_bone = {}
        for vg in mesh_obj.vertex_groups:
            if vg.name in bone_names:
                group_to_bone[vg.index] = bone_names.index(vg.name)

        for vi, v in enumerate(mesh_obj.data.vertices):
            if not v.groups:
                continue
            total_w = 0.0
            far_w   = 0.0
            for g in v.groups:
                bi = group_to_bone.get(g.group)
                if bi is None:
                    continue
                total_w += g.weight
                if dists[vi, bi] > bone_envelopes[bi]:
                    far_w += g.weight
            if total_w <= 0:
                continue
            total_count += 1
            if far_w / total_w > leaked_weight_threshold:
                bad_count += 1

    if total_count == 0:
        return True
    bad_fraction = bad_count / total_count
    print(f"  Weight locality check: {bad_count}/{total_count} vertices "
          f"({bad_fraction:.1%}) carry >{leaked_weight_threshold:.0%} weight "
          f"from bones much farther than their nearest bone")
    return bad_fraction <= leak_fraction


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

    if heat_succeeded and not _validate_heat_weights(mesh_objects, armature_obj):
        print("  Heat weighting produced leaked/non-local weights (likely "
              "from a very short bone) — falling back to segment weighting")
        heat_succeeded = False

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
