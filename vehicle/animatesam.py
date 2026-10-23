"""
animatesam.py  (Blender script)
================================
Step 3 of the vehicle pipeline.

Loads the separated GLB (one mesh per wheel + body), sets each wheel's
origin to the Taubin-computed pivot stored in blender_wheel_centroids,
then adds NLA clips: road wheels get drive (spin in place), drive_forward
and drive_backward (the whole vehicle rolls one wheel-turn) and steer_left /
steer_right (front wheels); gears and hinges get drive.

Coordinate system (Blender world space after GLB import)
---------------------------------------------------------
  X = vehicle length (front ↔ rear)
  Y = axle (left ↔ right)
  Z = height (up)

Wheels on a vehicle (side view) rotate around the Y axis (the axle).
Gears in a mechanical assembly use the rotation axis stored in classify JSON.

What changed vs the previous version
--------------------------------------
1.  Removed the dead `claude_pivots` / `remaining_blender` / `remaining_claude`
    pools — pivot is now looked up directly from blender_wheel_centroids by
    wheel object name, which is guaranteed to match because classify_wheels.py
    names separated objects after their centroid key.

2.  Removed the duplicate vertex-diagnostics block (was printed twice).

3.  `hint` variable in the hinge/gear branches is now looked up from
    joint_hints_by_name[wheel.name] rather than using an undefined variable.

4.  Gear speed-ratio uses brange computed from actual Blender mesh bounds
    rather than the undefined full_y_range / ref_r_mesh variables.
"""

import bpy, sys, math, json, os
import numpy as np

input_path    = sys.argv[sys.argv.index('--') + 1]
output_path   = sys.argv[sys.argv.index('--') + 2]
classify_json = (sys.argv[sys.argv.index('--') + 3]
                 if len(sys.argv) > sys.argv.index('--') + 3 else None)

# ── Load classify data ────────────────────────────────────────────────────────
wheel_joints            = []
blender_wheel_centroids = {}
joint_hints_by_name     = {}
is_mechanical           = False
reference_radius        = None
classify_data           = None

if classify_json and os.path.exists(classify_json):
    with open(classify_json) as f:
        classify_data = json.load(f)
    wheel_joints        = [j for j in classify_data.get('joint_hints', [])
                           if j.get('body_part') in ['wheel', 'gear']]
    joint_hints_by_name = {j['name']: j for j in classify_data.get('joint_hints', [])}
    is_mechanical       = (classify_data.get('is_mechanical', False) or
                       classify_data.get('category', '') == 'mechanical' or
                       classify_data.get('rig_type', '') == 'mechanical')
    reference_radius    = classify_data.get('reference_radius_normalized', None)

    # Keep full dict (centroid + radius) so animatesam can use radius for pivot
    raw_bwc = classify_data.get('blender_wheel_centroids', {})
    for name, v in raw_bwc.items():
        if v is None:
            continue
        blender_wheel_centroids[name] = v   # preserve {'centroid': [...], 'radius': ...}

    print(f"Loaded {len(blender_wheel_centroids)} Blender centroids")
    print(f"is_mechanical: {is_mechanical}  reference_radius: {reference_radius}")
else:
    print(f"classify_json not found or not provided: {classify_json}")

# ── Load scene ────────────────────────────────────────────────────────────────
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_path)

# Mesh_0 is an artefact some exporters add — skip it
mesh_objects = [o for o in bpy.data.objects
                if o.type == 'MESH' and o.name != 'Mesh_0']

# ── Compute overall mesh bounds for gear speed-ratio ─────────────────────────
all_co = np.array([(v.co.x, v.co.y, v.co.z)
                   for obj in mesh_objects for v in obj.data.vertices])
if len(all_co):
    mesh_bmin  = all_co.min(axis=0)
    mesh_bmax  = all_co.max(axis=0)
    mesh_brange = mesh_bmax - mesh_bmin
else:
    mesh_brange = np.ones(3)

# Reference radius in Blender world units (for gear speed ratios)
ref_r_mesh = (reference_radius * mesh_brange.max()) if reference_radius else None

# ── Verify and fill centroids ─────────────────────────────────────────────────
print(f"\nWheel centroid verification:")
for wheel in sorted(mesh_objects, key=lambda o: o.name):
    verts_np     = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    actual_mean  = verts_np.mean(axis=0)
    stored       = blender_wheel_centroids.get(wheel.name)

    if stored is not None:
        pos = stored['centroid'] if isinstance(stored, dict) else stored
        print(f"  {wheel.name}: Taubin pivot {[round(x,3) for x in pos]}, "
              f"mesh mean ({actual_mean[0]:.3f},{actual_mean[1]:.3f},{actual_mean[2]:.3f})")
    else:
        blender_wheel_centroids[wheel.name] = actual_mean.tolist()
        print(f"  {wheel.name}: no Taubin centroid — using mesh mean")

# ── Animation setup ───────────────────────────────────────────────────────────
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = 61

# ══════════════════════════════════════════════════════════════════════════════
# Blender world axes after GLB import (see classify_wheels.py):
#   X = vehicle length (front ↔ rear)
#   Y = axle (left ↔ right)
#   Z = height (up)
# Wheels turn about Y, their axle. The index names below are historical:
# fr_idx (1) is the axle, not front-rear.
# ══════════════════════════════════════════════════════════════════════════════
lr_idx = 0
fr_idx = 1
h_idx  = 2   # height

rotation_axis = fr_idx

print(f"Wheel rotation axis: {['X','Y','Z'][rotation_axis]} (index={rotation_axis})")

# ── Animate each wheel object ─────────────────────────────────────────────────
wheel_objects = sorted(mesh_objects, key=lambda o: o.name)
road_wheels   = []   # (object, radius): standard wheels, animated after the loop

for i, wheel in enumerate(wheel_objects):
    verts_np     = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    actual_center = verts_np.mean(axis=0)

    # Vertex spread diagnostics (once per wheel, not duplicated)
    distances = np.linalg.norm(verts_np - actual_center, axis=1)
    mean_dist = distances.mean()
    far_verts = np.sum(distances > mean_dist * 2)
    print(f"\n{wheel.name}: {len(verts_np)} verts  "
          f"std={distances.std():.3f}  far={far_verts} ({far_verts/len(verts_np)*100:.1f}%)")
    if distances.std() > mean_dist * 0.3 or far_verts > len(verts_np) * 0.1:
        print(f"  ⚠️  asymmetric — may contain non-wheel geometry")

    # ── Resolve pivot ────────────────────────────────────────────────────────
    # Object names after Blender separation may be Mesh_0.001 etc. rather than
    # wheel_fl. Match by finding the centroid whose X sign and Z value are
    # closest to this wheel object's mesh mean.
    pivot_entry = blender_wheel_centroids.get(wheel.name)

    if pivot_entry is None and blender_wheel_centroids:
        # Name mismatch — find nearest centroid to this object's mesh mean
        best_name = None
        best_dist = float('inf')
        for cname, centry in blender_wheel_centroids.items():
            cpos = centry['centroid'] if isinstance(centry, dict) else centry
            d = np.linalg.norm(np.array(cpos) - actual_center)
            if d < best_dist:
                best_dist = d
                best_name = cname
        if best_name:
            pivot_entry = blender_wheel_centroids[best_name]
            print(f"  name mismatch: matched {wheel.name} → {best_name} (dist={best_dist:.3f})")

    if pivot_entry is not None:
        cpos = pivot_entry['centroid'] if isinstance(pivot_entry, dict) else pivot_entry
        cx, cy, cz = float(cpos[0]), float(cpos[1]), float(cpos[2])
        print(f"  using Taubin pivot")
    else:
        # No centroid resolved — fall back to mesh mean. Note: if the mesh
        # is missing exterior verts the mean is biased and the wheel will
        # wobble; the Taubin pivot is preferred whenever available.
        cx = float(actual_center[0])
        cy = float(actual_center[1])
        cz = float(actual_center[2])
        print(f"  using mesh mean (no centroid match)")

    print(f"  pivot: ({cx:.4f}, {cy:.4f}, {cz:.4f})")

    # ── Set Blender origin to pivot ──────────────────────────────────────────
    # cursor.location is in world space — origin_set moves the object origin
    # to the cursor without transform_apply, which would shift the mesh.
    bpy.context.view_layer.objects.active = wheel
    wheel.select_set(True)
    bpy.context.scene.cursor.location = (cx, cy, cz)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    wheel.select_set(False)
    bpy.context.view_layer.update()

    # ── Build action ─────────────────────────────────────────────────────────
    wheel.rotation_mode = 'XYZ'
    action = bpy.data.actions.new(name=f"wheel_{i}_rot")

    # Look up the hint for this wheel (used by hinge / gear branches)
    hint      = joint_hints_by_name.get(wheel.name, {})
    body_part = hint.get('body_part', 'wheel')

    if body_part == 'hinge':
        hinge_axis  = hint.get('hinge_axis', 'y')
        hinge_range = hint.get('hinge_range', [-90, 0])
        axis_map    = {'x': 0, 'y': 1, 'z': 2}
        rot_idx     = axis_map.get(hinge_axis.lower(), 1)
        min_rad     = math.radians(hinge_range[0])
        max_rad     = math.radians(hinge_range[1])
        fc = action.fcurves.new(data_path='rotation_euler', index=rot_idx)
        fc.keyframe_points.add(3)
        fc.keyframe_points[0].co = (1,  max_rad)
        fc.keyframe_points[1].co = (30, min_rad)
        fc.keyframe_points[2].co = (61, max_rad)
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
        print(f"  hinge: axis={hinge_axis} range={hinge_range}°")

    elif body_part == 'gear' and is_mechanical and ref_r_mesh is not None:
        r_norm      = hint.get('wheel_radius_normalized', 0.15)
        # Get rotation direction from animations keyframes if available
        rot_dir = hint.get('rotation_direction', 1)
        animations = hint.get('animations', [])
        if animations:
            kf = animations[0].get('keyframes', [])
            if len(kf) >= 2:
                final_angle = kf[-1][1]
                rot_dir = -1 if final_angle < 0 else 1
        this_r_mesh = r_norm * mesh_brange.max()
        speed_ratio = ref_r_mesh / max(this_r_mesh, 1e-6)
        total_angle = math.pi * 2 * speed_ratio * rot_dir
        fc = action.fcurves.new(data_path='rotation_euler', index=rotation_axis)
        fc.keyframe_points.add(2)
        fc.keyframe_points[0].co = (1,  0.0)
        fc.keyframe_points[1].co = (61, total_angle)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        print(f"  gear: r_norm={r_norm:.3f} dir={rot_dir:+d} "
              f"ratio={speed_ratio:.2f} angle={math.degrees(total_angle):.1f}°")

    else:
        # Standard vehicle wheel: its clips are built after this loop, once
        # every wheel's pivot is known — which way is forward comes from
        # where the front wheels sit relative to the rear ones.
        bpy.data.actions.remove(action)
        radius = pivot_entry.get('radius') if isinstance(pivot_entry, dict) else None
        road_wheels.append((wheel, float(radius) if radius else
                            float(np.ptp(verts_np[:, h_idx])) / 2))
        print(f"  wheel: radius {road_wheels[-1][1]:.3f}, clips built below")
        continue

    # ── Push to NLA ──────────────────────────────────────────────────────────
    wheel.animation_data_create()
    wheel.animation_data.action = action
    track      = wheel.animation_data.nla_tracks.new()
    track.name = "drive"
    strip      = track.strips.new("drive", 1, action)
    wheel.animation_data.action = None
    print(f"  NLA strip 'drive' created")

# ── Road-wheel clips ──────────────────────────────────────────────────────────
# Actions are named "{clip}__{object}"; merge_animations.py joins each clip's
# actions across objects into one animation named {clip}.
#
# Rolling: wheels turn about Blender Y (the axle) and the vehicle runs along
# X. Turning +θ about +Y moves a wheel's contact point toward -X, so rolling
# a distance D along +X without slipping is +D/R, and along -X it is -D/R.
# Steering: +θ about the vertical (Z) turns any heading toward the vehicle's
# left (up × forward), whichever way it faces. Euler XYZ applies the roll
# about the axle first and the steer after it, as a real front wheel does.
STEER   = math.radians(25)
END     = 61


def add_clip(obj, clip, curves, interpolation='LINEAR'):
    """curves: [(data_path, index, [(frame, value), ...]), ...]"""
    act = bpy.data.actions.new(name=f"{clip}__{obj.name}")
    for path, idx, keys in curves:
        fc = act.fcurves.new(data_path=path, index=idx)
        fc.keyframe_points.add(len(keys))
        for kp, (frame, value) in zip(fc.keyframe_points, keys):
            kp.co = (frame, value)
            kp.interpolation = interpolation
    obj.animation_data_create()
    track      = obj.animation_data.nla_tracks.new()
    track.name = clip
    track.strips.new(clip, 1, act)


if road_wheels:
    def axle_x(names):
        xs = [w.location.x for w, _ in road_wheels if any(n in w.name for n in names)]
        return sum(xs) / len(xs) if xs else None

    front_x, rear_x = axle_x(('_fl', '_fr', '_f')), axle_x(('_rl', '_rr', '_r'))
    # Image left becomes -X, and most photos have the vehicle facing left
    fwd = (1.0 if front_x > rear_x else -1.0) if None not in (front_x, rear_x) else -1.0
    distance = 2 * math.pi * sum(r for _, r in road_wheels) / len(road_wheels)
    print(f"\nRoad wheels: forward is {'+' if fwd > 0 else '-'}X, "
          f"drive_forward travels {distance:.3f} (one turn)")

    for wheel, r in road_wheels:
        roll = fwd * distance / r
        add_clip(wheel, 'drive', [('rotation_euler', rotation_axis, [(1, 0.0), (END, roll)])])

    if not is_mechanical:
        movers = [o for o in bpy.data.objects if o.type == 'MESH']
        road   = {w.name: r for w, r in road_wheels}
        for clip, sign in (('drive_forward', 1.0), ('drive_backward', -1.0)):
            for obj in movers:
                x0 = obj.location.x
                curves = [('location', 0, [(1, x0), (END, x0 + sign * fwd * distance)])]
                if obj.name in road:
                    roll = sign * fwd * distance / road[obj.name]
                    curves.append(('rotation_euler', rotation_axis, [(1, 0.0), (END, roll)]))
                add_clip(obj, clip, curves)

        front = [w for w, _ in road_wheels if '_f' in w.name] or \
                [min(road_wheels, key=lambda wr: fwd * -wr[0].location.x)[0]]
        for clip, sign in (('steer_left', 1.0), ('steer_right', -1.0)):
            a = sign * STEER
            for wheel in front:
                add_clip(wheel, clip, [('rotation_euler', 2,
                                        [(1, 0.0), (21, a), (41, a), (END, 0.0)])],
                         interpolation='BEZIER')
        print(f"  clips: drive, drive_forward, drive_backward, steer_left, steer_right "
              f"(steering {[w.name for w in front]})")

# ── Export ────────────────────────────────────────────────────────────────────
bpy.ops.export_scene.gltf(
    filepath=output_path,
    export_format='GLB',
    export_animations=True,
    export_nla_strips=True,
    export_current_frame=False,
    export_force_sampling=True,
    export_image_format='JPEG',
    export_jpeg_quality=75,
)
print(f"\nExported: {output_path}")
