import bpy, sys, math, json, os
sys.path.insert(0, '/tmp/blender_packages')
import numpy as np

input_path    = sys.argv[sys.argv.index('--') + 1]
output_path   = sys.argv[sys.argv.index('--') + 2]
classify_json = sys.argv[sys.argv.index('--') + 3] \
                if len(sys.argv) > sys.argv.index('--') + 3 else None

# Load classify data
wheel_joints           = []
wheel_centroids        = {}
blender_wheel_centroids = {}
classify_data          = None

if classify_json and os.path.exists(classify_json):
    with open(classify_json) as f:
        classify_data = json.load(f)
    wheel_joints            = [j for j in classify_data.get('joint_hints', [])
                                if j.get('body_part') == 'wheel']
    wheel_centroids         = classify_data.get('wheel_centroids', {})
    blender_wheel_centroids = classify_data.get('blender_wheel_centroids', {})
    print(f"classify_json: {classify_json}")
    print(f"Loaded {len(wheel_joints)} wheel joints")
    print(f"Loaded {len(blender_wheel_centroids)} Blender centroids")
else:
    print(f"classify_json not found: {classify_json}")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_path)

mesh_objects  = [o for o in bpy.data.objects if o.type == 'MESH']
body_obj      = max(mesh_objects, key=lambda o: len(o.data.vertices))
wheel_objects = [o for o in mesh_objects if o != body_obj]

print(f"Body: {body_obj.name}")
print(f"Wheels: {[o.name for o in wheel_objects]}")

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = 61

# ── Build Claude pivot positions in Blender world space ───────────────────────
# Claude joint positions use y-up z-depth normalized convention.
# Blender imports GLB with Y→Z conversion, so after import:
#   Blender X = mesh X (left/right)   ← Claude x
#   Blender Y = mesh Y (front/rear)   ← Claude y
#   Blender Z = mesh Z (height)       ← Claude z
# Denormalize using the full scene bounding box.
claude_pivots = []   # list of (name, np.array([bx, by, bz]))

if wheel_joints:
    all_verts = [(v.co.x, v.co.y, v.co.z)
                 for obj in mesh_objects for v in obj.data.vertices]
    x_min = min(v[0] for v in all_verts); x_max = max(v[0] for v in all_verts)
    y_min = min(v[1] for v in all_verts); y_max = max(v[1] for v in all_verts)
    z_min = min(v[2] for v in all_verts); z_max = max(v[2] for v in all_verts)

    for wj in wheel_joints:
        p  = wj['position_normalized']
        bx = x_min + p.get('x', 0.5) * (x_max - x_min)
        by = y_min + p.get('y', 0.5) * (y_max - y_min)
        bz = z_min + p.get('z', 0.5) * (z_max - z_min)
        claude_pivots.append((wj['name'], np.array([bx, by, bz])))

    print(f"Claude wheel pivots (Blender space):")
    for name, pos in claude_pivots:
        print(f"  {name}: ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

# Centroid pools — pop matched ones to prevent duplicates
remaining_blender = {k: v for k, v in blender_wheel_centroids.items() if v is not None}
remaining_claude  = list(claude_pivots)

# ── Animate each wheel ────────────────────────────────────────────────────────
for i, wheel in enumerate(wheel_objects):
    verts_np  = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    median_pt = np.median(verts_np, axis=0)

    if remaining_blender:
        # PRIMARY: Blender-space centroids from classify_wheels.py
        # Already in Blender world space — no conversion needed
        best_name, best_dist, best_pos = None, float('inf'), None
        for name, pos in remaining_blender.items():
            dist = np.linalg.norm(np.array(pos) - median_pt)
            if dist < best_dist:
                best_dist = dist
                best_name = name
                best_pos  = pos
        remaining_blender.pop(best_name)
        cx, cy, cz = float(best_pos[0]), float(best_pos[1]), float(best_pos[2])
        print(f"  {wheel.name}: Blender centroid ({best_name})=({cx:.3f},{cy:.3f},{cz:.3f})")

    elif remaining_claude:
        # SECONDARY: Claude joint positions in Blender space
        best_name, best_dist, best_pos = None, float('inf'), None
        for name, pos in remaining_claude:
            dist = np.linalg.norm(pos - median_pt)
            if dist < best_dist:
                best_dist = dist
                best_name = name
                best_pos  = pos
        remaining_claude = [(n, p) for n, p in remaining_claude if n != best_name]
        cx, cy, cz = float(best_pos[0]), float(best_pos[1]), float(best_pos[2])
        print(f"  {wheel.name}: Claude pivot ({best_name})=({cx:.3f},{cy:.3f},{cz:.3f})")

    else:
        # LAST RESORT: vertex mean in Blender space
        cx = float(verts_np[:, 0].mean())
        cy = float(verts_np[:, 1].mean())
        cz = float(verts_np[:, 2].mean())
        print(f"  {wheel.name}: vertex mean pivot=({cx:.3f},{cy:.3f},{cz:.3f})")

    # ── Set origin to pivot ───────────────────────────────────────────────────
    bpy.context.view_layer.objects.active = wheel
    wheel.select_set(True)
    bpy.context.scene.cursor.location = (cx, cy, cz)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    wheel.select_set(False)

    # ── Create rotation animation ─────────────────────────────────────────────
    wheel.rotation_mode = 'XYZ'
    action = bpy.data.actions.new(name=f"wheel_{i}_rot")
    fc     = action.fcurves.new(data_path='rotation_euler', index=1)
    fc.keyframe_points.add(2)
    fc.keyframe_points[0].co = (1,  0.0)
    fc.keyframe_points[1].co = (61, math.pi * 2)

    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

    # ── Push to NLA strip named "drive" ───────────────────────────────────────
    wheel.animation_data_create()
    wheel.animation_data.action = action
    track      = wheel.animation_data.nla_tracks.new()
    track.name = "drive"
    strip      = track.strips.new("drive", 1, action)
    wheel.animation_data.action = None

    print(f"  {wheel.name}: NLA strip 'drive' created")

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
print(f"Exported: {output_path}")
