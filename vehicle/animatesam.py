import bpy, sys, math, json, os
sys.path.insert(0, '/tmp/blender_packages')
import numpy as np

input_path    = sys.argv[sys.argv.index('--') + 1]
output_path   = sys.argv[sys.argv.index('--') + 2]
classify_json = sys.argv[sys.argv.index('--') + 3] \
                if len(sys.argv) > sys.argv.index('--') + 3 else None

# Load Gemini wheel positions if available
wheel_joints     = []
gemini_positions = None


if classify_json and os.path.exists(classify_json):
    with open(classify_json) as f:
        classify_data = json.load(f)
    wheel_joints = [j for j in classify_data.get('joint_hints', [])
                    if j.get('body_part') == 'wheel']
    print(f"classify_json: {classify_json}")
    print(f"Loaded {len(wheel_joints)} wheel joints")
    print(f"All body_parts: {[j.get('body_part') for j in classify_data.get('joint_hints', [])]}")
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

# Build Gemini pivot lookup from world positions
if wheel_joints:
    x_min = min(v.co.x for obj in mesh_objects for v in obj.data.vertices)
    x_max = max(v.co.x for obj in mesh_objects for v in obj.data.vertices)
    y_min = min(v.co.y for obj in mesh_objects for v in obj.data.vertices)
    y_max = max(v.co.y for obj in mesh_objects for v in obj.data.vertices)
    z_min = min(v.co.z for obj in mesh_objects for v in obj.data.vertices)
    z_max = max(v.co.z for obj in mesh_objects for v in obj.data.vertices)

    gemini_positions = np.array([
        [
            x_min + wj['position_normalized']['x'] * (x_max - x_min),
            y_min + wj['position_normalized']['z'] * (y_max - y_min),
            z_min + wj['position_normalized']['y'] * (z_max - z_min),
        ]
        for wj in wheel_joints
    ])
    print(f"Gemini wheel positions: {gemini_positions}")

for i, wheel in enumerate(wheel_objects):
    verts_np  = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    median_pt = np.median(verts_np, axis=0)

    if gemini_positions is not None:
        dists       = np.linalg.norm(gemini_positions - median_pt, axis=1)
        nearest     = gemini_positions[np.argmin(dists)]
        cx, cy, cz  = float(nearest[0]), float(nearest[1]), float(nearest[2])
        print(f"  {wheel.name}: Gemini pivot=({cx:.3f},{cy:.3f},{cz:.3f})")
    else:
        # Bounding box center — true geometric center regardless of
        # vertex distribution. For a wheel (symmetric cylinder) this
        # is the axle center, which is what we want to rotate around.
        bbox_min = verts_np.min(axis=0)
        bbox_max = verts_np.max(axis=0)
        cx = float((bbox_min[0] + bbox_max[0]) / 2)
        cy = float((bbox_min[1] + bbox_max[1]) / 2)
        cz = float((bbox_min[2] + bbox_max[2]) / 2)
        print(f"  {wheel.name}: bbox pivot=({cx:.3f},{cy:.3f},{cz:.3f})")

    # ── Set origin to pivot ───────────────────────────────────────
    bpy.context.view_layer.objects.active = wheel
    wheel.select_set(True)
    bpy.context.scene.cursor.location = (cx, cy, cz)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    wheel.select_set(False)

    # ── Create rotation animation ─────────────────────────────────
    wheel.rotation_mode = 'XYZ'
    action = bpy.data.actions.new(name=f"wheel_{i}_rot")
    fc     = action.fcurves.new(data_path='rotation_euler', index=1)
    fc.keyframe_points.add(2)
    fc.keyframe_points[0].co = (1,  0.0)
    fc.keyframe_points[1].co = (61, math.pi * 2)
    
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

    # ── Push to NLA strip named "drive" ───────────────────────────
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
