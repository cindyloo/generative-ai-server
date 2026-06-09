import bpy, sys, math, json, os
sys.path.insert(0, '/tmp/blender_packages')
import numpy as np

input_path    = sys.argv[sys.argv.index('--') + 1]
output_path   = sys.argv[sys.argv.index('--') + 2]
classify_json = sys.argv[sys.argv.index('--') + 3] \
                if len(sys.argv) > sys.argv.index('--') + 3 else None

# Load classify data
wheel_joints            = []

# ── Load classify data ────────────────────────────────────────────────────────
blender_wheel_centroids = {}
fallback_wheel_centroids = {}
classify_data           = None
joint_hints_by_name     = {}
is_mechanical           = False
reference_radius        = None

if classify_json and os.path.exists(classify_json):
    with open(classify_json) as f:
        classify_data = json.load(f)
    wheel_joints            = [j for j in classify_data.get('joint_hints', [])
                                if j.get('body_part') in ['wheel', 'gear']]
    fallback_wheel_centroids         = classify_data.get('wheel_centroids', {})
    blender_wheel_centroids = classify_data.get('blender_wheel_centroids', {})
    joint_hints_by_name     = {j['name']: j for j in classify_data.get('joint_hints', [])}
    is_mechanical           = classify_data.get('is_mechanical', False)
    reference_radius        = classify_data.get('reference_radius_normalized', None)
    print(f"Loaded {len(fallback_wheel_centroids)} Fallback centroids {fallback_wheel_centroids}")
    print(f"Loaded {len(blender_wheel_centroids)} Blender centroids {blender_wheel_centroids}")
    print(f"is_mechanical: {is_mechanical}  reference_radius: {reference_radius}")
else:
    print(f"classify_json not found: {classify_json}")



centroids = {}

# Check if our High-Precision mathematical keys exist in the payload
if 'blender_wheel_centroids' in classify_data:
    print("INFO: Blender loading high-precision mathematical centroids keys.")
    for name, data in classify_data['blender_wheel_centroids'].items():
        # Extracted directly from our precise Taubin Circle Fit!
        precise_xyz = data['centroid']
        
        # Keep internal script dictionary structured smoothly
        centroids[name] = {
            'centroid': [precise_xyz[0], precise_xyz[1], precise_xyz[2]],
            'radius': float(data['radius'])
        }
    
else:
    # FALLBACK: Keep old bounding box code just in case a file misses processing
    print("WARNING: High-precision keys missing, falling back to box approximations.")
    for hint in classify_data.get('joint_hints', []):
        name = hint['name']
        p = hint.get('position_normalized', {})
        cx = bmin[0] + p.get('x', 0.5) * brange[0]
        cy = bmin[1] + (1.0 - p.get('y', 0.5)) * brange[1]
        cz = bmin[2] + p.get('z', 0.5) * brange[2]
        centroids[name] = {
            'centroid': [cx, cy, cz],
            'radius': hint.get('wheel_radius_normalized', 0.1) * np.max(brange)
        }
        
blender_wheel_centroids = {
    name: (v['centroid'] if isinstance(v, dict) else v)
    for name, v in blender_wheel_centroids.items()
    if v is not None
}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_path)

mesh_objects  = [o for o in bpy.data.objects if o.type == 'MESH' and o.name != 'Mesh_0']

# Identify body as the object with worst match to any centroid
def best_centroid_dist(obj):
    verts_np = np.array([obj.matrix_world @ v.co for v in obj.data.vertices])
    actual_center = verts_np.mean(axis=0)
    if not blender_wheel_centroids:
        return 0.0
    return min(
        np.linalg.norm(np.array(pos) - actual_center)
        for pos in blender_wheel_centroids.values()
        if pos is not None
    )

wheel_objects = sorted(mesh_objects, key=lambda o: o.name)

print(f"\nWheel centroid verification (no override):")
for wheel in wheel_objects:
    verts_np = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    actual_center = verts_np.mean(axis=0)
    stored = blender_wheel_centroids.get(wheel.name)
    
    if stored is not None:
        print(f"  {wheel.name}: using Taubin pivot {stored}, mesh mean was ({actual_center[0]:.3f},{actual_center[1]:.3f},{actual_center[2]:.3f})")
        # DO NOT override — keep the precise Taubin centroid
    else:
        # Only fall back to mesh mean if no stored centroid exists
        blender_wheel_centroids[wheel.name] = actual_center.tolist()
        print(f"  {wheel.name}: no stored centroid, falling back to mesh mean")
        
print(f"\nAssigned Blender centroids (UPDATED):")
for name, pos in blender_wheel_centroids.items():
    if pos:
        c = pos['centroid'] if isinstance(pos, dict) else pos
        print(f"  {name}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
        

bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end   = 61

# ══════════════════════════════════════════════════════════════════════════════
# CLAUDE CONVENTION (NO DETECTION)
# ══════════════════════════════════════════════════════════════════════════════
# Claude ALWAYS uses:
#   x = left-right
#   y = front-rear
#   z = height

lr_idx = 0   # left-right axis
fr_idx = 1   # front-rear axis
h_idx = 2    # height axis

print(f"\nUsing Claude convention axes:")
print(f"  axis 0 (X) = left-right")
print(f"  axis 1 (Y) = front-rear")
print(f"  axis 2 (Z) = height")

# ── Build Claude pivot positions in Blender world space ───────────────────────
claude_pivots = []

if wheel_joints:
    all_verts = [(v.co.x, v.co.y, v.co.z)
                 for obj in mesh_objects for v in obj.data.vertices]
    x_min = min(v[0] for v in all_verts); x_max = max(v[0] for v in all_verts)
    y_min = min(v[1] for v in all_verts); y_max = max(v[1] for v in all_verts)
    z_min = min(v[2] for v in all_verts); z_max = max(v[2] for v in all_verts)
    
    bounds_min = np.array([x_min, y_min, z_min])
    bounds_max = np.array([x_max, y_max, z_max])
    bounds_range = bounds_max - bounds_min
    
    for wj in wheel_joints:
        p = wj['position_normalized']
        cx = p.get('x', 0.5)
        cy = p.get('y', 0.5)
        cz = p.get('z', 0.5)
        
        world = np.zeros(3)
        world[lr_idx] = bounds_min[lr_idx] + cx * bounds_range[lr_idx]
        world[fr_idx] = bounds_min[fr_idx] + cy * bounds_range[fr_idx]
        world[h_idx] = bounds_min[h_idx] + cz * bounds_range[h_idx]
        
        claude_pivots.append((wj['name'], world))

# ── Rotation axis: wheels rotate around the height axis ─────────────────────
rotation_axis = fr_idx
print(f"Detected axes: front_rear={fr_idx}, left_right={lr_idx}, height={h_idx}")
print(f"Rotation axis: {['X','Y','Z'][rotation_axis]} (index={rotation_axis})")

# ── Initialize centroid pools ──────────────────────────────────────────────────
remaining_blender = {k: v for k, v in blender_wheel_centroids.items() if v is not None}
remaining_claude  = list(claude_pivots)

print(f"Available Blender centroids: {list(remaining_blender.keys())}")
print(f"Available Claude pivots: {[name for name, _ in remaining_claude]}")

# ── Animate each wheel ─────────────────────────────────────────────────────────
for i, wheel in enumerate(wheel_objects):
    verts_np = np.array([wheel.matrix_world @ v.co for v in wheel.data.vertices])
    actual_center = verts_np.mean(axis=0)
    distances = np.linalg.norm(verts_np - actual_center, axis=1)
    
    mean_dist = distances.mean()
    std_dist = distances.std()
    far_verts = np.sum(distances > mean_dist * 2)
    
    print(f"{wheel.name} VERTEX DIAGNOSTICS:")
    print(f"  Total: {len(verts_np)}, Std dev: {std_dist:.3f}, Verts > 2x mean: {far_verts} ({far_verts/len(verts_np)*100:.1f}%)")
    
    if std_dist > mean_dist * 0.3 or far_verts > len(verts_np) * 0.1:
        print(f"  ⚠️ ASYMMETRIC WHEEL - tire and rim may be separate!")
        sorted_dists = np.sort(distances)
        print(f"  Quartiles: {sorted_dists[len(sorted_dists)//4]:.3f}, {sorted_dists[len(sorted_dists)//2]:.3f}, {sorted_dists[3*len(sorted_dists)//4]:.3f}")

    # DIAGNOSTIC: Check vertex distribution
    distances = np.linalg.norm(verts_np - actual_center, axis=1)
    
    print(f"\n{wheel.name} VERTEX DIAGNOSTICS:")
    print(f"  Total vertices: {len(verts_np)}")
    print(f"  Centroid: ({actual_center[0]:.3f}, {actual_center[1]:.3f}, {actual_center[2]:.3f})")
    print(f"  Min distance from center: {distances.min():.3f}")
    print(f"  Max distance from center: {distances.max():.3f}")
    print(f"  Mean distance from center: {distances.mean():.3f}")
    print(f"  Std dev of distances: {distances.std():.3f}")
    
    # Check if bimodal - two clusters of verts?
    sorted_dists = np.sort(distances)
    quarter = len(sorted_dists) // 4
    print(f"  Distance quartiles: {sorted_dists[quarter]:.3f}, {sorted_dists[quarter*2]:.3f}, {sorted_dists[quarter*3]:.3f}")
    
    # How many verts are far from center?
    far_verts = np.sum(distances > distances.mean() * 2)
    print(f"  Verts > 2x mean distance: {far_verts} ({far_verts/len(verts_np)*100:.1f}%)")
    
    pivot = blender_wheel_centroids.get(wheel.name)
    cx = 0.0
    cy = 0.0
    cz = 0.0
    if pivot is not None:
        cx, cy, cz = float(pivot[0]), float(pivot[1]), float(pivot[2])
    else:
        cx = verts_np[:, lr_idx].mean()
        cy = verts_np[:, fr_idx].mean()
        cz = verts_np[:, h_idx].mean()


    print(f"  {wheel.name}: hub center ({cx:.3f}, {cy:.3f}, {cz:.3f}) ")
   
    print(f"  {wheel.name}: Using {cx}, {cy}, {cz}")
    
    # Set origin to pivot
    bpy.context.view_layer.objects.active = wheel
    wheel.select_set(True)
    bpy.context.scene.cursor.location = (cx, cy, cz)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    wheel.select_set(False)
    bpy.context.view_layer.update()

    # ── Vertex diagnostics ────────────────────────────────────────────────────
    distances = np.linalg.norm(verts_np - actual_center, axis=1)
    mean_dist = distances.mean()
    far_verts = np.sum(distances > mean_dist * 2)
    print(f"  verts={len(verts_np)} far={far_verts} ({far_verts/len(verts_np)*100:.1f}%)")
    if distances.std() > mean_dist * 0.3 or far_verts > len(verts_np) * 0.1:
        print(f"  ⚠️ may contain non-target geometry")

    # ── Build animation ───────────────────────────────────────────────────────
    wheel.rotation_mode = 'XYZ'
    action = bpy.data.actions.new(name=f"wheel_{i}_rot")

    body_part = wheel.get('body_part')
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
        print(f"  hinge animation: axis={hinge_axis} range={hinge_range}°")

    elif body_part == 'gear' and is_mechanical and ref_r_mesh is not None:
        r_norm      = hint.get('wheel_radius_normalized', 0.15)
        rot_dir     = hint.get('rotation_direction', 1)
        this_r_mesh = r_norm * full_y_range
        speed_ratio = ref_r_mesh / max(this_r_mesh, 0.001)
        total_angle = math.pi * 2 * speed_ratio * rot_dir
        fc = action.fcurves.new(data_path='rotation_euler', index=rotation_axis)
        fc.keyframe_points.add(2)
        fc.keyframe_points[0].co = (1,  0.0)
        fc.keyframe_points[1].co = (61, total_angle)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        print(f"  gear: r_norm={r_norm:.3f} dir={rot_dir:+d} "
              f"speed_ratio={speed_ratio:.2f} angle={math.degrees(total_angle):.1f}°")

    else:
        # Vehicle wheel or gear without speed ratio: one full rotation
        fc = action.fcurves.new(data_path='rotation_euler', index=rotation_axis)
        fc.keyframe_points.add(2)
        fc.keyframe_points[0].co = (1,  0.0)
        fc.keyframe_points[1].co = (61, math.pi * 2)
        for kp in fc.keyframe_points:
            kp.interpolation = 'LINEAR'
        print(f"  wheel/gear: one full rotation around axis {rotation_axis}")

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
