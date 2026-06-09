import os
import sys
import json
import numpy as np
import trimesh
from PIL import Image

# ── 1. Pipeline Arguments ─────────────────────────────────────────────────────
glb_path        = sys.argv[1]
classify_json   = sys.argv[2]
tire_verts_path = sys.argv[3]
texture_path    = sys.argv[4]

with open(classify_json) as f:
    classify_data = json.load(f)

joint_hints = classify_data.get('joint_hints', [])
wheel_hints = [h for h in joint_hints if h.get('body_part') in ['wheel', 'gear', 'chainring']]

scene = trimesh.load(glb_path)
if isinstance(scene, trimesh.Scene):
    mesh = scene.to_geometry()
else:
    mesh = scene

verts = np.array(mesh.vertices)
n_verts = len(verts)
print(f"Loaded mesh with {n_verts} vertices.")

# ── 2. Native Color Profile Mapping ───────────────────────────────────────────
wheel_colors_data = classify_data.get('wheel_colors_rgb', [])
body_colors_data = classify_data.get('body_colors_rgb', [])
has_texture = os.path.exists(texture_path) and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None

if has_texture and wheel_colors_data:
    try:
        uvs = mesh.visual.uv
        img = Image.open(texture_path).convert('RGB')
        iw, ih = img.size
        img_arr = np.array(img)

        px_u = (uvs[:, 0] * iw).astype(int) % iw
        px_v = ((1.0 - uvs[:, 1]) * ih).astype(int) % ih
        vert_colors = img_arr[px_v, px_u]

        # Normalize wheel color arrays
        if isinstance(wheel_colors_data, dict):
            wheel_colors = np.array([[int(c['color'][i]*255) for i in range(3)] for c in wheel_colors_data])
        elif isinstance(wheel_colors_data, (list, np.ndarray)):
            if len(wheel_colors_data) > 0 and max(np.array(wheel_colors_data).flatten()) <= 1.0:
                wheel_colors = np.array([[int(c[i]*255) for i in range(3)] for c in wheel_colors_data])
            else:
                wheel_colors = np.array(wheel_colors_data, dtype=int)
        
        # Normalize body color arrays for reference
        if body_colors_data:
            if max(np.array(body_colors_data).flatten()) <= 1.0:
                body_colors = np.array([[int(c[i]*255) for i in range(3)] for c in body_colors_data])
            else:
                body_colors = np.array(body_colors_data, dtype=int)
    except Exception as e:
        has_texture = False
        print(f"Texture extraction failed: {e}. Falling back to clean geometry math.")

# ── 3. High-Precision Mathematical Math Processing ────────────────────────────
output_centroids = {}

for hint in wheel_hints:
    name = hint['name']
    p = hint.get('position_normalized', {})
    bmin, bmax = verts.min(axis=0), verts.max(axis=0)
    brange = bmax - bmin
    
    hint_center = bmin + np.array([p.get('x', 0.5), 1.0 - p.get('y', 0.5), p.get('z', 0.5)]) * brange
    hint_dists = np.linalg.norm(verts - hint_center, axis=1)
    spatial_threshold = np.max(brange) * 0.25
    
    # Generate the local base color mask if texture space is valid
    local_color_mask = np.ones(n_verts, dtype=bool)
    if has_texture:
        # Step A: Find matching wheel color vertices
        match_mask = np.zeros(n_verts, dtype=bool)
        for wc in wheel_colors:
            dists = np.linalg.norm(vert_colors - wc, axis=1)
            match_mask |= (dists < 60)
            
        # Step B: Apply body exclusion ONLY for thin bike chainrings, NEVER for wheels or gears
        if "chainring" in name and body_colors_data:
            for bc in body_colors:
                body_dists = np.linalg.norm(vert_colors - bc, axis=1)
                match_mask &= (body_dists >= 80)
        
        if match_mask.sum() > 50:
            local_color_mask = match_mask

    # Combine spatial proximity with our local color mask
    local_mask = local_color_mask & (hint_dists < spatial_threshold)
    component_verts = verts[local_mask]
    
    if len(component_verts) < 10:
        # Final fallback: if color matching stripped everything, fall back to loose geometry profile
        component_verts = verts[hint_dists < spatial_threshold]
        if len(component_verts) < 10:
            continue

    # Adaptive density clustering rules
    if "chainring" in name:
        local_med = np.median(component_verts, axis=0)
        local_dists = np.linalg.norm(component_verts - local_med, axis=1)
        component_verts = component_verts[local_dists < np.percentile(local_dists, 60)]
    elif "wheel" in name or "gear" in name:
        local_med = np.median(component_verts, axis=0)
        local_dists = np.linalg.norm(component_verts - local_med, axis=1)
        component_verts = component_verts[local_dists < np.percentile(local_dists, 99)] # Keep full thickness

    # Run PCA Normal Vector Extraction
       # Run PCA Normal Vector Extraction
    mean = np.mean(component_verts, axis=0)
    centered = component_verts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    rotation_axis = eigenvectors[:, np.argmin(eigenvalues)]
    u_axis = eigenvectors[:, np.argmax(eigenvalues)]
    v_axis = np.cross(rotation_axis, u_axis)

    # ── NEW FILTER: Axial Thickness Cut (Removes Undercarriage/Axle Bleed) ──
    # Project vertices along the rotation axis line to measure their lateral depth
    depths = np.dot(centered, rotation_axis)
    
    # Calculate the thickness spread (standard deviation of depth)
    depth_std = np.std(depths)
    
    # If the selection bleeds too deep along the axis (common with matching chassis/axles),
    # slice it off at 2.0 standard deviations from the wheel center plane.
    # This keeps the full tire thickness but cuts off the undercarriage extending inward.
    if "wheel" in name or "gear" in name:
        axial_mask = np.abs(depths) < (depth_std * 2.0)
        component_verts = component_verts[axial_mask]
        
        # Re-center data after trimming the stray undercarriage parts
        mean = np.mean(component_verts, axis=0)
        centered = component_verts - mean
    # ─────────────────────────────────────────────────────────────────────────

    pts_2d = np.column_stack((np.dot(centered, u_axis), np.dot(centered, v_axis)))

    # Taubin Circle Fitting Loop
    # ── 4. Precise Pivot & Exact Radial Boundary Calculation ──────────────────
    try:
        X = pts_2d[:, 0]
        Y = pts_2d[:, 1]
        Z = X**2 + Y**2
        M_z = np.column_stack((Z, X, Y, np.ones(len(X))))
        M = np.dot(M_z.T, M_z) / len(X)
        
        w, v = np.linalg.eig(M)
        imin = np.argmin(np.abs(w))
        A, B, C, D = v[:, imin]
        
        center_2d_u = -B / (2 * A)
        center_2d_v = -C / (2 * A)
        
        # Calculate the high-precision mathematical pivot center
        precise_pivot = mean + (center_2d_u * u_axis) + (center_2d_v * v_axis)

        # ── Cylindrical re-filter using component_verts (not full verts) ──────────
        # All arrays must be scoped to component_verts to avoid size mismatch
        relative_positions = component_verts - precise_pivot                          # (N_component, 3)
        axial_depths = np.dot(relative_positions, rotation_axis)                      # (N_component,)
        radial_vecs = relative_positions - np.outer(axial_depths, rotation_axis)      # (N_component, 3)
        radial_dists = np.linalg.norm(radial_vecs, axis=1)                            # (N_component,)

        # Use 99th percentile of component verts as the true radius
        calculated_radius = float(np.percentile(radial_dists, 99))

        # Re-filter component_verts to the tight cylinder
        if "wheel" in name or "gear" in name:
            tight_mask = (
                (radial_dists <= calculated_radius * 1.05) &
                (np.abs(axial_depths) <= calculated_radius * 0.6)
            )
            component_verts = component_verts[tight_mask]

            # Recompute final radius from the cleaned verts
            relative_positions = component_verts - precise_pivot
            axial_depths = np.dot(relative_positions, rotation_axis)
            radial_dists = np.linalg.norm(
                relative_positions - np.outer(axial_depths, rotation_axis), axis=1
            )
            calculated_radius = float(np.percentile(radial_dists, 99))        # Hard limits to prevent runaway math on specific components
        elif "chainring" in name:
            calculated_radius = min(calculated_radius, np.max(brange) * 0.08)
        
            
    except Exception as e:
        print(f"Mathematical fit failed for {name} ({e}), using absolute fallbacks.")
        precise_pivot = mean
        calculated_radius = 0.25 if "wheel" in name else 0.05


    precise_pivot_list = [float(x) for x in precise_pivot]
    rotation_axis_list = [float(x) for x in rotation_axis]
    
    output_centroids[name] = {
        'centroid': precise_pivot_list,
        'radius': float(calculated_radius),
        'name': name,
        'axis': rotation_axis_list
    }

# ── 4. Pipeline Export Handshakes ─────────────────────────────────────────────
classify_data['wheel_centroids'] = output_centroids
with open(classify_json, 'w') as f:
    json.dump(classify_data, f, indent=4)

# Build standard and alternative centroid text paths explicitly
centroids_path = tire_verts_path.replace('.json', '_centroids.json') if tire_verts_path.endswith('.json') else os.path.splitext(tire_verts_path)[0] + '_centroids.json'
with open(centroids_path, 'w') as f:
    json.dump(output_centroids, f, indent=4)

print(f"INFO:seg_server:High-Precision Centroids Exported to {centroids_path}")
print(f"INFO:seg_server:Wheel centroids injected from find_tire_verts: {json.dumps(output_centroids)}")
