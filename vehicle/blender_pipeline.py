"""
blender_pipeline.py

Shared script pipeline loop used by both run_vehicle_pipeline.py and
run_mechanical_pipeline.py.  Import and call run_blender_pipeline().
"""

import os
import sys
import json
import struct
import subprocess
import logging

log = logging.getLogger(__name__)


def _blender_bin() -> str:
    return os.environ.get(
        'BLENDER_PATH',
        '/Applications/Blender.app/Contents/MacOS/blender'
    )


def extract_texture(glb_path: str, texture_path: str) -> None:
    """Extract the first texture from a GLB file."""
    try:
        with open(glb_path, 'rb') as f:
            f.read(12)
            json_len = struct.unpack('<I', f.read(4))[0]; f.read(4)
            j        = json.loads(f.read(json_len))
            bin_len  = struct.unpack('<I', f.read(4))[0]; f.read(4)
            binary   = f.read(bin_len)
        for img_data in j.get('images', []):
            bv   = j['bufferViews'][img_data['bufferView']]
            data = binary[bv['byteOffset']:bv['byteOffset'] + bv['byteLength']]
            with open(texture_path, 'wb') as tf:
                tf.write(data)
            break
    except Exception as e:
        log.warning(f"Texture extraction failed (non-fatal): {e}")


def run_blender_pipeline(
    classify_id: str,
    glb_path: str,
    classify_json: str,
    results_dir: str,
    vehicle_dir: str,
    mask_dir_valid: str | None,
    mechanical: bool = False,
) -> str:
    """
    Run the four-script Blender pipeline shared by vehicle and mechanical rigs.

    mechanical=True adds --mechanical to animatesam.py so it uses gear speed
    ratios and hinge oscillation instead of fixed wheel rotation.

    Returns the path to the final rigged GLB.
    """
    
    print("Blender pipeline...")
    separated_path = os.path.join(results_dir, f"{classify_id}_separated.glb")
    animated_path  = os.path.join(results_dir, f"{classify_id}_animated.glb")
    rigged_path    = os.path.join(results_dir, f"{classify_id}_rigged.glb")
    tire_verts     = os.path.join(results_dir, f"{classify_id}_tire_verts.json")
    texture_path   = os.path.join(results_dir, f"{classify_id}_texture.png")

    extract_texture(glb_path, texture_path)

    for script_name, input_path, out_path, runner in [
        ('find_tire_verts.py',  glb_path,       tire_verts,     'python'),
        ('classify_wheels.py',  glb_path,        separated_path, 'blender'),
        ('animatesam.py',       separated_path,  animated_path,  'blender'),
        ('merge_animations.py', animated_path,   rigged_path,    'python'),
    ]:
        script = os.path.join(vehicle_dir, script_name)
        log.info(f"{script}")
        if runner == 'blender':
            if script_name == 'classify_wheels.py':
                extra = [tire_verts]
                if mask_dir_valid and os.path.isdir(mask_dir_valid):
                    extra.append(mask_dir_valid)
                    log.info(f"classify_wheels: SAM2 mask_dir={mask_dir_valid}")
                else:
                    log.info("classify_wheels: no mask_dir, radius fallback")
            elif script_name == 'animatesam.py' and mechanical:
                extra = ['--mechanical']
            else:
                extra = []
            cmd = [_blender_bin(), '--background', '--factory-startup',
                   '--python', script, '--',
                   input_path, out_path, classify_json] + extra

        else:  # python
            if script_name == 'find_tire_verts.py':
                args = [glb_path, classify_json, tire_verts, texture_path]
                if mask_dir_valid and os.path.isdir(mask_dir_valid):
                    args.append(os.path.abspath(mask_dir_valid))
                    log.info(f"find_tire_verts: SAM2 mask_dir={os.path.abspath(mask_dir_valid)}")
                else:
                    log.info("find_tire_verts: geometry-only (no mask_dir)")
            else:
                args = [animated_path, rigged_path]
            cmd = [sys.executable, script] + args

        log.info(f"Running: {script_name}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        log.info(result.stdout)
        log.info(result.stderr)

        # Inject centroids into classify_json after find_tire_verts
        if script_name == 'find_tire_verts.py' and os.path.exists(tire_verts):
            try:
                centroids_path = tire_verts.replace('.json', '_centroids.json')
                if os.path.exists(centroids_path):
                    with open(centroids_path) as _f:
                        _centroids = json.load(_f)
                    with open(classify_json, 'r') as _f:
                        _cdata = json.load(_f)
                    _cdata['wheel_centroids'] = _centroids
                    with open(classify_json, 'w') as _f:
                        json.dump(_cdata, _f, indent=2)
                    log.info(f"Centroids injected: {len(_centroids)}")
                else:
                    log.warning("Centroids file not found — pivot will use bbox fallback")
            except Exception as e:
                log.warning(f"Centroid injection failed (non-fatal): {e}")

        log.info(f"{script_name} returncode: {result.returncode}")
        if result.stderr:
            log.info(f"{script_name} stderr: {result.stderr[-300:]}")

        if not os.path.exists(out_path):
            raise RuntimeError(f"{script_name} failed: {result.stderr[-200:]}")

    return rigged_path
