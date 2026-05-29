"""
run_vehicle_pipeline.py

Standalone script invoked by seg_server.py via subprocess:

    python run_vehicle_pipeline.py \
        <classify_id> <glb_path> <classify_json> \
        <results_dir> <vehicle_scripts_dir> [active_image_path]
"""

import sys
import os
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger('vehicle_pipeline')


def _add_to_path(vehicle_dir: str) -> None:
    parent = os.path.dirname(vehicle_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def run(classify_id: str, glb_path: str, classify_json: str,
        results_dir: str, vehicle_dir: str,
        mask_dir: str | None) -> str:

    _add_to_path(vehicle_dir)
    import utils as utils_mod
    from blender_pipeline import run_blender_pipeline

    with open(classify_json) as f:
        classify_data = json.load(f)

    joint_hints = classify_data.get('joint_hints', [])
    log.info(f"Vehicle pipeline: {len(joint_hints)} joint hints")
    for jh in joint_hints:
        p = jh.get('position_normalized', {})
        log.info(f"  {jh['name']} ({jh.get('body_part')}): "
                 f"x={p.get('x', 0):.2f} y={p.get('y', 0):.2f} z={p.get('z', 0):.2f}")

    # ── Run shared pipeline ───────────────────────────────────────────────────
    rigged_path = run_blender_pipeline(
        classify_id, glb_path, classify_json,
        results_dir, vehicle_dir, mask_dir,
        mechanical=False,
    )
    log.info(f"Vehicle pipeline complete: {rigged_path}")
    return rigged_path


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: run_vehicle_pipeline.py "
              "<classify_id> <glb_path> <classify_json> "
              "<results_dir> <vehicle_scripts_dir> [mask_dir]")
        sys.exit(1)

    rigged = run(
        classify_id       = sys.argv[1],
        glb_path          = sys.argv[2],
        classify_json     = sys.argv[3],
        results_dir       = sys.argv[4],
        vehicle_dir       = sys.argv[5],
        mask_dir = sys.argv[6] if len(sys.argv) > 6 else None,
    )
    print(rigged)
