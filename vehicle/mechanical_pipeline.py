"""
mechanical_pipeline.py

Standalone script invoked by seg_server.py via subprocess:

    python mechanical_pipeline.py \
        <classify_id> <glb_path> <classify_json> \
        <results_dir> <vehicle_scripts_dir> [mask_dir]

seg_server.py runs SAM2 in-process and passes mask_dir as the last arg.
This script just runs the Blender pipeline with mechanical=True.
"""

import sys
import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    stream=sys.stdout
)
log = logging.getLogger('mechanical_pipeline')


def _add_to_path(vehicle_dir: str) -> None:
    parent = os.path.dirname(vehicle_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def run(classify_id: str, glb_path: str, classify_json: str,
        results_dir: str, vehicle_dir: str,
        mask_dir: str | None) -> str:

    _add_to_path(vehicle_dir)
    from blender_pipeline import run_blender_pipeline

    with open(classify_json) as f:
        classify_data = json.load(f)

    joint_hints = classify_data.get('joint_hints', [])
    log.info(f"Mechanical pipeline: {len(joint_hints)} joint hints")
    for jh in joint_hints:
        p = jh.get('position_normalized', {})
        log.info(f"  {jh['name']} ({jh.get('body_part')}): "
                 f"x={p.get('x', 0):.2f} y={p.get('y', 0):.2f} z={p.get('z', 0):.2f} "
                 f"r={jh.get('wheel_radius_normalized', 'n/a')}")

    # ── Compute reference radius (largest gear = 1x speed reference) ─────────
    gear_hints = [j for j in joint_hints if j.get('body_part') == 'gear']
    if gear_hints:
        classify_data['reference_radius_normalized'] = max(
            j.get('wheel_radius_normalized', 0.15) for j in gear_hints
        )
        log.info(f"Reference radius: {classify_data['reference_radius_normalized']:.3f}")

    # ── Detect image_view ─────────────────────────────────────────────────────
    if len(gear_hints) >= 2:
        x_spread = (max(j['position_normalized']['x'] for j in gear_hints) -
                    min(j['position_normalized']['x'] for j in gear_hints))
        z_spread = (max(j['position_normalized']['z'] for j in gear_hints) -
                    min(j['position_normalized']['z'] for j in gear_hints))
        classify_data['image_view'] = 'side' if x_spread > z_spread else 'front'

    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)

    log.info(f"mask_dir: {mask_dir}")

    # ── Run shared Blender pipeline ───────────────────────────────────────────
    rigged_path = run_blender_pipeline(
        classify_id, glb_path, classify_json,
        results_dir, vehicle_dir, mask_dir,
        mechanical=True,
    )
    log.info(f"Mechanical pipeline complete: {rigged_path}")
    return rigged_path


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print("Usage: mechanical_pipeline.py "
              "<classify_id> <glb_path> <classify_json> "
              "<results_dir> <vehicle_scripts_dir> [mask_dir]")
        sys.exit(1)

    rigged = run(
        classify_id   = sys.argv[1],
        glb_path      = sys.argv[2],
        classify_json = sys.argv[3],
        results_dir   = sys.argv[4],
        vehicle_dir   = sys.argv[5],
        mask_dir      = sys.argv[6] if len(sys.argv) > 6 else None,
    )
    print(rigged)
