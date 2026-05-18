"""
utils.py

Shared helpers for seg_server.py.

Vision prompts
--------------
  _build_classify_prompt(tag_ctx)
      Lightweight — identify object_type, category, needs_augmentation only.
      Called by /classify. No joint placement.

  _build_joints_prompt(object_type, category, n_joints)
      Focused joint placement — receives known object_type/category as context.
      Called by /joints. No identification, no augmentation assessment.

  _build_vehicle_prompt()
      Combined identify + joint placement for vehicles (unchanged — vehicles
      are a special case where the joint schema is rigid enough that splitting
      adds no value).

The old _build_animal_prompt() is removed. classify_with_vision() now calls
_build_classify_prompt(), and classify_joints_with_vision() calls
_build_joints_prompt() from inside seg_server.py.
"""

import os
import sys
import io
import base64
import logging
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)

VEHICLE_KEYWORDS = {'car', 'truck', 'vehicle', 'bus', 'bike', 'motorcycle',
                    'van', 'auto', 'wheels'}

HUMANOID_KEYWORDS = {'human', 'person', 'character', 'humanoid', 'man',
                     'woman', 'boy', 'girl', 'robot', 'alien', 'zombie',
                     'totoro', 'creature', 'figure', 'monster'}
                     
ANIMAL_KEYWORDS = {'dog', 'cat', 'bird', 'fish', 'shark', 'whale', 'wolf', 'animal','bug', 'insect', 'butterfly'}


# ── Image helpers ─────────────────────────────────────────────────────────────

def resize_if_needed(img: Image.Image, max_size: int = 1024) -> Image.Image:
    if max(img.size) > max_size:
        ratio    = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img      = img.resize(new_size, Image.LANCZOS)
        log.info(f"Resized to {new_size}")
    return img


def img_to_b64(image: Image.Image, fmt: str = 'PNG') -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def detect_mime_type(img_bytes: bytes) -> str:
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    elif img_bytes[:3] == b'\xff\xd8\xff':
        return 'image/jpeg'
    elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        return 'image/webp'
    elif img_bytes[:6] in (b'GIF87a', b'GIF89a'):
        return 'image/gif'
    else:
        return 'image/png'


# ── Keyframe injection ────────────────────────────────────────────────────────
#
# Procedural walk animations injected into the skeleton after joint placement.
# This replaces asking the vision model to generate keyframes (which it does
# inconsistently). inject_keyframes() is called inside run_rig_pipeline()
# after the skeleton is assembled, before run_blender_rig().

# axis, phase (+1 normal / -1 inverted), base amplitude in radians
WALK_KEYFRAMES = {
    "root":      None,
    "pelvis":    ("z",  1,  0.05),
    "spine":     ("y", -1,  0.06),
    "chest":     ("y", -1,  0.05),
    "neck":      ("z",  1,  0.02),
    "head":      ("z", -1,  0.02),
    "shoulder":  ("x",  1,  0.35),
    "elbow":     ("x",  1,  0.20),
    "hand":      ("x",  1,  0.10),
    "hip":       ("x",  1,  0.20),
    "leg":       ("x",  1,  0.50),
    "foot":      ("x",  1,  0.10),
    "wing_base": ("z",  1,  0.30),
    "wing_mid":  ("z",  1,  0.20),
    "wing_tip":  ("z",  1,  0.10),
    "axle":      None,
    "body":      None,
}

# Right-side joints get phase flipped so left/right limbs oppose each other
RIGHT_SIDE_FLIP = {"shoulder", "elbow", "hand", "hip", "leg", "foot",
                   "wing_base", "wing_mid", "wing_tip"}


def build_walk_keyframes(body_part: str, joint_name: str,
                         bone_length: float = None) -> list:
    """
    Return an animations list for a joint given its body_part label.
    bone_length: world-space distance to nearest child, used to scale amplitude.
    """
    body_part_lower = (body_part or '').lower()

    if body_part_lower == "wheel":
        return [{
            "clip":      "drive",
            "property":  "rotation_euler",
            "axis":      "x",
            "keyframes": [[0, 0.0], [30, 3.14159], [60, 6.28318]],
            "loop":      True,
        }]

    params = WALK_KEYFRAMES.get(body_part_lower)
    if not params:
        return []

    axis, phase, base_amp = params
    is_right = "right" in joint_name.lower()
    if is_right and body_part_lower in RIGHT_SIDE_FLIP:
        phase *= -1

    amp = base_amp
    if bone_length is not None:
        REFERENCE_BONE = 0.4
        amp = base_amp * min(1.5, max(0.5, REFERENCE_BONE / bone_length))

    kf = [
        [1,  0.0],
        [15, round( phase * amp, 4)],
        [30, 0.0],
        [45, round(-phase * amp, 4)],
        [60, 0.0],
    ]
    return [{
        "clip":      "walk",
        "property":  "rotation_euler",
        "axis":      axis,
        "keyframes": kf,
        "loop":      True,
    }]


def inject_keyframes(skel: dict) -> dict:
    """
    Walk every joint in the skeleton and inject procedural walk keyframes
    based on its body_part label. Called after skel is assembled, before
    run_blender_rig().
    """
    positions = {j['id']: np.array(j['position']) for j in skel['joints']}
    children  = {}
    for bone in skel['bones']:
        children.setdefault(bone['parent'], []).append(bone['child'])

    for joint in skel['joints']:
        hint      = joint.get('hint') or {}
        body_part = hint.get('body_part', '')
        name      = joint.get('name', '')

        bone_length = None
        child_ids   = children.get(joint['id'], [])
        if child_ids:
            child_pos = positions.get(child_ids[0])
            if child_pos is not None:
                bone_length = float(np.linalg.norm(
                    np.array(joint['position']) - child_pos
                ))

        animations = build_walk_keyframes(body_part, name, bone_length)
        if joint.get('hint') is None:
            joint['hint'] = {}
        joint['hint']['animations'] = animations

    return skel


# ══════════════════════════════════════════════════════════════════════════════
# Vision prompts
# ══════════════════════════════════════════════════════════════════════════════

def _build_classify_prompt(tag_ctx: str = '') -> str:
    """
    Lightweight identify-only prompt for /classify.
    Determines object_type, category, rig_type, and whether augmentation is needed.
    Does NOT ask for joint placement — that is _build_joints_prompt().

    tag_ctx: optional string like '\nThe user identified this as: "dragon".'
    """
    return f"""Analyze this image to identify the object for 3D rigging.{tag_ctx}

Return ONLY valid JSON, no markdown, no extra text:

{{
  "object_type": "brief description of the object",
  "category": "animal|humanoid|vehicle|other",
  "rig_type": "humanoid|biped|quadruped|flying|vehicle|other",
  "needs_augmentation": false,
  "augment_prompt": ""
}}

Rules:
- object_type: concise noun phrase, e.g. "golden retriever", "toy truck", "anime girl"

- category: what the object IS
    animal    — any non-human creature (dog, dragon, bird, dinosaur)
    humanoid  — human or human-like figure (person, robot, zombie, alien)
    vehicle   — wheeled or motorised object (car, truck, bike)
    other     — furniture, food, abstract shapes, etc.

- rig_type: how the object MOVES and should be rigged — based on pose and structure,
    not on what the object is. A dog standing upright on two legs is "biped" not "quadruped".
    humanoid   — upright on two legs WITH arms (person, humanoid robot, zombie)
    biped      — upright on two legs WITHOUT arms (T-rex, penguin, bipedal statue)
    quadruped  — four legs, roughly horizontal spine (dog, horse, cat, lion)
    flying     — wings as primary limbs, may also have legs (bird, bat, dragon)
    vehicle    — wheels and axles (car, truck, bike, spacecraft)
    other      — no clear locomotion structure (snake, fish, furniture, abstract)

- needs_augmentation: true if the current pose will make rigging very difficult:
    • Limbs are bent, folded, or hidden (sitting, curled, wings closed)
    • Body parts overlap and cannot be separated
    • Extreme foreshortening hides limb structure
    Set false if pose is neutral/spread out or if rig_type is vehicle/other

- augment_prompt: only if needs_augmentation is true — describe what change
    would fix the pose. Leave empty string if needs_augmentation is false.
"""


def _build_joints_prompt(object_type: str, category: str,
                          n_joints: int | None = None,
                          mesh_bounds: dict | None = None,
                          rig_type: str = '') -> str:
    """
    Focused joint-placement prompt for /infer_joints.
    Receives object_type, category, and rig_type from /classify.
    rig_type drives which skeleton template is shown as an example.

    Key design decisions:
    - Z is always forced to 0.5 in code (center depth).
    - X and Y are the only meaningful coordinates from a frontal image.
    - We ask the model to cover the FULL object even if parts are cropped.

    mesh_bounds: dict with width/height in world units from the mesh GLB.
    n_joints: optional hint (treated as a suggestion).
    rig_type: humanoid|biped|quadruped|flying|vehicle|other
    """
    MIN_JOINTS, MAX_JOINTS = 3, 16
    if n_joints:
        joints_instruction = (
            f"\nAim for approximately {n_joints} joints total, "
            f"covering the FULL object anatomy from head to feet."
        )
    else:
        joints_instruction = (
            f"\nUse between {MIN_JOINTS} and {MAX_JOINTS} joints, "
            f"covering the FULL object anatomy from head to feet."
        )

    # Select example skeleton based on rig_type (how it moves),
    # falling back to category if rig_type is absent (old records)
    rt = (rig_type or category or '').lower()
    if rt == 'humanoid':
        example_json = _HUMANOID_EXAMPLE_JSON
    elif rt == 'biped':
        example_json = _BIPED_EXAMPLE_JSON
    elif rt == 'quadruped':
        example_json = _QUADRUPED_EXAMPLE_JSON
    elif rt == 'flying':
        example_json = _FLYING_EXAMPLE_JSON
    elif rt == 'vehicle':
        example_json = _VEHICLE_EXAMPLE_JSON
    else:
        # Fallback: use category
        if category == 'humanoid':
            example_json = _HUMANOID_EXAMPLE_JSON
        elif category == 'vehicle':
            example_json = _VEHICLE_EXAMPLE_JSON
        else:
            example_json = _ANIMAL_EXAMPLE_JSON

    if mesh_bounds:
        w = mesh_bounds['width']
        h = mesh_bounds['height']
        hw = h / w

        # Spine proportions by rig type — these are computed from mesh geometry,
        # not estimated from the image.
        if rt in ('biped', 'humanoid'):
            head_top = 1.00; head_bot = 0.75
            neck_y   = 0.72; chest_y  = 0.62
            pelvis_y = 0.42; spine_y  = 0.52
        elif rt == 'quadruped':
            head_top = 1.00; head_bot = 0.78
            neck_y   = 0.72; chest_y  = 0.58
            pelvis_y = 0.55; spine_y  = 0.57
        else:
            head_top = 1.00; head_bot = 0.78
            neck_y   = 0.72; chest_y  = 0.60
            pelvis_y = 0.45; spine_y  = 0.53

        mesh_context = f"""
MESH: width={w:.3f} height={h:.3f} ratio={hw:.2f}
y=0.0 = bottom of mesh (feet on ground), y=1.0 = top of mesh (top of head).

SPINE JOINTS — use these Y values directly, do NOT estimate from the image.
The image may be cropped or zoomed in, making the spine appear compressed.
These values are computed from the actual 3D mesh proportions:
  head:   y = {head_bot:.2f} – {head_top:.2f}
  neck:   y ≈ {neck_y:.2f}
  chest:  y ≈ {chest_y:.2f}
  spine:  y ≈ {spine_y:.2f}
  pelvis: y ≈ {pelvis_y:.2f}

LIMB JOINTS — estimate X and Y from the image:
  Look at where the legs/arms actually attach and articulate in this specific image.
  The hip x must be at the outer surface of the leg where it meets the pelvis.
  The knee y must be the midpoint between hip y and foot y.
  The foot y should be at or near 0.0 (ground level).
  Do NOT copy example x values — measure from the actual image.
"""
    else:
        mesh_context = ""

    return f"""You are placing a 3D skeleton on: "{object_type}"
rig_type: {rt or 'unknown'}  ← this tells you which skeleton structure to use
{joints_instruction}
{mesh_context}
COORDINATE RULES:
  x: 0.0=leftmost edge of mesh, 1.0=rightmost edge.
  y: 0.0=bottom of mesh (feet/ground), 1.0=top of mesh (head).
     For spine joints, use the precomputed values above — do NOT estimate from image.
     For limb joints, estimate from the image.

CRITICAL — COVER THE FULL OBJECT:
  Even if the image shows only part of the object, place joints for the ENTIRE
  anatomy. If the head is at the top, place head/neck joints at y≈1.0.
  If feet are at the bottom, place them at y≈0.0.

CRITICAL — ONLY REAL LIMBS:
  Only place joints where actual limbs exist. Count visible limbs carefully.
  Do NOT invent limbs that are not present on this object.

Do NOT include any "animations" key — animations are added automatically.

{example_json}

BODY_PART LABELS — use exactly these strings, they control the animations:
  Spine chain:  torso, pelvis, spine, chest, neck, head
  Arm chain:    shoulder, elbow, hand
  Leg chain:    hip, leg (for knee), foot
  Wing chain:   wing_base, wing_mid, wing_tip
  Vehicle:      body, axle, wheel

  The label "leg" means the KNEE joint — the middle joint of a leg chain.
  hip → leg (knee) → foot  is the correct leg hierarchy.
  NEVER use "shoulder/elbow/hand" labels for leg joints.
  If there are no arms, omit shoulder/elbow/hand entirely.
  ALWAYS use pelvis as the parent of hip joints (not chest).

LIMB PLACEMENT — measure each joint from the actual image:
  HIP:   at the outer surface of the leg where it visually branches from the pelvis.
    ✗ WRONG:   x=0.40 (too centered — inside the torso)
    ✓ CORRECT: x=0.25 (at the outer left surface of the leg)

  KNEE:  at the visible mid-leg articulation point in the image.
    The knee x is NOT the same as the hip x — legs taper, knee is more centered.
    The knee y = (hip_y + foot_y) / 2  — exact midpoint.
    ✗ WRONG:   knee x copied from hip x (leg has tapered — knee is narrower)
    ✓ CORRECT: knee x at the visible leg surface at knee height

  FOOT:  at the visible foot pad, near y=0.0 (ground level).
    x at the center of the foot pad as seen in the image.
"""


# ── Full example JSON skeletons per category ──────────────────────────────────
# These mirror the original _build_animal_prompt example closely.
# The model needs concrete coordinate values — a position guide table is weaker
# than seeing actual numbers inside real JSON.

_ANIMAL_EXAMPLE_JSON = """\
Return JSON in exactly this structure (adapt joint positions to match the image):

{
  "joint_hints": [
    {"name": "joint_root",           "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.42, "z": 0.5}},
    {"name": "joint_pelvis",         "body_part": "pelvis",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.42, "z": 0.5}},
    {"name": "joint_spine",          "body_part": "spine",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",          "body_part": "chest",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.65, "z": 0.5}},
    {"name": "joint_neck",           "body_part": "neck",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.75, "z": 0.5}},
    {"name": "joint_head",           "body_part": "head",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.88, "z": 0.5}},
    {"name": "joint_shoulder_left",  "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.15, "y": 0.62, "z": 0.5}},
    {"name": "joint_shoulder_right", "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.85, "y": 0.62, "z": 0.5}},
    {"name": "joint_elbow_left",     "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.08, "y": 0.50, "z": 0.5}},
    {"name": "joint_elbow_right",    "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.92, "y": 0.50, "z": 0.5}},
    {"name": "joint_hand_left",      "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.0,  "y": 0.38, "z": 0.5}},
    {"name": "joint_hand_right",     "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 1.0,  "y": 0.38, "z": 0.5}},
    {"name": "joint_hip_left",       "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.42, "z": 0.5}},
    {"name": "joint_hip_right",      "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.42, "z": 0.5}},
    {"name": "joint_knee_left",      "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.22, "z": 0.5}},
    {"name": "joint_knee_right",     "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.22, "z": 0.5}},
    {"name": "joint_foot_left",      "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.02, "z": 0.5}},
    {"name": "joint_foot_right",     "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.02, "z": 0.5}}
  ],
  "skeleton": [
    {"parent": "joint_root",           "child": "joint_pelvis"},
    {"parent": "joint_pelvis",         "child": "joint_spine"},
    {"parent": "joint_spine",          "child": "joint_chest"},
    {"parent": "joint_chest",          "child": "joint_neck"},
    {"parent": "joint_neck",           "child": "joint_head"},
    {"parent": "joint_chest",          "child": "joint_shoulder_left"},
    {"parent": "joint_chest",          "child": "joint_shoulder_right"},
    {"parent": "joint_shoulder_left",  "child": "joint_elbow_left"},
    {"parent": "joint_shoulder_right", "child": "joint_elbow_right"},
    {"parent": "joint_elbow_left",     "child": "joint_hand_left"},
    {"parent": "joint_elbow_right",    "child": "joint_hand_right"},
    {"parent": "joint_pelvis",         "child": "joint_hip_left"},
    {"parent": "joint_pelvis",         "child": "joint_hip_right"},
    {"parent": "joint_hip_left",       "child": "joint_knee_left"},
    {"parent": "joint_hip_right",      "child": "joint_knee_right"},
    {"parent": "joint_knee_left",      "child": "joint_foot_left"},
    {"parent": "joint_knee_right",     "child": "joint_foot_right"}
  ],
  "suggested_joints": 18
}

NOTE: The example above shows the full skeleton with both arms AND legs.
Adapt it to match the actual object:
  - If the object has NO arms: remove shoulder/elbow/hand joints entirely
  - If the object has NO legs: remove hip/leg/foot joints entirely
  - If the object has wings instead of arms: rename shoulder→wing_base, elbow→wing_mid, hand→wing_tip
  - ALWAYS use pelvis as the parent of hip joints (not chest)
  - ALWAYS use hip→leg(knee)→foot for the leg chain with body_part labels "hip", "leg", "foot"

KNEE PLACEMENT — most important rule:
  knee_y = (hip_y + foot_y) / 2   ← exact midpoint, no exceptions
  If hip_left is at y=0.42 and foot_left is at y=0.02, knee_left MUST be at y=0.22.
  NEVER place the knee closer to the foot than to the hip.

POSITION GUIDE (starting points only — override with what you actually see):
  Head:      y≈0.88,  x=0.5
  Neck:      y≈0.75,  x=0.5
  Chest:     y≈0.65,  x=0.5
  Spine:     y≈0.55,  x=0.5
  Pelvis:    y≈0.42,  x=0.5   ← parent of BOTH spine and hips
  Shoulders: y≈0.62,  x≈0.15 (left), x≈0.85 (right)
  Elbows:    y≈0.50,  x≈0.08 (left), x≈0.92 (right)
  Hands:     y≈0.38,  x=0.0  (left), x=1.0  (right)
  Hips:      y≈0.42,  x≈0.25 (left), x≈0.75 (right)  ← OUTER EDGE of leg, not center
  Knees:     y = midpoint(hip_y, foot_y), same x as hip
  Feet:      y≈0.02,  same x as hip"""


_HUMANOID_EXAMPLE_JSON = """\
Return JSON in exactly this structure (adapt joint positions to match the image):

{
  "joint_hints": [
    {"name": "joint_root",           "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.44, "z": 0.5}},
    {"name": "joint_pelvis",         "body_part": "pelvis",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.45, "z": 0.5}},
    {"name": "joint_spine",          "body_part": "spine",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",          "body_part": "chest",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.65, "z": 0.5}},
    {"name": "joint_neck",           "body_part": "neck",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.80, "z": 0.5}},
    {"name": "joint_head",           "body_part": "head",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.92, "z": 0.5}},
    {"name": "joint_shoulder_left",  "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.20, "y": 0.62, "z": 0.5}},
    {"name": "joint_shoulder_right", "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.80, "y": 0.62, "z": 0.5}},
    {"name": "joint_elbow_left",     "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.10, "y": 0.50, "z": 0.5}},
    {"name": "joint_elbow_right",    "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.90, "y": 0.50, "z": 0.5}},
    {"name": "joint_hand_left",      "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.05, "y": 0.38, "z": 0.5}},
    {"name": "joint_hand_right",     "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.95, "y": 0.38, "z": 0.5}},
    {"name": "joint_hip_left",       "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.28, "y": 0.44, "z": 0.5}},
    {"name": "joint_hip_right",      "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.72, "y": 0.44, "z": 0.5}},
    {"name": "joint_knee_left",      "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.28, "y": 0.22, "z": 0.5}},
    {"name": "joint_knee_right",     "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.72, "y": 0.22, "z": 0.5}},
    {"name": "joint_foot_left",      "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.28, "y": 0.0,  "z": 0.5}},
    {"name": "joint_foot_right",     "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.72, "y": 0.0,  "z": 0.5}}
  ],
  "skeleton": [
    {"parent": "joint_root",           "child": "joint_pelvis"},
    {"parent": "joint_pelvis",         "child": "joint_spine"},
    {"parent": "joint_spine",          "child": "joint_chest"},
    {"parent": "joint_chest",          "child": "joint_neck"},
    {"parent": "joint_neck",           "child": "joint_head"},
    {"parent": "joint_chest",          "child": "joint_shoulder_left"},
    {"parent": "joint_chest",          "child": "joint_shoulder_right"},
    {"parent": "joint_shoulder_left",  "child": "joint_elbow_left"},
    {"parent": "joint_shoulder_right", "child": "joint_elbow_right"},
    {"parent": "joint_elbow_left",     "child": "joint_hand_left"},
    {"parent": "joint_elbow_right",    "child": "joint_hand_right"},
    {"parent": "joint_pelvis",         "child": "joint_hip_left"},
    {"parent": "joint_pelvis",         "child": "joint_hip_right"},
    {"parent": "joint_hip_left",       "child": "joint_knee_left"},
    {"parent": "joint_hip_right",      "child": "joint_knee_right"},
    {"parent": "joint_knee_left",      "child": "joint_foot_left"},
    {"parent": "joint_knee_right",     "child": "joint_foot_right"}
  ],
  "suggested_joints": 18
}

KNEE PLACEMENT — most important rule:
  The knee y must be the midpoint between hip y and foot y.
  If hip_left is at y=0.44 and foot_left is at y=0.0, then knee_left MUST be at y=0.22.
  Formula: knee_y = (hip_y + foot_y) / 2
  NEVER place the knee closer to the foot than to the hip.

POSITION GUIDE (use FULL range 0.0–1.0 — starting points only, override with actual image):
  Head:      y≈0.92
  Neck:      y≈0.80
  Chest:     y≈0.65
  Shoulders: x≈0.20 (left), x≈0.80 (right), y≈0.62
  Elbows:    x≈0.10 (left), x≈0.90 (right), y≈0.50
  Hands:     x≈0.05 (left), x≈0.95 (right), y≈0.38
  Pelvis:    y≈0.44
  Hips:      x≈0.42 (left), x≈0.58 (right), y = top of leg
  Knees:     x same as hip, y = MIDPOINT between hip y and foot y
  Feet:      x≈0.40 (left), x≈0.60 (right), y=0.0"""


_BIPED_EXAMPLE_JSON = """\
This object is a BIPED — upright on two legs, NO arms.
Use this skeleton structure:

{
  "joint_hints": [
    {"name": "joint_root",        "body_part": "torso",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.42, "z": 0.5}},
    {"name": "joint_pelvis",      "body_part": "pelvis", "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.44, "z": 0.5}},
    {"name": "joint_spine",       "body_part": "spine",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",       "body_part": "chest",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.65, "z": 0.5}},
    {"name": "joint_neck",        "body_part": "neck",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.75, "z": 0.5}},
    {"name": "joint_head",        "body_part": "head",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.88, "z": 0.5}},
    {"name": "joint_hip_left",    "body_part": "hip",    "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.42, "z": 0.5}},
    {"name": "joint_hip_right",   "body_part": "hip",    "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.42, "z": 0.5}},
    {"name": "joint_knee_left",   "body_part": "leg",    "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.22, "z": 0.5}},
    {"name": "joint_knee_right",  "body_part": "leg",    "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.22, "z": 0.5}},
    {"name": "joint_foot_left",   "body_part": "foot",   "deforms_mesh": true,  "position_normalized": {"x": 0.25, "y": 0.02, "z": 0.5}},
    {"name": "joint_foot_right",  "body_part": "foot",   "deforms_mesh": true,  "position_normalized": {"x": 0.75, "y": 0.02, "z": 0.5}}
  ],
  "skeleton": [
    {"parent": "joint_root",       "child": "joint_pelvis"},
    {"parent": "joint_pelvis",     "child": "joint_spine"},
    {"parent": "joint_spine",      "child": "joint_chest"},
    {"parent": "joint_chest",      "child": "joint_neck"},
    {"parent": "joint_neck",       "child": "joint_head"},
    {"parent": "joint_pelvis",     "child": "joint_hip_left"},
    {"parent": "joint_pelvis",     "child": "joint_hip_right"},
    {"parent": "joint_hip_left",   "child": "joint_knee_left"},
    {"parent": "joint_hip_right",  "child": "joint_knee_right"},
    {"parent": "joint_knee_left",  "child": "joint_foot_left"},
    {"parent": "joint_knee_right", "child": "joint_foot_right"}
  ],
  "suggested_joints": 12
}

NO shoulder/elbow/hand joints — this is a biped with no arms.
Pelvis is the parent of both hips (NOT chest).
knee_y = (hip_y + foot_y) / 2  — place knee at the exact midpoint.

HIP/KNEE/FOOT X PLACEMENT — most common error:
  The hip x must be at the OUTER SURFACE of the leg, not near the body center.
  Look at the image: find the left leg and measure how far left it sits.
  ✗ WRONG:   joint_hip_left at x=0.40 (too centered — puts joint inside the torso)
  ✓ CORRECT: joint_hip_left at x=0.25 (at the outer left surface of the left leg)
  The knee and foot must share the same x as the hip on that side.
  If the legs are narrow and close together, the x values will be closer to 0.5.
  If the legs are wide apart, x values will be closer to 0.1 / 0.9.
  Do NOT copy the example x values — measure from the actual image.

POSITION GUIDE:
  Head:   y≈0.88, x=0.5
  Neck:   y≈0.75, x=0.5
  Chest:  y≈0.65, x=0.5
  Spine:  y≈0.55, x=0.5
  Pelvis: y≈0.42, x=0.5
  Hips:   y≈0.42, x≈0.40 (left) / 0.60 (right)  ← same y as pelvis
  Knees:  y = midpoint(hip_y, foot_y), same x as hip
  Feet:   y≈0.02, x≈0.38 (left) / 0.62 (right)"""


_QUADRUPED_EXAMPLE_JSON = """\
This object is a QUADRUPED — four legs, roughly horizontal spine.
Use this skeleton structure:

{
  "joint_hints": [
    {"name": "joint_root",             "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.0,  "z": 0.5}},
    {"name": "joint_pelvis",           "body_part": "pelvis",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.75}},
    {"name": "joint_spine",            "body_part": "spine",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.58, "z": 0.5}},
    {"name": "joint_chest",            "body_part": "chest",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.58, "z": 0.25}},
    {"name": "joint_neck",             "body_part": "neck",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.65, "z": 0.15}},
    {"name": "joint_head",             "body_part": "head",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.72, "z": 0.05}},
    {"name": "joint_front_hip_left",   "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.55, "z": 0.22}},
    {"name": "joint_front_hip_right",  "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.55, "z": 0.22}},
    {"name": "joint_front_knee_left",  "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.30, "z": 0.20}},
    {"name": "joint_front_knee_right", "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.30, "z": 0.20}},
    {"name": "joint_front_foot_left",  "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.02, "z": 0.18}},
    {"name": "joint_front_foot_right", "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.02, "z": 0.18}},
    {"name": "joint_rear_hip_left",    "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.55, "z": 0.78}},
    {"name": "joint_rear_hip_right",   "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.55, "z": 0.78}},
    {"name": "joint_rear_knee_left",   "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.30, "z": 0.80}},
    {"name": "joint_rear_knee_right",  "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.30, "z": 0.80}},
    {"name": "joint_rear_foot_left",   "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.02, "z": 0.82}},
    {"name": "joint_rear_foot_right",  "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.02, "z": 0.82}}
  ],
  "skeleton": [
    {"parent": "joint_root",            "child": "joint_pelvis"},
    {"parent": "joint_pelvis",          "child": "joint_spine"},
    {"parent": "joint_spine",           "child": "joint_chest"},
    {"parent": "joint_chest",           "child": "joint_neck"},
    {"parent": "joint_neck",            "child": "joint_head"},
    {"parent": "joint_chest",           "child": "joint_front_hip_left"},
    {"parent": "joint_chest",           "child": "joint_front_hip_right"},
    {"parent": "joint_front_hip_left",  "child": "joint_front_knee_left"},
    {"parent": "joint_front_hip_right", "child": "joint_front_knee_right"},
    {"parent": "joint_front_knee_left", "child": "joint_front_foot_left"},
    {"parent": "joint_front_knee_right","child": "joint_front_foot_right"},
    {"parent": "joint_pelvis",          "child": "joint_rear_hip_left"},
    {"parent": "joint_pelvis",          "child": "joint_rear_hip_right"},
    {"parent": "joint_rear_hip_left",   "child": "joint_rear_knee_left"},
    {"parent": "joint_rear_hip_right",  "child": "joint_rear_knee_right"},
    {"parent": "joint_rear_knee_left",  "child": "joint_rear_foot_left"},
    {"parent": "joint_rear_knee_right", "child": "joint_rear_foot_right"}
  ],
  "suggested_joints": 18
}

For quadrupeds, z is meaningful: front legs z≈0.2, rear legs z≈0.8.
Spine runs roughly horizontal (y stays nearly constant along z axis).
Front leg joints use shoulder/elbow/hand labels; rear leg joints use hip/leg/foot."""


_FLYING_EXAMPLE_JSON = """\
This object is a FLYING creature — wings as primary limbs.
Use this skeleton structure (adapt if it also has legs):

{
  "joint_hints": [
    {"name": "joint_root",           "body_part": "torso",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.45, "z": 0.5}},
    {"name": "joint_spine",          "body_part": "spine",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",          "body_part": "chest",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.60, "z": 0.5}},
    {"name": "joint_neck",           "body_part": "neck",      "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.72, "z": 0.5}},
    {"name": "joint_head",           "body_part": "head",      "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.88, "z": 0.5}},
    {"name": "joint_wing_base_left",  "body_part": "wing_base", "deforms_mesh": true,  "position_normalized": {"x": 0.20, "y": 0.60, "z": 0.5}},
    {"name": "joint_wing_base_right", "body_part": "wing_base", "deforms_mesh": true,  "position_normalized": {"x": 0.80, "y": 0.60, "z": 0.5}},
    {"name": "joint_wing_mid_left",   "body_part": "wing_mid",  "deforms_mesh": true,  "position_normalized": {"x": 0.08, "y": 0.55, "z": 0.5}},
    {"name": "joint_wing_mid_right",  "body_part": "wing_mid",  "deforms_mesh": true,  "position_normalized": {"x": 0.92, "y": 0.55, "z": 0.5}},
    {"name": "joint_wing_tip_left",   "body_part": "wing_tip",  "deforms_mesh": true,  "position_normalized": {"x": 0.02, "y": 0.50, "z": 0.5}},
    {"name": "joint_wing_tip_right",  "body_part": "wing_tip",  "deforms_mesh": true,  "position_normalized": {"x": 0.98, "y": 0.50, "z": 0.5}},
    {"name": "joint_hip_left",        "body_part": "hip",       "deforms_mesh": true,  "position_normalized": {"x": 0.42, "y": 0.42, "z": 0.5}},
    {"name": "joint_hip_right",       "body_part": "hip",       "deforms_mesh": true,  "position_normalized": {"x": 0.58, "y": 0.42, "z": 0.5}},
    {"name": "joint_foot_left",       "body_part": "foot",      "deforms_mesh": true,  "position_normalized": {"x": 0.42, "y": 0.02, "z": 0.5}},
    {"name": "joint_foot_right",      "body_part": "foot",      "deforms_mesh": true,  "position_normalized": {"x": 0.58, "y": 0.02, "z": 0.5}}
  ],
  "skeleton": [
    {"parent": "joint_root",           "child": "joint_spine"},
    {"parent": "joint_spine",          "child": "joint_chest"},
    {"parent": "joint_chest",          "child": "joint_neck"},
    {"parent": "joint_neck",           "child": "joint_head"},
    {"parent": "joint_chest",          "child": "joint_wing_base_left"},
    {"parent": "joint_chest",          "child": "joint_wing_base_right"},
    {"parent": "joint_wing_base_left", "child": "joint_wing_mid_left"},
    {"parent": "joint_wing_base_right","child": "joint_wing_mid_right"},
    {"parent": "joint_wing_mid_left",  "child": "joint_wing_tip_left"},
    {"parent": "joint_wing_mid_right", "child": "joint_wing_tip_right"},
    {"parent": "joint_root",           "child": "joint_hip_left"},
    {"parent": "joint_root",           "child": "joint_hip_right"},
    {"parent": "joint_hip_left",       "child": "joint_foot_left"},
    {"parent": "joint_hip_right",      "child": "joint_foot_right"}
  ],
  "suggested_joints": 15
}

If the creature has no visible legs (e.g. a bird in flight), remove hip/foot joints.
Wing tips should reach the very edges of the mesh (x≈0.02 and x≈0.98)."""


_VEHICLE_EXAMPLE_JSON = """\
Return JSON in exactly this structure (adapt joint positions to match the image):

{
  "joint_hints": [
    {"name": "body",       "body_part": "body",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.5,  "z": 0.5}},
    {"name": "front_axle", "body_part": "axle",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.25, "z": 0.15}},
    {"name": "rear_axle",  "body_part": "axle",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.75, "z": 0.15}},
    {"name": "wheel_fl",   "body_part": "wheel", "deforms_mesh": true,  "position_normalized": {"x": 0.15, "y": 0.25, "z": 0.15}},
    {"name": "wheel_fr",   "body_part": "wheel", "deforms_mesh": true,  "position_normalized": {"x": 0.85, "y": 0.25, "z": 0.15}},
    {"name": "wheel_rl",   "body_part": "wheel", "deforms_mesh": true,  "position_normalized": {"x": 0.15, "y": 0.75, "z": 0.15}},
    {"name": "wheel_rr",   "body_part": "wheel", "deforms_mesh": true,  "position_normalized": {"x": 0.85, "y": 0.75, "z": 0.15}}
  ],
  "skeleton": [
    {"parent": "body",       "child": "front_axle"},
    {"parent": "body",       "child": "rear_axle"},
    {"parent": "front_axle", "child": "wheel_fl"},
    {"parent": "front_axle", "child": "wheel_fr"},
    {"parent": "rear_axle",  "child": "wheel_rl"},
    {"parent": "rear_axle",  "child": "wheel_rr"}
  ],
  "suggested_joints": 7
}

Left wheels x<0.4, right wheels x>0.6.
front_axle y must match front wheel y, rear_axle y must match rear wheel y.
body and axles: deforms_mesh=false. wheels: deforms_mesh=true."""


def _build_vehicle_prompt() -> str:
    """
    Combined identify + joint placement prompt for vehicles.
    Vehicles are a special case - the joint schema is rigid enough that
    splitting identify/joints adds no value. Used by classify_with_vision()
    when VEHICLE_KEYWORDS are detected in the user tag.
    """
    return """Analyze this vehicle image for 3D rigging.

Return ONLY valid JSON with no markdown, no extra text, no backticks.

{
  "object_type": "toy truck",
  "category": "vehicle",
  "needs_augmentation": false,
  "augment_prompt": "",
  "wheel_colors_rgb": [
    [0.05, 0.05, 0.05],
    [0.85, 0.85, 0.85]
  ],
  "suggested_joints": 7,
  "joint_hints": [
    {
      "name": "body",
      "body_part": "body",
      "deforms_mesh": false,
      "position_normalized": {"x": 0.5, "y": 0.5, "z": 0.5},
      "animations": []
    },
    {
      "name": "front_axle",
      "body_part": "axle",
      "deforms_mesh": false,
      "position_normalized": {"x": 0.5, "y": 0.25, "z": 0.15},
      "animations": []
    },
    {
      "name": "rear_axle",
      "body_part": "axle",
      "deforms_mesh": false,
      "position_normalized": {"x": 0.5, "y": 0.75, "z": 0.15},
      "animations": []
    },
    {
      "name": "wheel_fl",
      "body_part": "wheel",
      "deforms_mesh": true,
      "position_normalized": {"x": 0.15, "y": 0.25, "z": 0.15},
      "wheel_radius_normalized": 0.18,
      "animations": [
        {"clip": "drive", "property": "rotation_euler", "axis": "x",
         "keyframes": [[0,0.0],[30,3.14159],[60,6.28318]], "loop": true}
      ]
    },
    {
      "name": "wheel_fr",
      "body_part": "wheel",
      "deforms_mesh": true,
      "position_normalized": {"x": 0.85, "y": 0.25, "z": 0.15},
      "wheel_radius_normalized": 0.18,
      "animations": [
        {"clip": "drive", "property": "rotation_euler", "axis": "x",
         "keyframes": [[0,0.0],[30,3.14159],[60,6.28318]], "loop": true}
      ]
    },
    {
      "name": "wheel_rl",
      "body_part": "wheel",
      "deforms_mesh": true,
      "position_normalized": {"x": 0.15, "y": 0.75, "z": 0.15},
      "wheel_radius_normalized": 0.18,
      "animations": [
        {"clip": "drive", "property": "rotation_euler", "axis": "x",
         "keyframes": [[0,0.0],[30,3.14159],[60,6.28318]], "loop": true}
      ]
    },
    {
      "name": "wheel_rr",
      "body_part": "wheel",
      "deforms_mesh": true,
      "position_normalized": {"x": 0.85, "y": 0.75, "z": 0.15},
      "wheel_radius_normalized": 0.18,
      "animations": [
        {"clip": "drive", "property": "rotation_euler", "axis": "x",
         "keyframes": [[0,0.0],[30,3.14159],[60,6.28318]], "loop": true}
      ]
    }
  ],
  "skeleton": [
    {"parent": "body",       "child": "front_axle"},
    {"parent": "body",       "child": "rear_axle"},
    {"parent": "front_axle", "child": "wheel_fl"},
    {"parent": "front_axle", "child": "wheel_fr"},
    {"parent": "rear_axle",  "child": "wheel_rl"},
    {"parent": "rear_axle",  "child": "wheel_rr"}
  ]
}

Rules:
- wheel_colors_rgb: identify actual tire and hub colors from the image as RGB 0-1
- front wheels y=0.25, rear wheels y=0.75 — DO NOT set all wheels to y=0.5
- left wheels x=0.15, right wheels x=0.85, all wheels z=0.15 (near bottom)
- front_axle y must match front wheel y, rear_axle y must match rear wheel y
- body and axles: deforms_mesh false, animations []
- wheels: deforms_mesh true, keep the drive spin animations exactly as shown
- x=0 is LEFT side of vehicle, x=1 is RIGHT side
- left wheels must have x < 0.4, right wheels must have x > 0.6
- z is DEPTH (front=0, rear=1), NOT left/right
"""
