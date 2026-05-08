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
    Determines object_type, category, rig_type, body_parts, and augmentation need.
    Does NOT ask for joint placement — that is _build_joints_prompt().
    """
    return f"""Analyze this image to identify the object for 3D rigging.{tag_ctx}

Return ONLY valid JSON, no markdown, no extra text:

{{
  "object_type": "brief description of the object",
  "category": "animal|humanoid|vehicle|other",
  "rig_type": "humanoid|biped|quadruped|flying|vehicle|other",
  "has_body_parts": {{
    "head": true,
    "neck": false,
    "torso": false,
    "arms": false,
    "legs": false,
    "jaw": false
  }},
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

- rig_type: how the object MOVES and should be rigged
    humanoid   — upright on two legs WITH arms (person, humanoid robot, zombie)
    biped      — upright on two legs WITHOUT arms (T-rex, penguin, bipedal statue)
    quadruped  — four legs, roughly horizontal spine (dog, horse, cat, lion)
    flying     — wings as primary limbs, may also have legs (bird, bat, dragon)
    vehicle    — wheels and axles (car, truck, bike, spacecraft)
    other      — no clear locomotion structure (snake, fish, furniture, abstract)


- has_body_parts: CRITICAL — which body parts PHYSICALLY EXIST on this object
    • head: true if has distinct head (even if it's the only part)
    • neck: true if neck is separate/articulated between head and torso
    • torso: true if has body/trunk distinct from head
    • arms: true if has arm limbs or upper limbs (includes wings for flying types)
    • legs: true if has leg limbs or lower limbs
    • jaw: true if jaw is visibly articulated (opening/closing)
    
    RULES FOR has_body_parts:
    ✓ Full humanoid → {{"head": true, "neck": true, "torso": true, "arms": true, "legs": true, "jaw": false}}
    ✓ humanoid missing limbs → {{"head": true, "neck": true, "torso": true}}
    ✓ four legged animal → {{"head": true, "neck": true, "torso": true, "arms": true, "legs": true, "jaw": false}}
    ✓ Bird flying → {{"head": true, "neck": true, "torso": true, "arms": true, "legs": true, "jaw": false}}
    ✓ Car (vehicle) → {{"head": false, "neck": false, "torso": false, "arms": false, "legs": false, "jaw": false}}


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
    """Simplified: core rules only. Validation happens server-side."""
    
    if mesh_bounds:
        w = mesh_bounds['width']
        h = mesh_bounds['height']
        mesh_context = f"MESH: width={w:.3f}, height={h:.3f} (y=0.0 bottom, y=1.0 top)"
    else:
        mesh_context = ""

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
        example_json = _ANIMAL_EXAMPLE_JSON

    return f"""Place joints on the image of: "{object_type}" (rig_type: {rt or 'unknown'})
{mesh_context}

COORDINATES: x ∈ [0,1] (left→right), y ∈ [0,1] (bottom→top), z = 0.5 (always)

RULE: NO PHANTOM LIMBS
You MUST regard rig_type higher priority than object_type
IF {rig_type} is bipedal and {object_type} is animal, only one set of legs!

LIMB STRUCTURE: Every limb has exactly and ONLY 3 joints: BASE → MIDDLE → END
  - Arm: shoulder → elbow → hand
  - Leg: hip → knee → foot
  - Wing: wing_base → wing_mid → wing_tip

BASE JOINT PLACEMENT (hip, shoulder, wing_base):
  1. Look at the LEFT limbs in this image. Where do they visibly ATTACH to the body?
   - Is it at the left edge, left-center, or center of the image?
   - Describe what you see.
   - Assign an X coordinate: 0.1-0.3 (far left), 0.3-0.4 (left-center), or other?

2. Look at the RIGHT limbs in this image. Where do they visibly ATTACH to the body?
   - Is it at the right edge, right-center, or center of the image?
   - Describe what you see.
   - Assign an X coordinate: 0.7-0.9 (far right), 0.6-0.7 (right-center), or other?
  
  
  ✗ WRONG: Place joints at body CENTER (x=0.5)
  x WRONG: joints markers are not touching the subject 
  x WRONG: joint markers are on white space
  ✗ WRONG: Guess a spread distance
  ✓ CORRECT: Find the actual junction/attachment point in this image
  
  For LEGS on a biped:
    - Look at where the LEFT leg meets the torso/hips
    - Place joint_hip_left AT that junction (typically x ≈ 0.25-0.35)
    - Look at where the RIGHT leg meets the torso/hips  
    - Place joint_hip_right AT that junction (typically x ≈ 0.65-0.75)
    - These positions will naturally separate because legs attach OUTWARD from body center
    
  For ARMS on a humanoid:
    - Find where LEFT arm shoulder meets the torso
    - Find where RIGHT arm shoulder meets the torso
    - Place shoulder joints AT those visible attachment points

VALIDATION:
  - Count only VISIBLE limbs (do NOT invent)
  - Total joints = 4 (spine) + 3×(number of limbs)
  - No decorative elements as joints
  - Use exact body_part labels: shoulder, elbow, hand, hip, leg, foot, etc.
  - Joints MUST be place ON part of the subject

{example_json}

Return ONLY valid JSON with no markdown.
"""


_BIPED_EXAMPLE_JSON = """\
BIPED: 2 legs + spine, NO arms. Total: 12 joints.
Adapt x/y positions to match your image.

{
  "joint_hints": [
    {"name": "joint_root",        "body_part": "torso",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.5,  "z": 0.5}},
    {"name": "joint_pelvis",      "body_part": "pelvis", "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.42, "z": 0.5}},
    {"name": "joint_spine",       "body_part": "spine",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",       "body_part": "chest",  "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.65, "z": 0.5}},
    {"name": "joint_neck",        "body_part": "neck",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.75, "z": 0.5}},
    {"name": "joint_head",        "body_part": "head",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.88, "z": 0.5}},
    {"name": "joint_hip_left",    "body_part": "hip",    "deforms_mesh": true,  "position_normalized": {"x": 0.40, "y": 0.42, "z": 0.5}},
    {"name": "joint_hip_right",   "body_part": "hip",    "deforms_mesh": true,  "position_normalized": {"x": 0.60, "y": 0.42, "z": 0.5}},
    {"name": "joint_knee_left",   "body_part": "leg",    "deforms_mesh": true,  "position_normalized": {"x": 0.40, "y": 0.22, "z": 0.5}},
    {"name": "joint_knee_right",  "body_part": "leg",    "deforms_mesh": true,  "position_normalized": {"x": 0.60, "y": 0.22, "z": 0.5}},
    {"name": "joint_foot_left",   "body_part": "foot",   "deforms_mesh": true,  "position_normalized": {"x": 0.40, "y": 0.02, "z": 0.5}},
    {"name": "joint_foot_right",  "body_part": "foot",   "deforms_mesh": true,  "position_normalized": {"x": 0.60, "y": 0.02, "z": 0.5}}
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

VARIATIONS:
• If object has ARMS: add shoulder/elbow/hand chains (parent to chest, not pelvis)
• If object has NO LEGS: delete all hip/knee/foot joints
• If object has WINGS instead of arms: use wing_base/wing_mid/wing_tip
• If object has NO SPINE: delete spine chain, parent limbs to root
"""

_ARMS_ONLY_EXAMPLE_JSON = """\
ARMS-ONLY: 2 arms + spine, NO legs. Total: 12 joints.

{
  "joint_hints": [
    {"name": "joint_root",           "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.5, "z": 0.5}},
    {"name": "joint_pelvis",         "body_part": "pelvis",   "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.45, "z": 0.5}},
    {"name": "joint_spine",          "body_part": "spine",    "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.55, "z": 0.5}},
    {"name": "joint_chest",          "body_part": "chest",    "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.65, "z": 0.5}},
    {"name": "joint_neck",           "body_part": "neck",     "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.80, "z": 0.5}},
    {"name": "joint_head",           "body_part": "head",     "deforms_mesh": false, "position_normalized": {"x": 0.5, "y": 0.92, "z": 0.5}},
    {"name": "joint_shoulder_left",  "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.20, "y": 0.62, "z": 0.5}},
    {"name": "joint_shoulder_right", "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.80, "y": 0.62, "z": 0.5}},
    {"name": "joint_elbow_left",     "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.10, "y": 0.50, "z": 0.5}},
    {"name": "joint_elbow_right",    "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.90, "y": 0.50, "z": 0.5}},
    {"name": "joint_hand_left",      "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.05, "y": 0.38, "z": 0.5}},
    {"name": "joint_hand_right",     "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.95, "y": 0.38, "z": 0.5}}
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
    {"parent": "joint_elbow_right",    "child": "joint_hand_right"}
  ],
  "suggested_joints": 12
}

NOTE: NO hip/knee/foot — object has no legs.
"""

_HUMANOID_EXAMPLE_JSON = """\
HUMANOID: 2 arms + 2 legs + spine. Total: 18 joints.

{
  "joint_hints": [
    {"name": "joint_root",           "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.5,  "z": 0.5}},
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
    {"name": "joint_hip_left",       "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.42, "y": 0.44, "z": 0.5}},
    {"name": "joint_hip_right",      "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.58, "y": 0.44, "z": 0.5}},
    {"name": "joint_knee_left",      "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.42, "y": 0.22, "z": 0.5}},
    {"name": "joint_knee_right",     "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.58, "y": 0.22, "z": 0.5}},
    {"name": "joint_foot_left",      "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.40, "y": 0.0,  "z": 0.5}},
    {"name": "joint_foot_right",     "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.60, "y": 0.0,  "z": 0.5}}
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
"""

_ANIMAL_EXAMPLE_JSON = """\
GENERIC ANIMAL: Adapt based on actual limbs present.
Example: quadruped with 4 legs + spine. Total: 18 joints.

{
  "joint_hints": [
    {"name": "joint_root",             "body_part": "torso",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.5,  "z": 0.5}},
    {"name": "joint_pelvis",           "body_part": "pelvis",   "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.55, "z": 0.5}},
    {"name": "joint_spine",            "body_part": "spine",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.58, "z": 0.5}},
    {"name": "joint_chest",            "body_part": "chest",    "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.62, "z": 0.5}},
    {"name": "joint_neck",             "body_part": "neck",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.70, "z": 0.5}},
    {"name": "joint_head",             "body_part": "head",     "deforms_mesh": false, "position_normalized": {"x": 0.5,  "y": 0.82, "z": 0.5}},
    {"name": "joint_hip_front_left",   "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.58, "z": 0.5}},
    {"name": "joint_hip_front_right",  "body_part": "shoulder", "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.58, "z": 0.5}},
    {"name": "joint_knee_front_left",  "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.35, "z": 0.5}},
    {"name": "joint_knee_front_right", "body_part": "elbow",    "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.35, "z": 0.5}},
    {"name": "joint_foot_front_left",  "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.35, "y": 0.05, "z": 0.5}},
    {"name": "joint_foot_front_right", "body_part": "hand",     "deforms_mesh": true,  "position_normalized": {"x": 0.65, "y": 0.05, "z": 0.5}},
    {"name": "joint_hip_rear_left",    "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.55, "z": 0.5}},
    {"name": "joint_hip_rear_right",   "body_part": "hip",      "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.55, "z": 0.5}},
    {"name": "joint_knee_rear_left",   "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.30, "z": 0.5}},
    {"name": "joint_knee_rear_right",  "body_part": "leg",      "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.30, "z": 0.5}},
    {"name": "joint_foot_rear_left",   "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.38, "y": 0.05, "z": 0.5}},
    {"name": "joint_foot_rear_right",  "body_part": "foot",     "deforms_mesh": true,  "position_normalized": {"x": 0.62, "y": 0.05, "z": 0.5}}
  ],
  "skeleton": [
    {"parent": "joint_root",            "child": "joint_pelvis"},
    {"parent": "joint_pelvis",          "child": "joint_spine"},
    {"parent": "joint_spine",           "child": "joint_chest"},
    {"parent": "joint_chest",           "child": "joint_neck"},
    {"parent": "joint_neck",            "child": "joint_head"},
    {"parent": "joint_chest",           "child": "joint_hip_front_left"},
    {"parent": "joint_chest",           "child": "joint_hip_front_right"},
    {"parent": "joint_hip_front_left",  "child": "joint_knee_front_left"},
    {"parent": "joint_hip_front_right", "child": "joint_knee_front_right"},
    {"parent": "joint_knee_front_left", "child": "joint_foot_front_left"},
    {"parent": "joint_knee_front_right","child": "joint_foot_front_right"},
    {"parent": "joint_pelvis",          "child": "joint_hip_rear_left"},
    {"parent": "joint_pelvis",          "child": "joint_hip_rear_right"},
    {"parent": "joint_hip_rear_left",   "child": "joint_knee_rear_left"},
    {"parent": "joint_hip_rear_right",  "child": "joint_knee_rear_right"},
    {"parent": "joint_knee_rear_left",  "child": "joint_foot_rear_left"},
    {"parent": "joint_knee_rear_right", "child": "joint_foot_rear_right"}
  ],
  "suggested_joints": 18
}

ADAPT THIS EXAMPLE:
• If wings instead of front legs: use wing_base/wing_mid/wing_tip, parent to chest
• If no spine/torso: remove spine chain, parent leg bases to root
"""
