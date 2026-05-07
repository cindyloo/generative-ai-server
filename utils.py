
import os
import sys
import io
import base64
import json
import logging
import time
from flask import Flask, request, jsonify, send_file
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from rembg import remove, new_session
from PIL import Image
import requests
import urllib3
import subprocess
import threading
from pathlib import Path
import hashlib
import numpy as np

VEHICLE_KEYWORDS = {'car', 'truck', 'vehicle', 'bus', 'bike', 'motorcycle', 'van', 'auto', 'wheels'}

HUMANOID_KEYWORDS = {'human', 'person', 'character', 'humanoid', 'man',
                             'woman', 'boy', 'girl', 'robot', 'alien', 'zombie',
                             'totoro', 'creature', 'figure', 'monster'}
 # ── Helpers ────────────────────────────────────────────────────────────────────

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

# ── Keyframe injection (procedural, replaces hardcoded animations) ─────────────

# axis, phase (+1 normal / -1 inverted), base amplitude in radians
WALK_KEYFRAMES = {
    "root":      None,
    "pelvis":    ("z",  1,  0.05),
    "spine":     ("y", -1,  0.06),   # counter-rotates vs pelvis
    "chest":     ("y", -1,  0.05),
    "neck":      ("z",  1,  0.02),
    "head":      ("z", -1,  0.02),
    "shoulder":  ("x",  1,  0.35),   # phase flipped per side below
    "elbow":     ("x",  1,  0.20),
    "hand":      ("x",  1,  0.10),
    "hip":       ("x",  1,  0.20),   # phase flipped per side below
    "leg":       ("x",  1,  0.50),
    "foot":      ("x",  1,  0.10),
    "wing_base": ("z",  1,  0.30),
    "wing_mid":  ("z",  1,  0.20),
    "wing_tip":  ("z",  1,  0.10),
    "axle":      None,               # static — child wheels spin
    "body":      None,               # static
    # "wheel" is special-cased in build_walk_keyframes
}

# Right-side joints get their phase flipped so left/right oppose each other
RIGHT_SIDE_FLIP = {"shoulder", "elbow", "hand", "hip", "leg", "foot",
                   "wing_base", "wing_mid", "wing_tip"}

def build_walk_keyframes(body_part: str, joint_name: str,
                         bone_length: float = None) -> list:
    """
    Returns an animations list for a joint given its body_part label.
    bone_length: world-space distance to nearest child, used to scale amplitude.
    """
    body_part_lower = (body_part or '').lower()

    # Wheels spin continuously — not an oscillation
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

    # Flip phase for right-side limbs so they oppose left
    if is_right and body_part_lower in RIGHT_SIDE_FLIP:
        phase *= -1

    # Scale amplitude by bone length so longer bones don't over-rotate
    amp = base_amp
    if bone_length is not None:
        REFERENCE_BONE = 0.4   # world units for a typical mesh
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
    run_blender_rig.
    """
    positions = {j['id']: np.array(j['position']) for j in skel['joints']}

    # Build parent→children map for bone-length calculation
    children = {}
    for bone in skel['bones']:
        p, c = bone['parent'], bone['child']
        children.setdefault(p, []).append(c)

    for joint in skel['joints']:
        hint      = joint.get('hint') or {}
        body_part = hint.get('body_part', '')
        name      = joint.get('name', '')

        # Bone length = distance to first child joint
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


# ── fal.ai image editing ───────────────────────────────────────────────────────

def edit_image_fal(img: Image.Image, prompt: str) -> Image.Image:
    """Edit image using fal.ai Qwen Image 2.0. No mask needed."""
    import fal_client

    data_uri = f"data:image/png;base64,{img_to_b64(img)}"
    log.info(f"fal.ai: {prompt[:80]}...")

    result  = fal_client.subscribe(
        "fal-ai/qwen-image-2/edit",
        arguments={"prompt": prompt, "image_urls": [data_uri], "num_images": 2}
    )
    out_url  = result["images"][0]["url"]
    out_url2 = result["images"][1]["url"]
    responseA = requests.get(out_url,  verify=False)
    responseB = requests.get(out_url2, verify=False)
    return (Image.open(io.BytesIO(responseA.content)).convert('RGB'),
            Image.open(io.BytesIO(responseB.content)).convert('RGB'))


# ── Vision prompts ─────────────────────────────────────────────────────────────
#
#  vehicle_prompt  — keeps animations (wheels spin, axles static)
#  animal_prompt   — NO animations; inject_keyframes() adds them procedurally
#
# Both are built inside classify_with_vision() so they can reference the local
# variables tag_context, bounds_info, joints_instruction, requested_joints.

def _build_vehicle_prompt() -> str:
    return """Analyze this vehicle image for 3D rigging.

Return ONLY valid JSON with no markdown, no extra text, no backticks.

{
  "object_type": "toy truck",
  "category": "vehicle",
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


def _build_animal_prompt(tag_context: str, bounds_info: str,
                         joints_instruction: str, requested_joints) -> str:
    return f"""Analyze this 3D object for Blender rigging. Return ONLY valid JSON.{tag_context}{bounds_info}{joints_instruction}

COORDINATES: x=0(left) 1(right), y=0(bottom) 1(top), z=0(front) 1(back)

{{
  "object_type": "...",
  "category": "animal|vehicle|humanoid|other",
  "needs_augmentation": false,
  "augment_prompt": "",
  "suggested_joints": {requested_joints or 12},
  "joint_hints": [
    {{
      "name": "joint_root",
      "body_part": "torso",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.0, "z": 0.5}}
    }},
    {{
      "name": "joint_pelvis",
      "body_part": "pelvis",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.20, "z": 0.5}}
    }},
    {{
      "name": "joint_spine",
      "body_part": "spine",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.40, "z": 0.5}}
    }},
    {{
      "name": "joint_chest",
      "body_part": "chest",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.55, "z": 0.5}}
    }},
    {{
      "name": "joint_neck",
      "body_part": "neck",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.70, "z": 0.5}}
    }},
    {{
      "name": "joint_head",
      "body_part": "head",
      "deforms_mesh": false,
      "position_normalized": {{"x": 0.5, "y": 0.88, "z": 0.5}}
    }},
    {{
      "name": "joint_shoulder_left",
      "body_part": "shoulder",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.15, "y": 0.50, "z": 0.5}}
    }},
    {{
      "name": "joint_shoulder_right",
      "body_part": "shoulder",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.85, "y": 0.50, "z": 0.5}}
    }},
    {{
      "name": "joint_elbow_left",
      "body_part": "elbow",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.08, "y": 0.42, "z": 0.5}}
    }},
    {{
      "name": "joint_elbow_right",
      "body_part": "elbow",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.92, "y": 0.42, "z": 0.5}}
    }},
    {{
      "name": "joint_hand_left",
      "body_part": "hand",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.0, "y": 0.35, "z": 0.5}}
    }},
    {{
      "name": "joint_hand_right",
      "body_part": "hand",
      "deforms_mesh": true,
      "position_normalized": {{"x": 1.0, "y": 0.35, "z": 0.5}}
    }},
    {{
      "name": "joint_hip_left",
      "body_part": "hip",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.38, "y": 0.22, "z": 0.5}}
    }},
    {{
      "name": "joint_hip_right",
      "body_part": "hip",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.62, "y": 0.22, "z": 0.5}}
    }},
    {{
      "name": "joint_knee_left",
      "body_part": "leg",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.38, "y": 0.15, "z": 0.5}}
    }},
    {{
      "name": "joint_knee_right",
      "body_part": "leg",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.62, "y": 0.15, "z": 0.5}}
    }},
    {{
      "name": "joint_foot_left",
      "body_part": "foot",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.35, "y": 0.0, "z": 0.5}}
    }},
    {{
      "name": "joint_foot_right",
      "body_part": "foot",
      "deforms_mesh": true,
      "position_normalized": {{"x": 0.65, "y": 0.0, "z": 0.5}}
    }}
  ],
  "skeleton": [
    {{"parent": "joint_root",           "child": "joint_pelvis"}},
    {{"parent": "joint_pelvis",         "child": "joint_spine"}},
    {{"parent": "joint_spine",          "child": "joint_chest"}},
    {{"parent": "joint_chest",          "child": "joint_neck"}},
    {{"parent": "joint_neck",           "child": "joint_head"}},
    {{"parent": "joint_chest",          "child": "joint_shoulder_left"}},
    {{"parent": "joint_chest",          "child": "joint_shoulder_right"}},
    {{"parent": "joint_shoulder_left",  "child": "joint_elbow_left"}},
    {{"parent": "joint_shoulder_right", "child": "joint_elbow_right"}},
    {{"parent": "joint_elbow_left",     "child": "joint_hand_left"}},
    {{"parent": "joint_elbow_right",    "child": "joint_hand_right"}},
    {{"parent": "joint_pelvis",         "child": "joint_hip_left"}},
    {{"parent": "joint_pelvis",         "child": "joint_hip_right"}},
    {{"parent": "joint_hip_left",       "child": "joint_knee_left"}},
    {{"parent": "joint_hip_right",      "child": "joint_knee_right"}},
    {{"parent": "joint_knee_left",      "child": "joint_foot_left"}},
    {{"parent": "joint_knee_right",     "child": "joint_foot_right"}}
  ]
}}

YOUR TASK:
  • Look at the image and FIND where limbs physically attach to the body
  • position_normalized must reflect the ACTUAL ATTACHMENT POINT, not the example values
  • Shoulders: at the visual junction where arms branch FROM the body
  • Neck: where head visibly connects to body
  • Hips: where legs branch from body
  • Elbows/knees: midpoint along the limb
  • Hands/feet: at the tips/extremities
  • Do NOT include any "animations" key — animations are added automatically
  • Adjust joint positions to match where limbs actually are in the image
  • Left/right sides must be symmetric
  • If object has no arms (e.g. snake, fish), remove shoulder/elbow/hand joints
  • If object has wings, rename: shoulder→wing_base, elbow→wing_mid, hand→wing_tip
  • needs_augmentation: true if pose prevents animation (e.g. wings folded, curled up)
  • augment_prompt: describe what to change if needs_augmentation is true

POSITION GUIDE (use FULL range 0.0–1.0):
  • Chest:     y≈0.55
  • Shoulders: x≈0.15 (left), x≈0.85 (right), y≈0.45–0.55
  • Elbows:    x≈0.08 (left), x≈0.92 (right), y≈0.40–0.45
  • Hands:     x=0.0  (left), x=1.0  (right), y≈0.35
  • Pelvis:    y≈0.20–0.25, parent of all leg joints
  • Hips:      x≈0.38 (left), x≈0.62 (right), y≈0.22
  • Knees:     x≈0.38 (left), x≈0.62 (right), y≈0.15
  • Feet:      x≈0.35 (left), x≈0.65 (right), y≈0.0

CRITICAL RULES:
  - DO NOT cluster joints near center (x=0.5) — spread across FULL WIDTH
  - Left side clearly left (x < 0.5), right side clearly right (x > 0.5)
  - Extremities at actual mesh edges (x≈0.0 or x≈1.0)
  - All joint names UNIQUE
  - Complete chains for BOTH sides
  - Shoulders are children of CHEST, not spine or pelvis
  - Shoulder positioned at LEFT/RIGHT OUTER EDGE of mesh, not body center
  - Do NOT add joints for facial features (jaw, teeth, tongue, eyes, ears)
    — these are static parts of the mesh, not articulated limbs

SKELETON HIERARCHY vs PHYSICAL POSITION:
  ✗ WRONG:   shoulder_left at chest center (x=0.5, y=0.5)
  ✓ CORRECT: shoulder_left at left branch point (x=0.15, y=0.45),
             skeleton still shows chest → shoulder_left
"""

