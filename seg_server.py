"""
seg_server.py

Flask server for MIT App Inventor image processing pipeline.

Routes
------
  GET  /health
  POST /segment                       remove background (rembg)
  POST /classify?tag=&force=          identify object, assess augmentation need
  POST /augment_image?classify_id=    generate two augmented variants via fal.ai
  POST /augment_image/confirm?classify_id=&choice=a|b  lock in chosen variant
  POST /joints?classify_id=&joints=&force=  place skeleton joints (repeatable)
  POST /mesh?classify_id=&force=      generate 3D mesh via Meshy (cached)
  GET  /mesh/status/<task_id>
  POST /rig?classify_id=&user_id=&force=  rig mesh with Blender (repeatable)
  GET  /rig/status/<task_id>
  GET  /results/<filename>
  GET  /gallery?user_id=&tag=&format=
  GET  /gallery_page
  GET  /gallery_data?user_id=
  POST /decimate?ratio=
  POST /convert_to_usdz?glb_url=

Pipeline
--------
  1. /segment          image → segmented PNG (stateless)
  2. /classify         segmented PNG → object_type, category, needs_augmentation
  3. /augment_image    [if needs_augmentation] → two improved PNGs
     /augment_image/confirm  → lock in chosen PNG as active_image_path
  4. /joints           active image → joint_hints, skeleton   ← iterate freely
  5. /mesh             active image → GLB                     ← cached after first run
  6. /rig              mesh + joints → rigged GLB             ← iterate freely

File naming — everything keyed on classify_id:
  {classify_id}_segmented.png
  {classify_id}_augmented_a.png / _b.png
  {classify_id}_mesh.glb / .usdz
  {classify_id}_decimated.glb
  {classify_id}_rigged.glb
  {classify_id}_viz.glb
  {classify_id}_skeleton.json

Environment variables
---------------------
  GEMINI_API_KEY, CLAUDE_API_KEY, OPENAI_API_KEY
  FAL_KEY, MESHY_API_KEY
  BLENDER_PATH   (default: /Applications/Blender.app/Contents/MacOS/blender)
  PIPELINE_STORE_BACKEND  json | tinydb | clouddb
  RESULTS_DIR    (default: results)

Bugs fixed from doc6 version
-----------------------------
  1. import model_store → import pipeline_store as ps
  2. _store.upsert_classify() called with unknown kwargs user_id/active_image_path
     → upsert_classify only accepts (classify_id, tag, info); user_id stored
       separately; active_image_path set by upsert_classify internally
  3. _build_joints_prompt in utils (doc7 version) referenced undefined locals
     tag_context / bounds_info / joints_instruction / bounds_info_text
     → utils.py already has the correct clean version; seg_server just calls it
  4. Route was /infer_joints — renamed back to /joints to match client calls
"""

import os
import sys
import io
import re
import base64
import json
import logging
import time
import uuid
import hashlib
import subprocess
import tempfile
import textwrap
import threading
import struct

import numpy as np
import requests
import urllib3
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from rembg import remove, new_session
from PIL import Image

import utils
import pipeline_store as ps          # FIX 1: was "import model_store as ms"
from pipeline_store import _local_url

# ── App setup ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.environ.get('RESULTS_DIR', 'results')

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = app.logger

os.makedirs(RESULTS_DIR, exist_ok=True)

dummy_user_id = "fb712dd7-73cc-43a5-8158-74f7cb8a7fb4"

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



# ── Singletons ─────────────────────────────────────────────────────────────────

_store        = ps.get_store()        # FIX 1: was ms.get_store()
_rig_tasks    = {}
_mesh_tasks   = {}
_blender_lock = threading.Lock()

# ── rembg ──────────────────────────────────────────────────────────────────────

log.info("Loading rembg model...")
rembg_session = new_session("u2net")
log.info("rembg ready.")


# ══════════════════════════════════════════════════════════════════════════════
# Error handlers
# ══════════════════════════════════════════════════════════════════════════════

@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({'error': 'File too large, max 64MB'}), 413


# ══════════════════════════════════════════════════════════════════════════════
# Vision helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_json(text: str) -> dict | None:
    """Extract JSON from a model response, tolerating markdown fences and
    trailing commas."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*',     '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError as e:
            log.debug(f"JSON parse error at {e.pos}: {e.msg}")
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ── Per-model callers (shared by classify and joints) ─────────────────────────

def _try_claude(img_base64: str, mime_type: str, prompt: str,
                max_tokens: int = 4096) -> dict | None:
    import anthropic
    api_key = os.environ.get('CLAUDE_API_KEY')
    if not api_key:
        return None
    img_base64 = img_base64.replace('\n', '').replace('\r', '').replace(' ', '')
    try:
        base64.b64decode(img_base64, validate=True)
    except Exception:
        log.warning("Claude: base64 validation failed")
        return None

    media_map  = {'image/jpeg': 'image/jpeg', 'image/png': 'image/png',
                  'image/webp': 'image/webp', 'image/gif': 'image/gif'}
    media_type = media_map.get(mime_type, 'image/png')
    client     = anthropic.Anthropic(api_key=api_key)

    for model in ['claude-sonnet-4-6', 'claude-haiku-4-5-20251001']:
        for attempt in range(2):
            try:
                resp = client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{'role': 'user', 'content': [
                        {'type': 'image',
                         'source': {'type': 'base64',
                                    'media_type': media_type,
                                    'data': img_base64}},
                        {'type': 'text', 'text': prompt},
                    ]}]
                )
                result = _extract_json(resp.content[0].text)
                if result:
                    return result
            except anthropic.NotFoundError:
                break
            except anthropic.RateLimitError:
                if attempt == 0:
                    time.sleep(30)
            except anthropic.BadRequestError as e:
                log.warning(f"Claude 400: {e.message}")
                if attempt == 1:
                    break
            except Exception as e:
                log.warning(f"Claude attempt {attempt}: {e}")
                if attempt == 1:
                    break
    return None


def _try_gemini(img_bytes: bytes, mime_type: str, prompt: str) -> dict | None:
    from google import genai
    from google.genai import types
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        return None
    client = genai.Client(api_key=api_key)
    for model in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                        prompt,
                    ]
                )
                raw = resp.text.strip()
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
                result = _extract_json(raw)
                if result:
                    return result
            except Exception as e:
                err = str(e)
                if '429' in err or 'RESOURCE_EXHAUSTED' in err:
                    m    = re.search(r'retryDelay.*?(\d+)s', err)
                    wait = max(int(m.group(1)) if m else 0, 15 * (attempt + 1))
                    if 'PerDay' in err or wait > 60:
                        break
                    time.sleep(wait)
                elif '503' in err or 'UNAVAILABLE' in err:
                    time.sleep(5 * (attempt + 1))
                elif attempt == 2:
                    log.warning(f"Gemini {model} failed: {err[:100]}")
    return None


def _try_openai(img_base64: str, mime_type: str, prompt: str,
                max_tokens: int = 2000) -> dict | None:
    import openai
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return None
    client = openai.OpenAI(api_key=api_key)
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model='gpt-4o-mini', max_tokens=max_tokens,
                messages=[{'role': 'user', 'content': [
                    {'type': 'image_url',
                     'image_url': {'url': f'data:{mime_type};base64,{img_base64}'}},
                    {'type': 'text', 'text': prompt},
                ]}]
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith('```'):
                raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
            result = _extract_json(raw)
            if result:
                return result
        except openai.RateLimitError:
            if attempt == 0:
                time.sleep(30)
        except Exception as e:
            log.warning(f"OpenAI attempt {attempt}: {e}")
    return None


# ── classify_with_vision: identify object ─────────────────────────────────────

def classify_with_vision(img_bytes: bytes, mime_type: str,
                         user_tag: str | None = None) -> dict:
    """
    Identify object_type, category, needs_augmentation.
    Does NOT place joints — call /joints separately.
    Uses utils._build_classify_prompt() or utils._build_vehicle_prompt().
    """
    img = Image.open(io.BytesIO(img_bytes))
    img = utils.resize_if_needed(img, max_size=1024)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    img_bytes = buf.getvalue()

    mime_type = utils.detect_mime_type(img_bytes)
    tag_words = set((user_tag or '').lower().split('+'))
    tag_ctx   = f'\nThe user identified this as: "{user_tag}".' if user_tag else ''

    prompt = (
        utils._build_vehicle_prompt()
        if tag_words & utils.VEHICLE_KEYWORDS
        else utils._build_classify_prompt(tag_ctx)
    )

    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    for label, fn, args in [
        ('Claude', _try_claude, (img_base64, mime_type, prompt)),
        ('Gemini', _try_gemini, (img_bytes,  mime_type, prompt)),
        ('OpenAI', _try_openai, (img_base64, mime_type, prompt)),
    ]:
        log.info(f"classify: trying {label}...")
        result = fn(*args)
        if result:
            log.info(f"classify: {label} succeeded")
            return result

    raise RuntimeError("All vision APIs exhausted for classify")


# ── classify_joints_with_vision: place joints ─────────────────────────────────

_MIN_JOINTS = 3
_MAX_JOINTS = 16


def _validate_joints(data: dict) -> bool:
    return (isinstance(data, dict)
            and isinstance(data.get('joint_hints'), list)
            and len(data['joint_hints']) >= 1)


def classify_joints_with_vision(img_bytes: bytes, mime_type: str,
                                 object_type: str, category: str,
                                 requested_joints: str | None = None,
                                 mesh_bounds: dict | None = None,
                                 rig_type: str = ""
                                 ) -> tuple[dict, str]:
    """
    Ask a vision model to place joints on the active image.
    object_type and category come from /classify — no re-identification.
    mesh_bounds: optional dict with width/height/depth in world units from the
                 mesh GLB, injected into the prompt so the model calibrates
                 coordinates to the full 3D mesh rather than the 2D image frame.

    Returns (joints_dict, model_name_used).
    Raises RuntimeError if all models fail.
    """
    n_joints = None
    if requested_joints:
        try:
            n_joints = max(_MIN_JOINTS, min(_MAX_JOINTS, int(requested_joints)))
        except (ValueError, TypeError):
            pass

    img = Image.open(io.BytesIO(img_bytes))
    img = utils.resize_if_needed(img, max_size=1024)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    img_bytes  = buf.getvalue()
    mime_type  = utils.detect_mime_type(img_bytes)
    prompt     = utils._build_joints_prompt(object_type, category, n_joints,
                                             mesh_bounds=mesh_bounds)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    for label, fn, args in [
        ('Claude', _try_claude, (img_base64, mime_type, prompt, 2048)),
        ('Gemini', _try_gemini, (img_bytes,  mime_type, prompt)),
        ('OpenAI', _try_openai, (img_base64, mime_type, prompt, 2000)),
    ]:
        log.info(f"joints: trying {label}...")
        result = fn(*args)
        if result and _validate_joints(result):
            return result, label.lower()

    raise RuntimeError("All vision APIs exhausted for joint placement")


# ══════════════════════════════════════════════════════════════════════════════
# fal.ai augmentation
# ══════════════════════════════════════════════════════════════════════════════

def edit_image_fal(img: Image.Image, prompt: str) -> tuple[Image.Image, Image.Image]:
    """Edit image using fal.ai Qwen Image 2.0. Returns two variants."""
    import fal_client
    data_uri = f"data:image/png;base64,{utils.img_to_b64(img)}"
    log.info(f"fal.ai prompt: {prompt[:80]}...")
    result = fal_client.subscribe(
        "fal-ai/qwen-image-2/edit",
        arguments={"prompt": prompt, "image_urls": [data_uri], "num_images": 2}
    )
    resp_a = requests.get(result["images"][0]["url"], verify=False)
    resp_b = requests.get(result["images"][1]["url"], verify=False)
    return (Image.open(io.BytesIO(resp_a.content)).convert('RGB'),
            Image.open(io.BytesIO(resp_b.content)).convert('RGB'))


# ══════════════════════════════════════════════════════════════════════════════
# Meshy / file helpers
# ══════════════════════════════════════════════════════════════════════════════

def meshy_reconstruct(img: Image.Image, object_type: str) -> tuple[str, str, str | None]:
    """Submit to Meshy, poll until done. Returns (task_id, glb_url, usdz_url)."""
    meshy_key = os.environ.get('MESHY_API_KEY')
    if not meshy_key:
        raise RuntimeError("MESHY_API_KEY not set")

    headers  = {"Authorization": f"Bearer {meshy_key}"}
    ot_lower = object_type.lower()
    pose_mode = (
        "t-pose" if any(w in ot_lower for w in ['human', 'person', 'humanoid'])
        else "a-pose" if any(w in ot_lower for w in
                             ['bird', 'dog', 'cat', 'horse', 'crab',
                              'fish', 'animal', 'creature'])
        else ""
    )
    payload = {
        "image_url":      f"data:image/png;base64,{utils.img_to_b64(img)}",
        "ai_model":       "meshy-6",
        "should_texture": True,
        "should_remesh":  False,
        "symmetry_mode":  "auto",
    }
    if pose_mode:
        payload["pose_mode"] = pose_mode

    log.info(f"Meshy submit: pose_mode={pose_mode or 'none'}")
    resp = requests.post("https://api.meshy.ai/openapi/v1/image-to-3d",
                         headers=headers, json=payload)
    resp.raise_for_status()
    task_id = resp.json()["result"]
    log.info(f"Meshy task: {task_id}")

    elapsed = 0
    while elapsed < 300:
        time.sleep(5)
        elapsed += 5
        poll = requests.get(
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
            headers=headers
        )
        poll.raise_for_status()
        task     = poll.json()
        status   = task["status"]
        progress = task.get("progress", 0)
        log.info(f"Meshy {task_id}: {status} ({progress}%)")
        if status == "SUCCEEDED":
            urls = task["model_urls"]
            return task_id, urls.get("glb"), urls.get("usdz")
        elif status == "FAILED":
            raise RuntimeError(
                f"Meshy failed: {task.get('task_error', {}).get('message', 'Unknown')}"
            )

    raise RuntimeError("Meshy timed out after 5 minutes")


def download_file(url: str, dest_path: str):
    log.info(f"Downloading: {url}")
    resp = requests.get(url, verify=False, timeout=120)
    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(resp.content)
    log.info(f"Saved: {dest_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Blender / rig helpers
# ══════════════════════════════════════════════════════════════════════════════

def _blender_bin() -> str:
    return os.environ.get('BLENDER_PATH',
                          '/Applications/Blender.app/Contents/MacOS/blender')


def _decimate_mesh(input_path: str, output_path: str, ratio: float = 0.1):
    script = textwrap.dedent(f"""
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=r'{input_path}')
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                bpy.context.view_layer.objects.active = obj
                mod = obj.modifiers.new('Decimate', 'DECIMATE')
                mod.ratio = {ratio}
                bpy.ops.object.modifier_apply(modifier='Decimate')
                print(f'Decimated to {{len(obj.data.vertices)}} vertices')
        bpy.ops.export_scene.gltf(filepath=r'{output_path}', export_format='GLB')
        print('Decimate done')
    """).strip()
    sf = tempfile.mktemp(suffix='.py')
    with open(sf, 'w') as f:
        f.write(script)
    result = subprocess.run([_blender_bin(), '--background', '--python', sf],
                            capture_output=True, text=True, timeout=120)
    os.unlink(sf)
    if not os.path.exists(output_path):
        raise RuntimeError(f"Decimation failed: {result.stderr[-300:]}")
    log.info(f"Decimation complete: {output_path}")


def run_skeleton_inference(glb_path: str, rigged_path: str,
                           n_joints: str | None = None) -> str:
    rig_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rig.py')
    if not os.path.exists(rig_script):
        raise RuntimeError(f"rig.py not found at {rig_script}")
    cmd = [sys.executable, rig_script,
           '--input',  os.path.abspath(glb_path),
           '--output', os.path.abspath(rigged_path),
           '--viz-only']
    if n_joints:
        cmd += ['--joints', str(n_joints)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"rig.py failed: {result.stderr[:200]}")
    json_path = rigged_path.replace('.glb', '_skeleton.json')
    if not os.path.exists(json_path):
        raise RuntimeError(f"Skeleton JSON not created: {json_path}")
    return json_path


def run_blender_rig(glb_path: str, json_path: str, rigged_path: str):
    with _blender_lock:
        rig_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rig.py')
        cmd = [_blender_bin(), '--background', '--python', rig_script, '--',
               '--from-json', os.path.abspath(json_path),
               '--input',     os.path.abspath(glb_path),
               '--output',    os.path.abspath(rigged_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        log.info(result.stdout[-2000:])
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed: {result.stderr[-200:]}")
        if not os.path.exists(rigged_path):
            raise RuntimeError(f"Rigged GLB not created: {result.stdout[-500:]}")


def joints_from_model(joints_data: dict, glb_path: str):
    """
    Map normalised joint positions (0–1) from the vision model onto mesh
    world-space coordinates via bounding box interpolation.
    Returns (joints, hierarchy, hint_objects).

    Claude always outputs:
      x = left/right (0=left, 1=right)
      y = bottom/top (0=bottom, 1=top)
      z = ignored (we force center depth)

    But the mesh may be Y-up or Z-up depending on how Meshy exported it.
    We detect the actual vertical axis as the one with the largest range,
    then map Claude's semantic y → that axis and force the depth axis to 0.5.
    """
    import trimesh

    hint_objects = joints_data.get('joint_hints', [])
    if not hint_objects or isinstance(hint_objects[0], str):
        return None, None, hint_objects

    mesh   = trimesh.load(glb_path, force='mesh')
    verts  = np.array(mesh.vertices)
    bmin   = verts.min(axis=0)
    bmax   = verts.max(axis=0)
    brange = bmax - bmin
    brange[brange == 0] = 1.0

    # Detect mesh orientation: Y-up (standard) vs Z-up (some Meshy exports)
    # We check explicitly rather than sorting by range, because X (side) and
    # Z (depth) can have similar ranges which causes the sort to assign them wrong.
    if brange[2] > brange[1] * 1.2:
        # Z range is significantly larger than Y → Z-up mesh
        vert_axis  = 2   # Z → vertical (Claude's y)
        side_axis  = 0   # X → left/right (Claude's x)
        depth_axis = 1   # Y → depth (forced to center)
    else:
        # Standard Y-up mesh
        vert_axis  = 1   # Y → vertical (Claude's y)
        side_axis  = 0   # X → left/right (Claude's x)
        depth_axis = 2   # Z → depth (forced to center)

    axis_names = ['x', 'y', 'z']
    log.info(f"Mesh axis mapping: vertical={axis_names[vert_axis]} "
             f"side={axis_names[side_axis]} depth={axis_names[depth_axis]} "
             f"ranges={brange.round(3)}")

    name_to_idx = {h['name']: i for i, h in enumerate(hint_objects)}

    joints = []
    for hint in hint_objects:
        p       = hint.get('position_normalized', {})
        norm_x  = np.clip(p.get('x', 0.5), 0.0, 1.0)
        norm_y  = np.clip(p.get('y', 0.5), 0.0, 1.0)

        # Build world position using detected axis mapping
        norm = np.array([0.5, 0.5, 0.5])
        norm[side_axis]  = norm_x   # Claude x → left/right axis
        norm[vert_axis]  = norm_y   # Claude y → vertical axis
        norm[depth_axis] = 0.5      # center depth always

        joints.append(tuple(bmin + norm * brange))

    hierarchy = []
    for bone in joints_data.get('skeleton', []):
        parent_ref = bone.get('parent')
        child_ref  = bone.get('child')
        if isinstance(parent_ref, int) and parent_ref < len(hint_objects):
            parent_ref = hint_objects[parent_ref]['name']
        if isinstance(child_ref, int) and child_ref < len(hint_objects):
            child_ref = hint_objects[child_ref]['name']
        p = name_to_idx.get(parent_ref)
        c = name_to_idx.get(child_ref)
        if p is not None and c is not None:
            hierarchy.append((p, c))

    return joints, hierarchy, hint_objects


def run_vehicle_pipeline(classify_id: str, glb_path: str,
                         classify_data: dict, host: str) -> str:
    seg_dir        = os.path.dirname(os.path.abspath(__file__))
    vehicle_dir = os.path.join(seg_dir, 'vehicle')
    separated_path = os.path.join(RESULTS_DIR, f"{classify_id}_separated.glb")
    animated_path  = os.path.join(RESULTS_DIR, f"{classify_id}_animated.glb")
    rigged_path    = os.path.join(RESULTS_DIR, f"{classify_id}_rigged.glb")
    classify_json  = os.path.join(RESULTS_DIR, f"{classify_id}_classify.json")
    tire_verts     = os.path.join(RESULTS_DIR, f"{classify_id}_tire_verts.json")
    texture_path   = os.path.join(RESULTS_DIR, f"{classify_id}_texture.png")

    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)

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
        log.info(f"Texture extracted: {texture_path}")
        break

    for script_name, input_path, out_path, runner in [
        ('find_tire_verts.py',  glb_path,        tire_verts,     'python'),
        ('classify_wheels.py',  glb_path,        separated_path, 'blender'),
        ('animatesam.py',       separated_path,  animated_path,  'blender'),
        ('merge_animations.py', animated_path,   rigged_path,    'python'),
    ]:
        script = os.path.join(vehicle_dir, script_name)
        if runner == 'blender':
            extra = [tire_verts] if script_name == 'classify_wheels.py' else []
            cmd   = [_blender_bin(), '--background', '--factory-startup',
                     '--python', script, '--',
                     input_path, out_path, classify_json] + extra
        else:
            args = ([glb_path, classify_json, tire_verts, texture_path]
                    if script_name == 'find_tire_verts.py'
                    else [animated_path, rigged_path])
            cmd  = [sys.executable, script] + args

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        log.info(f"separated_path exists: {os.path.exists(separated_path)} → {separated_path}")
        log.info(f"classify_wheels returncode: {result.returncode}")
        log.info(f"classify_wheels stderr: {result.stderr[-300:]}")
        log.info(result.stdout[-4000:])
        if result.returncode != 0:
            log.error(f"classify_wheels stderr: {result.stderr[-500:]}")

        if not os.path.exists(out_path):
            raise RuntimeError(f"{script_name} failed: {result.stderr[-200:]}")

    log.info(f"Vehicle pipeline complete: {rigged_path}")
    return rigged_path


def mesh_guided_joint_correction(joints_data: dict, glb_path: str,
                                  rig_type: str) -> dict:
    """
    Use actual mesh geometry to correct and fill in Claude's joint placement.
    Handles cases where Claude misses joints or places them incorrectly
    due to unusual aspect ratios or novel object types.
    """
    import trimesh

    hints = joints_data.get('joint_hints', [])
    if not hints:
        return joints_data

    mesh   = trimesh.load(glb_path, force='mesh')
    verts  = np.array(mesh.vertices)
    bmin   = verts.min(axis=0)
    bmax   = verts.max(axis=0)
    brange = bmax - bmin
    brange[brange == 0] = 1.0

    hint_map = {h['name']: h for h in hints}
    rt = (rig_type or '').lower()

    def world_to_norm(v):
        return {
            'x': float(np.clip((v[0] - bmin[0]) / brange[0], 0.0, 1.0)),
            'y': float(np.clip((v[1] - bmin[1]) / brange[1], 0.0, 1.0)),
            'z': 0.5,
        }

    if rt == 'flying':
        # Wing tips are the leftmost/rightmost mesh vertices — geometrically exact
        left_tip_vert  = verts[np.argmin(verts[:, 0])]
        right_tip_vert = verts[np.argmax(verts[:, 0])]


        # Center all spine joints at x=0.5 regardless of image perspective
        for name in ['joint_root', 'joint_pelvis', 'joint_spine',
                     'joint_chest', 'joint_neck', 'joint_head']:
            if name in hint_map:
                old_x = hint_map[name]['position_normalized']['x']
                if abs(old_x - 0.5) > 0.03:  # only correct if meaningfully off-center
                    hint_map[name]['position_normalized']['x'] = 0.5
                    log.info(f"  GeoCorrect {name} X: {old_x:.3f}→0.500 (spine centering)")
                

        for tip_name, tip_vert in [
            ('joint_wing_tip_left',  left_tip_vert),
            ('joint_wing_tip_right', right_tip_vert),
        ]:

            norm = world_to_norm(tip_vert)
            if tip_name in hint_map:
                old_x = hint_map[tip_name]['position_normalized']['x']
                hint_map[tip_name]['position_normalized']['x'] = norm['x']
                log.info(f"  GeoCorrect {tip_name} X: {old_x:.3f}→{norm['x']:.3f} "
                         f"(mesh extremity)")
            else:
                ref = next((h for h in hints if 'wing_tip' in h.get('name', '')), None)
                new_hint = {
                    'name':                tip_name,
                    'body_part':           'wing_tip',
                    'deforms_mesh':        ref['deforms_mesh'] if ref else True,
                    'position_normalized': norm,
                }
                hints.append(new_hint)
                hint_map[tip_name] = new_hint
                log.info(f"  GeoCorrect: created missing {tip_name} from geometry")

        # Wing mid — midpoint between base and tip
        for side in ['left', 'right']:
            base_name = f'joint_wing_base_{side}'
            mid_name  = f'joint_wing_mid_{side}'
            tip_name  = f'joint_wing_tip_{side}'
            if all(n in hint_map for n in [base_name, mid_name, tip_name]):
                base_x = hint_map[base_name]['position_normalized']['x']
                tip_x  = hint_map[tip_name]['position_normalized']['x']
                base_y = hint_map[base_name]['position_normalized']['y']
                tip_y  = hint_map[tip_name]['position_normalized']['y']
                hint_map[mid_name]['position_normalized']['x'] = (base_x + tip_x) / 2
                hint_map[mid_name]['position_normalized']['y'] = (base_y + tip_y) / 2
                log.info(f"  GeoCorrect {mid_name} → wing midpoint")

    elif rt in ('biped', 'humanoid', 'other'):
        # Find narrowest Y slice in mid-body = arm/stalk junction
        n_slices = 50
        slice_widths = []
        for i in range(n_slices):
            y_lo = bmin[1] + (i / n_slices) * brange[1]
            y_hi = bmin[1] + ((i + 1) / n_slices) * brange[1]
            sv   = verts[(verts[:, 1] >= y_lo) & (verts[:, 1] < y_hi)]
            if len(sv) > 10:
                width = sv[:, 0].max() - sv[:, 0].min()
                slice_widths.append((i / n_slices + 0.5 / n_slices, width))

        if slice_widths:
            lower_mid = [(y, w) for y, w in slice_widths if 0.25 < y < 0.65]
            if lower_mid:
                stalk_y_norm = min(lower_mid, key=lambda t: t[1])[0]
                for name in ['joint_shoulder_left', 'joint_shoulder_right']:
                    if name in hint_map:
                        old_y = hint_map[name]['position_normalized']['y']
                        if abs(old_y - stalk_y_norm) > 0.08:
                            hint_map[name]['position_normalized']['y'] = float(stalk_y_norm)
                            log.info(f"  GeoCorrect {name} Y: {old_y:.3f}→{stalk_y_norm:.3f} "
                                     f"(stalk junction)")

        # Hands = outermost X vertices in arm Y range
        arm_y_lo  = bmin[1] + 0.25 * brange[1]
        arm_y_hi  = bmin[1] + 0.70 * brange[1]
        arm_verts = verts[(verts[:, 1] >= arm_y_lo) & (verts[:, 1] < arm_y_hi)]
        if len(arm_verts) > 0:
            for name, selector in [
                ('joint_hand_left',  np.argmin),
                ('joint_hand_right', np.argmax),
            ]:
                if name in hint_map:
                    vert  = arm_verts[selector(arm_verts[:, 0])]
                    norm  = world_to_norm(vert)
                    old_x = hint_map[name]['position_normalized']['x']
                    if abs(old_x - norm['x']) > 0.05:
                        hint_map[name]['position_normalized']['x'] = norm['x']
                        log.info(f"  GeoCorrect {name} X: {old_x:.3f}→{norm['x']:.3f} "
                                 f"(mesh extremity)")

        # Feet = bottommost vertices split left/right
        foot_verts = verts[verts[:, 1] < bmin[1] + 0.15 * brange[1]]
        if len(foot_verts) > 10:
            mid_x = (bmin[0] + bmax[0]) / 2
            for name, mask_fn in [
                ('joint_foot_left',  lambda v: v[:, 0] < mid_x),
                ('joint_foot_right', lambda v: v[:, 0] >= mid_x),
            ]:
                if name in hint_map:
                    cluster = foot_verts[mask_fn(foot_verts)]
                    if len(cluster) > 0:
                        norm  = world_to_norm(cluster.mean(axis=0))
                        old_y = hint_map[name]['position_normalized']['y']
                        hint_map[name]['position_normalized']['y'] = norm['y']
                        log.info(f"  GeoCorrect {name} Y: {old_y:.3f}→{norm['y']:.3f} "
                                 f"(foot centroid)")

    elif rt == 'quadruped':
        foot_verts = verts[verts[:, 1] < bmin[1] + 0.15 * brange[1]]
        if len(foot_verts) > 20:
            mid_x = (bmin[0] + bmax[0]) / 2
            mid_z = (bmin[2] + bmax[2]) / 2
            clusters = {
                'joint_foot_front_left':  foot_verts[(foot_verts[:,0]<mid_x) & (foot_verts[:,2]<mid_z)],
                'joint_foot_front_right': foot_verts[(foot_verts[:,0]>=mid_x)& (foot_verts[:,2]<mid_z)],
                'joint_foot_rear_left':   foot_verts[(foot_verts[:,0]<mid_x) & (foot_verts[:,2]>=mid_z)],
                'joint_foot_rear_right':  foot_verts[(foot_verts[:,0]>=mid_x)& (foot_verts[:,2]>=mid_z)],
            }
            for name, cluster in clusters.items():
                if name in hint_map and len(cluster) > 0:
                    norm = world_to_norm(cluster.mean(axis=0))
                    hint_map[name]['position_normalized'] = norm
                    log.info(f"  GeoCorrect {name} → foot centroid")

    return joints_data


def enforce_bilateral_symmetry(joints_data: dict) -> dict:
    """
    For paired joints (left/right), if one side is missing, mirror the other.
    If both exist, enforce same Y height so they're level.
    """
    hints    = joints_data.get('joint_hints', [])
    hint_map = {h['name']: h for h in hints}

    pairs = [
        ('joint_wing_base_left',  'joint_wing_base_right'),
        ('joint_wing_mid_left',   'joint_wing_mid_right'),
        ('joint_wing_tip_left',   'joint_wing_tip_right'),
        ('joint_shoulder_left',   'joint_shoulder_right'),
        ('joint_elbow_left',      'joint_elbow_right'),
        ('joint_hand_left',       'joint_hand_right'),
        ('joint_hip_left',        'joint_hip_right'),
        ('joint_knee_left',       'joint_knee_right'),
        ('joint_foot_left',       'joint_foot_right'),
        ('joint_foot_front_left', 'joint_foot_front_right'),
        ('joint_foot_rear_left',  'joint_foot_rear_right'),
    ]

    for left_name, right_name in pairs:
        left  = hint_map.get(left_name)
        right = hint_map.get(right_name)

        if left and not right:
            lpos = left['position_normalized']
            new_hint = {**left, 'name': right_name,
                'position_normalized': {
                    'x': float(np.clip(1.0 - lpos['x'], 0.0, 1.0)),
                    'y': lpos['y'],
                    'z': lpos.get('z', 0.5),
                }}
            hints.append(new_hint)
            hint_map[right_name] = new_hint
            log.info(f"  Symmetry: mirrored {left_name} → {right_name}")

        elif right and not left:
            rpos = right['position_normalized']
            new_hint = {**right, 'name': left_name,
                'position_normalized': {
                    'x': float(np.clip(1.0 - rpos['x'], 0.0, 1.0)),
                    'y': rpos['y'],
                    'z': rpos.get('z', 0.5),
                }}
            hints.append(new_hint)
            hint_map[left_name] = new_hint
            log.info(f"  Symmetry: mirrored {right_name} → {left_name}")

        elif left and right:
            lpos  = left['position_normalized']
            rpos  = right['position_normalized']
            avg_y = (lpos['y'] + rpos['y']) / 2
            if abs(lpos['y'] - rpos['y']) > 0.05:
                lpos['y'] = avg_y
                rpos['y'] = avg_y
                log.info(f"  Symmetry: leveled {left_name}/{right_name} Y → {avg_y:.3f}")

    return joints_data


# ══════════════════════════════════════════════════════════════════════════════
# Rig pipeline (Blender only — mesh comes from /mesh)
# ══════════════════════════════════════════════════════════════════════════════

def run_rig_pipeline(task_id: str, classify_id: str, user_id: str, host: str):
    """
    Runs Blender rigging using mesh and joints already stored for classify_id.
    Mesh must exist. Joints are read from store; falls back to geometric if absent.
    Repeatable — re-run after /joints to get a new rig on the same mesh.
    """
    try:
        _rig_tasks[task_id] = {'status': 'rigging', 'progress': 10}
        _store.set_rig_status(classify_id, 'rigging')

        record        = _store.get(classify_id)
        classify_data = record.get('classify') or {}
        mesh_data     = record.get('mesh')     or {}
        joints_data   = record.get('joints')   or {}
        rig_type      = classify_data.get('rig_type', '').lower()

        # ── Validate mesh ─────────────────────────────────────────────────────
        glb_path = mesh_data.get('glb_path')
        if not glb_path or not os.path.exists(glb_path):
            raise RuntimeError(
                f"Mesh GLB not found at '{glb_path}' — run /mesh before /rig"
            )

        glb_url     = mesh_data.get('glb_url')
        category    = classify_data.get('category', '')
        object_type = classify_data.get('object_type', '')
        rigid_parts = classify_data.get('rigid_parts', [])
        tag_words   = set(object_type.lower().split())

        is_vehicle = (
            category == 'vehicle' or
            (category not in ('animal', 'humanoid', 'other') and
             bool(tag_words & utils.VEHICLE_KEYWORDS))
        )
        log.info(f"category='{category}' is_vehicle={is_vehicle} object_type='{object_type}' "
                 f"rigid_parts={rigid_parts}")

        decimated_path     = None
        skeleton_json_path = None
        viz_glb_path       = None

        if is_vehicle:
            _rig_tasks[task_id] = {'status': 'rigging', 'progress': 50}
            rigged_path = run_vehicle_pipeline(classify_id, glb_path,
                                               classify_data, host)
        else:
            # ── Decimate ──────────────────────────────────────────────────────
            decimated_path = os.path.join(RESULTS_DIR, f"{classify_id}_decimated.glb")
            if not os.path.exists(decimated_path):
                _rig_tasks[task_id] = {'status': 'decimating', 'progress': 20}
                _decimate_mesh(glb_path, decimated_path, ratio=0.1)
            active_glb  = decimated_path
            rigged_path = os.path.join(RESULTS_DIR, f"{classify_id}_rigged.glb")

            _rig_tasks[task_id] = {'status': 'inferring_skeleton', 'progress': 30}

            # ── Build skeleton from stored joints or geometric fallback ────────
            if joints_data.get('joint_hints'):
                log.info(f"Using stored joints "
                         f"(model={joints_data.get('model_used', '?')}, "
                         f"count={len(joints_data['joint_hints'])})")

                # Map vision-model normalized positions onto mesh world space.
                # Do NOT call run_skeleton_inference here — that runs geometric
                # inference which writes its own 4-joint skeleton JSON and
                # overwrites the vision model's joints before we can use them.
                joints, hierarchy, hint_objects = joints_from_model(
                    joints_data, active_glb
                )

                def _hint_name(h, i):
                    return h.get('name', f'joint_{i}') if isinstance(h, dict) else f'joint_{i}'

                if joints:
                    skel = {
                        'joints': [
                            {'id': i,
                             'name': _hint_name(hint_objects[i], i),
                             'position': list(joints[i]),
                             'hint': hint_objects[i]}
                            for i in range(len(joints))
                        ],
                        'bones': [
                            {'parent': p, 'child': c,
                             'name': (f"{_hint_name(hint_objects[p], p)}"
                                      f"_to_{_hint_name(hint_objects[c], c)}")}
                            for p, c in hierarchy
                            if p < len(hint_objects) and c < len(hint_objects)
                        ],
                        'rigid_parts': rigid_parts,
                    }
                else:
                    # position_normalized was missing/malformed — fall back to
                    # geometric inference as a last resort
                    log.warning("joints_from_model returned no positions — "
                                "falling back to geometric inference")
                    skeleton_json_path = run_skeleton_inference(active_glb, rigged_path)
                    with open(skeleton_json_path) as f:
                        skel = json.load(f)

                # Write the skeleton JSON ourselves — do not let rig.py do it
                skeleton_json_path = rigged_path.replace('.glb', '_skeleton.json')
                with open(skeleton_json_path, 'w') as f:
                    json.dump(skel, f, indent=2)
                log.info(f"Skeleton JSON written from vision joints: {skeleton_json_path}")

            else:
                log.info(f"No joints stored for {classify_id} — geometric inference")
                skeleton_json_path = run_skeleton_inference(active_glb, rigged_path)
                with open(skeleton_json_path) as f:
                    skel = json.load(f)

            # ── Inject keyframes ──────────────────────────────────────────────
            _rig_tasks[task_id] = {'status': 'injecting_keyframes', 'progress': 40}
            skel = utils.inject_keyframes(skel)
            with open(skeleton_json_path, 'w') as f:
                json.dump(skel, f, indent=2)

            # ── Visualisation (non-fatal) ─────────────────────────────────────
            try:
            
                _rig_tasks[task_id] = {'status': 'visualizing', 'progress': 50}
                from rig import visualize_skeleton
                viz_glb_path = os.path.join(RESULTS_DIR, f"{classify_id}_viz.glb")
                visualize_skeleton(
                    active_glb,
                    [tuple(j['position']) for j in skel['joints']],
                    [(b['parent'], b['child']) for b in skel['bones']],
                    viz_glb_path,
                    labels=[j['name'] for j in skel['joints']],
                    labels_raw=[j.get('hint') for j in skel['joints']],
                )
                log.info(f"Skeleton viz: {viz_glb_path}")
            except Exception as e:
                log.warning(f"Viz failed (non-fatal): {e}")

            _rig_tasks[task_id] = {'status': 'rigging_blender', 'progress': 60}
            run_blender_rig(active_glb, skeleton_json_path, rigged_path)

        # ── Update mesh record with decimated path ────────────────────────────
        if decimated_path:
            _store.upsert_mesh(classify_id, {**mesh_data,
                                             'decimated_glb_path': decimated_path})

        # ── Persist ───────────────────────────────────────────────────────────
        _rig_tasks[task_id] = {'status': 'finalizing', 'progress': 90}
        _store.upsert_rig(classify_id, {
            'rigged_glb_path':    rigged_path,
            'viz_glb_path':       viz_glb_path,
            'skeleton_json_path': skeleton_json_path,
            'status':             'ok',
            'user_id':            user_id,
        })

        _rig_tasks[task_id] = {
            'status':      'ok',
            'progress':    100,
            'rigged_url':  _local_url(rigged_path, host),
            'glb_url':     glb_url,
            'classify_id': classify_id,
        }
        log.info(f"Rig task {task_id} complete: {rigged_path}")

    except Exception as e:
        log.error(f"Rig task {task_id} failed: {e}")
        _rig_tasks[task_id] = {'status': 'error', 'error': str(e)}
        _store.set_rig_status(classify_id, 'error', str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Mesh pipeline (Meshy — background thread)
# ══════════════════════════════════════════════════════════════════════════════

def _run_mesh_task(task_id: str, classify_id: str, img: Image.Image,
                   object_type: str, mesh_hash: str, host: str):
    try:
        _mesh_tasks[task_id] = {'status': 'meshy', 'progress': 10}
        meshy_task_id, glb_url, usdz_url = meshy_reconstruct(img, object_type)

        _mesh_tasks[task_id] = {'status': 'downloading', 'progress': 70}
        glb_path  = os.path.join(RESULTS_DIR, f"{classify_id}_mesh.glb")
        usdz_path = None
        download_file(glb_url, glb_path)
        if usdz_url:
            usdz_path = os.path.join(RESULTS_DIR, f"{classify_id}_mesh.usdz")
            download_file(usdz_url, usdz_path)

        _store.upsert_mesh(classify_id, {
            'mesh_hash':     mesh_hash,
            'meshy_task_id': meshy_task_id,
            'glb_path':      glb_path,
            'glb_url':       glb_url,
            'usdz_path':     usdz_path,
            'usdz_url':      usdz_url,
        })

        _mesh_tasks[task_id] = {
            'status':        'ok',
            'progress':      100,
            'glb_url':       glb_url,
            'glb_local_url': _local_url(glb_path, host),
            'classify_id':   classify_id,
        }
        log.info(f"Mesh task {task_id} complete: {glb_path}")

    except Exception as e:
        log.error(f"Mesh task {task_id} failed: {e}")
        _mesh_tasks[task_id] = {'status': 'error', 'error': str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


# ── /segment ──────────────────────────────────────────────────────────────────

@app.route('/segment', methods=['GET', 'POST'])
def segment():
    """Remove background via rembg. Stateless — client holds the bytes."""
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No data received'}), 400
        img    = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img    = utils.resize_if_needed(img, max_size=1024)
        output = remove(img, session=rembg_session)
        buf    = io.BytesIO()
        output.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/segment error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /classify ─────────────────────────────────────────────────────────────────
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    Identify object_type, category, needs_augmentation.
    Does NOT place joints — call /joints separately.
    ?force=true bypasses cache but preserves confirmed augmented image.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        tag     = request.args.get('tag', '').strip()
        force   = request.args.get('force', '').lower() in ('1', 'true', 'yes')
        user_id = request.args.get('user_id', '').strip() or dummy_user_id
 
        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No data received'}), 400
 
        #same hash per img and tag...
        classify_id = hashlib.md5(img_bytes + tag.encode()).hexdigest()[:8]
 
        if not force:
            record = _store.get(classify_id)
            if record and record.get('classify'):
                log.info(f"classify cache hit: {classify_id}")
                return jsonify({
                    **record['classify'],
                    'classify_id':      classify_id,
                    'active_image_path': record.get('active_image_path'),
                })

        log.info(f"classify {'(force) ' if force else ''}running: {classify_id}")
        mime_type = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'
        info      = classify_with_vision(img_bytes, mime_type, tag or None)

        log.info(f"Classification: {info.get('object_type', '?')} | "
                 f"needs_augmentation={info.get('needs_augmentation', False)}")

        seg_path = os.path.join(RESULTS_DIR, f"{classify_id}_segmented.png")
        if not os.path.exists(seg_path):
            img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            img.save(seg_path, format='PNG')
            log.info(f"Segmented image saved: {seg_path}")

        info['segmented_image_path'] = seg_path

        # FIX 2: upsert_classify only takes (classify_id, tag, info).
        # It internally sets active_image_path = seg_path on first call,
        # and preserves it if already pointing at a confirmed augmented image.
        _store.upsert_classify(classify_id, tag, info)

        return jsonify({
            **info,
            'classify_id':      classify_id,
            'active_image_path': seg_path,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/classify error: {e}")
        return jsonify({'error': str(e)}), 500
 
 

# ── /augment_image ────────────────────────────────────────────────────────────

@app.route('/augment_image', methods=['GET', 'POST'])
def augment_image():
    """
    Generate two augmented variants of the active image via fal.ai.
    Reads active_image_path from the store.
    Client calls /augment_image/confirm to lock in choice.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id = request.args.get('classify_id', '').strip()
        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400
 
        record = _store.get(classify_id)
        if not record or not record.get('classify'):
            return jsonify({'error': f"classify_id '{classify_id}' not found — run /classify first"}), 404
 
        active_path = record.get('active_image_path')
        if not active_path or not os.path.exists(active_path):
            return jsonify({'error': f"Active image not found: '{active_path}'"}), 404
 
        classify_data   = record.get('classify') or {}
        object_type     = classify_data.get('object_type', '')
        rig_type        = classify_data.get('rig_type', '').lower()
        stored_augment  = classify_data.get('augment_prompt', '').strip()
        style           = classify_data.get('style', '').strip()
        user_prompt     = request.args.get('tag', '').lower().strip().replace('+', ' ')
 
        # Style preservation anchor — always prepend this so fal.ai doesn't
        # drift from the original appearance. Use the stored style description
        # if available — it captures the exact medium and technique.
        if style:
            style_anchor = (f"Keep the exact same {object_type}. "
                            f"Style: {style}. "
                            f"Do NOT add detail, change the art style, or alter "
                            f"the drawing technique. Only change the pose. ")
        else:
            style_anchor = (f"Keep the exact same {object_type} — identical "
                            f"colors, materials, textures, and visual style. "
                            f"Only change the pose. ")
 
        # Use stored augment_prompt from classify if available,
        # otherwise build a rig-type-appropriate pose prompt
        if stored_augment:
            pose_prompt = stored_augment
        elif user_prompt:
            pose_prompt = user_prompt
        elif rig_type == 'humanoid':
            pose_prompt = ("Repose into a T-pose or A-pose with arms extended "
                           "horizontally for easy 3D rigging.")
        elif rig_type == 'biped':
            pose_prompt = ("Repose standing upright on two legs in a neutral "
                           "A-pose with legs slightly apart, facing forward.")
        elif rig_type == 'flying':
            pose_prompt = ("Repose with wings fully extended horizontally, "
                           "facing forward, legs visible below if present.")
        elif rig_type == 'quadruped':
            pose_prompt = ("Repose standing with all four legs apart and "
                           "clearly visible, facing forward.")
        else:
            pose_prompt = f"Repose the {object_type} in a neutral spread pose for 3D rigging."
 
        prompt = style_anchor + pose_prompt + " Clear white background."
 
        log.info(f"Augment prompt: {prompt[:120]}...")
 
        img        = Image.open(active_path).convert('RGB')
        img        = utils.resize_if_needed(img, max_size=1024)
        img_a, img_b = edit_image_fal(img, prompt)
 
        path_a = os.path.join(RESULTS_DIR, f"{classify_id}_augmented_a.png")
        path_b = os.path.join(RESULTS_DIR, f"{classify_id}_augmented_b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        return jsonify({
            'status':      'ok',
            'classify_id': classify_id,
            'image_a_url': _local_url(path_a, request.host),
            'image_b_url': _local_url(path_b, request.host),
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/augment_image error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /augment_image/confirm ────────────────────────────────────────────────────

@app.route('/augment_image/confirm', methods=['GET', 'POST'])
def augment_image_confirm():
    """Lock in chosen augmented variant. Sets active_image_path in store."""
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id = request.args.get('classify_id', '').strip()
        choice      = request.args.get('choice', '').strip().lower()

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400
        if choice not in ('a', 'b'):
            return jsonify({'error': "choice must be 'a' or 'b'"}), 400

        chosen_path = os.path.join(RESULTS_DIR, f"{classify_id}_augmented_{choice}.png")
        if not os.path.exists(chosen_path):
            return jsonify({'error': "Augmented image not found — run /augment_image first"}), 404

        _store.set_active_image(classify_id, chosen_path)
        log.info(f"Active image → augmented_{choice} for {classify_id}")

        return jsonify({
            'status':           'ok',
            'classify_id':      classify_id,
            'choice':           choice,
            'active_image_url': _local_url(chosen_path, request.host),
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/augment_image/confirm error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /infer_joints ─────────────────────────────────────────────────────────────

@app.route('/infer_joints', methods=['GET', 'POST'])
def infer_joints():
    """
    Place skeleton joints on the active image using a vision model.
    Freely repeatable — each call overwrites previous result.
    Falls back to geometric inference if vision fails (requires /mesh first).

    ?classify_id=  required
    ?joints=N      optional hint (3–16), passed to vision model as suggestion
    ?force=true    bypass cache, re-run vision model

    If a mesh already exists for this classify_id, its bounding box dimensions
    are extracted and injected into the vision prompt so the model can reason
    about the 3D proportions of the full mesh rather than guessing from the
    2D image frame alone. This is the key fix for joints placed relative to
    the visible image crop rather than the full mesh extents.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id      = request.args.get('classify_id', '').strip()
        force            = request.args.get('force', '').lower() in ('1', 'true', 'yes')
        requested_joints = request.args.get('joints', '').strip() or None

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400

        record = _store.get(classify_id)
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found — run /classify first"}), 404

        if not force and record.get('joints'):
            log.info(f"joints cache hit: {classify_id}")
            return jsonify({
                **record['joints'],
                'classify_id':       classify_id,
                'active_image_path': record.get('active_image_path'),
            })

        classify_data = record.get('classify') or {}
        active_path   = record.get('active_image_path')

        if not active_path or not os.path.exists(active_path):
            return jsonify({'error': f"Active image not found: '{active_path}'"}), 404

        with open(active_path, 'rb') as f:
            img_bytes = f.read()

        mime_type   = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'
        object_type = classify_data.get('object_type', '')
        category    = classify_data.get('category', '')
        rig_type    = classify_data.get('rig_type', '')

        # ── Extract mesh bounding box if available ────────────────────────────
        mesh_bounds = None
        mesh_data   = record.get('mesh') or {}
        glb_path    = mesh_data.get('glb_path') or mesh_data.get('decimated_glb_path')

        if glb_path and os.path.exists(glb_path):
            try:
                import trimesh
                mesh   = trimesh.load(glb_path, force='mesh')
                verts  = np.array(mesh.vertices)
                bmin   = verts.min(axis=0)
                bmax   = verts.max(axis=0)
                brange = bmax - bmin
                # Meshy exports Y-up. x=left/right, y=bottom/top, z=front/back.
                # The y axis should be the tallest — log all three so we can
                # spot if the mesh has an unexpected up-axis.
                mesh_bounds = {
                    'width':  float(brange[0]),   # x: left→right
                    'height': float(brange[1]),   # y: bottom→top (up)
                    'depth':  float(brange[2]),   # z: front→back
                    'bmin':   bmin.tolist(),
                    'bmax':   bmax.tolist(),
                }
                tallest = max(enumerate(brange), key=lambda t: t[1])
                axis_names = ['x', 'y', 'z']
                if tallest[0] != 1:
                    log.warning(f"Mesh up-axis may not be Y: tallest axis is "
                                f"{axis_names[tallest[0]]} ({tallest[1]:.3f}), "
                                f"y={brange[1]:.3f}")
                log.info(f"Mesh bounds: x={brange[0]:.3f} y={brange[1]:.3f} "
                         f"z={brange[2]:.3f} (tallest={axis_names[tallest[0]]})")

            except Exception as e:
                log.warning(f"Could not extract mesh bounds: {e}")

        log.info(f"Placing joints for: '{object_type}' ({category}) "
                 f"requested={requested_joints} "
                 f"mesh_bounds={'yes' if mesh_bounds else 'no'}")

        # ── Vision model ──────────────────────────────────────────────────────
        joints_info = None
        model_used  = 'unknown'
        try:
            joints_info, model_used = classify_joints_with_vision(
                img_bytes, mime_type, object_type, category,
                requested_joints=requested_joints,
                mesh_bounds=mesh_bounds,
                rig_type=rig_type,
            )
        except Exception as e:
            log.warning(f"/infer_joints vision failed: {e} — trying geometric fallback")

        # ── Geometric fallback ────────────────────────────────────────────────
        if not joints_info:
            glb_path = glb_path or (mesh_data.get('glb_path') if mesh_data else None)
            if not glb_path or not os.path.exists(glb_path):
                return jsonify({
                    'error': ('Vision model unavailable and no mesh for geometric '
                              'fallback — run /mesh first, then retry /infer_joints')
                }), 422

            from rig import infer_skeleton_geometric
            import trimesh
            n = int(requested_joints) if requested_joints else None
            raw_joints, hierarchy, _ = infer_skeleton_geometric(glb_path, n)

            # raw_joints are world-space coordinates — normalize to 0–1
            # relative to the mesh bounding box so joints_from_model can
            # map them back correctly.
            if mesh_bounds is None:
                mesh   = trimesh.load(glb_path, force='mesh')
                verts  = np.array(mesh.vertices)
                bmin   = np.array(verts.min(axis=0))
                brange = np.array(verts.max(axis=0)) - bmin
            else:
                bmin   = np.array(mesh_bounds['bmin'])
                brange = np.array(mesh_bounds['bmax']) - bmin
            brange[brange == 0] = 1.0

            model_used  = 'geometric'
            joints_info = {
                'joint_hints': [
                    {'name': f'joint_{i}',
                     'body_part': f'joint_{i}',
                     'position_normalized': {
                         'x': float(np.clip((j[0] - bmin[0]) / brange[0], 0.0, 1.0)),
                         'y': float(np.clip((j[1] - bmin[1]) / brange[1], 0.0, 1.0)),
                         'z': float(np.clip((j[2] - bmin[2]) / brange[2], 0.0, 1.0)),
                     },
                     'animations': []}
                    for i, j in enumerate(raw_joints)
                ],
                'skeleton': [{'parent': p, 'child': c,
                               'name': f'joint_{p}_to_joint_{c}'}
                             for p, c in hierarchy],
                'suggested_joints': len(raw_joints),
            }

        joints_data = {
            **joints_info,
            'source_image_path': active_path,
            'model_used':        model_used,
        }
        if glb_path and os.path.exists(glb_path):
            try:
                joints_data = snap_joints_to_mesh(joints_data, glb_path)
            except Exception as e:
                log.warning(f"Snapping failed (non-fatal): {e}")

            try:
                rig_type = classify_data.get('rig_type', '')
                joints_data = mesh_guided_joint_correction(
                    joints_data, glb_path, rig_type)
            except Exception as e:
                log.warning(f"Mesh-guided correction failed (non-fatal): {e}")

            try:
                joints_data = enforce_bilateral_symmetry(joints_data)
            except Exception as e:
                log.warning(f"Symmetry enforcement failed (non-fatal): {e}")

            try:
                viz_path = os.path.join(RESULTS_DIR,
                                        f"{classify_id}_joints_normalized_viz.glb")
                visualize_normalized_joints(joints_data, glb_path, viz_path)
            except Exception as e:
                log.warning(f"Normalized joints viz failed (non-fatal): {e}")

        _store.upsert_joints(classify_id, joints_data)
        log.info(f"Joints stored: {classify_id} "
                 f"({len(joints_info['joint_hints'])} hints, model={model_used})")

        return jsonify({
            **joints_data,
            'classify_id':       classify_id,
            'active_image_path': active_path,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/infer_joints error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /mesh ─────────────────────────────────────────────────────────────────────

@app.route('/mesh', methods=['GET', 'POST'])
def mesh():
    """
    Generate 3D mesh via Meshy. Cached after first run.
    ?force=true regenerates (costs a Meshy credit).
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id = request.args.get('classify_id', '').strip()
        force       = request.args.get('force', '').lower() in ('1', 'true', 'yes')

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400

        record = _store.get(classify_id)
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found — run /classify first"}), 404

        active_path = record.get('active_image_path')
        if not active_path or not os.path.exists(active_path):
            return jsonify({'error': f"Active image not found: '{active_path}'"}), 404

        # ── Cache hit ─────────────────────────────────────────────────────────
        existing_mesh = record.get('mesh') or {}
        if not force and existing_mesh.get('glb_path') and \
                os.path.exists(existing_mesh['glb_path']):
            log.info(f"Mesh cache hit: {classify_id}")
            return jsonify({
                'status':        'ok',
                'task_id':       None,
                'glb_url':       existing_mesh.get('glb_url'),
                'glb_local_url': _local_url(existing_mesh['glb_path'], request.host),
                'classify_id':   classify_id,
            })

        with open(active_path, 'rb') as f:
            img_bytes = f.read()

        classify_data = record.get('classify') or {}
        object_type   = classify_data.get('object_type', '') or \
                        request.args.get('type', '').lower().strip().replace('+', ' ')
        img       = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img       = utils.resize_if_needed(img, max_size=1024)
        mesh_hash = hashlib.md5(img_bytes + object_type.encode()).hexdigest()[:12]

        # ── Cross-record mesh hash cache ──────────────────────────────────────
        if not force:
            cached = _store.get_mesh_by_hash(mesh_hash)
            if cached:
                log.info(f"Mesh hash cache hit: {mesh_hash}")
                _store.upsert_mesh(classify_id, cached)
                return jsonify({
                    'status':        'ok',
                    'task_id':       None,
                    'glb_url':       cached.get('glb_url'),
                    'glb_local_url': _local_url(cached['glb_path'], request.host),
                    'classify_id':   classify_id,
                })

        task_id = str(uuid.uuid4())[:8]
        _mesh_tasks[task_id] = {'status': 'started', 'progress': 0}
        threading.Thread(
            target=_run_mesh_task,
            args=(task_id, classify_id, img, object_type, mesh_hash, request.host),
            daemon=True,
        ).start()

        return jsonify({'status': 'processing', 'task_id': task_id,
                        'classify_id': classify_id})

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/mesh error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/mesh/status/<task_id>', methods=['GET'])
def mesh_status(task_id: str):
    task = _mesh_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


# ── /rig ──────────────────────────────────────────────────────────────────────

@app.route('/rig', methods=['GET', 'POST'])
def rig():
    """
    Rig mesh using joints from store. Repeatable.
    Falls back to geometric inference if no joints stored.
    ?force=true bypasses rig cache.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id = request.args.get('classify_id', '').strip()
        user_id     = request.args.get('user_id', '').strip() or dummy_user_id
        force       = request.args.get('force', '').lower() in ('1', 'true', 'yes')

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400

        record = _store.get(classify_id)
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found"}), 404

        mesh_data = record.get('mesh') or {}
        glb_path  = mesh_data.get('glb_path')
        if not glb_path or not os.path.exists(glb_path):
            return jsonify({
                'error':       'No mesh found — run /mesh before /rig',
                'classify_id': classify_id,
            }), 422

        rig_data = record.get('rig') or {}
        if not force and rig_data.get('status') == 'ok':
            rigged_path = rig_data.get('rigged_glb_path')
            if rigged_path and os.path.exists(rigged_path):
                log.info(f"Rig cache hit: {classify_id}")
                return jsonify({
                    'status':      'ok',
                    'task_id':     None,
                    'rigged_url':  _local_url(rigged_path, request.host),
                    'glb_url':     mesh_data.get('glb_url'),
                    'classify_id': classify_id,
                })

        if not record.get('joints'):
            log.warning(f"No joints for {classify_id} — geometric fallback will be used")

        task_id = str(uuid.uuid4())[:8]
        _rig_tasks[task_id] = {'status': 'started', 'progress': 0}
        _store.set_rig_status(classify_id, 'started')

        log.info(f"Starting rig task {task_id} for {classify_id}")
        threading.Thread(
            target=run_rig_pipeline,
            args=(task_id, classify_id, user_id, request.host),
            daemon=True,
        ).start()

        return jsonify({'status': 'processing', 'task_id': task_id,
                        'classify_id': classify_id})

    except Exception as e:
        log.error(f"/rig error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/rig/status/<task_id>', methods=['GET'])
def rig_status(task_id: str):
    task = _rig_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)


# ── /decimate ─────────────────────────────────────────────────────────────────

@app.route('/decimate', methods=['GET', 'POST'])
def decimate():
    """Decimate a GLB mesh. Accepts raw bytes or ?glb_url=..."""
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        ratio   = max(0.01, min(1.0, float(request.args.get('ratio', '0.1'))))
        glb_url = request.args.get('glb_url', '').strip()

        if glb_url:
            resp      = requests.get(glb_url, verify=False, timeout=60)
            resp.raise_for_status()
            glb_bytes = resp.content
        else:
            glb_bytes = request.stream.read()

        if not glb_bytes:
            return jsonify({'error': 'No GLB data'}), 400

        uid      = str(uuid.uuid4())[:8]
        in_path  = os.path.join(RESULTS_DIR, f"{uid}_input.glb")
        out_path = os.path.join(RESULTS_DIR, f"{uid}_decimated.glb")
        with open(in_path, 'wb') as f:
            f.write(glb_bytes)
        _decimate_mesh(in_path, out_path, ratio=ratio)
        os.unlink(in_path)

        return jsonify({'status': 'ok',
                        'url':    _local_url(out_path, request.host),
                        'ratio':  ratio})
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/decimate error: {e}")
        return jsonify({'error': str(e)}), 500


# ── /convert_to_usdz ──────────────────────────────────────────────────────────

def convert_glb_to_usdz(glb_path: str, usdz_path: str) -> str:
    """Convert GLB to USDZ using Blender's USD exporter."""
    import textwrap, tempfile, subprocess
    
    script = textwrap.dedent(f"""
        import bpy
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=r'{glb_path}')
        bpy.ops.wm.usd_export(
            filepath=r'{usdz_path}',
            export_animation=True,
            export_textures=True,
            export_materials=True,
        )
        print('USDZ export done')
    """).strip()
    
    sf = tempfile.mktemp(suffix='.py')
    with open(sf, 'w') as f:
        f.write(script)
    result = subprocess.run(
        [_blender_bin(), '--background', '--python', sf],
        capture_output=True, text=True, timeout=120
    )
    os.unlink(sf)
    
    if not os.path.exists(usdz_path):
        raise RuntimeError(f"USDZ export failed: {result.stderr[-300:]}")
    
    log.info(f"Converted {glb_path} → {usdz_path}")
    return usdz_path
    
@app.route('/convert_to_usdz', methods=['GET', 'POST'])
def convert_to_usdz():
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        glb_url = request.args.get('glb_url', '').strip()
        if not glb_url:
            return jsonify({'error': 'Missing glb_url parameter'}), 400
        resp = requests.get(glb_url, verify=False, timeout=60)
        resp.raise_for_status()
        uid       = str(uuid.uuid4())[:8]
        glb_path  = os.path.join(RESULTS_DIR, f"{uid}_temp.glb")
        usdz_path = os.path.join(RESULTS_DIR, f"{uid}_converted.usdz")
        with open(glb_path, 'wb') as f:
            f.write(resp.content)
        convert_glb_to_usdz(glb_path, usdz_path)
        os.unlink(glb_path)
        return jsonify({'status': 'ok',
                        'usdz_url': _local_url(usdz_path, request.host)})
    except Exception as e:
        log.error(f"/convert_to_usdz error: {e}")
        return jsonify({'error': str(e)}), 500


# ── Static / gallery ──────────────────────────────────────────────────────────

@app.route('/results/<filename>')
def serve_result(filename):
    return send_file(os.path.join(RESULTS_DIR, filename))


@app.route('/gallery_page')
def gallery_page():
    return send_file(os.path.join(os.path.dirname(__file__), 'gallery.html'))


@app.route('/gallery', methods=['GET'])
def gallery():
    tag     = request.args.get('tag', '').strip().lower()
    fmt     = request.args.get('format', 'json').strip()
    user_id = request.args.get('user_id', '').strip() or dummy_user_id
    try:
        records = (_store.search_by_tag(user_id, tag)
                   if tag else _store.get_by_user(user_id))
        records = [_store.with_urls(r, request.host) for r in records]
        if fmt == 'listview':
            return jsonify([
                f"{r.get('tag', 'model')}|{(r.get('rig') or {}).get('rigged_url', '')}"
                for r in records
            ])
        return jsonify(records)
    except Exception as e:
        log.error(f"/gallery error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/gallery_data', methods=['GET'])
def gallery_data():
    user_id = request.args.get('user_id', '').strip() or dummy_user_id
    try:
        records = _store.get_by_user(user_id)
        return jsonify([{
            'classify_id':     r.get('classify_id'),
            'label':           r.get('tag', 'model'),
            'tags':            r.get('tags', []),
            'rigged_path':     (r.get('rig')      or {}).get('rigged_glb_path'),
            'segmented_image': (r.get('classify') or {}).get('segmented_image_path'),
            'active_image':    r.get('active_image_path'),
            'has_joints':      bool(r.get('joints')),
            'has_mesh':        bool((r.get('mesh') or {}).get('glb_path')),
            'rig_status':      (r.get('rig') or {}).get('status'),
            'created_at':      (r.get('rig') or {}).get('created_at'),
        } for r in records])
    except Exception as e:
        log.error(f"/gallery_data error: {e}")
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)
