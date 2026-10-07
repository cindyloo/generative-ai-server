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
import shutil
import stat

import numpy as np
import requests
import urllib3
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from rembg import remove, new_session
from PIL import Image

import utils
import pipeline_store as ps          # FIX 1: was "import model_store as ms"
from pipeline_store import _local_url, hydrate

# ── App setup ──────────────────────────────────────────────────────────────────

RESULTS_DIR = os.environ.get("RESULTS_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)
EXAMPLES_DIR = os.environ.get("EXAMPLES_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "examples"
)

_MAC_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"

BLENDER_BIN = (
    os.environ.get("BLENDER_BIN")
    or shutil.which("blender")
    or (_MAC_BLENDER if os.path.isfile(_MAC_BLENDER) else "blender")
 )

def _rdir(classify_id: str) -> str:
    """Return (and create) the per-classify_id results subdirectory."""
    d = os.path.join(RESULTS_DIR, classify_id)
    os.makedirs(d, exist_ok=True)
    return d


app = Flask(__name__)
CORS(app)

@app.after_request
def add_ngrok_and_cors_headers(response):
    # Bypass the ngrok browser warning screen
    response.headers["ngrok-skip-browser-warning"] = "true"
    # Ensure browsers allow model-viewer to send the custom ngrok header
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, ngrok-skip-browser-warning"
    return response

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




def _validate_environment():
    """Fail fast with a clear message instead of an opaque PermissionError."""
    # Blender binary must exist and be an executable *file*, not a directory.
    resolved = shutil.which(BLENDER_BIN) or BLENDER_BIN
    if os.path.isdir(resolved):
        raise RuntimeError(
            f"BLENDER_BIN points to a directory ({resolved}). On macOS use "
            f"{resolved}/Contents/MacOS/Blender"
        )
    if not os.path.isfile(resolved):
        raise RuntimeError(f"Blender executable not found at {resolved!r}")
    if not os.access(resolved, os.X_OK):
        # Try to make it executable; otherwise tell the user exactly what to do.
        try:
            os.chmod(resolved, os.stat(resolved).st_mode | stat.S_IXUSR)
        except OSError:
            raise RuntimeError(
                f"{resolved} is not executable. Run: chmod +x {resolved}"
            )
 
    # Results directory must exist and be writable by *this* process.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.access(RESULTS_DIR, os.W_OK):
        raise RuntimeError(
            f"RESULTS_DIR ({RESULTS_DIR}) is not writable by the current user. "
            f"Either chown/chmod it, or set the RESULTS_DIR env var to a "
            f"writable path."
        )
 
    return resolved
 
 
BLENDER_BIN = _validate_environment()
 
 
def _make_workspace():
    """Create a unique, guaranteed-writable working directory."""
    uid = str(uuid.uuid4())[:8]
    ws = os.path.join(RESULTS_DIR, uid)
    os.makedirs(ws, exist_ok=True)
    return ws
 
# ══════════════════════════════════════════════════════════════════════════════
# SAM2 segmentation
# ══════════════════════════════════════════════════════════════════════════════

_sam2_predictor = None

def _get_sam2():
    global _sam2_predictor
    if _sam2_predictor is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2 as sam2_pkg
        sam2_dir        = os.path.dirname(sam2_pkg.__file__)
        sam2_config     = os.environ.get('SAM2_CONFIG',
            os.path.join(sam2_dir, 'configs/sam2.1/sam2.1_hiera_s.yaml'))
        sam2_checkpoint = os.environ.get('SAM2_CHECKPOINT',
            'checkpoints/sam2.1_hiera_small.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model           = build_sam2(sam2_config, sam2_checkpoint, device=device)
        _sam2_predictor = SAM2ImagePredictor(model)
        log.info(f"SAM2 ready ({device}).")
    return _sam2_predictor


def segment_parts_with_sam2(img_path: str, joint_hints: list,
                             classify_id: str, results_dir: str) -> dict:
    """
    Generate per-part SAM2 masks using Hough Circle detection to find the
    true center/radius, then build anatomically-correct prompt points.

    Saves masks to results_dir/<classify_id>_mask_<name>.png
    Saves wheel_centers.json to results_dir/<classify_id>_masks/

    Returns { hint_name: absolute_mask_path }
    """
    import torch
    import cv2
    from PIL import Image as PILImage, ImageDraw
    print("loading sam2")
    predictor = _get_sam2()

    img         = PILImage.open(img_path).convert('RGB')
    img_w, img_h = img.size
    img_rgba    = np.array(PILImage.open(img_path).convert('RGBA'))
    alpha       = img_rgba[:, :, 3]

    rows = np.where(alpha.max(axis=1) > 0)[0]
    cols = np.where(alpha.max(axis=0) > 0)[0]

    obj_top    = int(rows.min())
    obj_bottom = int(rows.max())
    obj_left   = int(cols.min())
    obj_right  = int(cols.max())
    obj_h      = obj_bottom - obj_top
    obj_w      = obj_right  - obj_left

    log.info(f"Object bounds from alpha: "
             f"top={obj_top} bottom={obj_bottom} "
             f"left={obj_left} right={obj_right} "
             f"obj_h={obj_h} obj_w={obj_w} "
             f"img_h={img_h} img_w={img_w}")

    img_np = np.array(img)
    predictor.set_image(img_np)

    wheel_hints = [h for h in joint_hints
                   if h.get('body_part') in ('wheel', 'gear', 'hinge')]

    mask_paths = {}
    centers    = {}

    centers['_image_bounds'] = {
        'obj_top':    obj_top,
        'obj_bottom': obj_bottom,
        'obj_left':   obj_left,
        'obj_right':  obj_right,
        'img_w':      img_w,
        'img_h':      img_h,
    }

    for hint in wheel_hints:
        name      = hint['name']
        p         = hint.get('position_normalized', {})
        r_norm    = hint.get('wheel_radius_normalized', 0.15)
        body_part = hint.get('body_part', 'wheel')
        is_gear   = (body_part == 'gear')

        # ── Step 1: rough bbox from normalized position ───────────────────────
        r_px = r_norm * img_h   # not max(img_w, img_h)
        pad  = 0.05 * img_h


        cx_rough = p.get('x', 0.5) * img_w
        cy_rough = (1.0 - p.get('y', 0.5)) * img_h

        x1 = max(0,     int(cx_rough - r_px - pad))
        y1 = max(0,     int(cy_rough - r_px - pad))
        x2 = min(img_w, int(cx_rough + r_px + pad))
        y2 = min(img_h, int(cy_rough + r_px + pad))

        log.info(f"SAM2 [{name}] rough bbox=({x1},{y1},{x2},{y2}) "
                 f"rough_center=({cx_rough:.0f},{cy_rough:.0f})")

        # ── Step 2: Hough circle inside bbox ─────────────────────────────────
        crop    = img_np[y1:y2, x1:x2]
        gray    = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        box_w, box_h = x2 - x1, y2 - y1
        min_r    = int(min(box_w, box_h) * 0.28)
        max_r    = int(min(box_w, box_h) * 0.62)
        min_dist = 1 if is_gear else min(box_w, box_h)

        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=min_dist,
            param1=80, param2=35,
            minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1.5, minDist=min(box_w, box_h),
                param1=60, param2=25,
                minRadius=int(min_r * 0.7), maxRadius=int(max_r * 1.2),
            )

        if circles is not None:
            bcx, bcy = box_w / 2, box_h / 2
            best     = min(circles[0],
                           key=lambda c: np.hypot(c[0] - bcx, c[1] - bcy))

            # Validate: reject if Hough result is too far from hint (gear only)
            cx_det = int(x1 + best[0])
            cy_det = int(y1 + best[1])
            if is_gear:
                dist_from_hint = np.hypot(cx_det - cx_rough, cy_det - cy_rough)
                if dist_from_hint > r_px * 2.0:
                    log.warning(f"SAM2 [{name}] Hough result ({cx_det},{cy_det}) "
                                f"is {dist_from_hint:.0f}px from hint — rejecting")
                    cx = int(cx_rough); cy = int(cy_rough)
                    radius = int(r_px * 0.9)
                else:
                    cx, cy, radius = cx_det, cy_det, int(best[2])
            else:
                cx, cy, radius = cx_det, cy_det, int(best[2])
            log.info(f"SAM2 [{name}] Hough circle: center=({cx},{cy}) radius={radius}")
        else:
            cx     = int(cx_rough)
            cy     = int(cy_rough)
            radius = int(r_px * 0.9)
            log.warning(f"SAM2 [{name}] Hough failed — using rough center ({cx},{cy}) r={radius}")

        # ── Step 3: anatomical prompt points ─────────────────────────────────
        def clamp_pt(x, y):
            return (int(np.clip(x, 0, img_w - 1)),
                    int(np.clip(y, 0, img_h - 1)))

        point_list = [clamp_pt(cx, cy)]
        label_list = [1]

        for deg in (0, 90, 180, 270):
            rad = np.radians(deg)
            point_list.append(clamp_pt(cx + radius * 0.90 * np.cos(rad),
                                       cy + radius * 0.90 * np.sin(rad)))
            label_list.append(1)

        for deg in (45, 135, 225, 315):
            rad = np.radians(deg)
            point_list.append(clamp_pt(cx + radius * 0.72 * np.cos(rad),
                                       cy + radius * 0.72 * np.sin(rad)))
            label_list.append(1)

        point_list.append(clamp_pt(cx + radius * 1.25, cy))
        label_list.append(0)

        point_coords = np.array(point_list)
        point_labels = np.array(label_list)

        margin = int(radius * 0.15)
        box = np.array([
            max(0,     cx - radius - margin),
            max(0,     cy - radius - margin),
            min(img_w, cx + radius + margin),
            min(img_h, cy + radius + margin),
        ])

        # ── Step 4: SAM2 predict ──────────────────────────────────────────────
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=True,
            )

        best_idx  = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)

        ys, xs  = np.where(best_mask)
        mask_cx = int(xs.mean()) if len(xs) > 0 else cx
        mask_cy = int(ys.mean()) if len(ys) > 0 else cy

        centers[name] = {
            'hough_cx': cx,
            'hough_cy': cy,
            'hough_r':  radius,
            'mask_cx':  mask_cx,
            'mask_cy':  mask_cy,
            'img_w':    img_w,
            'img_h':    img_h,
            'norm_cx':  mask_cx / img_w,
            'norm_cy':  mask_cy / img_h,
            'norm_r':   radius / max(img_w, img_h),
        }
        log.info(f"SAM2 [{name}] score={scores[best_idx]:.3f} "
                 f"pixels={best_mask.sum()} "
                 f"points={len(point_coords)} "
                 f"box={box.tolist()}")

        # ── Step 5: save mask ─────────────────────────────────────────────────
        mask_path = os.path.join(results_dir, f"mask_{name}.png")
        PILImage.fromarray((best_mask * 255).astype(np.uint8)).save(mask_path)
        mask_paths[name] = mask_path

        # ── Debug overlay ─────────────────────────────────────────────────────
        debug_img = PILImage.fromarray(img_np.copy())
        draw      = ImageDraw.Draw(debug_img)

        mask_rgba = np.zeros((*best_mask.shape, 4), dtype=np.uint8)
        mask_rgba[best_mask, 1] = 180
        mask_rgba[best_mask, 3] = 80
        overlay = PILImage.fromarray(mask_rgba, 'RGBA')
        debug_img.paste(overlay, mask=overlay)

        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     outline='cyan', width=2)
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

        for (px_c, py_c), lbl in zip(point_list, label_list):
            color = 'yellow' if lbl == 1 else 'red'
            r_dot = 7
            draw.ellipse([px_c - r_dot, py_c - r_dot,
                          px_c + r_dot, py_c + r_dot],
                         fill=color, outline='white', width=1)

        debug_path = os.path.join(results_dir, f"sam2_debug_{name}.png")
        debug_img.save(debug_path)
        log.info(f"SAM2 debug image → {debug_path}")

    # Save wheel_centers.json into mask_dir

    mask_dir = os.path.abspath(os.path.join(results_dir, 'masks'))
    centers_path = os.path.join(mask_dir, 'wheel_centers.json')
    os.makedirs(mask_dir, exist_ok=True)
    with open(centers_path, 'w') as f:
        json.dump(centers, f, indent=2)

    return mask_paths


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
    tag_words = set(re.split(r'[ +]', (user_tag or '').lower()))
    tag_ctx   = f'\nThe user identified this as: "{user_tag}".' if user_tag else ''

    prompt = (
        f"This is a vehicle {user_tag}.\n\n" + utils._build_vehicle_classify_prompt(tag_ctx)
        if tag_words & utils.VEHICLE_KEYWORDS or tag_words & utils.MECHANICAL_KEYWORDS
        else utils._build_classify_prompt(tag_ctx)
    )
    log.info(f"prompt first 500 chars: {prompt[:500]}")

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
    
    if rig_type == 'vehicle' or rig_type == "mechanical":
        prompt = utils._build_vehicle_prompt(object_type, category, n_joints,
                                             mesh_bounds=mesh_bounds, rig_type=rig_type)
    else:
        prompt = utils._build_joints_prompt(object_type, category, n_joints,
                                             mesh_bounds=mesh_bounds, rig_type=rig_type)

    log.info(f"prompt first 500 chars: {prompt[:500]}")
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
        log.info(result.stdout)
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed: {result.stderr[-200:]}")
        if not os.path.exists(rigged_path):
            raise RuntimeError(f"Rigged GLB not created: {result.stdout[-500:]}")


def joints_from_model(joints_data: dict, glb_path: str):
    """
    Map normalised joint positions (0–1) onto mesh world-space.

    Meshy exports Y-up (mesh index 0=X left/right, 1=Y up/down, 2=Z depth) —
    same convention used everywhere else (mesh_bounds, snap_joints_to_mesh,
    mesh_guided_joint_correction, visualize_normalized_joints). The vision
    prompt already returns y in this bottom-up convention (0=feet, 1=head),
    so no inversion is needed:
      Claude x (0=left, 1=right)   → mesh X (left/right)
      Claude y (0=feet, 1=head)    → mesh Y (up/down)
      Claude z (depth hint, ~0.5)  → mesh Z (front/back depth)

    The Y-up → Blender Z-up conversion happens later, in rig.py's
    --from-json branch (`(x, -z, y)`), not here.
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

    # Fixed axis mapping for Meshy Y-up + side-view camera
    # mesh axis indices: 0=X (left/right), 1=Y (up), 2=Z (front/rear)
    LR_IDX = 0   # mesh X  ← Claude z (depth, usually ~0.5)
    UP_IDX = 1   # mesh Y  ← Claude y (inverted: image top = world top)
    FR_IDX = 2   # mesh Z  ← Claude x (image horizontal = world front/rear)

    name_to_idx = {h['name']: i for i, h in enumerate(hint_objects)}
    joints = []

    for hint in hint_objects:
        p      = hint.get('position_normalized', {})
        norm_x = np.clip(p.get('x', 0.5), 0.0, 1.0)   # image left→right
        norm_y = np.clip(p.get('y', 0.5), 0.0, 1.0)   # image top→bottom
        norm_z = np.clip(p.get('z', 0.5), 0.0, 1.0)   # image depth

        world_pos = np.zeros(3)
        world_pos[0] = bmin[0] + np.clip(p.get('x', 0.5), 0.0, 1.0) * brange[0]  # Claude x → mesh X (left/right)
        world_pos[1] = bmin[1] + np.clip(p.get('y', 0.5), 0.0, 1.0) * brange[1]  # Claude y → mesh Y (up)
        world_pos[2] = bmin[2] + np.clip(p.get('z', 0.5), 0.0, 1.0) * brange[2]  # Claude z → mesh Z (depth)



        joints.append(tuple(world_pos))

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
    log.info(f"joints_from_model: {len(joints)} joints, {len(hierarchy)} bones "
         f"(Claude x→X, y→Y, z→Z)")

    return joints, hierarchy, hint_objects

def mirror_wheel_centers(mask_dir, joint_hints):
    centers_path = os.path.join(mask_dir, 'wheel_centers.json')
    if not os.path.exists(centers_path):
        return
    with open(centers_path) as f:
        centers = json.load(f)

    updated = False
    for hint in joint_hints:
        if not hint.get('mirrored'):
            continue
        name        = hint['name']
        source_name = hint.get('mirrored_from')
        if not source_name or source_name not in centers or name in centers:
            continue

        src  = centers[source_name]
        img_w = src['img_w']
        mirrored_cx = img_w - src['mask_cx']

        centers[name] = {
            'hough_cx': img_w - src['hough_cx'],
            'hough_cy': src['hough_cy'],
            'hough_r':  src['hough_r'],
            'mask_cx':  mirrored_cx,
            'mask_cy':  src['mask_cy'],
            'img_w':    img_w,
            'img_h':    src['img_h'],
            'norm_cx':  mirrored_cx / img_w,
            'norm_cy':  src['norm_cy'],
            'norm_r':   src['norm_r'],
            'mirrored': True,
        }
        updated = True
        log.info(f"Mirrored center: {source_name}→{name} "
                 f"cx {src['mask_cx']}→{mirrored_cx}")

    if updated:
        with open(centers_path, 'w') as f:
            json.dump(centers, f, indent=2)


def _detect_trunk_x_edges(verts_in_band: np.ndarray, mesh_center_x: float):
    """
    Measure trunk width at a height band without assuming any particular
    body shape or proportion. A naive min/max over the band is
    contaminated whenever a limb also passes through that height (e.g. an
    A-pose arm crossing waist height) — but a fixed "trunk can't be wider
    than X% of the mesh" cap is just as wrong in the other direction (a
    round/wide character body would be rejected as "implausible" even
    when the reading is completely correct).

    Instead, look for a real gap in the cross-section: sort all vertex X
    values in the band and find the largest gap between consecutive
    values. A limb crossing through is usually separated from the torso
    by open space (e.g. the armpit gap in an A-pose or spread stance), so
    a substantial gap means the band actually contains multiple
    disconnected clusters, not one continuous torso. When that happens,
    keep only the cluster nearest the mesh's own horizontal centerline —
    that's the trunk; the other cluster(s) are the offending limb(s).
    If there's no substantial gap, the band is one continuous blob and
    the full min/max is trusted as-is, however wide it is — this is what
    makes it work for a body shape with no slim waist at all.

    Returns (x_left, x_right) in world-space coordinates, or None if
    there aren't enough vertices in the band to measure.
    """
    if len(verts_in_band) < 10:
        return None

    xs = np.sort(verts_in_band[:, 0])
    total_span = xs[-1] - xs[0]
    if total_span <= 0:
        return None

    gaps = np.diff(xs)
    GAP_FRAC_THRESHOLD = 0.12
    split_idx = np.where(gaps / total_span >= GAP_FRAC_THRESHOLD)[0]

    if len(split_idx) == 0:
        # One continuous blob — no limb crossing detected, trust it fully.
        return float(xs[0]), float(xs[-1])

    # Split into clusters at every substantial gap (handles more than one
    # limb crossing at the same height, e.g. both arms in an A-pose).
    boundaries = [0] + (split_idx + 1).tolist() + [len(xs)]
    clusters = [xs[boundaries[i]:boundaries[i + 1]]
                for i in range(len(boundaries) - 1)]

    trunk_cluster = min(
        clusters, key=lambda c: abs((c[0] + c[-1]) / 2 - mesh_center_x)
    )
    return float(trunk_cluster[0]), float(trunk_cluster[-1])


def _measure_trunk_x_edges(verts: np.ndarray, bmin: np.ndarray,
                            bmax: np.ndarray, brange: np.ndarray,
                            center_frac: float = 0.50, span_frac: float = 0.21):
    """
    Robust trunk-width measurement: a single height band can still be
    contaminated even with gap detection (_detect_trunk_x_edges) — if a
    limb happens to sit close enough to the torso at that exact height,
    there's no real air gap to find (this genuinely happens on some
    meshes, e.g. a thick/low-poly A-pose character whose arm blends into
    the torso silhouette at certain heights, even though it's clearly
    separate at others).

    Sample several height bands spanning [center_frac - span_frac/2,
    center_frac + span_frac/2] and take the MEDIAN-width band's result. A
    limb only contaminates the bands its own height overlaps — the true
    trunk width tends to be far more consistent band-to-band than a
    contaminated reading — so the median is naturally resistant to the
    1-2 bands a limb happens to intrude on, without needing any fixed
    plausibility threshold.

    center_frac/span_frac default to the waist/lower-chest region (a
    reasonable default for a normal-proportioned humanoid), but callers
    measuring shoulder width should center this on the shoulder joint's
    own Y instead — for a body plan where the torso silhouette isn't
    roughly vertical between waist and shoulder (e.g. a wide, tapering
    crown/head mass sitting on a thin stalk), measuring at a fixed waist
    height and pairing the result with a shoulder Y from somewhere else
    entirely produces an (x, y) pair that doesn't correspond to any real
    cross-section — it can land outside the actual mesh surface.

    Returns (x_left, x_right) in world-space coordinates, or None if no
    band had enough vertices to measure.
    """
    mesh_center_x = (bmin[0] + bmax[0]) / 2
    band_results = []
    lo_bound = max(0.0, center_frac - span_frac / 2)
    hi_bound = min(1.0 - 0.05, center_frac + span_frac / 2)
    for lo_frac in np.arange(lo_bound, hi_bound + 1e-9, 0.05):
        y_lo = bmin[1] + lo_frac * brange[1]
        y_hi = bmin[1] + (lo_frac + 0.05) * brange[1]
        band_verts = verts[(verts[:, 1] >= y_lo) & (verts[:, 1] < y_hi)]
        edges = _detect_trunk_x_edges(band_verts, mesh_center_x)
        if edges is not None:
            band_results.append((edges[1] - edges[0], edges))

    if not band_results:
        return None

    band_results.sort(key=lambda t: t[0])
    return band_results[len(band_results) // 2][1]


def mesh_guided_joint_correction(joints_data: dict, mesh,
                                  rig_type: str) -> dict:
    """
    Use actual mesh geometry to correct and fill in Claude's joint placement.
    Handles cases where Claude misses joints or places them incorrectly
    due to unusual aspect ratios or novel object types.
    """
    hints = joints_data.get('joint_hints', [])
    if not hints:
        return joints_data

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
        # Shoulder Y is no longer derived from a "narrowest slice" search
        # here — the vision prompt now specifies shoulder height directly
        # (same as neck, see utils.py's shoulder_y), and snap_joints_to_mesh
        # already refines it against real geometry. The narrowest-slice
        # heuristic that used to run here was unreliable (it often found a
        # waist/torso taper rather than the actual armpit junction) and
        # could drag an already-correct shoulder Y back down.

        # Shoulder X = trunk edge, measured directly from the mesh — NOT
        # trusted from the vision guess even after snap_joints_to_mesh,
        # because a bad vision guess (anchored to a fixed fraction of the
        # full image) can land deep in arm territory in a T-pose, and
        # nearest-vertex search from a bad starting point just refines
        # near the same wrong spot instead of correcting it.
        # See _measure_trunk_x_edges for how arm-crossing contamination
        # (e.g. an A-pose arm passing through waist height) is filtered
        # out via gap detection across multiple height bands, rather than
        # a fixed width assumption.
        #
        # Measured separately per side, centered on THAT side's own
        # shoulder Y (not a fixed waist fraction) — pairing an X measured
        # at one height with a Y from somewhere else only lines up for a
        # body whose torso silhouette is roughly vertical between waist
        # and shoulder. For a body plan where it isn't (e.g. a wide,
        # tapering crown/head mass on a thin stalk, where the torso is
        # much narrower at "waist" height than up at real shoulder
        # height), that mismatch produces an (x, y) pair that lands
        # outside the actual mesh surface — confirmed via a vision-
        # verification render showing the shoulder marker floating in
        # empty space next to the body rather than on it.
        for side, name in [('left', 'joint_shoulder_left'),
                            ('right', 'joint_shoulder_right')]:
            if name not in hint_map:
                continue
            shoulder_y_frac = hint_map[name]['position_normalized'].get('y', 0.5)
            trunk_edges = _measure_trunk_x_edges(
                verts, bmin, bmax, brange, center_frac=shoulder_y_frac)
            if trunk_edges is None:
                continue
            trunk_x = trunk_edges[0] if side == 'left' else trunk_edges[1]
            new_x = float(np.clip((trunk_x - bmin[0]) / brange[0], 0.0, 1.0))
            old_x = hint_map[name]['position_normalized']['x']
            hint_map[name]['position_normalized']['x'] = new_x
            log.info(f"  GeoCorrect {name} X: {old_x:.3f}→{new_x:.3f} "
                     f"(trunk edge @ shoulder y={shoulder_y_frac:.3f})")

        # Hands = outermost X vertices in arm Y range. Band must be wide
        # enough to cover a hand at any arm pose — T-pose puts it near
        # shoulder height, which (like neck_y) is now measured per mesh and
        # can run above the old fixed 0.85 ceiling for a big-headed/
        # short-body mesh; A-pose or arms-at-sides puts it much lower
        # (~0.3-0.45) — a narrower band can miss the true extremity and
        # corrupt an already-correct X guess (same failure mode as the old
        # shoulder Y-band). Adopt the found vertex's Y as well as X — a
        # hand that flares outward at the wrist means the widest-X vertex
        # is a real hand vertex, so its height is real signal too, and it
        # can catch cases where the vision guess assumed the wrong arm pose
        # (e.g. guessed T-pose when the actual mesh is an A-pose).
        arm_y_lo  = bmin[1] + 0.15 * brange[1]
        arm_y_hi  = bmin[1] + 0.95 * brange[1]
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
                    old_y = hint_map[name]['position_normalized']['y']
                    if abs(old_x - norm['x']) > 0.05 or abs(old_y - norm['y']) > 0.05:
                        hint_map[name]['position_normalized'] = {
                            'x': norm['x'], 'y': norm['y'], 'z': 0.5,
                        }
                        log.info(f"  GeoCorrect {name} X: {old_x:.3f}→{norm['x']:.3f} "
                                 f"Y: {old_y:.3f}→{norm['y']:.3f} (mesh extremity)")

        # Elbow = the point along the arm's OWN geometry that bends
        # farthest away from a straight shoulder→hand line, computed
        # AFTER the shoulder/hand corrections above so it reflects their
        # geometrically-grounded positions rather than a possibly wrong
        # vision guess.
        #
        # This used to be a plain Y-midpoint between shoulder and hand
        # (mirroring knee_y = midpoint(hip_y, foot_y)) — a reasonable
        # assumption for a knee, since legs are load-bearing and stay
        # close to straight, but wrong for an arm that visibly droops or
        # bends (e.g. broccoli's floppy dangling arms, clearly curved in
        # the mesh, not a straight diagonal from shoulder to hand). Taking
        # the midpoint in that case lands the elbow in empty space between
        # the two ends rather than on the actual bend in the arm. Tracing
        # the arm's own cross-section (same clustering technique used for
        # shoulder's arm/trunk merge detection) and picking the vertex
        # that deviates most from the straight line finds the real bend
        # regardless of whether the arm is straight or curved.
        mesh_center_x = (bmin[0] + bmax[0]) / 2
        for side, sign, elbow_name, shoulder_name, hand_name in [
            ('left',  -1.0, 'joint_elbow_left',  'joint_shoulder_left',  'joint_hand_left'),
            ('right',  1.0, 'joint_elbow_right', 'joint_shoulder_right', 'joint_hand_right'),
        ]:
            if not (elbow_name in hint_map and shoulder_name in hint_map and hand_name in hint_map):
                continue
            sp = hint_map[shoulder_name]['position_normalized']
            hp = hint_map[hand_name]['position_normalized']
            shoulder_world = np.array([bmin[0] + sp['x'] * brange[0],
                                        bmin[1] + sp['y'] * brange[1]])
            hand_world     = np.array([bmin[0] + hp['x'] * brange[0],
                                        bmin[1] + hp['y'] * brange[1]])

            y_lo, y_hi = sorted([shoulder_world[1], hand_world[1]])
            n_bands = 24
            band_edges = np.linspace(y_lo, y_hi, n_bands + 1)
            samples = []
            for i in range(n_bands):
                band_verts = verts[(verts[:, 1] >= band_edges[i]) &
                                    (verts[:, 1] < band_edges[i + 1])]
                if len(band_verts) < 5:
                    continue
                # Outer edge of the arm's own side, away from the trunk —
                # avoids picking a torso/armpit vertex as if it were arm.
                outer = (band_verts[np.argmin(band_verts[:, 0])] if sign < 0
                         else band_verts[np.argmax(band_verts[:, 0])])
                if sign * (outer[0] - mesh_center_x) > 0:
                    samples.append(outer[:2])

            if not samples:
                continue
            samples = np.array(samples)
            line_vec  = hand_world - shoulder_world
            line_len2 = float(np.dot(line_vec, line_vec))
            if line_len2 < 1e-9:
                continue
            t = np.clip(np.dot(samples - shoulder_world, line_vec) / line_len2, 0.0, 1.0)
            proj  = shoulder_world[None, :] + t[:, None] * line_vec[None, :]
            perp_dist = np.linalg.norm(samples - proj, axis=1)
            bend_vert = samples[np.argmax(perp_dist)]

            old_norm = dict(hint_map[elbow_name]['position_normalized'])
            new_norm = {
                'x': float(np.clip((bend_vert[0] - bmin[0]) / brange[0], 0.0, 1.0)),
                'y': float(np.clip((bend_vert[1] - bmin[1]) / brange[1], 0.0, 1.0)),
                'z': 0.5,
            }
            hint_map[elbow_name]['position_normalized'] = new_norm
            log.info(f"  GeoCorrect {elbow_name}: {old_norm} → {new_norm} "
                     f"(arm bend point, {perp_dist.max():.3f} off the "
                     f"shoulder-hand line)")

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
                },
                'mirrored':      True,
                'mirrored_from': left_name,   # ← add this
            }
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
def run_mechanical_pipeline(classify_id: str, glb_path: str,
                             classify_data: dict, host: str) -> str:
    seg_dir       = os.path.dirname(os.path.abspath(__file__))
    vehicle_dir   = os.path.join(seg_dir, 'vehicle')
    _rd           = _rdir(classify_id)
    classify_json = os.path.abspath(os.path.join(_rd, f"{classify_id}_classify.json"))
    mask_dir      = os.path.abspath(os.path.join(_rd, f"{classify_id}_masks"))

    record      = hydrate(_store.get(classify_id))
    joints_data = (record.get('joints') or {}) if record else {}
    joint_hints = joints_data.get('joint_hints', [])
    wheel_colors = joints_data.get('wheel_colors_rgb')
    if wheel_colors:
        classify_data['wheel_colors_rgb'] = wheel_colors
        log.info(f"wheel_colors_rgb from joints: {wheel_colors}")
    classify_data['joint_hints'] = [
        j for j in joint_hints
        if j.get('body_part') in ('gear', 'hinge')
    ]

    gear_hints = [j for j in classify_data['joint_hints'] if j.get('body_part') == 'gear']
    if gear_hints:
        classify_data['reference_radius_normalized'] = max(
            j.get('wheel_radius_normalized', 0.15) for j in gear_hints
        )
        log.info(f"Reference radius: {classify_data['reference_radius_normalized']:.3f}")

    if len(gear_hints) >= 2:
        x_spread = (max(j['position_normalized']['x'] for j in gear_hints) -
                    min(j['position_normalized']['x'] for j in gear_hints))
        z_spread = (max(j['position_normalized']['z'] for j in gear_hints) -
                    min(j['position_normalized']['z'] for j in gear_hints))
        classify_data['image_view'] = 'side' if x_spread > z_spread else 'front'
        log.info(f"Image view: {classify_data['image_view']}")

    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)
    log.info(f"Injected {len(classify_data['joint_hints'])} mechanical joints")

    # ── SAM2 segmentation ─────────────────────────────────────────────────────
    active_path = record.get('active_image_path') if record else None
    if active_path and os.path.exists(active_path):
        try:
            os.makedirs(mask_dir, exist_ok=True)
            raw_mask_paths = segment_parts_with_sam2(
                active_path, classify_data['joint_hints'], classify_id, mask_dir
            )
            import shutil
            for hint_name, src_path in raw_mask_paths.items():
                dst_path = os.path.join(mask_dir, f"{hint_name}.png")
                if os.path.exists(dst_path):
                    os.unlink(dst_path)
                shutil.copy2(src_path, dst_path)
                log.info(f"Mask → {dst_path}")
            with open(classify_json, 'r') as f:
                _cdata = json.load(f)
            _cdata['sam2_masks']    = raw_mask_paths
            _cdata['sam2_mask_dir'] = mask_dir
            with open(classify_json, 'w') as f:
                json.dump(_cdata, f, indent=2)
            log.info(f"SAM2 masks ready: {list(raw_mask_paths.keys())}")
        except Exception as e:
            log.warning(f"SAM2 segmentation failed (non-fatal): {e}")
            mask_dir = None
    else:
        log.warning("No active image — SAM2 skipped")
        mask_dir = None

    # ── Invoke pipeline script ────────────────────────────────────────────────
    mech_script = os.path.join(seg_dir, 'vehicle', 'mechanical_pipeline.py')
    cmd = [sys.executable, mech_script,
           classify_id, glb_path, classify_json,
           _rd, vehicle_dir]
    if mask_dir and os.path.isdir(mask_dir):
        cmd.append(mask_dir)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    log.info(result.stdout)
    log.info(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Mechanical pipeline failed: {result.stderr[-300:]}")

    rigged_path = os.path.join(_rd, f"{classify_id}_rigged.glb")
    log.info(f"Mechanical pipeline complete: {rigged_path}")
    return rigged_path


def run_vehicle_pipeline(classify_id: str, glb_path: str,
                         classify_data: dict, host: str) -> str:
    seg_dir        = os.path.dirname(os.path.abspath(__file__))
    vehicle_dir    = os.path.join(seg_dir, 'vehicle')
    _rd            = _rdir(classify_id)
    separated_path = os.path.join(_rd, f"{classify_id}_separated.glb")
    animated_path  = os.path.join(_rd, f"{classify_id}_animated.glb")
    rigged_path    = os.path.join(_rd, f"{classify_id}_rigged.glb")
    classify_json  = os.path.join(_rd, f"{classify_id}_classify.json")
    tire_verts     = os.path.join(_rd, f"{classify_id}_tire_verts.json")
    texture_path   = os.path.join(_rd, f"{classify_id}_texture.png")


    record      = hydrate(_store.get(classify_id))
    joints_data = (record.get('joints') or {}) if record else {}
    joint_hints = joints_data.get('joint_hints', [])
    object_type = classify_data.get('object_type', '')
    
    wheel_colors = joints_data.get('wheel_colors_rgb')
    if wheel_colors:
        classify_data['wheel_colors_rgb'] = wheel_colors
        log.info(f"wheel_colors_rgb from joints: {wheel_colors}")

    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)
        
    # ADD THIS DEBUG:
    log.info(f"DEBUG: joints_data from store: {len(joint_hints)} hints")
    for jh in [j for j in joint_hints if j.get('body_part') == 'wheel']:
        p = jh.get('position_normalized', {})
        log.info(f"  {jh['name']}: x={p.get('x')}, y={p.get('y')}, z={p.get('z')}")

    TWO_WHEEL_KEYWORDS = ['bicycle', 'bike', 'motorcycle', 'motorbike', 'scooter', 'moped']
    is_two_wheel = any(kw in object_type.lower() for kw in TWO_WHEEL_KEYWORDS)

    if is_two_wheel:
        wheel_hints = [j for j in joint_hints if j.get('body_part') == 'wheel']
        if len(wheel_hints) == 4:
            # Sort by z to find front/rear pairs (z=front/rear in Claude convention)
            wheel_hints_sorted = sorted(wheel_hints, key=lambda w: w['position_normalized']['z'])
            front_pair = wheel_hints_sorted[:2]   # lower z = front
            rear_pair  = wheel_hints_sorted[2:]   # higher z = rear

            collapsed = []
            if front_pair:
                avg_z = sum(h['position_normalized']['z'] for h in front_pair) / len(front_pair)
                avg_y = sum(h['position_normalized']['y'] for h in front_pair) / len(front_pair)
                r_norm = front_pair[0].get('wheel_radius_normalized', 0.12)
                collapsed.append({**front_pair[0], 'name': 'wheel_fl',
                   'position_normalized': {'x': 0.5, 'y': avg_y, 'z': avg_z}})
            if rear_pair:
                avg_z = sum(h['position_normalized']['z'] for h in rear_pair) / len(rear_pair)
                avg_y = sum(h['position_normalized']['y'] for h in rear_pair) / len(rear_pair)
                collapsed.append({**rear_pair[0], 'name': 'wheel_rl',
                   'position_normalized': {'x': 0.5, 'y': avg_y, 'z': avg_z}})
            joint_hints = [j for j in joint_hints if j.get('body_part') != 'wheel'] + collapsed
            log.info(f"Collapsed 4 wheels to 2 for {object_type}")
            
    # Only inject wheel joints
    classify_data['joint_hints'] = [
        j for j in joint_hints
        if j.get('body_part') in ['wheel', 'gear']
    ]

    log.info(f"DEBUG: classify_data['joint_hints'] about to write: {len(classify_data['joint_hints'])} hints")
    for jh in classify_data['joint_hints']:
        p = jh.get('position_normalized', {})
        log.info(f"  {jh['name']}: x={p.get('x')}, y={p.get('y')}, z={p.get('z')}")


    wheel_joints_out = [j for j in classify_data['joint_hints'] if j.get('body_part') == 'wheel']
    if len(wheel_joints_out) == 2:
        z_spread = abs(wheel_joints_out[0]['position_normalized']['z'] -
                       wheel_joints_out[1]['position_normalized']['z'])
        y_spread = abs(wheel_joints_out[0]['position_normalized']['y'] -
                       wheel_joints_out[1]['position_normalized']['y'])
        classify_data['wheel_split_axis'] = 2 if z_spread > y_spread else 1
        log.info(f"Wheel split axis: {classify_data['wheel_split_axis']}")

    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)
    log.info(f"Injected {len(classify_data['joint_hints'])} wheel joints into classify_json")
    
    
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
        ('find_tire_verts.py', glb_path,        tire_verts,     'python'),
        ('classify_wheels.py', glb_path,         separated_path, 'blender'),
        ('animatesam.py',      separated_path,   animated_path,  'blender'),
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
        log.info(result.stdout[-5000:])




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
                    log.info(f"Wheel centroids injected from find_tire_verts: {_centroids}")
                else:
                    log.warning("Centroids file not found — pivot will use bbox fallback")
            except Exception as e:
                log.warning(f"Centroid injection failed (non-fatal): {e}")

        log.info(f"separated_path exists: {os.path.exists(separated_path)} → {separated_path}")
        log.info(f"classify_wheels returncode: {result.returncode}")
        log.info(f"classify_wheels stderr: {result.stderr[-300:] if result.stderr else ''}")

        if not os.path.exists(out_path):
            raise RuntimeError(f"{script_name} failed: {result.stderr[-200:]}")

    log.info(f"Vehicle pipeline complete: {rigged_path}")
    return rigged_path
    
    
# ══════════════════════════════════════════════════════════════════════════════
# Rig pipeline (Blender only — mesh comes from /mesh)
# ══════════════════════════════════════════════════════════════════════════════

def skeleton_from_joints(joints_data: dict, glb_path: str,
                         rigid_parts: list) -> dict | None:
    """
    Build the skeleton dict rig.py consumes from stored joint hints, mapped
    onto the mesh at glb_path. Returns None if no joint had a usable
    position_normalized.
    """
    joints, hierarchy, hint_objects = joints_from_model(joints_data, glb_path)
    if not joints:
        return None

    def _hint_name(h, i):
        return h.get('name', f'joint_{i}') if isinstance(h, dict) else f'joint_{i}'

    return {
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


def run_rig_pipeline(task_id: str, classify_id: str, user_id: str, host: str):
    """
    Runs Blender rigging using mesh and joints already stored for classify_id.
    Mesh must exist. Joints are read from store; falls back to geometric if absent.
    Repeatable — re-run after /joints to get a new rig on the same mesh.
    """
    try:
        _rig_tasks[task_id] = {'status': 'rigging', 'progress': 10}
        _store.set_rig_status(classify_id, 'rigging')

        record        = hydrate(_store.get(classify_id))
        classify_data = record.get('classify') or {}
        mesh_data     = record.get('mesh')     or {}
        joints_data   = record.get('joints')   or {}
        rig_type      = classify_data.get('rig_type', '').lower()

        # ── Validate mesh ─────────────────────────────────────────────────────
        glb_path = ps._resolve_path(classify_id, 'mesh.glb')
        if not os.path.exists(glb_path):
            raise RuntimeError(
                f"Mesh GLB not found at '{glb_path}' — run /mesh before /rig"
            )

        glb_url     = mesh_data.get('glb_url')
        category    = classify_data.get('category', '')
        object_type = classify_data.get('object_type', '')
        rigid_parts = classify_data.get('rigid_parts', [])
        tag_words   = set(object_type.lower().split())
        seg_dir       = os.path.dirname(os.path.abspath(__file__))
        vehicle_dir   = os.path.join(seg_dir, 'vehicle')
        _rd           = _rdir(classify_id)
        classify_json = os.path.join(_rd, f"{classify_id}_classify.json")
        mask_dir      = os.path.join(_rd, f"{classify_id}_masks")
        active_path   = record.get('active_image_path', '')

        is_vehicle = (
            category == 'vehicle' or
            (category not in ('animal', 'humanoid', 'other', 'mechanical') and
             bool(tag_words & utils.VEHICLE_KEYWORDS))
        )
        is_mechanical = (
            category == 'mechanical' or
            rig_type == 'mechanical' or
            bool(tag_words & utils.MECHANICAL_KEYWORDS)
        )

        log.info(f"category='{category}' is_vehicle={is_vehicle} object_type='{object_type}' "
                 f"rigid_parts={rigid_parts}")

        decimated_path     = None
        skeleton_json_path = None
        viz_glb_path       = None

        if is_vehicle:
            _rig_tasks[task_id] = {'status': 'rigging', 'progress': 50}
            rigged_path = run_vehicle_pipeline(classify_id, glb_path, classify_data, host)
        elif is_mechanical:
            _rig_tasks[task_id] = {'status': 'rigging', 'progress': 50}
            rigged_path = run_mechanical_pipeline(classify_id, glb_path, classify_data, host)
        else:
            # ── Decimate ──────────────────────────────────────────────────────
            decimated_path = os.path.join(_rd, f"{classify_id}_decimated.glb")
            existing_decimated = ps._resolve_path(classify_id, 'decimated.glb')
            if not os.path.exists(existing_decimated):
                _rig_tasks[task_id] = {'status': 'decimating', 'progress': 20}
                _decimate_mesh(glb_path, decimated_path, ratio=0.1)
            else:
                decimated_path = existing_decimated
            active_glb  = decimated_path
            rigged_path = os.path.join(_rd, f"{classify_id}_rigged.glb")

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
                skel = skeleton_from_joints(joints_data, active_glb, rigid_parts)
                if skel is None:
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
                viz_glb_path = os.path.join(_rd, f"{classify_id}_viz.glb")
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

        # ── Update mesh record (paths are not stored — reconstructed at read time)
        if decimated_path:
            _store.upsert_mesh(classify_id, {**mesh_data})

        # ── Persist ───────────────────────────────────────────────────────────
        _rig_tasks[task_id] = {'status': 'finalizing', 'progress': 90}
        _store.upsert_rig(classify_id, {
            'status':  'ok',
            'user_id': user_id,
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
        _rd       = _rdir(classify_id)
        glb_path  = os.path.join(_rd, f"{classify_id}_mesh.glb")
        usdz_path = None
        download_file(glb_url, glb_path)
        if usdz_url:
            usdz_path = os.path.join(_rd, f"{classify_id}_mesh.usdz")
            download_file(usdz_url, usdz_path)

        _mesh_tasks[task_id] = {'status': 'decimating', 'progress': 85}
        decimated_path = os.path.join(_rd, f"{classify_id}_decimated.glb")
        try:
            _decimate_mesh(glb_path, decimated_path, ratio=0.1)
            log.info(f"Decimated mesh: {decimated_path}")
        except Exception as e:
            log.warning(f"Decimation failed (non-fatal): {e}")
            decimated_path = None

        _store.upsert_mesh(classify_id, {
            'mesh_hash':          mesh_hash,
            'meshy_task_id':      meshy_task_id,
            'glb_path':           glb_path,
            'glb_url':            glb_url,
            'usdz_path':          usdz_path,
            'usdz_url':           usdz_url,
            'decimated_glb_path': decimated_path,
        })

        _mesh_tasks[task_id] = {
            'status':              'ok',
            'progress':            100,
            'glb_url':             glb_url,
            'glb_local_url':       _local_url(glb_path, host),
            'decimated_local_url': _local_url(decimated_path, host) if decimated_path else None,
            'classify_id':         classify_id,
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

# ── SAM ───────────────────────────────────────────────────────────────────────
_sam2_predictor = None

def _get_sam2():
    global _sam2_predictor
    if _sam2_predictor is None:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import sam2 as sam2_pkg
        sam2_dir    = os.path.dirname(sam2_pkg.__file__)
        sam2_config = os.environ.get('SAM2_CONFIG',
            os.path.join(sam2_dir, 'configs/sam2.1/sam2.1_hiera_s.yaml'))
        sam2_checkpoint = os.environ.get('SAM2_CHECKPOINT',
            'checkpoints/sam2.1_hiera_small.pt')
        model           = build_sam2(sam2_config, sam2_checkpoint,
                                     device='cpu')
        _sam2_predictor = SAM2ImagePredictor(model)
        log.info("SAM2 ready (CPU).")
    return _sam2_predictor
   
# ── /segment_parts ────────────────────────────────────────────────────────────

@app.route('/segment_parts', methods=['GET', 'POST'])
def segment_parts():
    """
    Run SAM2 part segmentation on the active image using joint positions as prompts.
    Requires /classify and /infer_joints to have run first.
    Saves per-wheel masks to RESULTS_DIR and stores mask_paths in the record.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        classify_id = request.args.get('classify_id', '').strip()
        force       = request.args.get('force', '').lower() in ('1', 'true', 'yes')

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400

        record = hydrate(_store.get(classify_id))
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found"}), 404

        # Cache check
        if not force and record.get('sam2_masks'):
            log.info(f"SAM2 mask cache hit: {classify_id}")
            return jsonify({
                'status':      'ok',
                'classify_id': classify_id,
                'mask_paths':  record['sam2_masks'],
            })

        joints_data = record.get('joints') or {}
        joint_hints = joints_data.get('joint_hints', [])
        if not joint_hints:
            return jsonify({'error': 'No joints found — run /infer_joints first'}), 422

        active_path = record.get('active_image_path')
        if not active_path or not os.path.exists(active_path):
            return jsonify({'error': f"Active image not found: '{active_path}'"}), 404

        log.info(f"Running SAM2 segmentation for {classify_id} "
                 f"({len([h for h in joint_hints if h.get('body_part') in ('wheel','gear')])} wheel/gear hints)")

        mask_paths = segment_parts_with_sam2(active_path, joint_hints, classify_id, _rdir(classify_id))

        # Store mask paths in record
        _store.upsert_classify(classify_id, record.get('tag', ''),
                               {**record.get('classify', {}),
                                'sam2_masks': mask_paths})

        return jsonify({
            'status':      'ok',
            'classify_id': classify_id,
            'mask_paths':  mask_paths,
            'mask_urls':   {name: _local_url(path, request.host)
                            for name, path in mask_paths.items()},
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/segment_parts error: {e}")
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
            record = hydrate(_store.get(classify_id))
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

        seg_path = os.path.join(_rdir(classify_id), f"{classify_id}_segmented.png")
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
 
        record = hydrate(_store.get(classify_id))
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
 
        path_a = os.path.join(_rdir(classify_id), f"{classify_id}_augmented_a.png")
        path_b = os.path.join(_rdir(classify_id), f"{classify_id}_augmented_b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        # The user explicitly requested augmentation by calling this endpoint —
        # record that, regardless of what /classify originally decided. This
        # keeps needs_augmentation an accurate reflection of "augmentation was
        # requested for this record" (e.g. a client working around a stale or
        # incorrect classify result), not just classify's own initial guess.
        if not classify_data.get('needs_augmentation'):
            _store.upsert_classify(classify_id, record.get('tag', ''),
                                   {**classify_data, 'needs_augmentation': True})

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

        chosen_path = os.path.join(_rdir(classify_id), f"{classify_id}_augmented_{choice}.png")
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

def compute_mesh_bounds(mesh) -> dict:
    """
    Measure the mesh's bounding box plus the geometric landmarks the
    joint pipeline relies on (trunk width, neck/pelvis narrowing,
    per-side arm/trunk merge height). Pure geometry, no vision model,
    so it's deterministic for a given mesh (see regression/).
    """
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

    # Trunk width at waist/lower-chest height — reliably below
    # where arms reach in a T-pose, but an A-pose's arm angles
    # down through this region too. See _measure_trunk_x_edges
    # for how arm-crossing contamination is filtered out via
    # gap detection across multiple height bands, rather than a
    # fixed width assumption (which would misfire on a body
    # shape with no slim waist at all, e.g. a round/tomato-like
    # character).
    trunk_edges = _measure_trunk_x_edges(verts, bmin, bmax, brange)
    if trunk_edges is not None:
        trunk_x_left  = float((trunk_edges[0] - bmin[0]) / brange[0])
        trunk_x_right = float((trunk_edges[1] - bmin[0]) / brange[0])
        mesh_bounds['trunk_x_left']  = trunk_x_left
        mesh_bounds['trunk_x_right'] = trunk_x_right
        log.info(f"Trunk width at waist height: x=[{trunk_x_left:.3f}, "
                 f"{trunk_x_right:.3f}]")

    # Neck height = local minimum of the cross-sectional width
    # profile between shoulders (wide) and head (wide again) —
    # a real geometric landmark, unlike a fixed fraction of
    # total height. Head-to-body ratio varies a lot per mesh
    # (e.g. a chibi/big-head character vs. adult proportions),
    # so a single fixed neck_y is wrong for one or the other.
    # Not attempted for the pelvis/waist the same way — an
    # A-pose arm passes through that height range and
    # contaminates the profile (same failure mode as the old
    # trunk-width measurement above).
    n_slices = 60
    width_profile = []
    for i in range(n_slices):
        y_lo = bmin[1] + (i / n_slices) * brange[1]
        y_hi = bmin[1] + ((i + 1) / n_slices) * brange[1]
        sv = verts[(verts[:, 1] >= y_lo) & (verts[:, 1] < y_hi)]
        if len(sv) > 5:
            w = float((sv[:, 0].max() - sv[:, 0].min()) / brange[0])
            width_profile.append((i / n_slices + 0.5 / n_slices, w))

    neck_candidates = [(y, w) for y, w in width_profile if 0.55 <= y <= 0.92]
    if neck_candidates:
        neck_y_detected, neck_w = min(neck_candidates, key=lambda t: t[1])
        boundary_w = max(
            neck_candidates[0][1], neck_candidates[-1][1]
        )
        # Require a genuine narrowing, not just the edge of the
        # search window (which would mean no real minimum exists).
        if boundary_w > 0 and neck_w < 0.7 * boundary_w:
            mesh_bounds['neck_y_detected'] = neck_y_detected
            log.info(f"Detected neck_y from mesh profile: "
                     f"{neck_y_detected:.3f} (width={neck_w:.3f})")

    # Pelvis/hip height = same local-minimum technique, applied to
    # the LOWER body — the waist narrowing between the legs and
    # the main body mass. A fixed pelvis_y (e.g. 0.42) assumes
    # legs take up a "normal" fraction of total height; for a
    # squatty character with very short legs relative to a large
    # round body, the real waist sits much lower, and hip was
    # otherwise left entirely to Claude's own guess (see
    # snap_joints_to_mesh, which only ever snapped hip X).
    #
    # Computed for ALL rig types now (not just biped) — a real
    # A-pose humanoid (aaf17890) confirmed an arm can pass
    # through this height range and mildly contaminate the
    # profile, but the resulting candidate still differs from
    # that character's fixed default by only ~0.1, well under
    # the ~0.27 seen on a genuinely broken case (broccoli) —
    # see verify_and_snap_joints's magnitude-gated override,
    # which uses that gap to apply this value only when it's
    # clearly needed, not on borderline/already-fine cases.
    # This mesh_bounds value itself is still only injected into
    # the PROMPT as an authoritative override for biped (see
    # _build_joints_prompt) — this broader computation exists so
    # the code-side safety net has a value to check against
    # regardless of rig_type.
    pelvis_candidates = [(y, w) for y, w in width_profile if 0.05 <= y <= 0.35]
    if pelvis_candidates:
        pelvis_y_detected, pelvis_w = min(pelvis_candidates, key=lambda t: t[1])
        boundary_w = max(
            pelvis_candidates[0][1], pelvis_candidates[-1][1]
        )
        if boundary_w > 0 and pelvis_w < 0.85 * boundary_w:
            mesh_bounds['pelvis_y_detected'] = pelvis_y_detected
            log.info(f"Detected pelvis_y from mesh profile: "
                     f"{pelvis_y_detected:.3f} (width={pelvis_w:.3f})")

    # Shoulder height = where each arm's own vertex cluster
    # actually merges into the main body mass, scanned bottom-up.
    # neck_y (used as shoulder_y's default/fallback above) is
    # only a good proxy for shoulder height on a normal
    # humanoid, where "narrow neck between wide shoulders and
    # wide head" and "where the arms attach" happen to sit at
    # about the same height. That assumption breaks for a body
    # plan where a large head/crown mass sits on a short torso
    # (e.g. broccoli) — the arms can attach much lower, near
    # the waist/collar, well below where the neck-narrowing
    # search (confined to y=0.55-0.92) ever looks. Tracing each
    # arm's own cluster directly answers "where does this
    # specific limb visually merge with the body" — the exact
    # question the vision prompt already asks, done
    # geometrically instead of trusting vision to trace it.
    mesh_center_x = (bmin[0] + bmax[0]) / 2
    n_bands = 48
    GAP_FRAC_THRESHOLD = 0.12
    for side, sign in [('left', -1.0), ('right', 1.0)]:
        merge_y = None
        for i in range(n_bands):
            y_lo = bmin[1] + (i / n_bands) * brange[1]
            y_hi = bmin[1] + ((i + 1) / n_bands) * brange[1]
            band_verts = verts[(verts[:, 1] >= y_lo) & (verts[:, 1] < y_hi)]
            if len(band_verts) < 10:
                continue
            xs = np.sort(band_verts[:, 0])
            total_span = xs[-1] - xs[0]
            if total_span <= 0:
                continue
            gaps = np.diff(xs)
            split_idx = np.where(gaps / total_span >= GAP_FRAC_THRESHOLD)[0]
            if len(split_idx) == 0:
                continue
            boundaries = [0] + (split_idx + 1).tolist() + [len(xs)]
            clusters = [xs[boundaries[j]:boundaries[j + 1]]
                        for j in range(len(boundaries) - 1)]
            has_arm_side = any(
                sign * ((c[0] + c[-1]) / 2 - mesh_center_x) > 0
                for c in clusters)
            has_trunk_side = any(
                sign * ((c[0] + c[-1]) / 2 - mesh_center_x) <= 0
                for c in clusters)
            if has_arm_side and has_trunk_side:
                merge_y = (i + 1) / n_bands  # top edge of last band still split
        if merge_y is not None:
            mesh_bounds[f'shoulder_y_detected_{side}'] = merge_y
            log.info(f"Detected shoulder_y ({side}) from arm/trunk "
                     f"merge: {merge_y:.3f}")

    return mesh_bounds


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

        record = hydrate(_store.get(classify_id))
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
         
        # Crop to alpha bounding box before sending to vision model.
        # The segmented PNG has black padding around the object — Claude's
        # normalized coordinates must be relative to the object bounds, not
        # the full padded image, so they map correctly onto mesh space.
        try:
            _img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
            _alpha = _img.split()[3]
            _bbox = _alpha.getbbox()   # (left, top, right, bottom)
            if _bbox:
                _img_cropped = _img.crop(_bbox)
                _buf = io.BytesIO()
                _img_cropped.save(_buf, format='PNG')
                img_bytes = _buf.getvalue()
                log.info(f"Cropped image to alpha bbox {_bbox} "
                         f"({_img.size} → {_img_cropped.size})")
            else:
                log.warning("Alpha bbox empty — using full image")
        except Exception as e:
            log.warning(f"Alpha crop failed (non-fatal): {e}")
 
        mime_type   = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'
        object_type = classify_data.get('object_type', '')
        category    = classify_data.get('category', '')
        rig_type    = classify_data.get('rig_type', '')

        # ── Extract mesh bounding box if available ────────────────────────────
        mesh_bounds = None
        mesh        = None
        mesh_data   = record.get('mesh') or {}
        # Prefer decimated mesh for bounds extraction; fall back to full mesh
        _dec = ps._resolve_path(classify_id, 'decimated.glb')
        _full = ps._resolve_path(classify_id, 'mesh.glb')
        glb_path = _dec if os.path.exists(_dec) else (_full if os.path.exists(_full) else None)

        if glb_path and os.path.exists(glb_path):
            try:
                import trimesh
                mesh   = trimesh.load(glb_path, force='mesh')
                mesh_bounds = compute_mesh_bounds(mesh)

            except Exception as e:
                log.warning(f"Could not extract mesh bounds: {e}")
                mesh = None

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
            if not glb_path or not os.path.exists(glb_path):
                return jsonify({
                    'error': ('Vision model unavailable and no mesh for geometric '
                              'fallback — run /mesh first, then retry /infer_joints')
                }), 422

            from rig import infer_skeleton_geometric
            n = int(requested_joints) if requested_joints else None
            raw_joints, hierarchy, _ = infer_skeleton_geometric(glb_path, n)

            # raw_joints are world-space coordinates — normalize to 0–1
            # relative to the mesh bounding box so joints_from_model can
            # map them back correctly.
            if mesh_bounds is None:
                if mesh is None:
                    import trimesh
                    mesh = trimesh.load(glb_path, force='mesh')
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
        log.info(f"GLB processing")
        if glb_path and os.path.exists(glb_path):
            if mesh is None:
                import trimesh
                mesh = trimesh.load(glb_path, force='mesh')
            try:
                joints_data = snap_joints_to_mesh(joints_data, mesh)
            except Exception as e:
                log.warning(f"Snapping failed (non-fatal): {e}")
            log.info(f"Snapped")
            try:
                rig_type = classify_data.get('rig_type', '')
                joints_data = mesh_guided_joint_correction(
                    joints_data, mesh, rig_type)
            except Exception as e:
                log.warning(f"Mesh-guided correction failed (non-fatal): {e}")
            log.info(f"Mesh correction")
            try:
                joints_data = enforce_bilateral_symmetry(joints_data)
            except Exception as e:
                log.warning(f"Symmetry enforcement failed (non-fatal): {e}")
            log.info(f"Mirrored")
            try:
                joints_data = verify_and_snap_joints(
                    joints_data, mesh, object_type, rig_type,
                    classify_id=classify_id, mesh_bounds=mesh_bounds)
            except Exception as e:
                log.warning(f"Vision-based joint verification failed (non-fatal): {e}")
            log.info(f"Verified")
            try:
                viz_path = os.path.join(_rdir(classify_id), f"{classify_id}_joints_normalized_viz.glb")
                visualize_normalized_joints(joints_data, mesh, viz_path)
            except Exception as e:
                log.warning(f"Normalized joints viz failed (non-fatal): {e}")
            log.info(f"Visualized")
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


def snap_joints_to_mesh(joints_data: dict, mesh) -> dict:
    """
    Post-processing correction that brings Claude's 2D-estimated positions
    into alignment with the actual 3D mesh geometry.
    For shoulders: snap X and Y to the arm attachment surface,
                   filtering to the stalk/floret junction Y range.
    For hips/wing_base: snap X only — Y is correct from Claude.
    """
    verts  = np.array(mesh.vertices)
    bmin   = verts.min(axis=0)
    bmax   = verts.max(axis=0)
    brange = bmax - bmin
    brange[brange == 0] = 1.0

    hints = joints_data.get('joint_hints', [])
    base_joint_names = [h['name'] for h in hints
                        if any(x in h.get('name', '').lower()
                               for x in ['hip', 'shoulder', 'wing_base'])]

    for hint in hints:
        if hint['name'] not in base_joint_names:
            continue

        pos_norm = hint.get('position_normalized', {})
        world_pos = np.array([
            bmin[0] + pos_norm.get('x', 0.5) * brange[0],
            bmin[1] + pos_norm.get('y', 0.5) * brange[1],
            bmin[2] + pos_norm.get('z', 0.5) * brange[2],
        ])

        is_shoulder = 'shoulder' in hint['name'].lower()

        if is_shoulder:
            # Snap X and Y to arm attachment surface within a Y band
            # centered on the CURRENT guess's own Y, not a fixed absolute
            # range. A fixed range doesn't work: it used to be 0.30-0.65
            # (excluded normal shoulder height on some meshes), then
            # widened to 0.45-0.92 once shoulder_y started tracking
            # neck_y (which can land above 0.82 for a big-headed/
            # short-body mesh) — but shoulder_y can now ALSO land well
            # BELOW 0.45 for a body plan where the arms attach near the
            # waist/collar rather than the neck (e.g. a large head/crown
            # mass on a short torso), which a fixed floor would exclude
            # just as badly as the old ceiling did. Centering on the
            # guess itself (already a real geometric estimate — see
            # mesh_bounds['shoulder_y_detected_left/right'] in
            # /infer_joints) means this step can only refine it against
            # nearby real geometry, not drag it back to some fixed range.
            guess_y = pos_norm.get('y', 0.5)
            band_half = 0.20
            y_lo = bmin[1] + max(0.0, guess_y - band_half) * brange[1]
            y_hi = bmin[1] + min(1.0, guess_y + band_half) * brange[1]
            mask = (verts[:, 1] >= y_lo) & (verts[:, 1] <= y_hi)
            candidates = verts[mask]

            if len(candidates) > 0:
                dists   = np.linalg.norm(candidates - world_pos, axis=1)
                closest = candidates[np.argmin(dists)]
                old_x   = pos_norm.get('x', 0.5)
                old_y   = pos_norm.get('y', 0.5)
                new_x   = float(np.clip((closest[0] - bmin[0]) / brange[0], 0.0, 1.0))
                new_y   = float(np.clip((closest[1] - bmin[1]) / brange[1], 0.0, 1.0))
                hint['position_normalized'] = {'x': new_x, 'y': new_y, 'z': 0.5}
                log.info(f"Snapped {hint['name']} X: {old_x:.2f}→{new_x:.2f} "
                         f"Y: {old_y:.2f}→{new_y:.2f} (3D surface snap)")
            else:
                log.warning(f"No candidates in Y band for {hint['name']}, skipping snap")

        else:
            # Snap X only — Y is correct from Claude
            distances    = np.linalg.norm(verts - world_pos, axis=1)
            closest_vert = verts[np.argmin(distances)]
            old_x        = pos_norm.get('x', 0.5)
            new_x        = float(np.clip((closest_vert[0] - bmin[0]) / brange[0], 0.0, 1.0))
            hint['position_normalized'] = {
                'x': new_x,
                'y': pos_norm.get('y', 0.5),
                'z': 0.5,
            }
            log.info(f"Snapped {hint['name']} X: {old_x:.2f}→{new_x:.2f} "
                     f"(Y unchanged: {pos_norm.get('y', 0.5):.2f})")

    return joints_data


def _joint_marker_color(name: str) -> list:
    """
    Shared color coding for joint markers, used by both the human-facing
    GLB visualization and the vision-model-facing PNG render, so the two
    are showing the literal same construct rather than two independently
    drawn (and potentially inconsistent) representations.
    Red = base joints (hip/shoulder/wing_base),
    yellow = middle joints (knee/elbow/wing_mid),
    green = end joints (hand/foot/head/wing_tip).
    """
    n = name.lower()
    if any(x in n for x in ['hip', 'shoulder', 'wing_base']):
        return [255, 50,  50,  220]
    elif any(x in n for x in ['knee', 'elbow', 'wing_mid']):
        return [255, 255, 50,  220]
    else:
        return [50,  220, 50,  220]


def _build_joint_spheres(joints_data: dict, mesh):
    """
    Build one small icosphere per joint hint, positioned at its world-
    space location and colored per _joint_marker_color. Returns a list of
    (sphere_mesh, joint_name) tuples.

    This is the single source of truth for "what does a joint marker look
    like in 3D" — both visualize_normalized_joints (exports these as part
    of a GLB for a human to inspect) and render_mesh_front_view (rasterizes
    these same spheres, from the same positions, into the PNG sent to the
    vision model for verification) build their markers from this function,
    so a human opening the GLB and the vision model looking at the render
    are guaranteed to be looking at the same joint placements, not two
    independently-implemented approximations of it.
    """
    import trimesh

    mesh_size = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    verts     = np.array(mesh.vertices)
    bmin      = verts.min(axis=0)
    brange    = verts.max(axis=0) - bmin
    brange[brange == 0] = 1.0

    sphere_r = mesh_size * 0.02
    spheres  = []
    for hint in joints_data.get('joint_hints', []):
        p = hint.get('position_normalized', {})
        world_x = bmin[0] + p.get('x', 0.5) * brange[0]
        world_y = bmin[1] + p.get('y', 0.5) * brange[1]
        world_z = bmin[2] + p.get('z', 0.5) * brange[2]

        sphere = trimesh.creation.icosphere(radius=sphere_r)
        sphere.apply_translation([world_x, world_y, world_z])
        sphere.visual.face_colors = _joint_marker_color(hint['name'])
        spheres.append((sphere, hint['name']))

    return spheres


def visualize_normalized_joints(joints_data: dict, mesh,
                                 output_path: str):
    """
    Create GLB showing Claude's normalized joint positions overlaid on mesh.

    Axis mapping (fixed):
      Claude x (left/right) → Blender X [0]
      Claude z (depth)      → Blender Y [1]  (always ~0.5 = center)
      Claude y (up/down)    → Blender Z [2]
    """
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name='mesh')

    for sphere, name in _build_joint_spheres(joints_data, mesh):
        scene.add_geometry(sphere, node_name=name)

    scene.export(output_path)
    log.info(f"Normalized joints viz: {output_path}")
    return output_path


def render_mesh_front_view(mesh, joints_data: dict | None = None,
                            size: tuple = (900, 1100)) -> bytes:
    """
    Orthographic front-view shaded render of the mesh (flat Lambertian
    shading, painter's-algorithm depth sort — no external renderer
    needed), with the SAME joint-marker spheres used by
    visualize_normalized_joints's GLB (built via _build_joint_spheres)
    rasterized alongside the mesh — not a separately-drawn flat overlay.
    Returns PNG bytes.

    Rendering the actual sphere geometry (rather than always-on-top flat
    circles) means occlusion behaves correctly: a joint embedded inside
    the mesh is properly hidden behind the mesh's own front surface, the
    same way it would look if you opened the GLB and looked at it — a
    human inspecting the GLB and the vision model looking at this render
    are seeing the same construct, not two independently approximated
    ones that could quietly drift out of sync.

    The pixel<->normalized-coordinate mapping is exact and uses the SAME
    convention as position_normalized elsewhere in this file (x: 0=left
    edge, 1=right edge; y: 0=bottom, 1=top) — a corrected pixel position
    read off this image by a vision model converts back to that format
    with no reinterpretation needed.
    """
    from PIL import ImageDraw

    W, H = size
    margin = 50
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    brange = bmax - bmin
    brange[brange == 0] = 1.0

    def to_px(v):
        xn = (v[0] - bmin[0]) / brange[0]
        yn = (v[1] - bmin[1]) / brange[1]
        return (xn * (W - 2 * margin) + margin, (1.0 - yn) * (H - 2 * margin) + margin)

    # Combine the mesh's triangles with every joint sphere's triangles
    # into one array so they can be depth-sorted and shaded together —
    # this is what makes correct occlusion between mesh and markers
    # possible, instead of markers always drawn on top regardless of depth.
    all_tri_verts   = [verts[faces]]
    all_base_colors = [np.tile(np.array([150, 150, 150]), (len(faces), 1))]

    joint_world_positions = []  # (name, world_xyz) for label placement
    if joints_data:
        for sphere, name in _build_joint_spheres(joints_data, mesh):
            sv = np.array(sphere.vertices)
            sf = np.array(sphere.faces)
            all_tri_verts.append(sv[sf])
            color = np.array(_joint_marker_color(name)[:3])
            all_base_colors.append(np.tile(color, (len(sf), 1)))
            joint_world_positions.append((name, sphere.vertices.mean(axis=0)))

    tri_verts   = np.concatenate(all_tri_verts, axis=0)
    base_colors = np.concatenate(all_base_colors, axis=0)

    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1
    normals = normals / norm_len
    depth = tri_verts[:, :, 2].mean(axis=1)
    order = np.argsort(depth)

    img  = Image.new('RGB', (W, H), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    # Key + fill lights from different angles, plus a high ambient floor.
    # A single light with a low floor (as this used to be) renders any
    # surface that's nearly edge-on to that one light as almost black —
    # indistinguishable from empty background at a glance. This matters
    # a lot for a thin limb (e.g. an arm) viewed near where it curves:
    # exactly the area a joint marker needs to land on can render as a
    # near-invisible dark sliver even though real geometry is there,
    # which misleads a human (or vision model) reviewing the render into
    # thinking a joint is floating off the mesh when it's actually on it.
    key_dir  = np.array([0.3, 0.4, 1.0]);  key_dir  /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.4, 0.1, 0.6]); fill_dir /= np.linalg.norm(fill_dir)
    ambient  = 0.45

    for i in order:
        n = normals[i]
        if n[2] <= 0:
            continue  # backface cull
        key_shade  = max(0.0, float(np.dot(n, key_dir)))
        fill_shade = max(0.0, float(np.dot(n, fill_dir))) * 0.4
        shade = min(1.0, ambient + key_shade + fill_shade)
        color = tuple(np.clip(base_colors[i] * shade, 0, 255).astype(int))
        draw.polygon([to_px(tri_verts[i, j]) for j in range(3)], fill=color)

    # Labels are the one thing drawn as a flat 2D overlay rather than
    # rasterized in 3D — text has no meaningful 3D depth. Still placed at
    # the exact same world position as the sphere itself (not a separate
    # mapping), and kept small/thin so it doesn't obscure the render.
    for name, world_pos in joint_world_positions:
        px, py = to_px(world_pos)
        draw.text((px + 10, py - 6), name.replace('joint_', ''), fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def verify_and_snap_joints(joints_data: dict, mesh, object_type: str,
                            rig_type: str, max_rounds: int = 1,
                            classify_id: str | None = None,
                            mesh_bounds: dict | None = None) -> dict:
    """
    Render the mesh with the current joint positions and ask a vision
    model to flag any joint that looks clearly wrong against the ACTUAL
    3D geometry (not just the 2D source photo used for the initial
    guess), then geometrically snap each flagged joint's coarse corrected
    position to the nearest real mesh surface point.

    Why the snap step: a vision model reading approximate coordinates off
    a static image is good at noticing "this is obviously wrong" and
    roughly where to look (confirmed: it correctly diagnoses body-plan
    issues like a giant-headed character and points corrections in the
    right direction), but isn't precise enough on its own to land exactly
    on the mesh surface — verified corrections still needed refinement
    round over round without fully converging. Snapping to the nearest
    vertex near its coarse guess is the same technique already used in
    snap_joints_to_mesh, just seeded by a vision-informed target instead
    of a fixed search band.

    Defaults to a single round. A second round was tried and reliably
    made things worse rather than refining them further — verified twice
    on real records: it dragged an already-good hip position from
    y≈0.09 (matching an independent geometric measurement) to y≈0.97
    while its own stated reasoning still said "bottom of the body",
    a self-contradictory answer that ignored the coordinate convention it
    had used correctly one round earlier. The model does not reliably
    "lock in" a correct answer once found — it can still overwrite it with
    a worse one on a later pass — so max_rounds is intentionally 1 rather
    than iterating for refinement.

    After the vision pass, also applies a deterministic geometric safety
    net for hip/pelvis specifically (see the pelvis_y_detected check
    below) — confirmed via two back-to-back /infer_joints?force=true runs
    on the SAME record that the vision pass is not reliable here: one run
    correctly flagged and fixed a badly-wrong hip (0.42, the fixed
    default, for a giant-headed body plan where the real value should be
    ~0.15), the very next run left it untouched at 0.42. Since a reliable
    geometric measurement already exists for this exact question (the
    same width-profile technique used for neck_y, applied to the lower
    body), it's used as a backstop rather than depending on the vision
    model to catch this every time.
    """
    verts  = np.array(mesh.vertices)
    bmin   = verts.min(axis=0)
    bmax   = verts.max(axis=0)
    brange = bmax - bmin
    brange[brange == 0] = 1.0

    hint_by_short_name = {
        h['name'].replace('joint_', ''): h
        for h in joints_data.get('joint_hints', [])
    }

    # Hard scope limit, independent of the prompt: this pass only ever
    # touches hip/shoulder/pelvis. knee/foot/elbow/hand already have
    # solid geometric handling elsewhere in the pipeline (mesh-extremity
    # search, foot centroid, midpoint formulas) — a real run showed the
    # vision pass can drag an already-correct hand joint onto the elbow
    # when allowed to "correct" them too, so this is enforced in code,
    # not just requested in the prompt (which the model doesn't always
    # follow exactly).
    ALLOWED_CORRECTION_NAMES = {
        'hip_left', 'hip_right', 'shoulder_left', 'shoulder_right', 'pelvis',
    }

    for round_i in range(max_rounds):
        try:
            png_bytes  = render_mesh_front_view(mesh, joints_data)
            if classify_id:
                # Persist exactly what the model was shown — without this,
                # the only way to know what drove a given correction was
                # to manually regenerate the render after the fact, which
                # isn't guaranteed to match (joints_data may have moved on).
                render_path = os.path.join(
                    _rdir(classify_id),
                    f"{classify_id}_joint_verification_round{round_i + 1}.png")
                with open(render_path, 'wb') as f:
                    f.write(png_bytes)
            prompt     = utils._build_joint_verification_prompt(object_type, rig_type)
            img_base64 = base64.b64encode(png_bytes).decode('utf-8')
            result     = _try_claude(img_base64, 'image/png', prompt, max_tokens=1024)
        except Exception as e:
            log.warning(f"Joint verification render/vision failed (non-fatal): {e}")
            break

        if not result:
            break
        corrections = result.get('corrections', [])
        log.info(f"Joint verification round {round_i + 1}: "
                 f"{result.get('body_plan_assessment', '')} "
                 f"({len(corrections)} correction(s))")
        if not corrections:
            break

        for c in corrections:
            name = c.get('name')
            if name not in ALLOWED_CORRECTION_NAMES:
                log.warning(f"  Ignoring out-of-scope correction for "
                            f"'{name}' (only hip/shoulder/pelvis allowed)")
                continue
            hint = hint_by_short_name.get(name)
            if hint is None or 'x' not in c or 'y' not in c:
                continue
            try:
                coarse_x = float(np.clip(c['x'], 0.0, 1.0))
                coarse_y = float(np.clip(c['y'], 0.0, 1.0))
            except (TypeError, ValueError):
                continue

            target_world = np.array([
                bmin[0] + coarse_x * brange[0],
                bmin[1] + coarse_y * brange[1],
                bmin[2] + 0.5 * brange[2],
            ])
            distances = np.linalg.norm(verts - target_world, axis=1)
            closest   = verts[np.argmin(distances)]
            new_x = float(np.clip((closest[0] - bmin[0]) / brange[0], 0.0, 1.0))
            new_y = float(np.clip((closest[1] - bmin[1]) / brange[1], 0.0, 1.0))

            old = dict(hint['position_normalized'])
            hint['position_normalized']['x'] = new_x
            hint['position_normalized']['y'] = new_y
            log.info(f"  Vision-verified snap: joint_{name} {old} → "
                     f"coarse=({coarse_x:.2f},{coarse_y:.2f}) → "
                     f"snapped=({new_x:.2f},{new_y:.2f}) [{c.get('reason', '')}]")

    # ── Deterministic geometric safety net for hip/pelvis ────────────────
    # Applied regardless of whether the vision pass above touched these
    # joints. Confirmed the vision pass alone isn't reliable for this:
    # two back-to-back /infer_joints?force=true runs on the same record
    # gave different outcomes (one correctly fixed a badly-wrong hip, the
    # next left it untouched). A reliable geometric measurement already
    # exists for this exact question (mesh_bounds['pelvis_y_detected'],
    # the same width-profile technique used for neck_y) — use it as a
    # backstop rather than depending on the vision model to catch this
    # every time.
    pelvis_y_detected = (mesh_bounds or {}).get('pelvis_y_detected')
    if pelvis_y_detected is not None:
        # Calibrated against two real cases: a genuinely broken hip
        # (broccoli, fixed 0.42 default vs. measured ~0.15 — a ~0.27 gap)
        # should trigger this; an already-reasonable value on a normal
        # A-pose humanoid, where the geometric candidate is itself mildly
        # contaminated by the arm (~0.42 vs ~0.325 — a ~0.10 gap), should
        # not. 0.15 sits between the two.
        MAGNITUDE_THRESHOLD = 0.15
        for name in ('hip_left', 'hip_right'):
            hint = hint_by_short_name.get(name)
            if hint is None:
                continue
            current_y = hint['position_normalized'].get('y', 0.5)
            gap = abs(current_y - pelvis_y_detected)
            if gap > MAGNITUDE_THRESHOLD:
                log.info(f"  Geometric safety net: joint_{name} y "
                         f"{current_y:.3f} → {pelvis_y_detected:.3f} "
                         f"(gap {gap:.3f} exceeds {MAGNITUDE_THRESHOLD} "
                         f"threshold vs. measured waist narrowing)")
                hint['position_normalized']['y'] = pelvis_y_detected

    # pelvis/root must match the hips, not be independently correct —
    # pelvis is structurally the convergence point of hip_left/hip_right,
    # and root is coincident with pelvis (see the "root: y ≈ pelvis_y"
    # prompt instruction). Checking pelvis against pelvis_y_detected
    # SEPARATELY from hip (as an earlier version of this safety net did)
    # missed a real case: hip_left/hip_right's gap fell under the
    # threshold and got left alone, but pelvis's own vision-pass value
    # had a smaller, also-under-threshold gap in a DIFFERENT direction —
    # leaving pelvis 0.08 away from where the hips actually ended up.
    # (This inconsistency is real and worth fixing on its own, but it
    # was NOT the cause of the leg/spine weight contamination seen on
    # the tomato — that turned out to be a separate bug in
    # build_segment_weights's spine/limb distance partition; see rig.py.)
    hip_left_hint  = hint_by_short_name.get('hip_left')
    hip_right_hint = hint_by_short_name.get('hip_right')
    if hip_left_hint is not None and hip_right_hint is not None:
        hip_center_y = (hip_left_hint['position_normalized'].get('y', 0.5)
                        + hip_right_hint['position_normalized'].get('y', 0.5)) / 2
        for name in ('pelvis', 'root'):
            hint = hint_by_short_name.get(name)
            if hint is None:
                continue
            old_y = hint['position_normalized'].get('y', 0.5)
            if abs(old_y - hip_center_y) > 1e-6:
                log.info(f"  Pelvis/root consistency: joint_{name} y "
                         f"{old_y:.3f} → {hip_center_y:.3f} "
                         f"(matching hip_left/hip_right convergence point)")
                hint['position_normalized']['y'] = hip_center_y

    return joints_data

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

        record = hydrate(_store.get(classify_id))
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found — run /classify first"}), 404

        active_path = record.get('active_image_path')
        if not active_path or not os.path.exists(active_path):
            return jsonify({'error': f"Active image not found: '{active_path}'"}), 404

        # ── Augmentation-skipped guard ────────────────────────────────────────
        # classify flagged this object as needing a pose change, but no
        # augmented variant has been confirmed as active yet — meshing now
        # would silently use the original (e.g. limbless) image instead.
        # This is exactly what happened for real records before this check
        # existed: augmentation was generated but /augment_image/confirm was
        # never called, so /mesh quietly used the un-augmented image and
        # spent a Meshy credit on a mesh that didn't match the intended pose.
        classify_data      = record.get('classify') or {}
        active_image_key   = record.get('active_image_key')
        if (classify_data.get('needs_augmentation') and not force
                and active_image_key in (None, '', 'segmented')):
            return jsonify({
                'error': (
                    'needs_augmentation is true for this object but no '
                    'augmented image has been confirmed — /mesh would use '
                    'the original, un-augmented image. Call /augment_image '
                    'then /augment_image/confirm first, or pass '
                    '?force=true to mesh the original image anyway.'
                ),
                'needs_augmentation': True,
                'active_image_key':   active_image_key,
                'classify_id':        classify_id,
            }), 422

        # ── Cache hit ─────────────────────────────────────────────────────────
        # glb_path is never persisted in the store (paths are reconstructed
        # from classify_id at read time, same as active_image_path) — it must
        # be resolved here rather than read off the record, or this check can
        # never succeed.
        existing_mesh     = record.get('mesh') or {}
        existing_glb_path = ps._resolve_path(classify_id, 'mesh.glb')
        mesh_exists        = bool(existing_mesh) and os.path.exists(existing_glb_path)

        # A confirmed augmented (or re-augmented) image can be newer than an
        # already-generated mesh — e.g. /mesh ran once against the original
        # image before augmentation was confirmed. Comparing filesystem
        # mtimes directly (image file vs. GLB file) catches this
        # automatically, without requiring the caller to remember
        # ?force=true every time the active image changes underneath an
        # existing mesh. Deliberately NOT comparing against the mesh's
        # stored created_at string — _now() truncates to whole-second
        # precision, so a mesh created within the same second as its own
        # source image would almost always look (falsely) older than the
        # image once truncated, making the cache spuriously miss on nearly
        # every real request. Two real mtimes have matching precision.
        mesh_is_stale = False
        if mesh_exists:
            try:
                if os.path.getmtime(active_path) > os.path.getmtime(existing_glb_path):
                    mesh_is_stale = True
                    log.info(f"Active image is newer than cached mesh for "
                             f"{classify_id} — regenerating instead of using cache")
            except OSError as e:
                log.warning(f"Could not compare mesh/image timestamps for "
                            f"{classify_id}: {e}")

        if mesh_exists and not force and not mesh_is_stale:
            log.info(f"Mesh cache hit: {classify_id}")
            return jsonify({
                'status':        'ok',
                'task_id':       None,
                'glb_url':       existing_mesh.get('glb_url'),
                'glb_local_url': _local_url(existing_glb_path, request.host),
                'classify_id':   classify_id,
            })

        with open(active_path, 'rb') as f:
            img_bytes = f.read()

        object_type = classify_data.get('object_type', '') or \
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


# ── /mesh/upload ──────────────────────────────────────────────────────────────

_RENDER_BACKGROUND = (235, 235, 235)


def front_view_cutout(mesh) -> bytes:
    """
    Front-view render of a mesh as a transparent PNG cropped to the mesh.

    Stands in for the segmented drawing that /classify normally stores: the
    vision models classify it, and /infer_joints crops the active image to its
    alpha bounding box and maps that box onto the mesh's x/y bounds. Cropping
    the render to the mesh's own silhouette keeps those two boxes identical.
    """
    render = Image.open(io.BytesIO(render_mesh_front_view(mesh))).convert('RGBA')
    pixels = np.array(render)
    background = np.all(pixels[:, :, :3] == _RENDER_BACKGROUND, axis=2)
    pixels[background, 3] = 0
    cutout = Image.fromarray(pixels)
    bbox = cutout.getchannel('A').getbbox()
    if bbox:
        cutout = cutout.crop(bbox)
    buf = io.BytesIO()
    cutout.save(buf, format='PNG')
    return buf.getvalue()


@app.route('/mesh/upload', methods=['GET', 'POST'])
def mesh_upload():
    """
    Start a pipeline record from an existing, unrigged GLB instead of a drawing.

    Body: raw GLB bytes. ?tag= and ?user_id= as for /classify.

    Saves the GLB as the record's mesh, renders its front view as the active
    image, and classifies that render — so /infer_joints and /rig then run on
    the returned classify_id exactly as they do after /classify + /mesh.
    Cached by content: the same GLB and tag return the same classify_id.
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        tag     = request.args.get('tag', '').strip()
        user_id = request.args.get('user_id', '').strip() or dummy_user_id

        glb_bytes = request.stream.read()
        if not glb_bytes:
            return jsonify({'error': 'No data received'}), 400
        if glb_bytes[:4] != b'glTF':
            return jsonify({'error': 'Body is not a binary glTF (GLB) file'}), 400

        classify_id = hashlib.md5(glb_bytes + tag.encode()).hexdigest()[:8]

        record = hydrate(_store.get(classify_id))
        glb_path = ps._resolve_path(classify_id, 'mesh.glb')
        if (record and record.get('classify') and record.get('mesh')
                and os.path.exists(glb_path)):
            log.info(f"mesh upload cache hit: {classify_id}")
            return jsonify({
                **record['classify'],
                'classify_id':       classify_id,
                'active_image_path': record.get('active_image_path'),
                'glb_local_url':     _local_url(glb_path, request.host),
            })

        import trimesh
        try:
            mesh = trimesh.load(io.BytesIO(glb_bytes), file_type='glb',
                                force='mesh')
        except Exception as e:
            return jsonify({'error': f'Could not read GLB: {e}'}), 400
        if mesh is None or len(getattr(mesh, 'faces', [])) == 0:
            return jsonify({'error': 'GLB contains no mesh geometry'}), 400

        _rd = _rdir(classify_id)
        glb_path = os.path.join(_rd, f"{classify_id}_mesh.glb")
        with open(glb_path, 'wb') as f:
            f.write(glb_bytes)

        png_bytes = front_view_cutout(mesh)
        seg_path = os.path.join(_rd, f"{classify_id}_segmented.png")
        with open(seg_path, 'wb') as f:
            f.write(png_bytes)

        info = classify_with_vision(png_bytes, 'image/png', tag or None)
        # The pose is whatever the uploaded mesh already has; there's no
        # drawing to repose, and /augment_image would only redraw the render.
        info['needs_augmentation'] = False
        info['source'] = 'mesh_upload'
        log.info(f"mesh upload {classify_id}: {info.get('object_type', '?')} "
                 f"({len(mesh.faces)} faces)")

        _store.upsert_classify(classify_id, tag, info)
        _store.upsert_mesh(classify_id, {
            'mesh_hash': hashlib.md5(glb_bytes).hexdigest()[:12],
            'source':    'upload',
            'user_id':   user_id,
            'glb_path':  glb_path,
        })

        return jsonify({
            **info,
            'classify_id':       classify_id,
            'active_image_path': seg_path,
            'glb_local_url':     _local_url(glb_path, request.host),
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/mesh/upload error: {e}")
        return jsonify({'error': str(e)}), 500

def _run_meshy_rig_task(task_id: str, classify_id: str, user_id: str, host: str):
    try:
        _rig_tasks[task_id] = {'status': 'rigging', 'progress': 10}
        _store.set_rig_status(classify_id, 'rigging')

        record    = hydrate(_store.get(classify_id))
        mesh_data = record.get('mesh') or {}
        glb_url   = mesh_data.get('glb_url')

        # ── Validate ──────────────────────────────────────────────────────────
        glb_path = ps._resolve_path(classify_id, 'mesh.glb')
        if not os.path.exists(glb_path):
            raise RuntimeError("Mesh GLB not found — run /mesh before /rig")

        meshy_task_id = mesh_data.get('meshy_task_id')
        if not meshy_task_id:
            raise RuntimeError(
                "should_autorig requires a meshy_task_id on the mesh record — "
                "ensure /mesh completed successfully via Meshy"
            )

        # ── Submit rig task ───────────────────────────────────────────────────
        _rig_tasks[task_id] = {'status': 'meshy', 'progress': 20}
        meshy_rig_task_id, rigged_glb_url, rigged_fbx_url = meshy_rig(meshy_task_id)

        # ── Download ──────────────────────────────────────────────────────────
        _rig_tasks[task_id] = {'status': 'downloading', 'progress': 70}
        _rd         = _rdir(classify_id)
        rigged_path = os.path.join(_rd, f"{classify_id}_rigged.glb")
        download_file(rigged_glb_url, rigged_path)

        # ── Persist ───────────────────────────────────────────────────────────
        _rig_tasks[task_id] = {'status': 'finalizing', 'progress': 90}
        _store.upsert_rig(classify_id, {
            'status':             'ok',
            'user_id':            user_id,
            'meshy_rig_task_id':  meshy_rig_task_id,
            'rigged_glb_url':     rigged_glb_url,
            'rigged_fbx_url':     rigged_fbx_url,
        })

        _rig_tasks[task_id] = {
            'status':      'ok',
            'progress':    100,
            'rigged_url':  _local_url(rigged_path, host),
            'glb_url':     glb_url,
            'classify_id': classify_id,
        }
        log.info(f"Meshy rig task {task_id} complete: {rigged_path}")

    except Exception as e:
        log.error(f"Meshy rig task {task_id} failed: {e}")
        _rig_tasks[task_id] = {'status': 'error', 'error': str(e)}
        _store.set_rig_status(classify_id, 'error', str(e))
        
def meshy_rig(meshy_task_id: str) -> tuple[str, str, str]:
    """
    Submit a Meshy rigging task and poll until done.
    Returns (rig_task_id, rigged_glb_url, rigged_fbx_url).
    Mirrors meshy_reconstruct's submit-then-poll pattern.
    """
    meshy_key = os.environ.get('MESHY_API_KEY')
    headers = {
        'Authorization': f'Bearer {meshy_key}',
        'Content-Type':  'application/json',
    }

    # Submit
    MESHY_API_BASE = 'https://api.meshy.ai/openapi/v1'
    resp = requests.post(
        f'{MESHY_API_BASE}/rigging',
        headers=headers,
        json={'input_task_id': meshy_task_id},
        timeout=30,
    )
    resp.raise_for_status()
    rig_task_id = resp.json()['result']
    log.info(f"Meshy rig task submitted: {rig_task_id}")

    # Poll
    while True:
        poll = requests.get(
            f'{MESHY_API_BASE}/rigging/{rig_task_id}',
            headers=headers,
            timeout=30,
        )
        poll.raise_for_status()
        data   = poll.json()
        status = data.get('status')
        log.info(f"Meshy rig {rig_task_id}: {status} {data.get('progress', 0)}%")

        if status == 'SUCCEEDED':
            result = data['result']
            return (
                rig_task_id,
                result['rigged_character_glb_url'],
                result['rigged_character_fbx_url'],
            )
        if status == 'FAILED':
            msg = data.get('task_error', {}).get('message', 'unknown error')
            raise RuntimeError(f"Meshy rig failed: {msg}")

        import time; time.sleep(5)
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
        should_autorig       = request.args.get('should_autorig', '').lower() in ('1', 'true', 'yes')

        if not classify_id:
            return jsonify({'error': 'Missing classify_id'}), 400

        record = hydrate(_store.get(classify_id))
        if not record:
            return jsonify({'error': f"classify_id '{classify_id}' not found"}), 404

        mesh_data = record.get('mesh') or {}
        glb_path  = ps._resolve_path(classify_id, 'mesh.glb')
        if not os.path.exists(glb_path):
            return jsonify({
                'error':       'No mesh found — run /mesh before /rig',
                'classify_id': classify_id,
            }), 422

        rig_data = record.get('rig') or {}
        if not force and rig_data.get('status') == 'ok':
            rigged_path = ps._resolve_path(classify_id, 'rigged.glb')
            if os.path.exists(rigged_path):
                log.info(f"Rig cache hit: {classify_id}")
                return jsonify({
                    'status':      'ok',
                    'task_id':     None,
                    'rigged_url':  _local_url(rigged_path, request.host),
                    'glb_url':     mesh_data.get('glb_url'),
                    'classify_id': classify_id,
                })

        task_id = str(uuid.uuid4())[:8]
        _rig_tasks[task_id] = {'status': 'started', 'progress': 0}
        _store.set_rig_status(classify_id, 'started')

        log.info(f"Starting rig task {task_id} for {classify_id}")
        if should_autorig:
            threading.Thread(
                target=_run_meshy_rig_task,
                args=(task_id, classify_id, user_id, request.host),
                daemon=True,
            ).start()
        else:
            if not record.get('joints'):
                log.warning(f"No joints for {classify_id} — geometric fallback will be used")
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


 
def _run_blender(script_source, workspace):
    """
    Run Blender headless with a script written to disk.
 
    Fixes vs. the original:
    - Script is written to a file and passed via --python (no shell-quoting /
      --python-expr edge cases with paths containing quotes or spaces).
    """
    script_path = os.path.join(workspace, "blender_job.py")
    with open(script_path, "w") as f:
        f.write(script_source)
 
    env = os.environ.copy()
    env.update({
        "TMPDIR": workspace,
        "TEMP": workspace,
        "TMP": workspace,
        "HOME": env.get("HOME") or workspace,
        "XDG_CACHE_HOME": os.path.join(workspace, ".cache"),
    })
 
    cmd = [
        BLENDER_BIN,
        "--background",
        "--factory-startup",
        "-noaudio",
        "--python-exit-code", "1",   # ← add this
        "--python", script_path,
    ]
    result = subprocess.run(
        cmd,
        cwd=workspace,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender exited with code {result.returncode}: "
            f"{result.stderr or result.stdout}"
        )
 
 
def _decimate_script(in_path, out_path, ratio, fmt):
    """Build the Blender-side script for GLB or USDZ decimation."""
    if fmt == "usdz":
        import_line = (
            f"bpy.ops.wm.usd_import(filepath={in_path!r}, import_materials=True)"
        )
        export_line = (
            f"bpy.ops.wm.usd_export(filepath={out_path!r}, "
            f"export_materials=True, export_textures=True)"
        )
    else:  # glb
        import_line = f"bpy.ops.import_scene.gltf(filepath={in_path!r})"
        export_line = (
            f"bpy.ops.export_scene.gltf(filepath={out_path!r}, "
            f"export_format='GLB')"
        )
 
    return f"""
import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
{import_line}

meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not meshes:
    raise RuntimeError("Import produced no mesh objects - check the input file")

for obj in meshes:
    mod = obj.modifiers.new(name="AutoDecimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = {ratio}
    if hasattr(mod, "use_collapse_triangulate"):
        mod.use_collapse_triangulate = True
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=mod.name)

{export_line}
"""
 
 
def _fetch_or_read_payload(url_param):
    """Get mesh bytes either from a URL query param or the request body."""
    url = request.args.get(url_param, "").strip()
    if url:
        # NOTE: verify=False disables TLS validation — keep only if you truly
        # need to hit self-signed internal hosts; prefer verify=True.
        resp = requests.get(url, verify=False, timeout=60)
        resp.raise_for_status()
        return resp.content
    return request.get_data()
 
 
def _handle_decimate(fmt):
    ratio = max(0.01, min(1.0, float(request.args.get("ratio", "0.1"))))
    payload = _fetch_or_read_payload(f"{fmt}_url")
    if not payload:
        return jsonify({"error": f"No {fmt.upper()} data received"}), 400
 
    ws = _make_workspace()
    in_path = os.path.abspath(os.path.join(ws, f"input.{fmt}"))
    out_path = os.path.abspath(os.path.join(ws, f"decimated.{fmt}"))
 
    with open(in_path, "wb") as f:
        f.write(payload)
 
    _run_blender(_decimate_script(in_path, out_path, ratio, fmt), ws)
 
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            "Blender ran but did not produce an output file. For USDZ export "
            "you need Blender 4.x — earlier versions cannot write .usdz."
        )
 
    # Best-effort cleanup; never let cleanup crash the request.
    try:
        os.unlink(in_path)
    except OSError:
        pass
 
    return jsonify({
        "status": "ok",
        "url": _local_url(out_path, request.host),  # noqa: F821 (defined elsewhere)
        "ratio": ratio,
    })
 
 
@app.route("/decimate_glb", methods=["GET", "POST"])  # noqa: F821
def decimate():
    """Decimate a GLB mesh. Accepts raw bytes or ?glb_url=..."""
    if request.method == "GET":
        return jsonify({"status": "ok"}), 200
    try:
        return _handle_decimate("glb")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/decimate_glb error: {e}")  # noqa: F821
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/decimate_usdz", methods=["GET", "POST"])  # noqa: F821
def decimate_usdz():
    """Decimate a USDZ mesh. Accepts raw bytes or ?usdz_url=..."""
    if request.method == "GET":
        return jsonify({"status": "ok"}), 200
    try:
        return _handle_decimate("usdz")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/decimate_usdz error: {e}")  # noqa: F821
        return jsonify({"error": str(e)}), 500

#── /convert_to_usdz ──────────────────────────────────────────────────────────

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
    log.info(result.stdout)
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
        _ud       = os.path.join(RESULTS_DIR, uid)
        os.makedirs(_ud, exist_ok=True)
        glb_path  = os.path.join(_ud, "temp.glb")
        usdz_path = os.path.join(_ud, "converted.usdz")
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

@app.route('/results/<path:filename>')
def serve_result(filename):
    full = os.path.join(RESULTS_DIR, filename)
    if not os.path.abspath(full).startswith(os.path.abspath(RESULTS_DIR)):
        return jsonify({'error': 'forbidden'}), 403
    if not os.path.exists(full):
        return jsonify({'error': 'not found'}), 404
    response = make_response(send_file(full))
    response.headers['ngrok-skip-browser-warning'] = 'true'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/examples', methods=['GET'])
def list_assets():
    """List all GLB files in the examples directory."""
    glbs = []
    for root, _, files in os.walk(EXAMPLES_DIR):
        for f in files:
            if f.endswith(('.glb', '.usdz', '.gltf')):
                rel = os.path.relpath(os.path.join(root, f), EXAMPLES_DIR)
                glbs.append({
                    'filename': rel,
                    'url': _local_url(os.path.join(ASSETS_DIR, rel), request.host),
                    'size_kb': round(os.path.getsize(os.path.join(root, f)) / 1024, 1),
                })
    return jsonify(glbs)

@app.route('/examples/<path:filename>')
def serve_asset(filename):
    """Serve a file from the assets directory."""
    full = os.path.abspath(os.path.join(EXAMPLES_DIR, filename))
    if not full.startswith(os.path.abspath(EXAMPLES_DIR)):
        return jsonify({'error': 'forbidden'}), 403
    if not os.path.exists(full):
        return jsonify({'error': 'not found'}), 404
    mime = 'model/gltf-binary' if filename.endswith('.glb') else \
           'model/vnd.usdz+zip' if filename.endswith('.usdz') else None
    return send_file(full, mimetype=mime)

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
    base = request.host_url.rstrip('/')
    try:
        # ── Store records (indexed by classify_id) ────────────────────────
        records = _store.get_by_user(user_id)
        store_map = {r['classify_id']: r for r in records}

        # ── Scan results dir for all folders with GLBs ────────────────────
        seen = set()
        items = []

        def make_item(cid, r=None):
            rigged_disk = ps._resolve_path(cid, 'rigged.glb')
            mesh_disk   = ps._resolve_path(cid, 'mesh.glb')
            folder      = os.path.join(RESULTS_DIR, cid)

            # All GLBs in folder, excluding utility files
            all_glbs = []
            if os.path.isdir(folder):
                all_glbs = sorted([
                    f for f in os.listdir(folder)
                    if f.endswith('.glb')
                    and not any(x in f for x in ('_viz', '_decimated', '_skeleton', '_armature'))
                ])

            has_rig  = os.path.exists(rigged_disk)
            has_mesh = os.path.exists(mesh_disk)

            rig_ok     = has_rig and (r is None or (r.get('rig') or {}).get('status') == 'ok')
            rigged_url = f"{base}/results/{cid}/{cid}_rigged.glb" if rig_ok  else None
            mesh_url   = f"{base}/results/{cid}/{cid}_mesh.glb"   if has_mesh else None

            # Fallback: use any GLB found in the folder
            if not rigged_url and not mesh_url and all_glbs:
                mesh_url = f"{base}/results/{cid}/{all_glbs[0]}"

            # Extra GLBs beyond the standard pair
            standard = {f"{cid}_rigged.glb", f"{cid}_mesh.glb"}
            extra_glbs = [
                f"{base}/results/{cid}/{f}" for f in all_glbs
                if f not in standard
            ]

            rig_data      = (r.get('rig')      or {}) if r else {}
            classify_data = (r.get('classify') or {}) if r else {}
            label = (r.get('tag') or classify_data.get('object_type') or cid) if r else cid
            tags  = [label] if label else [cid]

            return {
                'classify_id': cid,
                'label':       label,
                'tags':        tags,
                'has_joints':  bool(r.get('joints')) if r else False,
                'has_mesh':    has_mesh or bool(all_glbs),
                'rig_status':  rig_data.get('status'),
                'created_at':  rig_data.get('created_at'),
                'rigged_path': rigged_url,
                'active_path': mesh_url,
                'extra_glbs':  extra_glbs,
                'url':         rigged_url or mesh_url,
                'in_store':    r is not None,
            }

        # ── Walk results dir ──────────────────────────────────────────────
        if os.path.isdir(RESULTS_DIR):
            for cid in sorted(os.listdir(RESULTS_DIR), reverse=True):
                folder = os.path.join(RESULTS_DIR, cid)
                if not os.path.isdir(folder):
                    continue
                has_any_glb = any(f.endswith('.glb') for f in os.listdir(folder))
                if not has_any_glb:
                    continue
                seen.add(cid)
                r = store_map.get(cid)
                items.append(make_item(cid, r))

        # ── Add store records whose folder may have been deleted ──────────
        for cid, r in store_map.items():
            if cid not in seen:
                items.append(make_item(cid, r))

        # ── Sort: store records with rigs first, then by created_at ───────
        items.sort(key=lambda x: (
            not x['in_store'],
            x.get('created_at') or '',
        ), reverse=True)

        return jsonify(items)
    except Exception as e:
        log.error(f"/gallery_data error: {e}")
        return jsonify({'error': str(e)}), 500
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 6000)), debug=False)
