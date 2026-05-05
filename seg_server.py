"""
seg_server.py

Flask server for MIT App Inventor image processing pipeline.

Routes:
  GET  /health                   — health check
  POST /segment                  — remove background (rembg)
  POST /classify                 — identify object + infer rigging hints (Gemini)
  POST /augment_image?prompt=... — edit image via fal.ai
  POST /rig?type=bird            — full pipeline: Meshy → rig.py → rigged GLB URL
 

Pipeline:
  1. User takes photo
  2. POST /segment               → remove background, user approves clean subject
  3. POST /classify?tag=...      → Gemini analyzes segmented image + optional tag
                                   tag has two jobs:
                                     identity:   "bicycle" — what it is
                                     descriptor: "rolling" — how it moves
  4. if needs_augmentation
       POST /augment_image       → add wings/elements, user chooses original or augmented
  
   5. POST /rig                   → Meshy 3D + skeleton → rigged GLB URL for ModelNode
 

Requirements:
  pip install flask rembg pillow requests fal-client google-genai

Environment variables:
  GEMINI_API_KEY  — free at aistudio.google.com
  FAL_KEY         — get at fal.ai
  MESHY_API_KEY   — get at meshy.ai (paid)
"""

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

_rig_tasks = {}
_classify_cache = {}
_blender_lock = threading.Lock()
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_mesh_cache    = {}

HUMANOID_KEYWORDS = {'human', 'person', 'character', 'humanoid', 'man',
                             'woman', 'boy', 'girl', 'robot', 'alien', 'zombie',
                             'totoro', 'creature', 'figure', 'monster'}
                             
VEHICLE_KEYWORDS = {'car', 'vehicle', 'wheels', 'truck', 'auto', 'bus', 'bike', 'motorcycle', 'van'}

# ── App setup ─────────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
logging.basicConfig(level=logging.INFO)
log = app.logger

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load rembg once at startup ────────────────────────────────────────────────

log.info("Loading rembg model...")
rembg_session = new_session("u2net")
log.info("rembg ready.\n")


CLASSIFY_CACHE_FILE = os.path.join(RESULTS_DIR, '_classify_cache.json')

def load_classify_cache():
    if os.path.exists(CLASSIFY_CACHE_FILE):
        try:
            with open(CLASSIFY_CACHE_FILE) as f:
                data = json.load(f)
            log.info(f"Loaded {len(data)} cached classifications from disk")
            return data
        except Exception as e:
            log.warning(f"Could not load classify cache: {e}")
    return {}
    
    
    
MESH_CACHE_FILE = os.path.join(RESULTS_DIR, '_mesh_cache.json')

def load_mesh_cache():
    if os.path.exists(MESH_CACHE_FILE):
        try:
            with open(MESH_CACHE_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Could not load mesh cache: {e}")
    return {}

def save_mesh_cache():
    try:
        with open(MESH_CACHE_FILE, 'w') as f:
            json.dump(_mesh_cache, f, indent=2)
    except Exception as e:
        log.warning(f"Could not save mesh cache: {e}")

_classify_cache = load_classify_cache()
# ── fal.ai image editing ──────────────────────────────────────────────────────

def edit_image_fal(img: Image.Image, prompt: str) -> Image.Image:
    """Edit image using fal.ai Qwen Image 2.0. No mask needed."""
    import fal_client

    data_uri = f"data:image/png;base64,{img_to_b64(img)}"
    log.info(f"fal.ai: {prompt[:80]}...")

    result   = fal_client.subscribe(
        "fal-ai/qwen-image-2/edit",
        arguments={"prompt": prompt, "image_urls": [data_uri], "num_images": 2}
    )
    out_url  = result["images"][0]["url"]
    out_url2  = result["images"][1]["url"]
    responseA = requests.get(out_url, verify=False)
    responseB = requests.get(out_url2, verify=False)
    return Image.open(io.BytesIO(responseA.content)).convert('RGB'), Image.open(io.BytesIO(responseB.content)).convert('RGB')

_mesh_cache     = load_mesh_cache()

# ── Error handlers ─────────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({'error': 'File too large, max 64MB'}), 413


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/segment', methods=['GET', 'POST'])
def segment():
    """
    Step 1: Remove background using rembg.
    User approves clean subject before classification proceeds.

    Accepts: raw image bytes
    Returns: PNG with background removed
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200

    try:
        log.info(f"Content-Type: {request.content_type} | Length: {request.content_length}")
        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No data received'}), 400

        log.info(f"Bytes received: {len(img_bytes)}")
        img    = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img    = resize_if_needed(img, max_size=1024)
        output = remove(img, session=rembg_session)

        buf = io.BytesIO()
        output.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/segment error: {e}")
        return jsonify({'error': str(e)}), 500


def get_image_bytes() -> bytes:
    """Read image from wherever App Inventor's PostFile puts it."""
    # Raw bytes (curl --data-binary)
    data = request.stream.read()
    if data:
        return data
    # Multipart form upload (App Inventor PostFile)
    if request.files:
        f = next(iter(request.files.values()))
        return f.read()
    # form data body
    if request.data:
        return request.data
    return b''

def classify_with_vision(img_bytes, mime_type, user_tag=None,
                         mesh_bounds_size=None, requested_joints=None):  # ← add param
    
    import json
    import time
    import re
    import base64
    import os

    tag_context = f"\nThe user identified this as: '{user_tag}'." if user_tag else ""

    # Resize BEFORE base64 encoding
    print(f"Original image size: {len(img_bytes)} bytes")
    img = Image.open(io.BytesIO(img_bytes))
    print(f"Image dimensions: {img.size}")
    
    # Resize to fit Claude's 5MB limit
    img = resize_if_needed(img, max_size=1024)
    
    # Re-encode to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)  # optimize=True compresses better
    img_bytes = buf.getvalue()
    
    print(f"Resized image size: {len(img_bytes)} bytes")
    
    tag_context = f"\nThe user identified this as: '{user_tag}'." if user_tag else ""


    better_mime_type = detect_mime_type(img_bytes)
    MIN_JOINTS = 3
    MAX_JOINTS = 16
    if requested_joints:
        try:
            requested_joints = max(MIN_JOINTS, min(MAX_JOINTS, int(requested_joints)))
        except (ValueError, TypeError):
            requested_joints = None  # ignore bad input
    joints_instruction = (
        f"\nYou MUST generate EXACTLY {requested_joints} joints in joint_hints. "
        f"Set suggested_joints to {requested_joints}."
    ) if requested_joints else (
        f"\nGenerate between {MIN_JOINTS} and {MAX_JOINTS} joints. "
        f"Choose the count that best represents the object's articulation."
    )
    
    bounds_info = ""
    if mesh_bounds_size:
        bounds_info = (
            f"\nMesh bounding box: {mesh_bounds_size:.3f} units. "
            f"Scale keyframe values to max {mesh_bounds_size * 0.05:.4f} units."
        )
        


    tag_words = set((user_tag or '').lower().split('+'))

    if tag_words & VEHICLE_KEYWORDS:  # set intersection — any keyword matches
        prompt = VEHICLE_PROMPT
    else:
        prompt = animal_prompt
    
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')



    # 🔴  CLAUDE
    print("🔴 attempting Claude...")
    result = _try_claude(img_base64, better_mime_type, prompt)
    if result:
        print("✅ Claude succeeded")
        return result
        
    # 🔷 TRY GEMINI FIRST
    print("🔷 Attempting Gemini...")
    result = _try_gemini(img_bytes, better_mime_type, prompt)
    if result:
        print("✅ Gemini succeeded")
        return result
    # 🟠 FALLBACK TO OPENAI
    print("🟠 Claude failed, attempting OpenAI...")
    result = _try_openai(img_base64, better_mime_type, prompt)
    if result:
        print("✅ OpenAI succeeded")
        return result

    raise RuntimeError("All vision APIs exhausted or unavailable")

def _try_gemini(img_bytes: bytes, mime_type: str, prompt: str):
    """Try Gemini models with retry logic"""
    import os
    import time
    import re
    import json
    from google import genai
    from google.genai import types

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        print("  GEMINI_API_KEY not set")
        return None

    try:
        client = genai.Client(api_key=api_key)
        models = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash']
        
        for model in models:
            for attempt in range(3):
                try:
                    print(f"  Trying {model} (attempt {attempt + 1}/3)...")
                    response = client.models.generate_content(
                        model=model,
                        contents=[
                            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
                            prompt
                        ]
                    )
                    raw = response.text.strip()
                    if raw.startswith('```'):
                        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
                    return json.loads(raw.strip())

                except Exception as e:
                    err_str = str(e)
                    
                    if '429' in err_str or 'RESOURCE_EXHAUSTED' in err_str:
                        delay_match = re.search(r'retryDelay.*?(\d+)s', err_str)
                        suggested = int(delay_match.group(1)) if delay_match else 0
                        
                        if 'PerDay' in err_str or suggested > 60:
                            print(f"  {model} daily quota exhausted")
                            break  # Try next model
                        
                        wait = max(suggested, 15 * (attempt + 1))
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                    
                    elif '503' in err_str or 'UNAVAILABLE' in err_str:
                        wait = 5 * (attempt + 1)
                        print(f"  Service unavailable, waiting {wait}s...")
                        time.sleep(wait)
                    
                    elif attempt == 2:  # Last attempt
                        raise
                        
    except Exception as e:
        print(f"  ❌ Gemini failed: {str(e)[:100]}")
        return None


def _try_claude(img_base64: str, mime_type: str, prompt: str):
    """Try Claude with FULL error logging"""
    import os
    import json
    import time
    import anthropic
    import base64

    try:
        api_key = os.environ.get('CLAUDE_API_KEY')
        if not api_key:
            print("  CLAUDE_API_KEY not set")
            return None

        # ✅ Validate base64
        img_base64 = img_base64.replace('\n', '').replace('\r', '').replace(' ', '')
        
        try:
            base64.b64decode(img_base64, validate=True)
            print(f"  ✓ Base64 valid ({len(img_base64)} chars)")
        except Exception as e:
            print(f"  ❌ Base64 decode failed: {e}")
            return None

        client = anthropic.Anthropic(api_key=api_key)
        models_to_try = [
            "claude-opus-4-7",
            "claude-sonnet-4-6",
        ]
        
        for model_name in models_to_try:
            for attempt in range(2):
                try:
                    print(f"  Trying {model_name} (attempt {attempt + 1}/2)...")
                    
                    media_type_map = {
                        'image/jpeg': 'image/jpeg',
                        'image/png': 'image/png',
                        'image/webp': 'image/webp',
                        'image/gif': 'image/gif'
                    }
                    media_type = media_type_map.get(mime_type, 'image/jpeg')
                    
                    print(f"  Media type: {media_type}, base64 len: {len(img_base64)}")
                    print(f"  Prompt length: {len(prompt)}")
                    
                    # ✅ Build the request explicitly
                    request_payload = {
                        "model": model_name,
                        "max_tokens": 4096,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": img_base64
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": prompt  # ← Truncate
                                    }
                                ]
                            }
                        ]
                    }
                    
                    # DEBUG: Log the request structure
                    print(f"  Request keys: {list(request_payload.keys())}")
                    print(f"  Message content types: {[c['type'] for c in request_payload['messages'][0]['content']]}")
                    
                    response = client.messages.create(**request_payload)
                    
                    raw = response.content[0].text.strip()
                    print(f"  ✅ Claude responded ({len(raw)} chars)")
                    
                    json_obj = _extract_json(raw)
                    if json_obj:
                        return json_obj
                    else:
                        raise ValueError("Could not extract valid JSON from response")

                except anthropic.NotFoundError as e:
                    print(f"  {model_name} not available")
                    print(f"  Error details: {e}")
                    break
                    
                except anthropic.RateLimitError as e:
                    if attempt == 0:
                        print(f"  Rate limited: {e.message}")
                        time.sleep(30)
                    else:
                        raise
                        
                except anthropic.BadRequestError as e:
                    # ← Catch 400 Bad Request explicitly
                    print(f"  ❌ 400 Bad Request from Claude API")
                    print(f"  Full error message: {e.message}")  # ← THIS is the key
                    print(f"  Error body: {e.body if hasattr(e, 'body') else 'N/A'}")
                    print(f"  Error type: {type(e)}")
                    if attempt == 1:
                        break
                    else:
                        continue
                        
                except anthropic.APIStatusError as e:
                    print(f"  API Status Error {e.status_code}")
                    print(f"  Message: {e.message}")
                    print(f"  Body: {e.body if hasattr(e, 'body') else 'N/A'}")
                    if attempt == 1:
                        break
                    
                except Exception as e:
                    print(f"  Unexpected error type: {type(e).__name__}")
                    print(f"  Full error: {str(e)}")
                    if attempt == 1:
                        break

    except Exception as e:
        print(f"  ❌ Claude initialization failed: {str(e)}")
        return None
    
    return None
        
        
def _extract_json(text: str) -> dict:
    """Extract JSON from text, handling markdown, trailing text, etc."""
    import json
    import re
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try direct JSON parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object {...}
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as e:
            print(f"  JSON parse error at char {e.pos}: {e.msg}")
            print(f"  Context: ...{text[max(0, e.pos-50):e.pos+50]}...")
            pass
    
    # Try to find and fix common JSON issues
    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  Still invalid JSON: {e}")
        return None

def _try_openai(img_base64: str, mime_type: str, prompt: str):
    """Try OpenAI GPT-4V with image support"""
    import os
    import json
    import time

    try:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  OPENAI_API_KEY not set")
            return None

        import openai
        client = openai.OpenAI(api_key=api_key)

        # Determine media type
        media_type_map = {
            'image/jpeg': 'image/jpeg',
            'image/png': 'image/png',
            'image/webp': 'image/webp',
            'image/gif': 'image/gif'
        }
        media_type = media_type_map.get(mime_type, 'image/jpeg')

        for attempt in range(2):
            try:
                print(f"  Trying OpenAI (attempt {attempt + 1}/2)...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=2000,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{img_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ]
                )
                
                raw = response.choices[0].message.content.strip()
                if raw.startswith('```'):
                    raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
                return json.loads(raw.strip())

            except openai.RateLimitError as e:
                if attempt == 0:
                    print(f"  Rate limited, waiting 30s...")
                    time.sleep(30)
                else:
                    raise

    except Exception as e:
        print(f"  ❌ OpenAI failed: {str(e)[:100]}")
        return None
        
        
def save_classify_cache():
    try:
        with open(CLASSIFY_CACHE_FILE, 'w') as f:
            json.dump(_classify_cache, f, indent=2)
    except Exception as e:
        log.warning(f"Could not save classify cache: {e}")
        
        
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """
    Step 2: Identify object + infer rigging using Gemini.
    Receives the SEGMENTED image — clean subject, transparent background.

    Accepts:
      raw image bytes (segmented PNG from /segment)
      optional ?tag=flying+bird

    Returns JSON:
      {
        "object_type":        "glass bird figurine",
        "base_object":        "bird",
        "descriptor":         "flying",
        "fantasy_elements":   "",
        "suggested_joints":   8,
        "joint_hints":        ["root", "spine", ...],
        "movement_type":      "flaps wings and glides",
        "needs_augmentation": true,
        "augment_prompt":     "Extend wings outward as if in mid-flight...",
        "notes":              "wings currently folded"
      }

    App Inventor:
      if needs_augmentation = true
        POST /augment_image?prompt=augment_prompt
        show user original vs augmented, let them choose
      else
        POST /reconstruct directly
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200

    try:
        tag = request.args.get('tag', '').strip()
        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No data received'}), 400
            
        hash_input = img_bytes + tag.encode() # csb not yet + joints_str.encode()
        img_hash   = hashlib.md5(hash_input).hexdigest()[:8]

        # Check if we already have a result for this image
        existing = next(
            (cid for cid, data in _classify_cache.items()
             if data.get('img_hash') == img_hash),
            None
        )
        if existing:
            log.info(f"Cache hit for image hash {img_hash}, reusing {existing}")
            return jsonify({**_classify_cache[existing], 'classify_id': existing})


        user_tag      = request.args.get('tag', '').strip()
        
        requested_joints = request.args.get('joints', '').strip()  # ← add this
        mime_type     = 'image/png' if img_bytes[:4] == b'\x89PNG' else 'image/jpeg'

        info = classify_with_vision(img_bytes, mime_type, user_tag or None,
                                    requested_joints=requested_joints or None)

        import uuid
        classify_id = str(uuid.uuid4())[:8]
        _classify_cache[classify_id] = info

        info['img_hash'] = img_hash
        classify_id = img_hash  # use hash as the ID for natural deduplication
        _classify_cache[classify_id] = info
        save_classify_cache()

        return jsonify({**info, 'classify_id': classify_id})  # ← add classify_id to response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/classify error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/augment_image', methods=['GET', 'POST'])
def augment_image():
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200

    try:
        prompt = request.args.get('tag', '').lower().strip().replace("+", " ")
        if not prompt:
            return jsonify({'error': 'Missing tag'}), 400

        # For humanoids, append pose hint to the user's augment description
        
        tag_words = set(tag.lower().replace('+', ' ').split())

        if tag_words & HUMANOID_KEYWORDS:
            prompt = prompt + " Show the character in a T-pose or A-pose with arms extended outward for easy 3D rigging. Clear background"
            log.info(f"Humanoid tag '{tag}' — appending pose hint to prompt")

        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No image data'}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = resize_if_needed(img, max_size=1024)
        resultA, resultB = edit_image_fal(img, prompt)

        import uuid
        uid    = str(uuid.uuid4())[:8]
        path_a = os.path.join(RESULTS_DIR, f"aug_{uid}_a.png")
        path_b = os.path.join(RESULTS_DIR, f"aug_{uid}_b.png")
        resultA.save(path_a)
        resultB.save(path_b)

        host = request.host
        return jsonify({
            'status':  'ok',
            'image_a': f"http://{host}/results/aug_{uid}_a.png",
            'image_b': f"http://{host}/results/aug_{uid}_b.png",
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/augment_image error: {e}")
        return jsonify({'error': str(e)}), 500

 

@app.route('/reconstruct', methods=['GET', 'POST'])
def reconstruct():
    """
    Step 4: Generate 3D model from segmented (or augmented) image using Meshy.

    Accepts:
      raw image bytes
      optional ?type=bird  (sets Meshy pose_mode)

    Returns JSON:
      { "status": "ok", "task_id": "...", "glb_url": "...", "obj_url": "..." }

    Requirements:
      export MESHY_API_KEY=msy_your_key
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200

    try:
        meshy_key = os.environ.get('MESHY_API_KEY')
        if not meshy_key:
            return jsonify({'error': 'MESHY_API_KEY not set'}), 500
        log.info(f"Reconstruct:get params and image")
        
        img_bytes = get_image_bytes()
        log.info(f"reconstruct: got {len(img_bytes)} bytes via "
                 f"{'stream' if not request.files else 'multipart'}")
        if not img_bytes:
            return jsonify({'error': 'No image data'}), 400


        object_type = request.args.get('type', '').lower().strip()
        object_type = object_type.replace("+", " ")
        img         = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img         = resize_if_needed(img, max_size=1024)
        log.info(f"Reconstruct: {img.size}, type={object_type}")

        data_uri  = f"data:image/png;base64,{img_to_b64(img)}"
        headers   = {"Authorization": f"Bearer {meshy_key}"}
        pose_mode = ("t-pose" if object_type in ['human', 'person', 'humanoid']
                     else "a-pose" if object_type in ['bird', 'dog', 'cat', 'horse', 'crab', 'fish']
                     else "")

        payload = {
            "image_url":      data_uri,
            "ai_model":       "meshy-6",
            "should_texture": True,
            "should_remesh":  False,
            "symmetry_mode":  "auto",
        }
        if pose_mode:
            payload["pose_mode"] = pose_mode
            log.info(f"pose_mode: {pose_mode}")

        resp = requests.post(
            "https://api.meshy.ai/openapi/v1/image-to-3d",
            headers=headers, json=payload
        )
        resp.raise_for_status()
        task_id = resp.json()["result"]
        log.info(f"Meshy task: {task_id}")

        elapsed = 0
        while elapsed < 300:
            time.sleep(5)
            elapsed   += 5
            task_resp  = requests.get(
                f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
                headers=headers
            )
            task_resp.raise_for_status()
            task     = task_resp.json()
            status   = task["status"]
            progress = task.get("progress", 0)
            log.info(f"Meshy {task_id}: {status} ({progress}%)")

            if status == "SUCCEEDED":
                return jsonify({
                    'status':  'ok',
                    'task_id': task_id,
                    'glb_url': task["model_urls"]["glb"],
                    'obj_url': task["model_urls"].get("obj"),
                    'fbx_url': task["model_urls"].get("fbx"),
                })
            elif status == "FAILED":
                err = task.get("task_error", {}).get("message", "Unknown")
                return jsonify({'error': f"Meshy failed: {err}"}), 500

        return jsonify({'error': 'Meshy timed out after 5 minutes'}), 504

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/reconstruct error: {e}")
        return jsonify({'error': str(e)}), 500
        
# ── Rig pipeline helpers ──────────────────────────────────────────────────────
 
def meshy_reconstruct(img: Image.Image, object_type: str) -> str:
    """
    Submit image to Meshy, poll until done, return GLB URL.
    Raises RuntimeError on failure or timeout.
    """
    meshy_key = os.environ.get('MESHY_API_KEY')
    if not meshy_key:
        raise RuntimeError("MESHY_API_KEY not set")
 
    headers   = {"Authorization": f"Bearer {meshy_key}"}
    pose_mode = (
        "t-pose" if object_type in ['human', 'person', 'humanoid']
        else "a-pose" if any(w in object_type for w in
                             ['bird', 'dog', 'cat', 'horse', 'crab',
                              'fish', 'animal', 'creature'])
        else ""
    )
 
    payload = {
        "image_url":      f"data:image/png;base64,{img_to_b64(img)}",
        "ai_model":       "meshy-6",
        "should_texture": True,
        "should_remesh":  False,
        "symmetry_mode":  "auto",
    }
    if pose_mode:
        payload["pose_mode"] = pose_mode
 
    log.info("Submitting Meshy task...")
    resp = requests.post(
        "https://api.meshy.ai/openapi/v1/image-to-3d",
        headers=headers, json=payload
    )
    resp.raise_for_status()
    task_id = resp.json()["result"]
    log.info(f"Meshy task: {task_id}")
 
    elapsed = 0
    while elapsed < 300:
        time.sleep(5)
        elapsed  += 5
        poll      = requests.get(
            f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}",
            headers=headers
        )
        poll.raise_for_status()
        task     = poll.json()
        status   = task["status"]
        progress = task.get("progress", 0)
        log.info(f"Meshy {task_id}: {status} ({progress}%)")
 
        if status == "SUCCEEDED":
                model_urls = task["model_urls"]
                glb_url    = model_urls.get("glb")
                usdz_url   = model_urls.get("usdz")
                log.info(f"GLB: {glb_url}")
                log.info(f"USDZ: {usdz_url}")
                return task_id, glb_url, usdz_url

        elif status == "FAILED":
            err = task.get("task_error", {}).get("message", "Unknown")
            raise RuntimeError(f"Meshy failed: {err}")
 
    raise RuntimeError("Meshy timed out after 5 minutes")
 
 
def download_glb(glb_url: str, dest_path: str):
    """Download GLB from URL to local path."""
    resp = requests.get(glb_url, verify=False)
    resp.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(resp.content)
    log.info(f"GLB saved: {dest_path}")
 
 
def run_skeleton_inference(glb_path: str, rigged_path: str,
                           n_joints: str = None) -> str:
    rig_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rig.py')
    if not os.path.exists(rig_script):
        raise RuntimeError(f"rig.py not found at {rig_script}")

    cmd = [
        sys.executable, rig_script,
        '--input',  os.path.abspath(glb_path),
        '--output', os.path.abspath(rigged_path),
        '--viz-only',
    ]
    if n_joints:
        cmd += ['--joints', str(n_joints)]

    log.info("Running rig.py skeleton inference...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"rig.py failed: {result.stderr[:200]}")

    # JSON is saved next to --output, not --input
    json_path = rigged_path.replace(".glb","_skeleton.json")
    if not os.path.exists(json_path):
        raise RuntimeError(f"Skeleton JSON not created: {json_path}")
    return json_path


def run_blender_rig(glb_path: str, json_path: str, rigged_path: str):
    with _blender_lock:  # only one Blender at a time
        rig_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'rig.py'
        )
        
        blender_bin = os.environ.get(
            'BLENDER_PATH',
            '/Applications/Blender.app/Contents/MacOS/blender'
        )
        cmd = [
            blender_bin, '--background', '--python', rig_script,
            '--',
            '--from-json', os.path.abspath(json_path),
            '--input',     os.path.abspath(glb_path),
            '--output',    os.path.abspath(rigged_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        log.info(result.stdout[-2000:])
        log.error(f"Blender stderr: {result.stderr[-1000:]}")  # ← add this
        if result.returncode != 0:
            raise RuntimeError(f"Blender failed: {result.stderr[-200:]}")
        if not os.path.exists(rigged_path):
            raise RuntimeError(f"Rigged GLB not created. stdout: {result.stdout[-500:]}")


def snap_to_nearest_vertex(world_pos, verts, radius_fraction=0.15):
    """Snap a position to the centroid of nearby mesh vertices."""
    mesh_size = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    radius    = mesh_size * radius_fraction
    dists     = np.linalg.norm(verts - world_pos, axis=1)
    nearby    = verts[dists < radius]
    if len(nearby) >= 5:
        return nearby.mean(axis=0)
    # No nearby vertices — return nearest single vertex
    return verts[np.argmin(dists)]
    
def snap_shoulder_to_elbow_attachment(world_pos, verts, elbow_direction='left'):
    """Find the outermost edge of mesh where elbow branches."""
    import numpy as np
    
    # Get vertices at similar Y height (shoulder height)
    height_tolerance = 0.15  # ±15% of mesh height
    mesh_size = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    height_band = verts[np.abs(verts[:, 1] - world_pos[1]) < (mesh_size * height_tolerance)]
    
    if len(height_band) < 20:
        height_band = verts  # fallback to all verts
    
    # Split by left/right and pick the EXTREME edge
    verts_x_center = height_band[:, 0].mean()
    
    if elbow_direction == 'left':
        # LEFT: pick the leftmost (minimum x)
        attachment = height_band[np.argmin(height_band[:, 0])]
        log.info(f"Left shoulder snapped to x={attachment[0]:.3f} (leftmost)")
    else:
        # RIGHT: pick the rightmost (maximum x)
        attachment = height_band[np.argmax(height_band[:, 0])]
        log.info(f"Right shoulder snapped to x={attachment[0]:.3f} (rightmost)")
    
    return attachment

def find_extremity_vertex(world_pos, verts, body_part, name):
    """Find vertex for terminal joint, respecting left/right for feet."""

    mesh_size = np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
    body_part = (body_part or '').lower()
    is_left = 'left' in name.lower()

    if body_part in ('leg', 'foot'):
        # Split verts into left/right halves by x-coordinate
        verts_center = verts[:, 0].mean()
        if is_left:
            candidates = verts[verts[:, 0] < verts_center]
        else:
            candidates = verts[verts[:, 0] > verts_center]
        
        if len(candidates) < 5:
            candidates = verts
        
        # Find bottommost in the half
        return candidates[np.argmin(candidates[:, 1])]

    elif body_part in ('elbow', 'hand', 'wing'):
        if is_left:
            return verts[np.argmin(verts[:, 0])]  # leftmost
        else:
            return verts[np.argmax(verts[:, 0])]  # rightmost

    elif body_part == 'head':
        return verts[np.argmax(verts[:, 1])]  # topmost

    else:
        dists = np.linalg.norm(verts - world_pos, axis=1)
        return verts[np.argmin(dists)]

def joints_from_model(classify_data, glb_path):
    import trimesh
    import numpy as np

    if not classify_data:
        return None, None, None

    hint_objects = classify_data.get('joint_hints', [])
    if not hint_objects or isinstance(hint_objects[0], str):
        return None, None, hint_objects

    mesh = trimesh.load(glb_path, force='mesh')
    verts = np.array(mesh.vertices)
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    brange = bmax - bmin

    name_to_idx = {h['name']: i for i, h in enumerate(hint_objects)}

    # ✅ SIMPLE: Just use position_normalized as-is, no offset math
    joints = []
    for hint in hint_objects:
        p = hint.get('position_normalized', {})
        norm_pos = np.array([p.get('x', 0.5), p.get('y', 0.5), p.get('z', 0.5)])
        
        world = np.array([
            bmin[0] + norm_pos[0] * brange[0],
            bmin[1] + norm_pos[1] * brange[1],
            bmin[2] + norm_pos[2] * brange[2],
        ])
        
        joints.append(world)

    # Build hierarchy
    hierarchy = []
    skeleton = classify_data.get('skeleton', [])
    for bone in skeleton:
        parent_name = bone.get('parent')
        child_name = bone.get('child')
        if parent_name in name_to_idx and child_name in name_to_idx:
            hierarchy.append((name_to_idx[parent_name], name_to_idx[child_name]))

    joints = [tuple(j) for j in joints]
    return joints, hierarchy, hint_objects
    
def _decimate_mesh(input_path: str, output_path: str, ratio: float = 0.1):
    import tempfile
    import textwrap

    blender_bin = os.environ.get(
        'BLENDER_PATH',
        '/Applications/Blender.app/Contents/MacOS/blender'
    )

    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, '/tmp/blender_packages')
        import numpy as np
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

    script_file = tempfile.mktemp(suffix='.py')
    with open(script_file, 'w') as f:
        f.write(script)

    result = subprocess.run(
        [blender_bin, '--background', '--python', script_file],
        capture_output=True, text=True, timeout=120
    )
    os.unlink(script_file)

    log.info(f"Blender decimate stdout: {result.stdout[-500:]}")
    if not os.path.exists(output_path):
        raise RuntimeError(f"Decimation failed: {result.stderr[-300:]}")
    log.info(f"Decimation complete: {output_path}")

def run_vehicle_pipeline(glb_path: str, classify_data: dict, uid: str, host: str) -> str:

    blender_bin = os.environ.get('BLENDER_PATH', '/Applications/Blender.app/Contents/MacOS/blender')
    seg_dir     = os.path.dirname(os.path.abspath(__file__))

    separated_path  = os.path.join(RESULTS_DIR, f"{uid}_separated.glb")
    animated_path   = os.path.join(RESULTS_DIR, f"{uid}_animated.glb")
    rigged_path     = os.path.join(RESULTS_DIR, f"{uid}_rigged.glb")
    classify_json   = os.path.join(RESULTS_DIR, f"{uid}_classify.json")
    tire_verts_path = os.path.join(RESULTS_DIR, f"{uid}_tire_verts.json")
    texture_path    = os.path.join(RESULTS_DIR, f"{uid}_texture.png")

    # Save classify data
    with open(classify_json, 'w') as f:
        json.dump(classify_data, f, indent=2)

    # Step 1 — Extract texture from GLB
    log.info("Extracting texture from GLB...")
    import struct
    with open(glb_path, 'rb') as f:
        f.read(12)
        json_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)
        j       = json.loads(f.read(json_len))
        bin_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)
        binary  = f.read(bin_len)

    for img_data in j.get('images', []):
        bv   = j['bufferViews'][img_data['bufferView']]
        data = binary[bv['byteOffset']:bv['byteOffset']+bv['byteLength']]
        with open(texture_path, 'wb') as tf:
            tf.write(data)
        log.info(f"Texture extracted: {texture_path}")
        break

    # Step 2 — Find tire vertices by texture color + Gemini wheel colors
    log.info("Finding tire vertices...")
    find_script = os.path.join(seg_dir, 'find_tire_verts.py')
    result = subprocess.run([
        sys.executable, find_script,
        glb_path, classify_json, tire_verts_path, texture_path
    ], capture_output=True, text=True, timeout=60)
    log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"find_tire_verts failed: {result.stderr[-200:]}")

    # Step 3 — Separate wheels
    log.info("Separating wheels...")
    classify_script = os.path.join(seg_dir, 'classify_wheels.py')
    result = subprocess.run([
        blender_bin, '--background', '--factory-startup',
        '--python', classify_script, '--',
        glb_path, separated_path, classify_json, tire_verts_path
    ], capture_output=True, text=True, timeout=120)
    log.info(result.stdout[-500:])
    if not os.path.exists(separated_path):
        raise RuntimeError(f"classify_wheels failed: {result.stderr[-200:]}")

    # Step 4 — Animate wheels
    log.info("Animating wheels...")
    animate_script = os.path.join(seg_dir, 'animatesam.py')
    result = subprocess.run([
        blender_bin, '--background', '--factory-startup',
        '--python', animate_script, '--',
        separated_path, animated_path, classify_json
    ], capture_output=True, text=True, timeout=120)
    log.info(result.stdout[-500:])
    if not os.path.exists(animated_path):
        raise RuntimeError(f"animatesam failed: {result.stderr[-200:]}")

    # Step 5 — Merge into single drive animation
    log.info("Merging animations...")
    merge_script = os.path.join(seg_dir, 'merge_animations.py')
    result = subprocess.run([
        sys.executable, merge_script,
        animated_path, rigged_path
    ], capture_output=True, text=True, timeout=60)
    log.info(result.stdout)
    if not os.path.exists(rigged_path):
        raise RuntimeError(f"merge_animations failed: {result.stderr[-200:]}")

    log.info(f"Vehicle pipeline complete: {rigged_path}")
    return rigged_path

def run_rig_pipeline(task_id, img, object_type, n_joints,
                     classify_data, rig_hash, mesh_hash, host):
    try:
        _rig_tasks[task_id] = {'status': 'meshy', 'progress': 0}

        # ── Check mesh cache ──────────────────────────────────────
        glb_path = None
        glb_url  = None
        usdz_url = None

        if mesh_hash and mesh_hash in _mesh_cache:
            cached_glb = _mesh_cache[mesh_hash]['glb_path']
            if os.path.exists(cached_glb):
                log.info(f"Mesh cache hit: {mesh_hash}, skipping Meshy")
                glb_path = cached_glb
                glb_url  = _mesh_cache[mesh_hash]['glb_url']
                usdz_url = _mesh_cache[mesh_hash].get('usdz_url')  # may be None
            else:
                log.warning(f"Mesh cache points to missing file, re-running Meshy")
                del _mesh_cache[mesh_hash]
                save_mesh_cache()

        if glb_path is None:
            meshy_task_id, glb_url, usdz_url = meshy_reconstruct(img, object_type)
            uid      = meshy_task_id[:8]
            glb_path = os.path.join(RESULTS_DIR, f"{uid}_mesh.glb")
            download_glb(glb_url, glb_path)

            if usdz_url:
                usdz_path = os.path.join(RESULTS_DIR, f"{uid}_mesh.usdz")
                download_glb(usdz_url, usdz_path)
                log.info(f"USDZ saved: {usdz_path}")

            _mesh_cache[mesh_hash] = {
                'glb_path': glb_path,
                'glb_url':  glb_url,
                'usdz_url': usdz_url,
            }
            save_mesh_cache()
            log.info(f"Mesh cached: {mesh_hash}")

        uid = os.path.basename(glb_path).split('_')[0]

        # ── Branch by category ────────────────────────────────────────
        category = (classify_data or {}).get('category', '')

        # Trust Gemini's category first, fall back to object_type keywords
        VEHICLE_KEYWORDS = {'car', 'truck', 'vehicle', 'bus', 'bike', 'motorcycle', 'van', 'auto'}
        tag_words        = set(object_type.lower().split())

        if category == 'vehicle':
            is_vehicle = True
        elif category in ('animal', 'humanoid', 'other'):
            is_vehicle = False  # Gemini explicitly said not a vehicle
        else:
            # No category from Gemini — fall back to object_type keywords
            is_vehicle = bool(tag_words & VEHICLE_KEYWORDS)

        log.info(f"Category: '{category}' object_type: '{object_type}' is_vehicle: {is_vehicle}")

        if is_vehicle:
            log.info("Vehicle detected — using vehicle pipeline")
            _rig_tasks[task_id] = {'status': 'rigging', 'progress': 60}
            rigged_path = run_vehicle_pipeline(glb_path, classify_data or {}, uid, host)
        else:
            log.info("Non-vehicle — using character rigging pipeline")
            # Decimate mesh
            decimated_path = glb_path.replace('_mesh.glb', '_decimated.glb')
            _decimate_mesh(glb_path, decimated_path, ratio=0.1)
            glb_path = decimated_path
            log.info("Non-vehicle — decimated glb")
            usdz_path = _mesh_cache.get(mesh_hash, {}).get('usdz_path')
            log.info(f"usdz path {usdz_path}")
            if usdz_path and os.path.exists(usdz_path):
                decimated_usdz = usdz_path.replace('_mesh.usdz', '_decimated.usdz')
                try:
                    _decimate_usdz(usdz_path, decimated_usdz)
                    log.info(f"USDZ decimated: {decimated_usdz}")
                except Exception as e:
                    log.warning(f"USDZ decimation failed (non-fatal): {e}")
                    decimated_usdz = usdz_path  # use original

            rigged_path = os.path.join(RESULTS_DIR, f"{uid}_rigged.glb")
            _rig_tasks[task_id] = {'status': 'rigging', 'progress': 70}
            json_path = run_skeleton_inference(glb_path, rigged_path, n_joints)

            if classify_data:
                joints, _, hint_objects = joints_from_model(classify_data, glb_path)

                if joints:
                    log.info(f"Using model joint positions for {len(joints)} joints")
                    name_to_idx = {h['name']: i for i, h in enumerate(hint_objects)}
                    hierarchy   = []

                    for bone in classify_data.get('skeleton', []):
                        p = name_to_idx.get(bone.get('parent'))
                        c = name_to_idx.get(bone.get('child'))
                        if p is not None and c is not None:
                            hierarchy.append((p, c))

                    if not hierarchy:
                        with open(json_path) as f:
                            existing_skel = json.load(f)
                        hierarchy = [(b['parent'], b['child'])
                                     for b in existing_skel['bones']]
                        log.info("Using skeleton JSON hierarchy as fallback")

                    skel = {
                        'joints': [
                            {
                                'id':       i,
                                'name':     hint_objects[i].get('name', f'joint_{i}'),
                                'position': list(joints[i]),
                                'hint':     hint_objects[i],
                            }
                            for i in range(len(joints))
                        ],
                        'bones': [
                            {
                                'parent': p,
                                'child':  c,
                                'name':   f"{hint_objects[p]['name']}_to_{hint_objects[c]['name']}"
                            }
                            for p, c in hierarchy
                            if p < len(hint_objects) and c < len(hint_objects)
                        ]
                    }
                else:
                    log.info("No positions from model, patching names into geometric skeleton")
                    with open(json_path) as f:
                        skel = json.load(f)
                    for i, joint in enumerate(skel['joints']):
                        if i < len(hint_objects):
                            h = hint_objects[i]
                            joint['name'] = h.get('name', joint['name']) if isinstance(h, dict) else h
                            joint['hint'] = h if isinstance(h, dict) else None
            else:
                with open(json_path) as f:
                    skel = json.load(f)

            with open(json_path, 'w') as f:
                json.dump(skel, f, indent=2)

            try:
                from rig import visualize_skeleton
                corrected_joints    = [tuple(j['position']) for j in skel['joints']]
                corrected_hierarchy = [(b['parent'], b['child']) for b in skel['bones']]
                corrected_names     = [j['name'] for j in skel['joints']]
                corrected_hints     = [j.get('hint') for j in skel['joints']]
                viz_path            = rigged_path.replace('_rigged.glb', '_gemini_viz.glb')
                visualize_skeleton(
                    glb_path, corrected_joints, corrected_hierarchy, viz_path,
                    labels=corrected_names, labels_raw=corrected_hints
                )
                log.info(f"Gemini skeleton viz: {viz_path}")
            except Exception as e:
                log.warning(f"Viz regeneration failed (non-fatal): {e}")

            run_blender_rig(glb_path, json_path, rigged_path)

        # ── Done ──────────────────────────────────────────────────
        rigged_url = f"http://{host}/results/{os.path.basename(rigged_path)}"
        _rig_tasks[task_id] = {
            'status':     'ok',
            'progress':   100,
            'rigged_url': rigged_url,
            'glb_url':    glb_url,
        }

    except Exception as e:
        log.error(f"Rig task {task_id} failed: {e}")
        _rig_tasks[task_id] = {'status': 'error', 'error': str(e)}
        
        
@app.route('/decimate', methods=['GET', 'POST'])
def decimate():
    """
    Decimate a GLB mesh to reduce polygon count.
    
    Accepts:
      raw GLB bytes OR ?glb_url=... to fetch from URL
      optional ?ratio=0.1  (decimation ratio, default 0.1 = 10% of original)
    
    Returns:
      { "status": "ok", "url": "http://server/results/xxx_decimated.glb" }
    """
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200

    try:
        ratio   = float(request.args.get('ratio', '0.1'))
        ratio   = max(0.01, min(1.0, ratio))  # clamp 1%-100%
        glb_url = request.args.get('glb_url', '').strip()

        if glb_url:
            # Fetch GLB from URL
            log.info(f"Fetching GLB from: {glb_url}")
            resp     = requests.get(glb_url, verify=False, timeout=60)
            resp.raise_for_status()
            glb_bytes = resp.content
        else:
            glb_bytes = request.stream.read()

        if not glb_bytes:
            return jsonify({'error': 'No GLB data'}), 400

        # Save input
        import uuid
        uid      = str(uuid.uuid4())[:8]
        in_path  = os.path.join(RESULTS_DIR, f"{uid}_input.glb")
        out_path = os.path.join(RESULTS_DIR, f"{uid}_decimated.glb")

        with open(in_path, 'wb') as f:
            f.write(glb_bytes)

        log.info(f"Decimating {in_path} ratio={ratio}")
        _decimate_mesh(in_path, out_path, ratio=ratio)

        # Clean up input
        os.unlink(in_path)

        host = request.host
        return jsonify({
            'status': 'ok',
            'url':    f"http://{host}/results/{uid}_decimated.glb",
            'ratio':  ratio,
        })

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/decimate error: {e}")
        return jsonify({'error': str(e)}), 500
        
@app.route('/rig', methods=['GET', 'POST'])
def rig():
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        img_bytes = request.stream.read()
        if not img_bytes:
            return jsonify({'error': 'No image data'}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img = resize_if_needed(img, max_size=1024)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_bytes = buf.getvalue()

        object_type  = request.args.get('type', '').lower().strip().replace("+", " ")
        object_type +="in a t-pose or a-pose for easy rigging"
        n_joints     = request.args.get('joints', None)
        # Now accepts full classify JSON instead of flat comma string

        classify_id   = request.args.get('classify_id', '')
        classify_data = _classify_cache.get(classify_id) if classify_id else None
        if classify_id and classify_data is None:
            log.warning(f"classify_id '{classify_id}' not found in cache — rigging without hints")

        img = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
        img = resize_if_needed(img, max_size=1024)

        import uuid
        task_id = str(uuid.uuid4())[:8]
        _rig_tasks[task_id] = {'status': 'started', 'progress': 0}

        mesh_hash = hashlib.md5(
            img_bytes +
            object_type.encode()
        ).hexdigest()[:12]

        log.info(f"mesh_hash computed: {mesh_hash}")
        log.info(f"mesh_cache keys: {list(_mesh_cache.keys())}")
        log.info(f"mesh_hash in cache: {mesh_hash in _mesh_cache}")

        rig_hash = hashlib.md5(
            img_bytes +
            object_type.encode() +
            (n_joints or '').encode() +
            (classify_id or '').encode()
        ).hexdigest()[:12]

        thread = threading.Thread(
            target=run_rig_pipeline,
            args=(task_id, img, object_type, n_joints,
                  classify_data, rig_hash, mesh_hash, request.host),
            daemon=True
        
        )
        thread.start()
        return jsonify({'status': 'processing', 'task_id': task_id})

    except Exception as e:
        log.error(f"/rig error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/rig/status/<task_id>', methods=['GET'])
def rig_status(task_id: str):
    """Poll this until status = 'ok'."""
    task = _rig_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task)
    
    
@app.route('/results/<filename>')
def serve_result(filename):
    """Serve rigged GLB files for ModelNode."""
    return send_file(os.path.join(RESULTS_DIR, filename))
 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000, debug=False)


