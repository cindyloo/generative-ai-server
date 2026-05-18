"""
pipeline_store.py

Single source of truth for the 3D rigging pipeline.
Replaces model_store.py, _classify_cache, and _mesh_cache entirely.

Pipeline stages and the store fields they populate
---------------------------------------------------
  /segment          (stateless — client holds bytes)
  /classify         → classify{}, active_image_path = segmented_image_path
  /augment_image    → saves augmented PNGs to disk
  /augment_image/confirm → active_image_path = chosen augmented PNG
  /joints           → joints{}          ← repeatable with force=true
  /mesh             → mesh{}            ← cached, never re-runs without force=true
  /rig              → rig{}             ← repeatable with force=true

Schema (one record per classify_id)
------------------------------------
{
  "classify_id":      "a1b2c3d4",
  "tag":              "dragon",
  "tags":             ["animal", "dragon"],
  "pipeline_version": 2,

  "active_image_path": "results/a1b2c3d4_segmented.png",
  // set to segmented_image_path on /classify
  // overwritten to augmented path on /augment_image/confirm
  // /joints, /mesh, /rig all read this field — never the segmented path directly

  "classify": {
    "object_type":        "fire-breathing dragon",
    "category":           "animal",
    "needs_augmentation": false,
    "augment_prompt":     "",
    "segmented_image_path": "results/a1b2c3d4_segmented.png",
    "created_at":         "2025-01-01T00:00:00Z"
  },

  "joints": {                             // null until /joints completes
    "source_image_path":  "results/a1b2c3d4_segmented.png",
    "joint_hints":        [...],
    "skeleton":           [...],
    "suggested_joints":   8,
    "model_used":         "gemini-2.5-flash",
    "created_at":         "2025-01-01T00:00:05Z"
  },

  "mesh": {                               // null until /mesh completes
    "mesh_hash":          "abc123456789",
    "meshy_task_id":      "task-xyz",
    "glb_path":           "results/a1b2c3d4_mesh.glb",
    "glb_url":            "https://cdn.meshy.ai/...",
    "usdz_path":          "results/a1b2c3d4_mesh.usdz",
    "usdz_url":           "https://cdn.meshy.ai/...",
    "decimated_glb_path": "results/a1b2c3d4_decimated.glb",
    "created_at":         "2025-01-01T00:00:10Z"
  },

  "rig": {                                // null until /rig completes
    "rigged_glb_path":    "results/a1b2c3d4_rigged.glb",
    "viz_glb_path":       "results/a1b2c3d4_viz.glb",
    "skeleton_json_path": "results/a1b2c3d4_skeleton.json",
    "status":             "ok",
    "error":              null,
    "user_id":            "fb712dd7-...",
    "created_at":         "2025-01-01T00:01:00Z"
  }
}

Backends
--------
  PIPELINE_STORE_BACKEND=json     (default) single JSON file, zero extra deps
  PIPELINE_STORE_BACKEND=tinydb   TinyDB, good for single-server deployments
  PIPELINE_STORE_BACKEND=clouddb  MIT App Inventor CloudDB / Redis

Public API
----------
  store = get_store()

  # reads
  record  = store.get(classify_id)
  records = store.get_all()
  records = store.get_by_user(user_id)
  records = store.search_by_tag(user_id, tag)
  mesh    = store.get_mesh_by_hash(mesh_hash)
  tags    = store.all_tags_for_user(user_id)

  # writes
  store.upsert_classify(classify_id, tag, classify_data)
  store.set_active_image(classify_id, image_path)
  store.upsert_joints(classify_id, joints_data)
  store.upsert_mesh(classify_id, mesh_data)
  store.upsert_rig(classify_id, rig_data)
  store.set_rig_status(classify_id, status, error=None)

  # URL helpers
  record_with_urls = store.with_urls(record, request.host)

Environment variables
---------------------
  PIPELINE_STORE_BACKEND   json | tinydb | clouddb
  RESULTS_DIR              directory for local files (default: 'results')
  CLOUDDB_URL, CLOUDDB_TOKEN, CLOUDDB_PROJECT
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

PIPELINE_VERSION = 2


# ── Tag extraction ────────────────────────────────────────────────────────────

_SKIP_BODY_PARTS = {
    'torso', 'spine', 'chest', 'pelvis', 'axle', 'body', 'root',
}

_STOP_WORDS = {
    'a', 'an', 'the', 'in', 'or', 'for', 'no', 'on', 'and',
    't-pose', 'a-pose', 'easy', 'rigging', 'statue', 'two', 'walks',
}


def extract_tags(classify_data: dict, user_tag: str = '') -> list[str]:
    """
    Build sorted search tokens from user tag + classify result.

    Sources:
      1. user_tag                      e.g. "bronze+dog"
      2. classify_data['object_type']  e.g. "bronze dog statue (bipedal)"
      3. classify_data['category']     e.g. "animal"
    """
    tags: set[str] = set()

    if user_tag:
        words = user_tag.lower().replace('+', ' ').split()
        tags.update(w for w in words if w not in _STOP_WORDS and len(w) > 2)

    if classify_data:
        object_type = classify_data.get('object_type', '')
        cleaned = object_type.lower().translate(str.maketrans('(),.-', '     '))
        words = cleaned.split()
        tags.update(w for w in words if w not in _STOP_WORDS and len(w) > 2)

        category = classify_data.get('category', '')
        if category:
            tags.add(category.lower())

    return sorted(t for t in tags if t)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _local_url(path: str | None, host: str) -> str | None:
    if not path:
        return None
    return f"http://{host}/results/{os.path.basename(path)}"


def _inject_urls(record: dict, host: str) -> dict:
    """
    Return a shallow copy of record with http:// URLs added for all local paths.
    CDN urls (glb_url, usdz_url) are preserved unchanged.
    """
    r      = dict(record)
    cls    = dict(r.get('classify') or {})
    joints = dict(r.get('joints')   or {})
    msh    = dict(r.get('mesh')     or {})
    rig    = dict(r.get('rig')      or {})

    r['active_image_url']          = _local_url(r.get('active_image_path'),          host)
    cls['segmented_url']           = _local_url(cls.get('segmented_image_path'),     host)
    joints['source_image_url']     = _local_url(joints.get('source_image_path'),     host)
    msh['glb_local_url']           = _local_url(msh.get('glb_path'),                 host)
    msh['usdz_local_url']          = _local_url(msh.get('usdz_path'),                host)
    msh['decimated_local_url']     = _local_url(msh.get('decimated_glb_path'),       host)
    rig['rigged_url']              = _local_url(rig.get('rigged_glb_path'),          host)
    rig['viz_url']                 = _local_url(rig.get('viz_glb_path'),             host)

    r['classify'] = cls    or None
    r['joints']   = joints or None
    r['mesh']     = msh    or None
    r['rig']      = rig    or None
    return r


def _blank_record(classify_id: str, tag: str = '') -> dict:
    return {
        'classify_id':       classify_id,
        'tag':               tag,
        'tags':              [],
        'pipeline_version':  PIPELINE_VERSION,
        'active_image_path': None,
        'classify':          None,
        'joints':            None,
        'mesh':              None,
        'rig':               None,
    }


# ── JSON backend ──────────────────────────────────────────────────────────────

class JsonStore:
    """
    In-process JSON file store. Thread-safe via RLock.
    Writes are atomic: .tmp then os.replace(). Zero external dependencies.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data = json.load(f)
                log.info(f"pipeline_store(json): loaded {len(self._data)} records")
            except Exception as e:
                log.warning(f"pipeline_store(json): could not load {self._path}: {e}")

    def _save(self):
        tmp = self._path + '.tmp'
        try:
            os.makedirs(os.path.dirname(self._path) or '.', exist_ok=True)
            with open(tmp, 'w') as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as e:
            log.error(f"pipeline_store(json): save failed: {e}")

    def _get_or_create(self, classify_id: str, tag: str = '') -> dict:
        if classify_id not in self._data:
            self._data[classify_id] = _blank_record(classify_id, tag)
        return self._data[classify_id]

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(self, classify_id: str) -> dict | None:
        with self._lock:
            r = self._data.get(classify_id)
            return dict(r) if r else None

    def get_all(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._data.values()]

    def get_by_user(self, user_id: str) -> list[dict]:
        with self._lock:
            records = [
                dict(r) for r in self._data.values()
                if (r.get('rig') or {}).get('user_id') == user_id
            ]
            return sorted(records,
                          key=lambda r: (r.get('rig') or {}).get('created_at', ''),
                          reverse=True)
    #not using user_id atm
    def search_by_tag(self, user_id: str, tag: str) -> list[dict]:
        tag_lower = tag.lower().strip()
        with self._lock:
            return [
                dict(r) for r in self._data.values()
                if tag_lower in r.get('tags', [])
                and (r.get('rig') or {}) #.get('user_id') == user_id
            ]

    def get_mesh_by_hash(self, mesh_hash: str) -> dict | None:
        """Return mesh sub-object if mesh_hash matches and GLB file exists on disk."""
        with self._lock:
            for r in self._data.values():
                msh = r.get('mesh') or {}
                if msh.get('mesh_hash') == mesh_hash:
                    glb = msh.get('glb_path')
                    if glb and os.path.exists(glb):
                        return dict(msh)
            return None

    def all_tags_for_user(self, user_id: str) -> list[str]:
        with self._lock:
            tags: set[str] = set()
            for r in self._data.values():
                if (r.get('rig') or {}).get('user_id') == user_id:
                    tags.update(r.get('tags', []))
            return sorted(tags)

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert_classify(self, classify_id: str, tag: str, classify_data: dict):
        """
        Store classify result and set active_image_path to the segmented image.
        active_image_path is the single path all downstream steps read from.
        """
        with self._lock:
            record = self._get_or_create(classify_id, tag)
            record['tag']  = tag
            record['tags'] = extract_tags(classify_data, tag)
            record['classify'] = {
                **classify_data,
                'created_at': classify_data.get('created_at') or _now(),
            }
            # Set active_image_path to segmented image on first classify.
            # Only overwrite if not already set to an augmented image —
            # a force re-classify should not lose a previously confirmed augment.
            if not record.get('active_image_path'):
                record['active_image_path'] = classify_data.get('segmented_image_path')
            self._save()

    def set_active_image(self, classify_id: str, image_path: str):
        """
        Called by /augment_image/confirm to point all downstream steps
        at the chosen augmented image instead of the segmented original.
        """
        with self._lock:
            record = self._get_or_create(classify_id)
            record['active_image_path'] = image_path
            self._save()

    def upsert_joints(self, classify_id: str, joints_data: dict):
        """
        Store joint placement result from /joints.
        Overwrites any previous joints result — /joints is freely repeatable.
        joints_data should include: joint_hints, skeleton, suggested_joints,
        source_image_path, model_used.
        """
        with self._lock:
            record = self._get_or_create(classify_id)
            record['joints'] = {
                **joints_data,
                'created_at': joints_data.get('created_at') or _now(),
            }
            self._save()

    def upsert_mesh(self, classify_id: str, mesh_data: dict):
        with self._lock:
            record = self._get_or_create(classify_id)
            record['mesh'] = {
                **mesh_data,
                'created_at': mesh_data.get('created_at') or _now(),
            }
            self._save()

    def upsert_rig(self, classify_id: str, rig_data: dict):
        with self._lock:
            record = self._get_or_create(classify_id)
            record['rig'] = {
                'status':     'ok',
                'error':      None,
                **rig_data,
                'created_at': rig_data.get('created_at') or _now(),
            }
            self._save()

    def set_rig_status(self, classify_id: str, status: str,
                       error: str | None = None):
        with self._lock:
            record = self._get_or_create(classify_id)
            if record.get('rig') is None:
                record['rig'] = {}
            record['rig']['status'] = status
            record['rig']['error']  = error
            self._save()

    def with_urls(self, record: dict, host: str) -> dict:
        return _inject_urls(record, host)


# ── TinyDB backend ────────────────────────────────────────────────────────────

class TinyDbStore:
    """
    TinyDB backend. Same schema as JsonStore.
    Good for single-server deployments without running a full database.
    """

    def __init__(self, path: str):
        from tinydb import TinyDB, Query
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._db    = TinyDB(path)
        self._table = self._db.table('pipeline')
        self._Q     = Query()
        self._lock  = threading.RLock()
        log.info(f"pipeline_store(tinydb): {len(self._table)} records in {path}")

    def _get_raw(self, classify_id: str) -> dict | None:
        return self._table.get(self._Q.classify_id == classify_id)

    def _upsert(self, classify_id: str, updates: dict):
        existing = self._get_raw(classify_id) or _blank_record(classify_id)
        existing.update(updates)
        self._table.upsert(existing, self._Q.classify_id == classify_id)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(self, classify_id: str) -> dict | None:
        with self._lock:
            r = self._get_raw(classify_id)
            return dict(r) if r else None

    def get_all(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._table.all()]

    def get_by_user(self, user_id: str) -> list[dict]:
        with self._lock:
            records = [
                dict(r) for r in self._table.all()
                if (r.get('rig') or {}).get('user_id') == user_id
            ]
            return sorted(records,
                          key=lambda r: (r.get('rig') or {}).get('created_at', ''),
                          reverse=True)

    def search_by_tag(self, user_id: str, tag: str) -> list[dict]:
        tag_lower = tag.lower().strip()
        with self._lock:
            return [
                dict(r) for r in self._table.all()
                if tag_lower in r.get('tags', [])
                and (r.get('rig') or {}).get('user_id') == user_id
            ]

    def get_mesh_by_hash(self, mesh_hash: str) -> dict | None:
        with self._lock:
            for r in self._table.all():
                msh = r.get('mesh') or {}
                if msh.get('mesh_hash') == mesh_hash:
                    glb = msh.get('glb_path')
                    if glb and os.path.exists(glb):
                        return dict(msh)
            return None

    def all_tags_for_user(self, user_id: str) -> list[str]:
        with self._lock:
            tags: set[str] = set()
            for r in self._table.all():
                if (r.get('rig') or {}).get('user_id') == user_id:
                    tags.update(r.get('tags', []))
            return sorted(tags)

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert_classify(self, classify_id: str, tag: str, classify_data: dict):
        with self._lock:
            classify_data.setdefault('created_at', _now())
            existing = self._get_raw(classify_id) or {}
            updates = {
                'tag':      tag,
                'tags':     extract_tags(classify_data, tag),
                'classify': classify_data,
            }
            # Only set active_image_path if not already pointing at an augmented image
            if not existing.get('active_image_path'):
                updates['active_image_path'] = classify_data.get('segmented_image_path')
            self._upsert(classify_id, updates)

    def set_active_image(self, classify_id: str, image_path: str):
        with self._lock:
            self._upsert(classify_id, {'active_image_path': image_path})

    def upsert_joints(self, classify_id: str, joints_data: dict):
        with self._lock:
            joints_data.setdefault('created_at', _now())
            self._upsert(classify_id, {'joints': joints_data})

    def upsert_mesh(self, classify_id: str, mesh_data: dict):
        with self._lock:
            mesh_data.setdefault('created_at', _now())
            self._upsert(classify_id, {'mesh': mesh_data})

    def upsert_rig(self, classify_id: str, rig_data: dict):
        with self._lock:
            rig_data.setdefault('status', 'ok')
            rig_data.setdefault('error',  None)
            rig_data.setdefault('created_at', _now())
            self._upsert(classify_id, {'rig': rig_data})

    def set_rig_status(self, classify_id: str, status: str,
                       error: str | None = None):
        with self._lock:
            existing = self._get_raw(classify_id) or {}
            rig = dict(existing.get('rig') or {})
            rig['status'] = status
            rig['error']  = error
            self._upsert(classify_id, {'rig': rig})

    def with_urls(self, record: dict, host: str) -> dict:
        return _inject_urls(record, host)


# ── CloudDB backend ───────────────────────────────────────────────────────────

class CloudDbStore:
    """
    MIT App Inventor CloudDB (Redis-backed key-value store).

    Key scheme
    ----------
      pipeline:{classify_id}   full pipeline record as JSON
      user_index:{user_id}     JSON list of classify_ids, newest first
      mesh_index:{mesh_hash}   classify_id string (O(1) mesh lookup)
    """

    def __init__(self, url: str, token: str, project: str):
        import requests as _requests
        self._requests = _requests
        self._url      = url.rstrip('/')
        self._token    = token
        self._project  = project
        self._lock     = threading.RLock()
        log.info(f"pipeline_store(clouddb): {self._url} project={self._project}")

    def _headers(self) -> dict:
        return {
            'Authorization': f'Bearer {self._token}',
            'Content-Type':  'application/json',
        }

    def _get(self, key: str):
        try:
            resp = self._requests.get(
                f"{self._url}/v1/{self._project}/{key}",
                headers=self._headers(), timeout=10,
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            raw   = resp.json()
            value = raw.get('value') or raw.get('result')
            return json.loads(value) if isinstance(value, str) else value
        except Exception as e:
            log.error(f"CloudDB GET {key}: {e}")
            return None

    def _set(self, key: str, value) -> bool:
        try:
            resp = self._requests.post(
                f"{self._url}/v1/{self._project}/{key}",
                headers=self._headers(), timeout=10,
                json={'value': json.dumps(value)},
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            log.error(f"CloudDB SET {key}: {e}")
            return False

    def _delete(self, key: str) -> bool:
        try:
            resp = self._requests.delete(
                f"{self._url}/v1/{self._project}/{key}",
                headers=self._headers(), timeout=10,
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            log.error(f"CloudDB DELETE {key}: {e}")
            return False

    def _get_user_index(self, user_id: str) -> list[str]:
        return self._get(f"user_index:{user_id}") or []

    def _prepend_user_index(self, user_id: str, classify_id: str):
        index = self._get_user_index(user_id)
        if classify_id not in index:
            index.insert(0, classify_id)
            self._set(f"user_index:{user_id}", index)

    def _load_or_blank(self, classify_id: str, tag: str = '') -> dict:
        return self.get(classify_id) or _blank_record(classify_id, tag)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(self, classify_id: str) -> dict | None:
        return self._get(f"pipeline:{classify_id}")

    def get_all(self) -> list[dict]:
        log.warning("CloudDB get_all(): not supported, returning []")
        return []

    def get_by_user(self, user_id: str) -> list[dict]:
        with self._lock:
            index   = self._get_user_index(user_id)
            records = [self.get(cid) for cid in index]
            return [r for r in records if r]

    def search_by_tag(self, user_id: str, tag: str) -> list[dict]:
        tag_lower = tag.lower().strip()
        return [r for r in self.get_by_user(user_id)
                if tag_lower in r.get('tags', [])]

    def get_mesh_by_hash(self, mesh_hash: str) -> dict | None:
        classify_id = self._get(f"mesh_index:{mesh_hash}")
        if not classify_id:
            return None
        record = self.get(classify_id)
        if not record:
            return None
        msh = record.get('mesh') or {}
        return dict(msh) if msh.get('mesh_hash') == mesh_hash else None

    def all_tags_for_user(self, user_id: str) -> list[str]:
        tags: set[str] = set()
        for r in self.get_by_user(user_id):
            tags.update(r.get('tags', []))
        return sorted(tags)

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert_classify(self, classify_id: str, tag: str, classify_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id, tag)
            record['tag']      = tag
            record['tags']     = extract_tags(classify_data, tag)
            record['classify'] = {
                **classify_data,
                'created_at': classify_data.get('created_at') or _now(),
            }
            if not record.get('active_image_path'):
                record['active_image_path'] = classify_data.get('segmented_image_path')
            self._set(f"pipeline:{classify_id}", record)

    def set_active_image(self, classify_id: str, image_path: str):
        with self._lock:
            record = self._load_or_blank(classify_id)
            record['active_image_path'] = image_path
            self._set(f"pipeline:{classify_id}", record)

    def upsert_joints(self, classify_id: str, joints_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id)
            record['joints'] = {
                **joints_data,
                'created_at': joints_data.get('created_at') or _now(),
            }
            self._set(f"pipeline:{classify_id}", record)

    def upsert_mesh(self, classify_id: str, mesh_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id)
            mesh_data.setdefault('created_at', _now())
            record['mesh'] = mesh_data
            self._set(f"pipeline:{classify_id}", record)
            if mesh_data.get('mesh_hash'):
                self._set(f"mesh_index:{mesh_data['mesh_hash']}", classify_id)

    def upsert_rig(self, classify_id: str, rig_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id)
            rig_data.setdefault('status', 'ok')
            rig_data.setdefault('error',  None)
            rig_data.setdefault('created_at', _now())
            record['rig'] = rig_data
            self._set(f"pipeline:{classify_id}", record)
            user_id = rig_data.get('user_id', '')
            if user_id:
                self._prepend_user_index(user_id, classify_id)

    def set_rig_status(self, classify_id: str, status: str,
                       error: str | None = None):
        with self._lock:
            record = self._load_or_blank(classify_id)
            rig = dict(record.get('rig') or {})
            rig['status'] = status
            rig['error']  = error
            record['rig'] = rig
            self._set(f"pipeline:{classify_id}", record)

    def with_urls(self, record: dict, host: str) -> dict:
        return _inject_urls(record, host)


# ── Singleton factory ─────────────────────────────────────────────────────────

_store_instance: JsonStore | TinyDbStore | CloudDbStore | None = None
_store_lock = threading.Lock()


def get_store() -> JsonStore | TinyDbStore | CloudDbStore:
    """
    Return the module-level singleton, creating it on first call.
    Backend selected by PIPELINE_STORE_BACKEND env var: json | tinydb | clouddb
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    with _store_lock:
        if _store_instance is not None:
            return _store_instance

        backend     = os.environ.get('PIPELINE_STORE_BACKEND', 'json').lower()
        results_dir = os.environ.get('RESULTS_DIR', 'results')
        os.makedirs(results_dir, exist_ok=True)

        if backend == 'clouddb':
            url     = os.environ.get('CLOUDDB_URL', '')
            token   = os.environ.get('CLOUDDB_TOKEN', '')
            project = os.environ.get('CLOUDDB_PROJECT', '')
            if not all([url, token, project]):
                raise ValueError(
                    "PIPELINE_STORE_BACKEND=clouddb requires "
                    "CLOUDDB_URL, CLOUDDB_TOKEN, and CLOUDDB_PROJECT to be set."
                )
            _store_instance = CloudDbStore(url, token, project)

        elif backend == 'tinydb':
            path = os.path.join(results_dir, '_pipeline_store.json')
            _store_instance = TinyDbStore(path)

        else:
            path = os.path.join(results_dir, '_pipeline_store.json')
            _store_instance = JsonStore(path)

        log.info(f"pipeline_store: using '{backend}' backend")
        return _store_instance


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile
    logging.basicConfig(level=logging.INFO)
    print("=== pipeline_store smoke test ===\n")

    with tempfile.TemporaryDirectory() as tmp:
        store = JsonStore(os.path.join(tmp, '_pipeline_store.json'))

        seg_path = f'{tmp}/a1b2c3d4_segmented.png'
        aug_path = f'{tmp}/a1b2c3d4_augmented_a.png'

        classify_data = {
            'object_type':        'bronze dog statue (bipedal, no arms)',
            'category':           'animal',
            'needs_augmentation': False,
            'augment_prompt':     '',
            'segmented_image_path': seg_path,
        }

        # ── classify sets active_image_path to segmented ──────────────────────
        store.upsert_classify('a1b2c3d4', 'bronze+dog', classify_data)
        r = store.get('a1b2c3d4')
        assert r['active_image_path'] == seg_path,          "active = segmented initially"
        assert 'dog'    in r['tags'],                        "tag: dog"
        assert 'bronze' in r['tags'],                        "tag: bronze"
        assert 'animal' in r['tags'],                        "tag: category"
        assert 'spine'  not in r['tags'],                    "spine not in tags"
        print(f"tags: {r['tags']}")
        print("upsert_classify: OK")

        # ── force re-classify does not overwrite a confirmed augmented image ──
        store.set_active_image('a1b2c3d4', aug_path)
        store.upsert_classify('a1b2c3d4', 'bronze+dog', classify_data)  # force re-classify
        r = store.get('a1b2c3d4')
        assert r['active_image_path'] == aug_path,           "augmented path preserved on re-classify"
        print("set_active_image / re-classify preservation: OK")

        # ── joints ────────────────────────────────────────────────────────────
        store.upsert_joints('a1b2c3d4', {
            'source_image_path': aug_path,
            'joint_hints':       [{'name': 'hip'}, {'name': 'shoulder'}],
            'skeleton':          [{'parent': 0, 'child': 1}],
            'suggested_joints':  8,
            'model_used':        'gemini-2.5-flash',
        })
        r = store.get('a1b2c3d4')
        assert r['joints']['model_used']       == 'gemini-2.5-flash', "model_used stored"
        assert r['joints']['source_image_path']== aug_path,           "source_image_path stored"
        assert len(r['joints']['joint_hints']) == 2,                  "joint_hints stored"
        print("upsert_joints: OK")

        # ── joints are overwritable (re-iteration) ────────────────────────────
        store.upsert_joints('a1b2c3d4', {
            'source_image_path': aug_path,
            'joint_hints':       [{'name': 'hip'}, {'name': 'knee'}, {'name': 'shoulder'}],
            'skeleton':          [],
            'suggested_joints':  10,
            'model_used':        'claude-sonnet-4-6',
        })
        r = store.get('a1b2c3d4')
        assert len(r['joints']['joint_hints']) == 3,                  "joints overwritten"
        assert r['joints']['model_used']       == 'claude-sonnet-4-6'
        print("joints re-iteration (overwrite): OK")

        # ── mesh ──────────────────────────────────────────────────────────────
        store.upsert_mesh('a1b2c3d4', {
            'mesh_hash':     'abc123456789',
            'meshy_task_id': 'task-xyz',
            'glb_path':      f'{tmp}/a1b2c3d4_mesh.glb',
            'glb_url':       'https://cdn.meshy.ai/a1b2c3d4.glb',
            'usdz_path':     None,
            'usdz_url':      None,
        })
        assert store.get_mesh_by_hash('abc123456789') is None, "miss when file absent"
        open(f'{tmp}/a1b2c3d4_mesh.glb', 'w').close()
        msh = store.get_mesh_by_hash('abc123456789')
        assert msh is not None,                    "hit after file created"
        assert msh['meshy_task_id'] == 'task-xyz'
        print("upsert_mesh / get_mesh_by_hash: OK")

        # ── rig ───────────────────────────────────────────────────────────────
        store.upsert_rig('a1b2c3d4', {
            'rigged_glb_path':    f'{tmp}/a1b2c3d4_rigged.glb',
            'viz_glb_path':       f'{tmp}/a1b2c3d4_viz.glb',
            'skeleton_json_path': f'{tmp}/a1b2c3d4_skeleton.json',
            'user_id':            'user_42',
        })
        r = store.get('a1b2c3d4')
        assert r['rig']['status']  == 'ok'
        assert r['rig']['user_id'] == 'user_42'
        print("upsert_rig: OK")

        store.set_rig_status('a1b2c3d4', 'error', 'Blender crashed')
        r = store.get('a1b2c3d4')
        assert r['rig']['status'] == 'error'
        assert r['rig']['error']  == 'Blender crashed'
        print("set_rig_status: OK")

        # ── gallery queries ───────────────────────────────────────────────────
        assert len(store.get_by_user('user_42'))           == 1,  "get_by_user"
        assert len(store.search_by_tag('user_42', 'dog'))  == 1,  "tag hit"
        assert len(store.search_by_tag('user_42', 'cat'))  == 0,  "tag miss"
        assert 'dog' in store.all_tags_for_user('user_42'),       "all_tags"
        print("get_by_user / search_by_tag / all_tags_for_user: OK")

        # ── with_urls ─────────────────────────────────────────────────────────
        wu = store.with_urls(store.get('a1b2c3d4'), 'localhost:6000')
        assert wu['active_image_url']      == f'http://localhost:6000/results/a1b2c3d4_augmented_a.png'
        assert wu['rig']['rigged_url']     == 'http://localhost:6000/results/a1b2c3d4_rigged.glb'
        assert wu['mesh']['glb_local_url'] == 'http://localhost:6000/results/a1b2c3d4_mesh.glb'
        assert wu['mesh']['glb_url']       == 'https://cdn.meshy.ai/a1b2c3d4.glb', "CDN URL preserved"
        assert wu['joints']['source_image_url'] == f'http://localhost:6000/results/a1b2c3d4_augmented_a.png'
        print("with_urls: OK")

        # ── disk persistence ──────────────────────────────────────────────────
        store2 = JsonStore(os.path.join(tmp, '_pipeline_store.json'))
        r2 = store2.get('a1b2c3d4')
        assert r2 is not None
        assert r2['joints']['suggested_joints'] == 10
        assert r2['active_image_path'] == aug_path
        print("disk persistence: OK")

        # ── extract_tags edge cases ───────────────────────────────────────────
        assert extract_tags({}, '') == []
        assert 'the' not in extract_tags({'object_type': 'the big dog'}, 'the+dog')
        assert 'cat'     in extract_tags({}, 'a+cat')
        assert 'a'   not in extract_tags({}, 'a+cat')
        print("extract_tags edge cases: OK")

    print("\n✅ All tests passed")
