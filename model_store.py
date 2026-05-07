"""
pipeline_store.py

Single source of truth for the 3D rigging pipeline.
Replaces model_store.py, _classify_cache, and _mesh_cache entirely.

Every pipeline run produces one record, keyed on classify_id
(md5(image_bytes + tag)[:8]).  The record has three sub-objects that are
filled in progressively as the pipeline advances:

  classify  — filled by /classify
  mesh      — filled when Meshy finishes
  rig       — filled when Blender finishes

Schema
------
{
  "classify_id":      "a1b2c3d4",
  "tag":              "dragon",          # raw user input, e.g. "fire+dragon"
  "tags":             ["dragon","fire"], # extracted search tokens
  "pipeline_version": 1,

  "classify": {
    "object_type":        "fire-breathing dragon",
    "category":           "animal",         # animal | humanoid | vehicle | other
    "needs_augmentation": false,
    "augment_prompt":     "",
    "suggested_joints":   8,
    "joint_hints":        [...],
    "skeleton":           [...],
    "segmented_image_path": "results/a1b2c3d4_segmented.png",
    "created_at":         "2025-01-01T00:00:00Z"
  },

  "mesh": {                               # null until Meshy finishes
    "mesh_hash":          "abc123456789", # md5(image_bytes + object_type)[:12]
    "meshy_task_id":      "task-xyz",
    "glb_path":           "results/a1b2c3d4_mesh.glb",
    "glb_url":            "https://cdn.meshy.ai/...",   # CDN URL, may expire
    "usdz_path":          "results/a1b2c3d4_mesh.usdz", # None if not produced
    "usdz_url":           "https://cdn.meshy.ai/...",
    "decimated_glb_path": "results/a1b2c3d4_decimated.glb",
    "created_at":         "2025-01-01T00:00:10Z"
  },

  "rig": {                                # null until rigging finishes
    "rigged_glb_path":    "results/a1b2c3d4_rigged.glb",
    "viz_glb_path":       "results/a1b2c3d4_viz.glb",
    "skeleton_json_path": "results/a1b2c3d4_skeleton.json",
    "status":             "ok",           # started|meshy|rigging|ok|error
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

All three backends expose the same public interface via get_store():

  store = get_store()

  # reads
  record  = store.get(classify_id)
  records = store.get_all()
  records = store.get_by_user(user_id)
  records = store.search_by_tag(user_id, tag)
  mesh    = store.get_mesh_by_hash(mesh_hash)   # returns mesh sub-object | None
  tags    = store.all_tags_for_user(user_id)

  # writes
  store.upsert_classify(classify_id, tag, classify_data)
  store.upsert_mesh(classify_id, mesh_data)
  store.upsert_rig(classify_id, rig_data)
  store.set_rig_status(classify_id, status, error=None)

  # URL helpers
  record_with_urls = store.with_urls(record, request.host)

Environment variables
---------------------
  PIPELINE_STORE_BACKEND   json | tinydb | clouddb
  RESULTS_DIR              directory for local files (default: 'results')
  CLOUDDB_URL
  CLOUDDB_TOKEN
  CLOUDDB_PROJECT
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

PIPELINE_VERSION = 1


# ── Tag extraction (ported from model_store.py) ───────────────────────────────

# Joint names that are internal skeleton landmarks, not useful search terms
_SKIP_BODY_PARTS = {
    'torso', 'spine', 'chest', 'pelvis', 'axle', 'body', 'root',
}

# Generic words stripped from user tags and object_type strings
_STOP_WORDS = {
    'a', 'an', 'the', 'in', 'or', 'for', 'no', 'on', 'and',
    't-pose', 'a-pose', 'easy', 'rigging', 'statue', 'two', 'walks',
}


def extract_tags(classify_data: dict, user_tag: str = '') -> list[str]:
    """
    Build a sorted list of search tokens from the user tag and classify result.

    Sources (in priority order):
      1. user_tag                      e.g. "bronze+dog"
      2. classify_data['object_type']  e.g. "bronze dog statue (bipedal, no arms)"
      3. classify_data['category']     e.g. "animal"
      4. joint_hints[*]['body_part']   e.g. "hip", "shoulder"
                                       (excludes _SKIP_BODY_PARTS)
    """
    tags: set[str] = set()

    if user_tag:
        words = user_tag.lower().replace('+', ' ').split()
        tags.update(w for w in words if w not in _STOP_WORDS and len(w) > 2)

    if classify_data:
        object_type = classify_data.get('object_type', '')
        # Strip parens, commas, periods so "bipedal," doesn't become a tag
        cleaned = object_type.lower().translate(str.maketrans('(),.-', '     '))
        words = cleaned.split()
        tags.update(w for w in words if w not in _STOP_WORDS and len(w) > 2)

        category = classify_data.get('category', '')
        if category:
            tags.add(category.lower())

        for hint in classify_data.get('joint_hints', []):
            if not isinstance(hint, dict):
                continue
            bp = hint.get('body_part', '')
            if bp and bp.lower() not in _SKIP_BODY_PARTS:
                tags.add(bp.lower())

    return sorted(t for t in tags if t)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _local_url(path: str | None, host: str) -> str | None:
    """Convert a local file path to an http:// URL served by this host."""
    if not path:
        return None
    return f"http://{host}/results/{os.path.basename(path)}"


def _inject_urls(record: dict, host: str) -> dict:
    """
    Return a shallow copy of *record* with computed http:// URLs added to each
    sub-object.  The original CDN glb_url / usdz_url are preserved unchanged
    alongside the new local_* variants.
    """
    r   = dict(record)
    cls = dict(r.get('classify') or {})
    msh = dict(r.get('mesh')     or {})
    rig = dict(r.get('rig')      or {})

    cls['segmented_url']       = _local_url(cls.get('segmented_image_path'), host)
    msh['glb_local_url']       = _local_url(msh.get('glb_path'),             host)
    msh['usdz_local_url']      = _local_url(msh.get('usdz_path'),            host)
    msh['decimated_local_url'] = _local_url(msh.get('decimated_glb_path'),   host)
    rig['rigged_url']          = _local_url(rig.get('rigged_glb_path'),      host)
    rig['viz_url']             = _local_url(rig.get('viz_glb_path'),         host)

    r['classify'] = cls if cls else None
    r['mesh']     = msh if msh else None
    r['rig']      = rig if rig else None
    return r


def _blank_record(classify_id: str, tag: str = '') -> dict:
    return {
        'classify_id':      classify_id,
        'tag':              tag,
        'tags':             [],
        'pipeline_version': PIPELINE_VERSION,
        'classify':         None,
        'mesh':             None,
        'rig':              None,
    }


# ── JSON backend ──────────────────────────────────────────────────────────────

class JsonStore:
    """
    In-process JSON file store.  Thread-safe via RLock.
    Writes are atomic: write to .tmp then os.replace().
    Zero external dependencies — the default backend.
    """

    def __init__(self, path: str):
        self._path = path
        self._lock = threading.RLock()
        self._data: dict[str, dict] = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

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

    # ── internal ──────────────────────────────────────────────────────────────

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

    def search_by_tag(self, user_id: str, tag: str) -> list[dict]:
        tag_lower = tag.lower().strip()
        with self._lock:
            return [
                dict(r) for r in self._data.values()
                if tag_lower in r.get('tags', [])
                and (r.get('rig') or {}).get('user_id') == user_id
            ]

    def get_mesh_by_hash(self, mesh_hash: str) -> dict | None:
        """
        Return the mesh sub-object for the record whose mesh_hash matches,
        but only if the local GLB file still exists on disk.
        Returns None if the file is gone (Meshy will be re-called).
        """
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
        with self._lock:
            record = self._get_or_create(classify_id, tag)
            record['tag']      = tag
            record['tags']     = extract_tags(classify_data, tag)
            record['classify'] = {
                **classify_data,
                'created_at': classify_data.get('created_at') or _now(),
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
    TinyDB backend.  Same schema as JsonStore — one document per classify_id
    with classify / mesh / rig sub-dicts.  Good for single-server deployments
    where you want a proper document store without running a full database.
    """

    def __init__(self, path: str):
        from tinydb import TinyDB, Query
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._db    = TinyDB(path)
        self._table = self._db.table('pipeline')
        self._Q     = Query()
        self._lock  = threading.RLock()
        log.info(f"pipeline_store(tinydb): {len(self._table)} records in {path}")

    # ── internal ──────────────────────────────────────────────────────────────

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
            self._upsert(classify_id, {
                'tag':      tag,
                'tags':     extract_tags(classify_data, tag),
                'classify': classify_data,
            })

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
      mesh_index:{mesh_hash}   classify_id string (secondary index for mesh lookup)

    The mesh_index avoids a full scan in get_mesh_by_hash().
    Note: CloudDB has no local filesystem, so get_mesh_by_hash() cannot verify
    that the GLB file still exists — it trusts the stored record.
    """

    def __init__(self, url: str, token: str, project: str):
        import requests as _requests
        self._requests = _requests
        self._url      = url.rstrip('/')
        self._token    = token
        self._project  = project
        self._lock     = threading.RLock()
        log.info(f"pipeline_store(clouddb): {self._url} project={self._project}")

    # ── low-level HTTP ────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            'Authorization': f'Bearer {self._token}',
            'Content-Type':  'application/json',
        }

    def _get(self, key: str) -> Optional[dict | list | str]:
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

    # ── index helpers ─────────────────────────────────────────────────────────

    def _get_user_index(self, user_id: str) -> list[str]:
        return self._get(f"user_index:{user_id}") or []

    def _prepend_user_index(self, user_id: str, classify_id: str):
        index = self._get_user_index(user_id)
        if classify_id not in index:
            index.insert(0, classify_id)
            self._set(f"user_index:{user_id}", index)

    def _remove_from_user_index(self, user_id: str, classify_id: str):
        index = [i for i in self._get_user_index(user_id) if i != classify_id]
        self._set(f"user_index:{user_id}", index)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(self, classify_id: str) -> dict | None:
        return self._get(f"pipeline:{classify_id}")

    def get_all(self) -> list[dict]:
        # No global index in CloudDB — not practical without a separate list
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
        # Validate the hash matches (guards against stale index entries)
        return dict(msh) if msh.get('mesh_hash') == mesh_hash else None

    def all_tags_for_user(self, user_id: str) -> list[str]:
        tags: set[str] = set()
        for r in self.get_by_user(user_id):
            tags.update(r.get('tags', []))
        return sorted(tags)

    # ── writes ────────────────────────────────────────────────────────────────

    def _load_or_blank(self, classify_id: str, tag: str = '') -> dict:
        return self.get(classify_id) or _blank_record(classify_id, tag)

    def upsert_classify(self, classify_id: str, tag: str, classify_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id, tag)
            record['tag']      = tag
            record['tags']     = extract_tags(classify_data, tag)
            record['classify'] = {
                **classify_data,
                'created_at': classify_data.get('created_at') or _now(),
            }
            self._set(f"pipeline:{classify_id}", record)

    def upsert_mesh(self, classify_id: str, mesh_data: dict):
        with self._lock:
            record = self._load_or_blank(classify_id)
            mesh_data.setdefault('created_at', _now())
            record['mesh'] = mesh_data
            self._set(f"pipeline:{classify_id}", record)
            # Secondary index so get_mesh_by_hash() is O(1) not O(n)
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
            # Maintain user index so get_by_user() works
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

    Backend is selected by the PIPELINE_STORE_BACKEND environment variable:
      json     (default)  results/_pipeline_store.json  (no extra deps)
      tinydb              results/_pipeline_store.json  (via TinyDB)
      clouddb             requires CLOUDDB_URL, CLOUDDB_TOKEN, CLOUDDB_PROJECT
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

        else:  # json (default)
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

        classify_data = {
            'object_type':        'bronze dog statue (bipedal, no arms)',
            'category':           'animal',
            'needs_augmentation': False,
            'augment_prompt':     '',
            'suggested_joints':   6,
            'joint_hints': [
                {'name': 'hip',      'body_part': 'hip'},
                {'name': 'spine',    'body_part': 'spine'},    # skipped
                {'name': 'shoulder', 'body_part': 'shoulder'},
            ],
            'segmented_image_path': f'{tmp}/a1b2c3d4_segmented.png',
        }

        # ── classify ──────────────────────────────────────────────────────────
        store.upsert_classify('a1b2c3d4', 'bronze+dog', classify_data)
        r = store.get('a1b2c3d4')
        assert r is not None
        assert r['classify']['object_type'] == 'bronze dog statue (bipedal, no arms)'
        assert 'dog'      in r['tags'], f"missing 'dog' in {r['tags']}"
        assert 'bronze'   in r['tags'], f"missing 'bronze'"
        assert 'animal'   in r['tags'], f"missing 'animal' (category)"
        assert 'hip'      in r['tags'], f"missing 'hip' (joint body_part)"
        assert 'shoulder' in r['tags'], f"missing 'shoulder'"
        assert 'spine' not in r['tags'], "spine should be skipped"
        print(f"tags: {r['tags']}")
        print("upsert_classify: OK")

        # ── mesh (file absent → cache miss) ───────────────────────────────────
        store.upsert_mesh('a1b2c3d4', {
            'mesh_hash':     'abc123456789',
            'meshy_task_id': 'task-xyz',
            'glb_path':      f'{tmp}/a1b2c3d4_mesh.glb',
            'glb_url':       'https://cdn.meshy.ai/a1b2c3d4.glb',
            'usdz_path':     None,
            'usdz_url':      None,
        })
        assert store.get_mesh_by_hash('abc123456789') is None, \
            "should be None when GLB file absent"

        # create file → cache hit
        open(f'{tmp}/a1b2c3d4_mesh.glb', 'w').close()
        msh = store.get_mesh_by_hash('abc123456789')
        assert msh is not None,                     "cache hit after file created"
        assert msh['meshy_task_id'] == 'task-xyz',  "task id matches"
        print("upsert_mesh / get_mesh_by_hash: OK")

        # ── rig ───────────────────────────────────────────────────────────────
        store.upsert_rig('a1b2c3d4', {
            'rigged_glb_path':    f'{tmp}/a1b2c3d4_rigged.glb',
            'viz_glb_path':       f'{tmp}/a1b2c3d4_viz.glb',
            'skeleton_json_path': f'{tmp}/a1b2c3d4_skeleton.json',
            'user_id':            'user_42',
        })
        r = store.get('a1b2c3d4')
        assert r['rig']['status']  == 'ok',      "default status ok"
        assert r['rig']['user_id'] == 'user_42', "user_id stored"
        print("upsert_rig: OK")

        # ── set_rig_status ────────────────────────────────────────────────────
        store.set_rig_status('a1b2c3d4', 'error', 'Blender crashed')
        r = store.get('a1b2c3d4')
        assert r['rig']['status'] == 'error',           "status updated"
        assert r['rig']['error']  == 'Blender crashed', "error message stored"
        print("set_rig_status: OK")

        # ── gallery queries ───────────────────────────────────────────────────
        by_user = store.get_by_user('user_42')
        assert len(by_user) == 1, "get_by_user"
        assert store.search_by_tag('user_42', 'dog')    == [by_user[0]], "tag hit"
        assert store.search_by_tag('user_42', 'cat')    == [],           "tag miss"
        assert 'dog' in store.all_tags_for_user('user_42'), "all_tags"
        print("get_by_user / search_by_tag / all_tags_for_user: OK")

        # ── with_urls ─────────────────────────────────────────────────────────
        wu = store.with_urls(store.get('a1b2c3d4'), 'localhost:6000')
        assert wu['rig']['rigged_url']    == 'http://localhost:6000/results/a1b2c3d4_rigged.glb'
        assert wu['mesh']['glb_local_url']== 'http://localhost:6000/results/a1b2c3d4_mesh.glb'
        assert wu['mesh']['glb_url']      == 'https://cdn.meshy.ai/a1b2c3d4.glb', \
            "CDN URL preserved"
        print("with_urls: OK")

        # ── disk persistence (reload) ─────────────────────────────────────────
        store2 = JsonStore(os.path.join(tmp, '_pipeline_store.json'))
        r2 = store2.get('a1b2c3d4')
        assert r2 is not None
        assert r2['classify']['object_type'] == 'bronze dog statue (bipedal, no arms)'
        print("disk persistence: OK")

        # ── extract_tags edge cases ───────────────────────────────────────────
        assert extract_tags({}, '') == []
        assert 'the' not in extract_tags({'object_type': 'the big dog'}, 'the+dog')
        assert 'a'   not in extract_tags({}, 'a+cat')
        assert 'cat' in     extract_tags({}, 'a+cat')
        print("extract_tags edge cases: OK")

    print("\n✅ All tests passed")
