"""
model_store.py

Abstraction layer for saving and retrieving model records.
Supports TinyDB (local, default) and CloudDB (MIT App Inventor CloudDB / Redis).

Record schema:
    {
        "classify_id":        "clf_abc123",
        "mesh_id":            "019db81a",
        "user_id":            "user_42",
        "object_type":        "bronze dog statue (bipedal, no arms)",
        "category":           "animal",
        "needs_augmentation": false,
        "augment_prompt":     "",
        "segmented_image":    "clf_abc123_segmented.png",
        "glb_path":           "019db81a_mesh.glb",
        "glb_url":            "https://assets.meshy.ai/...",   (CDN, may expire)
        "usdz_path":          "019db81a_mesh.usdz",
        "rigged_path":        "019db81a_rigged.glb",
        "tags":               ["dog", "bronze", "animal"],
        "created_at":         1714000000.0
    }

Usage:
    from model_store import ModelStore

    store = ModelStore()                    # TinyDB (default)
    store = ModelStore(backend='clouddb',
                       clouddb_url=...,
                       clouddb_token=...,
                       clouddb_project=...)

Environment variables:
    MODEL_STORE_BACKEND   — 'tinydb' or 'clouddb'
    CLOUDDB_URL
    CLOUDDB_TOKEN
    CLOUDDB_PROJECT
    RESULTS_DIR           — directory for TinyDB file (default: 'results')
"""

import os
import time
import json
import logging
from typing import Optional

log = logging.getLogger(__name__)


# ── Tag extraction ─────────────────────────────────────────────────────────────

def extract_tags(classify_data: dict, user_tag: str = '') -> list:
    """Build a deduplicated sorted tag list from classify_data + user_tag."""
    SKIP_PARTS = {'torso', 'spine', 'chest', 'pelvis', 'axle', 'body', 'root'}
    tags = set()

    if user_tag:
        tags.update(user_tag.lower().replace('+', ' ').split())

    if classify_data:
        object_type = classify_data.get('object_type', '')
        tags.update(object_type.lower().split())

        category = classify_data.get('category', '')
        if category:
            tags.add(category.lower())

        for hint in classify_data.get('joint_hints', []):
            bp = hint.get('body_part', '')
            if bp and bp.lower() not in SKIP_PARTS:
                tags.add(bp.lower())

    return sorted(t for t in tags if t)


# ── Record builder ─────────────────────────────────────────────────────────────

def build_record(classify_id: str,
                 mesh_id: str,
                 user_id: str,
                 classify_data: dict,
                 segmented_image: str,
                 glb_path: str,
                 glb_url: str,
                 usdz_path: str,
                 rigged_path: str,
                 user_tag: str = '') -> dict:
    """
    Construct the canonical record dict.
    Paths are stored as basenames only — URLs are reconstructed at serve time.
    glb_url is the original CDN URL (stored as-is, may expire).
    """
    cd = classify_data or {}

    return {
        # ── Identity ───────────────────────────────────────────
        'classify_id':        classify_id,
        'mesh_id':            mesh_id,
        'user_id':            user_id,

        # ── Classification ─────────────────────────────────────
        'object_type':        cd.get('object_type', ''),
        'category':           cd.get('category', ''),
        'needs_augmentation': cd.get('needs_augmentation', False),
        'augment_prompt':     cd.get('augment_prompt', ''),

        # ── Files (basenames only) ─────────────────────────────
        'segmented_image':    os.path.basename(segmented_image) if segmented_image else None,
        'glb_path':           os.path.basename(glb_path)        if glb_path        else None,
        'glb_url':            glb_url,  # CDN URL — kept as-is
        'usdz_path':          os.path.basename(usdz_path)       if usdz_path       else None,
        'rigged_path':        os.path.basename(rigged_path)     if rigged_path     else None,

        # ── Search ─────────────────────────────────────────────
        'tags':               extract_tags(classify_data, user_tag),

        # ── Meta ───────────────────────────────────────────────
        'created_at':         time.time(),
    }


# ── TinyDB backend ─────────────────────────────────────────────────────────────

class TinyDBBackend:
    def __init__(self, results_dir: str = 'results'):
        from tinydb import TinyDB
        os.makedirs(results_dir, exist_ok=True)
        db_path    = os.path.join(results_dir, 'gallery.json')
        self.db    = TinyDB(db_path)
        self.table = self.db.table('models')
        log.info(f"TinyDB backend: {db_path}")

    def _q(self):
        from tinydb import Query
        return Query()

    def save(self, record: dict) -> dict:
        self.table.upsert(record, self._q().classify_id == record['classify_id'])
        log.info(f"TinyDB saved: {record['classify_id']}")
        return record
        
    def get_by_classify_id(self, classify_id: str) -> Optional[dict]:
        results = self.table.search(self._q().classify_id == classify_id)
        return results[0] if results else None

    def get_by_mesh_id(self, mesh_id: str) -> Optional[dict]:
        results = self.table.search(self._q().mesh_id == mesh_id)
        return results[0] if results else None

    def get_by_user(self, user_id: str) -> list:
        records = self.table.search(self._q().user_id == user_id)
        return sorted(records, key=lambda r: r.get('created_at', 0), reverse=True)

    def search_by_tag(self, user_id: str, tag: str) -> list:
        records = self.get_by_user(user_id)
        tag     = tag.lower().strip()
        return [r for r in records if tag in r.get('tags', [])]

    def delete(self, classify_id: str) -> bool:
        removed = self.table.remove(self._q().classify_id == classify_id)
        return len(removed) > 0

    def all_tags_for_user(self, user_id: str) -> list:
        records = self.get_by_user(user_id)
        tags    = set(t for r in records for t in r.get('tags', []))
        return sorted(tags)


# ── CloudDB backend ────────────────────────────────────────────────────────────

class CloudDBBackend:
    """
    MIT App Inventor CloudDB (Redis-backed key-value store).

    Key scheme:
        model:{classify_id}     → full record JSON
        user_index:{user_id}    → JSON list of classify_ids, newest first
    """

    def __init__(self, url: str, token: str, project: str):
        import requests as req
        self.url     = url.rstrip('/')
        self.token   = token
        self.project = project
        self.req     = req
        log.info(f"CloudDB backend: {self.url} project={self.project}")

    def _headers(self):
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type':  'application/json',
        }

    def _get(self, key: str) -> Optional[dict]:
        try:
            resp = self.req.get(
                f"{self.url}/v1/{self.project}/{key}",
                headers=self._headers(), timeout=10
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            raw   = resp.json()
            value = raw.get('value') or raw.get('result')
            return json.loads(value) if isinstance(value, str) else value
        except Exception as e:
            log.error(f"CloudDB GET {key} failed: {e}")
            return None

    def _set(self, key: str, value) -> bool:
        try:
            resp = self.req.post(
                f"{self.url}/v1/{self.project}/{key}",
                headers=self._headers(), timeout=10,
                json={'value': json.dumps(value)}
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            log.error(f"CloudDB SET {key} failed: {e}")
            return False

    def _delete_key(self, key: str) -> bool:
        try:
            resp = self.req.delete(
                f"{self.url}/v1/{self.project}/{key}",
                headers=self._headers(), timeout=10
            )
            return resp.status_code in (200, 204)
        except Exception as e:
            log.error(f"CloudDB DELETE {key} failed: {e}")
            return False

    def _get_user_index(self, user_id: str) -> list:
        return self._get(f"user_index:{user_id}") or []

    def _update_user_index(self, user_id: str, classify_id: str):
        index = self._get_user_index(user_id)
        if classify_id not in index:
            index.insert(0, classify_id)
        self._set(f"user_index:{user_id}", index)

    def save(self, record: dict) -> dict:
        classify_id = record['classify_id']
        self._set(f"model:{classify_id}", record)
        self._update_user_index(record['user_id'], classify_id)
        log.info(f"CloudDB saved: {classify_id}")
        return record

    def get_by_classify_id(self, classify_id: str) -> Optional[dict]:
        return self._get(f"model:{classify_id}")

    def get_by_mesh_id(self, mesh_id: str) -> Optional[dict]:
        log.warning("CloudDB get_by_mesh_id: no secondary index, not supported")
        return None

    def get_by_user(self, user_id: str) -> list:
        index   = self._get_user_index(user_id)
        records = [self.get_by_classify_id(cid) for cid in index]
        return [r for r in records if r]

    def search_by_tag(self, user_id: str, tag: str) -> list:
        records = self.get_by_user(user_id)
        tag     = tag.lower().strip()
        return [r for r in records if tag in r.get('tags', [])]

    def delete(self, classify_id: str) -> bool:
        record = self.get_by_classify_id(classify_id)
        if record:
            index = [i for i in self._get_user_index(record['user_id'])
                     if i != classify_id]
            self._set(f"user_index:{record['user_id']}", index)
            self._delete_key(f"model:{classify_id}")
            return True
        return False

    def all_tags_for_user(self, user_id: str) -> list:
        records = self.get_by_user(user_id)
        tags    = set(t for r in records for t in r.get('tags', []))
        return sorted(tags)


# ── ModelStore (public API) ────────────────────────────────────────────────────

class ModelStore:
    """
    Unified interface. Backend selected by constructor arg or
    MODEL_STORE_BACKEND env var ('tinydb' | 'clouddb').
    """

    def __init__(self,
                 backend: str = None,
                 results_dir: str = None,
                 clouddb_url: str = None,
                 clouddb_token: str = None,
                 clouddb_project: str = None):

        backend = (backend
                   or os.environ.get('MODEL_STORE_BACKEND', 'tinydb')).lower()

        if backend == 'clouddb':
            url     = clouddb_url     or os.environ.get('CLOUDDB_URL', '')
            token   = clouddb_token   or os.environ.get('CLOUDDB_TOKEN', '')
            project = clouddb_project or os.environ.get('CLOUDDB_PROJECT', '')
            if not all([url, token, project]):
                raise ValueError(
                    "CloudDB requires url, token, and project. "
                    "Pass as args or set CLOUDDB_URL / CLOUDDB_TOKEN / CLOUDDB_PROJECT."
                )
            self._backend = CloudDBBackend(url, token, project)
        else:
            results_dir   = results_dir or os.environ.get('RESULTS_DIR', 'results')
            self._backend = TinyDBBackend(results_dir)

        self.backend_name = backend

    # ── Write ──────────────────────────────────────────────────────────────────

    def save_model_record(self,
                          classify_id: str,
                          mesh_id: str,
                          user_id: str,
                          classify_data: dict,
                          segmented_image: str,
                          glb_path: str,
                          glb_url: str,
                          usdz_path: str,
                          rigged_path: str,
                          user_tag: str = '') -> dict:
        """
        Build and persist a model record. Returns the saved record.

        Args:
            classify_id     — from /classify response
            mesh_id         — Meshy task id prefix  e.g. '019db81a'
            user_id         — user identifier from App Inventor
            classify_data   — full dict from classify_with_vision()
            segmented_image — path to segmented PNG
            glb_path        — local path to raw mesh GLB
            glb_url         — original CDN URL from Meshy/Trellis (may expire)
            usdz_path       — local path to USDZ, or None
            rigged_path     — local path to rigged GLB
            user_tag        — raw user tag string e.g. 'bronze+dog'
        """
        record = build_record(
            classify_id, mesh_id, user_id, classify_data,
            segmented_image, glb_path, glb_url,
            usdz_path, rigged_path, user_tag
        )
        return self._backend.save(record)

    def delete_model_record(self, classify_id: str) -> bool:
        """Delete by classify_id. Returns True if found and deleted."""
        return self._backend.delete(classify_id)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_model_record(self, classify_id: str) -> Optional[dict]:
        """Retrieve a single record by classify_id."""
        return self._backend.get_by_classify_id(classify_id)

    def get_by_mesh_id(self, mesh_id: str) -> Optional[dict]:
        """Retrieve a single record by mesh_id."""
        return self._backend.get_by_mesh_id(mesh_id)

    def get_user_records(self, user_id: str) -> list:
        """All records for a user, newest first."""
        return self._backend.get_by_user(user_id)

    def search_by_tag(self, user_id: str, tag: str) -> list:
        """Filter a user's records by a single tag string."""
        return self._backend.search_by_tag(user_id, tag)

    def all_tags_for_user(self, user_id: str) -> list:
        """All unique tags across a user's models, sorted."""
        return self._backend.all_tags_for_user(user_id)

    # ── URL helpers ────────────────────────────────────────────────────────────

    def with_urls(self, record: dict, host: str) -> dict:
        """
        Inject live local URLs from stored filenames + current host.
        glb_url (CDN) is preserved as-is alongside the local URL.

            return jsonify(store.with_urls(record, request.host))
        """
        if not record:
            return record

        def local_url(filename):
            return f"http://{host}/results/{filename}" if filename else None

        return {
            **record,
            'rigged_url':     local_url(record.get('rigged_path')),
            'local_glb_url':  local_url(record.get('glb_path')),
            'local_usdz_url': local_url(record.get('usdz_path')),
            # glb_url left as-is (original CDN URL)
        }

    def user_records_with_urls(self, user_id: str, host: str) -> list:
        """Convenience: all user records with URLs injected."""
        return [self.with_urls(r, host) for r in self.get_user_records(user_id)]


# ── Module-level singleton ─────────────────────────────────────────────────────

store: Optional[ModelStore] = None

def init(backend: str = None, **kwargs) -> ModelStore:
    """Initialize the module-level singleton. Call once at startup."""
    global store
    store = ModelStore(backend=backend, **kwargs)
    return store


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import tempfile

    print("=== TinyDB smoke test ===\n")
    with tempfile.TemporaryDirectory() as tmp:
        s = ModelStore(backend='tinydb', results_dir=tmp)

        classify_data = {
            'object_type':        'bronze dog statue (bipedal, no arms)',
            'category':           'animal',
            'needs_augmentation': False,
            'augment_prompt':     '',
            'joint_hints': [
                {'body_part': 'hip'},
                {'body_part': 'spine'},
            ],
        }

        r = s.save_model_record(
            classify_id     = 'clf_abc123',
            mesh_id         = '019db81a',
            user_id         = 'user_42',
            classify_data   = classify_data,
            segmented_image = 'results/clf_abc123_segmented.png',
            glb_path        = 'results/019db81a_mesh.glb',
            glb_url         = 'https://assets.meshy.ai/signed/019db81a.glb',
            usdz_path       = 'results/019db81a_mesh.usdz',
            rigged_path     = 'results/019db81a_rigged.glb',
            user_tag        = 'bronze+dog',
        )
        print("Saved record:")
        print(json.dumps(r, indent=2))

        fetched = s.get_model_record('clf_abc123')
        assert fetched['object_type'] == 'bronze dog statue (bipedal, no arms)'
        assert fetched['mesh_id']     == '019db81a'
        assert fetched['glb_url']     == 'https://assets.meshy.ai/signed/019db81a.glb'
        assert fetched['glb_path']    == '019db81a_mesh.glb'   # basename only
        assert 'dog'    in fetched['tags']
        assert 'bronze' in fetched['tags']
        assert 'hip'    in fetched['tags']
        print("\nTags:", fetched['tags'])

        by_mesh = s.get_by_mesh_id('019db81a')
        assert by_mesh is not None
        print("get_by_mesh_id: OK")

        by_user = s.get_user_records('user_42')
        assert len(by_user) == 1
        print("get_user_records: OK")

        by_tag = s.search_by_tag('user_42', 'bronze')
        assert len(by_tag) == 1
        print("search_by_tag 'bronze': OK")

        no_match = s.search_by_tag('user_42', 'cat')
        assert len(no_match) == 0
        print("search_by_tag 'cat' (no match): OK")

        with_urls = s.with_urls(fetched, 'localhost:6000')
        assert with_urls['rigged_url']    == 'http://localhost:6000/results/019db81a_rigged.glb'
        assert with_urls['local_glb_url'] == 'http://localhost:6000/results/019db81a_mesh.glb'
        assert with_urls['glb_url']       == 'https://assets.meshy.ai/signed/019db81a.glb'
        print("with_urls: OK")

        assert s.delete_model_record('clf_abc123')
        assert s.get_model_record('clf_abc123') is None
        print("delete: OK")

    print("\n✅ All tests passed")
