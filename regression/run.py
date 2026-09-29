#!/usr/bin/env python3
"""
Snapshot regression check for the joint → skeleton → rig pipeline.

Each fixture is a real record from results/_pipeline_store.json (its classify
+ joints data, frozen into regression/fixtures/<label>.json) plus that
record's decimated mesh in results/<id>/ (not committed — too large). The
pipeline stages are re-run against that fixed input and compared to the
committed numbers in regression/baselines/<label>.json.

Stages
------
  mesh_bounds  compute_mesh_bounds() — trunk/neck/pelvis/shoulder landmarks
  joints       snap → mesh-guided correction → symmetry → verify
               (vision call stubbed out, so only the deterministic
               geometric safety net runs)
  skeleton     skeleton_from_joints() + utils.inject_keyframes()
  blender      [--blender] rig.py in Blender, then per-bone skin-weight stats
               read back from the rigged GLB

Usage
-----
  venv/bin/python regression/run.py                    # check fast stages
  venv/bin/python regression/run.py --blender          # + Blender rig (slow)
  venv/bin/python regression/run.py --only broccoli tomato
  venv/bin/python regression/run.py --update [--blender]   # accept current output
  venv/bin/python regression/run.py --add da008649 --label broccoli

Exit status is 1 if any stage differs from its baseline.
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import logging
import os
import struct
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(REPO, 'regression')
FIXTURES = os.path.join(HERE, 'fixtures')
BASELINES = os.path.join(HERE, 'baselines')

# Absolute tolerances. Stages 1–3 are all in normalized (0–1 of the mesh's
# bounding box) units or radians; blender compares vertex fractions.
TOL = {'mesh_bounds': 0.01, 'joints': 0.01, 'skeleton': 0.01, 'blender': 0.02}
FAST_STAGES = ['mesh_bounds', 'joints', 'skeleton']


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def _mesh_path(classify_id):
    return os.path.join(REPO, 'results', classify_id, f'{classify_id}_decimated.glb')


def _flatten(prefix, value, out):
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f'{prefix}.{k}' if prefix else str(k), v, out)
    elif isinstance(value, (list, tuple)) and value and all(
            isinstance(v, (int, float)) for v in value) and len(value) <= 4:
        for axis, v in zip('xyzw', value):
            out[f'{prefix}.{axis}'] = v
    else:
        out[prefix] = value
    return out


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 5)
    if isinstance(v, np.integer):
        return int(v)
    return v


def _norm(pos, bmin, brange):
    return [_clean((p - lo) / r) for p, lo, r in zip(pos, bmin, brange)]


# ── Stages ────────────────────────────────────────────────────────────────────

def stage_mesh_bounds(ss, mesh, fx):
    mb = ss.compute_mesh_bounds(mesh)
    # bmin/bmax/width/... are properties of the mesh file itself (guarded by
    # the fixture's mesh hash); only the detected landmarks are interesting.
    return {k: _clean(v) for k, v in mb.items()
            if k not in ('bmin', 'bmax', 'width', 'height', 'depth')}


def stage_joints(ss, mesh, fx):
    joints_data = copy.deepcopy(fx['joints'])
    rig_type = fx['classify'].get('rig_type', '')
    object_type = fx['classify'].get('object_type', '')
    mesh_bounds = ss.compute_mesh_bounds(mesh)

    joints_data = ss.snap_joints_to_mesh(joints_data, mesh)
    joints_data = ss.mesh_guided_joint_correction(joints_data, mesh, rig_type)
    joints_data = ss.enforce_bilateral_symmetry(joints_data)
    joints_data = ss.verify_and_snap_joints(
        joints_data, mesh, object_type, rig_type,
        classify_id=None, mesh_bounds=mesh_bounds)

    out = {}
    for h in joints_data.get('joint_hints', []):
        p = h.get('position_normalized') or {}
        for axis in 'xyz':
            if axis in p:
                out[f"{h['name']}.{axis}"] = _clean(p[axis])
    return out


def _build_skeleton(ss, utils, fx):
    glb = _mesh_path(fx['classify_id'])
    skel = ss.skeleton_from_joints(
        copy.deepcopy(fx['joints']), glb, fx['classify'].get('rigid_parts', []))
    if skel is None:
        raise RuntimeError('skeleton_from_joints returned None')
    return utils.inject_keyframes(skel)


def stage_skeleton(ss, mesh, fx, skel):
    verts = np.asarray(mesh.vertices)
    bmin = verts.min(axis=0)
    brange = verts.max(axis=0) - bmin
    brange[brange == 0] = 1.0

    names = {j['id']: j['name'] for j in skel['joints']}
    out = {}
    for j in skel['joints']:
        _flatten(f"pos.{j['name']}", _norm(j['position'], bmin, brange), out)
        for a in (j.get('hint') or {}).get('animations', []):
            vals = [kf[1] for kf in a.get('keyframes', [])]
            if not vals:
                continue
            key = f"anim.{j['name']}.{a['clip']}.{a['property']}.{a.get('axis', '')}"
            out[f'{key}.min'] = _clean(min(vals))
            out[f'{key}.max'] = _clean(max(vals))
            out[f'{key}.frames'] = ','.join(str(kf[0]) for kf in a['keyframes'])
    for b in skel['bones']:
        out[f"bone.{b['name']}"] = f"{names.get(b['parent'])}->{names.get(b['child'])}"
    return out


def _read_glb_skin(path):
    """Return (positions[N,3], joint_names, joints[N,K], weights[N,K])."""
    with open(path, 'rb') as f:
        data = f.read()
    magic, _, _ = struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67, 'not a GLB'
    off, gltf, binchunk = 12, None, b''
    while off < len(data):
        length, ctype = struct.unpack_from('<II', data, off)
        chunk = data[off + 8:off + 8 + length]
        if ctype == 0x4E4F534A:
            gltf = json.loads(chunk)
        elif ctype == 0x004E4942:
            binchunk = chunk
        off += 8 + length

    ctypes = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
    ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

    def accessor(i):
        acc = gltf['accessors'][i]
        bv = gltf['bufferViews'][acc['bufferView']]
        dt = np.dtype(ctypes[acc['componentType']])
        n = ncomp[acc['type']]
        stride = bv.get('byteStride') or dt.itemsize * n
        arr = np.ndarray((acc['count'], n), dtype=dt, buffer=binchunk,
                         offset=bv.get('byteOffset', 0) + acc.get('byteOffset', 0),
                         strides=(stride, dt.itemsize)).copy()
        if acc.get('normalized'):
            arr = arr.astype(np.float64) / np.iinfo(dt).max
        return arr

    skin = gltf['skins'][0]
    joint_names = [gltf['nodes'][n].get('name', f'node{n}') for n in skin['joints']]
    pos, jts, wts = [], [], []
    for node in gltf['nodes']:
        if 'mesh' not in node or 'skin' not in node:
            continue
        for prim in gltf['meshes'][node['mesh']]['primitives']:
            attrs = prim['attributes']
            if 'JOINTS_0' not in attrs:
                continue
            pos.append(accessor(attrs['POSITION']))
            j = [accessor(attrs['JOINTS_0'])]
            w = [accessor(attrs['WEIGHTS_0'])]
            if 'JOINTS_1' in attrs:
                j.append(accessor(attrs['JOINTS_1']))
                w.append(accessor(attrs['WEIGHTS_1']))
            jts.append(np.hstack(j))
            wts.append(np.hstack(w))
    k = max(a.shape[1] for a in jts)
    pad = lambda a: np.pad(a, ((0, 0), (0, k - a.shape[1])))  # noqa: E731
    return (np.vstack(pos), joint_names,
            np.vstack([pad(a) for a in jts]).astype(int),
            np.vstack([pad(a) for a in wts]).astype(np.float64))


def stage_blender(ss, mesh, fx, skel):
    glb = _mesh_path(fx['classify_id'])
    with tempfile.TemporaryDirectory() as tmp:
        skel_path = os.path.join(tmp, 'skeleton.json')
        rigged = os.path.join(tmp, 'rigged.glb')
        with open(skel_path, 'w') as f:
            json.dump(skel, f)
        ss.run_blender_rig(glb, skel_path, rigged)
        pos, names, jts, wts = _read_glb_skin(rigged)

    n = len(pos)
    bmin = pos.min(axis=0)
    brange = pos.max(axis=0) - bmin
    brange[brange == 0] = 1.0
    dominant = jts[np.arange(n), wts.argmax(axis=1)]

    top = wts.max(axis=1)
    out = {'multi_influence_frac': _clean(((wts >= 0.1).sum(axis=1) >= 2).mean()),
           'max_weight_median': _clean(np.median(top)),
           'max_weight_p10': _clean(np.percentile(top, 10))}
    for bi, name in enumerate(names):
        w = np.where(jts == bi, wts, 0.0).sum(axis=1)
        total = w.sum()
        out[f'{name}.dominant_frac'] = _clean((dominant == bi).mean())
        out[f'{name}.weight_share'] = _clean(total / n)
        if total > 1e-9:
            c = (pos * w[:, None]).sum(axis=0) / total
            _flatten(f'{name}.centroid', _norm(c, bmin, brange), out)
    return out


# ── Compare / report ──────────────────────────────────────────────────────────

def compare(stage, base, cur):
    tol = TOL[stage]
    diffs = []
    for k in sorted(set(base) | set(cur)):
        if k not in cur:
            diffs.append(f'{k}: missing (baseline {base[k]})')
        elif k not in base:
            diffs.append(f'{k}: new ({cur[k]})')
        else:
            b, c = base[k], cur[k]
            if isinstance(b, (int, float)) and isinstance(c, (int, float)) \
                    and not isinstance(b, bool):
                if abs(c - b) > tol:
                    diffs.append(f'{k}: {b:.3f} → {c:.3f} (Δ {c - b:+.3f}, tol {tol})')
            elif b != c:
                diffs.append(f'{k}: {b!r} → {c!r}')
    return diffs


def load_fixtures(only):
    fxs = []
    for fn in sorted(os.listdir(FIXTURES)):
        if not fn.endswith('.json'):
            continue
        label = fn[:-5]
        if only and label not in only:
            continue
        with open(os.path.join(FIXTURES, fn)) as f:
            fxs.append((label, json.load(f)))
    missing = set(only or []) - {label for label, _ in fxs}
    if missing:
        sys.exit(f'Unknown fixture(s): {", ".join(sorted(missing))}')
    return fxs


def add_fixture(classify_id, label):
    with open(os.path.join(REPO, 'results', '_pipeline_store.json')) as f:
        store = json.load(f)
    rec = store.get(classify_id)
    if not rec or not (rec.get('joints') or {}).get('joint_hints'):
        sys.exit(f'{classify_id}: no stored joints in the pipeline store')
    glb = _mesh_path(classify_id)
    if not os.path.exists(glb):
        sys.exit(f'{classify_id}: no decimated mesh at {glb}')
    fx = {
        'classify_id': classify_id,
        'mesh_sha256': _sha256(glb),
        'classify': rec['classify'],
        'joints': rec['joints'],
    }
    os.makedirs(FIXTURES, exist_ok=True)
    with open(os.path.join(FIXTURES, f'{label}.json'), 'w') as f:
        json.dump(fx, f, indent=2)
    print(f'Added fixture {label} ({classify_id}: {rec["classify"].get("object_type")})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--blender', action='store_true', help='also rig in Blender (slow)')
    ap.add_argument('--update', action='store_true', help='overwrite baselines with current output')
    ap.add_argument('--only', nargs='+', metavar='LABEL', help='run only these fixtures')
    ap.add_argument('--add', metavar='CLASSIFY_ID', help='freeze a store record as a new fixture')
    ap.add_argument('--label', help='fixture name for --add')
    ap.add_argument('-v', '--verbose', action='store_true', help='show pipeline logging')
    args = ap.parse_args()

    os.chdir(REPO)
    sys.path.insert(0, REPO)

    if args.add:
        if not args.label:
            sys.exit('--add needs --label')
        add_fixture(args.add, args.label)
        args.only, args.update = [args.label], True

    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    import seg_server as ss
    import utils
    import trimesh
    if not args.verbose:
        logging.getLogger().setLevel(logging.ERROR)

    # Joint verification is a live vision call; with it stubbed out only the
    # deterministic geometric safety net in verify_and_snap_joints runs.
    ss._try_claude = lambda *a, **k: None
    ss.render_mesh_front_view = lambda *a, **k: b''

    stages = FAST_STAGES + (['blender'] if args.blender else [])
    os.makedirs(BASELINES, exist_ok=True)
    failed = skipped = 0

    for label, fx in load_fixtures(args.only):
        cid = fx['classify_id']
        head = f"{label} ({cid})"
        glb = _mesh_path(cid)
        if not os.path.exists(glb):
            print(f'SKIP  {head}: mesh not found at {os.path.relpath(glb, REPO)}')
            skipped += 1
            continue
        if _sha256(glb) != fx['mesh_sha256']:
            print(f'SKIP  {head}: decimated mesh changed since fixture was captured '
                  f'(re-run --add {cid} --label {label} if intentional)')
            skipped += 1
            continue

        base_path = os.path.join(BASELINES, f'{label}.json')
        baseline = {}
        if os.path.exists(base_path):
            with open(base_path) as f:
                baseline = json.load(f)

        mesh = trimesh.load(glb, force='mesh')
        current, errors, skel = {}, {}, None
        for st in stages:
            # utils/rig print debug lines straight to stdout
            quiet = (contextlib.nullcontext() if args.verbose
                     else contextlib.redirect_stdout(io.StringIO()))
            try:
                with quiet:
                    if st in ('skeleton', 'blender'):
                        skel = skel or _build_skeleton(ss, utils, fx)
                        current[st] = (stage_skeleton if st == 'skeleton'
                                       else stage_blender)(ss, mesh, fx, skel)
                    else:
                        current[st] = {'mesh_bounds': stage_mesh_bounds,
                                       'joints': stage_joints}[st](ss, mesh, fx)
            except Exception as e:
                errors[st] = f'{type(e).__name__}: {e}'

        if args.update:
            if errors:
                print(f'ERROR {head}: not updating — ' +
                      '; '.join(f'{s}: {e}' for s, e in errors.items()))
                failed += 1
                continue
            baseline.update(current)
            with open(base_path, 'w') as f:
                json.dump(baseline, f, indent=2, sort_keys=True)
            print(f'WROTE {head}: {", ".join(current)}')
            continue

        report = []
        for st in stages:
            if st in errors:
                report.append((st, [f'crashed — {errors[st]}']))
            elif st not in baseline:
                report.append((st, ['no baseline yet (run --update)']))
            else:
                d = compare(st, baseline[st], current[st])
                if d:
                    report.append((st, d))
        if report:
            failed += 1
            print(f'FAIL  {head}')
            for st, diffs in report:
                print(f'      {st}:')
                for d in diffs[:25]:
                    print(f'        {d}')
                if len(diffs) > 25:
                    print(f'        … {len(diffs) - 25} more')
        else:
            print(f'ok    {head}: {", ".join(stages)}')

    print(f'\n{failed} failed, {skipped} skipped')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
