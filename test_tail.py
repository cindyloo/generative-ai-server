#!/usr/bin/env python3
"""
test_tail.py
============
Checks add_tail(): a fish or a bird gets joint_tail_base -> joint_tail_tip
behind its body, in the middle of its tail, parented to the pelvis (or to
root, for the flying skeleton, which has no pelvis).

Usage
-----
  python test_tail.py

Uses a synthetic fish (an ellipsoid body facing +Z with a thin fan of a tail
behind it), and the pufferfish record 865fee5d too if results/ has it.
"""

import copy
import json
import os

import numpy as np
import trimesh

from seg_server import add_tail, has_tail


def _joints(pelvis_z=0.5, pelvis=True):
    names = ['joint_root', 'joint_pelvis', 'joint_spine', 'joint_head']
    if not pelvis:
        names.remove('joint_pelvis')
    return {
        'joint_hints': [{'name': n, 'body_part': n[6:], 'deforms_mesh': False,
                         'position_normalized': {'x': 0.5, 'y': 0.5, 'z': pelvis_z}}
                        for n in names],
        'skeleton': [{'parent': a, 'child': b} for a, b in zip(names, names[1:])],
    }


def synthetic_fish():
    """Body: ellipsoid at the origin, face +Z. Tail: a narrow stalk, then a
    wide flat fan, behind it along -Z."""
    body = trimesh.creation.icosphere(subdivisions=3)
    body.apply_scale([0.6, 0.5, 0.8])
    stalk = trimesh.creation.box(extents=[0.12, 0.15, 0.3])
    stalk.apply_translation([0, 0, -0.9])
    fan = trimesh.creation.box(extents=[0.05, 0.8, 0.35])
    fan.apply_translation([0, 0, -1.2])
    return trimesh.util.concatenate([body, stalk, fan])


def tail_of(data):
    hints = {h['name']: h['position_normalized'] for h in data['joint_hints']}
    return hints.get('joint_tail_base'), hints.get('joint_tail_tip')


def world(mesh, p):
    v = np.asarray(mesh.vertices)
    bmin, brange = v.min(0), v.max(0) - v.min(0)
    return bmin + np.array([p['x'], p['y'], p['z']]) * brange


def check_synthetic():
    mesh = synthetic_fish()
    # Pelvis at the body centre: z=0 in a mesh spanning -1.375..0.8.
    v = np.asarray(mesh.vertices)
    pelvis_z = (0 - v[:, 2].min()) / np.ptp(v[:, 2])
    out = add_tail(_joints(pelvis_z), mesh, 'cartoon goldfish')

    base, tip = tail_of(out)
    assert base and tip, 'a fish should get a tail'
    base_w, tip_w = world(mesh, base), world(mesh, tip)
    # The tip is in the middle of the fan, not on one of its corners...
    assert -1.4 < tip_w[2] < -1.15 and abs(tip_w[0]) < 0.05 and abs(tip_w[1]) < 0.1, tip_w
    # ...and the base is at the narrow stalk, between body and fan.
    assert -1.05 < base_w[2] < -0.75, base_w
    assert {'parent': 'joint_pelvis', 'child': 'joint_tail_base'} in out['skeleton']
    assert {'parent': 'joint_tail_base', 'child': 'joint_tail_tip'} in out['skeleton']

    again = add_tail(copy.deepcopy(out), mesh, 'cartoon goldfish')
    assert len(again['joint_hints']) == len(out['joint_hints']), 'adds a tail only once'

    untouched = add_tail(_joints(pelvis_z), mesh, 'cartoon bee')
    assert tail_of(untouched) == (None, None), 'only fish get a tail'

    flat = trimesh.creation.box(extents=[1.0, 1.0, 0.05])
    assert tail_of(add_tail(_joints(), flat, 'fish drawing')) == (None, None), \
        'a flat mesh has nothing behind it to be a tail'
    print('synthetic fish: ok', base_w.round(2), tip_w.round(2))


def check_bird_and_words():
    mesh = synthetic_fish()   # the shape is all that matters here
    v = np.asarray(mesh.vertices)
    root_z = (0 - v[:, 2].min()) / np.ptp(v[:, 2])
    # The flying skeleton has no pelvis: the tail hangs off root.
    out = add_tail(_joints(root_z, pelvis=False), mesh, 'blue parrot toy')
    assert {'parent': 'joint_root', 'child': 'joint_tail_base'} in out['skeleton']

    assert has_tail('cartoon pufferfish') and has_tail('two ducks')
    assert has_tail('blackbird') and has_tail('eel')
    for no_tail in ('cartoon bee toy', 'wheelbarrow', 'kitchen robot', 'race car'):
        assert not has_tail(no_tail), no_tail
    print('bird and word matching: ok')


def check_pufferfish():
    store = 'results/_pipeline_store.json'
    mesh_path = 'results/865fee5d/865fee5d_mesh.glb'
    if not (os.path.exists(store) and os.path.exists(mesh_path)):
        print('pufferfish: not in results/, skipped')
        return
    record = json.load(open(store))['865fee5d']
    mesh = trimesh.load(mesh_path, force='mesh')
    out = add_tail(copy.deepcopy(record['joints']), mesh,
                        record['classify']['object_type'])
    base, tip = tail_of(out)
    assert base and tip
    tip_w = world(mesh, tip)
    # Behind the body (its face is at z ≈ +0.95) and near the midline.
    assert tip_w[2] < -0.7 and abs(tip_w[0]) < 0.15, tip_w
    print('pufferfish: ok', world(mesh, base).round(2), tip_w.round(2))


if __name__ == '__main__':
    check_synthetic()
    check_bird_and_words()
    check_pufferfish()
