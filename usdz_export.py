"""
usdz_export.py  (Blender script)

    blender --background --python usdz_export.py -- \
        <in.glb> <out.usdz> <decimate_ratio|none> <clip,order,...>

Converts a GLB to a Y-up USDZ (like glTF, and like the USDZ Meshy returns,
rather than Blender's Z-up).

USD has no named clips: a stage has one timeline, and AR Quick Look loops
it. So every clip in the GLB is baked onto that timeline back to back,
the first of the given clip order that the GLB has leading (it is what
Quick Look shows first), the rest in the GLB's order. The frame ranges go
in the root layer's customLayerData under "clips", and are printed as
USDZ_CLIPS <json> for the server. A RealityKit app can play any one clip
by trimming the timeline to its range.

Baking, not NLA strips laid end to end: when no strip drives a channel,
Blender leaves whatever the last strip wrote there, so a clip that doesn't
move the body (drive) would inherit the previous clip's last pose. Each
clip is played alone from the rest pose instead, on the objects that have
a slot in it, and every animated object's pose recorded on every frame.

decimate_ratio (0-1) collapses every mesh to that fraction of its faces
first, for rigs built on a full-resolution mesh (vehicles and mechanical
objects); creature rigs are built on the decimated mesh already.
"""
import json
import os
import sys
import tempfile

import bpy
from mathutils import Matrix

argv = sys.argv[sys.argv.index('--') + 1:]
glb_path, usdz_path = argv[0], argv[1]
ratio      = None if argv[2] == 'none' else float(argv[2])
clip_order = [c for c in argv[3].split(',') if c]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb_path)
scene = bpy.context.scene

# ── Bake every clip onto one timeline ─────────────────────────────────────────
animated = [o for o in bpy.data.objects
            if o.animation_data and (o.animation_data.action or o.animation_data.nla_tracks)]
actions  = list(bpy.data.actions)
lead     = next((a for n in clip_order for a in actions if a.name == n), None)
order    = ([lead] if lead else []) + [a for a in actions if a is not lead]

for o in animated:
    for track in o.animation_data.nla_tracks:
        track.mute = True
    o.animation_data.action = None
scene.frame_set(1)

rest_obj  = {o.name: o.matrix_basis.copy() for o in animated}
is_arm    = {o.name: o.type == 'ARMATURE' for o in animated}


def slot_for(action, obj):
    """The action's slot for obj, or None if the clip doesn't move it."""
    slots = getattr(action, 'slots', None)
    if slots is None:                      # Blender < 4.4: one object per action
        return True
    return next((s for s in slots if s.identifier == 'OB' + obj.name), None)


def reset():
    for o in animated:
        o.animation_data.action = None
        o.matrix_basis = rest_obj[o.name]
        if is_arm[o.name]:
            for pb in o.pose.bones:
                pb.matrix_basis = Matrix.Identity(4)


samples = []   # (timeline frame, {object: matrix_basis}, {armature: {bone: matrix_basis}})
ranges  = []
frame   = 1
for act in order:
    start, end = (int(round(x)) for x in act.frame_range)
    reset()
    for o in animated:
        slot = slot_for(act, o)
        if slot:
            o.animation_data.action = act
            if slot is not True:
                o.animation_data.action_slot = slot
    for f in range(start, end + 1):
        scene.frame_set(f)
        samples.append((frame + f - start,
                        {o.name: o.matrix_basis.copy() for o in animated},
                        {o.name: {pb.name: pb.matrix_basis.copy() for pb in o.pose.bones}
                         for o in animated if is_arm[o.name]}))
    ranges.append({'name': act.name, 'start': frame, 'end': frame + end - start})
    frame += end - start + 1
reset()

for o in animated:
    o.animation_data.action = None
    for track in list(o.animation_data.nla_tracks):
        o.animation_data.nla_tracks.remove(track)
    baked = bpy.data.actions.new(f'usdz_{o.name}')
    o.animation_data.action = baked
    o.rotation_mode = 'QUATERNION'
    bones = list(o.pose.bones) if is_arm[o.name] else []
    for pb in bones:
        pb.rotation_mode = 'QUATERNION'
    for f, objs, arms in samples:
        o.matrix_basis = objs[o.name]
        for path in ('location', 'rotation_quaternion', 'scale'):
            o.keyframe_insert(path, frame=f)
        for pb in bones:
            pb.matrix_basis = arms[o.name][pb.name]
            for path in ('location', 'rotation_quaternion', 'scale'):
                pb.keyframe_insert(path, frame=f)

if ranges:
    scene.frame_start, scene.frame_end = 1, frame - 1
print('USDZ clips:', ', '.join(f"{r['name']} {r['start']}-{r['end']}" for r in ranges))

# ── Decimate ──────────────────────────────────────────────────────────────────
if ratio:
    for obj in [o for o in bpy.data.objects if o.type == 'MESH']:
        mod = obj.modifiers.new('usdz_decimate', 'DECIMATE')
        mod.ratio = ratio
        with bpy.context.temp_override(object=obj, active_object=obj):
            bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)
            bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f'USDZ decimated to {ratio:.0%} of faces')

# ── Export: usdc, stamp the clip ranges, package as usdz ──────────────────────
from pxr import Gf, Sdf, UsdUtils

with tempfile.TemporaryDirectory() as tmp:
    usdc = os.path.join(tmp, os.path.splitext(os.path.basename(usdz_path))[0] + '.usdc')
    bpy.ops.wm.usd_export(
        filepath=usdc,
        export_animation=bool(ranges),
        export_armatures=True,
        export_textures=True,
        export_materials=True,
        convert_orientation=True,
        export_global_forward_selection='NEGATIVE_Z',
        export_global_up_selection='Y',
    )
    layer = Sdf.Layer.FindOrOpen(usdc)
    data = dict(layer.customLayerData)
    data['clips'] = {r['name']: Gf.Vec2i(r['start'], r['end']) for r in ranges}   # (first, last) frame
    layer.customLayerData = data
    layer.Save()
    if os.path.exists(usdz_path):
        os.unlink(usdz_path)
    if not UsdUtils.CreateNewARKitUsdzPackage(Sdf.AssetPath(usdc), usdz_path):
        UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(usdc), usdz_path)

print('USDZ_CLIPS ' + json.dumps(ranges))
print('USDZ export done')
