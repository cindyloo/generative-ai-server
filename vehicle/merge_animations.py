import sys, math
import struct, json

input_path  = sys.argv[1]
output_path = sys.argv[2]

with open(input_path, 'rb') as f:
    f.read(12)
    json_len = struct.unpack('<I', f.read(4))[0]
    f.read(4)
    j       = json.loads(f.read(json_len))
    bin_len = struct.unpack('<I', f.read(4))[0]
    f.read(4)
    binary  = f.read(bin_len)

anims = j.get('animations', [])
print(f"Input animations: {len(anims)}")
for a in anims:
    print(f"  {a['name']}: {len(a['channels'])} channels")

# Merge into one animation per clip. animatesam.py names each action
# "{clip}__{object}"; older per-wheel actions (wheel_0_rot, ...) are "drive".
clips = {}
for anim in anims:
    clip = anim['name'].split('__', 1)[0] if '__' in anim['name'] else 'drive'
    merged = clips.setdefault(clip, {'name': clip, 'channels': [], 'samplers': []})
    sampler_offset = len(merged['samplers'])
    merged['samplers'].extend(anim['samplers'])
    for ch in anim['channels']:
        merged['channels'].append({
            'sampler': ch['sampler'] + sampler_offset,
            'target':  ch['target'],
        })

j['animations'] = list(clips.values())

print(f"\nMerged into {len(clips)} animations: "
      + ", ".join(f"{c['name']} ({len(c['channels'])} channels)" for c in clips.values()))

# Write output GLB
json_bytes = json.dumps(j, separators=(',', ':')).encode('utf-8')
while len(json_bytes) % 4:
    json_bytes += b' '

total_len = 12 + 8 + len(json_bytes) + 8 + len(binary)

with open(output_path, 'wb') as f:
    f.write(b'glTF')
    f.write(struct.pack('<I', 2))
    f.write(struct.pack('<I', total_len))
    f.write(struct.pack('<I', len(json_bytes)))
    f.write(b'JSON')
    f.write(json_bytes)
    f.write(struct.pack('<I', len(binary)))
    f.write(b'BIN\x00')
    f.write(binary)

print(f"Written: {output_path}")
