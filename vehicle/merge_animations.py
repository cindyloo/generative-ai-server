import sys, math
sys.path.insert(0, '/tmp/blender_packages')
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

# Merge all into one animation named "drive"
merged_channels = []
merged_samplers = []

for anim in anims:
    sampler_offset = len(merged_samplers)
    merged_samplers.extend(anim['samplers'])
    for ch in anim['channels']:
        new_ch = {
            'sampler': ch['sampler'] + sampler_offset,
            'target':  ch['target']
        }
        merged_channels.append(new_ch)

j['animations'] = [{
    'name':     'drive',
    'channels': merged_channels,
    'samplers': merged_samplers,
}]

print(f"\nMerged into 1 animation 'drive' with {len(merged_channels)} channels")

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
