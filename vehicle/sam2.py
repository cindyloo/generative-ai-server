# segment_renders.py
import torch
import numpy as np
from PIL import Image
import json, os, sys
from pathlib import Path

# Install: pip install segment-anything
# Download checkpoint:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

render_dir  = sys.argv[1]   # /tmp/car_renders
output_dir  = sys.argv[2]   # /tmp/car_segments

os.makedirs(output_dir, exist_ok=True)

# Load SAM
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type     = "vit_b"
device         = "cuda" if torch.cuda.is_available() else "cpu"

sam  = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
    min_mask_region_area=500,
)

renders = ['front', 'right', 'left', 'back', 'top']
all_segments = {}

for view in renders:
    img_path = os.path.join(render_dir, f'{view}.png')
    if not os.path.exists(img_path):
        continue

    image = np.array(Image.open(img_path).convert('RGB'))
    masks = mask_generator.generate(image)

    print(f"\n{view}: {len(masks)} segments found")

    # Classify each mask as wheel or body based on:
    # - Position (wheels are near corners and low)
    # - Shape (wheels are roughly circular)
    # - Size (wheels are medium-sized, not too big or small)
    h, w   = image.shape[:2]
    labeled = []

    for i, mask in enumerate(masks):
        m    = mask['segmentation']
        area = mask['area']
        bbox = mask['bbox']

        ys, xs = np.where(m)
        cx = xs.mean() / w
        cy = ys.mean() / h

        bbox_w, bbox_h = bbox[2], bbox[3]
        aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0

        total_pixels  = h * w
        size_fraction = area / total_pixels

        is_corner = (cx < 0.35 or cx > 0.65) and cy > 0.40
        is_circular = aspect > 0.55                             # relaxed
        is_medium   = 0.005 < size_fraction < 0.30             # much more relaxed

        label = 'wheel' if (is_corner and is_circular and is_medium) else 'body'

        labeled.append({
            'id':       i,
            'label':    label,
            'centroid': [float(cx), float(cy)],
            'area':     area,
            'aspect':   float(aspect),
            'bbox':     list(bbox),
        })
        print(f"  [{i}] {label}: center=({cx:.2f},{cy:.2f}) "
              f"aspect={aspect:.2f} size={size_fraction:.3f} "
              f"corner={is_corner} circ={is_circular} med={is_medium}")
              
        all_segments[view] = labeled

print(f"About to save: {sum(len(v) for v in all_segments.values())} total segments")

with open(os.path.join(output_dir, 'segments.json'), 'w') as f:
    json.dump(all_segments, f, indent=2)

print(f"\nSegments saved to {output_dir}/segments.json")
