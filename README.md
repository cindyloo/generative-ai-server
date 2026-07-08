API Instructions: 
You need to set these exports in your environment:
FAL_KEY,
CLAUDE_API_KEY,
MESHY_API_KEY

And if using cloud db (advanced, otherwise we are just storing in server/results pipeline_store.json)

MODEL_STORE_BACKEND=clouddb
CLOUDDB_URL=https://clouddb.appinventor.mit.edu
CLOUDDB_TOKEN=your_token
CLOUDDB_PROJECT=your_project_id

PIPELINE
Begin with an image of something you want to animate
/segment
SEGMENT
`curl -s -X POST  <server>:6000/segment" --data-binary @assets/lego_truck.jpg -H "Content-Type: application/octet-stream" --output assets/lego_truck_segmented.png`
Segmentd that picture

CLASSIFY
`curl -s -X POST \\n  <server>:6000/classify?tag=a+lego+car&force=true" \\n  --data-binary @assets/lego_car_segmented.png`

returns
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","needs_augmentation":false,"object_type":"lego truck","rig_type":"vehicle","rigid_parts":["body","axle"],"segmented_image_path":"results/7b83d28d/7b83d28d_segmented.png","style":"physical LEGO toy, plastic brick construction, bright primary colors, hard-edged geometric shapes, studio white background photograph"}

and should describes what the object does, eg:
A walking piece of broccoli
A flying flower
A lego car
A two-legged dog statue
A bird

AUGMENT (optional, included as part of classification)
`curl -s -X POST <server>:6000/augment_image?classify_id=7b83d28d"`
POST /augment_image?classify_id=    generate two augmented variants via fal.ai
POST /augment_image/confirm?classify_id=&choice=a|b  lock in chosen variant
Use Claude (or any LLM) to augment the image in order to rig it properly for Meshy
As a humanoid (front, a-pose or t-pose_
As an animal (side)
As a vehicle (side)
As something mechanical (front)


CREATE MESH
`curl -s -X POST <server>:6000/mesh?classify_id=7b83d28d&joints=4&force=true"`
This will take several minutes
You should see something like
INFO:seg_server:Meshy 019ed5c4-d334-7bf4-a466-aedce08d8d8b: SUCCEEDED (100%)
INFO:seg_server:Downloading: https://assets.meshy.ai/260618c5-bbcd-4f54-a1d2-41e18c102816/tasks/019ed5c4-d334-7bf4-a466-aedce08d8d8b/output/model.glb?Expires=1781962317&Signature=OQL7Y~pITofDconNUMImgcsQAj0mSYn3Fh6JK0r3q4ucj0Suu9O2XDBMKbWtWMC6Le9arEwUDrcjJAJYfilMOKRMhC7e56Z0wgrt6PkGihBdLA53pD6DGvyULtyiWdC5VGU0MgdUfHg6qc5SVjKU7AN7ya0j7yAN~aCKxMXFJOLaXWkxege14-hB5omKZWC~~gp~2uQHJNe3qGVjLILfdqfGoGy5pPb891sZaCuGw5UAwKMJpJrd97PloB24nGuQgXzNpfrQIbP5Nvcizc57iqNKpuZjGKSDlQHBtLammGHDqRuCMp-MHsU9n2DzOKn7tiwVmLJlXgQhLxuQJ9svgQ__&Key-Pair-Id=K1VGYTHIYLM9UM
INFO:seg_server:Saved: results/7b83d28d/7b83d28d_mesh.glb
INFO:seg_server:Downloading: https://assets.meshy.ai/260618c5-bbcd-4f54-a1d2-41e18c102816/tasks/019ed5c4-d334-7bf4-a466-aedce08d8d8b/output/model.usdz?Expires=1781962317&Signature=Mz9l7byUgnV4a-kBDNEwenyzfXz5vqpepLzWl9f9-3JPL-rz08KDW2P9gc6GQcEXJkBsRq3l70YnxoIPIfRsok3yuRcyYkwBJ9-s3UkxHZI-c5JVr3eTWN38~~8JzccfvWWiqiw8yqNaa74LERfhq~ngPLTa0T1ZNIJduBqn2C1gFtCiJox2s7bnGGcB1~83dzu9~-4OFb~uV29ugaZRX-qp0qYixrPzcfEMXIH7sMZOqmfYUz-ePS71fsuZ81BJSa-~5f16MegBiIFChEbcpoxypD0yh7HxtBF10nSymkquuFdfwz~YryukVcr9RbmP-dKcTp7W3MgJXqNKGbID8A__&Key-Pair-Id=K1VGYTHIYLM9UM
INFO:seg_server:Saved: results/7b83d28d/7b83d28d_mesh.usdz
INFO:seg_server:Decimation complete: results/7b83d28d/7b83d28d_decimated.glb
INFO:seg_server:Decimated mesh: results/7b83d28d/7b83d28d_decimated.glb
INFO:seg_server:Mesh task f2e1b50b complete: results/7b83d28d/7b83d28d_mesh.glb


INFER JOINTS
`curl -s -X POST "<server>:6000/infer_joints?classify_id=1da7d94d&joints=4&force=true"`

You should see a response like   
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","joint_hints":[{...}] …}


CREATE RIGGED MESH
 `curl -s -X POST <server>:6000/rig?classify_id=1da7d94d&force=true"`
{"classify_id":"7b83d28d","status":"processing","task_id":"1d99d40e"}

Uses Claude joint positions as centroids when available 

Challenges:
Coordinate system issues
Claude ALWAYS uses x=left/right, y=front/rear, z=height, maps straight 1:1
trimesh and Blender have opposite Y/Z conventions:
trimesh: Y=height, Z=left-right
Blender: Y=left-right, Z=height

TL;DR
For better or worse, here is the research and decisions I made, with the resulting pipeline
With segmented picture from user, Claude determines if humanoid, animal, vehicle or other.
IF HUMANOID:
Claude determines if bipedal, with arms or legs
Claude determines if augmentation is necessary. T-pose or A-pose is enforced for humanoids. Side pose enforced if animal
Mesh is created
Using Claude gen’d pivot points, use Blender API to rig. Try to snap joints to edge of mesh (vs inside)

IF VEHICLE
Decision points: 
	How to find wheels in a mesh
	Does the mesh have loose/separate parts? From Meshy, no.
Generalized pipeline: Use Claude to identify wheel pivot points and wheel colors. Use estimated position from 2D to project onto 3D mesh, once created. Read GLB binary directly, sample texture at each vertex's UV coordinate, match against Gemini's wheel_colors_rgb palette. This became the foundation of the current approach.
