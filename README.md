<b>How to animate anything!</b>

#API Instructions: 
You need obtain and set these exports in your environment:
FAL_KEY,
CLAUDE_API_KEY,
MESHY_API_KEY

create a virtual env</br>
`source venv/bin/activate`</br>
install required libs (there may be a few more you need to add)</br>
`pip install -r requirements.txt`</br>
`python seg_server.py`

If using cloud db (advanced, otherwise we are just storing in server/results pipeline_store.json)
MODEL_STORE_BACKEND=clouddb
CLOUDDB_URL=https://clouddb.appinventor.mit.edu
CLOUDDB_TOKEN=your_token
CLOUDDB_PROJECT=your_project_id


***THE PIPELINE!***</br>
Begin with an image of something you want to animate. Create an assets folder and drop it there


<server> denotes where you have set up the server. 
<b>SEGMENT</b></br>
`curl -s -X POST  <server>:6000/segment" --data-binary @assets/lego_truck.jpg -H "Content-Type: application/octet-stream" --output assets/lego_truck_segmented.png`
Segmentd that picture

<b>CLASSIFY</b></br>
`curl -s -X POST \\n  <server>:6000/classify?tag=a+lego+truck&force=true" \\n  --data-binary @assets/lego_truck_segmented.png`

returns
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","needs_augmentation":false,"object_type":"lego truck","rig_type":"vehicle","rigid_parts":["body","axle"],"segmented_image_path":"results/7b83d28d/7b83d28d_segmented.png","style":"physical LEGO toy, plastic brick construction, bright primary colors, hard-edged geometric shapes, studio white background photograph"}
</br>
</br>
results should return a classify_id which is then used for the next successive commands. returned JSON should describe what the object does, eg:</br>
A walking piece of broccoli</br>
A flying flower</br>
A lego car</br>
A two-legged dog statue</br>
A bird</br>

<b>AUGMENT</b> (optional, included as part of classification)</br>
`curl -s -X POST <server>:6000/augment_image?classify_id=7b83d28d"`</br>
POST /augment_image?classify_id=    generate two augmented variants via fal.ai</br>
POST /augment_image/confirm?classify_id=&choice=a|b  lock in chosen variant</br>
Use Claude (or any LLM) to augment the image in order to rig it properly for Meshy</br>
As a humanoid (front, a-pose or t-pose_</br>
As an animal (side)</br>
As a vehicle (side)</br>
As something mechanical (front)</br>


<b>CREATE MESH</b></br>
`curl -s -X POST <server>:6000/mesh?classify_id=7b83d28d&joints=4&force=true"`
This will take several minutes
You should see something like
INFO:seg_server:Meshy 019ed5c4-d334-7bf4-a466-aedce08d8d8b: SUCCEEDED (100%)
INFO:seg_server:Downloading: https://assets.meshy.ai/260618c5-bbcd-4f54-a1d2-41e18c102816/tasks/019ed5c4-d334-7bf4-a466-aedce08d8d8b/output/model.glb?Expires=###
INFO:seg_server:Saved: results/7b83d28d/7b83d28d_mesh.usdz
INFO:seg_server:Decimation complete: results/7b83d28d/7b83d28d_decimated.glb
INFO:seg_server:Decimated mesh: results/7b83d28d/7b83d28d_decimated.glb
INFO:seg_server:Mesh task f2e1b50b complete: results/7b83d28d/7b83d28d_mesh.glb


<b>INFER JOINTS</b></br>
`curl -s -X POST "<server>:6000/infer_joints?classify_id=1da7d94d&joints=4&force=true"`
</br>
You should see a response like   
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","joint_hints":[{...}] …}


<b>CREATE RIGGED MESH</b></br>
 `curl -s -X POST <server>:6000/rig?classify_id=1da7d94d&force=true"`</br>
{"classify_id":"7b83d28d","status":"processing","task_id":"1d99d40e"}

Uses Claude joint positions as centroids when available </br>

Challenges:</br>
Coordinate system issues</br>
Claude ALWAYS uses x=left/right, y=front/rear, z=height, maps straight 1:1</br>
trimesh and Blender have opposite Y/Z conventions:</br>
trimesh: Y=height, Z=left-right</br>
Blender: Y=left-right, Z=height</br>
</br>
TL;DR
For better or worse, here is the research and decisions I made, with the resulting pipeline
With segmented picture from user, Claude determines if humanoid, animal, vehicle or other.
IF HUMANOID:
Claude determines if bipedal, with arms or legs
Claude determines if augmentation is necessary. T-pose or A-pose is enforced for humanoids. Side pose enforced if animal
Mesh is created
Using Claude gen’d pivot points, use Blender API to rig. Try to snap joints to edge of mesh (vs inside)
</br>
IF VEHICLE
Decision points: 
	How to find wheels in a mesh
	Does the mesh have loose/separate parts? From Meshy, no.
Generalized pipeline: Use Claude to identify wheel pivot points and wheel colors. Use estimated position from 2D to project onto 3D mesh, once created. Read GLB binary directly, sample texture at each vertex's UV coordinate, match against Gemini's wheel_colors_rgb palette. This became the foundation of the current approach.
