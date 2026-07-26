<b>How to animate anything!</b> 
<p><img width="300" height="200" alt="broccoli" src="https://github.com/user-attachments/assets/d4cc8a00-df4b-425d-996d-a2bf53978c59" />
 <img width="300" height="200" alt="Screenshot 2026-07-23 at 8 13 43 PM" src="https://github.com/user-attachments/assets/5f74de0b-6653-4d12-9b75-d5b6f9412e9b" />
<img width="300" height="200" alt="happy_broccoli_walking" src="https://github.com/user-attachments/assets/8edd0e55-a014-4bc1-a21c-0537c948b246" />
</p>

This api gives you an ability to start up a Flask server and create 3d animated models from one picture. <br/>

The picture you provide must clearly represent the object in full frame. Humans or humanoid objects need to be represented head to toe, ideally in a T or A pose, or facing "forward". Animals and vehicles need to be in full frame facing to either side (not facing the camera).<br/>


<br/>
<b>Getting Starting</b>
<br/>
You need obtain and set these exports in your environment:<br/>
FAL_KEY,<br/>
CLAUDE_API_KEY,<br/>
MESHY_API_KEY<br/>
<br/>
<br/>

Create a virtual env</br>
`source venv/bin/activate`
install required libs (there may be a few more you need to add)</br>
`pip install -r requirements.txt`
<br/>

Run the server (python 3.9 or higher)</br>
`python seg_server.py`
</br>
</br>
<p>If using cloud db (advanced, otherwise we are just storing in server/results pipeline_store.json)
MODEL_STORE_BACKEND=clouddb
CLOUDDB_URL=https://clouddb.appinventor.mit.edu
CLOUDDB_TOKEN=your_token
CLOUDDB_PROJECT=your_project_id </p>


***THE PIPELINE!***</br>
Begin with an image (as specified above) of something you want to animate. Create an assets folder (at the same level as vehicle and results) and drop it there


In curl commands, <server> denotes where you have set up the server, such as 128.31.33.33<br/>

In order to animate something, we need to first separate it from the background. This is the segmentation step.<br/>
<b>SEGMENT</b></br>
`curl -s -X POST  <server>:6000/segment" --data-binary @assets/lego_truck.jpg -H "Content-Type: application/octet-stream" --output assets/lego_truck_segmented.png`
Segmentd that picture

Next we need to classify the object, but we do that with a description of what we want the object to do or to be, such as "walking broccoli", "a lego truck", "a pink bike", "a two-legged dog statue"<br/>
<b>CLASSIFY</b></br>
`curl -s -X POST \\n  <server>:6000/classify?tag=a+lego+truck&force=true" \\n  --data-binary @assets/lego_truck_segmented.png`

returns
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","needs_augmentation":false,"object_type":"lego truck","rig_type":"vehicle","rigid_parts":["body","axle"],"segmented_image_path":"results/7b83d28d/7b83d28d_segmented.png","style":"physical LEGO toy, plastic brick construction, bright primary colors, hard-edged geometric shapes, studio white background photograph"}
</br>
</br>
Results should return a classify_id which is then used for the next successive commands. returned JSON should describe what the object does
</br>

Now it's time for the fun part. We ask Claude (or Gemini) to give us an image that add legs, arms, wings, wheels, etc. Try something fairly easy first and test what it does<br/>
<b>AUGMENT</b> (optional, included as part of classification)</br>
`curl -s -X POST <server>:6000/augment_image?classify_id=7b83d28d"`</br>
POST /augment_image?classify_id=    generate two augmented variants via fal.ai</br>
POST /augment_image/confirm?classify_id=&choice=a|b  lock in chosen variant</br>
Use Claude (or any LLM) to augment the image in order to rig it properly for Meshy</br>
As a humanoid (front, a-pose or t-pose_</br>
As an animal (side)</br>
As a vehicle (side)</br>
As something mechanical (front)</br>

You may want to run this a few times until you like what you get back - choice a or b<br/>
<br/>
Now ask Meshy to create your model for you!<br/>
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
<br/>
Meshy *can* infer joints, but it is limited. This next call uses Claude to infer where the joints are on the new model<br/>
<b>INFER JOINTS</b></br>
`curl -s -X POST "<server>:6000/infer_joints?classify_id=1da7d94d&joints=4&force=true"`
</br>
You should see a response like   
{"active_image_path":"results/7b83d28d/7b83d28d_segmented.png","augment_prompt":"","category":"vehicle","classify_id":"7b83d28d","joint_hints":[{...}] …}

<br/>
Lastly, super fun to see what you turn up with! you can drop your ###_rigged.glb or usdz file into bablylon at https://sandbox.babylonjs.com/ to view the animations<br/>
<b>CREATE RIGGED MESH</b></br>
 `curl -s -X POST <server>:6000/rig?classify_id=1da7d94d&force=true"`</br>
{"classify_id":"7b83d28d","status":"processing","task_id":"1d99d40e"}

Uses Claude joint positions as centroids when available </br>
<br/>
<br/>
<b>TLDR</b></br>
Challenges:</br>
Coordinate system issues</br>
Claude ALWAYS uses x=left/right, y=front/rear, z=height, maps straight 1:1</br>
trimesh and Blender have opposite Y/Z conventions:</br>
trimesh: Y=height, Z=left-right</br>
Blender: Y=left-right, Z=height</br>
</br>

For better or worse, here is the research and decisions I made, with the resulting pipeline<br/>

With segmented picture from user, Claude determines if humanoid, animal, vehicle or other.<br/>

IF HUMANOID:<br/>

Claude determines if bipedal, with arms or legs<br/>

Claude determines if augmentation is necessary. T-pose or A-pose is enforced for humanoids. Side pose enforced if animal<br/>

Mesh is created<br/>

Using Claude gen’d pivot points, use Blender API to rig. Try to snap joints to edge of mesh (vs inside)
</br>
IF VEHICLE<br/>

Decision points: <br/>
	How to find wheels in a mesh<br/>
	Does the mesh have loose/separate parts? From Meshy, no.<br/>

Generalized pipeline:<br/>
 Use Claude to identify wheel pivot points and wheel colors.<br/>
 Use estimated position from 2D to project onto 3D mesh, once created. <br/>
Read GLB binary directly, sample texture at each vertex's UV coordinate, match against Gemini's wheel_colors_rgb palette. <br/>
This became the foundation of the current approach.
