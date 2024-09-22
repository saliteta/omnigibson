from flask import Flask, request, jsonify, send_file, render_template
from scipy.spatial.transform import Rotation as R
from threading import Thread, Lock, Event
import queue
import omnigibson as og
from omnigibson.sensors import VisionSensor
from io import BytesIO
import numpy as np
import cv2

from rotation_utils import rotate_quaternion
import numpy as np
import torch
from helper import depth_to_point_cloud, transform_point_cloud
from config import cfg, move, rotation, start_pos



app = Flask(__name__)


        
env = og.Environment(cfg)

cam_mover = VisionSensor(
        prim_path="/World/viewer_camera",  # prim path
        name="my_vision_sensor",  # sensor name
        modalities=["rgb","depth_linear"],  # sensor mode
        enabled=True,  # enale sensor
        image_height=480,  # 
        image_width=640,  # 
        focal_length=17,  
        clipping_range=(0.01, 1000000.0),  # vision distance
    )

cam_mover.initialize()
# Allow camera teleoperation
current_pose=np.copy(start_pos)
_, current_orientation = cam_mover.get_position_orientation()
cam_mover.set_position_orientation(position=current_pose,orientation=current_orientation)



# Thread-safe queues and locks
camera_update_queue = queue.Queue()
cam_mover_lock = Lock()
image_lock = Lock()
image_ready_event = Event() 


# Define the environment's dimensions (in meters)

# Get the scene's bounding box


ENV_X_MIN = -30
ENV_X_MAX = 30
ENV_Y_MIN = -30
ENV_Y_MAX = 30
BEV_RESOLUTION = 0.1 # Meters per pixel
BEV_WIDTH = int((ENV_X_MAX - ENV_X_MIN) / BEV_RESOLUTION)
BEV_HEIGHT = int((ENV_Y_MAX - ENV_Y_MIN) / BEV_RESOLUTION)


# Shared variables for images
bev_map = torch.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=torch.uint8)  # 0: free, 255: occupied
current_rgb_image = None
current_depth_image = None
camera_path = []



current_position = np.copy(start_pos)
current_orientation = cam_mover.get_position_orientation()[1]
pose_lock = Lock()
bev_map_lock = Lock()
update_event = Event()

count = 0


def simulation_loop():
    global current_rgb_image, current_depth_image, camera_path, count
    while True:
        # Wait for an update from the Flask thread
        update_event.wait()

        with pose_lock:
            new_position = current_position.copy()
            new_orientation = current_orientation.copy()
            camera_path.append(new_position)

        with cam_mover_lock:
            cam_mover.set_position_orientation(position=new_position, orientation=new_orientation)
            update_environment()
            pose, orientation = cam_mover.get_position_orientation()
            np.allclose(pose, new_position, atol=1e-2) & np.allclose(orientation, new_orientation, atol=1e-2), \
            f"The camera setting is not correct, we have current_pose: {pose} set pose {new_position}, current_orientation: {orientation}, new orientation {new_orientation}"
            obs = cam_mover.get_obs()[0]
            rgb = obs["rgb"][..., :3][..., ::-1]
            depth = obs["depth_linear"]
            with image_lock:
                current_rgb_image = rgb.copy()
                current_depth_image = depth.copy()
                #np.savez_compressed(
                #    f"depth/depth_{count}",
                #    {
                #        'depth': current_depth_image,
                #        'pose': pose,
                #        'orientation': orientation
                #    }
                #)
                count += 1
                camera_points = depth_to_point_cloud(current_depth_image)
                world_points = transform_point_cloud(points=camera_points, pose = pose, orientation=orientation)
                update_bev_map(world_points)
         
                
        # Reset the events
        image_ready_event.set()
        update_event.clear()

def update_bev_map(points_world):
    global bev_map
    # Filter points with height (z) greater than 0.5 meters
    height_threshold = 0.5
    points_above_threshold = points_world[(points_world[:, 1] > height_threshold) & (points_world[:, 1] < 2)]

    # Convert world coordinates to BEV map indices
    x_indices = ((points_above_threshold[:, 0] - ENV_X_MIN) / BEV_RESOLUTION)
    y_indices = ((points_above_threshold[:, 2] - ENV_Y_MIN) / BEV_RESOLUTION)
    
    # Ensure indices are within the map boundaries
    valid_indices = (
        (x_indices >= 0) & (x_indices < BEV_WIDTH) &
        (y_indices >= 0) & (y_indices < BEV_HEIGHT)
    )
    x_indices = x_indices[valid_indices].long()
    y_indices = y_indices[valid_indices].long()

    # Update the BEV map
    with bev_map_lock:
        bev_map[y_indices, x_indices] = 255  # Mark as occupied

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_camera', methods=['POST'])
def update_camera():
    global current_position, current_orientation
    data = request.json
    key = data['key']

    with pose_lock:
        pos = current_position.copy()
        ori = current_orientation.copy()

    if key in move:
        new_position = position_change(key=key, pos=pos, ori=ori)
        new_orientation = ori
    elif key in rotation:
        new_orientation = orientation_change(ori, key_stroke=key)
        new_position = pos

    with pose_lock:
        current_position = new_position.copy()
        current_orientation = new_orientation.copy()
        position = current_position.tolist()
        orientation = current_orientation.tolist()

    # Signal the simulation loop that an update is available
    update_event.set()

    # Wait until the image is ready
    image_ready_event.wait()
    # clear image_ready for next processing
    image_ready_event.clear()
    
    
    return jsonify(success=True, position = position, orientation = orientation)

@app.route('/view_rgb')
def view_rgb():
    with image_lock:
        if current_rgb_image is not None:
            rgb = current_rgb_image.copy()
            cv2.imwrite('rgb.png', rgb)
        else:
            # Return a blank image or appropriate placeholder
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    # Encode the image to PNG
    _, encoded_image = cv2.imencode('.png', rgb)
    return send_file(BytesIO(encoded_image.tobytes()), mimetype='image/png')

@app.route('/view_depth')
def view_depth():
    with image_lock:
        if current_depth_image is not None:
            depth = current_depth_image.copy()
            
            # Create a mask for infinite values
            inf_mask = np.isinf(depth)
            
            # Replace infinite values with zero for visualization
            depth[inf_mask] = 0

            # Find the maximum depth value for normalization, excluding zeros
            max_depth = np.max(depth[depth > 0]) if np.any(depth > 0) else 1.0

            # Normalize depth for visualization (assuming depth is in meters)
            depth_normalized = (depth / max_depth * 255).astype(np.uint8)

            # Apply color map
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # Set areas where depth was infinite (now zero) to black
            depth_colored[inf_mask] = [0, 0, 0]
        else:
            # Return a blank image or appropriate placeholder
            depth_colored = np.zeros((480, 640, 3), dtype=np.uint8)

    # Encode the image to PNG
    _, encoded_image = cv2.imencode('.png', depth_colored)
    return send_file(BytesIO(encoded_image.tobytes()), mimetype='image/png')

@app.route('/view_bev')
def view_bev():
    with image_lock:
        bev_map_image = bev_map.cpu().numpy()
        print(bev_map.unique())
        # Optionally, convert to a color image for better visualization
        bev_map_color = cv2.cvtColor(bev_map_image, cv2.COLOR_GRAY2BGR)
        # You might want to flip or rotate the image to match coordinate conventions
        bev_map_color = cv2.flip(bev_map_color, 0)  # Flip vertically if needed

    _, encoded_image = cv2.imencode('.png', bev_map_color)
    return send_file(BytesIO(encoded_image.tobytes()), mimetype='image/png')

def update_environment():
    for _ in range(4): 
        env.step(np.array([0,0]))  # update the environment

def position_change(key, pos, ori):
    # Define movement step size
    step_size = 0.8

    # Define local movement vector based on key press
    if key == 'ArrowRight':
        local_move = np.array([step_size, 0, 0])  # Right
    elif key == 'ArrowLeft':
        local_move = np.array([-step_size, 0, 0])  # Left
    elif key == 'ArrowUp':
        local_move = np.array([0, 0, -step_size])  # Forward
    elif key == 'ArrowDown':
        local_move = np.array([0, 0, step_size])   # Backward
    elif key == ' ':
        local_move = np.array([0, step_size, 0])   # Up
    elif key == 'z':
        local_move = np.array([0, -step_size, 0])  # Down
    else:
        local_move = np.array([0, 0, 0])

    # Create a rotation object from the quaternion
    # Assuming ori is in [x, y, z, w] format
    rotation = R.from_quat([ori[0], ori[1], ori[2], ori[3]])

    # Rotate the local movement vector to global coordinate system
    global_move = rotation.apply(local_move)

    # Update position
    new_pos = pos + global_move

    return new_pos

def orientation_change(current_quat, key_stroke):
    if key_stroke == 'e':  # Pitch +10 degrees
        new_quat =  rotate_quaternion(current_quat, 'y', 10)
    elif key_stroke == 'q':  # Pitch -10 degrees
        new_quat =  rotate_quaternion(current_quat, 'y', -10)
    elif key_stroke == 'w':  # Roll +10 degrees
        new_quat =  rotate_quaternion(current_quat, 'x', 10)
    elif key_stroke == 's':  # Roll -10 degrees
        new_quat =  rotate_quaternion(current_quat, 'x', -10)
    elif key_stroke == 'a':  # Yaw +10 degrees
        new_quat =  rotate_quaternion(current_quat, 'z', 10)
    elif key_stroke == 'd':  # Yaw -10 degrees
        new_quat =  rotate_quaternion(current_quat, 'z', -10)
        
    if new_quat[0] < 0:
        new_quat = -new_quat
    return new_quat


if __name__ == '__main__':
    def run_flask_app():
        app.run(host='0.0.0.0', port=5000)
    
    
    # Start the Flask app in a separate thread
    flask_thread = Thread(target=run_flask_app)
    flask_thread.start()
    
    # Run the simulation loop in the main thread
    simulation_loop()