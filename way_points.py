import omnigibson as og
from pathlib import Path
import matplotlib.pyplot as plt
from omnigibson.sensors import VisionSensor
from omnigibson.maps.traversable_map import TraversableMap
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from rotation_utils import rotate_quaternion
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import medial_axis
import sknw
import networkx as nx

import numpy as np
from tqdm import tqdm
import logging

import logging

# Create a logger
logger = logging.getLogger('example_logger')
logger.setLevel(logging.INFO)  # Set the logging level to INFO

# Create a file handler
file_handler = logging.FileHandler('logging.log')
file_handler.setLevel(logging.INFO)  # Set the handler's level to INFO

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


# Initialize the environment with the desired scene
cfg = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "floor_plane_visible": True,
        "scene_model": "Beechwood_0_garden"
    }
}

SAVING_PATH = '/butian/RESULT/'


def visualization(trav_map: TraversableMap, map_visualization_location: Path):

    # Ensure that the map is loaded
    if not trav_map.floor_map:
        print("Traversability map is not loaded.")
    else:
        # For each floor map, display it
        for floor_num, floor_map in enumerate(trav_map.floor_map):
            plt.figure(figsize=(8, 8))
            plt.imshow(floor_map, cmap='gray')
            plt.title(f"Traversability Map - Floor {floor_num}")
            plt.axis('off')
            plt.savefig(map_visualization_location)

def orientation_prototype(current_quat: np.ndarray):
    rotation_list = []
    rotation_list.append(current_quat)
    for i in range(3):
        rotation_list.append(rotate_quaternion(rotation_list[i], 'z', 90))
    return np.array(rotation_list)

def camera_saving(camera_mover:VisionSensor, 
                  env: og.Environment, 
                  depth_saving_path: Path, 
                  RGB_saving_path: Path, 
                  position: np.ndarray,
                  orienation:np.ndarray,
                  ):
    # update camera mover location
    camera_mover.set_position_orientation(
        position=position,
        orientation=orienation
    )
    for _ in range(4): 
        env.step(np.array([0,0]))  # update the environment
    current_pose, current_orienation = camera_mover.get_position_orientation()
    np.allclose(position, current_pose, atol=1e-2) & np.allclose(orienation, current_orienation, atol=1e-2), \
    f"The camera setting is not correct, we have current_pose: {current_pose} set pose {position}, current_orientation: {current_orienation}, new orientation {orienation}"
    obs = camera_mover.get_obs()[0]
    
    cv2.imwrite(RGB_saving_path, obs["rgb"][..., :3][..., ::-1])
    saved_dict = {
        'position' : position,
        'orientation' : orienation,
        'depth' : obs["depth_linear"],
    }
    np.savez_compressed(depth_saving_path, **saved_dict)

def exploration_graph(map: np.ndarray, 
                      robotics_radius: float, 
                      map_resolution: float = 0.1,
                      plot_saving_location: str = None, 
                      visualization: bool = False) -> nx.Graph:
    """
    Generates a Voronoi graph for exploration, considering the robot's size, and plots it over the traversable map.

    Args:
        map (np.ndarray): 2D array representing the traversable map (0: obstacle, 255: free space).
        robotics_radius (float): Robot's radius in meters.
        plot_saving_location (str): File path to save the plotted graph.
        map_resolution (float): Map resolution in meters per pixel.
    
    Returns: 
        mst: The minimal spanning tree generated by the map we give
    """
    # Step 1: Convert map to binary image (obstacles: 0, free space: 1)
    _, binary_map = cv2.threshold(map, 127, 1, cv2.THRESH_BINARY)
    
    # Step 2: Dilate obstacles to account for robot's size
    radius_pixels = int(np.ceil(robotics_radius / map_resolution))
    kernel_size = 2 * radius_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_obstacles = cv2.dilate(1 - binary_map, kernel)
    
    # Step 3: Eroded free space (after accounting for robot's size)
    eroded_free_space = 1 - dilated_obstacles
    
    # Step 4: Compute the skeleton (medial axis) of the free space
    skeleton = medial_axis(eroded_free_space).astype(np.uint8)
    
    # Step 5: Build graph from skeleton
    graph = sknw.build_sknw(skeleton)
    
    # Step 6: Compute the minimum spanning tree (MST) to avoid loops
    mst:nx.Graph = nx.minimum_spanning_tree(graph)
    
    logger.info(f"we have {len(mst.nodes)} nodes, {len(mst.edges)} edges, in the minimal spanning tree")
    
    if visualization:
        # Step 7: Plot the MST over the original map
        plt.figure(figsize=(10, 10))
        plt.imshow(map, cmap='gray')

        # Draw edges
        for (s, e) in mst.edges():
            ps = np.array(mst[s][e]['pts'])
            plt.plot(ps[:, 1], ps[:, 0], 'r')

        # Draw nodes
        nodes = mst.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 1], ps[:, 0], 'bo')

        plt.axis('off')
        plt.savefig(plot_saving_location, bbox_inches='tight')
        plt.close()
        

    return mst

def generate_key_points(mst: nx.Graph, 
                       resolution: float = 1.0,
                       map_resolution: float = 0.1) -> np.ndarray:
    """
    Generates key points from the MST, including nodes and sampled points on edges.

    Args:
        mst (networkx.Graph): Minimum spanning tree graph.
        resolution (float): Desired distance between key points in meters.

    Returns:
        np.ndarray: Array of shape (N, 2) with (x, y) real-world coordinates.
    """
    key_points = []

    # Step 1: Add all nodes to key points
    for node_id, data in mst.nodes(data=True):
        # Assuming 'o' contains the (row, col) pixel coordinates
        x_pixel, y_pixel = data['o']
        key_points.append([x_pixel, y_pixel])

    # Step 2: Add sampled points along each edge
    for (start_node, end_node, _) in mst.edges(data=True):
        start_pos = np.array(mst.nodes[start_node]['o'])
        end_pos = np.array(mst.nodes[end_node]['o'])
        
        assert np.all(start_pos < 1000) and np.all(start_pos > -1000), "Strange start pose"
        assert np.all(end_pos < 1000) and np.all(end_pos > -1000), "Strange end pose"
        
        # Convert pixel coordinates to real-world coordinates (meters)
        start_pos_meters = start_pos * map_resolution
        end_pos_meters = end_pos * map_resolution

        # Calculate Euclidean distance between nodes
        distance = np.linalg.norm(start_pos_meters -end_pos_meters)

        # Determine the number of additional points to sample
        num_additional_points = int(np.floor(distance * map_resolution / resolution)) - 1
        if num_additional_points > 0:
            for i in range(1, num_additional_points + 1):
                # Calculate the interpolation factor
                t = i / (num_additional_points + 1)
                # Compute interpolated point coordinates
                interp_pos = start_pos + t * (end_pos - start_pos)
                key_points.append(interp_pos.tolist())

    # Step 3: Convert to NumPy array and remove duplicates
    key_points = np.array(key_points)
    key_points = np.unique(key_points, axis=0)

    return key_points

def main():
    # Initialize the environment
    env = og.Environment(configs=cfg)

    # Get the scene object
    scene: InteractiveTraversableScene = og.sim.scene

    cam_mover = VisionSensor(
            prim_path="/World/viewer_camera",  # prim path
            name="my_vision_sensor",  # sensor name
            modalities=["rgb","depth_linear"],  # sensor mode
            enabled=True,  # enale sensor
            image_height=480,  # 
            image_width=640,  # 
            focal_length=17,  
            clipping_range=(0.01, 1000000.0),  # vision distance,
        )
    cam_mover.initialize()
    _, current_orientation = cam_mover.get_position_orientation()
    # we need to set 6 degree orientations
    orientation_list = orientation_prototype(current_orientation)


    # Get the traversability map
    trav_map: TraversableMap = scene.trav_map
    floor_map = trav_map.floor_map[0]
    logger.info(f"This is the synthetic map resolution{trav_map.map_resolution}")
    mst = exploration_graph(map=floor_map, robotics_radius=0.5, map_resolution=trav_map.map_resolution, visualization=True, plot_saving_location='voronoi.png')
    points_location_xy = generate_key_points(mst=mst, map_resolution=trav_map.map_resolution)

    
    points_location_xy = trav_map.map_to_world(points_location_xy)
    
    points_location = np.hstack((points_location_xy, np.full((points_location_xy.shape[0],1), 1.2)))
    
    count = 0
    for location in tqdm(points_location):
        for orientation in orientation_list:
            depth_saving_path = Path(SAVING_PATH+f'/Depth/{count}.npz')
            RGB_saving_path = SAVING_PATH+f'/RGB/{count}.png'
            camera_saving(camera_mover=cam_mover, 
                          env=env, 
                          depth_saving_path=depth_saving_path,
                          RGB_saving_path=RGB_saving_path,
                          position=location, 
                          orienation=orientation)
            count += 1

if __name__ == "__main__":
    main()


