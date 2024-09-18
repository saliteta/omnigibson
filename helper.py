import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import math


CV2CG=torch.Tensor([
    [1,0,0],
    [0,0,-1],
    [0,-1,0]
]).cuda()

def get_intrinsic_matrix():
    K = np.zeros((3,3))
    w, h = 640, 480
    horiz_aperture = 20.955
    focal_length = 17.0
    horizontal_fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
    vertical_fov = horizontal_fov * h / w
    f_x = (w / 2.0) / np.tan(horizontal_fov / 2.0)
    f_y = (h / 2.0) / np.tan(vertical_fov / 2.0)
    K = np.array([
        [f_x, 0.0, w / 2.0],
        [0.0, f_y, h / 2.0],
        [0.0, 0.0, 1.0]
    ])

    return K

K = get_intrinsic_matrix()

def depth_to_point_cloud(depth_image_np:np.ndarray, K = K):
    """
    Converts a depth image to a point cloud using camera intrinsics.

    Parameters:
    depth_image (torch.Tensor): 2D tensor with depth values.
    K (torch.Tensor): Camera intrinsic matrix (3x3).

    Returns:
    points (torch.Tensor): Nx3 tensor with 3D points (X, Y, Z).
    """
    
    depth_image = torch.tensor(depth_image_np).to('cuda')
    dtype = depth_image.dtype
    height, width = depth_image.shape
    device = 'cuda'
    # Create meshgrid of pixel coordinates
    v_indices, u_indices = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing='ij'
    )

    # Flatten the arrays
    u = u_indices.flatten()
    v = v_indices.flatten()
    z = depth_image.flatten()

    # Filter out invalid depth values
    valid = ~torch.isinf(z)
    u = u[valid]
    v = v[valid]
    z = z[valid]

    # Extract intrinsic parameters
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Compute X, Y, Z coordinates
    X = (u - cx) * z / fx 
    Y = (v - cy) * z / fy
    Z = z

    # Stack to create Nx3 array of points
    points = torch.stack((X, Y, Z), dim=1)
    rotate = torch.matmul(points, CV2CG)
    return rotate

def transform_point_cloud(points, pose, orientation) -> np.ndarray:
    """
    Transforms a point cloud from camera coordinates to world coordinates.

    Parameters:
    points (torch.Tensor): Nx3 tensor of points in camera coordinates.
    pose (torch.Tensor): [x, y, z] translation vector.
    orientation (torch.Tensor): [x, y, z, w] quaternion.

    Returns:
    transformed_points (torch.Tensor): Nx3 tensor of points in world coordinates.
    """
    # Normalize the quaternion
    points = torch.Tensor(points).cuda()
    pose = torch.Tensor(pose).cuda()
    orientation = torch.Tensor(orientation).cuda()
    
    orientation = orientation / torch.norm(orientation)

    # Convert quaternion to rotation matrix
    qx, qy, qz, qw = orientation

    # Compute rotation matrix from quaternion
    R_quat = torch.tensor([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),         1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
    ], device=points.device, dtype=points.dtype)


    # Then apply rotation and translation
    rotated_points = torch.matmul(points, R_quat)
    #transformed_points = rotated_points + pose
    # pose need to permute (0,2,1)
    print(pose)
    pose = torch.tensor([pose[0],pose[2],pose[1]]).cuda()
    transformed_points = rotated_points + pose

    return transformed_points
