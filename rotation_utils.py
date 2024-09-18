# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utilities for generating and applying rotation matrices.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

ANGLE_EPS = 0.001


def normalize(v):
    return v / np.linalg.norm(v)


def get_r_matrix(ax_, angle):
    ax = normalize(ax_)
    if np.abs(angle) > ANGLE_EPS:
        S_hat = np.array(
            [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
            dtype=np.float32)
        R = np.eye(3) + np.sin(angle) * S_hat + \
            (1 - np.cos(angle)) * (np.linalg.matrix_power(S_hat, 2))
    else:
        R = np.eye(3)
    return R


def r_between(v_from_, v_to_):
    v_from = normalize(v_from_)
    v_to = normalize(v_to_)
    ax = normalize(np.cross(v_from, v_to))
    angle = np.arccos(np.dot(v_from, v_to))
    return get_r_matrix(ax, angle)


def rotate_camera_to_point_at(up_from, lookat_from, up_to, lookat_to):
    inputs = [up_from, lookat_from, up_to, lookat_to]
    for i in range(4):
        inputs[i] = normalize(np.array(inputs[i]).reshape((-1,)))
    up_from, lookat_from, up_to, lookat_to = inputs
    r1 = r_between(lookat_from, lookat_to)

    new_x = np.dot(r1, np.array([1, 0, 0]).reshape((-1, 1))).reshape((-1))
    to_x = normalize(np.cross(lookat_to, up_to))
    angle = np.arccos(np.dot(new_x, to_x))
    if angle > ANGLE_EPS:
        if angle < np.pi - ANGLE_EPS:
            ax = normalize(np.cross(new_x, to_x))
            flip = np.dot(lookat_to, ax)
            if flip > 0:
                r2 = get_r_matrix(lookat_to, angle)
            elif flip < 0:
                r2 = get_r_matrix(lookat_to, -1. * angle)
        else:
            # Angle of rotation is too close to 180 degrees, direction of
            # rotation does not matter.
            r2 = get_r_matrix(lookat_to, angle)
    else:
        r2 = np.eye(3)
    return np.dot(r2, r1)


def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ]

def normalize_quaternion(q):
    """Normalizes a quaternion to unit length."""
    norm = np.linalg.norm(q)
    return q / norm

def euler_to_quaternion(angle):
    """Converts Euler angles (in degrees) to a quaternion."""
    r = R.from_euler('xyz', angle, degrees=True)
    return r.as_quat()  # returns quaternion in the format [x, y, z, w]

def quaternion_to_euler(quaternion):
    # Create a quaternion
    
    # Convert to rotation object
    rotation = R.from_quat(quaternion)
    
    # Convert to Euler angles (using the 'xyz' convention as an example)
    euler_angles = rotation.as_euler('xyz', degrees=True)  # Return angles in degrees
    
    return euler_angles

def rotate_quaternion(current_quat, axis, angle_degrees=10):
    """Rotates the current quaternion around the specified axis by a given angle."""
    # first covert current quaternion to eular angle
    
    current_angle = quaternion_to_euler(current_quat)

    if axis == 'x':
        current_angle[0] = (current_angle[0]+angle_degrees)%360
    elif axis == 'y':
        current_angle[1] = (current_angle[1]+angle_degrees)%360
    elif axis == 'z':
        current_angle[2] = (current_angle[2]+angle_degrees)%360
    else:
        raise ValueError("Invalid axis! Use 'x', 'y', or 'z'.")

    # Multiply the current quaternion by the rotation quaternion
    new_quat = euler_to_quaternion(current_angle)
    return new_quat