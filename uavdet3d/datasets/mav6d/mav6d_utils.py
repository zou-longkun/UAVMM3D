# -*- coding: utf-8 -*-
"""
@Author : Ye Zheng
@Contact : zhengye@westlake.edu.cn
"""
import json
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import copy
import cv2



def write_label2txt(file_path, label):
    data = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
        int(label[0]), label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9],
        label[10], label[11],
        label[12], label[13], label[14], label[15], label[16], label[17], label[18], label[19], label[20])
    # print(data)
    with open(file_path, "w") as f:
        f.write(data)
        # print(file_path)

    return


def read_truth_Rt(labpath):

    if os.path.getsize(labpath):
        with open(labpath, 'r') as f:
            contents = f.readlines()
        contents = contents[0].strip().split(' ')
        pose = list(map(lambda x: float(x), contents))
        # the transformation: MAV coordinate frame with respect to VICON coordinate frame
        uav_pose = pose[9:]

        # change the keypoints from MAV coordinate frame to VICON coordinate frame
        r = R.from_quat([uav_pose[-4], uav_pose[-3], uav_pose[-2], uav_pose[-1]])
        vicon2uav_rotation = r.as_matrix()
        vicon2uav_translation = np.array([[uav_pose[0]], [uav_pose[1]], [uav_pose[2]]])
        vicon2uav_T = np.concatenate((vicon2uav_rotation, vicon2uav_translation), axis=1)
        vicon2uav_T_h = np.concatenate((vicon2uav_T, np.array([[0, 0, 0, 1]])), axis=0)

        # transformation matrix obtained by reprojection error minimization method
        camera2vicon_T_h = np.array([[0.6685859, -0.74342, 0.01787715, -0.3814142759180792],
                                     [0.01558769, -0.01002444, -0.99982825, 1.604836888783723],
                                     [0.74347153, 0.66874974, 0.004886, 3.035574156448842],
                                     [0, 0, 0, 1]])

        # transformation matrix: MAV coordinate frame with respect to camera coordinate frame
        camera2uav_T_h = camera2vicon_T_h.dot(vicon2uav_T_h)
        rotation_mat = np.reshape(camera2uav_T_h[:3, :3], (9,))

        translation_mat = np.reshape(camera2uav_T_h[:3, 3] , (3,))

    return rotation_mat, translation_mat

def draw_box9d_on_image(boxes9d, image, img_width=1920., img_height=1080., color=(255, 0, 0), intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None, offset=np.array([0,0,0])):
    """
    Draw multiple 9-DoF 3D boxes on the image, considering camera distortion parameters.

    :param boxes9d: (N, 9) Each box's parameters are (x, y, z, l, w, h, angle1, angle2, angle3)
    :param image: (H, W, 3) Input image
    :param img_width: Image width
    :param img_height: Image height
    :param color: Box color (B, G, R)
    :param intrinsic_mat: (3, 3) Camera intrinsic matrix
    :param extrinsic_mat: (4, 4) Camera extrinsic matrix
    :param distortion_matrix: (5, ) Camera distortion parameters [k1, k2, p1, p2, k3]
    :return: Image with 3D boxes drawn (H, W, 3)
    """
    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)

    # Define the 8 corner points of the 3D box (relative to the center point)
    corners_local = np.array([
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5]
    ], dtype=np.float32)

    corners_local += offset

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], box[8]

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        # Convert 3D points to homogeneous coordinates
        corners_homogeneous = np.hstack((corners, np.ones((corners.shape[0], 1))))

        # Project onto the image plane (considering distortion)
        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        # Draw the edges of the 3D box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Side faces
        ]
        for edge in edges:
            start, end = edge
            cv2.line(image, tuple(corners_2d[start]), tuple(corners_2d[end]), color, 2)

    return image


def box9d_to_key_points(boxes9d, offset=np.array([-0.15, 0.05, 0])):

    corners_local = np.array([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
    ], dtype=np.float32)

    corners_local += offset

    all_corners = []

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        all_corners.append(corners)

    return np.array(all_corners)


def box9d_to_2d_key_points(boxes9d, intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None, radius=2, offset=np.array([-0.15, 0.05, 0])):

    corners_local = np.array([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
    ], dtype=np.float32)

    corners_local += offset

    all_corners_3d = []
    all_corners_2d = []

    for box in boxes9d:
        x, y, z, l, w, h, angle1, angle2, angle3 = box

        # Scale the corner points to match the box dimensions
        corners = corners_local * np.array([l, w, h])

        # Rotate the corner points
        rotation_matrix = R.from_euler('xyz', [angle1, angle2, angle3], degrees=False).as_matrix()
        corners = np.dot(corners, rotation_matrix.T)

        # Translate the corner points to the box center
        corners += np.array([x, y, z])

        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat,
                                          distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2)

        all_corners_3d.append(corners)
        all_corners_2d.append(corners_2d)

    return np.array(all_corners_3d), np.array(all_corners_2d)

def draw_points_on_image(points, image, img_width=1920., img_height=1080., color=(255, 0, 0), intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None, radius=2):
    """
    Draw multiple 9-DoF 3D boxes on the image, considering camera distortion parameters.

    :param boxes9d: (M, N, 3)
    :param image: (H, W, 3) Input image
    :param img_width: Image width
    :param img_height: Image height
    :param color: Box color (B, G, R)
    :param intrinsic_mat: (3, 3) Camera intrinsic matrix
    :param extrinsic_mat: (4, 4) Camera extrinsic matrix
    :param distortion_matrix: (5, ) Camera distortion parameters [k1, k2, p1, p2, k3]
    :return: Image with 3D boxes drawn (H, W, 3)
    """

    if intrinsic_mat is None:
        intrinsic_mat = np.array([[img_width, 0, img_width / 2],
                                  [0, img_height, img_height / 2],
                                  [0, 0, 1]], dtype=np.float32)

    if extrinsic_mat is None:
        extrinsic_mat = np.eye(4)

    if distortion_matrix is None:
        distortion_matrix = np.zeros(5, dtype=np.float32)


    for corners in points:

        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        for cor in corners_2d:
            cv2.circle(image, (cor[0], cor[1]), radius=radius, color=color, thickness=radius)

    return image




def PnP_im_points_to_boxes9d(im_points, size, intrinsic_mat=None, distortion_matrix=None):
    """
    Convert key points in the image to the 9D parameters of the target (x, y, z, l, w, h, angle1, angle2, angle3).

    :param im_points: (N,4,2) Key point coordinates in the image, N is the number of targets, each target has 4 key points
    :param l: Target length
    :param w: Target width
    :param h: Target height
    :param intrinsic_mat: (3,3) Camera intrinsic matrix
    :param distortion_matrix: (5,) Camera distortion coefficients
    :return: (N,9) Each box's parameters are (x, y, z, l, w, h, angle1, angle2, angle3)
    """
    # Define the 3D corner points of the target in the local coordinate system
    corners_local = np.array([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0],
    ], dtype=np.float64)

    # Scale the local corner points according to the target size
    corners_local = corners_local * size

    # Initialize the result list
    boxes9d = []

    # Iterate over the image key points of each target
    for im_pts in im_points:
        # Use the PnP algorithm to solve for the target's rotation vector and translation vector
        success, rvec, tvec = cv2.solvePnP(corners_local, np.array(im_pts, dtype=np.float64), intrinsic_mat, distortion_matrix, flags=cv2.SOLVEPNP_AP3P)

        if success:
            # Convert the rotation vector to a rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles (rotation angles around x, y, z axes) from the rotation matrix
            sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            angle1 = np.arctan2(R[2, 1], R[2, 2])  # Rotation around x-axis
            angle2 = np.arctan2(-R[2, 0], sy)  # Rotation around y-axis
            angle3 = np.arctan2(R[1, 0], R[0, 0])  # Rotation around z-axis

            # Combine the translation vector and rotation angles into 9D parameters
            box9d = [tvec[0, 0], tvec[1, 0], tvec[2, 0], size[0], size[1], size[2], angle1, angle2, angle3]
            boxes9d.append(box9d)
        else:
            # If PnP solving fails, return default values
            boxes9d.append([0, 0, 0, size[0], size[1], size[2], 0, 0, 0])

    return np.array(boxes9d)


def pts3d_to_im_pts(pts=None,
                         intrinsic_mat=None,
                         extrinsic_mat=None,
                         distortion_matrix=None):

    all_im_pts = []
    for corners in pts:
        corners_2d, _ = cv2.projectPoints(corners, extrinsic_mat[:3, :3], extrinsic_mat[:3, 3], intrinsic_mat, distortion_matrix)
        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        all_im_pts.append(corners_2d)

    return all_im_pts


def draw_2d_points_on_image(points, image,  color=(255, 0, 0), radius=2):
    """
    Draw multiple 9-DoF 3D boxes on the image, considering camera distortion parameters.

    :param boxes9d: (M, N, 3)
    :param image: (H, W, 3) Input image
    :param img_width: Image width
    :param img_height: Image height
    :param color: Box color (B, G, R)
    :param intrinsic_mat: (3, 3) Camera intrinsic matrix
    :param extrinsic_mat: (4, 4) Camera extrinsic matrix
    :param distortion_matrix: (5, ) Camera distortion parameters [k1, k2, p1, p2, k3]
    :return: Image with 3D boxes drawn (H, W, 3)
    """

    for corners_2d in points:

        corners_2d = corners_2d.reshape(-1, 2).astype(int)

        for cor in corners_2d:
            cv2.circle(image, (cor[0], cor[1]), radius=radius, color=color, thickness=radius)

    return image
