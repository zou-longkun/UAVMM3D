
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
import torch

def key_point_encoder(boxes9d, encode_corner=None, intrinsic_mat=None, extrinsic_mat=None, distortion_matrix=None, offset=np.array([-0.15, 0.05, 0])):

    corners_local = np.array(encode_corner, dtype=np.float32)

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

def key_point_decoder( encode_corner,
                       pred_heat_map=None, # 1,1,4,W,H
                       pred_res_x=None, # 1,1,4,W,H
                       pred_res_y=None, # 1,1,4,W,H
                       new_im_width=None,
                       new_im_hight=None,
                       raw_im_width=None,
                       raw_im_hight=None,
                       stride=None,
                       im_num=None,
                       obj_num=None,
                       size=None,
                       intrinsic=None,
                       distortion=None,
                       PnP_algo = 'SOLVEPNP_EPNP'):
    im_num = im_num
    obj_num = obj_num

    corners_local = np.array(encode_corner)

    key_pts_num = len(corners_local)

    key_points_2d = torch.zeros(im_num, obj_num, key_pts_num, 2)

    confidence = torch.zeros(im_num, obj_num)

    for im_id in range(im_num):
        for ob_id in range(obj_num):

            all_conf = []
            for k_id in range(key_pts_num):
                this_heatmap = pred_heat_map[im_id, ob_id, k_id]
                this_res_x = pred_res_x[im_id, ob_id, k_id]
                this_res_y = pred_res_y[im_id, ob_id, k_id]

                shape_map = this_heatmap.shape

                flat_x = this_heatmap.flatten()
                values, linear_indices = torch.topk(flat_x, k=1)

                c_num = shape_map[-1]
                rows = linear_indices // c_num
                cols = linear_indices % c_num

                row = rows[0]
                col = cols[0]
                conf = values[0]
                all_conf.append(conf)

                res_x = this_res_x[row.long(), col.long()]
                res_y = this_res_y[row.long(), col.long()]

                y_cor = row.float() + res_y
                x_cor = col.float() + res_x

                delta0 = (new_im_width / raw_im_width / stride)
                delta1 = (new_im_hight / raw_im_hight / stride)

                key_points_2d[im_id, ob_id, k_id, 0] = (x_cor / delta0)
                key_points_2d[im_id, ob_id, k_id, 1] = (y_cor / delta1)

            confidence[im_id, ob_id] = torch.mean(torch.stack(all_conf))

    key_points_2d = key_points_2d.cpu().numpy()
    confidence = confidence.cpu().numpy()

    key_points_2d = key_points_2d.reshape(obj_num, key_pts_num, 2),
    confidence = confidence.reshape(obj_num)


    # Scale the local corner points according to the target size
    corners_local = corners_local * size

    # Initialize the result list
    pred_boxes9d = []

    # Iterate over the image key points of each target
    for im_pts in key_points_2d:
        # Use the PnP algorithm to solve for the target's rotation vector and translation vector
        if PnP_algo == 'SOLVEPNP_EPNP':
            success, rvec, tvec = cv2.solvePnP(corners_local, np.array(im_pts, dtype=np.float64), intrinsic, distortion,flags=cv2.SOLVEPNP_EPNP)#, flags=cv2.SOLVEPNP_AP3P
        elif PnP_algo == 'SOLVEPNP_AP3P':
            success, rvec, tvec = cv2.solvePnP(corners_local, np.array(im_pts, dtype=np.float64), intrinsic, distortion,
                                               flags=cv2.SOLVEPNP_AP3P)  # , flags=cv2.SOLVEPNP_AP3P
        else:
            success, rvec, tvec = cv2.solvePnP(corners_local, np.array(im_pts, dtype=np.float64), intrinsic, distortion,
                                               flags=cv2.SOLVEPNP_ITERATIVE)  # , flags=cv2.SOLVEPNP_AP3P

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
            pred_boxes9d.append(box9d)
        else:
            # If PnP solving fails, return default values
            pred_boxes9d.append([0, 0, 0, size[0], size[1], size[2], 0, 0, 0])

    pred_boxes9d = np.array(pred_boxes9d)

    return key_points_2d, confidence, pred_boxes9d


all_key_point_encoders = {'key_point_encoder': key_point_encoder,
                          'key_point_decoder': key_point_decoder,}

