import cv2
import numpy as np


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), " [!] Batch heatmaps should be numpy.ndarray!"
    assert batch_heatmaps.ndim == 4, " [!] Batch images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, axis=2)  # get index
    maxvals = np.amax(heatmaps_reshaped, axis=2)  # get big value

    idx = idx.reshape((batch_size, num_joints, 1))
    maxvals = maxvals.reshape((batch_size, num_joints, 1))

    # from one-dimentional index to get 2-dimentional index
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    # maxvals should be bigger than 0
    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)
    preds *= pred_mask

    return preds, maxvals


def get_final_preds_align(config, batch_heatmaps, center, scale, warp_matrix):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(coords[n][p][0])  # (math.floor(coords[n][p][0] + 0.5))
                py = int(coords[n][p][1])  # int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

    # preds = coords.copy() * 4

    if warp_matrix[0] is not None:
        new_landmarks = coords.copy()
        for i in range(coords.shape[0]):
            inv_warp_matrix = np.linalg.inv(warp_matrix[i])
            data = np.expand_dims(coords[i, :, :2], axis=0) * 4
            landmarks2d = cv2.perspectiveTransform(data, inv_warp_matrix)[0]
            new_landmarks[i, :, :2] = landmarks2d.copy()
        preds = new_landmarks

    return preds, maxvals


def get_final_preds_from_regressor(config, batch_heatmaps, center, scale):
    coords = (batch_heatmaps + 0.5) * config.MODEL.IMAGE_SIZE  # (N, 21, 2)
    maxvals = np.ones(
        (batch_heatmaps.shape[0], config.MODEL.NUM_JOINTS, 1), dtype=np.float32
    )
    height, width = config.MODEL.IMAGE_SIZE

    preds = coords.copy()
    # Transform back
    for i in range(coords.shape[0]):  # batch-size
        preds[i] = transform_preds(coords[i], center[i], scale[i], [width, height])

    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(coords[n][p][0])  # (math.floor(coords[n][p][0] + 0.5))
                py = int(coords[n][p][1])  # int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

    preds = coords.copy()

    # if warp_matrix[0] is not None:
    #     new_landmarks = coords.copy()
    #     for i in range(coords.shape[0]):
    #         inv_warp_matrix = np.linalg.inv(warp_matrix[i])
    #         data = np.expand_dims(coords[i, :, :2], axis=0) * 4
    #         landmarks2d = cv2.perspectiveTransform(data, inv_warp_matrix)[0]
    #         new_landmarks[i, :, :2] = landmarks2d.copy() * 0.25
    #     coords = new_landmarks

    # Transform back
    for i in range(coords.shape[0]):  # batch-size
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals
