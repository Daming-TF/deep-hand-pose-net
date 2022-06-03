import cv2
import math
import numpy as np


def get_rotation_matrix(radian):
    matrix = np.array(
        [
            [math.cos(radian), math.sin(radian), 0],
            [-math.sin(radian), math.cos(radian), 0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix


def get_translation_matrix(x, y):
    matrix = np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float32)
    return matrix


class NormalizedRect(object):
    def __init__(self):
        self.x_center = None
        self.y_center = None
        self.width = None
        self.height = None
        self.rotation = None

    def set_x_center(self, value):
        self.x_center = value

    def set_y_center(self, value):
        self.y_center = value

    def set_width(self, value):
        self.width = value

    def set_height(self, value):
        self.height = value

    def set_rotation(self, value):
        self.rotation = value


class HandInfo(object):
    def __init__(self, img_size):
        self.img_size = img_size
        self.num_joints_subset = 11
        self.landmark_subset = None
        self.kWristJoint = 0
        self.kIndexFingerPIPJoint = 4
        self.kMiddleFingerPIPJoint = 6
        self.kRingFingerPIPJoint = 8
        self.kTargetAngle = math.pi * 0.5  # 90 degree represented in radian
        self.shift_x = 0.0
        self.shift_y = -0.2  # -0.1 || -0.5
        self.scale_x = 2.1  # 2.0 || 2.6
        self.scale_y = 2.1  # 2.0 || 2.6
        self.square_long = True
        self.warp_matrix = None
        self.rect_roi_coord = None

    def __call__(self, img, landmark):
        rect = self.normalized_landmarks_list_to_rect(
            landmark, img_size=(img.shape[1], img.shape[0])
        )
        self.rect = self.rect_transformation(
            rect, img_width=img.shape[1], img_height=img.shape[0]
        )
        img_roi = self.get_rotated_rect_roi(img)
        return img_roi, self.warp_matrix

    def get_rotated_rect_roi(self, img):
        img_h, img_w = img.shape[0], img.shape[1]
        x_center = int(self.rect.x_center * img_w)
        y_center = int(self.rect.y_center * img_h)
        height = int(self.rect.height * img_h)
        half = 0.5 * height
        rotation_radian = -1.0 * self.rect.rotation

        rotate_matrix = get_rotation_matrix(rotation_radian)
        translation_matrix = get_translation_matrix(-x_center, -y_center)
        coords = np.array(
            [
                [x_center - half, x_center + half, x_center - half, x_center + half],
                [y_center - half, y_center - half, y_center + half, y_center + half],
                [1, 1, 1, 1],
            ]
        )

        # rotate * translation * coordinates
        result = np.matmul(rotate_matrix, np.matmul(translation_matrix, coords))

        pt1 = (int(result[0, 0] + x_center), int(result[1, 0] + y_center))
        pt2 = (int(result[0, 1] + x_center), int(result[1, 1] + y_center))
        pt3 = (int(result[0, 2] + x_center), int(result[1, 2] + y_center))
        pt4 = (int(result[0, 3] + x_center), int(result[1, 3] + y_center))

        spts = np.float32(
            [
                [pt1[0], pt1[1]],  # left-top
                [pt2[0], pt2[1]],  # right-top
                [pt3[0], pt3[1]],  # left-bottom
                [pt4[0], pt4[1]],
            ]
        )  # right-bottom
        dpts = np.float32(
            [
                [0, 0],  # left-top
                [self.img_size - 1, 0],  # right-top
                [0, self.img_size - 1],  # left-bottm
                [self.img_size - 1, self.img_size - 1],
            ]
        )  # right-bottom
        self.warp_matrix = cv2.getPerspectiveTransform(spts, dpts)
        img_roi = cv2.warpPerspective(
            img,
            self.warp_matrix,
            (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
        )

        self.rect_roi_coord = spts.copy()

        return img_roi

    def rect_transformation(self, rect, img_width, img_height):
        width = rect.width
        height = rect.height
        rotation = rect.rotation

        if rotation == 0.0:
            rect.set_x_center(rect.x_center + width * self.shift_x)
            rect.set_y_center(rect.y_center + height * self.shift_y)
        else:
            x_shift = (
                img_width * width * self.shift_x * math.cos(rotation)
                - img_height * height * self.shift_y * math.sin(rotation)
            ) / img_width
            y_shift = (
                img_width * width * self.shift_x * math.sin(rotation)
                + img_height * height * self.shift_y * math.cos(rotation)
            ) / img_height

            rect.set_x_center(rect.x_center + x_shift)
            rect.set_y_center(rect.y_center + y_shift)

        if self.square_long:
            long_side = np.maximum(width * img_width, height * img_height)
            width = long_side / img_width
            height = long_side / img_height

        rect.set_width(width * self.scale_x)
        rect.set_height(height * self.scale_y)

        return rect

    def normalized_landmarks_list_to_rect(self, landmark, img_size):
        rotation = self.compute_rotation(landmark)
        revese_angle = self.normalize_radians(-rotation)

        # Find boundaries of landmarks.
        max_x = np.max(self.landmark_subset[:, 0])
        max_y = np.max(self.landmark_subset[:, 1])
        min_x = np.min(self.landmark_subset[:, 0])
        min_y = np.min(self.landmark_subset[:, 1])

        axis_aligned_center_x = (max_x + min_x) * 0.5
        axis_aligned_center_y = (max_y + min_y) * 0.5

        # Find boundaries of rotated landmarks.
        original_x = self.landmark_subset[:, 0] - axis_aligned_center_x
        original_y = self.landmark_subset[:, 1] - axis_aligned_center_y

        projected_x = original_x * math.cos(revese_angle) - original_y * math.sin(
            revese_angle
        )
        projected_y = original_x * math.sin(revese_angle) + original_y * math.cos(
            revese_angle
        )

        max_x = np.max(projected_x)
        max_y = np.max(projected_y)
        min_x = np.min(projected_x)
        min_y = np.min(projected_y)

        projected_center_x = (max_x + min_x) * 0.5
        projected_center_y = (max_y + min_y) * 0.5

        center_x = (
            projected_center_x * math.cos(rotation)
            - projected_center_y * math.sin(rotation)
            + axis_aligned_center_x
        )
        center_y = (
            projected_center_x * math.sin(rotation)
            + projected_center_y * math.cos(rotation)
            + axis_aligned_center_y
        )

        width = (max_x - min_x) / img_size[0]
        height = (max_y - min_x) / img_size[1]

        rect = NormalizedRect()
        rect.set_x_center(center_x / img_size[0])
        rect.set_y_center(center_y / img_size[1])
        rect.set_width(width)
        rect.set_height(height)
        rect.set_rotation(rotation)

        return rect

    def compute_rotation(self, landmark):
        self.landmark_subset = np.zeros(
            (self.num_joints_subset, landmark.shape[1]), dtype=np.float32
        )
        self.landmark_subset[0:3] = landmark[
            0:3
        ].copy()  # Wrist and thumb's two indexes
        self.landmark_subset[3:5] = landmark[5:7].copy()  # Index MCP & PIP
        self.landmark_subset[5:7] = landmark[9:11].copy()  # Middle MCP & PIP
        self.landmark_subset[7:9] = landmark[13:15].copy()  # Ring MCP & PIP
        self.landmark_subset[9:11] = landmark[17:19].copy()  # Pinky MPC & PIP

        x0, y0 = (
            self.landmark_subset[self.kWristJoint][0],
            self.landmark_subset[self.kWristJoint][1],
        )

        x1 = (
            self.landmark_subset[self.kIndexFingerPIPJoint][0]
            + self.landmark_subset[self.kRingFingerPIPJoint][0]
        ) * 0.5
        y1 = (
            self.landmark_subset[self.kIndexFingerPIPJoint][1]
            + self.landmark_subset[self.kRingFingerPIPJoint][1]
        ) * 0.5
        x1 = (x1 + self.landmark_subset[self.kMiddleFingerPIPJoint][0]) * 0.5
        y1 = (y1 + self.landmark_subset[self.kMiddleFingerPIPJoint][1]) * 0.5

        rotation = self.normalize_radians(
            self.kTargetAngle - math.atan2(-(y1 - y0), x1 - x0)
        )
        return rotation

    @staticmethod
    def normalize_radians(angle):
        return angle - 2 * math.pi * np.floor((angle - (-math.pi)) / (2 * math.pi))
