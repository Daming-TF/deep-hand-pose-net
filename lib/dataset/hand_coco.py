import copy
import cv2
import os
import torch
import logging
import numpy as np
import random
import json_tricks as json
from collections import defaultdict, OrderedDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset

from lib.core.inference import get_affine_transform, affine_transform
from lib.dataset.augmentation import (
    adjust_contrast_brightness,
    gaussian_blur,
    motion_blur,
)
from lib.nms.nms import oks_nms
from lib.utils.mediapipe_alignments import HandInfo


logger = logging.getLogger(__name__)


class HandCOCODataset(Dataset):
    """
    该程序主要用于整理需要的数据格式
    初始化输入:
    cfg：通过输入的yaml文件更新后的CN容器
    root：数据集路径      如：/workspace/cpfs-data/Data/hand_coco_v2_3'
    image_set: 训练集文件名
    is_train：是否训练用的
    transform：数据处理块
    """
    def __init__(self, cfg, root, image_set, is_train=True, transform=None):
        self.nms_thre = cfg.TEST.NMS_THRE  # 1.0
        self.image_thre = cfg.TEST.IMAGE_THRE  # 0.0
        self.soft_nms = cfg.TEST.SOFT_NMS  # False
        self.oks_thre = cfg.TEST.OKS_THRE  # 0.9
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE  # 0.2
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE  # ''
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX  # true
        self.image_width = cfg.MODEL.IMAGE_SIZE[1]  # width
        self.image_height = cfg.MODEL.IMAGE_SIZE[0]  # height
        self.aspect_ratio = self.image_width * 1.0 / self.image_height      # 纵横比
        self.pixel_std = 200        # 像素缩放比例
        # self.is_alignment = cfg.MODEL.ALIGNMENT

        self.is_train = is_train        # __getitem__里面只有在训练模式下才会进行数据增强
        self.root = root    # 根路径
        self.image_set = image_set      # ‘train2017’
        self.data_format = cfg.DATASET.DATA_FORMAT  # .jpg
        self.color_rgb = cfg.DATASET.COLOR_RGB  # true
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)  # (height, width)
        self.transform = transform  # normalize and convert to tensor
        self.target_type = cfg.MODEL.TARGET_TYPE  # gaussian
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)  # (height/4, width/4)
        self.sigma = cfg.MODEL.SIGMA  # 1
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT  # true
        # Agumentation factors
        self.scale_factor = cfg.DATASET.SCALE_FACTOR  # 0.1
        self.rotation_factor = cfg.DATASET.ROT_FACTOR  # 20
        self.shift = cfg.DATASET.SHIFT  # true
        self.flip = cfg.DATASET.FLIP  # true
        self.shift_factor = cfg.DATASET.SHIFT_FACTOR  # 0.05
        self.color_jitter = cfg.DATASET.COLOR  # true
        self.blur = cfg.DATASET.BLUR  # true
        self.gray = cfg.DATASET.GRAY  # true

        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ["__background__"] + cats
        logger.info("=> classes: {}".format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()     # 图片id列表
        self.num_images = len(self.image_set_index)
        logger.info("=> num_images: {}".format(self.num_images))

        self.num_joints = 21
        self.parent_ids = None

        self.joints_weight = np.array(
            [
                1.0,
                1.0,
                1.2,
                1.2,
                1.5,
                1.0,
                1.2,
                1.2,
                1.5,
                1.0,
                1.2,
                1.2,
                1.5,
                1.0,
                1.2,
                1.2,
                1.5,
                1.0,
                1.2,
                1.2,
                1.5,
            ],
            dtype=np.float32,
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()        # 每个单元是一个列表，对应一张图片（每个对象皆为一个类别的字典{'images':image_dir,'center','scale','joints_3d','joints_3d_vis'}）
        logger.info("=> load {} samples".format(len(self.db)))

    def _get_ann_file_keypoint(self):       # 在构造函数时调用，为了获取annotations中的json文件
        data_path = os.path.join(
            self.root, "annotations", "person_keypoints_" + self.image_set + ".json"
        )
        return data_path

    def _load_image_set_index(self):    # 利用coco的接口，在构造函数中获取所有载入数据的图片id，并存储在self.image_set_index
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):      # 通过接口：_load_coco_keypoint_annotations， 载入设定的数据存储单元
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    @staticmethod
    def _load_coco_person_detection_results():
        # 抛出一个异常
        raise Exception(" [!] Not Implment detection-bbox input!")

    def _load_coco_keypoint_annotations(self):      # 载入设定的数据存储单元
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):     # 输入图片id处理，annos，获得设定的存储单元
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        # im_ann_list = self.coco.loadImgs(index)
        # if len(im_ann_list) > 1:
        #     pass
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]

        ann_ids = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(ann_ids)      # objs返回整个让图片的对象

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))       # 令框的有边界不超出图片边界
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2 - x1, y2 - y1]      # bbox：{x_left, y_top, width, height}
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj["category_id"]]
            if cls != 1:        # 手的cls在这里设置为1，所以当类别不为1，则直接跳过
                continue

            # ignore objs without keypoints annotation
            if max(obj["keypoints"]) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj["keypoints"][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj["keypoints"][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj["keypoints"][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj["clean_bbox"][:4])     # center：bbox中心， scale：图片放缩大小
            image_path_ = self.image_path_from_index(index)
            rec.append(
                {
                    "image": image_path_,  # self.image_path_from_index(index),
                    "center": center,
                    "scale": scale,
                    "joints_3d": joints_3d,
                    "joints_3d_vis": joints_3d_vis,
                    "filename": "",
                    "imgnum": 0,
                }
            )

        return rec

    def _box2cs(self, box):     # 输入clean_bbox
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        # 把短的边界拉长直到w=h，aspect_ratio = image_width * 1.0 / image_height
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std], dtype=np.float32
        )
        if center[0] != -1:
            scale = scale * 1.0

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = "%012d.jpg" % index
        if "2014" in self.image_set:
            file_name = "COCO_%s_" % self.image_set + file_name

        prefix = "test2017" if "test" in self.image_set else self.image_set
        data_name = prefix + ".zip@" if self.data_format == "zip" else prefix
        image_path = os.path.join(self.root, "images", data_name, file_name)

        return image_path

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec["image"]
        filename = db_rec["filename"] if "filename" in db_rec else ""
        imgnum = db_rec["imgnum"] if "imgnum" in db_rec else ""

        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )

        if self.color_rgb:  # true
            try:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            except Exception as ex:
                print(f" [!] {ex}, ERROR image_file: {image_file}!")

        if data_numpy is None:
            logger.error("=> fail to read {}".format(image_file))
            raise ValueError("Fail to read {}".format(image_file))

        joints = db_rec["joints_3d"]
        joints_vis = db_rec["joints_3d_vis"]

        c = db_rec["center"]
        s = db_rec["scale"]
        score = db_rec["score"] if "score" in db_rec else 1  # 1, no 'score' item in rec
        r = 0

        shift = np.array([0, 0])

        if self.is_train:  #
            # sf = self.scale_factor
            # rf = self.rotation_factor
            # # s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            # s = 1 - sf + 2 * sf * np.random.uniform()  # [1 - sf, 1 + sf]
            # # r = np.clip(np.random.randn() * rf, -rf * 1, rf * 1) if random.random() <= 0.6 else 0
            # r = -1 * rf + 2 * rf * np.random.uniform()
            sf = self.scale_factor  # 0.1
            rf = self.rotation_factor  # 20
            # numpy.random.rand生成[0,1)范围数字
            # 得到0.9-1.1的随机数     3/4概率, s=0.9
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            # 3/5概率, r∈(-40,40)
            r = (
                np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
                if random.random() <= 0.6
                else 0
            )

            tf = self.shift_factor
            if self.shift and random.random() <= 0.5:
                shift_x = np.clip(np.random.randn(), -1.0, 1.0) * tf
                shift_y = np.clip(np.random.randn(), -1.0, 1.0) * tf
                shift = np.array([shift_x, shift_y])

            if self.flip and random.random() <= 0.5:
                # Flip horizontal
                data_numpy = data_numpy[:, ::-1, :]  # image horizontal-flip
                joints[:, 0] = (
                    data_numpy.shape[1] - joints[:, 0] - 1
                )  # landmark horizontal-flip
                c[0] = data_numpy.shape[1] - c[0] - 1

            if self.color_jitter and random.random() < 0.5:
                max_b = 45
                brightness_adjust = np.clip(
                    np.random.randn() * max_b, -max_b * 2, max_b * 2
                )
                max_c = 0.2
                contrast_adjust = np.clip(
                    np.random.randn() * max_c + 1, 1 - max_c, 1 + max_c
                )
                data_numpy = adjust_contrast_brightness(
                    data_numpy, contrast_adjust, brightness_adjust
                )

            if self.blur and random.random() < 0.5:  # blur
                if random.random() < 0.3:
                    data_numpy = gaussian_blur(data_numpy)
                else:
                    data_numpy = motion_blur(data_numpy)

            if self.gray and random.random() < 0.2:  # gray
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_RGB2GRAY)
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_GRAY2RGB)

        trans = get_affine_transform(c, s, r, self.image_size, shift)       # 获得仿射变换旋转矩阵(2, 3)
        # image affine
        input_data = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR,
        )

        if self.transform:
            input_data = self.transform(input_data)

        # landmark affine 把关键点也进行仿射变换
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight, target_reg = self.generate_target(joints, joints_vis)        # 把gt的关键点信息转成21张高斯热图

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        target_reg = torch.from_numpy(target_reg)

        meta = {
            "image": image_file,
            "filename": filename,
            "imgnum": imgnum,
            "joints": joints,
            "joints_vis": joints_vis,
            "center": c,
            "scale": s,
            "rotation": r,
            "score": score,
        }

        return input_data, target, target_weight, meta, target_reg

    def generate_target(self, joints, joints_vis):      # 在__getitem__调用
        """
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        target_reg = np.zeros((self.num_joints, 2), dtype=np.float32)
        for joint_id in range(self.num_joints):
            target_reg[joint_id, 0] = joints[joint_id, 0] / self.image_size[0] - 0.5    # 归一化到-0.5~0.5之间
            target_reg[joint_id, 1] = joints[joint_id, 1] / self.image_size[1] - 0.5

        assert self.target_type == "gaussian", "Only support gaussian map now!"

        target = None
        if self.target_type == "gaussian":
            target = np.zeros(
                (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )
            tmp_size = self.sigma * 3       # 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)      # 转到heatmap的坐标系下
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds: 检查高斯的任何部分是否在界内
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if (
                    ul[0] >= self.heatmap_size[0]
                    or ul[1] >= self.heatmap_size[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    # If not, just return the image as is: 如果不是，则按原样返回图像
                    target_weight[joint_id] = 0
                    continue

                # Generate gaussian: 生成高斯
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1: 高斯未归一化，我们希望中心值等于 1
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range: 可用高斯范围
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range: 图像范围
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:     # 只要权重大于0.5的关键点才产生对应的高斯热值
                    target[joint_id][img_y[0]: img_y[1], img_x[0]: img_x[1]] = g[
                        g_y[0]: g_y[1], g_x[0]: g_x[1]
                    ]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight, target_reg

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path, *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, "results")
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception as ex:
                logger.error(f" [!] {ex} Fail to make {res_folder}!")

        res_file = os.path.join(
            res_folder, f"keypoints_{self.image_set}_results_{rank}.json"
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append(
                {
                    "keypoints": kpt,
                    "center": all_boxes[idx][0:2],
                    "scale": all_boxes[idx][2:4],
                    "area": all_boxes[idx][4],
                    "score": all_boxes[idx][5],
                    "image": int(img_path[idx][-16:-4]),
                }
            )

        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt["image"]].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints  # 21
        in_vis_thre = self.in_vis_thre  # 0.2
        oks_thre = self.oks_thre  # 0.9
        oks_nmsed_kpts = []

        for img in kpts.keys():
            img_kpts = kpts[img]

            for n_p in img_kpts:
                box_score = n_p["score"]
                kpt_score = 0
                valid_num = 0

                for n_jt in range(0, num_joints):
                    t_s = n_p["keypoints"][n_jt][2]

                    if t_s > in_vis_thre:  # 0.2
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1

                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p["score"] = kpt_score * box_score

            keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], oks_thre)

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(oks_nmsed_kpts, res_file)

        info_str = self._do_python_keypoint_eval(res_file)
        name_value = OrderedDict(info_str)      # OrderedDict：有顺序的dict
        return name_value, name_value["AP"]

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                "cat_id": self._class_to_coco_ind[cls],
                "cls_ind": cls_ind,
                "cls": cls,
                "ann_type": "keypoints",
                "keypoints": keypoints,
            }
            for cls_ind, cls in enumerate(self.classes)
            if not cls == "__background__"
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info("=> writing results json to %s" % res_file)
        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, "r") as f:
                for line in f:
                    content.append(line)
            content[-1] = "]"
            with open(res_file, "w") as f:
                for c in content:
                    f.write(c)

    def _do_python_keypoint_eval(self, res_file):
        print(f"res_file: {res_file}")

        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, "keypoints")
        coco_eval.params.useSegm = None
        kpt_oks_sigmas = np.ones(self.num_joints) * 0.35 / 10.0
        # np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        coco_eval.params.kpt_oks_sigmas = kpt_oks_sigmas
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = [
            "AP",
            "Ap .5",
            "AP .75",
            "AP (M)",
            "AP (L)",
            "AR",
            "AR .5",
            "AR .75",
            "AR (M)",
            "AR (L)",
        ]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack["cat_id"]
        keypoints = data_pack["keypoints"]
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array(
                [img_kpts[k]["keypoints"] for k in range(len(img_kpts))]
            )
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    "image_id": img_kpts[k]["image"],
                    "category_id": cat_id,
                    "keypoints": list(key_points[k]),
                    "score": img_kpts[k]["score"],
                    "center": list(img_kpts[k]["center"]),
                    "scale": list(img_kpts[k]["scale"]),
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results
