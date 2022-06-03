import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train keypoints network")
    # parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--cfg",
        help="experiment configure file name",
        default="/workspace/nas-data/deep-hand-pose-net/experiments/hand_coco/debug.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
