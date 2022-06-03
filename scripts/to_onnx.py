import os
import torch
import torchvision.transforms as transforms

# noinspection PyUnresolvedReferences
import _init_path
import lib.models
from lib.config import cfg, update_config, parse_args
from lib.utils import create_logger, set_cudnn_related_setting, get_model_summary


def main():
    args = parse_args()
    update_config(cfg, args)

    _, final_output_dir, _ = create_logger(cfg, args.cfg, "to_onnx")

    set_cudnn_related_setting(cfg)  # cudnn related setting
    model = getattr(lib.models, cfg.MODEL.NAME).get_pose_net(cfg, is_train=False)

    if cfg.TEST.MODEL_FILE:
        print("=> Loading model from {}".format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(final_output_dir, "model_best.pth")
        print("=> loading model from {}".format(model_state_file))
        model.load_state_dict(torch.load(model_state_file), strict=False)
    model.eval()

    ####################################################################################################################
    # Convert to ONNX
    file_name = os.path.basename(final_output_dir)
    onnx_out_file = os.path.join(final_output_dir, file_name + ".onnx")
    print("*" * 50)
    print(f"onnx_out_file: {onnx_out_file}")
    print("*" * 50)

    test_inp = torch.randn(
        1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0], device="cpu"
    )

    # 将模型以ONNX格式导出并保存.
    torch.onnx.export(
        model,
        test_inp,
        onnx_out_file,
        export_params=True,
        verbose=False,
        input_names=["input"],
        output_names=["output"],
        opset_version=9,
    )
    print(f" [!] Convert onnx SUCCESS!")
    ####################################################################################################################

    # calculate model params. and MFlops info
    result = get_model_summary(
        model, torch.rand((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    )
    print(result)


if __name__ == "__main__":
    main()
