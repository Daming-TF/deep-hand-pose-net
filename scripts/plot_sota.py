import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text


pro_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pro_dir)


def convert_str_to_num(arr):
    # Convert K, M, G string to float number.
    data = list()
    num_arr = arr.copy()
    for i, f in enumerate(num_arr):
        data.append(float(f[0]))
    return num_arr


def get_name_and_color(models, resolutions, alphas, decoders):
    mobilenet_colors = np.array([[164, 196, 0], [96, 169, 23], [0, 138, 0],])
    mediapipe_colors = np.array([[227, 200, 0]])
    others_color = np.array([[240, 163, 10], [170, 0, 255],])

    # Preprocessing
    exp_names = []
    colors = []
    transparent = 0.8

    for i, (model, resolution, alpha, decoder) in enumerate(
        zip(models, resolutions, alphas, decoders)
    ):
        alpha = str(alpha[0])
        color, model_name = None, None

        if "Small" in model[0]:
            color = others_color[0] / 255.0
            model_name = model[0]
            alpha = ""
        elif ("Large" in model[0]) and ("MobileNet" not in model[0]):
            color = others_color[1] / 255.0
            model_name = model[0]
            alpha = ""
        elif model[0] == "MobileNet-v1":
            color = mobilenet_colors[0] / 255.0
            model_name = "MNetv1"
        elif model[0] == "MobileNet-v2":
            color = mobilenet_colors[1] / 255.0
            model_name = "MNetv2"
        elif model[0] == "MobileNet-v2*":
            color = mobilenet_colors[1] / 255.0
            model_name = "MNetv2*"
        elif model[0] == "MobileNet-v3-Large":
            color = mobilenet_colors[2] / 255.0
            model_name = "MNetv3L"
        elif model[0] == "Mediapipe-Full":
            color = mediapipe_colors[0] / 255.0
            model_name = "Mediapipe-F"
        elif model[0] == "Mediapipe-Lite":
            color = mediapipe_colors[0] / 255.0
            model_name = "Mediapipe-L"

        resolution = resolution[0][: len(resolution[0]) // 2]

        if decoder[0] == "CoarseRefineDecoder":
            decoder = "CRD"
        elif decoder[0] == "IterativeHeadDecoder":
            decoder = "IHD"
        else:  # SimpleDecoder
            decoder = "SD"

        print(f"model_name: {model_name}")
        print(f"resolution: {resolution}")
        print(f"alpha: {alpha}")
        print(f"decoder: {decoder}")
        name = "-".join([model_name, resolution, alpha, decoder])
        name = name.replace("---", "-")
        name = name.replace("--", "-")
        exp_names.append(name)
        colors.append((*color, transparent))

    return exp_names, colors


def main():
    # Read data from excel
    exp_data = pd.read_excel(
        r"F:\BOBBY\Code\H0097-deep-hand-pose\deep-hand-pose-net\experiments\academic\SOTA.xlsx",
        engine="openpyxl",
    )

    models = pd.DataFrame(exp_data, columns=["Model"]).values.tolist()
    resolutions = pd.DataFrame(exp_data, columns=["Resolution"]).values.tolist()
    alphas = pd.DataFrame(exp_data, columns=["Alpha"]).values.tolist()
    decoders = pd.DataFrame(exp_data, columns=["Decoder"]).values.tolist()
    params = pd.DataFrame(exp_data, columns=["#params"]).values.tolist()
    mflops = pd.DataFrame(exp_data, columns=["MFlops"]).values.tolist()
    f1 = pd.DataFrame(exp_data, columns=["F1"]).values.tolist()

    exp_names, colors = get_name_and_color(models, resolutions, alphas, decoders)
    mflops = convert_str_to_num(mflops)
    params = convert_str_to_num(params)
    f1 = convert_str_to_num(f1)

    min_f1, max_f1 = min(60, np.amin(f1) - 2), max(75, int(np.amax(f1)) + 2)
    min_mflops, max_mflops = min(0, np.amin(mflops)), max(np.amax(mflops), 30e2)

    # Plot
    plt.rcParams["figure.figsize"] = [40, 20]
    plt.rcParams["font.size"] = 24

    fig, ax = plt.subplots()
    params = [p[0] / 1000 for p in params]
    ax.scatter(mflops, f1, marker="o", s=params, c=colors)
    ax.set_xscale("log")  # FLOPs are in log scale

    texts = []
    for i, exp in enumerate(exp_names):
        texts.append(plt.text(mflops[i][0], f1[i][0], exp, fontsize=20))

    # Adjust text to avoid overlapping
    adjust_text(
        texts, expand_objects=(2, 2), arrowprops=dict(arrowstyle="->", color="r", lw=2)
    )

    plt.xlabel("MFLOPs")
    plt.ylabel("F1-Score")
    ax.grid(True)

    plt.xlim((min_mflops, max_mflops))
    plt.ylim((min_f1, max_f1))  # f1 score
    plt.title("SOTA")

    # plt.show()
    plt.savefig(
        r"F:\BOBBY\Code\H0097-deep-hand-pose\deep-hand-pose-net\experiments\academic\sota.png"
    )
    plt.close()


if __name__ == "__main__":
    main()
