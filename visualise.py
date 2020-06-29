import os
import sys
import joblib
n_jobs = 8
import yaml

from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.data_utils import load_inference_results
from deeplab_v3plus_tfkeras.vis_utils import convert_y_to_image_array
from deeplab_v3plus_tfkeras.vis_utils import visualise_inference_result

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)

threshold = 0.0

model_dir = conf["model_dir"]
label_file_path = conf["label_file_path"]
test_x_dirs = conf["test_x_dirs"]

which_to_visualise = conf["which_to_visualise"]

fig_dir = os.path.join(model_dir, "figure")
label = Label(label_file_path)


# train data
if "train" in which_to_visualise:
    fig_out_dir = os.path.join(fig_dir, "train")
    os.makedirs(fig_out_dir, exist_ok=True)

    fpath = os.path.join(model_dir, "train_inference.h5")
    x, extra_x, y, pred, basenames, last_activation = load_inference_results(fpath)

    y_pred = convert_y_to_image_array(pred,
                                      label,
                                      threshold=threshold,
                                      activation=last_activation)
    y_true = convert_y_to_image_array(y,
                                      label,
                                      activation=last_activation)

    joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend="loky")(joblib.delayed(visualise_inference_result)(
            i,
            x,
            y_pred,
            fig_out_dir,
            last_activation,
            label=label,
            y_true=y_true,
            extra_x=extra_x,
            basenames=basenames) for i in range(x.shape[0]))

# valid data
if "valid" in which_to_visualise:
    fig_out_dir = os.path.join(fig_dir, "valid")
    os.makedirs(fig_out_dir, exist_ok=True)

    fpath = os.path.join(model_dir, "valid_inference.h5")
    x, extra_x, y, pred, basenames, last_activation = load_inference_results(fpath)

    y_pred = convert_y_to_image_array(pred,
                                      label,
                                      threshold=threshold,
                                      activation=last_activation)
    y_true = convert_y_to_image_array(y,
                                      label,
                                      activation=last_activation)

    joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend="loky")(joblib.delayed(visualise_inference_result)(
            i,
            x,
            y_pred,
            fig_out_dir,
            last_activation,
            label=label,
            y_true=y_true,
            extra_x=extra_x,
            basenames=basenames) for i in range(x.shape[0]))
# test data
if "test" in which_to_visualise:
    for test_data_dir in test_x_dirs:
        print(test_data_dir)
        test_name = test_data_dir.split(os.sep)[-1]
        fig_out_dir = os.path.join(fig_dir, "test_" + test_name)
        os.makedirs(fig_out_dir, exist_ok=True)

        fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
        x, extra_x, y, pred, basenames, last_activation = load_inference_results(fpath)

        y_pred = convert_y_to_image_array(pred,
                                          label,
                                          threshold=threshold,
                                          activation=last_activation)

        joblib.Parallel(
            n_jobs=n_jobs,
            verbose=10,
            backend="loky")(joblib.delayed(visualise_inference_result)(
                i,
                x,
                y_pred,
                fig_out_dir,
                last_activation,
                extra_x=extra_x,
                label=label,
                basenames=basenames) for i in range(len(x)))
