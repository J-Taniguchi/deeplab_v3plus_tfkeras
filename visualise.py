import os
import sys
import joblib
n_jobs = 36
import yaml

from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.data_utils import load_inference_results
from deeplab_v3plus_tfkeras.vis_utils import convert_y_to_image_array
from deeplab_v3plus_tfkeras.vis_utils import visualise_true_pred
from deeplab_v3plus_tfkeras.vis_utils import visualise_pred

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)


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
    x, y, pred, last_activation = load_inference_results(fpath)

    y_pred = convert_y_to_image_array(pred,
                                      label,
                                      threshold=0.5,
                                      activation=last_activation)
    y_true = convert_y_to_image_array(y,
                                      label,
                                      activation=last_activation)

    joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend="threading")(joblib.delayed(visualise_true_pred)(
            i,
            x,
            y_true,
            y_pred,
            fig_out_dir,
            last_activation,
            label) for i in range(x.shape[0]))

# valid data
if "valid" in which_to_visualise:
    fig_out_dir = os.path.join(fig_dir, "valid")
    os.makedirs(fig_out_dir, exist_ok=True)

    fpath = os.path.join(model_dir, "valid_inference.h5")
    x, y, pred, last_activation = load_inference_results(fpath)

    y_pred = convert_y_to_image_array(pred,
                                      label,
                                      threshold=0.5,
                                      activation=last_activation)
    y_true = convert_y_to_image_array(y,
                                      label,
                                      activation=last_activation)

    joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend="threading")(joblib.delayed(visualise_true_pred)(
            i,
            x,
            y_true,
            y_pred,
            fig_out_dir,
            last_activation,
            label) for i in range(x.shape[0]))

# test data
if "test" in which_to_visualise:
    for test_data_dir in test_x_dirs:
        print(test_data_dir)
        test_name = test_data_dir.split(os.sep)[-1]
        fig_out_dir = os.path.join(fig_dir, "test_" + test_name)
        os.makedirs(fig_out_dir, exist_ok=True)

        fpath = os.path.join(model_dir, "test_" + test_name + "_inference.h5")
        x, _, pred, last_activation = load_inference_results(fpath)

        y_pred = convert_y_to_image_array(pred,
                                          label,
                                          threshold=0.5,
                                          activation=last_activation)

        joblib.Parallel(
            n_jobs=n_jobs,
            verbose=10,
            backend="threading")(joblib.delayed(visualise_pred)(
                i,
                x,
                y_pred,
                fig_out_dir,
                last_activation,
                label) for i in range(len(x)))
