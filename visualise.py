import os
import sys
import joblib

import yaml

from deeplab_v3plus_tfkeras.label import Label
from deeplab_v3plus_tfkeras.data_utils import load_inference_results
from deeplab_v3plus_tfkeras.input_data_processing import make_data_path_list
from deeplab_v3plus_tfkeras.vis_utils import convert_y_to_image_array
from deeplab_v3plus_tfkeras.vis_utils import visualise_inference_result

conf_file = sys.argv[1]
with open(conf_file, "r") as f:
    conf = yaml.safe_load(f)

threshold = 0.0

model_dir = conf["model_dir"]
label_file_path = conf["label_file_path"]

n_extra_channels = conf.get("n_extra_channels", 0)

train_x_dirs = conf["train_x_dirs"]
train_extra_x_dirs = conf.get("train_extra_x_dirs", None)
train_y_dirs = conf["train_y_dirs"]

valid_x_dirs = conf["valid_x_dirs"]
valid_extra_x_dirs = conf.get("valid_extra_x_dirs", None)
valid_y_dirs = conf["valid_y_dirs"]

test_x_dirs = conf.get("test_x_dirs", None)
test_extra_x_dirs = conf.get("test_extra_x_dirs", None)
test_y_dirs = conf.get("test_y_dirs", None)

which_to_visualise = conf["which_to_visualise"]
njobs_for_visualise = conf.get("njobs_for_visualise", 1)

fig_dir = os.path.join(model_dir, "figure")
label = Label(label_file_path)

n_jobs = njobs_for_visualise


for target in which_to_visualise:
    x_dirs = conf["{}_x_dirs".format(target)]
    extra_x_dirs = conf.get("{}_extra_x_dirs".format(target), None)
    y_dirs = conf.get("{}_y_dirs".format(target), None)

    fig_out_dir = os.path.join(fig_dir, target)
    os.makedirs(fig_out_dir, exist_ok=True)

    fpath = os.path.join(model_dir, "{}_inference.h5".format(target))
    pred, last_activation = load_inference_results(fpath)

    path_list = make_data_path_list(
        x_dirs,
        y_dirs=y_dirs,
        extra_x_dirs=extra_x_dirs)

    y_pred = convert_y_to_image_array(pred,
                                      label,
                                      threshold=threshold,
                                      activation=last_activation)

    joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        backend="loky")(joblib.delayed(visualise_inference_result)(
            i,
            path_list,
            y_pred,
            fig_out_dir,
            last_activation,
            label=None) for i in range(len(y_pred)))
