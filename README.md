# deeplab_v3plus_tfkeras

Programs for semantic segmentation.
If you want to use Jupyter Notebook, see this [branch](https://github.com/J-Taniguchi/deeplab_v3plus_tfkeras/tree/jupyter)

use like run_sample.sh.

you can make annotation file as RGB image

- you can use polygon for annotation.([labelme](https://github.com/wkentaro/labelme) json file.)
  - please work with --nodata flag like ```labelme --nodata```
  - aftar make jsons, use omake/convert_labelme_json_to_palette_png.py
- It is better to modify data_augment.py for your objective.

You can use pipenv to create virtual environment.
``` bash
pipenv sync
```

# config.yaml

Describe these keywords in YAML format in your configuration file.

See conf_sample.yml.

|  key  |  description  |
| :---: | :--- |
|use_devices | use gpu number. If you want to do distributed learning, write like "0,1,2" |
|model_dir                                           |all outputs are written here. <br>e.g., trained model, inference results, etc..|
|  train_x_dirs<br> valid_x_dirs<br> test_x_dirs  | list of directories where input images exist.|
|  train_y_dirs<br> valid_y_dirs                   | list of directories where segmentation image exsit.<br> Each segmentation image name must be the same for corresponding input image.  |
|which_to_inference <br> which_to_visualise          | chosse from "train", "valid", "test".|
|output_activation                                   | chosse "softmax" or "sigmoid". <br>softmax means each pixcel is assigned to 1 label.<br>sigmoid means each pixel can assigned 1 or more labels.|
|batch_size | batch size|
|n_epochs   |nuber of epochs |
|image_size | [height, width] |
|optimizer  |"Adam" or "Nadam" or "SGD" |
|loss       |choose one .<br>"CE": cross entropy <br> "FL": focal loss <br>"GDL": generalized dice loss
|use_tensorboard| if you want to use, True. If not, False.|

# train.py

Training with the data written in train_x_dirs and train_y_dirs, watch validation with the data written in valid_x_dirs and valid_y_dirs.

Trained model and training log are written in model_dir.
use like
``` bash
python train.py conf.yml
```


# train_continue.py

Training continues from finale_epoch.h5.

Trained model and log are overwrite to model_dir.



# inference.py

Inference to the dataset written in which_to_inference.

Inference results are written in model_dir with h5 format.  This file is use for visualise.py

use like
``` bash
python inference.py conf.yml
```

# visualise.py

visualise thre inference results written in  which_to_visualise.

use like
``` bash
python visualise.py conf.yml
```

# omake/convert_labelme_json_to_palette_png.py

```
usage: convert_labelme_json_to_palette_png.py [-h]
                                              label_list_path input_dir
                                              output_dir

positional arguments:
  label_list_path  path to label list csv file
  input_dir        path to input dir. all json files under this directory will
                   be used to generate segmentation image file.
  output_dir       path to output dir. segmentation images will be generated
                   to this directory.

optional arguments:
  -h, --help       show this help message and exit
```