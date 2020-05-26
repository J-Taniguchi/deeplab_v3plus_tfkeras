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

# train.py

Training with the data written in train_x_paths and train_y_paths, watch validation with the data written in valid_x_paths and valid_y_paths.

Trained model and training log are written in model_dir.



# train_continue.py

Training continues from finale_epoch.h5.

Trained model and log are overwrite to model_dir.



# inference.py

Inference to the dataset written in which_to_inference.

Inference results are written in model_dir with h5 format.  This file is use for visualise.py

# visualise.py

visualise thre inference results written in  which_to_visualise.


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