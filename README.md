# deeplab_v3plus_tfkeras

Programs for semantic segmentation. ~~Useage is written in ipynb files.~~ 

use like run_sample.sh. 

- you can make annotation file as
  - RGB image
  - index png image(like pascal voc2012)
  - polygon ([labelme](https://github.com/wkentaro/labelme) json file.)
    - please work with --nodata flag like ```labelme --nodata```
- It is better to modify data_augment.py for your objective.
  - [albumentations](https://github.com/albumentations-team/albumentations) is used for data augment.

# train.py

Training with the data written in train_data_paths, watch validation with the data written in  valid_data_paths.

Trained model and training log are written in model_dir.



# train_continue.py

Training continues from finale_epoch.h5.

Trained model and log are overwrite to model_dir.



# inference.py

Inference to the dataset written in which_to_inference.

Inference results are written in model_dir with h5 format.  This file is use for visualise.py

# visualise.py

visualise thre inference results written in  which_to_visualise.