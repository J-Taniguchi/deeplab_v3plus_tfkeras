# deeplab_v3plus_tfkeras

Programs for semantic segmentation. Useage is written in ipynb files.

- you can make annotation file as
  - RGB image
  - index png image(like pascal voc2012)
  - polygon ([labelme](https://github.com/wkentaro/labelme) json file.)
    - please work with --nodata flag like ```labelme --nodata```
- It is better to modify data_augment.py for your objective.
  - [albumentations](https://github.com/albumentations-team/albumentations) is used for data augment.
