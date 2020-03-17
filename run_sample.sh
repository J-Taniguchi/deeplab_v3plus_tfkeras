#!/bin/bash -Ceu
export PYTHONPATH=path/to/deeplab_v3plus_tfkeras/
conf_file=conf.yml
python train.py $conf_file
#python train_continue.py $conf_file
python inference.py $conf_file
python visualise.py $conf_file
