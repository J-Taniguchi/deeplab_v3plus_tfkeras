#!/bin/bash -Ceu
export PYTHONPATH=path/to/this_dir/
out_dir="path/to/out_dir"
out2_dir="path/to/out2_dir"
traindata_dir='path/to/traindata_dir'
validdata_dir='path/to/validdata_dir'
testdata_dir='path/to/testdata_dir'
python train.py $out_dir $traindata_dir $validdata_dir
#python train_continue.py $out_dir $out2_dir $traindata_dir $validdata_dir
python inference.py $out_dir $traindata_dir $testdata_dir
python visualise.py $out_dir $traindata_dir $testdata_dir
