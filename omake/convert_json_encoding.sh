#!/bin/bash -Ceu

tar_file=$(ls *.json)

for f in $tar_file; do
    echo $f
    nkf -w -Lu --overwrite $f
done