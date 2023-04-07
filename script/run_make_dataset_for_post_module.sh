#!/bin/tcsh

set gpu_id = 0

mkdir -p ../out/for_post_module/train

python3 ../tool/make_dataset_for_post_module.py ../config/config.py $gpu_id
