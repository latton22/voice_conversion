#!/bin/tcsh

set gpu_id = 0

mkdir -p ../out/ppg

python3 ../tool/predictor.py ../config/config.py $gpu_id
