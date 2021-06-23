#!/bin/bash
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=4  python train_interpreter.py  --exp experiments/car_20.json --resume ./model_dir/car_20 --num_sample  2500 --start_step 10000 --generate_data True &
CUDA_VISIBLE_DEVICES=5  python train_interpreter.py  --exp experiments/car_20.json --resume ./model_dir/car_20 --num_sample  2500 --start_step 12500 --generate_data True &
CUDA_VISIBLE_DEVICES=6  python train_interpreter.py  --exp experiments/car_20.json --resume ./model_dir/car_20 --num_sample  2500 --start_step 15000 --generate_data True &
CUDA_VISIBLE_DEVICES=7  python train_interpreter.py  --exp experiments/car_20.json --resume ./model_dir/car_20 --num_sample  2500 --start_step 17500 --generate_data True



CUDA_VISIBLE_DEVICES=0  python train_interpreter.py  --exp experiments/bedroom_19.json --resume ./model_dir/bedroom_19 --num_sample  10  --save_vis True  --generate_data True
