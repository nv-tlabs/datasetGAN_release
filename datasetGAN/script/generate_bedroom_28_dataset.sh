#!/bin/bash
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=4  python train_interpreter.py  --exp experiments/bedroom_28.json --resume ./model_dir/bedroom_28 --num_sample  2500 --start_step 0 --generate_data True &
CUDA_VISIBLE_DEVICES=5  python train_interpreter.py  --exp experiments/bedroom_28.json --resume ./model_dir/bedroom_28 --num_sample  2500 --start_step 2500 --generate_data True &
CUDA_VISIBLE_DEVICES=6  python train_interpreter.py  --exp experiments/bedroom_28.json --resume ./model_dir/bedroom_28 --num_sample  2500 --start_step 5000 --generate_data True &
CUDA_VISIBLE_DEVICES=7  python train_interpreter.py  --exp experiments/bedroom_28.json --resume ./model_dir/bedroom_28 --num_sample  2500 --start_step 7500 --generate_data True

