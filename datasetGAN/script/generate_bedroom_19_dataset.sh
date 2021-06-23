#!/bin/bash
export PYTHONPATH=$PWD

CUDA_VISIBLE_DEVICES=0  python train_interpreter.py  --exp experiments/bedroom_19.json --resume ./model_dir/bedroom_19 --num_sample  2500 --start_step 0 --generate_data True &
CUDA_VISIBLE_DEVICES=1  python train_interpreter.py  --exp experiments/bedroom_19.json --resume ./model_dir/bedroom_19 --num_sample  2500 --start_step 2500 --generate_data True &
CUDA_VISIBLE_DEVICES=2  python train_interpreter.py  --exp experiments/bedroom_19.json --resume ./model_dir/bedroom_19 --num_sample  2500 --start_step 5000 --generate_data True &
CUDA_VISIBLE_DEVICES=3  python train_interpreter.py  --exp experiments/bedroom_19.json --resume ./model_dir/bedroom_19 --num_sample  2500 --start_step 7500 --generate_data True
