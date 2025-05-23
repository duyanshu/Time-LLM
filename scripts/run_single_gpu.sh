#!/bin/bash

accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=no \
    run_main.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_96_96 \
    --model_comment single_gpu \
    --model TimeLLM \
    --data ETTh1 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1 \
    --d_model 512 \
    --n_heads 8 \
    --checkpoints ./checkpoints 