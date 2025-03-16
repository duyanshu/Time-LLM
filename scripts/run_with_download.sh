#!/bin/bash

# 检查模型是否已下载
if [ ! -d "pretrained_models/llama-7b" ]; then
    echo "Downloading LLaMA model..."
    python scripts/download_models.py
fi

# 运行训练脚本
bash scripts/run_single_gpu.sh 