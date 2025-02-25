#!/bin/bash

python prepare-training-data.py

#mlx_lm.lora --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --train --data data --batch-size 2 --iters 300
#mlx_lm.lora --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --train --data data --batch-size 2 --num-layers 4 --iters 300 --grad-checkpoint
mlx_lm.lora --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --train --data data --batch-size 2 --iters 300 --grad-checkpoint

echo "Now run 'ollama create bestisblessed/<new-model-name>'"
# ollama create bestisblessed/new-model
