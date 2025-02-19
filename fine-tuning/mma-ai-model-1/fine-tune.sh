#!/bin/bash

# python -m transformers.convert_graph_to_gguf --model mma-ai-model-1 --output mma-ai-model-1.gguf
# python -m llama_cpp.convert_hf_to_gguf --model_dir mma-ai-model-1 --outfile mma-ai-model-1.gguf

wget -O mma-ai-model-1/config.json https://huggingface.co/cognitivecomputations/TinyDolphin-2.8-1.1b/raw/main/config.json

# python transformers/src/transformers/models/llama/convert_llama_weights_to_gguf.py \
python /opt/homebrew/Cellar/llama.cpp/4739/bin/convert_hf_to_gguf.py \
    --model_path ../mma-ai-model-1 \
    --output_path ../mma-ai-model-1.gguf \
    --quantization q4_0  # Choose quantization (q4_0, q5_0, q8_0, etc.)