vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --trust-remote-code \
  --port 8000 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90