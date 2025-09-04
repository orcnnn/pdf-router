CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-VL-72B-Instruct --port 8000 \
 --max-model-len 16384 --tensor-parallel-size 4 --gpu_memory_utilization 0.8 > logs/vllm_server.log 2>&1 &