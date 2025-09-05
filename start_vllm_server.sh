
export VLLM_API_URL="http://10.128.41.141:8000/v1"   
export VLLM_API_KEY="dummy"
# Eski süreçleri kapat
pkill -f "vllm serve" 2>/dev/null

# (İsteğe bağlı) P2P cache'i temizle (yeniden ölçsün)
rm -f ~/.cache/vllm/gpu_p2p_access_cache_for_0,1,2,3.json

# Ortam
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# "expandable_segments" bu makinada desteklenmiyor uyarısı veriyor; kaldırıyoruz:
unset PYTORCH_CUDA_ALLOC_CONF

# Sunucu (eager + custom all-reduce kapalı)
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  --swap-space 24 \
  --disable-custom-all-reduce \
  --enforce-eager \
  > logs/vllm_server.log 2>&1 & disown

