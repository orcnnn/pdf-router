#!/bin/bash

#SBATCH -J "vllm"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100x4q                        # kuyruk (partition/queue) adi

#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:4                    # ilave kaynak (1 gpu gerekli)
#SBATCH -o logs/%j_run.log                 # Output file path

set -euo pipefail

# Slurm submit klasörüne geç (paylaşımlı fs varsayımı)
cd "${SLURM_SUBMIT_DIR:-$PWD}"

PORT="${PORT:-8000}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log"

echo "[INFO] Launching vLLM on port ${PORT}, logging to ${LOG_FILE}"

# ---- VLLM SUNUCUYU BAŞLAT (KENDİ ARGÜMANLARINLA) ----
# ÖRNEK: argümanları kendine göre uyarlayabilirsin
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --trust-remote-code \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.90 \
  --swap-space 16 \
  --disable-custom-all-reduce \
  --enforce-eager \
  > "${LOG_FILE}" 2>&1 &

VLLM_PID=$!
echo "${VLLM_PID}" > "${LOG_DIR}/vllm_${SLURM_JOB_ID}.pid"
echo "[INFO] vLLM PID=${VLLM_PID}"

# ---- HAZIRLIK/GÖZETLEYİCİ: PORT AÇILINCA ENDPOINT JSON YAZ ----
(
  for i in {1..240}; do
    if ss -ltn | awk '{print $4}' | grep -q ":${PORT}\$"; then
      HOST=$(hostname -f)
      IP=$(ip -4 route get 1.1.1.1 | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
      URL="http://${HOST}:${PORT}/v1"

      EP_FILE="${SLURM_SUBMIT_DIR:-$PWD}/vllm_endpoint_${SLURM_JOB_ID}.json"
      printf '{"host":"%s","ip":"%s","port":%s,"url":"%s","job_id":"%s","pid":%s}\n' \
             "$HOST" "$IP" "$PORT" "$URL" "$SLURM_JOB_ID" "$VLLM_PID" > "$EP_FILE"
      ln -sf "$EP_FILE" "${SLURM_SUBMIT_DIR:-$PWD}/vllm_endpoint_latest.json"

      echo "[INFO] Endpoint file written: ${EP_FILE}"
      exit 0
    fi
    sleep 1
  done
  echo "[WARN] vLLM port ${PORT} did not open in time; no endpoint file written."
  exit 1
) &

# ---- JOB'U vLLM'YE BAĞLA (vLLM kapanınca job biter) ----
wait "${VLLM_PID}"

