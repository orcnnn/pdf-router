#!/bin/bash
set -euo pipefail

echo ">>> Starting Marker and vLLM servers for Sanity Test..."

# Script dizinine geç (log/endpoint dosyaları burada oluşsun)
cd "$(dirname "$0")"

# ------- 1) Marker'ı başlat -------
sh start_marker.sh || true
echo "[INFO] Marker servers requested. Waiting 5s..."
sleep 5

# ------- 2) vLLM job'unu Slurm'da başlat -------
SUBMIT_OUT=$(sbatch start_vllm_server.sh)
echo "[INFO] sbatch output: $SUBMIT_OUT"

# Slurm farklı formatlar basabilir; rakamları çekelim
JID=$(echo "$SUBMIT_OUT" | grep -oE '[0-9]+' | tail -1)
if [[ -z "${JID}" ]]; then
  echo "[ERROR] Could not parse Job ID from sbatch output."
  exit 1
fi
echo "[INFO] vLLM Job ID: ${JID}"

# ------- 3) Endpoint dosyasını bekle -------
EP_FILE="vllm_endpoint_${JID}.json"
LATEST_FILE="vllm_endpoint_latest.json"
TIMEOUT=1200   # saniye (10 dk)
SLEEP=2

echo "[INFO] Waiting for endpoint file: ${EP_FILE} (timeout=${TIMEOUT}s)"
elapsed=0
while [[ ! -f "$EP_FILE" && $elapsed -lt $TIMEOUT ]]; do
  sleep $SLEEP
  elapsed=$((elapsed + SLEEP))
done

# ------- 4) URL/Host belirleme (ÖNCE latest.json'dan IP'yi dene) -------
# Not: PORT env ile override edilebilir, yoksa dosyadaki port, o da yoksa 8000 kullanılır.
PORT_DEFAULT=8000
PORT_FROM_FILE=""
IP_FROM_FILE=""
VLLM_API_URL=""

if [[ -f "$LATEST_FILE" ]]; then
  echo "[INFO] Using IP from ${LATEST_FILE}"
  if command -v jq >/dev/null 2>&1; then
    IP_FROM_FILE=$(jq -r '.ip // ( .url | capture("://(?<h>[^:/]+)").h ) // empty' "$LATEST_FILE")
    PORT_FROM_FILE=$(jq -r '.port // ( .url | capture(":(?<p>[0-9]+)") | .p ) // empty' "$LATEST_FILE" 2>/dev/null || echo "")
  else
    # jq yoksa Python ile çıkar
    IP_FROM_FILE=$(python - "$LATEST_FILE" <<'PY'
import json, sys, re
d=json.load(open(sys.argv[1]))
u=d.get("url","")
ip=d.get("ip")
if not ip:
    m=re.search(r'://([^:/]+)', u)
    ip=m.group(1) if m else ""
print(ip)
PY
)
    PORT_FROM_FILE=$(python - "$LATEST_FILE" <<'PY'
import json, sys, re
d=json.load(open(sys.argv[1]))
u=d.get("url","")
p=d.get("port")
if p is None:
    m=re.search(r':(\d+)', u)
    p=m.group(1) if m else ""
print(p)
PY
)
  fi
fi

# PORT önceliği: $PORT (env) > latest.json'daki port > 8000
PORT="${PORT:-${PORT_FROM_FILE:-$PORT_DEFAULT}}"

if [[ -n "${IP_FROM_FILE}" ]]; then
  VLLM_API_URL="http://${IP_FROM_FILE}:${PORT}/v1"
  echo "[INFO] Built URL from latest.json IP: ${VLLM_API_URL}"
fi

# Eğer latest.json'dan URL kurulamadıysa EP_FILE veya fallback kullan
if [[ -z "${VLLM_API_URL}" ]]; then
  if [[ -f "$EP_FILE" ]]; then
    echo "[INFO] Endpoint file found: ${EP_FILE}"
    if command -v jq >/dev/null 2>&1; then
      VLLM_API_URL=$(jq -r .url "$EP_FILE")
    else
      VLLM_API_URL=$(python - <<PY
import json
print(json.load(open("$EP_FILE"))["url"])
PY
)
    fi
  else
    echo "[WARN] Endpoint file was not created in time. Falling back to node discovery."
    NODE_RAW=""
    # squeue bazen boş dönebilir; kısa bir beklemeyle tekrar dene
    for i in {1..60}; do
      NODE_RAW=$(squeue -j "$JID" -h -o "%N" || true)
      [[ -n "$NODE_RAW" ]] && break
      sleep 2
    done
    if [[ -z "$NODE_RAW" ]]; then
      echo "[ERROR] Could not obtain node for Job $JID via squeue."
      exit 1
    fi
    NODE=$(scontrol show hostnames "$NODE_RAW" | head -n1)
    echo "[INFO] vLLM node: $NODE"
    VLLM_API_URL="http://${NODE}:${PORT}/v1"
  fi
fi

export VLLM_API_URL
export VLLM_API_KEY="${VLLM_API_KEY:-dummy}"   # boş olmasın diye dummy koy

echo "[INFO] VLLM_API_URL=${VLLM_API_URL}"

# ------- 5) Sağlık kontrolü (endpoint açılana kadar bekle) -------
echo "[INFO] Waiting for vLLM endpoint to become ready..."
READY_TIMEOUT=300
elapsed=0
until curl -sSf "${VLLM_API_URL}/models" >/dev/null 2>&1; do
  sleep 2
  elapsed=$((elapsed + 2))
  if [[ $elapsed -ge $READY_TIMEOUT ]]; then
    echo "[ERROR] vLLM endpoint did not become ready in ${READY_TIMEOUT}s."
    LOG_CANDIDATE="logs/vllm_server_${JID}.log"
    [[ -f "$LOG_CANDIDATE" ]] && { echo "---- vLLM LOG (tail) ----"; tail -n 200 "$LOG_CANDIDATE"; }
    exit 1
  fi
done
echo "[INFO] vLLM endpoint is reachable."

# ------- 6) Python işini çalıştır -------
echo ">>> Starting the Python processing script..."
python main.py run.yaml

echo ">>> Python processing completed."

# ------- 7) vLLM job'unu kapat -------
echo ">>> Stopping vLLM server..."
if scancel "$JID" 2>/dev/null; then
  echo "[INFO] vLLM job $JID cancelled successfully."
else
  echo "[WARN] Failed to cancel vLLM job $JID (may have already finished)."
fi

# ------- 8) Marker servislerini kapat -------
echo ">>> Stopping Marker servers..."
pkill -f "marker_server" || true
echo "[INFO] Marker servers stopped."

echo ">>> Sanity Test script finished."

