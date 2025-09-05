#!/bin/bash
echo ">>> Starting Marker and VLLM servers for Sanity Test..."

# Sunucuları başlat (eğer zaten çalışmıyorsa)
# Bekleme süreleri test için düşürüldü
sh start_marker_1.sh
echo "Marker servers started. Sleeping for 200 seconds..."
sleep 200

sbatch start_vllm_server.sh
echo "VLLM servers started. Sleeping for 800 seconds..."
sleep 1000




echo ">>> Starting the Python processing script..."
python main.py run.yaml

echo ">>> Sanity Test script finished."
