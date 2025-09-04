#!/bin/bash
echo ">>> Starting Marker and VLLM servers for Sanity Test..."

# Sunucuları başlat (eğer zaten çalışmıyorsa)
# Bekleme süreleri test için düşürüldü
sh start_marker_1.sh
echo "Marker servers started. Sleeping for 200 seconds..."
sleep 200

sh start_vllm.sh
echo "VLLM servers started. Sleeping for 800 seconds..."
sleep 800




echo ">>> Starting the Python processing script..."
python main.py run.yaml

echo ">>> Sanity Test script finished."
