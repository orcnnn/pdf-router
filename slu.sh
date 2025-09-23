#!/bin/bash

#SBATCH -J "Pdf-Router-Prod"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100x4q                        # kuyruk (partition/queue) adi

#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:4                    # ilave kaynak (1 gpu gerekli)
#SBATCH -o logs/%j_run.log                 # Output file path

echo ">>> Starting Marker and vLLM servers for Sanity Test..."

# ------- 1) Marker'ı başlat -------
sh start_marker.sh > marker.log 2>&1 & || true
echo "[INFO] Marker servers requested. Waiting 5s..."
sleep 5

sh start_vllm_2.sh > vllm.log 2>&1 & || true
echo "[INFO] VLLM servers requested. Waiting 600s..."
sleep 600

echo ">>> Starting the Python processing script..."
python main.py run1.yaml

echo ">>> Python processing completed."