#!/bin/bash

#SBATCH -J "Pdf-Router-Prod"                         # isin adi

#SBATCH -A sdmmtv                           # account / proje adi
#SBATCH -p a100x4q                        # kuyruk (partition/queue) adi

#SBATCH -n 64                            # cekirdek / islemci sayisi
#SBATCH -N 1                              # bilgisayar sayisi
#SBATCH --gres=gpu:4                    # ilave kaynak (1 gpu gerekli)
#SBATCH -o logs/%j_run.log                 # Output file path

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Submit time: $(date)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.cuda.device_count() =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

sh run_job.sh

echo "Job completed at $(date)"
