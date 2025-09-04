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

sh run_job.sh

echo "Job completed at $(date)"
