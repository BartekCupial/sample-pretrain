#!/bin/bash

#SBATCH --job-name=NetHackRL
#SBATCH --output=logger-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --gres=gpu:2
#SBATCH --qos=normal  # test (1 GPU, 1 hour), quick (1 GPU, 1 day), normal (2 GPU, 2 days), big (4 GPU, 7 days)
#SBATCH --partition=dgx  # dgxa100 (mini-servers), dgxteam (dgx1 for Team-Net), dgxmatinf (dgx2 for WMiI), dgx (dgx1 and dgx2)

#squeue -u ${USER} --Format "JobID:.6 ,Partition:.4 ,Name:.10 ,StateCompact:.2 ,TimeUsed:.11 ,Qos:.7 ,TimeLeft:.11 ,ReasonList:.16 ,Command:.40"

singularity exec --nv \
    -H /home/gmcupial/sample-pretrain \
    --env WANDB_API_KEY=d2f9309c1cee36dc7ad726c57e4eba04974d9914 \
    --env WANDBPWD=$PWD \
    -B /shared/results/gmcupial/dungeons/nle:/nle \
    -B /shared/results/gmcupial/dungeons/ttyrecs:/ttyrecs \
    -B $TMPDIR:/tmp \
    --bind /usr/lib/x86_64-linux-gnu/libGL.so.1:/usr/lib/x86_64-linux-gnu/libGL.so.1 \
    --bind /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 \
    --bind /usr/lib/x86_64-linux-gnu/libGLX.so.0:/usr/lib/x86_64-linux-gnu/libGLX.so.0 \
    /shared/results/gmcupial/sp.sif \
    ./train.sh
