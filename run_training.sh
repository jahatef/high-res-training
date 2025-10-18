#!/bin/bash
#SBATCH -t 48:0:00
#SBATCH -N 4
#SBATCH -p gpu
#SBATCH -A PZS0622
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive


cd /users/PAS2312/jahatef/cardinal/high-res/
source setup.sh
cd high-res-training

OMP_NUM_THREADS=12 torchrun --nproc_per_node=4  train_4k_fsdp.py \
  --data-dir /fs/ess/PAS2699/jahatef/soybeans/ \
  --epochs 15 \
  --batch-size 3 \
  --mixed-precision \
  --lr 0.000001 \
  --min-lr 0.00000001 \
  --cpt-power 2 \
  --num-workers 8 \
  --output-dir ./checkpoints-soybeans-ablate/ \
  --use-wandb \
  --resume /users/PAS2312/jahatef/cardinal/high-res/high-res-training/checkpoints-stage3-ablate/vit_large_patch16_384_epoch4.pt
  #--resume /users/PAS2312/jahatef/ascend/vrwkv/high-res-training/checkpoints-stage3-ablate/vit_large_patch16_384_epoch1.pt
