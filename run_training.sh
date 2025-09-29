#!/bin/bash
#SBATCH -t 01:0:00
#SBATCH -N 1
#SBATCH -p quad
#SBATCH -A PZS0622
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --exclusive


cd /users/PAS2312/jahatef/ascend/vrwkv
source setup.sh
cd high-res-training

OMP_NUM_THREADS=12 torchrun --nproc_per_node=4  train_4k_fsdp.py \
  --data-dir /fs/ess/PAS2699/jahatef/imagenet-1k-wds-subset3-super \
  --epochs 15 \
  --batch-size 12 \
  --mixed-precision \
  --lr 0.000001 \
  --min-lr 0.00000001 \
  --cpt-power 2 \
  --num-workers 8 \
  --output-dir ./checkpoints-stage3-ablate/ \
  --use-wandb \
  --resume /users/PAS2312/jahatef/ascend/vrwkv/checkpoints-stage2-ablate/vit_large_patch16_384_epoch10.pt
