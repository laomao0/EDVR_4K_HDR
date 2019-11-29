#!/usr/bin/env bash

#      --master_port=1868 \
python -m torch.distributed.launch \
      --nproc_per_node=4 \
      train_parallel.py \
      --launcher pytorch \
      --opt /DATA7_DB7/data/4khdr/codes/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug_ssim_parral_predeblur.yml

