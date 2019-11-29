#!/usr/bin/env bash

#      --master_port=1868 \
/DATA5_DB8/data/4khdr/torch11/bin/python -m torch.distributed.launch \
      --nproc_per_node=4 \
      train_parallel.py \
      --launcher pytorch \
      --opt /DATA7_DB7/data/4khdr/codes/codes/codes/options/train/train_EDVR_L_Preload_30000_nobug_nf192_rb80_ssim_parall.yml

