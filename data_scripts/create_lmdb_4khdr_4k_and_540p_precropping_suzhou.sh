#!/usr/bin/env bash

python create_lmdb_4khdr_precropping_suzhou.py \
              --dataset 4KHDR \
              --mode both \
              --small 0 \
              --batch 100 \
              --name train_precropping