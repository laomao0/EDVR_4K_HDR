#!/usr/bin/env bash

python create_lmdb_4khdr_suzhou.py \
                        --dataset 4KHDR \
                        --mode train_540p \
                        --small 0 \
                        --batch 100 \
                        --name train_540p_all

