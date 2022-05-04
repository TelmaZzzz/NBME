#!/bin/sh
# source ~/.bashrc
# source activate NBME

ROOT="/users10/lyzhang/opt/tiger/NBME"
export PYTHONPATH="$HOME/opt/tiger/NBME"


PRETRAIN="deberta_v3_large"

python ../src/Base.py \
--predict \
--valid_path="$ROOT/data/sgcvs_valid" \
--data_path="$ROOT/data" \
--fold=1 \
--valid_batch_size=32 \
--fix_length=512 \
> ../log/Base_predict.log 2>&1 &

# fold0 0.8919 -> 0.8946
# fold1 0.8858 -> 0.8883
# fold2 0.8887 -> 0.8914
# fold3 0.8972 -> 0.8972
