#!/bin/sh
source ~/.bashrc
source activate NBME

ROOT="/users10/lyzhang/opt/tiger/NBME"
export PYTHONPATH="$HOME/opt/tiger/NBME"


# PRETRAIN="deberta_v3_large"
# PRETRAIN="structbert_large"
# PRETRAIN="bart_large"
# PRETRAIN="t5_efficient_large"
PRETRAIN="deberta_xlarge"
# PRETRAIN="roberta_large"


python ../src/Pretrain.py \
--train_path="$ROOT/language_model/tmp/train.txt" \
--valid_path="$ROOT/language_model/tmp/valid.txt" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--save_path="$ROOT/model/Pretrain/PMC_Task" \
# > ../log/Pretrain.log 2>&1 &