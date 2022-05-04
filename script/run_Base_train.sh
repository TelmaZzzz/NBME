#!/bin/sh
# source ~/.bashrc
# source activate NBME

ROOT="/users10/lyzhang/opt/tiger/NBME"
export PYTHONPATH="$HOME/opt/tiger/NBME"


# PRETRAIN="deberta_v3_large"
PRETRAIN="deberta_v2_xlarge"
# PRETRAIN="roberta_large"
# PRETRAIN="structbert_large"
# PRETRAIN="bart_large"
# PRETRAIN="t5_efficient_large"
# PRETRAIN="deberta_xlarge"

python ../src/Base.py \
--train \
--debug \
--train_path="$ROOT/data/sgcvs_train" \
--valid_path="$ROOT/data/sgcvs_valid" \
--data_path="$ROOT/data" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--tokenizer_path="$HOME/model/$PRETRAIN" \
--model_save="$ROOT/model/Base" \
--fold=1 \
--epoch=50 \
--lr=7e-6 \
--min_lr=1e-6 \
--eval_step=50 \
--valid_batch_size=16 \
--train_batch_size=6 \
--fix_length=512 \
--scheduler="get_cosine_schedule_with_warmup" \
--mask_prob=0.0 \
--mask_ratio=0.0 \
--mode="base" \
--fgm \
--deberta \
> ../log/Base_train.log 2>&1 &
# --da \
# --da_path="$ROOT/data/pseudo/submission_0_fold_0.csv" \