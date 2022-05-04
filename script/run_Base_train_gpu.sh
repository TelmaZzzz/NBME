#!/bin/sh
source /users10/lyzhang/.bashrc
source activate NBME

ROOT="/users10/lyzhang/opt/tiger/NBME"
export PYTHONPATH="$HOME/opt/tiger/NBME"


PRETRAIN="deberta_v3_large"
# PRETRAIN="roberta_large"
# PRETRAIN="deberta_v2_xlarge"
# PRETRAIN="structbert_large"
# PRETRAIN="t5_efficient_large"
# PRETRAIN="bart_large"
# PRETRAIN="deberta_xlarge"

SCHEDULER="get_cosine_schedule_with_warmup"
# SCHEDULER="MultiStepLR"

# "$HOME/model/$PRETRAIN"
# --pretrain_path="/users10/lyzhang/opt/tiger/NBME/model/Pretrain/$PRETRAIN" \

python ../src/Base.py \
--train \
--train_path="$ROOT/data/sgcvs_train" \
--valid_path="$ROOT/data/sgcvs_valid" \
--data_path="$ROOT/data" \
--pretrain_path="$HOME/model/$PRETRAIN" \
--tokenizer_path="$HOME/model/$PRETRAIN" \
--model_save="$ROOT/model/Base" \
--fold=0 \
--epoch=3 \
--lr=1e-5 \
--min_lr=1e-6 \
--eval_step=1000 \
--valid_batch_size=32 \
--train_batch_size=10 \
--opt_step=1 \
--fix_length=512 \
--dropout=0.1 \
--scheduler="$SCHEDULER" \
--seed=520794 \
--fgm \
--da \
--da_path="$ROOT/data/pseudo/submission_0_fold_3.csv" \
--deberta \
# --mode="char" \
# --record="pretrained" \
# --swa \
# --swa_start_step=8000 \
# --swa_update_step=100 \
# --swa_lr=3e-5 \
# --ema \
# --mode="generate" \
# --offset_fix \
