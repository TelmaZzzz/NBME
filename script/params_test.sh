#!/bin/sh
source ~/.bashrc
source activate NBME

ROOT="/users10/lyzhang/opt/tiger/NBME"
export PYTHONPATH="$HOME/opt/tiger/NBME"

python ../src/Ensemble_Test.py
