#!/bin/bash
# STraTS on a Chronicle cohort: overall survival (OS). See "Benchmarking on Chronicle data" in README.md.
#   CHRONICLE_DATA_DIR  output directory of chronicle-preprocess (<prefix>_{train,val,test}.csv)
#   GLOBAL_FORMAT       feature schema used for the Chronicle run
# e.g. CHRONICLE_DATA_DIR=.../data/essen_new_labels GLOBAL_FORMAT=.../global_format_essen_12_03.csv ./run_chronicle.sh

cd src/
python preprocess_chronicle.py --data_dir ${CHRONICLE_DATA_DIR:?set CHRONICLE_DATA_DIR} \
    --global_format ${GLOBAL_FORMAT:?set GLOBAL_FORMAT}

# hyperparameters of the STraTS (ss-) runs in run_main.sh (no self-supervised pretraining)
python main.py --dataset chronicle --train_frac 1.0 --model_type strats --hid_dim 64 --num_layers 2 \
    --num_heads 16 --dropout 0.2 --attention_dropout 0.2 --lr 5e-4
