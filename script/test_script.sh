#!/bin/sh

# general train test
python main.py --fold 1 --batch_size 256 --max_epoch 10 \
                --save_path './logs/general_test'

# online train test
python main.py --fold 1 --batch_size 256 --max_epoch 10 \
                --save_path './logs/online_test' \
                --hem_extract_mode 'hem-emb-online'


# offline train test
python offline_main.py --fold 1 --batch_size 256 --max_epoch 10 \
                    --save_path './logs/offline_test' \
                    --n_stage 2 \
                    --n_mini_fold 4 \
                    --n_dropout 3