#!/usr/bin/env zsh



torchrun --nproc_per_node=1 --master_port=22002 "/data/train.py" 