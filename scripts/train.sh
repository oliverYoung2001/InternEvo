#!/bin/bash

# set -x 

export PARTITION=Mix
export HOST=nico1
export CLUSTER_NAME=nico
# export PARTITION=octave
# export HOST=octave
# export CLUSTER_NAME=yes
export MASTER_ADDR=localhost
export NNODES=1
export GPUS_PER_NODE=8

export MASTER_PORT=$((RANDOM % 12000 + 10000))
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

SLURM_ARGS="
-p $PARTITION \
-w $HOST \
-N $NNODES \
--ntasks-per-node=$GPUS_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
--gpus-per-task=1 \
-K \
"
SLURM_ARGS="
-p $PARTITION \
-w $HOST \
-N $NNODES \
--ntasks-per-node=$GPUS_PER_NODE \
--gres=gpu:$GPUS_PER_NODE \
-K \
"

# export NCCL_DEBUG_SUBSYS=GRAPH
# export NCCL_DEBUG=INFO
srun $SLURM_ARGS \
python train.py \
    --config ./configs/7B_sft.py \
