#!/bin/bash

BENCHMARK="workarena_l1"

LLM_CONFIG="azure/gpt-5-mini-2025-08-07"
# PARALLEL_BACKEND="sequential"
PARALLEL_BACKEND="ray"

N_JOBS=5
N_RELAUNCH=3

python experiments/generic/run_generic_agent.py \
    --benchmark $BENCHMARK \
    --llm-config $LLM_CONFIG \
    --parallel-backend $PARALLEL_BACKEND \
    --n-jobs $N_JOBS \
    --n-relaunch $N_RELAUNCH