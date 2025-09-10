#!/bin/bash

BENCHMARK="workarena_l1"

LLM_CONFIG="azure/gpt-5-mini-2025-08-07"
# PARALLEL_BACKEND="sequential"
PARALLEL_BACKEND="ray"

HINT_TYPE="docs"    # human, llm, docs
HINT_INDEX_TYPE="sparse" # sparse, dense
HINT_QUERY_TYPE="goal" # goal, llm
HINT_NUM_RESULTS=5

HINT_INDEX_PATH="indexes/servicenow-docs-bm25"
# HINT_INDEX_PATH="indexes/servicenow-docs-embeddinggemma-300m"
HINT_RETRIEVER_PATH="google/embeddinggemma-300m"

N_JOBS=6

python experiments/hint/run_hinter_agent.py \
    --benchmark $BENCHMARK \
    --llm-config $LLM_CONFIG \
    --parallel-backend $PARALLEL_BACKEND \
    --n-jobs $N_JOBS \
    --hint-type $HINT_TYPE \
    --hint-index-type $HINT_INDEX_TYPE \
    --hint-query-type $HINT_QUERY_TYPE \
    --hint-index-path $HINT_INDEX_PATH \
    --hint-retriever-path $HINT_RETRIEVER_PATH \
    --hint-num-results $HINT_NUM_RESULTS \
    --relaunch