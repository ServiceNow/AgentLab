#!/bin/bash

# Launch script for JephHinter workflow
# This script runs the complete workflow:
# 1. Run experiments without hints
# 2. Generate hints from traces using JephHinter
# 3. Run experiments with hints enabled

set -e  # Exit on any error

export SNOW_INSTANCE_PWD='SNOW_INSTANCE_PWD'
export SNOW_INSTANCE_URL='https://researchworkarenademo.service-now.com'
export SNOW_INSTANCE_UNAME='admin'

export OPENAI_API_KEY="OPENAI_API_KEY"
export ANTHROPIC_API_KEY="ANTHROPIC_API_KEY"
# Configuration
EXP_ROOT_NO_HINT="PATH_TO_TRACES"
EXP_ROOT_WITH_HINT="${EXP_ROOT_NO_HINT}-with_jeph"
HINT_DB_PATH="${EXP_ROOT_NO_HINT}/hint_db_updated.csv"

export MINIWOB_URL="MINIWOB_URL"

echo "=== JephHinter Workflow ==="
echo "No-hint experiments: ${EXP_ROOT_NO_HINT}"
echo "With-hint experiments: ${EXP_ROOT_WITH_HINT}"
echo "Hint database: ${HINT_DB_PATH}"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "${EXP_ROOT_NO_HINT}"
# mkdir -p "${EXP_ROOT_WITH_HINT}"
echo "Directories created successfully!"
echo ""

# Step 1: Run experiments without hints
export AGENTLAB_EXP_ROOT="${EXP_ROOT_NO_HINT}"

echo "=== Step 1: Running experiments without hints ==="
echo "Running main_jephhinter.py with exp-root=${EXP_ROOT_NO_HINT} (hints disabled by default)"
python main_jephhinter.py \
    --exp-root "${EXP_ROOT_NO_HINT}" \
    --hint-db-path "${HINT_DB_PATH}"

echo "Step 1 complete!"
echo ""

# Number of recursive hinting rounds
NUM_ROUNDS=3
PREV_ROOT="${EXP_ROOT_NO_HINT}"

for ((i=1; i<=NUM_ROUNDS; i++)); do
    echo "=== Recursive Hinting Round $i ==="

    # Step 2: Generate hints using JephHinter (from PREV_ROOT)
    echo "=== Step 2: Generating hints from traces ==="
    echo "Running JephHinter with root-dir=${PREV_ROOT}"
    python src/agentlab/agents/tool_use_agent/jeph_hinter.py \
        --root-dir "${PREV_ROOT}" \
        --output-path "${HINT_DB_PATH}"

    echo "Step 2 complete!"
    echo ""

    # Step 3: Run experiments with hints enabled
    EXP_ROOT_WITH_HINT_ROUND="${EXP_ROOT_WITH_HINT}_round${i}"
    echo "=== Step 3: Running experiments with hints ==="
    export AGENTLAB_EXP_ROOT="${EXP_ROOT_WITH_HINT_ROUND}"

    # Check if hint database exists
    if [ ! -f "${HINT_DB_PATH}" ]; then
        echo "Error: Hint database not found at ${HINT_DB_PATH}"
        echo "Please ensure Step 2 completed successfully."
        exit 1
    fi

    echo "Running main_jephhinter.py with exp-root=${EXP_ROOT_WITH_HINT_ROUND} (hints enabled)"
    python main_jephhinter.py \
        --exp-root "${EXP_ROOT_WITH_HINT_ROUND}" \
        --use-task-hint \
        --hint-db-path "${HINT_DB_PATH}"

    echo "Step 3 complete!"
    echo ""

    # Prepare for next round
    PREV_ROOT=${EXP_ROOT_WITH_HINT_ROUND}
    # EXP_ROOT_WITH_HINT remains the same base for next round, but will be suffixed again

done

echo "=== JephHinter Workflow Complete ==="
echo "Results without hints: ${EXP_ROOT_NO_HINT}"
echo "Results with hints: ${EXP_ROOT_WITH_HINT}"
echo "Hint database: ${HINT_DB_PATH}" 