#!/bin/bash

# Launch script for JephHinter workflow
# This script runs the complete workflow:
# 1. Run experiments without hints
# 2. Generate hints from traces using JephHinter
# 3. Run experiments with hints enabled

set -e  # Exit on any error

# Configuration
BASE_DIR="/Users/had.nekoeiqachkanloo/hadi/AgentLab"
EXP_ROOT_NO_HINT="${BASE_DIR}/agentlab_results_no_hint_miniwob10"
EXP_ROOT_WITH_HINT="${BASE_DIR}/agentlab_results_no_hint/with_jeph"
HINT_DB_PATH="${EXP_ROOT_NO_HINT}/hint_db_updated.csv"

echo "=== JephHinter Workflow ==="
echo "Base directory: ${BASE_DIR}"
echo "No-hint experiments: ${EXP_ROOT_NO_HINT}"
echo "With-hint experiments: ${EXP_ROOT_WITH_HINT}"
echo "Hint database: ${HINT_DB_PATH}"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p "${EXP_ROOT_NO_HINT}"
mkdir -p "${EXP_ROOT_WITH_HINT}"
echo "Directories created successfully!"
echo ""

# Step 1: Run experiments without hints
echo "=== Step 1: Running experiments without hints ==="
echo "Running main_jephhinter.py with exp-root=${EXP_ROOT_NO_HINT} (hints disabled by default)"
python main_jephhinter.py \
    --exp-root "${EXP_ROOT_NO_HINT}" \
    --hint-db-path "${HINT_DB_PATH}"

echo "Step 1 complete!"
echo ""

# Step 2: Generate hints using JephHinter
echo "=== Step 2: Generating hints from traces ==="
echo "Running JephHinter with root-dir=${EXP_ROOT_NO_HINT}"
python src/agentlab/agents/tool_use_agent/jeph_hinter.py \
    --root-dir "${EXP_ROOT_NO_HINT}" \
    --output-path "${HINT_DB_PATH}"

echo "Step 2 complete!"
echo ""

# Step 3: Run experiments with hints enabled
echo "=== Step 3: Running experiments with hints ==="

# Check if hint database exists
if [ ! -f "${HINT_DB_PATH}" ]; then
    echo "Error: Hint database not found at ${HINT_DB_PATH}"
    echo "Please ensure Step 2 completed successfully."
    exit 1
fi

echo "Running main_jephhinter.py with exp-root=${EXP_ROOT_WITH_HINT} (hints enabled)"
python main_jephhinter.py \
    --exp-root "${EXP_ROOT_WITH_HINT}" \
    --use-task-hint \
    --hint-db-path "${HINT_DB_PATH}"

echo "Step 3 complete!"
echo ""

echo "=== JephHinter Workflow Complete ==="
echo "Results without hints: ${EXP_ROOT_NO_HINT}"
echo "Results with hints: ${EXP_ROOT_WITH_HINT}"
echo "Hint database: ${HINT_DB_PATH}" 