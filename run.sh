#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-baseline}"
DATASET="${2:-ETTh1}"
CONFIG="${CONFIG:-config_deepseek.yaml}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,4}"
MPL_DIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

case "$MODE" in
  baseline)
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    MPLCONFIGDIR="$MPL_DIR" \
    ORCHESTRATION_MODE=deterministic \
    python run_experiment.py --config "$CONFIG" --dataset "$DATASET"
    ;;

  llm)
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    MPLCONFIGDIR="$MPL_DIR" \
    ORCHESTRATION_MODE=llm \
    ORCHESTRATION_BACKEND=langgraph \
    python run_experiment.py --config "$CONFIG" --dataset "$DATASET"
    ;;

  llm-direct)
    HTTP_PROXY= \
    HTTPS_PROXY= \
    http_proxy= \
    https_proxy= \
    ALL_PROXY= \
    all_proxy= \
    NO_PROXY='*' \
    no_proxy='*' \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    MPLCONFIGDIR="$MPL_DIR" \
    ORCHESTRATION_MODE=llm \
    ORCHESTRATION_BACKEND=langgraph \
    python run_experiment.py --config "$CONFIG" --dataset "$DATASET"
    ;;

  *)
    echo "Usage:"
    echo "  ./run.sh baseline [DATASET]    # deterministic baseline"
    echo "  ./run.sh llm [DATASET]         # LLM orchestration via LangGraph"
    echo "  ./run.sh llm-direct [DATASET]  # LLM orchestration without proxy env vars"
    echo
    echo "Optional env overrides:"
    echo "  CONFIG=config.yaml CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib ./run.sh llm-direct ETTh1"
    exit 2
    ;;
esac
