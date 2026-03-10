#!/usr/bin/env bash
# =============================================================================
# run_use_case_comparison.sh
#
# Orchestrates the 11-model × 2-agent × 5-benchmark eval comparison.
#
# Open-source models run sequentially (one vLLM server at a time).
# Cloud models run via API (no GPU needed).
#
# Usage:
#   ./scripts/run_use_case_comparison.sh               # Full comparison
#   ./scripts/run_use_case_comparison.sh --cloud-only   # Cloud models only
#   ./scripts/run_use_case_comparison.sh --oss-only     # Open-source models only
#   ./scripts/run_use_case_comparison.sh --model "GLM-4.7-Flash"  # Single OSS model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# === Configuration ===
EVAL_CMD="uv run jarvis eval run"
CONFIGS_DIR="src/openjarvis/evals/configs"
VLLM_PORT=8000
VLLM_HEALTH_URL="http://localhost:${VLLM_PORT}/health"
VLLM_STARTUP_TIMEOUT=300  # seconds
VLLM_PID=""
LOG_DIR="logs/eval-comparison"

# Open-source models: name|hf_id|tensor_parallel_size
OSS_MODELS=(
    "GLM-4.7-Flash|unsloth/GLM-4.7-Flash-GGUF|2"
    "Qwen3.5-122B|unsloth/Qwen3.5-122B-A10B-GGUF|4"
    "gpt-oss-120b|unsloth/gpt-oss-120b-GGUF|4"
    "GLM-5|unsloth/GLM-5-GGUF|4"
    "Qwen3.5-397B|unsloth/Qwen3.5-397B-A17B-GGUF|8"
)

# === Helpers ===

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

ensure_dirs() {
    mkdir -p "$LOG_DIR"
    mkdir -p results/use-cases-opensource-orchestrator
    mkdir -p results/use-cases-opensource-openhands
    mkdir -p results/use-cases-cloud-orchestrator
    mkdir -p results/use-cases-cloud-openhands
}

start_vllm() {
    local model_id="$1"
    local tp_size="$2"
    local log_file="$3"

    log "Starting vLLM: model=$model_id tp=$tp_size"
    vllm serve "$model_id" \
        --tensor-parallel-size "$tp_size" \
        --port "$VLLM_PORT" \
        --disable-log-requests \
        > "$log_file" 2>&1 &
    VLLM_PID=$!
    log "vLLM PID: $VLLM_PID"
}

wait_for_vllm() {
    local elapsed=0
    log "Waiting for vLLM to be ready at $VLLM_HEALTH_URL ..."
    while [ $elapsed -lt $VLLM_STARTUP_TIMEOUT ]; do
        if curl -sf "$VLLM_HEALTH_URL" > /dev/null 2>&1; then
            log "vLLM is ready (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "ERROR: vLLM failed to start within ${VLLM_STARTUP_TIMEOUT}s"
    stop_vllm
    return 1
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        log "Stopping vLLM (PID $VLLM_PID)"
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
    # Also kill any lingering vllm serve processes on our port
    pkill -f "vllm serve.*--port $VLLM_PORT" 2>/dev/null || true
    sleep 2
}

run_eval() {
    local config="$1"
    local filter="${2:-}"
    local start_time end_time

    start_time=$(date +%s)
    log "Running: $config ${filter:+(filter: $filter)}"

    if [ -n "$filter" ]; then
        $EVAL_CMD -c "$config" --model-filter "$filter" -v || {
            log "WARNING: eval run failed for $config (filter: $filter)"
            return 1
        }
    else
        $EVAL_CMD -c "$config" -v || {
            log "WARNING: eval run failed for $config"
            return 1
        }
    fi

    end_time=$(date +%s)
    log "Completed in $((end_time - start_time))s: $config ${filter:+(filter: $filter)}"
}

# === Execution Phases ===

run_oss_models() {
    local model_filter="${1:-}"

    for entry in "${OSS_MODELS[@]}"; do
        IFS='|' read -r name model_id tp_size <<< "$entry"

        # Skip if --model flag set and doesn't match
        if [ -n "$model_filter" ] && [[ "$name" != *"$model_filter"* ]]; then
            continue
        fi

        log "========================================"
        log "OSS MODEL: $name ($model_id, TP=$tp_size)"
        log "========================================"

        local log_file="$LOG_DIR/vllm_${name}.log"

        # Start vLLM for this model
        start_vllm "$model_id" "$tp_size" "$log_file"
        if ! wait_for_vllm; then
            log "Skipping $name — vLLM failed to start"
            continue
        fi

        # Run orchestrator config (filtered to this model)
        run_eval "$CONFIGS_DIR/use_case_opensource_orchestrator.toml" "$name" || true

        # Run openhands config (filtered to this model)
        run_eval "$CONFIGS_DIR/use_case_opensource_openhands.toml" "$name" || true

        # Stop vLLM
        stop_vllm

        log "Completed all benchmarks for $name"
    done
}

run_cloud_models() {
    log "========================================"
    log "CLOUD MODELS"
    log "========================================"

    # Source .env for API keys
    if [ -f "$PROJECT_DIR/.env" ]; then
        # shellcheck disable=SC1091
        set -a && source "$PROJECT_DIR/.env" && set +a
        log "Loaded API keys from .env"
    else
        log "WARNING: .env not found — cloud runs may fail"
    fi

    # Orchestrator agent
    run_eval "$CONFIGS_DIR/use_case_cloud_orchestrator.toml" || true

    # CodeAct agent
    run_eval "$CONFIGS_DIR/use_case_cloud_openhands.toml" || true

    log "Completed cloud model runs"
}

run_comparison() {
    log "========================================"
    log "COMPARISON"
    log "========================================"

    local result_files=()
    for dir in results/use-cases-{opensource-orchestrator,opensource-openhands,cloud-orchestrator,cloud-openhands}; do
        if [ -d "$dir" ]; then
            while IFS= read -r -d '' f; do
                result_files+=("$f")
            done < <(find "$dir" -name '*.jsonl' -print0 2>/dev/null)
        fi
    done

    if [ ${#result_files[@]} -eq 0 ]; then
        log "No result files found — skipping comparison"
        return
    fi

    log "Comparing ${#result_files[@]} result files"
    uv run jarvis eval compare "${result_files[@]}" || {
        log "WARNING: comparison failed"
    }
}

# === Main ===

main() {
    local mode="all"
    local model_filter=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --cloud-only) mode="cloud"; shift ;;
            --oss-only)   mode="oss"; shift ;;
            --model)      model_filter="$2"; mode="oss"; shift 2 ;;
            --compare)    mode="compare"; shift ;;
            -h|--help)
                echo "Usage: $0 [--cloud-only|--oss-only|--model NAME|--compare]"
                echo ""
                echo "Options:"
                echo "  --cloud-only   Run cloud models only (no GPU needed)"
                echo "  --oss-only     Run open-source models only (needs GPU + vLLM)"
                echo "  --model NAME   Run a single open-source model by name substring"
                echo "  --compare      Only run comparison on existing results"
                exit 0
                ;;
            *) echo "Unknown option: $1"; exit 1 ;;
        esac
    done

    log "Starting eval comparison (mode=$mode)"
    ensure_dirs

    # Trap to ensure vLLM is stopped on exit
    trap stop_vllm EXIT

    case "$mode" in
        all)
            run_oss_models ""
            run_cloud_models
            run_comparison
            ;;
        oss)
            run_oss_models "$model_filter"
            ;;
        cloud)
            run_cloud_models
            ;;
        compare)
            run_comparison
            ;;
    esac

    log "Done!"
}

main "$@"
