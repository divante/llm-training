#!/usr/bin/env bash
# Launch vLLM instances for model serving on Strix Halo.
#
# Usage:
#   ./scripts/serve.sh --session coding       # Start code + research instances
#   ./scripts/serve.sh --model code           # Start single model instance
#   ./scripts/serve.sh --stop-all             # Stop all vLLM instances
#   ./scripts/serve.sh --list                 # List running instances
#
# Requires: vLLM >= 0.11.0, yq (for YAML parsing)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${ROOT_DIR}/config/serve/instances.yaml"
PID_DIR="${ROOT_DIR}/logs/serve_pids"

mkdir -p "$PID_DIR"

# --- Helpers ---

die() { echo "ERROR: $*" >&2; exit 1; }

check_deps() {
    command -v python3 >/dev/null 2>&1 || die "python3 not found"
    # Use python to parse YAML instead of requiring yq
}

parse_yaml() {
    # Parse instances.yaml using Python (more portable than yq)
    python3 -c "
import yaml, json, sys
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(json.dumps(cfg))
"
}

get_instance_config() {
    local model_name="$1"
    python3 -c "
import yaml, json
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
inst = cfg.get('instances', {}).get('$model_name')
if inst:
    print(json.dumps(inst))
else:
    print('null')
"
}

get_session_models() {
    local session_name="$1"
    python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
models = cfg.get('sessions', {}).get('$session_name', [])
for m in models:
    print(m)
"
}

start_instance() {
    local model_name="$1"
    local config_json
    config_json="$(get_instance_config "$model_name")"

    if [ "$config_json" = "null" ]; then
        echo "ERROR: Model '$model_name' not found in config" >&2
        return 1
    fi

    # Parse config
    local port path max_model_len gpu_mem
    port="$(echo "$config_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['port'])")"
    path="$(echo "$config_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['path'])")"
    max_model_len="$(echo "$config_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('max_model_len', 4096))")"
    gpu_mem="$(echo "$config_json" | python3 -c "import json,sys; print(json.load(sys.stdin).get('gpu_memory_utilization', 0.3))")"

    # Check if port is already in use
    if lsof -i ":$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $port already in use (model '$model_name' may already be running)"
        return 0
    fi

    # Check if model files exist
    local full_path="${ROOT_DIR}/${path}"
    if [ ! -d "$full_path" ]; then
        die "Model directory not found: $full_path"
    fi

    # Build vLLM command
    local cmd="python3 -m vllm.entrypoints.openai.api_server"
    cmd+=" --model ${full_path}"
    cmd+=" --served-model-name ${model_name}"
    cmd+=" --port ${port}"
    cmd+=" --max-model-len ${max_model_len}"
    cmd+=" --gpu-memory-utilization ${gpu_mem}"
    cmd+=" --quantization gptq"
    cmd+=" --dtype float16"
    cmd+=" --trust-remote-code"

    # Optional: LoRA adapter
    local lora_adapter
    lora_adapter="$(echo "$config_json" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
print(cfg.get('lora_adapter', ''))
")"
    if [ -n "$lora_adapter" ]; then
        local full_adapter="${ROOT_DIR}/${lora_adapter}"
        if [ -d "$full_adapter" ]; then
            cmd+=" --enable-lora"
            cmd+=" --lora-modules adapter=${full_adapter}"
        else
            echo "WARNING: LoRA adapter not found at $full_adapter, serving without it"
        fi
    fi

    # Optional: expert parallel for MoE
    local expert_parallel
    expert_parallel="$(echo "$config_json" | python3 -c "
import json, sys
cfg = json.load(sys.stdin)
print(cfg.get('enable_expert_parallel', False))
")"
    if [ "$expert_parallel" = "True" ]; then
        cmd+=" --enable-expert-parallel"
    fi

    echo "Starting vLLM instance: $model_name on port $port"
    echo "  Command: $cmd"

    # Launch in background
    nohup $cmd > "${ROOT_DIR}/logs/serve_${model_name}.log" 2>&1 &
    local pid=$!
    echo "$pid" > "${PID_DIR}/${model_name}.pid"

    echo "  PID: $pid"
    echo "  Log: logs/serve_${model_name}.log"

    # Wait briefly and check if it started
    sleep 3
    if kill -0 "$pid" 2>/dev/null; then
        echo "  Status: running"
    else
        echo "  Status: FAILED (check log)"
        return 1
    fi
}

stop_instance() {
    local model_name="$1"
    local pid_file="${PID_DIR}/${model_name}.pid"

    if [ ! -f "$pid_file" ]; then
        echo "No PID file for '$model_name'"
        return 0
    fi

    local pid
    pid="$(cat "$pid_file")"

    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $model_name (PID $pid)..."
        kill "$pid"
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "Stopped."
    else
        echo "$model_name not running (stale PID file)"
    fi

    rm -f "$pid_file"
}

stop_all() {
    echo "Stopping all vLLM instances..."
    for pid_file in "${PID_DIR}"/*.pid; do
        [ -f "$pid_file" ] || continue
        local name
        name="$(basename "$pid_file" .pid)"
        stop_instance "$name"
    done
    echo "All instances stopped."
}

list_instances() {
    echo "Running vLLM instances:"
    local found=false
    for pid_file in "${PID_DIR}"/*.pid; do
        [ -f "$pid_file" ] || continue
        local name pid status
        name="$(basename "$pid_file" .pid)"
        pid="$(cat "$pid_file")"
        if kill -0 "$pid" 2>/dev/null; then
            status="running"
        else
            status="dead (stale PID)"
        fi
        echo "  $name: PID=$pid, status=$status"
        found=true
    done
    if [ "$found" = false ]; then
        echo "  (none)"
    fi
}

# --- Main ---

check_deps

case "${1:-}" in
    --session)
        [ -n "${2:-}" ] || die "Usage: $0 --session <session_name>"
        models="$(get_session_models "$2")"
        if [ -z "$models" ]; then
            die "Session '$2' not found or empty"
        fi
        echo "Starting session '$2'..."
        while IFS= read -r model; do
            start_instance "$model"
        done <<< "$models"
        echo "Session '$2' started."
        ;;
    --model)
        [ -n "${2:-}" ] || die "Usage: $0 --model <model_name>"
        start_instance "$2"
        ;;
    --stop)
        [ -n "${2:-}" ] || die "Usage: $0 --stop <model_name>"
        stop_instance "$2"
        ;;
    --stop-all)
        stop_all
        ;;
    --list)
        list_instances
        ;;
    *)
        echo "Usage: $0 {--session <name> | --model <name> | --stop <name> | --stop-all | --list}"
        exit 1
        ;;
esac
