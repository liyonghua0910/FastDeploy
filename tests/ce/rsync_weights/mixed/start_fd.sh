#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p "$(pwd)/output/fd"

FD_API_PORT=${FD_API_PORT:-"8180"}
MODEL_DIR=${MODEL_DIR:-"/ssd4/liyonghua/models/TP2"}
TP_SIZE=${TP_SIZE:-"2"}
BACKEND=${BACKEND:-"mooncake"}
BUCKET_SIZE_MB=${BUCKET_SIZE_MB:-"2048"}
REDIS_HOST=${REDIS_HOST:-"127.0.0.1"}
REDIS_PORT=${REDIS_PORT:-"6379"}

RSYNC_CONFIG=$(printf '{"index":0,"backend":"%s","bucket_size_mb":%s,"redis_host":"%s","redis_port":%s}' \
    "${BACKEND}" "${BUCKET_SIZE_MB}" "${REDIS_HOST}" "${REDIS_PORT}")

export CUDA_VISIBLE_DEVICES=2,3
export ENABLE_V1_KVCACHE_SCHEDULER=1
export FD_LOG_DIR="$(pwd)/output/fd/log"

nohup python -m fastdeploy.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port "${FD_API_PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len 32768 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.9 \
    --model "${MODEL_DIR}" \
    --dynamic-load-weight \
    --load-strategy rsync \
    --rsync-config "${RSYNC_CONFIG}" \
    >"$(pwd)/output/fd/stdout.log" 2>"$(pwd)/output/fd/stderr.log" &

echo $! >"$(pwd)/output/fd/fd_api_server.pid"

for _ in $(seq 1 300); do
    if (echo > /dev/tcp/127.0.0.1/"${FD_API_PORT}") >/dev/null 2>&1; then
        echo "API server is up on port ${FD_API_PORT}"
        exit 0
    fi
    sleep 1
done

echo "Timeout: API server did not start within 300 seconds (port ${FD_API_PORT})" >&2
exit 1
