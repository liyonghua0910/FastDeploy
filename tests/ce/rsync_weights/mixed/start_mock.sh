#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p "$(pwd)/output/mock/publishers"

MODEL_DIR=${MODEL_DIR:-"/ssd4/liyonghua/models/TP2"}
MODEL_VERSION=${MODEL_VERSION:-"0"}
TP_SIZE=${TP_SIZE:-"2"}
BACKEND=${BACKEND:-"mooncake"}
BUCKET_SIZE_MB=${BUCKET_SIZE_MB:-"2048"}
REDIS_HOST=${REDIS_HOST:-"127.0.0.1"}
REDIS_PORT=${REDIS_PORT:-"6379"}
VERIFY_CHECKSUM="${VERIFY_CHECKSUM}"

VERIFY_CHECKSUM_ARGSTR=""
if [[ "${VERIFY_CHECKSUM}" == "true" ]]; then
    VERIFY_CHECKSUM_ARGSTR="--verify-checksum"
fi

if ! redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1; then
    if command -v redis-server >/dev/null 2>&1; then
        nohup redis-server --port "${REDIS_PORT}" --bind 0.0.0.0 >"$(pwd)/output/mock/redis.log" 2>&1 &
    else
        bash /ssd4/liyonghua/async_weight_update/download_and_start_redis.sh /ssd4/liyonghua/async_weight_update >/dev/null
    fi
    for _ in $(seq 1 20); do
        if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping >/dev/null 2>&1 || {
        echo "Redis failed to start at ${REDIS_HOST}:${REDIS_PORT}" >&2
        exit 1
    }
fi

for ((rank=0; rank<TP_SIZE; rank++)); do
    printf -v shard_path "${MODEL_DIR}/model_state.tp%02d.pdparams" "${rank}"
    [[ -f "${shard_path}" ]] || {
        echo "checkpoint shard not found: ${shard_path}" >&2
        exit 1
    }

    ready_file="$(pwd)/output/mock/publishers/rank${rank}.${MODEL_VERSION}.ready"
    rm -f "${ready_file}"

    nohup python "$(pwd)/publish_weights.py" \
        --state-path "${shard_path}" \
        --version "${MODEL_VERSION}" \
        --backend "${BACKEND}" \
        --bucket-size-mb "${BUCKET_SIZE_MB}" \
        --redis-host "${REDIS_HOST}" \
        --redis-port "${REDIS_PORT}" \
        --global-rank "${rank}" \
        --group-size "${TP_SIZE}" \
        --ready-file "${ready_file}" \
        ${VERIFY_CHECKSUM_ARGSTR} \
        --hold-seconds 600 \
        >"$(pwd)/output/mock/publisher.rank${rank}.${MODEL_VERSION}.log" 2>&1 &

    echo $! >"$(pwd)/output/mock/publishers/rank${rank}.pid"
    echo "publisher rank=${rank} pid=$(cat "$(pwd)/output/mock/publishers/rank${rank}.pid") shard=${shard_path}"
done

for ((rank=0; rank<TP_SIZE; rank++)); do
    ready_file="$(pwd)/output/mock/publishers/rank${rank}.${MODEL_VERSION}.ready"
    for _ in $(seq 1 300); do
        [[ -f "${ready_file}" ]] && break
        sleep 1
    done
    [[ -f "${ready_file}" ]] || {
        echo "publisher rank=${rank} not ready for version=${MODEL_VERSION}" >&2
        exit 1
    }
done

VERSION_FILE="${MODEL_DIR}/version.yaml" VERSION_VALUE="${MODEL_VERSION}" python - <<'PY'
import os
import yaml
with open(os.environ["VERSION_FILE"], "w", encoding="utf-8") as f:
    yaml.safe_dump({"step": os.environ["VERSION_VALUE"]}, f, sort_keys=False)
PY

echo "all mock publishers ready for version=${MODEL_VERSION}"
