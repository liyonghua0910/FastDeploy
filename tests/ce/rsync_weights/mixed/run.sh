#!/bin/bash
set -e

FD_API_PORT=${FD_API_PORT:-"8180"}
MODEL_DIR=${MODEL_DIR:-"/ssd4/liyonghua/models/TP2"}
MODEL_VERSION=${MODEL_VERSION:-"0"}
VERIFY_CHECKSUM=${VERIFY_CHECKSUM:-"false"}

cd "$(dirname "${BASH_SOURCE[0]}")"
rm -rf "$(pwd)/output"

echo
echo "================================================================"
echo "Step 1/5  Start Mock Weights"
echo "================================================================"
echo
bash $(pwd)/start_mock.sh

echo
echo "================================================================"
echo "Step 2/5  Start FastDeploy"
echo "================================================================"
echo
bash $(pwd)/start_fd.sh

request() {
    local name="$1"
    shift
    local result http_code body
    result=$(curl -sS -w "%{http_code}" "$@")
    http_code="${result: -3}"
    body="${result%???}"
    [[ "${http_code}" == "200" ]] || {
        echo "${name} failed, http=${http_code}, body=${body}" >&2
        exit 1
    }
    echo "${name}: ${body}"
}

echo
echo "================================================================"
echo "Step 3/5  Request After Startup"
echo "================================================================"
echo
request startup_chat \
    -X POST "http://127.0.0.1:${FD_API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Hello!\"}
      ],
      \"stream\": false,
      \"max_tokens\": 200
    }"

echo
echo "================================================================"
echo "Step 4/5  Update Weights"
echo "================================================================"
echo
request pause -X POST "http://127.0.0.1:${FD_API_PORT}/v1/pause"
echo
request update_weights \
    -X POST "http://127.0.0.1:${FD_API_PORT}/v1/update_weights" \
    -H "Content-Type: application/json" \
    -d "{\"version\":\"${MODEL_VERSION}\",\"verify_checksum\":${VERIFY_CHECKSUM}}"
echo
request resume -X POST "http://127.0.0.1:${FD_API_PORT}/v1/resume"

echo
echo "================================================================"
echo "Step 5/5  Request After Resume"
echo "================================================================"
echo
request resume_chat \
    -X POST "http://127.0.0.1:${FD_API_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"messages\": [
        {\"role\": \"user\", \"content\": \"Hello!\"}
      ],
      \"stream\": false,
      \"max_tokens\": 200
    }"
