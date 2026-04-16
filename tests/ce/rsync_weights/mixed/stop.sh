#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f "$(pwd)/output/fd/fd_api_server.pid" ]]; then
    pid="$(cat "$(pwd)/output/fd/fd_api_server.pid")"
    if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" || true
        echo "stopped fd pid=${pid}"
    fi
    rm -f "$(pwd)/output/fd/fd_api_server.pid"
fi

if compgen -G "$(pwd)/output/mock/publishers/*.pid" >/dev/null; then
    for pid_file in "$(pwd)/output/mock/publishers"/*.pid; do
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" || true
            echo "stopped mock pid=${pid}"
        fi
        rm -f "${pid_file}"
    done
fi
