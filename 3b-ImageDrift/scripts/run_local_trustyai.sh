#!/usr/bin/env bash

set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_IMAGE="${TRUSTYAI_IMAGE:-quay.io/opendatahub/odh-trustyai-service-py:odh-stable}"
LOCAL_IMAGE="${TRUSTYAI_LOCAL_IMAGE:-localhost/odh-trustyai-service-py:odh-stable-image-mmd-demo}"
CONTAINER_NAME="${TRUSTYAI_CONTAINER_NAME:-trustyai-local}"
TRUSTYAI_PORT="${TRUSTYAI_PORT:-18081}"
TRUSTYAI_URL="http://127.0.0.1:${TRUSTYAI_PORT}"

command -v podman >/dev/null || {
    printf 'podman is required.\n' >&2
    exit 1
}
command -v curl >/dev/null || {
    printf 'curl is required.\n' >&2
    exit 1
}

if ! podman image exists "$LOCAL_IMAGE"; then
    podman build \
        --platform linux/amd64 \
        --build-arg "TRUSTYAI_BASE_IMAGE=${BASE_IMAGE}" \
        -f "${DEMO_ROOT}/local/Containerfile" \
        -t "$LOCAL_IMAGE" \
        "$DEMO_ROOT"
fi

if podman container exists "$CONTAINER_NAME"; then
    if [[ "$(podman inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
        printf 'Container %s exists but is not running; remove it with: podman rm %s\n' \
            "$CONTAINER_NAME" "$CONTAINER_NAME" >&2
        exit 1
    fi
else
    podman run -d \
        --platform linux/amd64 \
        --name "$CONTAINER_NAME" \
        -p "${TRUSTYAI_PORT}:8081" \
        -e SERVICE_STORAGE_FORMAT=PVC \
        -e STORAGE_DATA_FOLDER=/tmp \
        -e STORAGE_DATA_FILENAME=trustyai-local.hdf5 \
        "$LOCAL_IMAGE" >/dev/null
fi

for _ in $(seq 1 60); do
    if curl --silent --fail --max-time 2 "${TRUSTYAI_URL}/info" >/dev/null; then
        printf 'TrustyAI is ready at %s\n' "$TRUSTYAI_URL"
        printf 'Export TRUSTYAI_URL=%s before opening the notebook.\n' "$TRUSTYAI_URL"
        exit 0
    fi
    if [[ "$(podman inspect --format '{{.State.Running}}' "$CONTAINER_NAME")" != "true" ]]; then
        podman logs "$CONTAINER_NAME" >&2
        exit 1
    fi
    sleep 1
done

podman logs "$CONTAINER_NAME" >&2
printf 'TrustyAI did not become ready at %s.\n' "$TRUSTYAI_URL" >&2
exit 1
