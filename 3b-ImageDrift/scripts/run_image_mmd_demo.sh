#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="image-mmd-model"
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCE_FILE="${DEMO_DIR}/resources/inferenceservice.yaml"
DATA_ROOT="${IMAGE_MMD_DATA_ROOT:-${DEMO_DIR}/data}"
GENERATED_DIR="${DATA_ROOT}/generated"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/image-mmd-run.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_command oc
require_command curl
require_command jq

: "${PREDICTOR_IMAGE:?Set PREDICTOR_IMAGE to the pushed predictor image reference.}"
if [[ ! "$PREDICTOR_IMAGE" =~ ^[[:alnum:]][[:alnum:]./_:@+-]*$ ]]; then
  fail "PREDICTOR_IMAGE contains unsupported characters: ${PREDICTOR_IMAGE}"
fi

CURL_TLS_ARGS=()
if [[ -n "${TRUSTYAI_CA_BUNDLE:-}" ]]; then
  test -r "$TRUSTYAI_CA_BUNDLE" || fail \
    "TRUSTYAI_CA_BUNDLE is not readable: ${TRUSTYAI_CA_BUNDLE}"
  CURL_TLS_ARGS=(--cacert "$TRUSTYAI_CA_BUNDLE")
fi

: "${TRUSTYAI_NAMESPACE:?Set TRUSTYAI_NAMESPACE to the dedicated demo project.}"

test -f "${GENERATED_DIR}/reference_upload.json" || fail \
  "generated payloads are missing; run prepare_data.py download, build, and verify first"

CURRENT_PROJECT="$(oc project -q)"
printf 'Using OpenShift project: %s\n' "$CURRENT_PROJECT"
if [[ "$CURRENT_PROJECT" != "$TRUSTYAI_NAMESPACE" ]]; then
  fail "current OpenShift project ${CURRENT_PROJECT} does not match TRUSTYAI_NAMESPACE=${TRUSTYAI_NAMESPACE}"
fi

if [[ -z "${TOKEN:-}" ]]; then
  TOKEN="$(oc create token user-one -n "$TRUSTYAI_NAMESPACE")" || fail \
    "unable to create a token for user-one in ${CURRENT_PROJECT}; apply the TrustyAI service resources first"
fi
if [[ "$TOKEN" == *$'\n'* || "$TOKEN" == *$'\r'* ]]; then
  fail "TOKEN contains a newline and cannot be used safely"
fi

escaped_token="${TOKEN//\\/\\\\}"
escaped_token="${escaped_token//\"/\\\"}"
CURL_AUTH_CONFIG="$TMP_DIR/curl-auth.conf"
umask 077
printf 'header = "Authorization: Bearer %s"\n' "$escaped_token" > "$CURL_AUTH_CONFIG"
CURL_ARGS=(--config "$CURL_AUTH_CONFIG" -sS --max-redirs 0)
if ((${#CURL_TLS_ARGS[@]} > 0)); then
  CURL_ARGS+=("${CURL_TLS_ARGS[@]}")
fi

TRUSTY_HOST="$(oc get route trustyai-service -n "$TRUSTYAI_NAMESPACE" -o jsonpath='{.spec.host}')" || fail \
  "TrustyAI route trustyai-service was not found in ${CURRENT_PROJECT}"
EXPECTED_TRUSTY_ROUTE="https://${TRUSTY_HOST}"
if [[ -z "${TRUSTY_ROUTE:-}" ]]; then
  TRUSTY_ROUTE="$EXPECTED_TRUSTY_ROUTE"
elif [[ "$TRUSTY_ROUTE" != "$EXPECTED_TRUSTY_ROUTE" && "${TRUSTYAI_ALLOW_CUSTOM_ROUTE:-}" != "1" ]]; then
  fail "TRUSTY_ROUTE must match the discovered route; set TRUSTYAI_ALLOW_CUSTOM_ROUTE=1 only for an intentional custom route"
fi
if [[ ! "$TRUSTY_ROUTE" =~ ^https://[^/]+$ ]]; then
  fail "TRUSTY_ROUTE must be an HTTPS origin without a path or query string"
fi

http_post_file() {
  local url="$1"
  local input_file="$2"
  local output_file="$3"
  curl "${CURL_ARGS[@]}" \
    -o "$output_file" \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data-binary "@${input_file}" \
    "$url"
}

http_post_json() {
  local url="$1"
  local json_body="$2"
  local output_file="$3"
  curl "${CURL_ARGS[@]}" \
    -o "$output_file" \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data "$json_body" \
    "$url"
}

get_counter() {
  local info_file="$TMP_DIR/info.json"
  local status
  status="$(curl "${CURL_ARGS[@]}" -o "$info_file" -w '%{http_code}' \
    "${TRUSTY_ROUTE}/info")"
  [[ "$status" =~ ^2 ]] || return 1
  jq -er --arg model "$MODEL_NAME" \
    '(.[$model].data.observations // 0) | if type == "number" then . else tonumber end' \
    "$info_file"
}

printf 'Checking MMD availability at %s/metrics/drift/mmd/definition\n' "$TRUSTY_ROUTE"
MMD_DEFINITION_FILE="$TMP_DIR/mmd-definition.json"
MMD_STATUS="$(curl "${CURL_ARGS[@]}" -o "$MMD_DEFINITION_FILE" -w '%{http_code}' \
  "${TRUSTY_ROUTE}/metrics/drift/mmd/definition")"
if [[ "$MMD_STATUS" == "404" ]]; then
  cat "$MMD_DEFINITION_FILE" >&2
  fail "MMD endpoint is not enabled (HTTP 404)"
elif [[ "$MMD_STATUS" == "503" ]]; then
  cat "$MMD_DEFINITION_FILE" >&2
  fail "MMD definition endpoint is unavailable (HTTP 503); verify TrustyAI readiness and the optional MMD dependency"
elif [[ ! "$MMD_STATUS" =~ ^2 ]]; then
  cat "$MMD_DEFINITION_FILE" >&2
  fail "MMD definition check failed with HTTP ${MMD_STATUS}"
fi
jq . "$MMD_DEFINITION_FILE"

RENDERED_RESOURCE="$TMP_DIR/inferenceservice.yaml"
sed "s|IMAGE_PLACEHOLDER|${PREDICTOR_IMAGE}|g" "$RESOURCE_FILE" > "$RENDERED_RESOURCE"
oc apply --dry-run=server -n "$TRUSTYAI_NAMESPACE" -f "$RENDERED_RESOURCE" >/dev/null
oc apply -n "$TRUSTYAI_NAMESPACE" -f "$RENDERED_RESOURCE"
oc wait -n "$TRUSTYAI_NAMESPACE" --for=condition=Ready inferenceservice/${MODEL_NAME} --timeout=600s

MODEL_HOST="$(oc get route "$MODEL_NAME" -n "$TRUSTYAI_NAMESPACE" -o jsonpath='{.spec.host}')" || fail \
  "KServe route ${MODEL_NAME} was not found"
MODEL_ROUTE="https://${MODEL_HOST}/v2/models/${MODEL_NAME}/infer"
printf 'KServe model route: %s\n' "$MODEL_ROUTE"

REFERENCE_FILE="${GENERATED_DIR}/reference_upload.json"
jq -e '.data_tag == "IMAGE_BASELINE"' "$REFERENCE_FILE" >/dev/null
REFERENCE_RESPONSE="$TMP_DIR/reference-response.json"
REFERENCE_STATUS="$(http_post_file "${TRUSTY_ROUTE}/data/upload" "$REFERENCE_FILE" "$REFERENCE_RESPONSE")"
if [[ ! "$REFERENCE_STATUS" =~ ^2 ]]; then
  cat "$REFERENCE_RESPONSE" >&2
  fail "reference upload failed with HTTP ${REFERENCE_STATUS}"
fi
jq -e '.status == "success" and (.message | contains("30 datapoints"))' \
  "$REFERENCE_RESPONSE" >/dev/null || {
  cat "$REFERENCE_RESPONSE" >&2
  fail "reference upload did not report 30 successful datapoints"
}
printf 'Reference upload response:\n'
jq . "$REFERENCE_RESPONSE"

FIT_COLUMNS="$(jq -cn '[range(0;54) | "embedding-\(.)"]')"

for case in identical near mild strong; do
  CURRENT_FILE="${GENERATED_DIR}/current_${case}.json"
  test -f "$CURRENT_FILE" || fail "missing current payload: $CURRENT_FILE"
  BEFORE="$(get_counter)" || fail "unable to read TrustyAI observation count before ${case}"
  printf '\nSending %s current batch; observations before send: %s\n' "$case" "$BEFORE"

  KSERVE_RESPONSE="$TMP_DIR/kserve-${case}.json"
  KSERVE_STATUS="$(curl "${CURL_ARGS[@]}" \
    -o "$KSERVE_RESPONSE" \
    -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data-binary "@${CURRENT_FILE}" \
    "$MODEL_ROUTE")"
  if [[ ! "$KSERVE_STATUS" =~ ^2 ]]; then
    cat "$KSERVE_RESPONSE" >&2
    fail "KServe request for ${case} failed with HTTP ${KSERVE_STATUS}"
  fi

  jq -e --arg model "$MODEL_NAME" '
    .model_name == $model
    and (.outputs | length) == 1
    and .outputs[0].name == "embedding"
    and .outputs[0].datatype == "FP32"
    and .outputs[0].shape == [30, 54]
    and (.outputs[0].data | length) == 1620
    and ([.outputs[0].data[] | select((type != "number") or (isfinite | not))] | length) == 0
  ' "$KSERVE_RESPONSE" >/dev/null || {
    cat "$KSERVE_RESPONSE" >&2
    fail "KServe response for ${case} failed the [30,54] FP32 contract"
  }

  AFTER=""
  for attempt in $(seq 1 60); do
    if AFTER="$(get_counter 2>/dev/null)" && (( AFTER >= BEFORE + 30 )); then
      break
    fi
    sleep 2
  done
  [[ -n "$AFTER" ]] && (( AFTER >= BEFORE + 30 )) || {
    printf 'Last observed count: %s\n' "${AFTER:-unavailable}" >&2
    oc get inferenceservice "$MODEL_NAME" -n "$TRUSTYAI_NAMESPACE" -o yaml >&2 || true
    fail "TrustyAI observation count did not increase by 30 for ${case}"
  }
  printf 'Observations after send: %s\n' "$AFTER"

  MMD_REQUEST="$(jq -cn \
    --arg model "$MODEL_NAME" \
    --argjson fitColumns "$FIT_COLUMNS" \
    '{modelId: $model, referenceTag: "IMAGE_BASELINE", batchSize: 30, method: "ctt", alpha: 0.05, seed: 7, numPermutations: 199, fitColumns: $fitColumns}')"
  MMD_RESPONSE="$TMP_DIR/mmd-${case}.json"
  MMD_STATUS="$(http_post_json "${TRUSTY_ROUTE}/metrics/drift/mmd" "$MMD_REQUEST" "$MMD_RESPONSE")"
  if [[ ! "$MMD_STATUS" =~ ^2 ]]; then
    cat "$MMD_RESPONSE" >&2
    if [[ "$MMD_STATUS" == "503" ]]; then
      fail "MMD computation for ${case} returned HTTP 503; verify the optional MMD dependency in the TrustyAI image"
    fi
    fail "MMD computation for ${case} failed with HTTP ${MMD_STATUS}"
  fi
  jq -e --argjson fitColumns "$FIT_COLUMNS" '
    .status == "success"
    and .fit_columns == $fitColumns
    and (.drift_detected | type) == "boolean"
  ' "$MMD_RESPONSE" >/dev/null || {
    cat "$MMD_RESPONSE" >&2
    fail "MMD response for ${case} failed the result contract"
  }
  printf 'MMD response for %s:\n' "$case"
  jq '{value, p_value, threshold, drift_detected, fit_columns}' "$MMD_RESPONSE"

  if [[ "$case" == "identical" ]]; then
    jq -e '.drift_detected == false' "$MMD_RESPONSE" >/dev/null || \
      fail "identical case unexpectedly detected drift"
  elif [[ "$case" == "strong" ]]; then
    jq -e '.drift_detected == true' "$MMD_RESPONSE" >/dev/null || \
      fail "strong case did not detect drift"
  fi
done

printf '\nImage MMD demo completed successfully.\n'
