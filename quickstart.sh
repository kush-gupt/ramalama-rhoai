#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE=""
MODE="oci"
TIMEOUT=600
NAME=""

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Deploy a RamaLama CUDA ServingRuntime and InferenceService on OpenShift AI.

Options:
  -n, --namespace NS    Target namespace (default: current oc project)
  --oci                 Use OCI modelcar image for model delivery (default)
  --pvc                 Use PVC + download job for model delivery
  --timeout SECS        Wait timeout in seconds (default: 600)
  -h, --help            Show this help

Examples:
  $(basename "$0") -n gpu-test --oci
  $(basename "$0") -n gpu-test --pvc
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--namespace) NAMESPACE="$2"; shift 2 ;;
        --oci)          MODE="oci"; shift ;;
        --pvc)          MODE="pvc"; shift ;;
        --timeout)      TIMEOUT="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

info()  { printf "\033[1;34m==> %s\033[0m\n" "$*"; }
ok()    { printf "\033[1;32m  ✓ %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m  ! %s\033[0m\n" "$*"; }
fail()  { printf "\033[1;31m  ✗ %s\033[0m\n" "$*"; exit 1; }

# --- Prerequisites -----------------------------------------------------------

info "Checking prerequisites"

command -v oc &>/dev/null || fail "'oc' CLI not found in PATH"
ok "oc CLI found"

oc whoami &>/dev/null || fail "Not logged into an OpenShift cluster (run 'oc login' first)"
CLUSTER=$(oc whoami --show-server 2>/dev/null)
ok "Logged in as $(oc whoami) on $CLUSTER"

if [[ -z "$NAMESPACE" ]]; then
    NAMESPACE=$(oc project -q 2>/dev/null)
fi
oc get namespace "$NAMESPACE" &>/dev/null || fail "Namespace '$NAMESPACE' does not exist"
ok "Namespace: $NAMESPACE"

GPU_NODES=$(oc get nodes -l nvidia.com/gpu.present=true --no-headers 2>/dev/null | wc -l | tr -d ' ')
if [[ "$GPU_NODES" -eq 0 ]]; then
    warn "No nodes with label nvidia.com/gpu.present=true detected"
    warn "The deployment may fail without GPU nodes"
else
    ok "Found $GPU_NODES GPU node(s)"
fi

# --- Deploy ServingRuntime ----------------------------------------------------

info "Applying ServingRuntime via Kustomize"
oc apply -k "$SCRIPT_DIR/base/" -n "$NAMESPACE"
ok "ServingRuntime applied"

# --- Deploy model + InferenceService -----------------------------------------

if [[ "$MODE" == "pvc" ]]; then
    NAME="ramalama-example"
    info "Deploying with PVC model delivery"

    oc apply -f "$SCRIPT_DIR/model-pvc.yaml" -n "$NAMESPACE"
    ok "PVC created"

    oc delete job download-gemma-gguf -n "$NAMESPACE" --ignore-not-found &>/dev/null
    oc apply -f "$SCRIPT_DIR/download-model-job.yaml" -n "$NAMESPACE"
    ok "Download job created"

    info "Waiting for model download (timeout: ${TIMEOUT}s)"
    if oc wait --for=condition=Complete "job/download-gemma-gguf" -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
        ok "Model downloaded"
    else
        fail "Download job did not complete within ${TIMEOUT}s"
    fi

    oc apply -f "$SCRIPT_DIR/inference-service-example.yaml" -n "$NAMESPACE"
    ok "InferenceService (PVC) created"
else
    NAME="ramalama-oci-car"
    info "Deploying with OCI modelcar image"
    oc apply -f "$SCRIPT_DIR/inference-service-oci-car.yaml" -n "$NAMESPACE"
    ok "InferenceService (OCI) created"
fi

# --- Wait for readiness -------------------------------------------------------

info "Waiting for InferenceService '$NAME' to become Ready (timeout: ${TIMEOUT}s)"
if oc wait --for=condition=Ready "inferenceservice/$NAME" -n "$NAMESPACE" --timeout="${TIMEOUT}s"; then
    ok "InferenceService is Ready"
else
    warn "InferenceService did not reach Ready within ${TIMEOUT}s"
    echo ""
    echo "Debug with:"
    echo "  oc get inferenceservice $NAME -n $NAMESPACE -o yaml"
    echo "  oc get pods -l serving.kserve.io/inferenceservice=$NAME -n $NAMESPACE"
    echo "  oc logs deploy/${NAME}-predictor -c kserve-container -n $NAMESPACE"
    exit 1
fi

# --- Smoke test ---------------------------------------------------------------

info "Running smoke test (via oc exec into the pod)"
DEPLOY="${NAME}-predictor"

RESPONSE=$(oc exec "deploy/$DEPLOY" -c kserve-container -n "$NAMESPACE" -- \
    curl -s --max-time 120 http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one word\"}],\"max_tokens\":16}" 2>&1) || true

if echo "$RESPONSE" | grep -q '"choices"'; then
    ok "Inference working"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    warn "Smoke test did not return expected response (model may still be warming up):"
    echo "$RESPONSE"
fi

# --- Summary ------------------------------------------------------------------

URL=$(oc get inferenceservice "$NAME" -n "$NAMESPACE" -o jsonpath='{.status.url}' 2>/dev/null || echo "")

echo ""
info "Deployment complete"
echo "  Namespace:  $NAMESPACE"
echo "  Mode:       $MODE"
echo "  Name:       $NAME"
[[ -n "$URL" ]] && echo "  URL:        $URL"
echo ""
echo "  Test via oc exec:"
echo "    oc exec deploy/${DEPLOY} -c kserve-container -n $NAMESPACE -- \\"
echo "      curl -s http://localhost:8080/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
if [[ -n "$URL" ]]; then
    echo ""
    echo "  Test via external route:"
    echo "    curl -sk '${URL}/v1/chat/completions' \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64}'"
fi
