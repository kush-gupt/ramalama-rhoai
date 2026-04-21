# RamaLama CUDA ServingRuntime for OpenShift AI

Serve GGUF models on NVIDIA GPUs in Red Hat OpenShift AI using
`quay.io/ramalama/cuda` and `llama-server` (llama.cpp).

This guide walks through every step from a fresh RHOAI cluster to a working
inference endpoint visible in the Gen AI Studio playground.

## Architecture

Architecture diagram

## Quickstart

The `quickstart.sh` script automates the full deployment workflow:

```bash
# OCI modelcar image (recommended)
./quickstart.sh -n <namespace> --oci

# PVC with direct download
./quickstart.sh -n <namespace> --pvc
```

The script validates prerequisites, applies the ServingRuntime via Kustomize,
deploys the InferenceService, waits for readiness, and runs a smoke test.

## Prerequisites

- OpenShift 4.x cluster with **RHOAI** (Red Hat OpenShift AI) installed and
KServe managed
- At least one worker node with an NVIDIA GPU and the GPU operator configured
- `oc` CLI logged into the cluster (cluster-admin for one-time setup steps)
- `ramalama` CLI installed locally (for building OCI model images)
- `podman` available locally (used by `ramalama push` to build and push images)

## 1. Cluster Setup

### Enable Headed Services

RHOAI's RawDeployment mode defaults to headless Kubernetes Services. Headed
(ClusterIP) services are required so the default KServe URL works on port 80
without callers needing to know the container port (8080 by default):

```bash
oc patch datasciencecluster default-dsc --type merge \
  -p '{"spec":{"components":{"kserve":{"rawDeploymentServiceConfig":"Headed"}}}}'
```

This is cluster-wide. Existing InferenceServices must be recreated to pick up
the change. To revert, replace `Headed` with `Headless`.

### Label the Target Namespace

The namespace must be recognized as a Data Science Project for the RHOAI
dashboard to manage resources in it:

```bash
oc label namespace <your-namespace> opendatahub.io/dashboard=true
```

## 2. Install the ServingRuntime

### Option A: Global install via OpenShift Template (recommended)

Install the runtime as an OpenShift Template in `redhat-ods-applications` so it
appears alongside the built-in runtimes (vLLM, OVMS) in every project:

```bash
oc apply -f serving-runtime-template.yaml -n redhat-ods-applications
```

The template supports parameters for customization:

```bash
oc process -f serving-runtime-template.yaml \
  -p IMAGE=quay.io/ramalama/cuda:latest \
  -p LLAMA_CTX_SIZE=8192 \
  | oc apply -n redhat-ods-applications -f -
```


| Parameter                | Default                        | Description                    |
| ------------------------ | ------------------------------ | ------------------------------ |
| `IMAGE`                  | `quay.io/ramalama/cuda:latest` | Container image                |
| `LLAMA_NGL`              | `999`                          | GPU layers to offload          |
| `LLAMA_CTX_SIZE`         | `4096`                         | Context window size in tokens  |
| `LLAMA_CACHE_REUSE`      | `256`                          | KV cache reuse slots           |
| `LLAMA_THREADS`          | `4`                            | CPU threads                    |
| `LLAMA_REASONING_BUDGET` | `0`                            | Reasoning token budget (0=off) |


The template includes metadata the dashboard uses for display and linking:


| Metadata                                                   | Purpose                                                          |
| ---------------------------------------------------------- | ---------------------------------------------------------------- |
| `opendatahub.io/template-name` (label)                     | Links a namespace-level runtime back to this template            |
| `opendatahub.io/template-display-name` (annotation)        | Name shown in the dashboard instead of "Unknown Serving Runtime" |
| `opendatahub.io/apiProtocol` (annotation)                  | Tells the dashboard the API type (REST)                          |
| `opendatahub.io/modelServingSupport` (template annotation) | Marks it as single-model serving                                 |


### Option B: Single namespace via Kustomize

```bash
oc apply -k base/ -n <namespace>
```

Or use the overlay:

```bash
oc apply -k overlays/namespace/ -n <namespace>
```

To remove:

```bash
oc delete template ramalama-cuda-runtime-template -n redhat-ods-applications
# or
oc delete -k base/ -n <namespace>
```

### Why `vLLM` appears in `supportedModelFormats`

The RHOAI dashboard hardcodes `{ name: 'vLLM' }` as the model format for all
generative model deployments
([source](https://github.com/opendatahub-io/odh-dashboard/blob/main/packages/model-serving/src/components/deploymentWizard/fields/ModelFormatField.tsx)).
Without `vLLM` in the ServingRuntime's `supportedModelFormats`, the dashboard
filters out the runtime when deploying generative models, making it invisible.
The `vLLM` entry is required for dashboard compatibility even though the runtime
uses llama.cpp.

## 3. Provide a GGUF Model

The ServingRuntime expects a GGUF model file under `/mnt/models/`. There are two
supported approaches for delivering the model.

### Option A: PVC (direct download)

Best when you want a single specific quantization from a HuggingFace repo that
offers many variants.

```bash
oc apply -f model-pvc.yaml # 5 gig by default
oc apply -f download-model-job.yaml # hf/bartowski/google_gemma-4-E2B-it-GGUF/google_gemma-4-E2B-it-Q4_K_M.gguf by default
oc wait --for=condition=Complete job/download-gemma-gguf --timeout=600s
```

To use a different model, edit `download-model-job.yaml` and change the curl URL
and output filename.

### Option B: OCI Modelcar Image (recommended)

Package the model as an OCI container image using `ramalama push --type car`.
KServe mounts the image as a sidecar (the "modelcar" pattern) -- no PVC, no
download job, and the model travels with its version tag.

**Build and push the image**:

```bash
# Authenticate to your registry
podman login quay.io

# Build and push a modelcar image
ramalama push --type car \
  hf://bartowski/google_gemma-4-E2B-it-GGUF/google_gemma-4-E2B-it-Q4_K_M.gguf \
  oci://quay.io/kugupta/gemma-4-e2b-q4km:car
```

The resulting image is `FROM ubi10-micro` with the model at `/models/model.file`
(a symlink into the model directory). It carries the label
`org.containers.type=ai.image.model.car`.

Ensure the registry repo is **public** or configure a pull secret on the cluster.

#### Why `--type car` and not `--type raw`?

KServe's modelcar pattern runs the OCI image as a sidecar container that
executes `sh -c "ln -sf /proc/$$/root/models /mnt/models && sleep infinity"`.
Raw images (`--type raw`) are built `FROM scratch` and contain no shell, so the
sidecar fails with `executable file 'sh' not found`. Car images include a
minimal base OS (UBI micro) with `sh`.

## 4. Deploy an InferenceService

### Using the OCI modelcar image

```bash
oc apply -f inference-service-oci-car.yaml -n <namespace>
```

The key field for OCI is `storageUri: oci://quay.io/kugupta/gemma-4-e2b-q4km:car`.

### Using a PVC

```bash
oc apply -f inference-service-example.yaml -n <namespace>
```

The key field is `storageUri: pvc://ramalama-models/`.

### Wait for readiness

```bash
oc wait --for=condition=Ready inferenceservice/<name> --timeout=600s
```

The first startup takes some minutes while the OCI image is pulled (3+ GB).

Subsequent restarts are faster since the image is cached on the node/cluster.

## 5. Smoke Test Inference

### Via the external HTTPS route

```bash
URL=$(oc get inferenceservice <name> -o jsonpath='{.status.url}')

curl -sk "${URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<name>",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "max_tokens": 128
  }'
```

### Via the in-cluster service (headed mode)

```bash
curl -s "http://<name>-predictor.<namespace>.svc.cluster.local/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "<name>", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}'
```

### Via localhost (exec into the pod)

```bash
oc exec deploy/<name>-predictor -c kserve-container -- \
  curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## 6. Gen AI Studio Playground

The InferenceService examples include labels that register the model in the Gen AI Studio playground:


| Label / Annotation                                   | Purpose                                    |
| ---------------------------------------------------- | ------------------------------------------ |
| `opendatahub.io/genai-asset: "true"` (label)         | Registers on the AI asset endpoints page   |
| `opendatahub.io/dashboard: "true"` (label)           | Makes it visible in the RHOAI dashboard    |
| `networking.kserve.io/visibility: exposed` (label)   | Creates an external HTTPS route            |
| `opendatahub.io/model-type: generative` (annotation) | Tags the model type for the dashboard      |
| `openshift.io/display-name` (annotation)             | Human-readable name shown in the dashboard |


### Cluster prerequisites

Enable the LlamaStack operator in the DataScienceCluster CRD (under `spec.components`):

```yaml
llamastackoperator:
  managementState: Managed
```

Enable Gen AI Studio for all RHOAI Dashboard users (existing users will need to refresh their browser and/or sign in again):

```bash
oc patch odhdashboardconfig odh-dashboard-config -n redhat-ods-applications \
  --type merge -p '{"spec":{"dashboardConfig":{"genAiStudio":true}}}'
```

## Troubleshooting

### Cold-start timeout / "Error in input stream" in playground

`llama-server` processes the KV cache on the first request after startup, which
can take 30-40 seconds for larger models. The RHOAI playground proxy has a
~30-second timeout.

Mitigations:

- Remove `--no-warmup` from the llama-server args (already done in the provided
YAMLs) so it warms up at startup
- Send a manual warmup request after the pod starts:
  ```bash
  oc exec deploy/<name>-predictor -c kserve-container -- \
    curl -s http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"warmup","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
  ```

### OCI raw image fails with `CreateContainerError`

`--type raw` images are built `FROM scratch` and have no shell. KServe's
modelcar sidecar needs `sh` to create the procfs symlink. Use `--type car`
instead.

## Files


| File                             | Description                                                        |
| -------------------------------- | ------------------------------------------------------------------ |
| `base/serving-runtime.yaml`      | ServingRuntime CR (Kustomize base, single source of truth)         |
| `base/kustomization.yaml`        | Kustomize base configuration                                       |
| `overlays/namespace/`            | Kustomize overlay for namespace-scoped install                     |
| `serving-runtime-template.yaml`  | OpenShift Template for global availability via the RHOAI dashboard |
| `inference-service-example.yaml` | InferenceService using a PVC-backed GGUF model                     |
| `inference-service-oci-car.yaml` | InferenceService using an OCI modelcar image                       |
| `model-pvc.yaml`                 | 5Gi PVC for model storage                                          |
| `download-model-job.yaml`        | Job that downloads a specific GGUF file to the PVC                 |
| `quickstart.sh`                  | Automated deploy + smoke test script                               |


## Configuration

### Model discovery

The ServingRuntime uses a shell wrapper to locate the GGUF model:

1. Check for `/mnt/models/model.file` (ramalama OCI images always include this symlink)
2. Glob for `*.gguf` files directly in `/mnt/models/` or one level deep
3. Fall back to `/mnt/models`

### llama-server parameters

Tunable `llama-server` flags are exposed as environment variables with sensible defaults. Override them in the InferenceService `env` block or by editing the ServingRuntime directly:


| Env Variable             | llama-server Flag    | Default | Description                                       |
| ------------------------ | -------------------- | ------- | ------------------------------------------------- |
| `LLAMA_NGL`              | `-ngl`               | `999`   | GPU layers to offload (999 = all layers)          |
| `LLAMA_CTX_SIZE`         | `--ctx-size`         | `4096`  | Context window size in tokens                     |
| `LLAMA_CACHE_REUSE`      | `--cache-reuse`      | `256`   | KV cache reuse slots for repeated prompt prefixes |
| `LLAMA_THREADS`          | `--threads`          | `4`     | CPU threads for non-GPU work                      |
| `LLAMA_REASONING_BUDGET` | `--reasoning-budget` | `0`     | Thinking/reasoning token budget (0 = disabled)    |


The `--no-webui` flag is always set since the built-in web UI is not needed in a headless serving context.

### Security context

The ServingRuntime container runs with a restrictive security context:

```yaml
securityContext:
  allowPrivilegeEscalation: false
  runAsNonRoot: true
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop: ["ALL"]
```

This has been tested and confirmed working with NVIDIA GPU access on OpenShift.

## Storage URI Reference


| URI Format                 | Behavior                              | Notes                                              |
| -------------------------- | ------------------------------------- | -------------------------------------------------- |
| `pvc://pvc-name/`          | Mounts PVC at `/mnt/models/`          | Best for single-file GGUF downloads                |
| `oci://registry/image:tag` | Mounts OCI image via modelcar sidecar | Use `--type car` images (not raw)                  |
| `hf://owner/repo`          | Downloads entire HF repo              | Impractical for GGUF repos with many quantizations |
| `s3://bucket/key`          | Downloads from S3                     | Requires S3 credentials secret                     |


