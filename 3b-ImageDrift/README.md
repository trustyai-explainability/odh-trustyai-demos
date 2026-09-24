# Image MMD KServe demo

This demo shows image drift detection with TrustyAI Maximum Mean Discrepancy (MMD):

1. encode image bytes with a small deterministic KServe predictor;
2. upload baseline embeddings to TrustyAI;
3. send identical, near, mildly changed, and strongly changed image batches;
4. query the MMD metric after each observation.

The predictor is intentionally lightweight and CPU-only. Its embedding is a deterministic 54-dimensional feature vector: pooled RGB features plus per-channel mean and standard deviation. It is a demo signal, not a production vision model.
KServe is the only supported deployment path for the remote workflow.
The demo predictor accepts batches of at most 30 images, with a 32 MiB request
limit and bounded image dimensions for predictable CPU and memory use.

## Dataset

The default manifest uses the MIT Multi-Illumination Dataset test archive. The demo selects 30 indoor scenes and four illumination conditions per scene:

| Case | Reference/current illumination | MMD statistic | p-value | Drift |
| --- | --- | ---: | ---: | --- |
| identical | 0 / 0 | approximately 0 | 1.000 | false |
| near | 0 / 1 | 0.0324905 | 1.000 | false |
| mild | 0 / 14 | 0.0897650 | 0.035 | true |
| strong | 0 / 24 | 0.2354222 | 0.005 | true |

These measurements use method ctt, alpha 0.05, seed 7, and 199 permutations
over all 54 embedding columns.

The selected archive is 214,841,949 bytes and is verified by SHA-256 before extraction:

    7a142f0f4dcf8c6b038f91a32eee5962a12aa68e5c4ee43adf0d3059ea0f0ce0

The source page documents the dataset and its CC-BY licensing: [MIT Multi-Illumination Dataset](https://projects.csail.mit.edu/illumination/). The accompanying paper is [A Multi-Illumination Dataset of Indoor Object Appearance](https://arxiv.org/abs/1910.08131).

The default source is:

    https://data.csail.mit.edu/multilum/multi_illumination_test_mip2_jpg.zip

The dataset is not committed. Only the manifest and generated-data contract are tracked.

## Dataset switching

To switch datasets, edit data/dataset_manifest.json only when the replacement follows the same contract:

- one ZIP source with a pinned SHA-256;
- explicit reference and current image members for each sample;
- a case label and expected drift outcome;
- the same generated V2 payload shape.

Then run:

    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py download --manifest 3b-ImageDrift/data/dataset_manifest.json
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py build --manifest 3b-ImageDrift/data/dataset_manifest.json
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py verify --manifest 3b-ImageDrift/data/dataset_manifest.json

The predictor and OpenShift resources do not contain dataset-specific paths or labels.
For an alternate profile kept outside the checked-in demo data directory, set
IMAGE_MMD_MANIFEST and IMAGE_MMD_DATA_ROOT before running the notebook or
runner. Use the same values for the preparation commands, for example:

    export IMAGE_MMD_MANIFEST=/path/to/dataset_manifest.json
    export IMAGE_MMD_DATA_ROOT=/path/to/image-mmd-data
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py download --manifest "$IMAGE_MMD_MANIFEST" --data-root "$IMAGE_MMD_DATA_ROOT"
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py build --manifest "$IMAGE_MMD_MANIFEST" --data-root "$IMAGE_MMD_DATA_ROOT"
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py verify --manifest "$IMAGE_MMD_MANIFEST" --data-root "$IMAGE_MMD_DATA_ROOT"

The demo uses `3b-ImageDrift/pyproject.toml` and its checked-in `uv.lock` as
the dependency source. Base dependencies cover the predictor and data
preparation. The `local` extra adds Jupyter, plotting, pandas, and notebook
execution support. MMD itself runs in the TrustyAI service container.

## Prerequisites

- oc authenticated to the target OpenShift cluster.
- Podman authenticated to Quay.
- `skopeo` for resolving the registry's immutable digest after pushing.
- uv for the local notebook and data tooling.
- A namespace-admin capable identity for the new demo project.
- A healthy TrustyAI service and KServe installation in the cluster.

Cluster installation, TrustyAI, TLS/logger, and monitoring prerequisites are
documented in [1-Installation](../1-Installation/README.md).

Create or select the dedicated project:

    oc project image-mmd-kserve-demo || oc new-project image-mmd-kserve-demo --display-name="Image MMD KServe Demo"

The repository includes the namespace-level TrustyAI prerequisite used by the existing demos. Apply it once in the new project:

    oc apply -f 1-Installation/resources/trustyai.yaml

The default Quay repository is private. Choose an auth-file path for Podman,
log in with that file, then create a project-scoped pull secret and link it to
the predictor's default service account:

    export QUAY_AUTH_FILE=/actual/path/to/podman-auth.json
    podman login --authfile "$QUAY_AUTH_FILE" quay.io
    jq -c '{auths: {"quay.io": .auths["quay.io"]}}' "$QUAY_AUTH_FILE" | oc create secret generic quay-pull --from-file=.dockerconfigjson=/dev/stdin --type=kubernetes.io/dockerconfigjson --dry-run=client -o yaml | oc apply -f -
    oc secrets link default quay-pull --for=pull

Wait until the TrustyAI service is ready before running the remote workflow:

    oc get trustyaiservice trustyai-service -w

The cluster-level TrustyAI operator must be running for that resource to reconcile. The KServe logger configuration and service CA bundle are cluster-managed; the demo does not replace them.
The platform-level TrustyAI installation alone is not sufficient: this
namespace-level resource creates the service, its storage, the demo service
account, and the injected logger CA bundle.

## Prepare the data

From the repository root:

    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py download
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py build
    uv run --project 3b-ImageDrift python 3b-ImageDrift/scripts/prepare_data.py verify

The download command verifies the pinned archive and extracts only the manifest-selected members. Build creates ignored files under data/generated/.

## Run locally

Run the predictor contract self-test:

    uv run --project 3b-ImageDrift python 3b-ImageDrift/predictor/server.py --self-test

Run the local, image-rich notebook against a local TrustyAI service container.
It displays manifest-selected image pairs, embedding-distance plots, the MMD
statistics table, and the statistic versus threshold plot. From the demos
repository root:

    bash 3b-ImageDrift/scripts/run_local_trustyai.sh
    export TRUSTYAI_URL=http://127.0.0.1:18081
    export DEMOS_ROOT="$(pwd)"
    uv run --project 3b-ImageDrift --extra local jupyter lab

For a non-interactive verification run, execute the notebook with nbconvert:

    TRUSTYAI_URL=http://127.0.0.1:18081 uv run --project 3b-ImageDrift --extra local python -m nbconvert --to notebook --execute --output /tmp/image-mmd-local.executed.ipynb "$DEMOS_ROOT/3b-ImageDrift/notebooks/local_image_mmd_demo.ipynb"

The notebook never imports a local TrustyAI checkout. The runner builds a small
local compatibility image from
`quay.io/opendatahub/odh-trustyai-service-py:odh-stable`, repairs the current
image's dependency mismatch, and exposes its HTTP application on port 18081.
The service image computes MMD; the notebook's local embedding arrays are
diagnostic plots only. Before upload, the notebook removes the large base64
image request bodies while preserving the request shape, row count, and
response embeddings. This is required by the service image's default local
PVC row-size limit and does not change the MMD inputs.

The original `mmd_image_drift.ipynb` is also service-backed and uses the same
HTTP client; it no longer requires `TRUSTYAI_SERVICE_ROOT` or a sibling source
checkout. The local runner is intended for development only. Remove its
container with `podman rm -f trustyai-local` when finished.

For a remote HTTPS service, set `TRUSTYAI_URL`, `TRUSTYAI_ALLOWED_HOSTS`, and
`TRUSTYAI_TOKEN`. The client verifies HTTPS by default and accepts
`TRUSTYAI_CA_BUNDLE` for a cluster CA bundle. `TRUSTYAI_INSECURE=1` is limited
to loopback development endpoints.

## Build and push the predictor image

The default image reference for this branch is:

    quay.io/rh-ee-sudsinha/image-mmd-kserve-predictor:feat-image-mmd-kserve-demo

From the repository root:

    podman build --platform linux/amd64 -f 3b-ImageDrift/predictor/Containerfile -t quay.io/rh-ee-sudsinha/image-mmd-kserve-predictor:feat-image-mmd-kserve-demo 3b-ImageDrift
    podman push quay.io/rh-ee-sudsinha/image-mmd-kserve-predictor:feat-image-mmd-kserve-demo

Use the pushed immutable digest for deployment:

    export PREDICTOR_IMAGE="$(skopeo inspect docker://quay.io/rh-ee-sudsinha/image-mmd-kserve-predictor:feat-image-mmd-kserve-demo | jq -r '.Name + "@" + .Digest')"

The image contains only the predictor and shared encoder. Dataset acquisition remains a local or notebook concern.
The Containerfile resolves the base predictor dependencies from the checked-in
`pyproject.toml` and `uv.lock`; it does not install the optional `local` extra.

## Run on OpenShift

Prepare data and set the image:

    export PREDICTOR_IMAGE="$(skopeo inspect docker://quay.io/rh-ee-sudsinha/image-mmd-kserve-predictor:feat-image-mmd-kserve-demo | jq -r '.Name + "@" + .Digest')"
    export TRUSTYAI_NAMESPACE=image-mmd-kserve-demo
    # export TRUSTYAI_CA_BUNDLE=/actual/path/to/cluster-ca-bundle.pem

The runner obtains a short-lived token for the demo service account, discovers the TrustyAI route, applies the InferenceService manifest, and validates the V2 response shape before querying MMD:

    bash 3b-ImageDrift/scripts/run_image_mmd_demo.sh

The runner expects:

- TrustyAI service route /info, /data, and /metrics/drift/mmd;
- a user-one service account from trustyai.yaml;
- generated reference and current payloads under data/generated/;
- the image-mmd-model InferenceService to become Ready.

The runner verifies HTTPS using the system trust store by default and never
disables certificate verification. Set TRUSTYAI_CA_BUNDLE to a readable bundle
containing the required public and internal CAs when the route uses an
internal CA.

For a schema-only check before applying the resource:

    sed "s|IMAGE_PLACEHOLDER|$PREDICTOR_IMAGE|g" 3b-ImageDrift/resources/inferenceservice.yaml | oc apply --dry-run=server -f -

The runner preserves the deployed demo resources. To remove only the model:

    oc delete inferenceservice image-mmd-model --ignore-not-found

## Troubleshooting

- If data preparation reports a missing archive, run the download command. Build and verify intentionally do not download implicitly.
- If the checksum fails, remove the local archive and download it again. Do not bypass checksum verification.
- If oc get trustyaiservice remains Pending, inspect the TrustyAI operator deployment and pod events before changing the CR.
- If the MMD definition endpoint returns 404, the service version does not expose MMD. If the definition or computation returns 503, inspect TrustyAI readiness, storage, and the optional MMD dependency in the deployed image.
- If the InferenceService remains Pending, inspect KServe controller, storage-initializer, queue-proxy, and predictor pod events.
- If the predictor image cannot be pulled, verify Quay authentication and the image tag.
- If a route returns 401 or 403, obtain a fresh token with oc create token user-one -n image-mmd-kserve-demo and confirm the service account binding.
- The payload validator checks one image input, BYTES datatype, [30] input shape, and [30,54] FP32 output shape. A schema mismatch is a contract failure, not an MMD failure.
