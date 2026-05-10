#!/usr/bin/env bash
# Build and push the Falcon fused-MoE calibration TPU image via Cloud Build.
#
# This follows Falcon's image-build pattern: local Docker is not required;
# gcloud uploads the source context and Cloud Build pushes to Artifact Registry.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PROJECT="${FALCON_PROJECT:-tpu-service-473302}"
REGION="${FALCON_BUILD_REGION:-us-central1}"
REGISTRY="${FALCON_REGISTRY:-us-central1-docker.pkg.dev}"
AR_REPO="${FALCON_AR_REPO:-primatrix}"
IMAGE_NAME="${FALCON_CALIBRATION_IMAGE_NAME:-sglang-jax-fused-moe-calibration}"
IMAGE="${REGISTRY}/${PROJECT}/${AR_REPO}/${IMAGE_NAME}"

GIT_SHA="$(git rev-parse HEAD)"
GIT_SHA_SHORT="$(git rev-parse --short=12 HEAD)"
TREE_STATE="clean"
if ! git diff --quiet || ! git diff --cached --quiet; then
  TREE_STATE="dirty"
fi

if [[ -z "${IMAGE_TAG:-}" ]]; then
  IMAGE_TAG="${GIT_SHA_SHORT}"
  if [[ "${TREE_STATE}" == "dirty" ]]; then
    IMAGE_TAG="${IMAGE_TAG}-dirty"
  fi
fi

echo "Submitting Cloud Build for ${IMAGE}:${IMAGE_TAG}"
echo "  project      = ${PROJECT}"
echo "  region       = ${REGION}"
echo "  gitCommit    = ${GIT_SHA}"
echo "  gitTreeState = ${TREE_STATE}"

gcloud builds submit \
  --project="${PROJECT}" \
  --region="${REGION}" \
  --config=hack/cloudbuild-falcon-calibration.yaml \
  --substitutions="_IMAGE=${IMAGE},_TAG=${IMAGE_TAG},_GIT_SHA=${GIT_SHA}" \
  .

echo
echo "Pushed: ${IMAGE}:${IMAGE_TAG}"
