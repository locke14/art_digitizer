#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/deploy_cloud_run.sh <PROJECT_ID> <REGION> [SERVICE_NAME]
# Example: ./scripts/deploy_cloud_run.sh my-project us-central1 art-digitizer

if [[ ${1-} == "" || ${2-} == "" ]]; then
  echo "Usage: $0 <PROJECT_ID> <REGION> [SERVICE_NAME]" >&2
  exit 1
fi

PROJECT_ID="$1"
REGION="$2"
SERVICE_NAME="${3:-art-digitizer}"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "==> Setting gcloud project to ${PROJECT_ID}"
gcloud config set project "${PROJECT_ID}" 1>/dev/null

echo "==> Enabling required APIs (Run, Build)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com 1>/dev/null

echo "==> Building image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}"

echo "==> Deploying to Cloud Run: ${SERVICE_NAME} in ${REGION}"
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --allow-unauthenticated \
  --region "${REGION}" \
  --memory 1Gi

URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')
echo "==> Deployed: ${URL}"

