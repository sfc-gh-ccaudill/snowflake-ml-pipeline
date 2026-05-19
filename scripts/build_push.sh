#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <image_repo_url> <image_name> <path_to_dockerfile_dir>"
  echo ""
  echo "  image_repo_url        Snowflake image registry URL (from SHOW IMAGE REPOSITORIES)"
  echo "  image_name            Name and optional tag (e.g. patient_risk_xgb:latest)"
  echo "  path_to_dockerfile_dir  Directory containing the Dockerfile"
  echo ""
  echo "Example:"
  echo "  $0 sfsenorthamerica-demo.registry.snowflakecomputing.com/db/schema/repo patient_risk_xgb:latest ./job_payload"
  exit 1
}

if [ $# -lt 3 ]; then
  usage
fi

REPO_URL="$1"
IMAGE_NAME="$2"
DOCKERFILE_DIR="$3"

REGISTRY_HOST=$(echo "$REPO_URL" | cut -d'/' -f1)
FULL_TAG="${REPO_URL}/${IMAGE_NAME}"

echo "=== Snowflake Image Build & Push ==="
echo "  Registry:   ${REGISTRY_HOST}"
echo "  Image:      ${FULL_TAG}"
echo "  Dockerfile: ${DOCKERFILE_DIR}"
echo ""

echo ">> Logging in to ${REGISTRY_HOST} ..."
if [ -n "${SNOWFLAKE_USER:-}" ] && [ -n "${SNOWFLAKE_TOKEN:-}" ]; then
  echo "${SNOWFLAKE_TOKEN}" | docker login "${REGISTRY_HOST}" --username "${SNOWFLAKE_USER}" --password-stdin
elif command -v snow &> /dev/null; then
  snow spcs image-registry login
else
  read -rp "Snowflake username: " SNOWFLAKE_USER
  read -rsp "Snowflake password/token: " SNOWFLAKE_TOKEN
  echo ""
  echo "${SNOWFLAKE_TOKEN}" | docker login "${REGISTRY_HOST}" --username "${SNOWFLAKE_USER}" --password-stdin
fi
echo ""

echo ">> Building image ..."
docker build --platform linux/amd64 \
  --build-arg SNOWFLAKE_REGISTRY="${REGISTRY_HOST}" \
  -t "${IMAGE_NAME}" "${DOCKERFILE_DIR}"
echo ""

echo ">> Tagging as ${FULL_TAG} ..."
docker tag "${IMAGE_NAME}" "${FULL_TAG}"
echo ""

echo ">> Pushing to Snowflake registry ..."
docker push "${FULL_TAG}"
echo ""

echo "=== Done ==="
echo "Image available at: ${FULL_TAG}"
