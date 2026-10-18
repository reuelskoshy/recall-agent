#!/usr/bin/env bash
# One-time Google Cloud setup for deploying Recall to Cloud Run from GitHub Actions.
#
# Run it in Cloud Shell (console.cloud.google.com -> the >_ icon), which has gcloud
# installed and logged in:
#
#   bash gcp-setup.sh <project-id> <github-owner/repo> [region]
#
# It creates (idempotently, so re-running is safe):
#   - an Artifact Registry repository for the images (keeps the 3 newest),
#   - service accounts: recall-run (the app; reads its secrets) and recall-deploy
#     (GitHub Actions; pushes images and deploys),
#   - Secret Manager secrets: DATABASE_URL, LLM_API_KEY, DEMO_PASSWORD (prompted,
#     never echoed) and READONLY_DB_PASSWORD (generated),
#   - a Workload Identity Federation pool that trusts only this GitHub repository.
# and prints the four GitHub repository variables the deploy job needs.
set -euo pipefail

PROJECT="${1:?usage: bash gcp-setup.sh <project-id> <github-owner/repo> [region]}"
REPO="${2:?usage: bash gcp-setup.sh <project-id> <github-owner/repo> [region]}"
REGION="${3:-us-east4}" # N. Virginia, next to Neon's AWS us-east-1

RUN_SA="recall-run@${PROJECT}.iam.gserviceaccount.com"
DEPLOY_SA="recall-deploy@${PROJECT}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="github"

gcloud config set project "$PROJECT" >/dev/null
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"

echo "==> Enabling APIs (takes a minute the first time)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
  sts.googleapis.com

echo "==> Artifact Registry repository 'recall' in $REGION"
if ! gcloud artifacts repositories describe recall --location "$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create recall --repository-format docker \
    --location "$REGION" --description "Recall container images"
fi
POLICY="$(mktemp)"
cat >"$POLICY" <<'JSON'
[
  {"name": "keep-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-rest", "action": {"type": "Delete"}, "condition": {"tagState": "any"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies recall --location "$REGION" \
  --policy "$POLICY" --no-dry-run >/dev/null
rm -f "$POLICY"

echo "==> Service accounts"
for name in recall-run recall-deploy; do
  if ! gcloud iam service-accounts describe "${name}@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" --display-name "Recall ${name#recall-}"
  fi
done
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member "serviceAccount:$DEPLOY_SA" \
    --role "$role" --condition None >/dev/null
done
# The deploy account may start the service as recall-run (and nothing else).
gcloud iam service-accounts add-iam-policy-binding "$RUN_SA" \
  --member "serviceAccount:$DEPLOY_SA" --role roles/iam.serviceAccountUser >/dev/null

# put_secret NAME VALUE: create the secret or add a new version, readable by recall-run only.
put_secret() {
  if ! gcloud secrets describe "$1" >/dev/null 2>&1; then
    gcloud secrets create "$1" --replication-policy automatic >/dev/null
  fi
  printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=- >/dev/null
  gcloud secrets add-iam-policy-binding "$1" --member "serviceAccount:$RUN_SA" \
    --role roles/secretmanager.secretAccessor >/dev/null
  echo "    stored $1"
}

# ask_secret NAME PROMPT: prompt without echoing; Enter keeps an existing value.
ask_secret() {
  local value
  read -r -s -p "$2 (input hidden${3:+; Enter keeps the current one}): " value
  echo
  if [ -n "$value" ]; then
    put_secret "$1" "$value"
  elif [ -z "${3:-}" ]; then
    echo "    $1 is required" >&2
    exit 1
  fi
}

exists() { gcloud secrets describe "$1" >/dev/null 2>&1 && echo yes || true; }

echo "==> Secrets (paste each value, then Enter)"
ask_secret recall-database-url "Neon connection string (postgresql://...)" "$(exists recall-database-url)"
ask_secret recall-llm-api-key "LLM API key" "$(exists recall-llm-api-key)"
ask_secret recall-demo-password "Demo password to give visitors" "$(exists recall-demo-password)"
if [ -z "$(exists recall-readonly-db-password)" ]; then
  put_secret recall-readonly-db-password "$(openssl rand -hex 24)"
fi

echo "==> Workload Identity Federation for $REPO"
if ! gcloud iam workload-identity-pools describe "$POOL" --location global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location global \
    --display-name "GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" --location global \
  --workload-identity-pool "$POOL" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location global \
    --workload-identity-pool "$POOL" --display-name "GitHub" \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition "assertion.repository == '$REPO'"
fi
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${REPO}" \
  >/dev/null

cat <<EOF

Done. In GitHub: https://github.com/${REPO}/settings/variables/actions
add these repository *variables* (Variables tab, not Secrets; none of them is secret):

  GCP_PROJECT_ID                  ${PROJECT}
  GCP_REGION                      ${REGION}
  GCP_WORKLOAD_IDENTITY_PROVIDER  projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}
  GCP_SERVICE_ACCOUNT             ${DEPLOY_SA}

Then re-run the latest workflow on the Actions tab (or push to main).
EOF
