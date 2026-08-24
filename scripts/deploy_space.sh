#!/usr/bin/env bash
# Assemble and push the HF Space. Usage: scripts/deploy_space.sh <hf-username>
set -euo pipefail
cd "$(dirname "$0")/.."
USER=${1:?usage: deploy_space.sh <hf-username>}
SPACE=$USER/knee-mri-detect
HF=.venv/bin/hf   # NOT the bare `hf` on PATH: Homebrew's higgsfield package aliases itself as `hf`
$HF auth whoami >/dev/null || { echo "run: .venv/bin/hf auth login"; exit 1; }
(cd frontend && npm run build)
$HF repo create "$SPACE" --repo-type space 2>/dev/null || true
D=$(mktemp -d)
cp deploy/Dockerfile deploy/README.md "$D/"
cp requirements.txt "$D/"
rsync -a --exclude __pycache__ backend ml "$D/"
rsync -a frontend/dist "$D/frontend/"
rsync -a samples "$D/"
mkdir -p "$D/ml/models"
cp ml/models/*.pt ml/models/eval.json ml/models/*_metrics.json "$D/ml/models/"
$HF upload "$SPACE" "$D" . --repo-type space --commit-message "deploy $(date +%F-%H%M)"
rm -rf "$D"
echo "https://huggingface.co/spaces/$SPACE"
