#!/usr/bin/env bash
# One-time helper to push this repo to YOUR GitHub account.
# Usage:  ./push_to_github.sh https://github.com/<you>/<repo>.git
set -euo pipefail
REMOTE="${1:-}"
if [[ -z "$REMOTE" ]]; then
  echo "Usage: ./push_to_github.sh https://github.com/<you>/<repo>.git"
  exit 1
fi
git init -b main
git add .
git commit -m "QSQ-FS: quorum-sensing feature selection for multimodal diabetes risk prediction"
git remote add origin "$REMOTE" 2>/dev/null || git remote set-url origin "$REMOTE"
git push -u origin main
echo
echo "Pushed. Next: deploy on Streamlit Community Cloud (see DEPLOYMENT.md)."
echo "Main file path: app.py"
