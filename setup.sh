#!/usr/bin/env bash
# One-time environment setup for running the notebooks.
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=youtube-spam-detection --display-name="YouTube Spam Detection"

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Launch Jupyter:       jupyter lab"
echo "In Cursor: select kernel 'YouTube Spam Detection' or .venv/bin/python"
