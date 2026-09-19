#!/usr/bin/env bash
# One-time environment setup for the QMOF-CGCNN project.
# Usage:  bash setup.sh
# Options (env vars):
#   ENV_NAME=.venv                     venv directory name
#   TORCH_INDEX=auto|<index-url>       "auto" detects GPU; or force e.g.
#                                      https://download.pytorch.org/whl/cu118
set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-auto}"

echo "==> [1/4] Creating virtualenv ($ENV_NAME)"
if [ ! -d "$ENV_NAME" ]; then
    python3 -m venv "$ENV_NAME"
fi
# shellcheck disable=SC1091
source "$ENV_NAME/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "==> [2/4] Installing PyTorch"
if [ "$TORCH_INDEX" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "    GPU detected -> CUDA 12.1 wheels"
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    else
        echo "    No GPU detected -> CPU wheels"
        pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
else
    pip install torch --index-url "$TORCH_INDEX"
fi

echo "==> [3/4] Installing project dependencies"
pip install -r requirements.txt

echo "==> [4/4] Creating project directories"
mkdir -p data/raw/cifs data/processed \
         outputs/graphs outputs/splits outputs/models \
         outputs/figures outputs/metrics outputs/predictions \
         logs

python - <<'EOF'
import torch, torch_geometric, pymatgen, pandas, sklearn
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("pymatgen / pandas / sklearn OK")
EOF

echo ""
echo "Setup complete. Activate with:  source $ENV_NAME/bin/activate"
