#!/usr/bin/env bash
# Usage:
#   chmod +x custom-instructions/setup.sh
#   ./custom-instructions/setup.sh
#
# After the script completes, activate the virtualenv with:
#   source custom-instructions/.venv/bin/activate

set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

DOCTOR_TOOLCHAIN_ROOT="${RISCV_GNU_TOOLCHAIN_ROOT:-}"
if [ -n "$DOCTOR_TOOLCHAIN_ROOT" ]; then
  python "$PWD/riscv_ci.py" doctor --toolchain-root "$DOCTOR_TOOLCHAIN_ROOT" || true
else
  python "$PWD/riscv_ci.py" doctor || true
fi

cat <<'EOF'
Setup complete.
Activate the virtualenv with:
  source custom-instructions/.venv/bin/activate
Then run the CLI, for example:
  python custom-instructions/riscv_ci.py validate
EOF
