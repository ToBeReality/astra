#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv_astra"
ENV_FILE="${SCRIPT_DIR}/env_astra.sh"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SCRIPT_DIR}/requirements_Astra.txt"
python -m pip install -e "${SCRIPT_DIR}"
python -m pip install -e "${SCRIPT_DIR}/lmms-eval"

cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DECORD_NUM_THREADS=8
export PYTHONPATH=${SCRIPT_DIR}:${SCRIPT_DIR}/lmms-eval
export PATH=${VENV_DIR}/bin:\${PATH}
EOF

chmod +x "${ENV_FILE}"
echo "Astra setup completed."
echo "Run: source ${ENV_FILE}"
echo "Then: bash ${SCRIPT_DIR}/run_llava_onevision.sh"
