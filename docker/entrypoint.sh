#!/bin/bash
set -e

export PATH="/usr/local/bin:${PATH:-/usr/bin:/bin}"
eval "$(micromamba shell hook --shell bash --root-prefix ${MAMBA_ROOT_PREFIX:-/opt/conda})"
micromamba activate openvla_env

# Editable install (project mounted at /workspace/openvla-oft)
if [ -f "/workspace/openvla-oft/pyproject.toml" ]; then
    echo ">> Installing editable package..."
    cd /workspace/openvla-oft && pip install -e . && cd - > /dev/null
fi

# packaging + ninja (for flash-attn build); flash-attn when INCLUDE_FLASH_ATTN=1 , must be installed after editable install
echo ">> Installing packaging ninja..."
pip install packaging ninja
if [ "${INCLUDE_FLASH_ATTN:-1}" = "1" ]; then
    if pip show flash-attn > /dev/null 2>&1; then
        echo ">> flash-attn already installed, skipping."
    else
        echo ">> Installing flash-attn==2.5.5..."
        pip install "flash-attn==2.5.5" --no-build-isolation
    fi
fi

echo ">> Ready."
exec "$@"
