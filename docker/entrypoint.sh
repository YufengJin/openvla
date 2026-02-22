#!/bin/bash
set -e

export PATH="/usr/local/bin:${PATH:-/usr/bin:/bin}"
eval "$(micromamba shell hook --shell bash --root-prefix ${MAMBA_ROOT_PREFIX:-/opt/conda})"
micromamba activate openvla_env

# Editable install (project mounted at /workspace/openvla-oft)
if [ -f "/workspace/openvla-oft/pyproject.toml" ]; then
    echo ">> Installing editable package..."
    cd /workspace/openvla-oft && pip install -e . --no-deps > /dev/null 2>&1 && cd - > /dev/null
fi

echo ">> Ready."
exec "$@"
