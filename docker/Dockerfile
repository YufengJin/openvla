# =============================================================================
# Base: CUDA 11.8 (OpenVLA-OFT)
# =============================================================================
ARG CUDA_VERSION=11.8
FROM nvidia/cuda:${CUDA_VERSION}.0-cudnn8-devel-ubuntu20.04
ARG INCLUDE_LIBERO=0
ARG INCLUDE_FLASH_ATTN=1
ARG ENV_NAME=openvla_env

# Basic env
ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# -----------------------------------------------------------------------------
# 1. System dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    curl wget tar procps bzip2 ca-certificates git build-essential cmake ninja-build unzip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "${INCLUDE_LIBERO}" = "1" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
        libx11-6 libxext6 libxrender1 libsm6 libegl1 libosmesa6 \
        lsof psmisc \
    && rm -rf /var/lib/apt/lists/*; fi

# -----------------------------------------------------------------------------
# 2. Micromamba
# -----------------------------------------------------------------------------
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# -----------------------------------------------------------------------------
# 3. Python environment (3.10)
# -----------------------------------------------------------------------------
RUN micromamba create -y -n ${ENV_NAME} python=3.10.* -c conda-forge && micromamba clean -a -y
RUN micromamba shell init --shell bash
RUN echo '' >> /root/.bashrc && \
    echo '# Micromamba' >> /root/.bashrc && \
    echo 'if [ -f /opt/conda/etc/profile.d/mamba.sh ]; then . /opt/conda/etc/profile.d/mamba.sh; fi' >> /root/.bashrc && \
    echo "micromamba activate ${ENV_NAME}" >> /root/.bashrc
RUN echo 'if [ -f ~/.bashrc ]; then . ~/.bashrc; fi' >> /root/.bash_profile

# -----------------------------------------------------------------------------
# 4. PyTorch (2.2.0 + CUDA 11.8)
# -----------------------------------------------------------------------------
RUN micromamba run -n ${ENV_NAME} pip install --upgrade pip
RUN micromamba run -n ${ENV_NAME} pip install \
    torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu118

# -----------------------------------------------------------------------------
# 5. openvla-oft (editable install at build time; entrypoint re-runs when mounted)
# -----------------------------------------------------------------------------
WORKDIR /tmp/openvla-oft
COPY . .
RUN micromamba run -n ${ENV_NAME} pip install -e .

# -----------------------------------------------------------------------------
# 6. Flash Attention 2 (for training; optional)
# -----------------------------------------------------------------------------
RUN micromamba run -n ${ENV_NAME} pip install packaging ninja
RUN if [ "${INCLUDE_FLASH_ATTN}" = "1" ]; then \
    micromamba run -n ${ENV_NAME} pip install "flash-attn==2.5.5" --no-build-isolation; fi

# -----------------------------------------------------------------------------
# 7. LIBERO (optional; for simulation benchmark)
# -----------------------------------------------------------------------------
RUN if [ "${INCLUDE_LIBERO}" = "1" ]; then \
    mkdir -p /opt/third_party && cd /opt/third_party && \
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && \
    micromamba run -n ${ENV_NAME} pip install -e /opt/third_party/LIBERO && \
    micromamba run -n ${ENV_NAME} pip install -r /tmp/openvla-oft/experiments/robot/libero/libero_requirements.txt; fi

ENV INCLUDE_LIBERO=${INCLUDE_LIBERO}
ENV INCLUDE_FLASH_ATTN=${INCLUDE_FLASH_ATTN}

# -----------------------------------------------------------------------------
# 8. Workspace (project mounted at runtime; entrypoint does pip install -e .)
# -----------------------------------------------------------------------------
WORKDIR /workspace/openvla-oft
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]
