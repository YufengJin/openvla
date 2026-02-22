# Docker Setup for OpenVLA-OFT

## Prerequisites

- Docker (Compose v2+)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

## Build

From project root:

```bash
# Base (OpenVLA-OFT + flash-attn)
docker compose -f docker/docker-compose.headless.yaml build

# With LIBERO simulation (~extra deps)
docker compose -f docker/docker-compose.headless.yaml build --build-arg INCLUDE_LIBERO=1

# Skip flash-attn if build fails (inference-only use case)
docker compose -f docker/docker-compose.headless.yaml build --build-arg INCLUDE_FLASH_ATTN=0
```

| Build arg           | Default | Description                                    |
|--------------------|---------|------------------------------------------------|
| `INCLUDE_LIBERO`   | 0       | Set to `1` for LIBERO simulation benchmark     |
| `INCLUDE_FLASH_ATTN` | 1    | Set to `0` to skip flash-attn (long build)     |
| `CUDA_VERSION`     | 11.8    | Base image CUDA version                        |

## Volume mounts

Update paths in the compose file to match your system.

| Host path                | Container path              | Purpose                         |
|--------------------------|-----------------------------|---------------------------------|
| `../` (openvla-oft)      | `/workspace/openvla-oft`    | Project (editable install)       |
| `/path/to/datasets`      | `/workspace/datasets`       | RLDS datasets (optional, ro)    |
| `${HOME}/.cache/huggingface` | `/root/.cache/huggingface` | HuggingFace model cache         |

**Datasets**: For LIBERO fine-tuning, download RLDS data via `git clone https://huggingface.co/datasets/openvla/modified_libero_rlds` and mount the parent directory so that `--data_root_dir /workspace/datasets/rlds` (or the path containing `libero_spatial_no_noops`, etc.) is correct. Remove or comment out the datasets volume if unused.

## Usage

### Headless (training / inference)

```bash
docker compose -f docker/docker-compose.headless.yaml up -d
docker exec -it openvla-dev-headless bash
# Inside: cd /workspace/openvla-oft
```

### One-off

```bash
docker run --rm --gpus all \
  -v /path/to/openvla-oft:/workspace/openvla-oft \
  -v /path/to/datasets:/workspace/datasets \
  -v ${HOME}/.cache/huggingface:/root/.cache/huggingface \
  openvla-oft:latest python -c "print('hello')"
```

## Verify installation

After starting the container, run the Quick Start verification (from [README.md](../README.md)):

```bash
docker exec openvla-dev-headless bash -c "eval \"\$(micromamba shell hook --shell bash --root-prefix /opt/conda)\" && micromamba activate openvla_env && cd /workspace/openvla-oft && python docker/verify_env.py"
```

Or from inside the container (`docker exec -it openvla-dev-headless bash`):

```bash
cd /workspace/openvla-oft
python docker/verify_env.py
```

This loads the pretrained checkpoint and generates an action chunk from a sample observation.

## Entrypoint

On each start:

1. Activates `openvla_env`
2. Runs `pip install -e . --no-deps` when project is mounted at `/workspace/openvla-oft`

## Example commands

### LIBERO evaluation (requires pretrained checkpoint)

```bash
cd /workspace/openvla-oft
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --task_suite_name libero_spatial
```

### LIBERO fine-tuning

Ensure RLDS datasets are mounted under `/workspace/datasets`. Example:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /workspace/datasets/rlds \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir /workspace/openvla-oft/runs \
  --use_l1_regression True \
  --batch_size 8 \
  --wandb_entity "YOUR_ENTITY" \
  --wandb_project "YOUR_PROJECT"
```

Adjust `--data_root_dir` to match your mount layout (e.g. if `modified_libero_rlds` is at `/workspace/datasets`, use `--data_root_dir /workspace/datasets`).

## LIBERO

When built with `INCLUDE_LIBERO=1`:

- LIBERO is installed from `/opt/third_party/LIBERO`
- Use for running LIBERO simulation evaluations and training
- See [LIBERO.md](../LIBERO.md) for dataset download and usage details
