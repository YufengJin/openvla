# OpenVLA-OFT

**Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success** — [Project website](https://openvla-oft.github.io/) · [Paper](https://arxiv.org/abs/2502.19645) · [Summary video](https://youtu.be/T3Zkkr_NTSA)

Based on [moojink/openvla-oft](https://github.com/moojink/openvla-oft) (Stanford) — OpenVLA with OFT fine-tuning.

**This fork** adds installation optimizations and [policy-websocket](https://github.com/YufengJin/policy_websocket) integration for remote inference. Compatible with [role-ros2](https://github.com/YufengJin/role-ros2) — robot learning full stack on ROS2. See [role-ros2 README](https://github.com/YufengJin/role-ros2/blob/main/README.md) for policy deployment.

## Requirements

| Mode        | GPU memory (typical) |
| ----------- | -------------------- |
| Inference (LIBERO sim) | ~16 GB VRAM   |
| Inference (ALOHA)      | ~18 GB VRAM   |
| Training               | 1–8 GPUs, 27–80 GB each (bfloat16); see [FAQ](https://openvla-oft.github.io/#train-compute) |

## Installation

```bash
git clone https://github.com/YufengJin/openvla.git
cd openvla
```

Conda / micromamba environment: follow **[SETUP.md](SETUP.md)** (`pip install -e .` pulls `policy-websocket` from `pyproject.toml`).

## Docker

```bash
# Build and start (headless). Default compose enables LIBERO + flash-attn build args.
docker compose -f docker/docker-compose.headless.yaml up --build -d

# Enter container
docker exec -it openvla-dev-headless bash
cd /workspace/openvla-oft
```

**Policy server in Docker**:

```bash
docker exec -it openvla-dev-headless bash
cd /workspace/openvla-oft
micromamba run -n openvla_env python vla-scripts/policy_server.py --port 8000
```

See [docker/README.md](docker/README.md) for build args (`INCLUDE_LIBERO`, `INCLUDE_FLASH_ATTN`), X11 GUI compose, dataset mounts.

## Policy Server (policy-websocket)

Serves OpenVLA-OFT over WebSocket. Compatible with:

- [role-ros2](https://github.com/YufengJin/role-ros2) — robot learning full stack on ROS2
- [RoboCasa](https://robocasa.github.io/) — large-scale simulation benchmark
- [LIBERO](https://github.com/YufengJin/LIBERO) — lifelong robot learning benchmark

```bash
# Default HF checkpoint + port 8000
python vla-scripts/policy_server.py --port 8000

# One-step mode (execute_steps=1)
python vla-scripts/policy_server.py --execute_steps 1 --port 8000

# Custom checkpoint / unnorm key
python vla-scripts/policy_server.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --unnorm_key libero_spatial_no_noops
```

**Client usage** (use **`--arm_controller cartesian_pose`**, not `joint_vel`):

```bash
# LIBERO
python LIBERO/scripts/run_demo.py --arm_controller cartesian_pose --policy_server_addr localhost:8000 --task_suite_name libero_10

# RoboCasa
python robocasa/scripts/run_demo.py --arm_controller cartesian_pose --policy_server_addr localhost:8000 --task_name PnPCounterToCab
```

Output: **action_dim 7** (cartesian pose + gripper).

**Dependency**: declared in `pyproject.toml` as `policy-websocket` (Git install). After `pip install -e .` it should be present; otherwise:  
`pip install "policy-websocket @ git+https://github.com/YufengJin/policy_websocket.git"`

## Inference (Quick)

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="libero_spatial_no_noops",
)
vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as f:
    observation = pickle.load(f)
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)
```

## Training & Evaluation

- [LIBERO.md](LIBERO.md) — LIBERO sim fine-tuning / eval  
- [ALOHA.md](ALOHA.md) — real-world ALOHA tasks  
- [docs/ARCHITECTURE_AND_FINETUNE.md](docs/ARCHITECTURE_AND_FINETUNE.md) — model & fine-tuning details

## More

- [SETUP.md](SETUP.md) — full conda setup  
- [Upstream openvla-oft](https://github.com/moojink/openvla-oft) — original repo

## Support

Open a GitHub issue; if no reply within ~2 business days, email Moo Jin Kim (moojink@cs.stanford.edu).

## Citation

```bibtex
@article{kim2025fine,
  title={Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success},
  author={Kim, Moo Jin and Finn, Chelsea and Liang, Percy},
  journal={arXiv preprint arXiv:2502.19645},
  year={2025}
}
```
