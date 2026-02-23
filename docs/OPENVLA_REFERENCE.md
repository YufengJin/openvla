# OpenVLA-OFT 技术参考文档

本文档汇总 OpenVLA-OFT 的 checkpoint 类型、观测/动作接口格式，以及 LIBERO/ALOHA 实验中的数据格式转换流程。

---

## 1. Checkpoint 概览

### 1.1 基础模型

| Checkpoint | unnorm_key | 说明 |
|------------|------------|------|
| `openvla/openvla-7b` | `bridge_orig` | 预训练基座模型，在 Bridge 数据集上训练 |

### 1.2 LIBERO 微调 Checkpoint（HuggingFace Hub）

| Checkpoint | unnorm_key | 任务套件 |
|------------|------------|----------|
| [moojink/openvla-7b-oft-finetuned-libero-spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial) | `libero_spatial_no_noops` | LIBERO-Spatial |
| [moojink/openvla-7b-oft-finetuned-libero-object](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object) | `libero_object_no_noops` | LIBERO-Object |
| [moojink/openvla-7b-oft-finetuned-libero-goal](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-goal) | `libero_goal_no_noops` | LIBERO-Goal |
| [moojink/openvla-7b-oft-finetuned-libero-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10) | `libero_10_no_noops` | LIBERO-10 (LIBERO-Long) |
| [moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10) | `libero_4_task_suites_no_noops` | 四任务套件联合训练 |

### 1.3 ALOHA 微调

使用本地 checkpoint 路径。`unnorm_key` 为数据集名称，例如：
- `aloha1_put_X_into_pot_300_demos`
- `aloha1_fold_shorts_20_demos`
- `aloha1_fold_shirt_30_demos`
- `aloha1_scoop_X_into_bowl_45_demos`

---

## 2. 观测格式与 Action Space

### 2.1 观测格式 (Observation)

推理时传入 `get_vla_action` 的 observation 为 dict，结构如下（参见 [experiments/robot/openvla_utils.py](../experiments/robot/openvla_utils.py) 中 `get_vla_action`）：

```python
observation = {
    "full_image": primary_image,       # 主视角图像 (numpy, HxWx3, uint8)
    # LIBERO: 1 个手腕相机
    "wrist_image": wrist_image,
    # ALOHA: 左手腕 + 右手腕（num_images_in_input=3 时）
    "left_wrist_image": left_wrist_image,
    "right_wrist_image": right_wrist_image,
    "state": proprio_state,            # 本体感知状态 (numpy array)
    # deploy 接口使用 "instruction" 作为任务描述
    "task_description": str,           # 或 "instruction"
}
```

**图像要求：**
- shape: `(H, W, 3)`，dtype: `np.uint8`
- 推理前会 resize 到 224x224（`OPENVLA_IMAGE_SIZE`）
- 若训练时使用 `image_aug=True`，推理需设 `center_crop=True`，做 90% 中心裁剪以与训练分布一致

**本体状态维度**（[prismatic/vla/constants.py](../prismatic/vla/constants.py)）：

| 平台 | PROPRIO_DIM | 构成 |
|------|-------------|------|
| LIBERO | 8 | ee_pos (3) + quat2axisangle (3) + gripper_qpos (2) |
| ALOHA | 14 | 双臂 14 维关节角 (qpos) |

### 2.2 输出 Action Space

来自 [prismatic/vla/constants.py](../prismatic/vla/constants.py)：

| 平台 | ACTION_DIM | NUM_ACTIONS_CHUNK | 动作含义 |
|------|------------|-------------------|----------|
| LIBERO | 7 | 8 | 末端位姿 (6D) + 夹爪 (1D)，相对动作，BOUNDS_Q99 归一化 |
| ALOHA | 14 | 25 | 双臂 14 维关节角（绝对），BOUNDS 归一化 |

**LIBERO 动作约定：**
- 前 6 维：末端位姿 (x, y, z + axis-angle)
- 第 7 维：夹爪。训练时 RLDS dataloader 对齐为 0=close、1=open；环境执行前需调用 `invert_gripper_action` 还原为 -1=open、+1=close（参见 [experiments/robot/robot_utils.py](../experiments/robot/robot_utils.py) 与 [experiments/robot/libero/run_libero_eval.py](../experiments/robot/libero/run_libero_eval.py) 中 `process_action`）

---

## 3. 数据格式与转换

### 3.1 LIBERO 流程

```mermaid
flowchart LR
    HDF5[原始 HDF5] --> Regenerate[regenerate_libero_dataset.py]
    Regenerate --> HDF5NoOps[HDF5 no_noops]
    HDF5NoOps --> RLDS[RLDS Builder 转换为 RLDS]
    RLDS --> TrainData[训练数据]
    TrainData --> LiberoTransform[libero_dataset_transform]
```

**转换内容：**

1. **regenerate_libero_dataset.py**（[experiments/robot/libero/regenerate_libero_dataset.py](../experiments/robot/libero/regenerate_libero_dataset.py)）：
   - 图像分辨率 256x256（原始为 128x128）
   - 过滤 no-op 动作（接近零且夹爪不变）
   - 过滤失败 demo
   - 注：HDF5 → RLDS 时会将图像旋转 180°

2. **libero_dataset_transform**（[prismatic/vla/datasets/rlds/oxe/transforms.py](../prismatic/vla/datasets/rlds/oxe/transforms.py) 第 827–841 行）：
   - 夹爪：-1~1 → clip 到 0~1 → flip → 得到 0=close、1=open
   - 从 `observation.state` 抽取 `EEF_state`（前 6 维）和 `gripper_state`（后 2 维）

3. **评估时**（[experiments/robot/libero/libero_utils.py](../experiments/robot/libero/libero_utils.py)）：
   - `agentview_image` → `full_image`
   - `robot0_eye_in_hand_image` → `wrist_image`
   - 图像 `[::-1, ::-1]` 旋转 180° 与训练一致

### 3.2 ALOHA 流程

```mermaid
flowchart LR
    Raw[原始 HDF5 480x640] --> Preprocess[preprocess_split_aloha_data.py]
    Preprocess --> HDF5_256[256x256 HDF5 + train/val split]
    HDF5_256 --> RLDS[RLDS Builder 转 RLDS]
    RLDS --> TrainData[训练数据]
    TrainData --> AlohaTransform[aloha_dataset_transform]
```

**转换内容：**

1. **preprocess_split_aloha_data.py**（[experiments/robot/aloha/preprocess_split_aloha_data.py](../experiments/robot/aloha/preprocess_split_aloha_data.py)）：
   - 图像 480x640 → 256x256（BICUBIC）
   - 按 episode 划分 train/val
   - 可选保存 `relative_action`

2. **aloha_dataset_transform**（[prismatic/vla/datasets/rlds/oxe/transforms.py](../prismatic/vla/datasets/rlds/oxe/transforms.py) 第 844–846 行）：
   - **无转换**：`return trajectory`（RLDS 已满足 OpenVLA 格式）

3. **评估时**（[experiments/robot/aloha/run_aloha_eval.py](../experiments/robot/aloha/run_aloha_eval.py)）：
   - `cam_high` → `full_image`
   - `cam_left_wrist` → `left_wrist_image`，`cam_right_wrist` → `right_wrist_image`
   - 图像 resize 到 256x256（与 preprocessing 一致）

---

## 4. 相关文件索引

| 用途 | 文件路径 |
|------|----------|
| VLA 部署服务 | [vla-scripts/deploy.py](../vla-scripts/deploy.py) |
| OpenVLA 验证 | [vla-scripts/extern/verify_openvla.py](../vla-scripts/extern/verify_openvla.py) |
| 推理与 obs 处理 | [experiments/robot/openvla_utils.py](../experiments/robot/openvla_utils.py) |
| 通用 robot 工具 | [experiments/robot/robot_utils.py](../experiments/robot/robot_utils.py) |
| LIBERO 评估 | [experiments/robot/libero/run_libero_eval.py](../experiments/robot/libero/run_libero_eval.py) |
| LIBERO 工具 | [experiments/robot/libero/libero_utils.py](../experiments/robot/libero/libero_utils.py) |
| LIBERO 数据 regenerate | [experiments/robot/libero/regenerate_libero_dataset.py](../experiments/robot/libero/regenerate_libero_dataset.py) |
| ALOHA 评估 | [experiments/robot/aloha/run_aloha_eval.py](../experiments/robot/aloha/run_aloha_eval.py) |
| ALOHA 预处理 | [experiments/robot/aloha/preprocess_split_aloha_data.py](../experiments/robot/aloha/preprocess_split_aloha_data.py) |
| 常量定义 | [prismatic/vla/constants.py](../prismatic/vla/constants.py) |
| RLDS 数据集配置 | [prismatic/vla/datasets/rlds/oxe/configs.py](../prismatic/vla/datasets/rlds/oxe/configs.py) |
| RLDS 数据转换 | [prismatic/vla/datasets/rlds/oxe/transforms.py](../prismatic/vla/datasets/rlds/oxe/transforms.py) |
| 架构与微调说明 | [docs/ARCHITECTURE_AND_FINETUNE.md](ARCHITECTURE_AND_FINETUNE.md) |
| WebSocket Policy Server | [vla-scripts/policy_server.py](../vla-scripts/policy_server.py) |
