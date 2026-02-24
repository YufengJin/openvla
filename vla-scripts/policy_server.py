"""
policy_server.py — OpenVLA policy server over WebSocket.

Serves OpenVLA-OFT as a WebSocket policy compatible with LIBERO and RoboCasa.
Client sends raw robosuite obs; server handles all remapping (raw LIBERO, raw RoboCasa, or prepared).
Output: action_dim 7 (cartesian_pose + gripper). Use --arm_controller cartesian_pose on client.

Usage:
    python vla-scripts/policy_server.py --port 8000
    python vla-scripts/policy_server.py --execute_steps 1 --port 8000  # one-step mode

Client (use --arm_controller cartesian_pose):
    python LIBERO/scripts/run_demo.py --policy_server_addr localhost:8000 --task_suite_name libero_10
    python robocasa/scripts/run_demo.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Required for `from experiments.robot...` when run as python vla-scripts/policy_server.py
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation remapping: client format -> OpenVLA format
# ---------------------------------------------------------------------------

# Required keys for each format (for validation)
RAW_LIBERO_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
RAW_ROBOCASA_KEYS = ("robot0_agentview_left_image", "robot0_eye_in_hand_image")
PREPARED_KEYS = ("primary_image", "wrist_image")


def _validate_keys(obs: Dict[str, Any], required: tuple, format_name: str) -> None:
    """Raise ValueError if any required key is missing."""
    missing = [k for k in required if k not in obs or obs[k] is None]
    if missing:
        raise ValueError(
            f"Observation format '{format_name}' requires keys {list(required)}. "
            f"Missing: {missing}. Received keys: {list(obs.keys())[:20]}..."
        )


def prepare_obs_from_libero(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw LIBERO (robosuite) obs to OpenVLA format.
    Raw keys: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos.
    """
    _validate_keys(raw_obs, RAW_LIBERO_KEYS, "raw_libero")
    primary = np.asarray(raw_obs["agentview_image"])
    wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"])
    # LIBERO renders upside-down: flipud then fliplr for 180 deg to match training
    primary = np.fliplr(np.flipud(primary))
    wrist = np.fliplr(np.flipud(wrist))
    out = {
        "full_image": primary,
        "wrist_image": wrist,
        "task_description": raw_obs.get("task_description", ""),
    }
    if "robot0_eef_pos" in raw_obs and raw_obs["robot0_eef_pos"] is not None:
        out["state"] = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"]),
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"]),
        ]).astype(np.float64)
    else:
        out["state"] = np.zeros(8, dtype=np.float64)
    return out


def prepare_obs_from_robocasa(raw_obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert raw RoboCasa (robosuite) obs to OpenVLA format.
    Raw keys: robot0_agentview_left_image, robot0_eye_in_hand_image, robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos.
    """
    _validate_keys(raw_obs, RAW_ROBOCASA_KEYS, "raw_robocasa")
    primary = np.asarray(raw_obs["robot0_agentview_left_image"])
    wrist = np.asarray(raw_obs["robot0_eye_in_hand_image"])
    primary = np.fliplr(np.flipud(primary))
    wrist = np.fliplr(np.flipud(wrist))
    out = {
        "full_image": primary,
        "wrist_image": wrist,
        "task_description": raw_obs.get("task_description", ""),
    }
    if "robot0_eef_pos" in raw_obs and raw_obs["robot0_eef_pos"] is not None:
        out["state"] = np.concatenate([
            np.asarray(raw_obs["robot0_eef_pos"]),
            _quat2axisangle(raw_obs["robot0_eef_quat"]),
            np.asarray(raw_obs["robot0_gripper_qpos"]),
        ]).astype(np.float64)
    else:
        out["state"] = np.zeros(8, dtype=np.float64)
    return out


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle. Same as libero_utils.quat2axisangle."""
    quat = np.asarray(quat).flatten()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def remap_obs_prepared(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap prepared observation (primary_image, wrist_image, state/proprio) to OpenVLA format.
    Client sends: primary_image (flipud), wrist_image (flipud), task_description,
    and optionally state or robot0_eef_pos/robot0_eef_quat/robot0_gripper_qpos or proprio.
    """
    _validate_keys(obs, PREPARED_KEYS, "prepared")
    primary = np.asarray(obs["primary_image"])
    wrist = np.asarray(obs["wrist_image"])
    out = {
        "full_image": np.fliplr(primary),
        "wrist_image": np.fliplr(wrist),
        "task_description": obs.get("task_description", ""),
    }
    # State: LIBERO sends state (8D) or robot0_*; RoboCasa sends proprio (9D: gripper+eef_pos+eef_quat)
    if "state" in obs and obs["state"] is not None:
        out["state"] = np.asarray(obs["state"], dtype=np.float64)
    elif "proprio" in obs and obs["proprio"] is not None:
        p = np.asarray(obs["proprio"]).flatten()
        if len(p) >= 9:
            gripper, eef_pos, eef_quat = p[:2], p[2:5], p[5:9]
            axis_angle = _quat2axisangle(eef_quat)
            out["state"] = np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float64)
        else:
            out["state"] = np.zeros(8, dtype=np.float64)
    elif "robot0_eef_pos" in obs and obs["robot0_eef_pos"] is not None:
        out["state"] = np.concatenate([
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]).astype(np.float64)
    else:
        out["state"] = np.zeros(8, dtype=np.float64)
    return out


def remap_obs_to_openvla(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect obs format and remap to OpenVLA format (full_image, wrist_image, state, task_description).
    Supports: raw LIBERO, raw RoboCasa, prepared (primary_image/wrist_image).
    Raises ValueError if required keys are missing.
    """
    # Format A: raw LIBERO (agentview_image)
    if "agentview_image" in obs and obs["agentview_image"] is not None:
        return prepare_obs_from_libero(obs)
    # Format B: raw RoboCasa (robot0_agentview_left_image)
    if "robot0_agentview_left_image" in obs and obs["robot0_agentview_left_image"] is not None:
        return prepare_obs_from_robocasa(obs)
    # Format C: prepared (primary_image, wrist_image)
    if "primary_image" in obs and obs["primary_image"] is not None:
        return remap_obs_prepared(obs)
    raise ValueError(
        "Observation format not recognized. Expected one of: "
        "raw LIBERO (agentview_image, robot0_eye_in_hand_image), "
        "raw RoboCasa (robot0_agentview_left_image, robot0_eye_in_hand_image), "
        f"prepared (primary_image, wrist_image). Received keys: {list(obs.keys())[:25]}..."
    )


# ---------------------------------------------------------------------------
# Action post-processing: OpenVLA output -> env format
# ---------------------------------------------------------------------------

def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Normalize gripper from [0,1] to [-1,+1]. Same as robot_utils.normalize_gripper_action."""
    out = action.copy()
    orig_low, orig_high = 0.0, 1.0
    out[..., -1] = 2 * (out[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        out[..., -1] = np.sign(out[..., -1])
    return out


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign for env where -1=open, +1=close. Same as robot_utils.invert_gripper_action."""
    out = action.copy()
    out[..., -1] *= -1.0
    return out


def postprocess_action_for_env(action: np.ndarray, invert_gripper: bool = True) -> np.ndarray:
    """
    Convert OpenVLA action (0=close, 1=open) to env format (-1=open, +1=close).
    Apply normalize_gripper + invert for LIBERO/RoboCasa OSC_POSE.
    """
    action = _normalize_gripper_action(action, binarize=True)
    if invert_gripper:
        action = _invert_gripper_action(action)
    return np.asarray(action, dtype=np.float64)


# ---------------------------------------------------------------------------
# OpenVLA Policy
# ---------------------------------------------------------------------------

@dataclass
class PolicyServerConfig:
    pretrained_checkpoint: str = ""
    unnorm_key: str = "libero_spatial_no_noops"
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    lora_rank: int = 32
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    invert_gripper: bool = True  # LIBERO/RoboCasa env: -1=open, +1=close
    execute_steps: int = 8  # Action chunk: how many steps to execute per query (1 = one-step)


class OpenVLAPolicy:
    """Wraps OpenVLA inference with obs remapping and action post-processing."""

    def __init__(self, cfg: PolicyServerConfig) -> None:
        self.cfg = cfg

        # Lazy load heavy deps
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_processor,
            get_proprio_projector,
            get_vla,
            get_vla_action,
        )
        from experiments.robot.robot_utils import get_image_resize_size

        # Build a minimal config object for openvla_utils
        class Cfg:
            pass

        c = Cfg()
        c.pretrained_checkpoint = cfg.pretrained_checkpoint
        c.use_l1_regression = cfg.use_l1_regression
        c.use_diffusion = cfg.use_diffusion
        c.use_film = cfg.use_film
        c.num_images_in_input = cfg.num_images_in_input
        c.use_proprio = cfg.use_proprio
        c.center_crop = cfg.center_crop
        c.lora_rank = cfg.lora_rank
        c.load_in_8bit = cfg.load_in_8bit
        c.load_in_4bit = cfg.load_in_4bit
        c.unnorm_key = cfg.unnorm_key
        c.model_family = "openvla"

        self._cfg = c

        self.vla = get_vla(c)
        self.processor = get_processor(c)
        self.action_head = get_action_head(c, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(c, self.vla.llm_dim, 8)
        self.resize_size = get_image_resize_size(c)

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Init: action_dim only, no images
        has_images = any(
            k in obs and obs.get(k) is not None
            for k in ("primary_image", "agentview_image", "robot0_agentview_left_image")
        )
        if "action_dim" in obs and not has_images:
            return {"actions": np.zeros(int(obs["action_dim"]), dtype=np.float64)}

        openvla_obs = remap_obs_to_openvla(obs)
        from experiments.robot.openvla_utils import get_vla_action

        actions = get_vla_action(
            self._cfg,
            self.vla,
            self.processor,
            openvla_obs,
            openvla_obs["task_description"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        # actions: List[np.ndarray], each (action_dim,)
        actions_array = np.stack(actions, axis=0)
        actions_array = postprocess_action_for_env(
            actions_array, invert_gripper=self.cfg.invert_gripper
        )
        # Slice to execute_steps for ActionChunkBroker (use first N of chunk)
        actions_array = actions_array[: self.cfg.execute_steps]
        return {"actions": actions_array}

    def reset(self) -> None:
        """Reset internal state (OpenVLA is stateless)."""
        pass


def main():
    import argparse

    from policy_websocket import ActionChunkBroker, BasePolicy, WebsocketPolicyServer

    parser = argparse.ArgumentParser(
        description="OpenVLA policy server (WebSocket, auto-detects obs format)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--pretrained_checkpoint",
        type=str,
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
        help="OpenVLA checkpoint path or HF repo",
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="libero_spatial_no_noops",
        help="Dataset key for action un-normalization",
    )
    parser.add_argument(
        "--use_film",
        action="store_true",
        help="Use FiLM (for ALOHA-style checkpoints)",
    )
    parser.add_argument(
        "--num_images_in_input",
        type=int,
        default=2,
        help="Number of images (2 for LIBERO, 3 for ALOHA)",
    )
    parser.add_argument(
        "--no_proprio",
        action="store_true",
        dest="no_proprio",
        help="Disable proprio input",
    )
    parser.add_argument(
        "--execute_steps",
        type=int,
        default=8,
        help="Action chunk: steps to execute per query (1 = one-step)",
    )
    parser.add_argument(
        "--no_invert_gripper",
        action="store_true",
        dest="no_invert_gripper",
        help="Do not invert gripper (env expects 0=open, 1=close)",
    )
    args = parser.parse_args()

    cfg = PolicyServerConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        unnorm_key=args.unnorm_key,
        use_film=args.use_film,
        num_images_in_input=args.num_images_in_input,
        use_proprio=not args.no_proprio,
        invert_gripper=not args.no_invert_gripper,
        execute_steps=args.execute_steps,
    )

    logger.info("Loading OpenVLA checkpoint: %s", cfg.pretrained_checkpoint)
    inner = OpenVLAPolicy(cfg)

    if cfg.execute_steps > 1:
        broker = ActionChunkBroker(inner, action_horizon=cfg.execute_steps)

        class ResetOnInit(BasePolicy):
            def __init__(self, p: BasePolicy):
                self._p = p

            def infer(self, obs):
                has_images = any(
                    k in obs and obs.get(k) is not None
                    for k in ("primary_image", "agentview_image", "robot0_agentview_left_image")
                )
                if "action_dim" in obs and not has_images:
                    self._p.reset()
                return self._p.infer(obs)

            def reset(self):
                self._p.reset()

        policy = ResetOnInit(broker)
    else:
        policy = inner

    metadata = {
        "policy_name": "OpenVLAPolicy",
        "action_dim": 7,
        "execute_steps": cfg.execute_steps,
        "checkpoint": cfg.pretrained_checkpoint,
    }

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    print(
        f"OpenVLA policy server on ws://{args.host}:{args.port} "
        f"(execute_steps={cfg.execute_steps})"
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print("Server stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
