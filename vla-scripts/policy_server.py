"""
policy_server.py — OpenVLA policy server over WebSocket.

Serves OpenVLA-OFT as a WebSocket policy compatible with LIBERO and RoboCasa run_eval
clients. Supports both one-step and action-chunk modes. All observation remapping and
action post-processing conversions are centralized here.

Usage:
    # Action chunk mode (predict 8, execute 8 — default for LIBERO checkpoint):
    python vla-scripts/policy_server.py \\
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \\
        --obs_remap libero \\
        --unnorm_key libero_spatial_no_noops \\
        --port 8000

    # One-step mode (re-query every step):
    python vla-scripts/policy_server.py \\
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \\
        --obs_remap libero \\
        --execute_steps 1 \\
        --port 8000

    # RoboCasa obs format:
    python vla-scripts/policy_server.py \\
        --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \\
        --obs_remap robocasa \\
        --port 8000

Then connect with:
    python LIBERO/scripts/run_eval.py --policy_server_addr localhost:8000
    python robocasa/scripts/run_eval.py --policy_server_addr localhost:8000
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

# Ensure openvla-oft project root is on path (vla-scripts/ is inside project)
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation remapping: client format -> OpenVLA format
# ---------------------------------------------------------------------------

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


def remap_obs_libero(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap LIBERO run_eval observation to OpenVLA format.

    Client sends: primary_image (flipud), wrist_image (flipud), task_description.
    OpenVLA training uses 180 deg rotation [::-1, ::-1]. Client uses flipud only,
    so we apply fliplr to get 180 deg and match training.
    """
    primary = np.asarray(obs["primary_image"])
    wrist = np.asarray(obs["wrist_image"])
    out = {
        "full_image": np.fliplr(primary),
        "wrist_image": np.fliplr(wrist),
        "task_description": obs.get("task_description", ""),
    }
    # LIBERO client typically does not send proprio; use zeros if missing
    if "state" in obs:
        out["state"] = np.asarray(obs["state"], dtype=np.float64)
    elif "robot0_eef_pos" in obs:
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        gripper_qpos = obs["robot0_gripper_qpos"]
        out["state"] = np.concatenate([
            np.asarray(eef_pos),
            _quat2axisangle(eef_quat),
            np.asarray(gripper_qpos),
        ]).astype(np.float64)
    else:
        out["state"] = np.zeros(8, dtype=np.float64)  # PROPRIO_DIM for LIBERO
        logger.debug("LIBERO obs has no proprio; using zeros")
    return out


def remap_obs_robocasa(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remap RoboCasa run_eval observation to OpenVLA format.

    Client sends: primary_image (flipud), secondary_image, wrist_image (flipud), proprio, task_description.
    RoboCasa proprio = gripper_qpos (2) + eef_pos (3) + eef_quat (4) = 9.
    OpenVLA LIBERO state = eef_pos (3) + axis_angle (3) + gripper_qpos (2) = 8.
    Apply fliplr to match 180 deg rotation (client uses flipud).
    """
    primary = np.asarray(obs["primary_image"])
    wrist = np.asarray(obs["wrist_image"])
    out = {
        "full_image": np.fliplr(primary),
        "wrist_image": np.fliplr(wrist),
        "task_description": obs.get("task_description", ""),
    }
    if "proprio" in obs:
        p = np.asarray(obs["proprio"]).flatten()
        # proprio = [gripper(2), eef_pos(3), eef_quat(4)]
        if len(p) >= 9:
            gripper = p[:2]
            eef_pos = p[2:5]
            eef_quat = p[5:9]
            axis_angle = _quat2axisangle(eef_quat)
            out["state"] = np.concatenate([eef_pos, axis_angle, gripper]).astype(np.float64)
        else:
            out["state"] = np.zeros(8, dtype=np.float64)
            logger.debug("RoboCasa proprio dim < 9; using zeros")
    elif "state" in obs:
        out["state"] = np.asarray(obs["state"], dtype=np.float64)
    else:
        out["state"] = np.zeros(8, dtype=np.float64)
        logger.debug("RoboCasa obs has no proprio; using zeros")
    return out


OBS_REMAP_FN = {
    "libero": remap_obs_libero,
    "robocasa": remap_obs_robocasa,
}


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
    obs_remap: str = "libero"  # libero | robocasa
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
        self._remap_fn = OBS_REMAP_FN.get(cfg.obs_remap)
        if self._remap_fn is None:
            raise ValueError(f"Unknown obs_remap: {cfg.obs_remap}. Choose: libero, robocasa")

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
        if "action_dim" in obs and "primary_image" not in obs:
            return {"actions": np.zeros(int(obs["action_dim"]), dtype=np.float64)}

        openvla_obs = self._remap_fn(obs)
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
        description="OpenVLA policy server (WebSocket, obs_remap libero/robocasa)",
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
        "--obs_remap",
        type=str,
        choices=["libero", "robocasa"],
        default="libero",
        help="Observation format from client (LIBERO or RoboCasa run_eval)",
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
        obs_remap=args.obs_remap,
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
                if "action_dim" in obs and "primary_image" not in obs:
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
        "obs_remap": cfg.obs_remap,
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
        f"(obs_remap={cfg.obs_remap}, execute_steps={cfg.execute_steps})"
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
