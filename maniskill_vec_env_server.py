"""
ManiSkill Vector Environment Server.

Serves a vectorized ManiSkill environment (multiple parallel envs on GPU) over websockets.
This is designed for high-throughput RL training scenarios.

Note: Vector envs use GPU simulation and auto-reset. When a sub-env terminates,
it automatically resets and the returned obs is the new initial observation.
"""

import logging
import asyncio
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
import torch
import websockets
import websockets.asyncio.server as _server

import mani_skill.envs
from pi_link import msgpack_numpy
from pi_link.spaces import gym_space_to_spec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _now_s() -> float:
    return time.time()


class ManiSkillVecEnvServer:
    """Server for vectorized ManiSkill environment.
    
    Unlike the single-env server which uses worker processes, this server
    runs the vectorized environment directly in the main process since
    ManiSkill's GPU simulation is already parallelized.
    
    Key differences from single-env server:
    - No worker pool (GPU sim handles parallelization)
    - Single session only (one vec env instance)
    - Observations/actions are batched tensors
    - Auto-reset is handled by ManiSkill
    """
    
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        session_idle_timeout_s: float = 60.0,
        env_id: str = "PickCube-v1",
        num_envs: int = 16,
        obs_mode: str = "state",
        control_mode: str = "pd_joint_delta_pos",
        device: str = "cuda",
    ) -> None:
        self._host = host
        self._port = port
        self._session_idle_timeout_s = session_idle_timeout_s
        self._env_id = env_id
        self._num_envs = num_envs
        self._obs_mode = obs_mode
        self._control_mode = control_mode
        self._device_str = device
        
        # Create environment
        self._env = self._create_env()
        self._device = self._env.unwrapped.device
        
        # Session state
        self._session_id: Optional[str] = None
        self._last_used_s: float = 0.0
        self._current_obs: Optional[Any] = None
        self._lock = asyncio.Lock()
        
        # Cache space specs
        self._space_specs = self._infer_space_specs()
        
        logger.info(f"Created ManiSkill vec env: {env_id} with {num_envs} envs on {self._device}")
    
    def _create_env(self):
        """Create vectorized ManiSkill environment."""
        env_kwargs = {
            "obs_mode": self._obs_mode,
            "control_mode": self._control_mode,
            "num_envs": self._num_envs,
            "robot_uids": "panda_wristcam",
        }
        
        env = gym.make(self._env_id, **env_kwargs)
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
        return env
    
    def _infer_space_specs(self) -> dict:
        """Infer observation and action space specs."""
        # Reset to get sample observation
        obs, _ = self._env.reset(seed=0)
        self._current_obs = obs
        
        # Convert spaces to specs
        obs_space = self._env.observation_space
        action_space = self._env.action_space
        
        return {
            "observation_space_spec": gym_space_to_spec(obs_space),
            "action_space_spec": gym_space_to_spec(action_space),
        }
    
    def _tensor_to_numpy(self, x):
        """Convert torch tensor to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, dict):
            return {k: self._tensor_to_numpy(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._tensor_to_numpy(v) for v in x)
        return x
    
    def _numpy_to_tensor(self, x):
        """Convert numpy array to torch tensor on device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self._device)
        elif isinstance(x, dict):
            return {k: self._numpy_to_tensor(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(self._numpy_to_tensor(v) for v in x)
        return x
    
    def _process_observation(self, obs):
        """Convert observation to numpy for transmission."""
        obs = {
            'state': obs['state'],
            'image': obs['sensor_data']['base_camera']['rgb'],
            'wrist_image': obs['sensor_data']['hand_camera']['rgb'],
        }
        return self._tensor_to_numpy(obs)
    
    def _process_action(self, action):
        """Convert action from numpy to tensor."""
        return self._numpy_to_tensor(action)
    
    def _do_reset(self, seed: Optional[int] = None) -> dict:
        """Reset all environments."""
        if seed is not None:
            obs, info = self._env.reset(seed=int(seed))
        else:
            obs, info = self._env.reset()
        
        self._current_obs = obs
        
        return {
            "obs": self._process_observation(obs),
            "info": self._tensor_to_numpy(info) if info else {},
        }
    
    def _do_step(self, action) -> dict:
        """Step all environments."""
        if self._current_obs is None:
            raise RuntimeError("Environment not reset; call reset first.")
        
        # Convert action to tensor
        gym_action = self._process_action(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = self._env.step(gym_action)
        self._current_obs = obs
        
        # Convert tensors to numpy for transmission
        # Note: done envs are auto-reset by ManiSkill, obs is already the new obs
        # Ensure tensors are on the same device before bitwise operation
        if isinstance(terminated, torch.Tensor):
            terminated = terminated.to(self._device)
        if isinstance(truncated, torch.Tensor):
            truncated = truncated.to(self._device)
        done = terminated | truncated
        
        return {
            "obs": self._process_observation(obs),
            "reward": self._tensor_to_numpy(reward),
            "terminated": self._tensor_to_numpy(terminated),
            "truncated": self._tensor_to_numpy(truncated),
            "done": self._tensor_to_numpy(done),
            "info": self._tensor_to_numpy(info) if info else {},
        }
    
    def get_metadata(self) -> dict:
        """Return server metadata."""
        return {
            "kind": "maniskill_vec_env_server",
            "env_id": self._env_id,
            "num_envs": self._num_envs,
            "obs_mode": self._obs_mode,
            "control_mode": self._control_mode,
            "device": str(self._device),
            "idle_timeout_s": self._session_idle_timeout_s,
        }
    
    def _err(
        self,
        *,
        code: str,
        message: str,
        details: Optional[dict] = None,
        request_id: Any = None,
    ) -> dict:
        """Create error response."""
        out: dict = {"error": {"code": code, "message": message}}
        if details:
            out["error"]["details"] = details
        if request_id is not None:
            out["request_id"] = request_id
        return out
    
    async def _reap_idle_session(self) -> None:
        """Background task to auto-release idle session."""
        if self._session_idle_timeout_s <= 0:
            return
        
        while True:
            await asyncio.sleep(1.0)
            
            if self._session_id is None:
                continue
            
            if _now_s() - self._last_used_s >= self._session_idle_timeout_s:
                if not self._lock.locked():
                    logger.info(f"Auto-releasing idle session: {self._session_id}")
                    self._session_id = None
    
    async def _handler(self, ws: _server.ServerConnection) -> None:
        """Handle websocket connection."""
        packer = msgpack_numpy.Packer()
        
        # Send handshake metadata
        metadata = self.get_metadata()
        metadata.update({
            "observation_space_spec": self._space_specs.get("observation_space_spec"),
            "action_space_spec": self._space_specs.get("action_space_spec"),
        })
        metadata["session_protocol"] = {
            "enabled": True,
            "max_sessions": 1,
            "idle_timeout_s": self._session_idle_timeout_s,
        }
        
        await ws.send(packer.pack(metadata))
        
        # Main message loop
        while True:
            try:
                req_raw = await ws.recv()
                req = msgpack_numpy.unpackb(req_raw)
                
                if not isinstance(req, dict):
                    await ws.send(packer.pack(self._err(
                        code="bad_request",
                        message=f"Expected dict, got {type(req)}",
                    )))
                    continue
                
                cmd = req.get("cmd")
                request_id = req.get("request_id")
                session_id = req.get("session_id")
                
                # Handle close
                if cmd == "close":
                    await ws.send(packer.pack({"ok": True, "request_id": request_id}))
                    await ws.close(
                        code=websockets.frames.CloseCode.NORMAL_CLOSURE,
                        reason="Closed by client.",
                    )
                    return
                
                # Handle close_session
                if cmd == "close_session":
                    if session_id and str(session_id) == self._session_id:
                        self._session_id = None
                        await ws.send(packer.pack({
                            "ok": True,
                            "session_id": str(session_id),
                            "request_id": request_id,
                        }))
                    else:
                        await ws.send(packer.pack(self._err(
                            code="invalid_session",
                            message="Invalid session_id",
                            details={"session_id": session_id},
                            request_id=request_id,
                        )))
                    continue
                
                # Handle ping
                if cmd == "ping":
                    if session_id and str(session_id) == self._session_id:
                        self._last_used_s = _now_s()
                        await ws.send(packer.pack({
                            "ok": True,
                            "cmd": "pong",
                            "request_id": request_id,
                        }))
                    else:
                        await ws.send(packer.pack(self._err(
                            code="invalid_session",
                            message="ping requires valid session_id",
                            details={"session_id": session_id},
                            request_id=request_id,
                        )))
                    continue
                
                # Handle reset
                if cmd == "reset":
                    async with self._lock:
                        new_session = bool(req.get("new_session", False))
                        
                        # Allocate session if needed
                        if new_session or self._session_id is None:
                            if self._session_id is not None and not new_session:
                                # Session exists, reuse it
                                pass
                            else:
                                # Create new session
                                if self._session_id is not None:
                                    logger.info(f"Replacing session {self._session_id}")
                                self._session_id = uuid.uuid4().hex
                        
                        sid = self._session_id
                        self._last_used_s = _now_s()
                        
                        try:
                            resp = self._do_reset(req.get("seed"))
                            await ws.send(packer.pack({
                                "session_id": sid,
                                "obs": resp["obs"],
                                "info": resp.get("info", {}),
                                "request_id": request_id,
                            }))
                        except Exception as e:
                            logger.exception("Reset failed")
                            await ws.send(packer.pack(self._err(
                                code="reset_failed",
                                message=str(e),
                                request_id=request_id,
                            )))
                    continue
                
                # Handle step
                if cmd == "step":
                    if not session_id or str(session_id) != self._session_id:
                        await ws.send(packer.pack(self._err(
                            code="invalid_session",
                            message="step requires valid session_id",
                            details={"session_id": session_id},
                            request_id=request_id,
                        )))
                        continue
                    
                    async with self._lock:
                        self._last_used_s = _now_s()
                        
                        try:
                            resp = self._do_step(req.get("action"))
                            await ws.send(packer.pack({
                                "session_id": self._session_id,
                                "obs": resp["obs"],
                                "reward": resp["reward"],
                                "terminated": resp["terminated"],
                                "truncated": resp["truncated"],
                                "done": resp["done"],
                                "info": resp.get("info", {}),
                                "request_id": request_id,
                            }))
                        except Exception as e:
                            logger.exception("Step failed")
                            await ws.send(packer.pack(self._err(
                                code="step_failed",
                                message=str(e),
                                request_id=request_id,
                            )))
                    continue
                
                # Unknown command
                await ws.send(packer.pack(self._err(
                    code="unknown_cmd",
                    message=f"Unknown cmd: {cmd}",
                    request_id=request_id,
                )))
            
            except websockets.ConnectionClosed:
                return
            except Exception as e:
                logger.exception("Unexpected error in handler")
                try:
                    await ws.send(packer.pack(self._err(
                        code="server_error",
                        message=str(e),
                        details={"type": type(e).__name__},
                    )))
                except Exception:
                    pass
                return
    
    async def run(self) -> None:
        """Async server main loop."""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logger.info(f"ManiSkill vec env server listening on ws://{self._host}:{self._port}")
            logger.info(f"  Environment: {self._env_id}")
            logger.info(f"  Num envs: {self._num_envs}")
            logger.info(f"  Device: {self._device}")
            
            reaper_task = asyncio.create_task(self._reap_idle_session())
            try:
                await server.serve_forever()
            finally:
                reaper_task.cancel()
    
    def serve_forever(self) -> None:
        """Start server and block forever."""
        asyncio.run(self.run())
    
    def close(self):
        """Close the environment."""
        if self._env is not None:
            self._env.close()
            self._env = None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ManiSkill Vector Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9100, help="Port to bind")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment ID")
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel environments")
    parser.add_argument("--obs-mode", default="rgb+state", help="Observation mode")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose", help="Control mode")
    parser.add_argument("--idle-timeout", type=float, default=60.0, help="Session idle timeout (seconds)")
    
    args = parser.parse_args()
    
    server = ManiSkillVecEnvServer(
        host=args.host,
        port=args.port,
        session_idle_timeout_s=args.idle_timeout,
        env_id=args.env_id,
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
    )
    
    try:
        server.serve_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
