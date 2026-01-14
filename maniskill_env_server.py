"""
ManiSkill Single Environment Server.

Serves a single ManiSkill gym environment over websockets using the RemoteEnv protocol.
"""

import logging
import gymnasium as gym
import mani_skill.envs
from pi_link.gym_env_server import GymEnvServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManiSkillEnvServer(GymEnvServer):
    """Server for single ManiSkill environment."""
    
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_sessions: int = 1,
        session_idle_timeout_s: float = 30.0,
        env_id: str = "PickCube-v1",
        obs_mode: str = "state",
        control_mode: str = "pd_ee_delta_pose",
        render_mode: str = None,
        **kwargs,
    ) -> None:
        self._env_id = env_id
        self._obs_mode = obs_mode
        self._control_mode = control_mode
        self._render_mode = render_mode
        
        super().__init__(
            host=host,
            port=port,
            max_sessions=max_sessions,
            session_idle_timeout_s=session_idle_timeout_s,
            **kwargs,
        )
    
    def create_env(self, **kwargs):
        """Create a ManiSkill environment."""
        env_kwargs = {
            "obs_mode": self._obs_mode,
            "control_mode": self._control_mode,
        }
        if self._render_mode:
            env_kwargs["render_mode"] = self._render_mode
        
        env = gym.make(self._env_id, **env_kwargs)
        logger.info(f"Created ManiSkill env: {self._env_id}")
        logger.info(f"  Observation space: {env.observation_space}")
        logger.info(f"  Action space: {env.action_space}")
        return env
    
    def process_observation(self, obs):
        """Process observation for client.
        
        ManiSkill state observations are already numpy arrays, pass through.
        """
        return obs
    
    def process_action(self, action):
        """Process action from client.
        
        Actions should be numpy arrays matching action_space.
        """
        return action
    
    def get_metadata(self) -> dict:
        """Return server metadata."""
        base = super().get_metadata()
        base.update({
            "env_id": self._env_id,
            "obs_mode": self._obs_mode,
            "control_mode": self._control_mode,
        })
        return base
    
    def _get_picklable_init_kwargs(self):
        """Get picklable init kwargs for worker process."""
        base = super()._get_picklable_init_kwargs()
        base.update({
            "env_id": self._env_id,
            "obs_mode": self._obs_mode,
            "control_mode": self._control_mode,
            "render_mode": self._render_mode,
        })
        return base


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ManiSkill Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--max-sessions", type=int, default=1, help="Max concurrent sessions")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment ID")
    parser.add_argument("--obs-mode", default="state", help="Observation mode")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose", help="Control mode")
    parser.add_argument("--idle-timeout", type=float, default=30.0, help="Session idle timeout (seconds)")
    
    args = parser.parse_args()
    
    server = ManiSkillEnvServer(
        host=args.host,
        port=args.port,
        max_sessions=args.max_sessions,
        session_idle_timeout_s=args.idle_timeout,
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
    )
    
    logger.info(f"Starting ManiSkill env server...")
    logger.info(f"  Environment: {args.env_id}")
    logger.info(f"  Obs mode: {args.obs_mode}")
    logger.info(f"  Control mode: {args.control_mode}")
    
    server.serve_forever()


if __name__ == "__main__":
    main()
