import gymnasium as gym
import mani_skill.envs
import time
from tqdm import tqdm

env = gym.make(
    "PickCube-v1",
    # obs_mode="state", # there is also "state_dict", "rgbd", ...
    obs_mode="rgb+state", # there is also "state_dict", "rgbd", ...
    num_envs=16,
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    robot_uids="panda_wristcam",
    # render_mode="human"
)
print(env.observation_space) # will now have shape (16, ...)
print(env.action_space) # will now have shape (16, ...)
# env.single_observation_space and env.single_action_space provide non batched spaces
device = env.unwrapped.device

# for i in range(100):
start_time = time.time()
step_count = 0
obs, _ = env.reset(seed=0) # reset with a seed for determinism
for i in tqdm(range(1000)):
    action = env.action_space.sample() # this is batched now
    obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated | truncated
    # done = terminated.to(device) | truncated.to(device)
    pass


# obs['sensor_data']['base_camera']['rgb']
# obs['sensor_data']['hand_camera']['rgb']
# print(f"Obs shape: {obs.shape}, Reward shape {reward.shape}, Done shape {done.shape}")
env.close()