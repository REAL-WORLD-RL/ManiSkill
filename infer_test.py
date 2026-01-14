import gymnasium as gym
import mani_skill.envs
import time

env = gym.make(
    "PickCube-v1", # there are more tasks e.g. "PushCube-v1", "PegInsertionSide-v1", ...
    obs_mode="state", # there is also "state_dict", "rgbd", ...
    # obs_mode="rgb+state", # there is also "state_dict", "rgbd", ...
    control_mode="pd_ee_delta_pose", # there is also "pd_joint_delta_pos", ...
    robot_uids="panda_wristcam",
    # render_mode="human"
)
print("Observation space", env.observation_space)
print("Action space", env.action_space)

for i in range(100):
    start_time = time.time()
    obs, _ = env.reset(seed=0) # reset with a seed for determinism
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # print(obs)
        # print(action)
        # print('---'*100)
        # env.render()  # a display is required to render
        step_count += 1
    print(f'Episode {i} completed, timecost: {time.time() - start_time}, step_count: {step_count}')
env.close()