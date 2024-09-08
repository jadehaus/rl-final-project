import gymnasium as gym

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO, DQN

# Customize your training settings here
ENV_NAME = ???
TRAINED_TOTAL_TIMESTEPS = ???
ALGORITHM = ???
POLICY = ???

# Environment name mapping
# Add your own version mapping for convenience
ATARI_ENVS = [
    'Boxing',
    'Breakout',
]
ENV_VERSION_MAPPING = {
    'BipedalWalker': '-v3',
    'Boxing': 'NoFrameskip-v4',
    'Breakout': 'NoFrameskip-v4',
}

version = ENV_VERSION_MAPPING[ENV_NAME]

if ENV_NAME in ATARI_ENVS:
    env = make_atari_env(f"{ENV_NAME}{version}", n_envs=1)
    env = VecFrameStack(env, n_stack=1)
else:
    env = gym.make(f"{ENV_NAME}{version}", render_mode='human')

try:
    timesteps_str = f'{TRAINED_TOTAL_TIMESTEPS // 1000}k'
    file_path = f'./trained_weights/{ENV_NAME}_{ALGORITHM.__name__}_{POLICY}_{timesteps_str}'
    model = ALGORITHM.load(file_path, env, verbose=1)
    print(f"Loaded trained models from checkpoint {file_path}.zip.")
except FileNotFoundError:
    print(f"Trained model not found. Training from scratch..")
    model = ALGORITHM(POLICY, env, verbose=1)
    model.learn(total_timesteps=TRAINED_TOTAL_TIMESTEPS)
    model.save(file_path)

if ENV_NAME not in ATARI_ENVS:
    env = model.get_env()

obs = env.reset()
total_rewards = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, dones, info = env.step(action)
    env.render("human")
    total_rewards += rewards
    
print(f"Total Return: {total_rewards}")