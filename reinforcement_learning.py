import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = gym.make("BipedalWalker-v3", render_mode="human")
# env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
# env = VecFrameStack(env, n_stack=4)

# Learn yourself or...
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

# model = A2C("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# load from checkpoint
# checkpoint = load_from_hub(
# 	repo_id="sb3/ppo-BipedalWalker-v3",
# 	filename="{MODEL FILENAME}.zip",
# )
# model = PPO.load(checkpoint)

print("Evaluating")
env = model.get_env()
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render("human")

env.close()