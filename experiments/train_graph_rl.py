import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure

from environment.Gymenv_graph import GraphGymEnv

#Registering the environment for SB3, so it can create it
gym.register(id="PacmanGraph-v0", entry_point=GraphGymEnv, max_episode_steps=300, kwargs={"layoutName": "originalClassic"})

def main():
    env = gym.make("PacmanGraph-v0")

    #log_dir = "data/logs_graph"
    #logger = configure(log_dir, ["stdout", "tensorboard"])

    #A2C model
    model = A2C(policy="MlpPolicy", env=env, verbose=1,
    #tensorboard_log=log_dir
    )

    model.learn(total_timesteps=10_000)

    model.save("data/models/graph_a2c")

    env.close()

if __name__ == "__main__":
    main()
