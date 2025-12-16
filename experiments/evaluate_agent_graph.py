"""
Evaluation script for trained Q-Learning agents.
Loads a trained Q-table and configuration to run the agent in the environment.
"""
import sys
import os
import argparse
import yaml
import pickle
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.gymenv_graph import GraphEnv
from agents.graph_naive_qlearning_agent import NaiveGraphQLearningAgent
from agents.graphPruned_qlearning_agent import GraphPrunedQLearningAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_reward_config_dict(rewards_yaml):
    return {
        'TIME_PENALTY': rewards_yaml.get('time_penalty', -1),
        'EAT_FOOD': rewards_yaml.get('eat_food', 10),
        'EAT_GHOST': rewards_yaml.get('eat_ghost', 200),
        'WIN': rewards_yaml.get('win', 500),
        'LOSE': rewards_yaml.get('lose', -500),
        'CAPSULE': rewards_yaml.get('capsule', 10)
    }

def evaluate(experiment_dir, num_episodes=10, render=False, delay=0.1):
    # Paths
    config_path = os.path.join(experiment_dir, "config.yaml")
    q_table_path = os.path.join(experiment_dir, "q_table.pkl")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    if not os.path.exists(q_table_path):
        print(f"Error: Q-table file not found at {q_table_path}")
        return
        
    # Load Config
    config = load_config(config_path)
    print(f"Loaded config from {experiment_dir}")
    
    # Setup Environment
    reward_config = create_reward_config_dict(config['environment']['rewards'])
    render_mode = "human" if render else None
    
    env = GraphEnv(
        layoutName=(config['environment']['layout']),
        render_mode=render_mode,
        reward_config=reward_config,
        record=False # Don't record during evaluation by default
    )
    
    # Setup Agent
    if config["graphAgent"]["type"] == "naive":
        agent = NaiveGraphQLearningAgent(
            action_space_size=4,
            learning_rate=0, # No learning during evaluation
            discount_factor=config['agent']['discount_factor'],
            epsilon=0,       # Greedy policy
            epsilon_decay=0,
            epsilon_min=0,
        )
    elif config["graphAgent"]["type"] == "pruned":
        agent = GraphPrunedQLearningAgent(
            action_space_size=4,
            learning_rate=0, # No learning during evaluation
            discount_factor=config['agent']['discount_factor'],
            epsilon=0,       # Greedy policy
            epsilon_decay=0,
            epsilon_min=0,
            hops_prune_limit=4
        )
    else:
        raise Exception("No valid graph agent selected")
    
    # Load Q-table
    print(f"Loading Q-table from {q_table_path}...")
    agent.load(q_table_path)
    print(f"Q-table loaded with {len(agent.q_table)} states.")
    
    # Evaluation Loop
    wins = 0
    total_score = 0
    total_reward = 0
    
    print(f"\nStarting evaluation for {num_episodes} episodes...")
    print("-" * 50)
    
    for i in range(num_episodes):
        obs, info = env.reset()

        done = False
        episode_reward = 0
        steps = 0

        while not done:
            if render:
                time.sleep(delay)

                
            action = agent.get_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            obs = next_obs
            episode_reward += reward
            steps += 1
            
        total_reward += episode_reward
        total_score += info['score']
        
        # if episode_reward > 0: # Simple win check based on positive reward
        #     wins += 1
        if(info['win']):
            win += 1
            
        print(f"Episode {i+1}: Reward = {episode_reward}, Score = {info['score']}, Steps = {steps}")
        
    env.close()
    
    print("-" * 50)
    print(f"Evaluation Complete")
    print(f"Win Rate: {wins/num_episodes:.2f}")
    print(f"Avg Reward: {total_reward/num_episodes:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Pacman agent")
    parser.add_argument("experiment_dir", help="Path to the experiment output directory")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between frames when rendering")
    
    args = parser.parse_args()
    
    evaluate(args.experiment_dir, args.episodes, args.render, args.delay)
