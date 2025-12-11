"""
Experiment runner for Q-Learning on Pacman with YAML configuration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import csv
from datetime import datetime

from environment.gymenv_graph import GraphEnv
from agents.graph_naive_qlearning_agent import NaiveGraphQLearningAgent
from agents.graphPruned_qlearning_agent import GraphPrunedQLearningAgent     
#from environment.state_abstraction import StateAbstraction #comment for now

from torch.utils.tensorboard import SummaryWriter


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_reward_config_dict(rewards_yaml):
    """
    Convert YAML rewards section to format expected by GymEnv
    
    Arguments:
        rewards_yaml: Dict from YAML config['environment']['rewards']
    
    Returns:
        Dict compatible with game rules reward structure
    """
    return {
        'TIME_PENALTY': rewards_yaml.get('time_penalty', -1),
        'EAT_FOOD': rewards_yaml.get('eat_food', 10),
        'EAT_GHOST': rewards_yaml.get('eat_ghost', 200),
        'WIN': rewards_yaml.get('win', 500),
        'LOSE': rewards_yaml.get('lose', -500),
        'CAPSULE': rewards_yaml.get('capsule', 10)
    }


def run_experiment(config_path):
    """
    Run experiment based on config file
    """
    

    config = load_config(config_path)

    print("#" * 69)
    print("EXPERIMENTS, YESSS")
    print("#" * 69)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Layout: {config['environment']['layout']}")
    print(f"Episodes: {config['training']['num_episodes']}")
    #print(f"State Abstraction: {config['state_abstraction']['feature_type']}")
    print("#" * 69)

    
    writer = SummaryWriter(log_dir=config['output']['base_dir'] + "/runs")


    # Setup environment with reward config
    reward_config = create_reward_config_dict(config['environment']['rewards'])
    env = GraphEnv(
        layoutName=(config['environment']['layout']),
        render_mode=(config['environment']['render_mode']), # Should make this paramater controllable via terminal
        reward_config=reward_config,
        #record=config['output'].get('record_games', False), #We dont have recording in graph enviroment yet, so we comment this until we do
        #record_interval=config['output'].get('record_interval', 10)
    )

    #Set up agent (Unchanged from experiment_runner expect NaiveGraohQLearningAgent)
    if config["graphAgent"]["type"] == "naive":
        print("Going graph/naive")
        agent = NaiveGraphQLearningAgent(
            action_space_size=5,  # E, N, W, S (STOP removed)
            learning_rate=config['agent']['learning_rate'],
            discount_factor=config['agent']['discount_factor'],
            epsilon=config['agent']['epsilon'],
            epsilon_decay=config['agent']['epsilon_decay'],
            epsilon_min=config['agent']['epsilon_min']
        )
    elif config["graphAgent"]["type"] == "pruned":
        print("Going graph/pruned")
        agent = GraphPrunedQLearningAgent(
            action_space_size=5,  # E, N, W, S (STOP removed)
            learning_rate=config['agent']['learning_rate'],
            discount_factor=config['agent']['discount_factor'],
            epsilon=config['agent']['epsilon'],
            epsilon_decay=config['agent']['epsilon_decay'],
            epsilon_min=config['agent']['epsilon_min'],
            hops_prune_limit=4
        )


     # Training setup
    metrics = []
    print_interval = config['output']['print_interval']
    
    print("\nStarting training...")
    print("-" * 70)
    
    # Training loop
    for episode in range(config['training']['num_episodes']):
        obs, info = env.reset()
        #####DEBUG REGION START#####
        import numpy as np
        print("env._direction_to_action:", env._direction_to_action)
        print("env._inv_direction_to_action:", env._inv_direction_to_action)

        # Agent belief from graph observation
        pac_legal, pac_idx = agent.extractPacState(obs)
        print("pacNodeIndex:", pac_idx, "agent_legal_from_graph (ints):", pac_legal, 
              "agent_legal_named:", [env._inv_direction_to_action[a] for a in pac_legal])

        # Show outgoing edges for pac node: (edge_idx, action_int, action_name, (u,v), dest_coord, wall_at_dest)
        pac_edges = []
        for ei, (edge, feat) in enumerate(zip(obs["edges"], obs["edge_features"])):
            if int(edge[0]) == int(pac_idx):
                action_int = int(feat[0])
                action_name = env._inv_direction_to_action.get(action_int, str(action_int))
                dest = int(edge[1])
                dest_coord = tuple(obs["nodesXY"][dest])
                # Check layout/wall if available
                wall_at_dest = None
                try:
                    wall_at_dest = bool(env.layout.walls[dest_coord[0]][dest_coord[1]])
                except Exception:
                    wall_at_dest = "unknown"
                pac_edges.append((ei, action_int, action_name, (int(edge[0]), dest), dest_coord, wall_at_dest))

        print("pac_edges:", pac_edges)

        # Print environment's legal moves from game state
        env_legal_dirs = env.game.state.getLegalPacmanActions()
        env_legal = [env._direction_to_action[d] for d in env_legal_dirs]
        print("env_legal (ints):", env_legal, "env_legal_named:", env_legal_dirs)

        # Also print pac node coords and node features for context
        print("pac node coords:", tuple(obs["nodesXY"][pac_idx]), "pac node features:", tuple(obs["nodes"][pac_idx]))

        print("shares_memory edges/env.edges:", np.shares_memory(obs["edges"], env.edges))
        print("shares_memory edge_features/env.edge_features:", np.shares_memory(obs["edge_features"], env.edge_features))
        print("ids: obs_edges", id(obs["edges"]), "env.edges", id(env.edges))
        ######DEBUG REGION END######



        state = obs #define state here as well
        
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Agent selects action
            action = agent.get_action(state, training=True)

            # DEBUG: compare agent belief vs environment legal moves
            env_legal = [env._direction_to_action[a] for a in env.game.state.getLegalPacmanActions()]
            if action not in env_legal:
                print("ACTION MISMATCH: ", action, " agent_legal_from_graph: ", agent.extractPacState(state)[0], " env_legal: ", env_legal)

            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)  
            done = terminated or truncated
            
            next_state = next_obs
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)

            #Check again, this should work now our state is the entire graph now (which is given by obs)
            state = next_state

            episode_reward += reward
            episode_steps += 1
        
        # Decay epsilon after episode
        agent.decay_epsilon()
        
        # Track metrics
        win = 1 if episode_reward > 0 else 0
        countableWin = 1 if info["winState"] == True else 0
        avg_q = agent.get_average_q_value()
        q_table_size = len(agent.q_table)
        
        metrics.append({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': episode_steps,
            'win': win,
            'countableWin': countableWin,
            'epsilon': agent.epsilon,
            'avg_q_value': avg_q,
            'q_table_size': q_table_size
        })

        
        writer.add_scalar('steps',episode_steps,episode+1)
        writer.add_scalar('epsilon',agent.epsilon,episode+1)
        writer.add_scalar('avg_q_value',avg_q,episode+1)
        writer.add_scalar('q_table_size',q_table_size,episode+1)
        
        # Print progress
        if (episode + 1) % print_interval == 0:
            recent_metrics = metrics[-print_interval:]
            avg_reward = sum(m['reward'] for m in recent_metrics) / len(recent_metrics)
            win_rate = sum(m['win'] for m in recent_metrics) / len(recent_metrics)
            countable_win_rate = sum(m['countableWin'] for m in recent_metrics) / len(recent_metrics)
            
            print(f"Episode {episode + 1:4d}/{config['training']['num_episodes']} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Win Rate: {win_rate:.2f} | "
                  f"Countable Win Rate: {countable_win_rate:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Q-table: {q_table_size:6d} states | "
                  f"Avg Q: {avg_q:6.2f}")
            
            writer.add_scalar('avg_reward',avg_reward, (episode + 1) // print_interval)
            writer.add_scalar('win_rate',win_rate, (episode + 1) // print_interval)
    
    env.close()


     # Save results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config['experiment_name']
    output_base = config['output'].get('base_dir', 'data/experiments')
    output_dir = os.path.join(output_base,f"{timestamp}_{experiment_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to CSV
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'reward', 'steps', 'win', 'countableWin', 'epsilon', 'avg_q_value', 'q_table_size'
        ])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Metrics saved to: {csv_path}")
    
    # Save Q-table
    if config['output']['save_q_table']:
        q_table_path = os.path.join(output_dir, "q_table.pkl")
        agent.save(q_table_path)
        print(f"Q-table saved to: {q_table_path}")
    
    # Save configuration
    config_save_path = os.path.join(output_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")
    
    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"  Q-table size: {len(agent.q_table)} states")
    print(f"  Final epsilon: {agent.epsilon:.4f}")
    
    last_100 = metrics[-100:] if len(metrics) >= 100 else metrics
    avg_reward_100 = sum(m['reward'] for m in last_100) / len(last_100)
    win_rate_100 = sum(m['win'] for m in last_100) / len(last_100)
    print(f"  Last {len(last_100)} episodes avg reward: {avg_reward_100:.1f}")
    print(f"  Last {len(last_100)} episodes win rate: {win_rate_100:.2f}")
    print("=" * 70)
    
    return output_dir


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python experiment_runner.py <config.yaml>")
        print("\nExample: python experiment_runner.py configurations/qlearning_simple_smallGrid.yaml")
        sys.exit(1)
    
    config_file = sys.argv[1]
    if not os.path.exists("experiments/"+ config_file):
        print(f"Error: Configuration file '{config_file}' not found!")
        sys.exit(1)
    
    output_dir = run_experiment("experiments/"+ config_file)
    print(f"\nResults saved to: {output_dir}")
