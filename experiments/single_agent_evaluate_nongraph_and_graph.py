from experiments.evaluate_agent import evaluate as evalNonGraph
from experiments.evaluate_agent_graph import evaluate as evalGraph
import argparse
import os
import sys
import csv
import operator as op


parser = argparse.ArgumentParser(description="Evaluate a trained Pacman agent")
parser.add_argument("experiment_dir", help="Path to the experiment output directory")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
parser.add_argument("--render", action="store_true", help="Render the game")
parser.add_argument("--delay", type=float, default=0.05, help="Delay between frames when rendering")
parser.add_argument("--isGraph", action="store_true", help="Indicates that a graph agent needs to be used")
    
args = parser.parse_args()

eval_result = []
    
if  "config.yaml" in os.listdir(args.experiment_dir) and "q_table.pkl" in os.listdir(args.experiment_dir):
    
    if args.isGraph:
        print(f"evaluating: {args.experiment_dir}")
        win_rate, avg_reward, avg_score =  evalGraph(args.experiment_dir,args.episodes,args.render,args.delay)
        eval_result.append({
            'experiment': args.experiment_dir,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_score': avg_score
        })
    else:
        print(f"evaluating: {args.experiment_dir}")
        win_rate, avg_reward, avg_score =  evalNonGraph(args.experiment_dir,args.episodes,args.render,args.delay)
        eval_result.append({
            'experiment': args.experiment_dir,
            'win_rate': win_rate,
            'avg_reward': avg_reward,
            'avg_score': avg_score
        })

num_of_prev_experiments = len([e for e in os.listdir(args.experiment_dir) if op.contains(e,f"evaluation_{args.episodes}")])

file_name = f"evaluation_{args.episodes}e({num_of_prev_experiments}).csv"
csv_path = os.path.join(args.experiment_dir, file_name)

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'experiment','win_rate','avg_reward', 'avg_score'
    ])
    writer.writeheader()
    writer.writerows(eval_result)
print(f"Evaluation saved to: {csv_path}")

        



