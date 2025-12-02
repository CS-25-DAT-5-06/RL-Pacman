# Reinforcement Learning in Pac-Man

A systematic exploration of reinforcement learning methods applied to the Berkeley Pac-Man framework


## Project Structure
```

pacman-qlearning/
│
├── README.md                          # Project overview
├── requirements.txt                   # Python dependencies
│
├── berkeley_pacman/                   # Core Pacman game files
│   ├── layouts/
│   ├── game.py
│   ├── ghostAgents.py
│   ├── layout.py
│   ├── pacman.py
│   ├── graphicsDisplay.py
│   ├── textDisplay.py
│   └── ...
│
├── environment/                      # Gymnasium wrappers and environment abstractions
│   ├── __init__.py
│   ├── gymenv_v2.py                  
│   └── state_extractor.py            # State extraction module
│
├── agents/                            # RL agents
│   ├── __init__.py
│   └── qlearning_agent.py         
│
├── experiments/                      # Training scripts and configs
│   ├── __init__.py
│   ├── configs/                      # .ini configuration files
│   ├── train_qlearning.py            # Main training scripts
│   ├── evaluate.py                   # Evaluation scripts
│   └── ...
│
├── data/                          # Training outputs (gitignored)
│   ├── logs/                         # Training logs
│   ├── models/                       # Saved Q-tables
│   ├── plots/                        # Visualization plots
│   ├── metrics/                      # CSV/JSON metrics
|   └── recordings/                   # Stores folders with session
│
└── docs/                             # Documentation
    ├── NOTES.md                      # Running log of group observations
    └── USER_GUIDE.md                # How to train agents
    
 ```   


## Project Overview

This project demonstrates the progression from classical tabular reinforcement learning to modern deep RL approaches. We use the Berkeley AI Pac-Man environment as our testbed.

The core narrative examines how state space complexity drives the need for optimizing, illustrating fundamental concepts in reinforcement learning through concrete experimental results.

## Architecture

The project follows a layered architecture that separates concerns and enables modular experimentation:

- **Berkeley Pac-Man Framework**: Core game mechanics and physics
- **Gymnasium Wrapper**: Standardized RL interface following OpenAI Gym conventions
- **RL Agents**: Tabular Q-learning implementation
- **Experimental Framework**: Training scripts and evaluation tools

## Current Status

In development

## Requirements

- Python 3.8+
- NumPy
- Gymnasium
- Stable Baselines3 (for future deep RL experiments)

## Getting Started

## Running Experiments

To run an experiment, use the `experiment_runner.py` script with a configuration file:

```bash
python experiments/experiment_runner.py experiments/configurations/your_config.yaml
```

### Configuration
Configuration files are located in `experiments/configurations/`. You can control various aspects of the experiment, including:
- **Environment**: Layout, rewards
- **Agent**: Learning rate, epsilon, discount factor
- **State Abstraction**: Feature type (`simple`, `medium`, `rich`, `relative`, `relative_radius`, `relative_grid`, `relative_crisis`, `relative_crisis_bfs`)
- **Output**: Logging, saving models, and **recording games**


To enable game recording, add the following to your config:
```yaml
output:
  record_games: true
  record_interval: 10 # Record every 10th game
```

## Replaying Games

If you have enabled recording in your experiment, you can replay the games using the `replay.py` tool.
Recordings are saved in `data/recordings/session-TIMESTAMP/`.

To replay all games in a session:
```bash
python tools/replay.py session-TIMESTAMP -a
```

To replay a specific game (e.g., game 5):
```bash
python tools/replay.py session-TIMESTAMP -g 5
```

To replay the first and last game:
```bash
python tools/replay.py session-TIMESTAMP -fl
```


## Evaluating Agents

To watch a trained agent play without training:

```bash
python experiments/evaluate_agent.py data/experiments/YOUR_EXPERIMENT_FOLDER --render
```

- `YOUR_EXPERIMENT_FOLDER`: The directory containing `config.yaml` and `q_table.pkl`.
- `--render`: Enable graphics.
- `--episodes N`: Number of episodes to run (default: 10).
- `--delay X`: Delay between frames in seconds (default: 0.05).



## Academic Context

This work is part of a 5th semester computer science project exploring the applications of reinforcement learning algorithms.
