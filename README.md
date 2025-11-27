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
├── environments/                     # Gymnasium wrappers
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

More detailed setup and usage instructions will be added as the project develops.

## Academic Context

This work is part of a 5th semester computer science project exploring the applications of reinforcement learning algorithms.
