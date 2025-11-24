
import numpy as np

"""
STATE ABSTRACTION ARCHITECTURE
================================

Purpose:
--------
Reduces the Pacman state space into discrete, hashable states.

Flow:
----------
    Pacman Game
         ↓
    GymEnv (gymenv_v2.py)
         ↓ [Rich observation: full game state]
         ↓ {agent: [x,y], food: [...], ghosts: [...], capsules: [...]}
         ↓
    StateAbstraction
         ↓ [Converts to simple hashable tuple]
         ↓ (x, y, ghost_direction, ghost_dist_bucket, food_directions, ...)
         ↓
    Q-Learning Agent
         ↓ [Uses tuple as key in Q-table]
         Q-table[(x, y, ghost_dir, ...)] → [Q-values for actions]

When It's Called:
-----------------
- Called EVERY STEP during training:
  1. After env.reset() → extract initial state
  2. After env.step(action) → extract next state
  
- The agent never  sees raw observations, only abstracted states
- Different abstraction levels (simple/medium/rich) create different-sized state spaces

Abstraction Levels, TBD/WIP, pending results:
-------------------
- simple: Minimal features (position + nearest ghost + food directions)
          → Smallest state space, may lose important information
          
- medium: Balanced features (position + 2 nearest ghosts + food info)
          → Medium state space, balanced trade-off
          
- rich:   Maximum features (detailed ghost info + food counts)
          → Larger state space, more information preserved
"""

class StateAbstraction:
    """
    Converts rich Gymnasium observation into reduced observed state space
    """

    def __init__(self, grid_width, grid_height, feature_type="simple"):
        """
        Arguments:
            state: Width of the grid
            action: Height of the grid
            feature_type:
                "simple": Minimal abstractions
                "medium":
                "rich":      
        """
        self.grid_width = grid_width #env.layout.width
        self.grid_height = grid_height #env.layout.height
        self.feature_type = feature_type
        
    def extract_state(self, observation):
        """
        Extracts a discrete hashable state from the Gymnasium
        Arguments:
            observation: We pass the entire observable dict from the gymnasiaum, 
            returned by gymenv
        Returns:
            state: Hashable tuple representing the state.
        """
        agent_pos = tuple(observation['agent'])
        ghosts = observation['ghosts'].reshape(-1,2) # Converts the flat array from gym dict to pairs: [x1,y1,x2,y2,...] to [[x1,y1], [x2,y2],...]
        food = observation['food'].reshape(self.grid_width, self.grid_height)

        if self.feature_type == "simple":
            return self._extract_simple_state(agent_pos, ghosts, food)
        elif self.feature_type == "medium":
            return self._extract_medium_state(agent_pos, ghosts, food, observation)
        elif self.feature_type == "rich":
            return self._extract_rich_state(agent_pos, ghosts, food, observation)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
    
    def _extract_simple_state(self, agent_pos, ghosts, food):
        """
        Simplest state representation - minimal features for small state space
        Features: agent position, nearest ghost info, food directions
        """
        x, y = agent_pos
        
        # Get info about 1 closest ghost
        ghost_info = self._get_closest_ghost_info(agent_pos, ghosts, n=1)
        
        # Check food in cardinal directions
        food_dirs = self._get_food_directions(agent_pos, food)
        
        state = (x, y, ghost_info, food_dirs)
        return state



    """
    Helper methods:
    """

    def _get_closest_ghost_info(self, agent_pos, ghosts, n=2, bucket_size=3):

        """
        Get informatio about the n closests ghosts
        
        Arguments:
            agent_pos: (x, y) tuple of agent position
            ghosts: Array of ghost positions [[x1, y1], [x2,y2], ..]
            n: Numbner of closest ghost to track
            bucket_size: Size of distance bukets, default 3:
        """
        if len(ghosts) == 0:
            # No ghosts, return dummy values
            return tuple([((0, 0), 3)] * n )



    def _manhattan_distance(self, pos1, pos2):

        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    