
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


        if self.feature_type == "none":
            return self._extract_raw_state(agent_pos, ghosts, food)
        elif self.feature_type == "simple":
            return self._extract_simple_state(agent_pos, ghosts, food)
        elif self.feature_type == "medium":
            return self._extract_medium_state(agent_pos, ghosts, food, observation)
        elif self.feature_type == "rich":
            return self._extract_rich_state(agent_pos, ghosts, food, observation)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
        
    # =================
    # STATE EXTRACTORS:
    # =================



    def _extract_raw_state(self, agent_pos, ghosts, food):
        """
        No abstraction - full raw state representation
        Features: exact positions of everything
        """
        x, y = agent_pos
        
        # Convert food grid to tuple of food positions (makes it hashable)
        food_positions = tuple(
            (i, j) for i in range(self.grid_width) 
            for j in range(self.grid_height) 
            if food[i, j]
        )
        
        # Ghost positions as tuples
        ghost_positions = tuple(tuple(ghost) for ghost in ghosts)
        
        state = (agent_pos, ghost_positions, food_positions)
        return state

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


    def _extract_medium_state(self, agent_pos, ghosts, food, observation):
        """
        Medium complexity
        Features: agent position, 2 closest ghosts, food directions, nearby food count
        """
        x, y = agent_pos

        # Infro about 2 closest ghosts
        ghost_info = self._get_closest_ghost_info(agent_pos, ghosts, n=2)

        # Food, directional indicators
        food_dirs = self._get_food_directions(agent_pos, food)

        # count food within 3 tiles and put into bucket
        nearby_food_count = self._count_nearby_food(agent_pos, food, radius=3)
        raw_bucket_index = nearby_food_count // 5
        capped_food_bucket = min(raw_bucket_index, 4)  # Buckets: 0-4, 5-9, 10-14, 15-19, 20+
        

        state = (x, y, ghost_info, food_dirs, capped_food_bucket)
        return state
        
    def _extract_rich_state(self, agent_pos, ghosts, food, observation):
        """
        "Rich" complexity, max features, largest state space
        Features: agent position, 3 closest ghosts, food directions, nearby food count, capsules
        """

        x, y = agent_pos

        # infro about 3 closest ghosts, more distance buckets
        ghost_info = self._get_closest_ghost_info(agent_pos, ghosts, n=3, bucket_size=2)

        # food, directional indicators
        food_dirs = self._get_food_directions(agent_pos, food)

        # count nearby food with larger radius
        nearby_food_count = self._count_nearby_food(agent_pos, food, radius=5)
        raw_bucket_index = nearby_food_count // 3
        capped_food_bucket = min(raw_bucket_index, 5)

        # Capsule information
        capsule_data = observation['capsules']
        if len(capsule_data) > 0:
            capsules = capsule_data.reshape(-1, 2)
        else:
            capsules = np.array([])
        
        capsule_info = self._get_closest_capsule_info(agent_pos, capsules)
        
        state = (x, y, ghost_info, food_dirs, capped_food_bucket, capsule_info)
        return state
    

    # =================
    # HELPER METHODS
    # =================
    
    def _get_closest_ghost_info(self, agent_pos, ghosts, n=2, distance_per_bucket=3):

        """
        Get informatio about the n closests ghosts
        
        Arguments:
            agent_pos: (x, y) tuple of agent position
            ghosts: Array of ghost positions [[x1, y1], [x2,y2], ..]
            n: Numbner of closest ghost to track
            distance_per_bucket: distance units per bucket, deefault = 3, (0-2, 3-5, 6-8, 9+)
        Returns:
            Tuple of ghost features: ((direction, dist_bucket), (direction, dist_bucket),...)
            which direction the ghost is, relative to pacman, and the distance "bucket"
        """
        if len(ghosts) == 0:          
            # No ghosts, return dummy values
            dummy_ghost = ((0, 3), 3)
            return tuple([dummy_ghost] * n)
        
        # Calculate distance for all ghosts and store positions
        ghost_distances_and_positions = []
        for ghost in ghosts:
            distance = self._manhattan_distance(agent_pos, tuple(ghost))
            ghost_distances_and_positions.append((distance, ghost))

        
        # Sort by distance, closest ghost first
        def extract_distance(distance_position_pair):
            distance = distance_position_pair[0]
            return distance
    
        ghost_distances_and_positions.sort(key=extract_distance)

        # Extract features for the n closest ghosts
        ghost_features = []
        n_closest_ghosts = ghost_distances_and_positions[:n]

        for distance, ghost_position in n_closest_ghosts:
            direction = self._get_direction(agent_pos, tuple(ghost_position))
            bucket_number = distance // distance_per_bucket
            final_bucket = min(bucket_number, 3) # cap at bucket size 
            ghost_features.append((direction, final_bucket))

        # if fewer ghosts than n exists, pad with dummy
        while len(ghost_features) < n:
            dummy_ghost_info = ((0,0), 3) #Dummy, no direction, far away
            ghost_features.append(dummy_ghost_info)

        return tuple(ghost_features)
    

    def _get_direction(self, pos1, pos2):
        """
        Get normalized direction vector from pos1 to pos2
        Calculates the relative direction from pos1 (typically pacman) to pos2 (food, ghost or capsule)
        Pacman at (5,5), ghost at (8,7)
        dx = 8-5= 3, normalized to 1 (east)
        dy = 7-5 = 2, normalized to 1 (north)
        returns 1,1, meaning, target is north east of pacman

        

        """

        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        # Normalize to -1, 0, or 1
        if dx == 0:
            dir_x = 0
        elif dx > 0:
            dir_x = 1
        else:
            dir_x = -1

        if dy == 0:
            dir_y = 0
        elif dy > 0:
            dir_y = 1
        else:
            dir_y = -1

        return(dir_x, dir_y)
    

    def _get_food_directions(self, agent_pos, food, radius=3):
        """
        Check if food exists in all directions + 3 (radius=3)
        args:
            agent_pos
            food: 2d boolean array, true = food exists
            radius: How many steps to look in each direction

        returns:
            tuple of 4 booleans, north, east, south, west
 
        """
        x, y = agent_pos

        # Check directions for food, in radius
        has_food_north = False
        for y_coord in range(y +1, min(y + radius + 1, self.grid_height)):
            if food[x, y_coord]:
                has_food_north = True
                break
        
        has_food_south = False
        for y_coord in range(max(0, y - radius), y):
            if food[x, y_coord]:
                has_food_south = True
                break
        
        has_food_east = False
        for x_coord in range(x + 1, min(x + radius + 1, self.grid_width)):
            if food[x_coord, y]:
                has_food_east = True
                break
                
        has_food_west = False
        for x_coord in range(max(0, x - radius), x):
            if food[x_coord, y]:
                has_food_west = True
                break

        return (has_food_north, has_food_east, has_food_south, has_food_west)



    def _count_nearby_food(self, agent_pos, food, radius=3):
        """
        count food pellets within manhatten distance radius
        args:
            agent_pos
            food: 2d, boolean array
            radius: max manhattan distance
        returns:
            Integer count of nearby food
        """
        x, y = agent_pos
        food_count = 0

        # Check for food within radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                manhattan_dist = abs(dx) + abs(dy)
                
                if manhattan_dist <= radius:
                    neighbor_x = x + dx
                    neighbor_y = y + dy
                    
                    is_within_width = (0 <= neighbor_x < self.grid_width)
                    is_within_height = (0 <= neighbor_y < self.grid_height)
                    
                    if is_within_width and is_within_height:
                        if food[neighbor_x, neighbor_y]:
                            food_count += 1
        
        return food_count


    def _get_closest_capsule_info(self, agent_pos, capsules):
        """
        Get info about the closest power up
        args:
            agent_pos: x, y of pacman
            capsules: array of capsule positions

        return:
            Turple: (Direction, dist_bucket), or 0,0 if no capsules
        """

        if len(capsules) == 0:
            no_capsules_info = ((0, 0), 3)
            return no_capsules_info
        
        # Calculate distances for all capsules
        capsule_distances_and_positions = []
        for capsule in capsules:
            distance = self._manhattan_distance(agent_pos, tuple(capsule))
            capsule_distances_and_positions.append((distance, capsule))
        
        # Get the closest capsule
        def extract_distance(distance_position_pair):
            distance = distance_position_pair[0]
            return distance
        
        closest_capsule = min(capsule_distances_and_positions, key=extract_distance)
        distance_to_closest = closest_capsule[0]
        position_of_closest = closest_capsule[1]
        
        direction = self._get_direction(agent_pos, tuple(position_of_closest))
        raw_bucket = distance_to_closest // 3
        capped_bucket = min(raw_bucket, 3)
        
        return (direction, capped_bucket)
    

    def _manhattan_distance(self, pos1, pos2):
        """
        Manhatten distance, you only move horizontally or vertically, does not account for walls etc.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    