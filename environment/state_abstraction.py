import numpy as np
from collections import deque

"""
STATE ABSTRACTION ARCHITECTURE
================================

Purpose:
--------
Reduces the Pacman state space into discrete, hashable states.

Flow:
----------
    Pacman Game
         ->
    GymEnv (gymenv_v2.py)
         -> [Rich observation: full game state]
         -> {agent: [x,y], food: [...], ghosts: [...], capsules: [...]}
         ->
    StateAbstraction
         -> [Converts to simple hashable tuple]
         -> (x, y, ghost_direction, ghost_dist_bucket, food_directions, ...)
         ->
    Q-Learning Agent
         -> [Uses tuple as key in Q-table]
         -> Q-table[(x, y, ghost_dir, ...)] → [Q-values for actions]

When It's Called:
-----------------
- Called EVERY STEP during training:
  1. After env.reset() → extract initial state
  2. After env.step(action) → extract next state
  
- The agent never  sees raw observations, only abstracted states
- Different abstraction levels (simple/medium/rich) create different-sized state spaces

Abstraction Levels, TBD/WIP, pending results:
-------------------
Explanation of abstractions:


"""

class StateAbstraction:
    """
    Converts rich Gymnasium observation into reduced observed state space
    """

    def __init__(self, grid_width, grid_height, walls=None, feature_type="simple"):
        """
        Arguments:
            grid_width: Width of the grid
            grid_height: Height of the grid
            walls: Grid of walls (Grid object or 2D array)
            feature_type:
                "simple": Minimal abstractions
                "medium":
                "rich":      
        """
        self.grid_width = grid_width #env.layout.width
        self.grid_height = grid_height #env.layout.height
        self.walls = walls
        self.feature_type = feature_type
        
    def extract_state(self, observation, last_action=None):
        """
        Extracts a discrete hashable state from the Gymnasium
        Arguments:
            observation: We pass the entire observable dict from the gymnasiaum, 
            returned by gymenv
            last_action: The action taken in the previous step (int) or None
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
        elif self.feature_type == "relative":
            return self._extract_relative_state(agent_pos, ghosts, food, observation)
        elif self.feature_type == "relative_radius":
            return self._extract_relative_radius_state(agent_pos, ghosts, food, observation)
        elif self.feature_type == "relative_grid":
            return self._extract_relative_grid_state(agent_pos, ghosts, food, observation)
        elif self.feature_type == "relative_crisis":
            return self._extract_relative_crisis_state(agent_pos, ghosts, food, observation, last_action)
        elif self.feature_type == "relative_crisis_bfs":
            return self._extract_relative_crisis_bfs_state(agent_pos, ghosts, food, observation, last_action)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def _extract_relative_crisis_state(self, agent_pos, ghosts, food, observation, last_action):
        """
        Crisis vs. Greed vs. Hunt complexity
        Mode 0 (Greed): Safe (Ghost > 4). State = (0, FoodDir, LegalMoves, LastAction)
        Mode 1 (Crisis): Threatened (Ghost <= 4) & Not Scared. State = (1, GhostDir, LegalMoves, LastAction)
        Mode 2 (Hunt): Threatened (Ghost <= 4) & Scared. State = (2, GhostDir, LegalMoves, LastAction)
        """
        # 1. Determine Mode and Target Direction
        closest_ghost_dist = float('inf')
        closest_ghost_pos = None
        closest_ghost_idx = -1
        
        for i, ghost in enumerate(ghosts):
            dist = self._manhattan_distance(agent_pos, tuple(ghost))
            if dist < closest_ghost_dist:
                closest_ghost_dist = dist
                closest_ghost_pos = ghost
                closest_ghost_idx = i
                
        if closest_ghost_dist <= 4:
            # Check if scared
            scared_timers = observation.get('ghost_scared_timers', [])
            is_scared = False
            if len(scared_timers) > closest_ghost_idx:
                # Only hunt if timer is sufficient to reach (dist < timer)
                if scared_timers[closest_ghost_idx] > closest_ghost_dist:
                    is_scared = True

            if is_scared:
                # HUNT MODE
                mode = 2
                # Target: Ghost
                target_dir = self._get_direction(agent_pos, tuple(closest_ghost_pos))
            else:
                # CRISIS MODE
                mode = 1
                # Target: Ghost (to run away from)
                target_dir = self._get_direction(agent_pos, tuple(closest_ghost_pos))
        else:
            # GREED MODE
            mode = 0
            # Get direction to closest food
            food_list = np.argwhere(food)
            if len(food_list) > 0:
                # Find closest food
                closest_food_dist = float('inf')
                closest_food_pos = None
                for f in food_list:
                    dist = self._manhattan_distance(agent_pos, tuple(f))
                    if dist < closest_food_dist:
                        closest_food_dist = dist
                        closest_food_pos = f
                
                target_dir = self._get_direction(agent_pos, tuple(closest_food_pos))
            else:
                target_dir = (0, 0) # No food left

        # 2. Legal moves signature
        legal_moves = self._get_legal_moves_signature(observation)
        
        # 3. Last Action (to prevent oscillation)
        # last_action is an int (0-4) or None. Convert None to -1.
        last_action_val = last_action if last_action is not None else -1

        state = (mode, target_dir, legal_moves, last_action_val)
        return state

    def _extract_relative_crisis_bfs_state(self, agent_pos, ghosts, food, observation, last_action):
        """
        Same as relative_crisis but uses BFS for Greed Mode navigation.
        """
        # 1. Determine Mode and Target Direction
        closest_ghost_dist = float('inf')
        closest_ghost_pos = None
        closest_ghost_idx = -1
        
        for i, ghost in enumerate(ghosts):
            dist = self._manhattan_distance(agent_pos, tuple(ghost))
            if dist < closest_ghost_dist:
                closest_ghost_dist = dist
                closest_ghost_pos = ghost
                closest_ghost_idx = i
                
        if closest_ghost_dist <= 4:
            # Check if scared
            scared_timers = observation.get('ghost_scared_timers', [])
            is_scared = False
            if len(scared_timers) > closest_ghost_idx:
                # Only hunt if timer is sufficient to reach (dist < timer)
                if scared_timers[closest_ghost_idx] > closest_ghost_dist:
                    is_scared = True

            if is_scared:
                # HUNT MODE
                mode = 2
                # Target: Ghost
                target_dir = self._get_direction(agent_pos, tuple(closest_ghost_pos))
            else:
                # CRISIS MODE
                mode = 1
                # Target: Ghost (to run away from)
                target_dir = self._get_direction(agent_pos, tuple(closest_ghost_pos))
        else:
            # GREED MODE
            mode = 0
            # Get direction to closest food
            food_list = np.argwhere(food)
            if len(food_list) > 0:
                # Find closest food
                closest_food_dist = float('inf')
                closest_food_pos = None
                for f in food_list:
                    dist = self._manhattan_distance(agent_pos, tuple(f))
                    if dist < closest_food_dist:
                        closest_food_dist = dist
                        closest_food_pos = f
                
                # Use BFS if walls are available, otherwise fallback to Manhattan
                if self.walls is not None:
                    target_dir = self._get_maze_direction(agent_pos, tuple(closest_food_pos))
                else:
                    target_dir = self._get_direction(agent_pos, tuple(closest_food_pos))
            else:
                target_dir = (0, 0) # No food left

        # 2. Legal moves signature
        legal_moves = self._get_legal_moves_signature(observation)
        
        # 3. Last Action (to prevent oscillation)
        # last_action is an int (0-4) or None. Convert None to -1.
        last_action_val = last_action if last_action is not None else -1

        state = (mode, target_dir, legal_moves, last_action_val)
        return state

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
        ghost_info = self._get_closest_ghost_info(agent_pos, ghosts, n=3, distance_per_bucket=2)

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
    
    def _extract_relative_state(self, agent_pos, ghosts, food, observation):
        """
        Relative complexity - removes absolute coordinates
        Features: 1 closest ghost, food directions, legal moves
        """
        # Get info about 1 closest ghost
        ghost_info = self._get_closest_ghost_info(agent_pos, ghosts, n=1)
        
        # Check food in cardinal directions
        food_dirs = self._get_food_directions(agent_pos, food)
        
        # Legal moves signature
        legal_moves = self._get_legal_moves_signature(observation)
        
        state = (ghost_info, food_dirs, legal_moves)
        return state
    
    def _extract_relative_radius_state(self, agent_pos, ghosts, food, observation):
        """
        Relative Radius complexity
        Features: Ghosts within radius (cardinal dirs), food dirs, legal moves
        """
        # Get info about ghosts within radius
        ghost_info = self._get_ghost_info_radius(agent_pos, ghosts, radius=5)
        
        # Check food in cardinal directions
        food_dirs = self._get_food_directions(agent_pos, food)
        
        # Legal moves signature
        legal_moves = self._get_legal_moves_signature(observation)
        
        state = (ghost_info, food_dirs, legal_moves)
        return state
    
    def _extract_relative_grid_state(self, agent_pos, ghosts, food, observation):
        """
        Relative Grid complexity
        Features: Ghost threat grid (N, E, S, W), food dirs, legal moves
        """
        # Get threat grid
        threat_grid = self._get_ghost_threat_grid(agent_pos, ghosts, radius=5)
        
        # Check food in cardinal directions
        food_dirs = self._get_food_directions(agent_pos, food)
        
        # Legal moves signature
        legal_moves = self._get_legal_moves_signature(observation)
        
        state = (threat_grid, food_dirs, legal_moves)
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
            
        return (dir_x, dir_y)

    def _get_food_directions(self, agent_pos, food):
        """
        Check for food in cardinal directions (N, E, S, W) within radius 5.
        Returns a tuple of 4 booleans.
        """
        x, y = agent_pos
        radius = 5
        
        has_food_north = False
        for y_coord in range(y + 1, min(y + radius + 1, self.grid_height)):
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

    def _count_nearby_food(self, agent_pos, food, radius=5):
        """
        Count food within a certain radius
        """
        x, y = agent_pos
        food_count = 0
        
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

    def _get_legal_moves_signature(self, observation):
        """
        Extracts legal moves as a hashable tuple of booleans (N, E, S, W)
        """
        # nextLegalMoves is an array of 5 integers, where -1 is padding
        # 0: STOP, 1: EAST, 2: NORTH, 3: WEST, 4: SOUTH
        legal_moves = observation['nextLegalMoves']
        
        has_north = 2 in legal_moves
        has_east = 1 in legal_moves
        has_south = 4 in legal_moves
        has_west = 3 in legal_moves
        
        return (has_north, has_east, has_south, has_west)

    def _get_ghost_info_radius(self, agent_pos, ghosts, radius=5):
        """
        Get info about ghosts within radius using simplified cardinal directions.
        Returns a tuple of ghost states.
        Each ghost state is: (cardinal_direction, distance_bucket) or None if outside radius.
        """
        if len(ghosts) == 0:
            return tuple()

        ghost_features = []
        
        # Calculate distance for all ghosts
        ghost_distances_and_positions = []
        for ghost in ghosts:
            distance = self._manhattan_distance(agent_pos, tuple(ghost))
            ghost_distances_and_positions.append((distance, ghost))
            
        # Sort by distance
        ghost_distances_and_positions.sort(key=lambda x: x[0])
        
        for distance, ghost_position in ghost_distances_and_positions:
            if distance > radius:
                ghost_features.append(0) # 0 = Safe/Far
            else:
                # Ghost is close!
                # 1. Get Cardinal Direction
                dx = ghost_position[0] - agent_pos[0]
                dy = ghost_position[1] - agent_pos[1]
                
                # Simplify to N, E, S, W
                if abs(dx) > abs(dy):
                    # Horizontal
                    direction = 1 if dx > 0 else 3 # East or West
                else:
                    # Vertical
                    direction = 2 if dy > 0 else 4 # North or South
                    
                # 2. Get Distance Bucket
                # Critical (<= 2) or Near (3-5)
                dist_bucket = 1 if distance <= 2 else 2
                
                ghost_features.append((direction, dist_bucket))

        return tuple(ghost_features)
    

    def _get_ghost_threat_grid(self, agent_pos, ghosts, radius=5):
        """Get threat levels for cardinal directions (N, E, S, W)."""
        # Initialize grid: [North, East, South, West]
        # Using list for mutability, convert to tuple at end
        grid = [0, 0, 0, 0] 
        
        if len(ghosts) == 0:
            return tuple(grid)
            
        for ghost in ghosts:
            distance = self._manhattan_distance(agent_pos, tuple(ghost))
            
            if distance > radius:
                continue
                
            # Determine direction
            dx = ghost[0] - agent_pos[0]
            dy = ghost[1] - agent_pos[1]
            
            # 0: North, 1: East, 2: South, 3: West
            direction_idx = -1
            if abs(dx) > abs(dy):
                # Horizontal
                if dx > 0: direction_idx = 1 # East
                else:      direction_idx = 3 # West
            else:
                # Vertical
                if dy > 0: direction_idx = 0 # North
                else:      direction_idx = 2 # South
                
            # Determine threat level
            threat_level = 0
            if distance <= 2:
                threat_level = 2 # Critical
            else:
                threat_level = 1 # Warning
                
            # Update grid with max threat
            grid[direction_idx] = max(grid[direction_idx], threat_level)
            
        return tuple(grid)
    


    def _get_maze_direction(self, start_pos, end_pos):
        """
        Returns the direction of the first step in the shortest path from start_pos to end_pos
        using BFS, accounting for walls.
        """
        if start_pos == end_pos:
            return (0, 0)

        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Queue stores (x, y, first_move_direction)
        queue = deque()
        visited = set()
        visited.add(start_pos)

        # Initialize queue with neighbors
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            next_x, next_y = start_x + dx, start_y + dy
            if not self.walls[next_x][next_y]: # Check if not wall
                queue.append(((next_x, next_y), (dx, dy)))
                visited.add((next_x, next_y))

        while queue:
            (curr_x, curr_y), first_dir = queue.popleft()
            
            if (curr_x, curr_y) == end_pos:
                return first_dir
            
            for dx, dy in directions:
                next_x, next_y = curr_x + dx, curr_y + dy
                if not self.walls[next_x][next_y] and (next_x, next_y) not in visited:
                    visited.add((next_x, next_y))
                    queue.append(((next_x, next_y), first_dir))
                    
        # If no path found (shouldn't happen in connected maze), fallback to Manhattan
        return self._get_direction(start_pos, end_pos)
