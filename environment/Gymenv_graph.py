import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt

from berkeley_pacman import layout as bp_layout
from .gymenv import GymEnv

from stable_baselines3 import PPO


#Build static graph from layout
def build_graph_from_layout(layout_obj):
    walls = layout_obj.walls
    width = layout_obj.width
    height = layout_obj.height

    G = nx.Graph()

    #Add nodes for all non-wall tiles
    for x in range(width):
        for y in range(height):
            if not walls[x][y]:
                G.add_node((x, y))

    #Add edges to potentially 4 adjacent nodes (check up, down, left, write for non-wall nodes)
    for (x, y) in list(G.nodes):
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in G.nodes:
                G.add_edge((x, y), (nx_, ny_)) #Graph is undirected right now (maybe change later)

    #visual_of_nodes_and_edges(G)

    return G

#Can visually show nodes and edges so it looks like pacman layout
def visual_of_nodes_and_edges(G):
    pos = { (x,y): (x, -y) for (x,y) in G.nodes }  #-y rotates so it looks more like Pac-man game
    nx.draw(G, pos, with_labels=False, node_size=50) #Node size
    nx.draw_networkx_labels(G, pos, font_size=5) #Label font size
    plt.gca().set_aspect("equal") #keep square proportions
    plt.show()


#We are wrapping GymEnv, but changing the observation so its a np.array with shape (294, 4) 
class GraphGymEnv(GymEnv):
    def __init__(
        self, layoutName, record=False, record_interval=None, config="/experiments/configurations/default.ini", render_mode=None,
    ):
        #Calls parent class constructor GymEnv
        super().__init__(layoutName=layoutName, record=record, record_interval=record_interval, config=config, render_mode=render_mode,) 

        
        self.graph = build_graph_from_layout(self.layout) #The NetworkX graph built from the layout
        self.node_list = sorted(self.graph.nodes())  #Make sure the matrix is sorted
        self.num_nodes = len(self.node_list) #Number of walkable tiles in total

        #print(self.node_list, self.num_nodes)

        #Each node will contain: [pellet, capsule, ghost, pacman]
        self.num_features = 4

        #The observation space is a matrix of shape num_nodes x num_features (should be 294 x 4) where each value is either 0 or 1
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_features), dtype=np.int8,)

    #Build a (num_nodes, 4) feature matrix for each graph node
    def build_node_features(self):
        state = self.game.state 

        #Reads the game info
        food_grid = state.getFood().asNpArray()       #bool [x][y]
        capsules = set(state.getCapsules())           #set[(x, y)]
        ghosts = set(state.getGhostPositions())       #set[(x, y)]
        pacman_pos = state.getPacmanPosition()        #(x, y)

        feats = np.zeros((self.num_nodes, self.num_features), dtype=np.int8) #Create empty feature matrix of shape (num_nodes, num_features)

        #Loop through every node, if something is present we put 1, otherwise 0
        for i, (x, y) in enumerate(self.node_list):
            feats[i, 0] = 1 if food_grid[x][y] else 0
            feats[i, 1] = 1 if (x, y) in capsules else 0
            feats[i, 2] = 1 if (x, y) in ghosts else 0
            feats[i, 3] = 1 if (x, y) == pacman_pos else 0

        return feats
    
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options) #GymEnv.reset()
        obs = self.build_node_features() #Use observations from build_node_features not from GymEnv.py

        return obs, {} #Gym requires a return of "observation, info", so we just return empty dict

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action) #We ignore base_obs, we want our own graph based-observation, but everything else from GymEnv.step()

        obs = self.build_node_features() 

        return obs, reward, terminated, truncated, info #Return graph-based observation instead of dict

"""
#Test if works
if __name__ == "__main__":
    debug_layout("originalClassic")
    env = GraphGymEnv("originalClassic")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
"""