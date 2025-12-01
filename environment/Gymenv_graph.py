import numpy as np
import gymnasium as gym
import networkx as nx
import matplotlib.pyplot as plt

from berkeley_pacman import layout as bp_layout
from .gymenv import GymEnv

from stable_baselines3 import PPO

"""
If config file changes use this
#[REWARDS] 
TIME_PENALTY = -1
EAT_FOOD = 10
EAT_GHOST = 200
WIN = 500
LOSE = -500
CAPSULE = 0
"""
#Build static graph from layout
def _build_graph_from_layout(layout_obj):
    walls = layout_obj.walls
    width = layout_obj.width
    height = layout_obj.height

    G = nx.DiGraph() #Only do DiGraph if you want edges pointing in direction of action

    #Add nodes for all non-wall tiles
    for x in range(width):
        for y in range(height):
            if not walls[x][y]:
                G.add_node((x, y))
    
    action_direction = {
        (0,1): "North",
        (1,0): "East",
        (0,-1): "South",
        (-1,0): "West",
    }
    
    #Add edges to potentially 4 adjacent nodes (check up, down, left, write for non-wall nodes)
    for node in list(G.nodes):
        (x,y) = node
        for x_direction, y_direction in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            adj_x_node, adj_y_node = x + x_direction, y + y_direction

            if (adj_x_node, adj_y_node) in G.nodes:
                G.add_edge((x,y), (adj_x_node,adj_y_node), action_direction =[(x_direction, y_direction)])

            #Both these print functions print edges and the actions between those two edges, dont fully understand the difference between them, but they both print similar things
            #DEBUG: print(list(G.edges((x,y), data=True))) prints all edges and the action_direction between those edges
            #DEBUG: print(list(G.edges(data=True))) #Prints EVERY edges in whole graph

    #DEBUG: print(list(G.edges((6,6), data=True))) #Can write a specific node to test, if you print the visual NetworkX graph below you can find that specific node and see if the adjacency nodes and action direction is correct
    #DEBUG: visual_of_nodes_and_edges(G)#Visual of NetworkX graph, change to directed graph or something else by editing G variable above

    return G
    
#Can visually show nodes and edges so it looks like pacman layout
def visual_of_nodes_and_edges(G, show_labels=True):
    pos = { (x,y): (x, y) for (x,y) in G.nodes }  #-y rotates so it looks more like Pac-man game, but then the NetworkX graph isnt an accurate representation so keep it like it is
    nx.draw(G, pos, with_labels=False, node_size=50) #Node size
    nx.draw_networkx_labels(G, pos, font_size=10) #Label font size
    plt.gca().set_aspect("equal") #keep equal square porpotions so it resembles pacman game
    plt.show()


#We are wrapping GymEnv, but changing the observation so its a np.array with shape (294, 4) 
class GraphGymEnv(GymEnv):
    def __init__(
        self, layoutName, record=False, record_interval=None, config="/experiments/configurations/default.ini", render_mode=None, #If config file changes, use rewards above
        ):
        #Calls parent class constructor GymEnv
        super().__init__(layoutName=layoutName, record=record, record_interval=record_interval, config=config, render_mode=render_mode,) 
        
        self.graph = _build_graph_from_layout(self.layout) #The NetworkX graph built from the layout
        self.node_list = sorted(self.graph.nodes())  #Make sure the matrix is sorted
        self.num_nodes = len(self.node_list) #Number of walkable tiles in total

        #DEBUG: print(self.node_list, self.num_nodes)

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
        pacman_pos = state.getPacmanPosition()        #(x,y) 
        
        #DEBUG: print(pacman_pos) #Use to see pacman position through out training, should see him moving around and occasionally dying and returning to start position
        
        node_features = np.zeros((self.num_nodes, self.num_features), dtype=np.int8) #Create empty feature matrix of shape (num_nodes, num_features)

        #Loop through every node, if something is present we put 1, otherwise 0 should look something like [1, 0, 0, 0] or another combination of 1 or 0
        for i,(x, y) in enumerate(self.node_list):
            node_features[i,0] = 1 if food_grid[x][y] else 0
            node_features[i,1] = 1 if (x, y) in capsules else 0
            node_features[i,2] = 1 if (x, y) in ghosts  else 0
            node_features[i,3] = 1 if (x, y) == pacman_pos else 0

        #DEBUG: print(f"Node {(x,y)} -> features {node_features[i]}") #print the feature vector for a specific node every iteration, use for testing
        return node_features


    
    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options) #GymEnv.reset()
        obs = self.build_node_features() #Use observations from build_node_features not from GymEnv.py

        return obs, {} #Gym requires a return of "observation, info", so we just return empty dict

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action) #We ignore base_obs, we want our own graph based-observation, but everything else from GymEnv.step(), should work

        obs = self.build_node_features() 

        return obs, reward, terminated, truncated, info #Return graph-based observation instead of dict (like gymenv.py)


#Test if works
if __name__ == "__main__":
    #debug_layout("originalClassic")
    env = GraphGymEnv("originalClassic") #Keep uncommented if you run this file like this; "python -m environment.Gymenv_graph"
    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=5000)
