import sys
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Graph, Box
from . import gymenv as ge
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from berkeley_pacman import pacman as pm
from berkeley_pacman.util import *
from berkeley_pacman import layout
from berkeley_pacman.game import Directions


#Settings constants
TIMEOUT = 30
ZOOM = 1.0
FRAME_TIME = 0.001
GHOST_AGENT = "RandomGhost"
CATCH_EXCEPTIONS = False
HORIZON = -1
PACMAN = "KeyboardAgent"

RECORDING_PATH = "data/recordings/"


class GraphEnv(ge.GymEnv):
    def __init__(self, layoutName, record=False, record_interval=None, reward_config=None, render_mode=None):
        #Getting all of the necessary variables initialized just like in gymenv.py
        super().__init__(layoutName, record, record_interval, reward_config, render_mode)        

        if reward_config is None:
            pm.rewardConfig = {
            "EAT_FOOD": 10,
            "EAT_GHOST": 200,
            "TIME_PENALTY": -1,
            "WIN": 500,
            "LOSE": -500,
            "CAPSULE": 10
        }
        else:
            pm.rewardConfig = reward_config
        # walkable_nodes amount of nodes, 7 feautures each
        # Node Features (id, pellet/food, capsule, ghostAgentPresent, pacmanAgentPresent, trueX, trueY)
        self.nodes, self.edges, self.edge_features = self.createGraphFromLayout()

        #np.set_printoptions(threshold=sys.maxsize)
        #print(f"Nodes:  {self.nodes}")
        #print(f"Edges:  {self.edges}")
        #print(f"Edge Features:  {self.edge_features}")

        # TODO: Create the observation space as a gym graph (with gymnasium.spaces.Graph)
        self.nodeFeatureDim = self.nodes.shape[1]
        self.edgeFeatureDim = self.edge_features.shape[1]

        self.observation_space = Graph(
            node_space=Box(
                low=-np.inf, high=np.inf,
                shape=(self.nodeFeatureDim, ),
                dtype=np.int64
                ),
            
            edge_space=Box(
                low=-np.inf, high=np.inf,
                shape=(self.edgeFeatureDim, ),
                dtype=np.int64
            )
        )

        #Getting the NetworkX version of the graph, for visualization purposes
        self.environmentNXGraph = self.gymGraphToNXGraph()
        #self.visual_of_nodes_and_edges(self.environmentNXGraph)
        self.fig, self.ax = plt.subplots()
    

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.nodes, self.edges, self.edge_features = self.createGraphFromLayout()
        self.environmentNXGraph = self.gymGraphToNXGraph()

        observation = dict({
            "nodes": self.nodes,
            "edges": self.edges,
            "edge_features": self.edge_features
        })

        return observation, dict()


    def step(self, action):
        gymEnvObs, reward, terminated, truncated, info = super().step(action)

        # Update PacMan location
        pacNodeIndex, ghostNodesIndicies = self.getAgentsNodes()
        newPacNodeIndex, newGhostNodesIndicies = self.parseGymEnvObs(gymEnvObs)

        # Remove old PacMan and ghost Position in Graph
        self.nodes[pacNodeIndex][4] = 0
        self.environmentNXGraph.nodes[pacNodeIndex]["features"] = self.nodes[pacNodeIndex]
        for oldGhost in ghostNodesIndicies:
            self.nodes[oldGhost][3] = 0
            self.environmentNXGraph.nodes[oldGhost]["features"] = self.nodes[oldGhost]
        
        # Add new PacMan and Ghost position
        self.nodes[newPacNodeIndex][4] = 1
        self.environmentNXGraph.nodes[newPacNodeIndex]["features"] = self.nodes[newPacNodeIndex]
        for newGhost in newGhostNodesIndicies:
            self.nodes[newGhost][3] = 1
            self.environmentNXGraph.nodes[newGhost]["features"] = self.nodes[newGhost]

        # Eat Food/Pellet/Capsule
        self.nodes[newPacNodeIndex][1] = 0
        self.nodes[newPacNodeIndex][2] = 0

        observation = dict({
            "nodes": self.nodes,
            "edges": self.edges,
            "edge_features": self.edge_features
        })

        return observation, reward, terminated, truncated, info



    

    #region BUILD LAYOUT GRAPH

    def createGraphFromLayout(self):
        walkable_nodes = 0

        node_list = []
        edge_link_list = []
        edge_features_list = []

        for y in range(self.layout.height):
            for x in range(self.layout.width):
                if self.layout.walls[x][y] != True:
                    walkable_nodes += 1
                    capsulePresent = self.checkCapsulePresence(x, y)
                    ghostAgentPresent, pacmanAgentPresent = self.checkAgentPresence(x, y)

                    node_list.append([walkable_nodes - 1, self.layout.food[x][y], capsulePresent, ghostAgentPresent, pacmanAgentPresent, x, y])

        edge_link_list, edge_features_list = self.connectNodesToSurroundingNodes(node_list)
                    
        
        #All of this computing... just to make it into a NumPy array for more computing :)
        return np.array(node_list, dtype=np.int64), np.array(edge_link_list, dtype=np.int64), np.array(edge_features_list, dtype=np.int64)


    def checkCapsulePresence(self, xCoordinate, yCoordinate):
        capsuleFound = 0
        for capsule in self.layout.capsules:
            if capsule[0] == xCoordinate and capsule[1] == yCoordinate:
                capsuleFound = 1
        
        return capsuleFound
    
    def checkAgentPresence(self, xCoordinate, yCoordinate):
        ghostPresent = 0
        pacmanPresent = 0
        for agents in self.layout.agentPositions:
            if agents[1][0] == xCoordinate and agents[1][1] == yCoordinate:
                if agents[0] == 1:
                    ghostPresent = 1
                if agents[0] == 0:
                    pacmanPresent = 1

        return ghostPresent, pacmanPresent



    def connectNodesToSurroundingNodes(self, nodeList):
        running_edge_link_list = []
        running_edge_feature_list = []

        # We go through every node
        for node in nodeList:
            # We calculate the adjacent coordinates in each direction
            # Future Improvement opportunity: Calculate all of of the adjacent directions instead of looping through them and check them all in one go in the loop below
            for x, y in [(1,0), (-1,0), (0,1), (0,-1)]:
                nodeX = node[5]
                nodeY = node[6]
                xCoordinateToTest, yCoordinateToTest = x + nodeX, y + nodeY
                for nodeToTest in nodeList:
                    # Testing if any of the nodeToTest nodes is adjacent to node
                    if nodeToTest[5] == xCoordinateToTest and nodeToTest[6] == yCoordinateToTest:
                        # Adding edge and direction feature
                        if x == 1:
                            running_edge_link_list.append([node[0], nodeToTest[0]])
                            running_edge_feature_list.append([self._direction_to_action["East"]])
                            break

                        elif x == -1:
                            running_edge_link_list.append([node[0], nodeToTest[0]])
                            running_edge_feature_list.append([self._direction_to_action["West"]])
                            break

                        elif y == 1:
                            running_edge_link_list.append([node[0], nodeToTest[0]])
                            running_edge_feature_list.append([self._direction_to_action["North"]])
                            break

                        elif y == -1:
                            running_edge_link_list.append([node[0], nodeToTest[0]])
                            running_edge_feature_list.append([self._direction_to_action["South"]])
                            break
                    
        return running_edge_link_list, running_edge_feature_list
    
    #endregion

    
    def gymGraphToNXGraph(self):
        G = nx.MultiDiGraph()

        #Starting by adding nodes with their features. Generating labels from their x, y values.
        for i, features in enumerate(self.nodes):
            nodeId = self.nodes[i][0]
            nodeX = self.nodes[i][5]
            nodeY = self.nodes[i][6]
            nodeLabel = f"({nodeId}, ({nodeX}, {nodeY}))"
            G.add_node(i, features=features, pos=(nodeX, nodeY), label=nodeLabel)

        for i, (u, v) in enumerate(self.edges):
            G.add_edge(u, v, action=self.edge_features[i][0], label=self._inv_direction_to_action[self.edge_features[i][0]])

        return G

    # Got this function from David. Danke!
    def visual_of_nodes_and_edges(self, G, show_labels=True):
        pos = nx.get_node_attributes(G, "pos")  #-y rotates so it looks more like Pac-man game, but then the NetworkX graph isnt an accurate representation so keep it like it is
        nx.draw(G, pos, with_labels=True, node_size=150) #Node size
        nx.draw_networkx_labels(G, pos, font_size=10) #Label font size
        plt.gca().set_aspect("equal") #keep equal square porpotions so it resembles pacman game
        plt.show()

    def animate_graph(self, frame):
        #Random valid action (0..4). 
        action = int(self.action_space.sample())

        #Step underlying environment; this updates game state
        observation, reward, terminated, truncated, info = self.step(action)

        #Use ax instead
        self.ax.cla()

        G = self.environmentNXGraph
        pos = nx.get_node_attributes(G, "pos")

        colors = []
        for _, data in G.nodes(data=True):
            features = data["features"]
            ghostPresent = features[3]
            pacmanPresent = features[4]

            if pacmanPresent == 1:
                colors.append("yellow")
            elif ghostPresent == 1:
                colors.append("red")
            else:
                colors.append("skyblue")

        nx.draw(G, pos, node_color=colors, with_labels=True, node_size=150, ax=self.ax)
        self.ax.set_aspect("equal") #Get it ax instead of clear


    #Function to convert the observation from gymenv.py to changes in graph
    def parseGymEnvObs(self, gymEnvObs):
        newPacNodeIndex = None
        newGhostNodeIndicies = []

        newPacNodeX = gymEnvObs["agent"][0]
        newPacNodeY = gymEnvObs["agent"][1]
        newGhostPosList = gymEnvObs["ghosts"].reshape(-1, 2)

        newPacNodeIndex = None
        newGhostNodeIndicies = []

        for i, node in enumerate(self.nodes):       
            if node[5] == newPacNodeX and node[6] == newPacNodeY:
                newPacNodeIndex = i                

            for ghostPos in newGhostPosList:
                if node[5] == ghostPos[0] and node[6] == ghostPos[1]:
                    newGhostNodeIndicies.append(i)   

        return newPacNodeIndex, newGhostNodeIndicies 
        
        

        

    def getAgentsNodes(self):
        pacNodeIndex = None
        ghostNodesIndices = []
        for i, node in enumerate(self.nodes):
            if node[4] == 1:
                pacNodeIndex = i
            if node[3] == 1:
                ghostNodesIndices.append(i)

        return pacNodeIndex, ghostNodesIndices
    
    
    #def getNodeOutgoingEdges(self, nodeIndex):
    #    list_of_outgoing_edges = []
    #    for edge in self.edges:
    #        if edge[0] == nodeIndex:
    #            list_of_outgoing_edges.append(edge)
    #    
    #    return list_of_outgoing_edges


#Gotta test the __init__ function of the GraphEnv class         
if __name__ == "__main__":
    env = GraphEnv(layoutName="originalClassic")

    #Reset once to initialize everything to start
    obs, info = env.reset()

    #Create a Matplotlib figure and start the animation
    ani = animation.FuncAnimation(env.fig, frames=500, interval=200, repeat=False)
    plt.show()
