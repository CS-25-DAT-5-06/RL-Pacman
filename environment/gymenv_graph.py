import sys
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Graph, Box
from . import gymenv as ge
import networkx as nx
import matplotlib.pyplot as plt


class GraphEnv(ge.GymEnv):
    def __init__(self, layoutName, record=False, record_interval=None, config="/experiments/configurations/default.ini", render_mode=None):
        #Getting all of the necessary variables initialized just like in gymenv.py
        super().__init__(layoutName, record, record_interval, config, render_mode)        

        # walkable_nodes amount of nodes, 6 feautures each
        # Node Features (pellet/food, capsule, ghostAgentPresent, pacmanAgentPresent, trueX, trueY)
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


        # TODO: Call the gym to nx graph conversion function and visualize it

        # TODO: (Optional) Redefine the action space here for clarity

        # TODO: Actually build the reset() function


        # TODO: Actually build the step() function


    

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

                    node_list.append([self.layout.food[x][y], capsulePresent, ghostAgentPresent, pacmanAgentPresent, x, y])

                    # If there is somebody to actually/potentially connect to
                    if walkable_nodes > 1:
                        extension_for_edge_link_list, extension_for_edge_features_list = self.connectNodeToSurrounding(x, y, node_list)
                        edge_link_list.extend(extension_for_edge_link_list)
                        edge_features_list.extend(extension_for_edge_features_list)
                    
        
        # All of this computing... just to make it into a NumPy array for more computing :)
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



    def connectNodeToSurrounding(self, xCoordinate, yCoordinate, nodeList):
        running_edge_link_list = []
        running_edge_feature_list = []

        for x, y in [(1,0), (-1,0), (0,1), (0,-1)]:
            xCoordinateToTest, yCoordinateToTest = x + xCoordinate, y + yCoordinate
            for node in nodeList:
                # Testing if any of the x and y coordinates are nodes adjacent to the one we want to connect
                if node[4] == xCoordinateToTest and node[5] == yCoordinateToTest:
                    if x == 1 or x == -1:
                        #Adding edge and direction feature
                        running_edge_link_list.append([self.convert_xy_coordinates_to_id(xCoordinate, yCoordinate), self.convert_xy_coordinates_to_id(xCoordinateToTest, yCoordinateToTest)])
                        running_edge_feature_list.append([self._direction_to_action["East"]])

                        #Adding the inverse direction
                        running_edge_link_list.append([self.convert_xy_coordinates_to_id(xCoordinateToTest, yCoordinateToTest), self.convert_xy_coordinates_to_id(xCoordinate, yCoordinate)])
                        running_edge_feature_list.append([self._direction_to_action["West"]])

                    if y == 1 or y == -1:
                        #Adding edge and direction feature
                        running_edge_link_list.append([self.convert_xy_coordinates_to_id(xCoordinate, yCoordinate), self.convert_xy_coordinates_to_id(xCoordinateToTest, yCoordinateToTest)])
                        running_edge_feature_list.append([self._direction_to_action["North"]])

                        #Adding the inverse direction
                        running_edge_link_list.append([self.convert_xy_coordinates_to_id(xCoordinateToTest, yCoordinateToTest), self.convert_xy_coordinates_to_id(xCoordinate, yCoordinate)])
                        running_edge_feature_list.append([self._direction_to_action["South"]])

        return running_edge_link_list, running_edge_feature_list
    
    def convert_xy_coordinates_to_id(self, xCoordinate, yCoordinate, width=10):
        return yCoordinate * width + xCoordinate
    
    #endregion

    # TODO: Implement the gym to nx graph conversion function
    def gymGraphToNXGraph(self):
        G = nx.MultiDiGraph()

        #Starting by adding nodes with their features. Generating labels from their x, y values.



# Gotta test the __init__ function of the GraphEnv class         
if __name__ == "__main__":
    testObj = GraphEnv(layoutName="originalClassic")

