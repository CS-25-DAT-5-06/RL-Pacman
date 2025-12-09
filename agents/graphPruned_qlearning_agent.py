import numpy as np
import pickle
from collections import defaultdict
from .graph_naive_qlearning_agent import NaiveGraphQLearningAgent

class GraphPrunedQLearningAgent(NaiveGraphQLearningAgent):
    def __init__(self, action_space_size=5, learning_rate=0.1, discount_factor=0.9, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, hops_prune_limit=4):
        super().__init__(action_space_size, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)
        self.hopsPruneLimit = hops_prune_limit


    def get_action(self, stateGraph, training=True):
        # Start off by checking distances and prune graph paths where the ghost is too close
        prunedStateGraph = self.pruneStateGraph(stateGraph)
        
        return super().get_action(prunedStateGraph, training=training)
        

    def pruneStateGraph(self, stateGraph):
        __, pacNodeIndex = self.extractPacState(stateGraph)

        #Find what out what pacman is connected too
        pacOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(stateGraph=stateGraph, nodeId=pacNodeIndex)
        
        #Look for ghost in hopsPruneLimit nodes
        for pacEdge in pacOutgoingEdges:
            #Checking for ghosts in immediate vicinity
            pacEdgeDestinationNode = stateGraph["edges"][pacEdge][1]
            foundGhostOnPath = self.checkNodePathsForGhosts(stateGraph=stateGraph, nodeToCheckId=pacEdgeDestinationNode, hopNumber=0) # Check for ghost present on the PacEdge path and beyond
            if foundGhostOnPath == True:
                stateGraph["edge_feature"][pacEdge][0] = -1 # Invalidate / Cut-Off edge as legal 

        
        return stateGraph


        

    def checkNodePathsForGhosts(self, stateGraph, nodeToCheckId, hopNumber):
        if hopNumber < self.hopsPruneLimit: #Check that we are within our hops limit
            if stateGraph["nodes"][nodeToCheckId][3] == 1: #Check ghost pressence
                return True
            else:
                #We find out which nodes this one is connected to and hop to those
                nodeOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(stateGraph=stateGraph, nodeId=nodeToCheckId)
                for edgeIndex in nodeOutgoingEdges:
                    hopNumber += 1
                    nextNodeToCheckId = stateGraph["edges"][edgeIndex][1]
                    self.checkNodePathsForGhosts(stateGraph=stateGraph, nodeToCheckId=nextNodeToCheckId, hopNumber=hopNumber)
        
        return False
        




    
    def findOutgoingEdgesConnectedFromNode(self, stateGraph, nodeId):
        edgeIndexList = []
        node = stateGraph["nodes"][nodeId]

        for i, edge in enumerate(stateGraph["edges"]):
            if edge[0] == node[0]:
                edgeIndexList.append(i)

        return edgeIndexList
