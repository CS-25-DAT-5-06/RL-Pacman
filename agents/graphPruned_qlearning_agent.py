import numpy as np
import pickle
from collections import defaultdict
from .graph_naive_qlearning_agent import NaiveGraphQLearningAgent

class GraphPrunedQLearningAgent(NaiveGraphQLearningAgent):
    def __init__(self, action_space_size=5, learning_rate=0.1, discount_factor=0.9, epsilon=1, epsilon_decay=0.995, epsilon_min=0.01, hops_prune_limit=4):
        super().__init__(action_space_size, learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)
        self.hopsPruneLimit = hops_prune_limit


    def get_action(self, graphDict, training=True):
        # Start off by checking distances and prune graph paths where the ghost is too close
        if training == True:
            prunedGraphDict = self.pruneStateGraph(graphDict=graphDict)
        
        return super().get_action(prunedGraphDict, training=training)
        

    def pruneStateGraph(self, graphDict):
        __, pacNodeIndex = self.extractPacState(graphDict)

        #Find what out what pacman is connected too
        pacOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(graphDict=graphDict, nodeId=pacNodeIndex)
        
        #Look for ghost in hopsPruneLimit nodes
        for pacEdge in pacOutgoingEdges:
            #Checking for ghosts in immediate vicinity
            pacEdgeDestinationNode = graphDict["edges"][pacEdge][1]
            foundGhostOnPath = self.checkNodePathsForGhosts(graphDict=graphDict, nodeToCheckId=pacEdgeDestinationNode, hopNumber=0) # Check for ghost present on the PacEdge path and beyond

            #print(stateGraph.keys())
            if foundGhostOnPath == True:
                graphDict["edge_features"][pacEdge][0] = -1 # Invalidate / Cut-Off edge as legal 
        return graphDict


        

    def checkNodePathsForGhosts(self, graphDict, nodeToCheckId, hopNumber):
        if hopNumber < self.hopsPruneLimit: #Check that we are within our hops limit
            if graphDict["nodes"][nodeToCheckId][2] == 1: #Check ghost pressence
                return True
            else:
                #We find out which nodes this one is connected to and hop to those
                nodeOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(graphDict=graphDict, nodeId=nodeToCheckId)
                for edgeIndex in nodeOutgoingEdges:
                    hopNumber += 1
                    nextNodeToCheckId = graphDict["edges"][edgeIndex][1]
                    self.checkNodePathsForGhosts(graphDict=graphDict, nodeToCheckId=nextNodeToCheckId, hopNumber=hopNumber)
        
        return False
    
    def findOutgoingEdgesConnectedFromNode(self, graphDict, nodeId):
        edgeIndexList = []
        node = graphDict["nodes"][nodeId]

        for i, edge in enumerate(graphDict["edges"]):
            if edge[0] == node[0]:
                edgeIndexList.append(i)

        return edgeIndexList
