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
        # Pruning regardless of training or not, beacuse the agent trains for a specific map now (See how we represent states in the Naive agent)
        prunedGraphDict = self.pruneStateGraph(pruned=graphDict)
        return super().get_action(prunedGraphDict, training=training)
        

    def pruneStateGraph(self, pruned):
        # Make copies of the arrays so we don't modify the env's internal state.
        # graphDict is the raw observation coming from the env (it references env arrays).
        # And since dictionaries are mutable data types, it essentially means that graphDict is a reference to the env's internal state
        pruned = {
            "nodes": pruned["nodes"].copy(),
            "nodesXY": pruned["nodesXY"].copy(),
            "edges": pruned["edges"].copy(),
            "edge_features": pruned["edge_features"].copy()
        }

        __, pacNodeIndex = self.extractPacState(pruned)

        #Find what out what pacman is connected too
        pacOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(graphDict=pruned, nodeId=pacNodeIndex)
        
        #Look for ghost in hopsPruneLimit nodes
        for pacEdge in pacOutgoingEdges:
            #Checking for ghosts in immediate vicinity
            pacEdgeDestinationNode = pruned["edges"][pacEdge][1]
            foundGhostOnPath = self.checkNodePathsForGhosts(graphDict=pruned, nodeToCheckId=pacEdgeDestinationNode, hopNumber=0) # Check for ghost present on the PacEdge path and beyond

            #print(stateGraph.keys())
            if foundGhostOnPath == True:
                pruned["edge_features"][pacEdge][0] = -1 # Invalidate / Cut-Off edge as legal 
        return pruned

    
    def update(self, state, action, reward, next_state, done):
        pruned_state = self.pruneStateGraph(pruned=state)
        pruned_next_state = self.pruneStateGraph(pruned=next_state)
        
        #print("\nDEBUG UPDATE:")
        #print("Original outgoing edges:", state["edges"])
        #print("Pruned outgoing edges:  ", pruned_state["edge_features"])
        #print("Action taken:", action)
        #print("Legal actions passed to update():", self.extractPacState(state)[0])
        
        super().update(pruned_state, action, reward, pruned_next_state, done)
        

    def checkNodePathsForGhosts(self, graphDict, nodeToCheckId, hopNumber):
        if hopNumber < self.hopsPruneLimit: #Check that we are within our hops limit
            if graphDict["nodes"][nodeToCheckId][2] == 1: #Check ghost pressence
                return True
            else:
                #We find out which nodes this one is connected to and hop to those
                nodeOutgoingEdges = self.findOutgoingEdgesConnectedFromNode(graphDict=graphDict, nodeId=nodeToCheckId)
                for edgeIndex in nodeOutgoingEdges:
                    nextNodeToCheckId = graphDict["edges"][edgeIndex][1]
                    if self.checkNodePathsForGhosts(graphDict=graphDict, nodeToCheckId=nextNodeToCheckId, hopNumber=hopNumber+1):
                        return True
        
        return False
    
    def findOutgoingEdgesConnectedFromNode(self, graphDict, nodeId):
        edgeIndexList = []
        node = graphDict["nodes"][nodeId]

        for i, edge in enumerate(graphDict["edges"]):
            if edge[0] == int(nodeId):
                edgeIndexList.append(i)

        return edgeIndexList
