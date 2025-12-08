import numpy as np
import pickle
from collections import defaultdict


class NaiveGraphQLearningAgent:
    def __init__(
        self,
        action_space_size=5, #Change action space to 5 because Berkley Pacman has 5 possible actions. later change so we remove stop entirely, temp fix
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01       
     ):
        self.action_space_size = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table, (state, action) - > Q-value
        # Using defaultdict, so unseen states start with Q-value of 0

        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

        # Stats for tracking:
        self.episodes_trained = 0
        self.total_steps = 0

    
    
    def get_action(self, stateGraph, training=True):
        """
        Selects an using epsilon-greedy policy
            state: Current state
            training: If True, use epsilon-greedy. If False, use greedy (for evaluation)
        Returns:
            action: integer action index (0-3)
        """
        

        # During evaluation, always exploit:
        if not training:
            epsilon = 0.0
        else:
            epsilon = self.epsilon

        #Get legal actions an the node pacman is on
        legalActions, pacNodeId = self.extractPacState(stateGraph)

        # Epsilon-greedy action selection:
        if np.random.random() < epsilon:
            if len(legalActions) == 0:
                return np.random.randint(self.action_space_size)  # <-- emergency fallback
            return np.random.choice(legalActions)

        else:
            # Exploit: best action according to q-values
            # Tuple format of state serves as the key in the Q-table, then you can give it an action (as an index) and then see the Q-value of that action
            q_values = self.q_table[tuple(stateGraph["nodes"][pacNodeId])] #array of four numbers, q-value for each of the four actions
            
            maximum = max(q_values)

            equalValueActionList = []
            # We loop through all actions and add those who are equal to the max q-value action of the state
            for action in range(0, len(q_values)):
                if q_values[action] == maximum:
                    equalValueActionList.append(action)

            equalValueActionList = np.array(equalValueActionList)

            #returns a random choice between the arguments with the indices storing maximum values 
            return np.random.choice(equalValueActionList)
        

    def update(self, stateGraph, action, reward, next_stateGraph, done):
        """
        Update gets called every single action agent takes in the environment
        Update Q-value using the Q-learning update rule

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Arguments:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after taking action
            done: episode terminated?
        """
        __, pacNodeId = self.extractPacState(stateGraph) 
        #print("DEBUG Q row length:", len(self.q_table[tuple(stateGraph["nodes"][pacNodeId])]),"Action:", action)
        current_q = self.q_table[tuple(stateGraph["nodes"][pacNodeId])][action]

        if done:
            # No future rewards if episode ended
            target_q = reward
        else:
            # Bellman equation: current reward 0 discounted max future Q
            # Temporal difference learning
            _, nextPacNodeId = self.extractPacState(next_stateGraph)
            max_next_q = np.max(self.q_table[tuple(next_stateGraph["nodes"][nextPacNodeId])])
            target_q = reward + self.gamma * max_next_q

        # Q-learning update
        new_q = current_q + self.alpha * (target_q - current_q)
        self.q_table[tuple(stateGraph["nodes"][pacNodeId])][action] = new_q

        self.total_steps += 1

    
    def decay_epsilon(self):
        """
        Decay epsilonm after each episode
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_trained += 1
        

    def get_average_q_value(self):
        """
        Calculate average Q-value across all states and actions in Q-table
        for tracking learning progress
        
        Returns:
            float: Average Q-value, or 0.0 if Q-table is empty
        """
        if len(self.q_table) == 0:
            return 0.0
        
        total_q = 0.0
        total_entries = 0
        
        for state_q_values in self.q_table.values():
            total_q += np.sum(state_q_values)
            total_entries += len(state_q_values)
        
        return total_q / total_entries if total_entries > 0 else 0.0


    def extractPacState(self, stateGraph):
        pacNodeIndex = 0
        edgeIndicies = []

        legalActions = []

        #print("First 10 edges:", stateGraph["edges"][:10])
        for i, node in enumerate(stateGraph["nodes"]): #Searching for PacNode
            #print("Checking this nodee;", node) #Prints every node that we check
            
            if node[4] == 1: # PacNode found
                pacNodeIndex = i #Save nodeId/nodeIndex
                #print("Pacman is at node:", node) #Confirm that acutally find pacman node

                edgeIndexCounter = 0

                for edge in stateGraph["edges"]: #Looking after the outgoing edges the node is connected to
                    if edge[0] == node[0] or edge[1] == node[0]: #If found, add the action ( basically the direction) for this edge
                         legalActions.append(stateGraph["edge_features"][edgeIndexCounter][0]) #Add edge index to list
                    edgeIndexCounter += 1
                
                for edge in edgeIndicies: # Go through all edges the node is connected to and add their direction to the list of legal actions
                    legalActions.append(stateGraph["edge_features"][edge]) 
            
                break
            
        return legalActions, pacNodeIndex
    
    
    def save(self, filepath):
        """
        Save Q-table and agent parameters to file
        
        Arguments:
            filepath: Path where to save the agent (e.g., 'models/agent.pkl')
        """
        save_dict = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'action_space_size': self.action_space_size,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episodes_trained': self.episodes_trained,
            'total_steps': self.total_steps
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)



    def load(self, filepath):
        """
        Load Q-table and agent parameters from file
        
        Arguments:
            filepath: Path to saved agent file
        """
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
        self.q_table.update(save_dict['q_table'])
        
        # Restore parameters
        self.action_space_size = save_dict['action_space_size']
        self.alpha = save_dict['alpha']
        self.gamma = save_dict['gamma']
        self.epsilon = save_dict['epsilon']
        self.epsilon_decay = save_dict['epsilon_decay']
        self.epsilon_min = save_dict['epsilon_min']
        self.episodes_trained = save_dict['episodes_trained']
        self.total_steps = save_dict['total_steps']
