import numpy as np
import gymnasium as gym
from reinforcement import pacman as pm
from reinforcement.util import *
from reinforcement import layout
from reinforcement.game import Directions
import os

from reinforcement.rewardConfig import readConfig

from stable_baselines3 import A2C

import stable_baselines3.common.env_checker as ec

def count(list):
    i = 0
    for e in list:
        i += 1
    return i

#Settings constants
TIMEOUT = 30
ZOOM = 1.0
FRAME_TIME = 0.001
GHOST_AGENT = "RandomGhost"
CATCH_EXCEPTIONS = False
HORIZON = -1
PACMAN = "KeyboardAgent"

RECORDING_PATH = "recordings/"

#Takes an array of tuples and returns an array of arrays
def tupleArrayToArrayArray(list):
    result = []
    for e in list:
        result.append([e[0],e[1]])
        #np.append(result,np.array(np.array([e[0],e[1]])))
    
    return np.array(result, dtype=np.int64)

class GymEnv(gym.Env):
    metadata = {"render_modes":["human"]}

    #Initializes the environment
    def __init__(self, layoutName,record = False, record_interval = None,config =  "../reward_configs/default.ini",render_mode = None):
        self.config = readConfig(config)
        self.record = record
        if record_interval != None:
            self.record_interval = record_interval
        else:
            self.record_interval = 0
        self.gameCount = 0

        if record:
            import time
            dir_name = 'session-' + '-'.join([str(t) for t in time.localtime()[1:6]])
            self.recordings_dir = RECORDING_PATH + "/" + dir_name
            os.makedirs(self.recordings_dir)


        #Load the specified layout
        self.layout = layout.getLayout(layoutName + ".lay")
        if(self.layout == None):
            raise Exception("The layout " + layoutName + " cannot be found")
        
        self.render_mode = render_mode

        #The size of the capsules array
        self.shapeCapsules = 2*count(self.layout.capsules)

        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low = np.array([0,0]),high=np.array([self.layout.width-1,self.layout.height-1]),shape=(2,),dtype=np.int64),
            "food": gym.spaces.Box(low = 0, high = 1, shape = (self.layout.width*self.layout.height,),dtype=np.bool),
            "ghosts": gym.spaces.Box(low = 0,high = max(self.layout.width - 1,self.layout.height - 1),shape=(2*self.layout.numGhosts,) ,dtype=np.int64),
            "capsules": gym.spaces.Box(low = -1,high = max(self.layout.width - 1,self.layout.height - 1),shape=(self.shapeCapsules,), dtype=np.int64),
            "nextLegalMoves": gym.spaces.Box(low = -1, high=4,shape=(5,),dtype=np.int64)
        })
        
        self.action_space = gym.spaces.Discrete(5)

        self._action_to_direction = {
            0: Directions.STOP,
            1: Directions.EAST,
            2: Directions.NORTH,
            3: Directions.WEST,
            4: Directions.SOUTH,
        }
        self._direction_to_action = {
            "Stop": 0,
            "East": 1,
            "North": 2,
            "West": 3,
            "South": 4,
        }

        self._inv_direction_to_action = {
            0: "Stop",
            1: "East",
            2: "North",
            3: "West",
            4: "South"
        }

        print(1)

    def reset(self, seed=None, options=None):
        #From runGames in pacman.py
        self.rules = pm.ClassicGameRules(TIMEOUT)

        ghostType = pm.moduleLoadAgent(GHOST_AGENT,True)
        ghosts = [ghostType(i+1) for i in range(self.layout.numGhosts)]

        pacman = pm.moduleLoadAgent(PACMAN, False)
        
        
        if self.render_mode == None:
            self.beQuiet = True
            # Suppress output and graphics
            import reinforcement.textDisplay as textDisplay
            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.beQuiet = False
            import reinforcement.graphicsDisplay as graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(
            ZOOM, frameTime=FRAME_TIME)
            self.gameDisplay = display
            self.rules.quiet = False
        self.game = self.rules.newGame(self.layout, HORIZON, pacman, ghosts,
                             self.gameDisplay, self.beQuiet, CATCH_EXCEPTIONS, config=self.config)
        
        print(1)
        #FROM game.run in game.py

        self.game.display.initialize(self.game.state.data)
        self.game.numMoves = 0
        
        self.agentIndex = self.game.startingIndex
        self.numAgents = len(self.game.agents)

        currState = self.game.state
        obsLegalActions = np.empty(shape=(5,),dtype=np.int64)
        legalActions = self.game.state.getLegalPacmanActions()
        for i in range(0,5):
            if(i < len(legalActions)):
                obsLegalActions[i] = self._direction_to_action[legalActions[i]]
            else:
                obsLegalActions[i] = -1

        observation = dict({
            "agent": np.array([currState.getPacmanPosition()[0],currState.getPacmanPosition()[1]]),
            "food":  currState.getFood().asNpArray().flatten(),
            "ghosts": tupleArrayToArrayArray(currState.getGhostPositions()).flatten(),
            "capsules": tupleArrayToArrayArray(currState.getCapsules()).flatten(),
            "nextLegalMoves": obsLegalActions
        })
        
        return observation, dict()

    def step(self, action):

        #Check if action is in illegal actions
        if self._inv_direction_to_action[action] not in self.game.state.getLegalPacmanActions():          
            action = 0 #STOP

        #Used for calculating reward
        prevScore = self.game.state.getScore()

        action = self._action_to_direction[action]
        
        #Iterates over all the agents (Pacman and ghosts)
        for agentIndex in range(0, self.numAgents):
            agent = self.game.agents[agentIndex]

            #Only true for ghosts
            if agentIndex != 0:
                observation = self.game.state.deepCopy()
                action = agent.getAction(observation)

            #Execute action
            self.game.moveHistory.append((agentIndex, action))
            self.game.state = self.game.state.generateSuccessor(agentIndex, action)
            self.game.display.update(self.game.state.data) #Updates visuals
            self.game.rules.process(self.game.state, self.game)

           
            if(self.game.gameOver):
                terminated = True
                self.gameCount += 1     
                if self.record:
                    if(self.gameCount % self.record_interval == 0):
                        import pickle
                        fname = self.recordings_dir + ('/recorded-game-%d' % self.gameCount)
                        f = open(fname, 'wb')
                        components = {'layout': self.layout, 'actions': self.game.moveHistory}
                        pickle.dump(components, f)
                        f.close()
                break
            terminated = False
        
        currState = self.game.state



        obsLegalActions = np.empty(shape=(5,),dtype=np.int64)
        legalActions = self.game.state.getLegalPacmanActions()
        for i in range(0,5):
            if(i < len(legalActions)):
                obsLegalActions[i] = self._direction_to_action[legalActions[i]]
            else:
                obsLegalActions[i] = -1 #Pad with -1 to ensure correct size and that elements are within low and high

        # Bugfix: Changed from getGhostPositions() to getCapsules()       
        capsules = tupleArrayToArrayArray(currState.getCapsules()).flatten()

        # Bugfix: Replaced while loop with np.pad() which returns a new array
        if len(capsules) < self.shapeCapsules:
            capsules = np.pad(capsules, (0, self.shapeCapsules - len(capsules)), constant_values=-1)
        
        observation = dict({
            "agent": np.array([currState.getPacmanPosition()[0],currState.getPacmanPosition()[1]]),
            "food":  currState.getFood().asNpArray().flatten(),
            "ghosts": tupleArrayToArrayArray(currState.getGhostPositions()).flatten(),
            "capsules": capsules,
            "nextLegalMoves": obsLegalActions
        })

        reward = self.game.state.getScore() - prevScore

        return observation, reward, terminated, False, dict()
    
    def render(self):
        self.render_mode = "human"
#TensorBoard stuff
models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

if __name__ == '__main__':
    print(1)
    gym.register(id="berkley-pacman",entry_point=GymEnv,max_episode_steps=300,kwargs = {"layoutName": "openClassic", "record": False, "record_interval": None, "config": "../reward_configs/default.ini", "render_mode": None})
    env = gym.make("berkley-pacman", layoutName = "originalClassic", record = True, record_interval=2,config = "../reward_configs/inverseDefault.ini", render_mode = None) #Removed rendering during training
    
    #Training x amount of times (without rendering)
    model = A2C("MultiInputPolicy",env, verbose=1, tensorboard_log=logdir) #"python -m tensorboard.main --logdir=logs --port=6006"
    model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name="A2C") 
    model.save("trained_pacman") #Save last model, "trained pacman"
    env.close() 

    #Rendering last model after training is finished, showing "trained pacman"
    env = gym.make("berkley-pacman", layoutName = "originalClassic", config = "../reward_configs/inverseDefault.ini", render_mode = "human")
    model = A2C.load("trained_pacman", env=env) #Change to whatever algorithm we are using 

    obs, info = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        action = int(action)  #SB3 returns action as np.array, have to convert to int so env.step() gets an int
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

