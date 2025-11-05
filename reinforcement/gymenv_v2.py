import numpy as np
import gymnasium as gym
import pacman as pm
from util import *
import time
import os
import traceback
import sys
import layout
import ghostAgents as ga
from game import Directions

def count(list):
    i = 0
    for e in list:
        i += 1
    return i


TIMEOUT = 30
ZOOM = 1.0
FRAME_TIME = 0.0
GHOST_AGENT = "RandomGhost"
CATCH_EXCEPTIONS = False
HORIZON = -1

PACMAN = "KeyboardAgent"
GHOSTS = "RandomGhost"

#Takes an array of tuples and returns an array of arrays
def tupleArrayToArrayArray(list):
    result = []
    for e in list:
        result.append([e[0],e[1]])
        #np.append(result,np.array(np.array([e[0],e[1]])))
    
    return np.array(result, dtype=np.int64)

class GymEnv(gym.Env):
    metadata = {"render_modes":["human"]}

    def __init__(self, layoutName,render_mode = None):
        self.layout = layout.getLayout(layoutName + ".lay")
        if(self.layout == None):
            raise Exception("The layout " + layoutName + " cannot be found")
        
        self.render_mode = render_mode

        max = np.array([self.layout.width-1,self.layout.height-1])
        
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low = np.array([0,0]),high=max,shape=(2,),dtype=np.int64),
            "food": gym.spaces.Box(low = 0, high = 1, shape = (self.layout.width,self.layout.height), dtype=np.bool),
            "ghosts": gym.spaces.Box(low = np.array([[0,0]]),high=np.array([[self.layout.width - 1,self.layout.height - 1]]),shape=(self.layout.getNumGhosts(),2),dtype=np.int64),
            "capsules": gym.spaces.Box(low = np.array([[-1,-1]]),high=np.array([[self.layout.width - 1,self.layout.height - 1]]),shape=(count(self.layout.capsules),2),dtype=np.int64)
        })
        
        
        self.action_space = gym.spaces.Discrete(5)

        self._action_to_direction = {
            0: Directions.STOP,
            1: Directions.EAST,
            2: Directions.NORTH,
            3: Directions.WEST,
            4: Directions.SOUTH,
        }

    def reset(self, seed=None, options=None):
        #From runGames in pacman.py
        self.rules = pm.ClassicGameRules(TIMEOUT)

        ghostType = pm.loadAgent(GHOST_AGENT,True)
        ghosts = [ghostType(i+1) for i in range(self.layout.numGhosts)]


        pacman = pm.loadAgent(PACMAN, False)
        

        if self.render_mode == None:
            self.beQuiet = True
            # Suppress output and graphics
            import textDisplay
            self.gameDisplay = textDisplay.NullGraphics()
            self.rules.quiet = True
        else:
            self.beQuiet = False
            import graphicsDisplay
            display = graphicsDisplay.PacmanGraphics(
            ZOOM, frameTime=FRAME_TIME)
            self.gameDisplay = display
            self.rules.quiet = False
        self.game = self.rules.newGame(self.layout, HORIZON, pacman, ghosts,
                             self.gameDisplay, self.beQuiet, CATCH_EXCEPTIONS)
        
        #FROM game.run in game.py

        self.game.display.initialize(self.game.state.data)
        self.game.numMoves = 0
        
        self.agentIndex = self.game.startingIndex
        self.numAgents = len(self.game.agents)
        timestep = 0


        currState = self.game.state

        observation = dict({
            "agent": np.array([currState.getPacmanPosition()[0],currState.getPacmanPosition()[1]]),
            "food":  currState.getFood().asNpArray(),
            "ghosts": tupleArrayToArrayArray(currState.getGhostPositions()),
            "capsules": tupleArrayToArrayArray(currState.getCapsules())
        })
        
        return observation, dict()

    def step(self, action):
        
        action = self._action_to_direction[action]
        #for agent in self.game.agents:
        for agentIndex in range(0, self.numAgents):
            agent = self.game.agents[agentIndex]
            if agentIndex != 0:
                observation = self.game.state.deepCopy()
                action = agent.getAction(observation)
                print("here")

            self.game.moveHistory.append((agentIndex, action))

            self.game.state = self.game.state.generateSuccessor(agentIndex, action)

            self.game.display.update(self.game.state.data)

            self.game.rules.process(self.game.state, self.game)

            if(self.game.gameOver):
                terminated = True
                break
            terminated = False
        
        currState = self.game.state

        observation = dict({
            "agent": np.array([currState.getPacmanPosition()[0],currState.getPacmanPosition()[1]]),
            "food":  currState.getFood().asNpArray(),
            "ghosts": tupleArrayToArrayArray(currState.getGhostPositions()),
            "capsules": tupleArrayToArrayArray(currState.getCapsules())
        })
        reward = 0
        

        return observation, reward, terminated, False, dict()

if __name__ == '__main__':
    gym.register(id="berkley-pacman",entry_point=GymEnv,max_episode_steps=300,kwargs = {"layoutName": "openClassic", "render_mode": None})
    env = gym.make("berkley-pacman", render_mode = "human")

    env.reset()

    terminated = False
    while terminated == False:
        observation, reward, terminated, truncated, info = env.step(1)
