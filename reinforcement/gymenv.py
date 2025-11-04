import numpy as np
import gymnasium as gym
import pacman as pm
from util import *
import time
import os
import traceback
import sys

def count(list):
    i = 0
    for e in list:
        i += 1
    return i


class PacManEnv(gym.Env):
    def __init__(self, args):
        self.args = args
        self.width = args['layout'].width
        self.height = args['layout'].height
        self.layout = args['layout']

        max = np.array([self.width - 1,self.height - 1])
        
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low = np.array([0,0]),high=max,shape=(2,)),
            "food": gym.spaces.Box(0,1,shape=(self.width,self.height)),
            "ghosts": gym.spaces.Box(low = np.array([[0,0],[0,0]]),high=np.array([[self.width - 1,self.height - 1],[self.width - 1,self.height -1 ]]),shape=(self.layout.getNumGhosts(),2)),
            "capsules": gym.spaces.Box(low = np.array([[-1,-1],[-1,-1]]),high=np.array([[self.width - 1,self.height - 1],[self.width - 1,self.height - 1]]),shape=(count(self.layout.capsules),2))
        })

        self.action_space = gym.spaces.Discrete(5)



    def step(self, action):
        print("not implemented")

    def reset(self, *, seed = None, options = None):
        import __main__
        __main__.__dict__['_display'] = self.args['display']

        rules = pm.ClassicGameRules

        gameDisplay = self.args['display']
        rules.quiet = False
        beQuiet = False

        self.game = rules.newGame(self.args['layout'], self.args['horizon'],self.args['pacman'],self.args['ghosts'],
                             gameDisplay, beQuiet, self.args['catchExceptions'])
        
        self.game.display.initialize(self.state.data)
        self.game.numMoves = 0

        for i in range(len(self.game.agents)):
            agent = self.game.agents[i]
            if not agent:
                self.game.mute(i)
                # this is a null agent, meaning it failed to load
                # the other team wins
                print("Agent %d failed to load" % i, file=sys.stderr)
                self.game.unmute()
                self.game._agentCrash(i, quiet=True)
                return
            if ("registerInitialState" in dir(agent)):
                self.game.mute(i)
                if self.game.catchExceptions:
                    try:
                        timed_func = TimeoutFunction(
                            agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
                        try:
                            start_time = time.time()
                            timed_func(self.state.deepCopy())
                            time_taken = time.time() - start_time
                            self.game.totalAgentTimes[i] += time_taken
                        except TimeoutFunctionException:
                            print("Agent %d ran out of time on startup!" %
                                  i, file=sys.stderr)
                            self.game.unmute()
                            self.game.agentTimeout = True
                            self.game._agentCrash(i, quiet=True)
                            return
                    except Exception as data:
                        self.game._agentCrash(i, quiet=False)
                        self.game.unmute()
                        return
                else:
                    agent.registerInitialState(self.state.deepCopy())
                # TODO: could this exceed the total time
                self.game.unmute()

        self.agentIndex = self.game.startingIndex
        self.numAgents = len(self.game.agents)
        self.timestep = 0


        return super().reset(seed=seed, options=options)
    



