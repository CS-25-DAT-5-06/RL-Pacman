import numpy as np
import gymnasium as gym
import pacman as pm
from util import *
import time
import os
import traceback
import sys
import layout

class gymenv(gym.Env):
    def __init__(self, layoutName):
        self.layout = layout.getLayout(layoutName + ".lay")
        if(self.layout == None):
            raise Exception("The layout " + layoutName + " cannot be found")
        
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low = np.array([0,0]),high=max,shape=(2,)),
            "food": gym.spaces.Box(0,1,shape=(self.width,self.height)),
            "ghosts": gym.spaces.Box(low = np.array([[0,0],[0,0]]),high=np.array([[self.width - 1,self.height - 1],[self.width - 1,self.height -1 ]]),shape=(self.layout.getNumGhosts(),2)),
            "capsules": gym.spaces.Box(low = np.array([[-1,-1],[-1,-1]]),high=np.array([[self.width - 1,self.height - 1],[self.width - 1,self.height - 1]]),shape=(count(self.layout.capsules),2))
        })

        self.action_space = gym.spaces.Discrete(5)


    