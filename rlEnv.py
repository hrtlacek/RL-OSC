#!/usr/bin/env python3

#from typing import Optional
import numpy as np
import logging
import warnings
import time
import threading
import util
import os
import argparse
#from pythonosc import dispatcher, osc_server

from util import ColorFormatter
from RLlib import OscEnv
# RL related imports
#import gymnasium as gym
#from pythonosc.udp_client import SimpleUDPClient
#from gymnasium.utils.env_checker import check_env
from stable_baselines3 import A2C

parser = argparse.ArgumentParser(
                    prog='RL OSC',
                    description='An OSC based Reinforcement Learning Environment and model training.',
                    epilog='Made in The Project "Spirits in Comnplexity", funded by the Austrian Science Fund [10.55776/AR821]')

parser.add_argument('--inport', type=int, default=5000)
parser.add_argument('--outport', type=int, default=3030)
parser.add_argument('-v','--verbose', action='store_true', help='Verbose mode.')
parser.add_argument('-vv','--vverbose', action='store_true', help='Very verbose mode.')

parser.add_argument('-dt','--deltaTime', type=int, default=10, help='Time delay between steps in milliseconds.')
parser.add_argument('-nI', '--numInput', type=int, default=8, help='Input array size (dimensionality of one observation)')
parser.add_argument('-nO', '--numOutput',type=int, default=8, help='Output array size (dimensionality of one action)')

parser.add_argument('-iA', '--inAddress', type=str, default='/toRLosc', help='the osc adress on which observations are received.')
parser.add_argument('-oA', '--outAddress', type=str, default='/fromRLosc', help='the osc adress on which actions are sent.')

parser.add_argument('-n', '--numSteps', type=int, default=1_000_000, help='The number of steps to train.')
parser.add_argument('-r', '--internalReward', action='store_true', help="If set, the reward function is just the sum of the obersvations and no reward needs to be sent.")

parser.add_argument('-s', '--agentSpeed', type=float, default = 0.01, help='Agent speed. The maximum step the agent can do when trying to move through space.')
parser.add_argument('-eS', '--episodeSteps', type=int, default=500, help='Maximum Number of steps per episode.')

args = parser.parse_args()

INPORT =  args.inport
OUTPORT = args.outport
verbose = args.verbose

vverbose = args.vverbose

nSteps =  args.numSteps
INADDR =  args.inAddress
OUTADDR = args.outAddress

nObserv = args.numInput
nAction = args.numOutput
dt = args.deltaTime/1000.
internalReward = args.internalReward
agentSpeed = args.agentSpeed
maxEpisodeSteps = args.episodeSteps



warnings.filterwarnings("ignore", message="X does not have valid feature names")
logger = logging.getLogger("colored_logger")

if vverbose:
    logger.setLevel(logging.DEBUG)
    modelVerbosity = 1
elif verbose:
    logger.setLevel(logging.INFO)
    modelVerbosity = 1
else:
    logger.setLevel(logging.CRITICAL)
    modelVerbosity = 0

env = OscEnv(inport=INPORT, 
             outport=OUTPORT, 
             inAddr=INADDR, 
             outAddr=OUTADDR, 
             nObserv=nObserv, 
             nAction=nAction, 
             dt=dt,
             internalReward = internalReward,
             agentSpeed=agentSpeed,
             maxEpisodeSteps=maxEpisodeSteps )
obs, _ = env.reset()

model = A2C("MultiInputPolicy", env, verbose=modelVerbosity)
print("Starting Training.")
print(f"Will train for approximately :{nSteps * dt/60} minutes.")

try:
    model.learn(total_timesteps=nSteps)
except KeyboardInterrupt:
    print() 
    print("Stopping…")
    env.close()

