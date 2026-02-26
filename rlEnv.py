from typing import Optional
import numpy as np
import gymnasium as gym
import logging
from util import ColorFormatter
import warnings
import time
import threading
import util
import os
from pythonosc import dispatcher, osc_server
from pythonosc.udp_client import SimpleUDPClient
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import A2C
import argparse

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
# https://gymnasium.farama.org/introduction/create_custom_env/

# === Key design questions ===
# What skill should the agent learn?
# To produce offsets to the latency space of a rave model that maximizes own output and minimizes audio input(makes other agents shut up)

# What information does the agent need?
# Own input (other models output), own state, own output


# What actions can the agent take?

# How do we measure success?

# When should episodes end?



class OscEnv(gym.Env):
    def __init__(self, inport=5000, outport=3030, 
                 inAddr = '/toRLosc', 
                 outAddr = 'fromRLosc', 
                 nObserv=8, 
                 nAction=8, 
                 dt=0.1,
                 internalReward = True,
                 agentSpeed=0.01, maxEpisodeSteps = 500 ):
        self.agentSpeed = agentSpeed      
        self.IN_PORT = inport
        self.IN_IP = '0.0.0.0'
        self.IN_ADDR = inAddr #"/toRLosc"
        self.maxEpisodeSteps = maxEpisodeSteps
        self.OUT_IP = '127.0.0.1'
        self.OUT_PORT = outport
        self.OUT_ADDR = outAddr #"/fromRLosc"

        self.size = nObserv
        self.nAction = nAction
        self.dt = dt
        self.last_obs = np.zeros(self.size, dtype=np.float32)
        self.internalReward = internalReward 
        
        self.disp = dispatcher.Dispatcher()
        self.disp.map(self.IN_ADDR, self.handle_osc_input)
        if not self.internalReward:
            self.last_reward = 0
            self.disp.map('/reward', self.handle_osc_input)

        self.server = osc_server.ThreadingOSCUDPServer((self.IN_IP, self.IN_PORT), self.disp)
        logger.info(f"Listening for OSC on port {self.IN_PORT}...")
        
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True
        )
        self.thread.start()

         # client for actions
        self.client = SimpleUDPClient(self.OUT_IP, self.OUT_PORT)

        self.observation_space = gym.spaces.Dict(
            {
                "agentState": gym.spaces.Box(low=0, high=1, shape=(self.size,), dtype=np.float32),   # own state, possibly current latent space
                "envState": gym.spaces.Box(low=0, high=1, shape=(self.size,), dtype=np.float32),  # input audio features (other agents state).
            }
        )

        self.action_space = gym.spaces.Box(low=-1,high=1, shape=(nAction,), dtype=np.float32)

    def handle_osc_input(self, addr, *args):
        if addr==self.IN_ADDR:
            logger.debug(args)
            data = np.array(args[:self.size],dtype=np.float32)
            logger.debug(f"data received: {data}")
            if len(data) != self.size:
                logger.critical("Received the wrong number of observations!")
                logger.critical(f"Data received: {data}")
                #self.last_obs = np.zeros(self.size)
                return
            self.last_obs = data

        elif addr == '/reward':
            rew = float(args[0])
            logger.debug(f'Received reward: {rew}')
            self.last_reward = rew
        else:
            print(f'Rceived something weird: {addr}')

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agentState": self._agent_location, "envState": self.last_obs}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "inputFeatures": 
                self.last_obs
        }

    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        # direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        #self._agent_location = np.clip(
        #    self._agent_location + action*self.agentSpeed, -1, 1
        #)
        self.step_count += 1

        self._agent_location = self._agent_location*0.99 + action*self.agentSpeed

        # print(self._agent_location)
        # print(type(self._agent_location))
        # self.client.send_message(self.OUT_ADDR, self._agent_location)
        self.act(self._agent_location)

        # Check if agent reached the target
        #terminated = np.array_equal(self._agent_location, self._target_location)
        #terminated = False

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        #truncated = False
        truncated = self.step_count >= self.maxEpisodeSteps
        if self.internalReward:
            reward = float(np.sum(np.array(self.last_obs)))
        else:
            reward = self.last_reward

        atBorder = np.sum(np.abs(self._agent_location)) > len(self._agent_location)/1.1

        if atBorder:
            self.stuckCount += 1
            trunctaed=1
        else:
            self.stuckCount = 0
        
        stuckLim = 50
        if self.stuckCount>stuckLim:
            truncated = 1
            print('Agent stuck at border. Truncating.')

        truncated = truncated or reward < -10
        terminated = reward >= 0.99
        
        if truncated:
            print('trunc')
        if terminated:
            print('term')

        self.client.send_message("/reward", reward)
        logger.info(f"Reward: {reward}")

        observation = self._get_obs()
        logger.debug(f"Observation: {observation}")
        info = self._get_info()
        time.sleep(self.dt)
        return observation, reward, terminated, truncated, info
    


    def act(self, vals):
        nativeList = [float(v) for v in vals] 
        self.client.send_message(self.OUT_ADDR, nativeList)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        #super().reset(seed=seed)
        self.step_count = 0
        self.stuckCount = 0
        # Randomly place the agent anywhere in the space
        self._agent_location = self.np_random.random(size=self.size, dtype=np.float32)*2-1

        #self._target_location = self.np_random.random(size=self.size, dtype=np.float32)
        # # Randomly place target, ensuring it's different from agent position
        # self._target_location = self._agent_location
        # while np.array_equal(self._target_location, self._agent_location):
        #     self._target_location = self.np_random.integers(
        #         0, self.size, size=2, dtype=int
        #     )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        if hasattr(self, "server"):
            self.server.shutdown()   # stops serve_forever()
            self.server.server_close()
        if hasattr(self, "thread"):
            self.thread.join(timeout=1.0)

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


#vec_env = model.get_env()
#obs = vec_env.reset()
#for i in range(nSteps):
#    action, _state = model.predict(obs, deterministic=True)
#    obs, reward, done, info = vec_env.step(action)
    
    # vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

# try:
#     # optional check
#     check_env(env)

#     logger.info("Running... press Ctrl+C to stop.")

#     while True:
#         time.sleep(1)      # or your RL loop
#         action = env.action_space.sample()
#         env.step(action)

# except KeyboardInterrupt:
#     print("Stopping…")
#     env.close()
