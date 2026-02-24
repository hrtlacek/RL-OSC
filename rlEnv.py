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
parser.add_argument('--outport', type=int, default=3000)
parser.add_argument('-v','--verbose', action='store_true', help='Verbose mode.')
parser.add_argument('-dt','--deltatime', type=int, help='Time delay between steps in milliseconds.')
parser.add_argument('-Ni', '--numInput', type=int, default=8, help='Input array size (dimensionality of one observation)')
parser.add_argument('-No', '--numOutput',type=int, default=8, help='Output array size (dimensionality of one action)')

parser.add_argument('-vv','--vverbose', action='store_true', help='Very verbose mode.')



args = parser.parse_args()
INPORT = args.inport
OUTPORT = args.outport
verbose = args.verbose

vverbose = args.vverbose



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
    modelVerbosity = 1
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
    def __init__(self, inport=5000, outport=3030):
        
        self.IN_PORT = inport
        self.IN_IP = '0.0.0.0'
        self.IN_ADDR = "/toRLosc"

        self.OUT_IP = '127.0.0.1'
        self.OUT_PORT = outport
        self.OUT_ADDR = "/fromRLosc"

        self.size = 8

        self.last_obs = np.zeros(self.size, dtype=np.float32)

        self.disp = dispatcher.Dispatcher()
        self.disp.map(self.IN_ADDR, self.handle_osc_input)
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

        self.action_space = gym.spaces.Box(low=-1,high=1, shape=(8,), dtype=np.float32)

    def handle_osc_input(self, addr, *args):
        logger.debug(args)
        data = np.array(args[:self.size],dtype=np.float32)
        logger.debug(f"data received: {data}")
        self.last_obs = data


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
                self._target_location
        }

    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction
        # direction = self._action_to_direction[action]

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + action, -1, 1
        )
        # print(self._agent_location)
        # print(type(self._agent_location))
        # self.client.send_message(self.OUT_ADDR, self._agent_location)
        self.act(self._agent_location)
        # Check if agent reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        terminated = False

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        
        #ownLoudness = self.last_obs[1]
        #otherLoudness = self.last_obs[0]

        #reward = float(abs(ownLoudness/(self.last_obs[0]+1e-6)))
        # reward = float(1/((np.sum(self.last_obs)) +1e-5)) #1 if terminated else 0
        reward = float(np.sum(np.array(self.last_obs)))
        
        self.client.send_message("/reward", reward)
        logger.info(f"Reward: {reward}")

        observation = self._get_obs()
        logger.debug(f"Observation: {observation}")
        info = self._get_info()
        time.sleep(0.1)
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
        super().reset(seed=seed)

        # Randomly place the agent anywhere in the space
        self._agent_location = self.np_random.random(size=self.size, dtype=np.float32)

        self._target_location = self.np_random.random(size=self.size, dtype=np.float32)
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




env = OscEnv(inport=INPORT, outport=OUTPORT)
obs, _ = env.reset()




# env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MultiInputPolicy", env, verbose=modelVerbosity)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    
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
