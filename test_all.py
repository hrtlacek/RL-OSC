import pytest
import RLlib
import numpy as np


def test_util():
    import util

#def test_importRLlib():
#    import RLlib


def clacRewardForTest(agentPos, targetPos):
    diff = agentPos - targetPos
    return float(np.linalg.norm(diff))


def test_converge():
    env = RLlib.OscEnv(inport=5000, outport=3030, nObserv=2, nAction=2)
    targetPos = [0.2, 0.3]
    


    disp = dispatcher.Dispatcher()
    disp.map(self.IN_ADDR, self.handle_osc_input)
    client = SimpleUDPClient(self.OUT_IP, self.OUT_PORT)

    model = A2C("MultiInputPolicy", env, verbose=modelVerbosity)
   




