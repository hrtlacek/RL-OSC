# RL OSC 
This Project creates a simple OSC (Open Sound Control) Reinforcement Learning Environment and an Agent that learns in it in real time. Actions and Observations are sent/received via UDP / OSC. 

## Install
```
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
## Tests
Tested on linux only.

To run simple tests, make sure the requirements are installed and the environment is activated. Then:
`python -m pytest`

## Run:

- Open the example max patch.
- in a terminal, with the activated virtual environment run a learning session via `python rlEnv.py -nI 2 -nO 2 -dt 1`.
- The agent should learn how to follow the target.
- Action space is currently normalized to a -1 to 1 range. Observations are normalized to a 0 to 1 range.

By default, reward needs to be sent as well as observations. There is a flag `-r` that calculates reward from the sum of observations and makes it unnecessary to send reward externally.

## Inference
Currently there is no possibility to run a trained network. This would be easy to do but this project is more interested in the learning and observing exploration in real time.



## Acknowledgements
Made in The Project "Spirits in Comnplexity", funded by the Austrian Science Fund [10.55776/AR821].


## Todos
- Document default adresses in readme.
- shebang & rights.
- manage requirements better
- add pytest to requirements.
- write actual test with python server
- offer more models
