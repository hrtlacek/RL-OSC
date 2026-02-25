# RL OSC 
This Project creates a simple OSC (Open Sound Control) Reinforcement Learning Environment and an Agent that learns in it in real time. Actions and Observations are sent/received via UDP / OSC. 

## Install
```
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run:
- in a terminal, with the activated virtual environment run a learning session via `python rlEnv.py --inport 5000 --outport 3030`
- Use any OSC-capable program (pd, max) to send observations and receive actions.

- Action space is currently normalized to a -1 to 1 range. Observations are normalized to a 0 to 1 range.


## Acknowledgements
Made in The Project "Spirits in Comnplexity", funded by the Austrian Science Fund [10.55776/AR821].


## Todos
- Warnings for data received in wrong formats (wrong list lengths specifically)
- Document default adresses in readme.
- shebang & rights.
