# RL OSC Environment 
This Project creates a simple OSC (Open Sound Control) Reinforcement Learning Environment.


## Install
```
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run:
- in a terminal, with the activated virtual environment run a learning session via `python rlEnv.py --inport 5000 --outport 3030`
- Use any OSC-capable program (pd, max) to send observations and receive actions.
