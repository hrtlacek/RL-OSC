# RL Environment + RAVE Example
The example uses 2 RAVE models so it needs quite a bit of CPU/GPU resources.

## Install
```
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run:
- open pd patch `testPdEnv.pd`. Uses `[nn~]` and was only tested using `plugdata`, so might use additional non-vanilla objects.
- in a terminal, with the activated virtual environment run a learning session via `python rlEnv.py --inport 5000 --outport 3030`
