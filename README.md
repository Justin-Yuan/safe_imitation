# safe_imitation

safe imitation learning

## Installation

Install via `setup.py` as developer mode. Once pip installed, module import can start from module `safe_il`, i.e. `import safe_il.xxx.xxx`. any changes in the `safe_il` folder will also be reflected.

```bash
pip install -e .
```

For Gym-Minigrid (Currently Not Used!):
Install the gym-minigrid dependency and matplotlib prior to setting up the main project

```bash
cd dependencies/gym-minigrid
pip install -e .
pip install matplotlib
```

## Examples

Use the gym API to create and step through the environments. You can find the 
environment ID's in the __init__.py file

```python
import gym
gym.make('safe_il:BoneDrilling2D-v0')
```

suppose there's a main script (using stuff in `safe_il`)

```bash
python scripts/main.py
```

## Structure

- `safe_il` folder keeps all python source scripts
- `scripts` folder keeps all runnable main scripts (python or bash)
- `results` folder keeps all experimental results (added to `.gitignore` so not present in github)

## References

1. previous repo [link](https://github.com/StafaH/graph-imitation-learning)
2. safety gym [link](https://github.com/openai/safety-gym)
3. recovery rl [link](https://github.com/abalakrishna123/recovery-rl)
