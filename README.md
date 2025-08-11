# RLProject_StateEmbeddings ☄️

A research framework for learning and evaluating state embeddings in reinforcement learning (RL) environments. This project provides tools for pretraining, training, and evaluating RL agents with learned state representations, including transformer-based embeddings and reconstruction objectives.

## Features 🪐

- **Transformer-based State Embeddings:** Learn compact representations of environment states using transformer encoders.
- **Reconstruction Objectives:** Jointly train RL agents with state reconstruction losses.
- **Flexible Environment Wrappers:** Context and embedding wrappers for stacking observations and using learned embeddings.
- **Pretraining & Evaluation:** Pretrain embeddings with various objectives and evaluate with linear probes.
- **Support for Multiple Environments:** Includes experiments for CartPole, MinAtar Breakout, and MiniGrid Unlock.

## Repository Structure 🌜

```
RLProject_StateEmbeddings/ 
│
├── src/
│   ├── main.py                                 # Entry point for running experiments
│   ├── breakout_experiment.py                  # Breakout (MinAtar) experiment routines
│   ├── cartpole_experiment.py                  # CartPole experiment routines
│   ├── minigrid_memory_experiment.py           # MiniGrid Memory experiment routines
│   ├── minigrid_unlock_experiment.py           # MiniGrid Unlock experiment routines
│   ├── seaquest_experiment.py                  # MinAtar Seaquest experiment routines
│   ├── seaquestmarkov_unlock_experiment.py     # MinAtar Seaquest_alt experiment routines
│   ├── state_embedding/
│       ├── env.py                              # Context and embedding environment wrappers
│       ├── embedding.py                        # StateEmbedding and StateDecoder modules
│       ├── embedding_eval.py                   # Linear probe for embedding evaluation
│       ├── train.py                            # Pretraining routines for embeddings
│       ├── callbacks.py                        # Custom RL training callbacks
│       └── dqn/
│           ├── dqn.py                          # DQN with reconstruction loss
│           └── qnetwork.py                     # Q-network with embedding and reconstruction
│   └── envs/
│       └── seaquest_markov.py                  # Implementation of an alternative Seaquest env
│
└── README.md                                   # This file
```

## Installation 🌌

This project uses [uv](https://github.com/astral-sh/uv) for Python package management and virtual environments.

1. **Clone this repo:**
    ```bash
    git clone https://github.com/thibautklenke/RLProject_StateEmbeddings
    cd RLProject_StateEmbeddings
    ```

2. **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    uv sync
    ```

## Usage 🥢

### Running Experiments

The main entry point is `src/main.py`, which manages pretraining and training for all supported environments.

To run all experiments (pretraining and training for each environment and seed):

```bash
uv run main
```

- By default, this will run pretraining once per environment, then train agents with different seeds.
- You can comment/uncomment lines in `main.py` to select which environments to run.

### Customizing Experiments 

- **Add new environments:** Create a new `*_experiment.py` file following the structure of `cartpole_experiment.py`.
- **Change embedding architecture:** Modify `state_embedding/embedding.py` or pass different `embedding_kwargs` in experiment files.
- **Adjust training parameters:** Edit variables like `n_pretrain`, `n_train`, or `net_arch` in the experiment scripts.


---