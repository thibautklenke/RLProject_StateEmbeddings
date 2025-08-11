import numpy as np
import cartpole_experiment as ce
import minigrid_unlock_experiment as mue
import random
import torch as th
from minatar.gym import register_envs
from stable_baselines3.common.utils import set_random_seed


def main() -> None:
    seeds = range(10)
    register_envs()  # Register minatar namespace for gymnasium

    set_random_seed(0)
    th.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Pretrain only once
    # mme.pretrain(0)
    mue.pretrain(0)
    # be.pretrain(0)
    ce.pretrain(0)
    # se.pretrain(0)
    # sem.pretrain(0)

    for SEED in seeds:
        print(SEED)
        set_random_seed(SEED)
        th.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # mme.train(SEED)
        mue.train(SEED)
        # be.train(SEEd)
        ce.train(SEED)
        # se.train(SEED)
        # sem.train(SEED)


if __name__ == "__main__":
    main()
