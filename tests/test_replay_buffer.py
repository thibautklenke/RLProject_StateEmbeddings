from state_embedding.replay_buffer import ContextualizedReplayBuffer
from gymnasium.spaces import Box, Discrete

import numpy as np

buffer_size = 20
window_size = 5
obs_space = Box(low=-1.0, high=1.0, shape=(2, 2), dtype=np.float32)
action_space = Discrete(4)

def init_buffer(n_envs: int = 1) -> ContextualizedReplayBuffer:
    return ContextualizedReplayBuffer(
        buffer_size=buffer_size,
        observation_space=obs_space,
        action_space=action_space,
        handle_timeout_termination=False,
        window_size=window_size,
        n_envs=n_envs,
    )

def test_sanity():
    n_envs = 2
    buffer = init_buffer(n_envs)

    samples = buffer_size // n_envs

    assert buffer.buffer_size == samples
    assert buffer.observations.shape[0] == samples
    assert buffer.contexts.shape[0] == samples

    assert buffer._context_index.shape == (n_envs, )
    assert buffer._context_tracker.shape == (window_size, n_envs, *obs_space.shape)

def test_single_observation_stored():
    buffer = init_buffer()

    obs = obs_space.sample()
    obs_next = obs_space.sample()
    action = action_space.sample()
    buffer.add(obs, obs_next, action, np.zeros(1), [False], [])

    assert (buffer.observations[0, 0] == obs).all()
    assert (buffer.next_observations[0, 0] == obs_next).all()
    assert (buffer.actions[0, 0] == action).all()

def test_no_ref():
    buffer = init_buffer()

    obs = obs_space.sample()
    obs_next = obs_space.sample()
    action = action_space.sample()
    buffer.add(obs, obs_next, action, np.zeros(1), [False], [])

    assert (buffer.contexts[0, 0] == obs).all()
    # Check that the context is not a reference to the observation
    obs[0, 0] = 100  # Modify the observation
    assert not (buffer.contexts[0, 0] == obs).all()

def test_single_discrete_observation():
    discrete_space = Discrete(10)
    buffer = ContextualizedReplayBuffer(
        buffer_size=buffer_size,
        observation_space=discrete_space,
        action_space=action_space,
        handle_timeout_termination=False,
        window_size=window_size,
        n_envs=2,
    )

    obs = np.array([discrete_space.sample(), discrete_space.sample()])
    obs_next = np.array([discrete_space.sample(), discrete_space.sample()])
    action = np.array([action_space.sample(), action_space.sample()])
    buffer.add(obs, obs_next, action, np.zeros(2), [False, True], [])

    assert (buffer.observations[0].squeeze() == obs).all()
    assert (buffer.next_observations[0].squeeze() == obs_next).all()
    assert (buffer.actions[0].squeeze() == action).all()

def test_rollover():
    # Init environment
    buffer = init_buffer()

    # Fill buffer with window_size + 2 observations
    # and save last window_size observations in local variable
    for i in range(2 * window_size):
        obs = obs_space.sample()
        obs_next = obs_space.sample()
        action = action_space.sample()
        buffer.add(obs, obs_next, action, np.zeros(1), [False], [])

        correct_context = buffer.observations[max(buffer.pos - window_size, 0):buffer.pos]
        correct_context = np.pad(correct_context, ((0, max(0, window_size - correct_context.shape[0])), (0, 0), (0, 0), (0, 0) ), mode='constant', constant_values=0)
        assert (buffer.contexts[i] == correct_context).all()

# TODO: test n_envs > 1 and done signals at different times

def test_simple_oversample():
    buffer = init_buffer()

    obs = obs_space.sample()
    obs_next = obs_space.sample()
    action = action_space.sample()
    buffer.add(obs, obs_next, action, np.zeros(1), [False], [])

    bs = 5
    replay_data, contexts = buffer.sample_with_context(batch_size=bs)
    assert (replay_data.observations == np.stack([obs] * bs)).all()
    assert (replay_data.next_observations == np.stack([obs_next] * bs)).all()
    assert (replay_data.actions == np.stack([action] * bs)).all()

    zero = np.zeros_like(obs)
    obs_context = np.stack(np.stack([obs] + [zero]*4))
    assert (contexts == np.stack([obs_context] * bs)).all()

    
    