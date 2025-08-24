"""Utility functions for the TidyBot environment."""

import numpy as np


def get_rng(seed=None):
    """Get a random number generator (RNG) based on the provided seed.

    Args:
        seed (int, optional): Seed for the RNG. If None, a default RNG is used.

    Returns:
        np.random.Generator: A NumPy random number generator.
    """
    if seed is not None:
        assert isinstance(seed, int)
        assert seed >= 0, "Seed must be a non-negative integer."

        seed_seq = np.random.SeedSequence(entropy=seed)
        rng = np.random.Generator(np.random.PCG64(seed_seq.entropy))
        return rng
    return np.random.default_rng()
