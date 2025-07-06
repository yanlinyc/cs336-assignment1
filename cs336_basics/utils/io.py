import os
import pickle


__all__ = ["load_pickle", "save_pickle"]


def save_pickle(path: str | os.PathLike, data):
    """Save an object to a pickle file."""
    if not isinstance(path, str) and not isinstance(path, os.PathLike):
        raise TypeError("Path must be a string or os.PathLike object.")

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | os.PathLike):
    """Load an object from a pickle file."""

    with open(path, "rb") as f:
        return pickle.load(f)
