import os
import pickle


__all__ = ["load_pickle", "save_pickle"]


def save_pickle(path: str | os.PathLike, data):
    """Save an object to a pickle file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str | os.PathLike):
    """Load an object from a pickle file."""

    with open(path, "rb") as f:
        return pickle.load(f)
