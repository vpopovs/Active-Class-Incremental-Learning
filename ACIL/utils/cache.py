import os
import pickle


def cache_file(func):
    """Decorator to cache the result of a function in a file."""

    def wrapper(*args, **kwargs):
        """Wrapper function to cache the result of a function in a file."""
        if isinstance(args[0], dict):
            log_or_print = args[0].get("log").info if args[0].get("log") else print
            cache = args[0].get("cache", None)
        else:
            log_or_print = args[0].log.info if hasattr(args[0], "log") else print
            cache = getattr(args[0], "cache", None)

        if not cache:
            log_or_print("No cache attribute found")
            return func(*args, **kwargs)
        parent_dir = os.path.dirname(os.path.dirname(cache))
        if parent_dir == "data" and not any(parent_dir.split("data")[1:]):
            log_or_print("Cache path cannot be in the data folder")
            return func(*args, **kwargs)

        if not os.path.exists(cache):
            result = func(*args, **kwargs)
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "wb") as f:
                pickle.dump(result, f)
            log_or_print(f"Cache saved at {cache}")
        else:
            with open(cache, "rb") as f:
                result = pickle.load(f)
            log_or_print(f"Cache loaded from {cache}")
        return result

    return wrapper
