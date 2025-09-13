import time
from utils.loggin_config import LogManager
from functools import wraps

def timed(func):
    """Decorator that logs the execution time of the function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LogManager.logger.info(f"{func.__name__} took {elapsed:.4f} s")
        return result
    return wrapper