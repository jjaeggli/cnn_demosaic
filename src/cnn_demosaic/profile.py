import functools
import logging
import time


def profile(logger: logging.Logger | None = None, level=logging.DEBUG):
    def decorator(func):
        # Use functools.wraps to preserve the original function's name, docstring, etc.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine which logger instance to use
            log_instance = logger if logger is not None else logging.getLogger(func.__module__)

            start_time = time.perf_counter()

            try:
                # Execute the original function
                result = func(*args, **kwargs)
            finally:
                # Ensure profiling and logging happen even if the function raises an error
                end_time = time.perf_counter()
                duration = end_time - start_time

                # Log the duration
                log_instance.log(level, "'%s' executed in %.4f seconds", func.__name__, duration)

            return result

        return wrapper

    return decorator
