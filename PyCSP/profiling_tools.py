import time
from functools import wraps

def profile_cpu_time(attribute_name, log=False):
    """A decorator to profile the CPU time of a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.process_time()
            result = func(self, *args, **kwargs)
            elapsed_time = time.process_time() - start_time
            if not hasattr(self, attribute_name):
                setattr(self, attribute_name, 0.0)
            setattr(self, attribute_name, getattr(self, attribute_name) + elapsed_time)
            if log:
                print(f"Function {func.__name__} took {elapsed_time:.6f} seconds.")
            return result
        return wrapper
    return decorator

import time
from functools import wraps

def profile_cpu_time_and_count(time_attribute, count_attribute, log=False):
    """
    A decorator to profile the CPU time and count the number of calls to a function.
    
    Args:
        time_attribute (str): The attribute name to store the total CPU time.
        count_attribute (str): The attribute name to store the call count.
        log (bool): Whether to log the time for each call.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Start CPU time measurement
            start_time = time.process_time()
            
            # Execute the original function
            result = func(self, *args, **kwargs)
            
            # Stop CPU time measurement
            elapsed_time = time.process_time() - start_time
            
            # Update the total CPU time
            if not hasattr(self, time_attribute):
                setattr(self, time_attribute, 0.0)
            current_time = getattr(self, time_attribute)
            setattr(self, time_attribute, current_time + elapsed_time)
            
            # Update the call count
            if not hasattr(self, count_attribute):
                setattr(self, count_attribute, 0)
            current_count = getattr(self, count_attribute)
            setattr(self, count_attribute, current_count + 1)
            
            # Optionally log the time for this call
            if log:
                print(f"Function {func.__name__} call #{current_count + 1} took {elapsed_time:.6f} seconds.")
            
            return result
        return wrapper
    return decorator
