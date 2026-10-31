import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def window_segment(x, T, S):
    """Return complete 1D windows; T and S are measured in samples.

    S must be a positive integer. Any trailing samples that cannot form
    another complete window at the requested stride are left unused.
    """
    if isinstance(S, (bool, np.bool_)) or not isinstance(S, (int, np.integer)) or S <= 0:
        raise ValueError("S must be a positive integer")

    x = np.asarray(x) # make x an array
    n = x.shape[0] # determine number of samples in x
    if n < T: # return no windows if even one complete window cannot fit
        print(f"Number of samples {n} is smaller than the window size of {T}")
        return np.empty((0, T), dtype=x.dtype)
        
    return sliding_window_view(x, window_shape=T)[::S] # windows share the underlying array's data buffer
    
