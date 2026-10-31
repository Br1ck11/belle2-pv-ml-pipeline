from decimal import Decimal



def split_time_grid(grid_ns, mode, train_fraction):
    """Partition a shared time grid for training/validation or testing.

    In "train_val" mode, assign the first floor(train_fraction * N) grid
    points to training and the remaining points to validation, where N is
    the grid length. Interpret the fraction through Decimal(str(...)) to
    avoid binary floating-point errors in the split count. The fraction
    refers to grid points before windowing or filtering, not raw PV counts.
    In "test" mode, return the entire grid without partitioning it.

    Args:
        grid_ns: One-dimensional int64 NumPy array containing a shared,
            chronological, regular grid of UTC Unix timestamps in
            nanoseconds, such as the output of make_time_grid. The caller
            is responsible for this format; it is not validated here.
        mode: Either "train_val" or "test".
        train_fraction: Training proportion strictly between zero and one
            in "train_val" mode. This argument must be supplied but is
            ignored in "test" mode.

    Returns:
        In "train_val" mode, a dictionary with "train" and "val" arrays.
        Both are nonempty and together contain every input grid point
        exactly once in its original order. In "test" mode, a dictionary
        with a "test" array containing the full grid, which may be empty.

    Raises:
        ValueError: If mode is unknown or, in "train_val" mode, the fraction
            is outside the open interval (0, 1) or either split is empty.

    Notes:
        The function does not modify the input. Training and validation
        arrays are views sharing the input's storage; the test array is
        the input object itself. Mutating a returned array therefore also
        changes the corresponding entries in the original grid.

        The first validation timestamp defines the shared boundary:
        training sources must precede it, while validation sources may
        include it. Apply these bounds to every PV's source measurements
        before sampling each split.
    """
    
    if mode == "test":
        return {
        "test": grid_ns,
        }
    
    elif mode == "train_val":
        if not 0 < train_fraction < 1: # for no splitting use mode = "test"
            raise ValueError("train_fraction must be a float between 0 and 1")
    
        split_index = int(Decimal(str(train_fraction)) * len(grid_ns)) # Round down convention such that training doesn't allocate more than the requested fraction (this is a choice)
        
        if not 0 < split_index < len(grid_ns):
            raise ValueError("The split must leave at least one grid point in each part")
        
        # Allocate first data points of grid_ns up to split_index to train set and remaning ones to val set
        train_grid_ns = grid_ns[:split_index]
        val_grid_ns = grid_ns[split_index:]
        
        return {
        "train": train_grid_ns,
        "val": val_grid_ns,
        }
        
    else:
        raise ValueError("Select a valid mode: 'test' or 'train_val'")
    
    
    
