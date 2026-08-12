import numpy as np
import random

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    dropout_pattern = rng.choice(
        [0.0, 1.0 / (1 - p)],
        size=x.shape,
        p=[p, 1 - p]
    )

    output = x * dropout_pattern

    return output, dropout_pattern
