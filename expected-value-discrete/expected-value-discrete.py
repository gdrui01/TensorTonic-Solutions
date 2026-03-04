import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if np.allclose(1, sum(p)):
        e_val = 0
        for i, el in enumerate(x):
            e_val += el*p[i]
        return e_val

    else:
        raise ValueError
            
