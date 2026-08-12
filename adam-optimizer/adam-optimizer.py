import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    
    # Write code here
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    m_new = beta1 * m + (1-beta1)*grad
    v_new = beta2 * v + (1 - beta2)*pow(grad,2)
    m_bias = m_new /(1 -pow(beta1,t)) 
    v_bias = v_new /(1 - pow(beta2,t))
    param_new = param - lr*m_bias/(v_bias**(1.0/2) + eps)
    return (param_new, m_new, v_new)