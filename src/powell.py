import numpy as np 
from .line_search import gss


def powell(f, x_0: np.ndarray, direction_set: np.ndarray, tol: float) -> float:

    x = x_0

    while f(x) < tol:

        temp = x
        for i in range(0, direction_set.size, 1):
            phi = lambda alpha: f(temp + alpha * direction_set[i])
            temp = gss(phi, 0, 100)
        
        net_displacement = temp - x
        # add net displacement to direction set
        x = temp 
    
    return x