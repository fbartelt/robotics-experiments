#%%
import warnings
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import logm, expm

class LieGroup(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def diff(self):
        pass

class GL(LieGroup, ABC):
    """ General Linear Group"""
    def __init__(self):
        pass

class O(GL):
    """Indefinite Orthogonal Group O(n, m). For orthogonal group, use m=0
    """
    def __init__(self, n:int, m:int=0, matrix:np.ndarray=None):
        self.n = n
        self.m = m
        self.matrix = matrix if matrix is not None else O.random(n, m)

    @classmethod
    def g_class(cls, n:int, m:int) -> np.ndarray:
        g = np.block([
                    [np.eye(n), np.zeros((n, m))],
                    [np.zeros((m, n)), -np.eye(m)]
                ])
        return g
    
    def g(self, n:int=None, m:int=None) -> np.ndarray:
        match (n, m):
            case (None, None):
                n = self.n
                m = self.m
            case (None, _) | (_, None):
                raise ValueError('Both m and n need to be specified or both None')
            case _:
                pass
        return self.g_class(n, m)
    
    @classmethod
    def S_class(cls, omega:np.ndarray, n:int, m:int) -> np.ndarray:
        omega = omega.copy().ravel()
        k = len(omega)
        r = m + n
        test_k = int(r * (r - 1) / 2)
        if k < test_k:
            omega = np.pad(omega, (0, test_k - k))
            warnings.warn(f'omega should be {test_k}-dimensional. Padding with zeros')
        elif k > test_k:
            raise ValueError(f'omega should be {test_k}-dimensional [(n+m)*(n+m-1)/2]. ')
        S_ = np.zeros((r, r))
        triu_indices = np.triu_indices(r, k=1)
        S_[triu_indices] = omega[:len(triu_indices[0])]
        X = cls.g_class(n, m)
        S_ = X @ (S_ - S_.T)
        return S_
    
    def S(self, omega:np.ndarray, r:int|None=None) -> np.ndarray:
        return O.S_class(omega, self.n, self.m)

    def diff(self):
        pass

    @staticmethod
    def random(n, m, seed:int=42):
        X = O.g_class(n, m)
        r = n + m
        rng = np.random.default_rng(seed=seed)
        omega = 10*rng.random(int(r * (r - 1) / 2))
        M = expm(X @ O.S_class(omega, n, m))
        return M


# %%
