import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from scipy.linalg import logm, expm

class LieAlgebra(ABC):
    def __init__(self):
        pass

class gl(LieAlgebra, ABC):
    def __init__(self):
        pass

class o(gl):
    """Indefinite orthogonal Lie algebra"""
    def __init__(self, n:int, m:int=0):
        self.n = n
        self.m = m
        self.dim = int((n + m) * (n + m - 1) / 2)
        self.size = n + m

    @staticmethod
    def g(n:int, m:int) -> np.ndarray:
        g = np.block([
                    [np.eye(n), np.zeros((n, m))],
                    [np.zeros((m, n)), -np.eye(m)]
                ])
        return g
    
    @staticmethod
    def S(omega:np.ndarray, n:int|tuple, m:int|None=None) -> np.ndarray:
        if m is None:
            if isinstance(n, tuple):
                n, m = n
            else:
                raise ValueError('n should be a tuple (n, m) if m is None')
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
        S_ = np.diag(omega[:r-1], 1)
        el = k - (r - 1);
        diag_num = 2
        # while el:
        #     i = k - el
        #     diag_len = len(np.diag(S_, diag_num))
        #     if el >= diag_len:
        #         S_ += np.diag(omega[i: i + diag_len], diag_num);
        #         el -= diag_len;
        #     else:
        #         raise ValueError('AAAA')
        #     #     aux = [omega(i+1:end) zeros(1, diag_len - el)];
        #     #     S += np.diag(aux, diag_num);
        #     #     el -= el
        #     diag_num = diag_num + 1;
        triu_indices = np.triu_indices(r, k=1)
        S_[triu_indices] = omega[:len(triu_indices[0])]
        X = o.g(n, m)
        S_ = X @ (S_ - S_.T)
        return S_
    
    @staticmethod
    def manifold_dim(n:tuple):
        r = n[0] + n[1]
        return int(r * (r - 1) / 2)
    
class sl(gl):
    """Special Linear Lie algebra"""
    def __init__(self, n:int):
        self.n = n
        self.dim = int(n**2 - 1)
        self.size = n
    
    @staticmethod
    def S(omega:np.ndarray, n:int) -> np.ndarray:
        omega = omega.copy().ravel()
        k = len(omega)
        test_k = sl.manifold_dim(n)
        if k < test_k:
            omega = np.pad(omega, (0, test_k - k))
            warnings.warn(f'omega should be {test_k}-dimensional. Padding with zeros')
        elif k > test_k:
            raise ValueError(f'omega should be {test_k}-dimensional [n**2 - 1]. ')
        S_ = np.zeros((n, n))
        triu_indices = np.triu_indices(n, k=1)
        tril_indices = np.tril_indices(n, k=-1)
        S_[triu_indices] = omega[:int(n*(n-1)/2)]
        S_[tril_indices] = omega[int(n*(n-1)/2) : int(n*(n-1))]
        diag_ = omega[int(n*(n-1)):]
        diag_ = np.hstack([diag_, -np.sum(diag_)])
        S_ += np.diag(diag_)
        return S_

    @staticmethod
    def manifold_dim(n:int) -> int:
        return int(n ** 2 - 1)