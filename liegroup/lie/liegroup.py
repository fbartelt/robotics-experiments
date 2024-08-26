#%%
import warnings
import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from scipy.linalg import logm, expm
from liealgebra import o, sl, gl

class LieGroup(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def diff(self):
        pass

class GL(LieGroup, ABC):
    """ General Linear Group"""
    algebra_ = gl
    
    def __init__(self):
        pass

    @classmethod
    def random(cls, n:int|tuple, seed:int|None=None):
        rng = np.random.default_rng(seed=seed)
        dim = cls.algebra_.manifold_dim(n)
        omega = rng.random(dim)
        M = expm(cls.algebra_.S(omega, n))
        return M
    
    def jacobian(self):
        M = self.matrix.copy()
        E = np.eye(self.algebra.dim)
        W = []
        for mi in M.T:
            W_i = np.array([self.algebra.S(e, self.dim) @ mi for e in E.T]).T
            W.append(W_i)
        W = np.vstack(W)
        return W

class O(GL):
    """Indefinite Orthogonal Group O(n, m). For orthogonal group O(n), use m=0
    """
    algebra_ = o
    def __init__(self, n:int, m:int=0, matrix:np.ndarray=None):
        self.n = n
        self.m = m
        self.dim = (n, m)
        self.matrix = matrix if matrix is not None else O.random((n, m))
        self.algebra = o(n, m)
    
    def __array__(self):
        return self.matrix
    
    def __str__(self):
        group = f'({self.n}, {self.m})' if self.m else f'({self.n})'
        s = f'''Matrix in O{group} {'Indefinite'*bool(self.m)} Orthogonal Group.\n{self.matrix}'''
        return s
    
    def __repr__(self):
        return self.__str__()
    
    # def g(self, n:int=None, m:int=None) -> np.ndarray:
    #     match (n, m):
    #         case (None, None):
    #             n = self.n
    #             m = self.m
    #         case (None, _) | (_, None):
    #             raise ValueError('Both m and n need to be specified or both None')
    #         case _:
    #             pass
    #     return self.algebra.g(n, m)
    
    # def S(self, omega:np.ndarray, r:int|None=None) -> np.ndarray:
    #     return self.algebra.S(omega, self.n, self.m)

    def geodesic(self, M:Union[np.ndarray, 'O'], M2:Union[np.ndarray, 'O', None]=None):
        if M2 is None:
            M2 = np.array(M).copy()
            M = self.matrix.copy()
        return O.geodesic_(M, M2, self.n, self.m)
    
    @staticmethod
    def geodesic_(M1:Union[np.ndarray, 'O'], M2:Union[np.ndarray, 'O'], n:int, m:int):
        M1 = np.array(M1)
        M2 = np.array(M2)
        I = np.eye(n + m)
        dist = np.linalg.norm(I - np.linalg.inv(M2) @ M1, 'fro') ** 2
        # dist = np.linalg.norm((M2 - M1), 'fro') ** 2
        return dist
    
    @staticmethod
    def geodesic_diff(M1:Union[np.ndarray, 'O'], M2:Union[np.ndarray, 'O'], n:int, m:int, delta:float, wrt:int):
        M1 = np.array(M1)
        M2 = np.array(M2)
        curr_geodesic = O.geodesic_(M1, M2, n, m)
        r = n + m
        dgeo = np.zeros(M1.size)
        for i, _ in enumerate(M1.ravel()):
            row, col = divmod(i, M1.shape[0])
            delta_M = np.zeros(M1.shape)
            delta_M[row, col] = delta
            if wrt == 2:
                next_geodesic = O.geodesic_(M1, M2 + delta_M, n, m)
            else:
                next_geodesic = O.geodesic_(M1 + delta_M, M2, n, m)
            dgeo[col*r + row] = (next_geodesic - curr_geodesic) / delta
        return dgeo
    
    # def jacobian(self):
    #     M = self.matrix.copy()
    #     r = self.n + self.m
    #     k = int(r*(r - 1)/2)
    #     E = np.eye(k)
    #     W = []
    #     for mi in M.T:
    #         W_i = np.array([self.algebra.S(e, self.n, self.m) @ mi for e in E.T]).T
    #         W.append(W_i)
    #     W = np.vstack(W)
    #     return W
            
    def diff(self):
        pass

    # @staticmethod
    # def random(n, m, seed:int|None=None):
    #     X = o.g(n, m)
    #     r = n + m
    #     rng = np.random.default_rng(seed=seed)
    #     omega = 1*rng.random(int(r * (r - 1) / 2))
    #     M = expm(o.S(omega, n, m))
    #     return M

class SL(GL):
    """Special Linear Group SL(n).
    """
    algebra_ = sl
    def __init__(self, n:int, matrix:np.ndarray=None):
        self.n = n
        self.dim = n
        self.matrix = matrix if matrix is not None else SL.random(n)
        self.algebra = sl(n)
    
    def __array__(self):
        return self.matrix
    
    def __str__(self):
        s = f'''Matrix in SL({self.n}) Special Linear Group.\n{self.matrix}'''
        return s
    
    def __repr__(self):
        return self.__str__()

    def geodesic(self, M:Union[np.ndarray, 'SL'], M2:Union[np.ndarray, 'SL', None]=None):
        if M2 is None:
            M2 = np.array(M).copy()
            M = self.matrix.copy()
        return SL.geodesic_(M, M2, self.n)
    
    @staticmethod
    def geodesic_(M1:Union[np.ndarray, 'SL'], M2:Union[np.ndarray, 'SL'], n:int):
        M1 = np.array(M1)
        M2 = np.array(M2)
        I = np.eye(n)
        dist = np.linalg.norm(I - np.linalg.inv(M2) @ M1, 'fro') ** 2
        # dist = np.linalg.norm((M2 - M1), 'fro') ** 2
        return dist
    
    @staticmethod
    def geodesic_diff(M1:Union[np.ndarray, 'SL'], M2:Union[np.ndarray, 'SL'], n:int, delta:float, wrt:int):
        M1 = np.array(M1)
        M2 = np.array(M2)
        curr_geodesic = SL.geodesic_(M1, M2, n)
        dgeo = np.zeros(M1.size)
        for i, _ in enumerate(M1.ravel()):
            row, col = divmod(i, M1.shape[0])
            delta_M = np.zeros(M1.shape)
            delta_M[row, col] = delta
            if wrt == 2:
                next_geodesic = SL.geodesic_(M1, M2 + delta_M, n)
            else:
                next_geodesic = SL.geodesic_(M1 + delta_M, M2, n)
            dgeo[col*n + row] = (next_geodesic - curr_geodesic) / delta
        return dgeo
    
    # def jacobian(self):
    #     M = self.matrix.copy()
    #     k = self.algebra.size
    #     E = np.eye(k)
    #     W = []
    #     for mi in M.T:
    #         W_i = np.array([self.algebra.S(e, self.n) @ mi for e in E.T]).T
    #         W.append(W_i)
    #     W = np.vstack(W)
    #     return W
            
    def diff(self):
        pass

    # @staticmethod
    # def random(n, seed:int|None=None):
    #     rng = np.random.default_rng(seed=seed)
    #     omega = rng.random(int(n ** 2 - 1))
    #     M = expm(sl.S(omega, n))
    #     return M



# %%
n, m = 5,5
H = O(n, m)
Hd = O(n, m)
# H.matrix =  np.array([[1.2201,    0.8484  ,  1.0993],
#     [-0.0738 ,   1.4250   , 1.0178],
#     [0.7030 ,   1.3230 ,   1.8012]])
# Hd.matrix =  np.array([[8.3724 ,   5.8009  ,  1.9357],
#     [-5.5637 ,   8.3724 ,   1.0259],
#     [1.0256 ,   1.9359 ,  10.2372]])

dist = H.geodesic(Hd.matrix)
Wq = H.jacobian();
Wqd = Hd.jacobian()
dDdq = H.geodesic_diff(H.matrix, Hd.matrix, n, m, 1e-6, 1)
dDdqd = H.geodesic_diff(H.matrix, Hd.matrix, n, m, 1e-6, 2)

alpha = -1
print(f'dist: {dist}')
print(f'cond = {np.linalg.norm(dDdq @ Wq - (alpha * dDdqd @ Wqd))}')
# %%
n = 4
H = SL(n)
Hd = SL(n)
# H.matrix =  np.array([[1.2201,    0.8484  ,  1.0993],
#     [-0.0738 ,   1.4250   , 1.0178],
#     [0.7030 ,   1.3230 ,   1.8012]])
# Hd.matrix =  np.array([[8.3724 ,   5.8009  ,  1.9357],
#     [-5.5637 ,   8.3724 ,   1.0259],
#     [1.0256 ,   1.9359 ,  10.2372]])

dist = H.geodesic(Hd.matrix)
Wq = H.jacobian();
Wqd = Hd.jacobian()
dDdq = H.geodesic_diff(H.matrix, Hd.matrix, n, 1e-6, 1)
dDdqd = H.geodesic_diff(H.matrix, Hd.matrix, n, 1e-6, 2)

alpha = -1
print(f'dist: {dist}')
print(f'cond = {np.linalg.norm(dDdq @ Wq - (alpha * dDdqd @ Wqd))}')
# %%
