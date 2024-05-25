import warnings
import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import logm, expm

class LieAlgebra(ABC):
    def __init__(self):
        pass

class sl(LieAlgebra, ABC):
    def __init__(self):
        pass

class o(sl):
    """Indefinite orthogonal lie algebra"""
    def __init__(self,):
        pass