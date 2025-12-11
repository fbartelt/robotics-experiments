# %%
import numpy as np
import smoothfunctions as sf
from distances import holder_mean

# Test all functions in smoothfunctions C++ module
def test_holder():
    a = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    b = np.array([1.0, 2.0])
    r = 0.1
    pyres1, pygrad1 = holder_mean(a, r=r, compute_gradient=True)
    pyres2, pygrad2 = holder_mean(b, r=r, compute_gradient=True)
    res1 = sf.holderMean(a, r=r)
    res2 = sf.holderMean(b, r=r)
    assert np.allclose(res1, 0.0)
    assert np.allclose(res2, 1.0, rtol=1e-3)
    grad1 = sf.holderMeanGradient(a, r=r)
    grad2 = sf.holderMeanGradient(b, r=r)
    print("Holder Mean Results:", grad1, grad2)
    print("Holder Mean Py Results:", pygrad1, pygrad2)

test_holder()




