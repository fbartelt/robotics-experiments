# %%
import numpy as np
import smoothfunctions as sf
from distances import holder_mean, smooth_min, smooth_max


# Test all functions in smoothfunctions C++ module
def test_holder():
    # a = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    # b = np.array([1.0, 2.0])
    a, b = 0.0, 1.0
    c, d = 1.5, 2.0
    r = 0.1
    pyres1, pygrad1 = holder_mean([a, b], r=r, compute_gradient=True)
    pyres2, pygrad2 = holder_mean([c, d], r=r, compute_gradient=True)
    res1 = sf.holderMean(a, b, r=r)
    res2 = sf.holderMean(c, d, r=r)
    assert np.allclose(res1, pyres1)
    assert np.allclose(res2, pyres2)
    grad1 = sf.holderMeanGradient(a, b, r=r)
    grad2 = sf.holderMeanGradient(c, d, r=r)
    assert np.allclose(grad1, pygrad1)
    assert np.allclose(grad2, pygrad2)


def test_smooth_min():
    a = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    b = np.array([1.0, 2.0])
    c = np.array([-10.0, -5.0, 0.0, 5.0, 10.0, -2.0])
    r = 0.1
    pyres1, pygrad1 = smooth_min(a, r=r, compute_gradient=True)
    pyres2, pygrad2 = smooth_min(b, r=r, compute_gradient=True)
    pyres3, pygrad3 = smooth_min(c, r=r, compute_gradient=True)
    res1 = sf.smoothMinList(a, r=r)
    res2 = sf.smoothMinList(b, r=r)
    res3 = sf.smoothMinList(c, r=r)
    assert np.allclose(res1, pyres1)
    assert np.allclose(res2, pyres2)
    assert np.allclose(res3, pyres3)
    grad1 = sf.smoothMinListGradient(a, r=r)
    grad2 = sf.smoothMinListGradient(b, r=r)
    grad3 = sf.smoothMinListGradient(c, r=r)
    print(f"grad1: {grad1}. Pygrad1: {pygrad1}")
    assert np.allclose(grad1, pygrad1, rtol=1e-3)
    assert np.allclose(grad2, pygrad2, rtol=1e-3)
    assert np.allclose(grad3, pygrad3, rtol=1e-3)


def test_smooth_max():
    a = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    b = np.array([1.0, 2.0])
    c = np.array([10.0, -5.0, 0.0, 11.0, 5.0, 10.0])
    r = 0.1
    pyres1, pygrad1 = smooth_max(a, r=r, compute_gradient=True)
    pyres2, pygrad2 = smooth_max(b, r=r, compute_gradient=True)
    pyres3, pygrad3 = smooth_max(c, r=r, compute_gradient=True)
    res1 = sf.smoothMaxList(a, r=r)
    res2 = sf.smoothMaxList(b, r=r)
    res3 = sf.smoothMaxList(c, r=r)
    assert np.allclose(res1, pyres1)
    assert np.allclose(res2, pyres2)
    assert np.allclose(res3, pyres3)
    grad1 = sf.smoothMaxListGradient(a, r=r)
    grad2 = sf.smoothMaxListGradient(b, r=r)
    grad3 = sf.smoothMaxListGradient(c, r=r)
    print(f"grad3: {grad3}. Pygrad3: {pygrad3}")
    try:
        assert np.allclose(grad1, pygrad1, rtol=1e-3)
    except AssertionError:
        print(f"res1: {res1}, pyres1: {pyres1}")
        print(f"grad1: {grad1}, pygrad1: {pygrad1}")
        raise
    try:
        assert np.allclose(grad2, pygrad2, rtol=1e-3)
    except AssertionError:
        print(f"res2: {res2}, pyres2: {pyres2}")
        print(f"grad2: {grad2}, pygrad2: {pygrad2}")
        raise
    try:
        assert np.allclose(grad3, pygrad3, rtol=1e-3)
    except AssertionError:
        print(f"res3: {res3}, pyres3: {pyres3}")
        print(f"grad3: {grad3}, pygrad3: {pygrad3}")
        raise


test_holder()
print("Holder mean tests passed.")
test_smooth_min()
print("Smooth min tests passed.")
test_smooth_max()
print("Smooth max tests passed.")
