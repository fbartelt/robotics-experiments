# %%
import numpy as np

def holder_mean(values, r=0.1):
    v = np.array(values) + 1e-10
    if np.any(v < 0):
        raise ValueError("All values must be non-negative")
    # Raise error if values is a list of lists or nested arrays
    if v.ndim > 1:
        raise ValueError("Input must be a 1D array or list")

    raised = list(map(lambda x: x**(-1/r), v))
    res = np.sum(raised)**(-r)
    return res

def holder_mean_derivative(values, r=0.1):
    v = np.array(values) + 1e-10
    if np.any(v < 0):
        raise ValueError("All values must be non-negative")
    # Raise error if values is a list of lists or nested arrays
    if v.ndim > 1:
        raise ValueError("Input must be a 1D array or list")

    outer_der = np.array(list(map(lambda x: x**(-1/r), v)))
    outer_der = -r * (np.sum(outer_der)**(-r - 1))
    inner_der = np.array(list(map(lambda x: -1/r * (x**(-1/r - 1)), v)))
    return outer_der * inner_der

def _smooth_min_two_elements(x, y, r=0.1):
    if x >= 0 and y >= 0:
        return holder_mean([x, y], r)
    elif x < 0 and y < 0:
        return -1/holder_mean([-1/x, -1/y], r)
    else:
        return min(x, y)

def _smooth_min_list(values, r=0.1):
    if len(values) == 0:
        raise ValueError("List of values cannot be empty")
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise NotImplementedError("Input must be a 1D array in the current implementation")
        values = values.tolist()
    if len(values) == 1:
        return values[0]
    min_value = values[0]
    for val in values[1:]:
        min_value = _smooth_min_two_elements(min_value, val, r)
    return min_value

def smooth_min(x, y=None, r=0.1):
    match (x, y):
        case (list() | np.ndarray(), None):
            return _smooth_min_list(x, r)
        case (_, None):
            return x
        case (list() | np.ndarray(), _):
            if isinstance(y, (list, np.ndarray)):
                raise NotImplementedError("Cannot perform smooth_min on two lists or arrays directly")
            return _smooth_min_two_elements(_smooth_min_list(x, r), y, r)
        case (_, _):
            return _smooth_min_two_elements(x, y, r)

def _smooth_argmin_two_elements(x, y, r=0.1):
    if x >= 0 and y >= 0:
        grad = holder_mean_derivative([x, y], r)
        return np.argmax(grad)
    elif x < 0 and y < 0:
        grad = holder_mean_derivative([-1/x, -1/y], r)
        return np.argmax(grad)
    else:
        xs = np.array([x, y])
        imin = np.argmin(xs)
        return imin
        # return xs[imin]

def _smooth_argmin_list(values, r=0.1):
    if len(values) == 0:
        raise ValueError("List of values cannot be empty")
    if isinstance(values, np.ndarray):
        if values.ndim > 1:
            raise NotImplementedError("Input must be a 1D array in the current implementation")
        values = values.tolist()
    if len(values) == 1:
        return 0
    min_index = 0
    min_value = values[0]
    for i, val in enumerate(values[1:]):
        min_index_ = _smooth_argmin_two_elements(min_value, val, r)
        if min_index_ == 1:
            min_value = val
        min_index = min_index_ if min_index_ == 0 else i + 1
    return min_index

def smooth_argmin(x, y=None, r=0.1):
    match (x, y):
        case (list() | np.ndarray(), None):
            return _smooth_argmin_list(x, r)
        case (_, None):
            return x
        case (list() | np.ndarray(), _):
            if isinstance(y, (list, np.ndarray)):
                raise NotImplementedError("Cannot perform smooth_argmin on two lists or arrays directly")
            aux = _smooth_argmin_list(x, r) # This will be an array
            return _smooth_argmin_list([aux, y], r)
        case (_, _):
            return _smooth_argmin_two_elements(x, y, r)

def test_associativity():
    r = 0.1
    xs = np.arange(-10, 10, 1)
    ys, zs = xs.copy(), xs.copy()  # Ensure ys and zs are the same as xs for testing
    for x in xs:
        for y in ys:
            for z in zs:
                left = smooth_min(smooth_min(x, y, r), z, r)
                right = smooth_min(x, smooth_min(y, z, r), r)
                # print(f"Testing associativity for {x}, {y}, {z}: {left} vs {right}")
                assert np.isclose(left, right), f"Failed associativity for {x}, {y}, {z}"
                left_argmin = smooth_argmin(smooth_argmin(x, y, r), z, r)
                right_argmin = smooth_argmin(x, smooth_argmin(y, z, r), r)
                assert np.isclose(left_argmin, right_argmin), f"Failed argmin associativity for {x}, {y}, {z}"

def test_argmin():
    r = 0.1
    xs = np.arange(-10, 10, 1)
    ys = xs.copy()
    for x in xs:
        for y in ys:
            argmin = smooth_argmin(x, y, r)
            print(f"Testing argmin for {x}, {y}: {argmin}")
    print(smooth_argmin(xs, r=0.1))
    print(smooth_argmin(xs[::-1], r=0.1))
    print(smooth_argmin(xs, -20, r=0.1))

test_associativity()
test_argmin()
print("Associativity test passed.")
