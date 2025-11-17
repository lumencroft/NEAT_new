import numpy as np

def make_circle(n=1000, r=3.0, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-5, 5, size=(n, 2))
    y = (np.linalg.norm(X, axis=1) > r).astype(np.int32)
    X += rng.normal(0, noise, X.shape)
    return X, y

def make_spiral(n=2000, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    theta = np.sqrt(rng.random(n2)) * 3 * np.pi
    r = 2 * theta
    X1 = np.c_[r*np.cos(theta), r*np.sin(theta)] + rng.normal(scale=noise, size=(n2, 2))
    X2 = np.c_[-r*np.cos(theta), -r*np.sin(theta)] + rng.normal(scale=noise, size=(n2, 2))
    X = np.vstack([X1, X2])
    y = np.array([0]*n2 + [1]*n2)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    return X, y 

def make_xor(n=1000, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:,0] > 0) ^ (X[:,1] > 0)).astype(np.int32)
    X += rng.normal(0, noise, X.shape)
    return X, y