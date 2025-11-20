import numpy as np

def make_circle(n=1000, r=3.0, noise=0.4):
    rng = np.random.default_rng()
    X = rng.uniform(-5, 5, size=(n, 2))
    y = (np.linalg.norm(X, axis=1) > r).astype(np.int32)
    X += rng.normal(0, noise, X.shape)
    return X, y

def make_spiral(n=2000, noise=0.2):
    y = np.repeat([0, 1], n // 2)
    
    theta = np.sqrt(np.random.rand(n)) * 3 * np.pi
    r = 2 * theta
    
    rotation = theta + (y * np.pi)
    
    X = np.c_[r * np.cos(rotation), r * np.sin(rotation)]
    X += np.random.randn(n, 2) * noise
    
    return X, y

def make_xor(n=100, noise=0.1):
    X = np.random.randn(n, 2) * noise
    centers = np.random.choice([-1, 1], size=(n, 2))
    X += centers
    y = (centers[:, 0] != centers[:, 1]).astype(int)
    return X, y