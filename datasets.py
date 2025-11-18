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

def make_xor(n=100, noise=0.1):
    """
    Generates a 2D XOR dataset.
    n: total number of samples (will be divided by 4)
    noise: std deviation of Gaussian noise added to the points
    """
    n_per_quad = int(np.ceil(n / 4))
    n_total = n_per_quad * 4
    
    X = np.zeros((n_total, 2))
    y = np.zeros(n_total, dtype=int)
    
    quads = [
        ([1, 1], 0),   # Top-right, class 0
        ([-1, 1], 1),  # Top-left, class 1
        ([-1, -1], 0), # Bottom-left, class 0
        ([1, -1], 1)   # Bottom-right, class 1
    ]
    
    idx = 0
    for (center, label) in quads:
        points = np.random.randn(n_per_quad, 2) * noise + center
        X[idx:idx + n_per_quad] = points
        y[idx:idx + n_per_quad] = label
        idx += n_per_quad
        
    # Shuffle the data
    indices = np.arange(n_total)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y