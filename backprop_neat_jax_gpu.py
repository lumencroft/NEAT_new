import argparse, json
import numpy as np

from neat_core import NEATPop, Genome, load_genome, save_genome, DIFF_ACTIVATIONS
from datasets import make_circle, make_spiral, make_xor

import jax
import jax.numpy as jnp

from functools import partial

ACT_POOL = list(DIFF_ACTIVATIONS.keys())

def make_task(name, n=2000, seed=0):
    if name == "circle":
        X, y = make_circle(n=n, seed=seed)
    elif name == "spiral":
        X, y = make_spiral(n=n, seed=seed)
    elif name == "xor":
        X, y = make_xor(n=n, seed=seed)
    else:
        raise ValueError("unknown task")
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    y = y.astype(np.int32)
    return X, y

def param_shapes(genome: Genome):
    conns = [c for c in genome.conns.values() if c.enabled]
    n_bias = sum(1 for n in genome.nodes.values() if n.type != "in")
    return len(conns), n_bias

def unpack_params(genome: Genome, theta):
    conns = [c for c in genome.conns.values() if c.enabled]
    n_w, n_b = param_shapes(genome)
    w = theta[:n_w]
    b = theta[n_w : n_w + n_b]
    conn_ws = {}
    for i, c in enumerate(conns):
        conn_ws[(c.in_id, c.out_id)] = w[i]
    node_bias = {}
    idx = 0
    for nid, n in sorted(genome.nodes.items()):
        if n.type != "in":
            node_bias[nid] = b[idx]
            idx += 1
    return conn_ws, node_bias

def jax_forward(genome: Genome, theta, X):
    conn_ws, node_bias = unpack_params(genome, theta)
    order = sorted(genome.nodes.keys())

    def single(x):
        values = {}
        for i in range(genome.n_inputs):
            values[i] = x[i]
        for nid in order:
            n = genome.nodes[nid]
            if n.type == "in":
                continue
            inc = 0.0
            # accumulate inputs
            for (i, j), w in conn_ws.items():
                if j == nid:
                    inc += values.get(i, 0.0) * w
            z = inc + node_bias[nid]
            # activation
            if n.activation == "relu":
                a = jnp.maximum(0.0, z)
            elif n.activation == "tanh":
                a = jnp.tanh(z)
            elif n.activation == "sigmoid":
                a = 1.0 / (1.0 + jnp.exp(-z))
            else:  # identity
                a = z
            values[nid] = a
        outs = [values[genome.n_inputs + j] for j in range(genome.n_outputs)]
        return jnp.stack(outs)
    return jax.vmap(single)(X)

def loss_fn(genome: Genome, theta, X, y, complexity_penalty=1e-3):
    logits = jax_forward(genome, theta, X)
    if genome.n_outputs == 1:
        y_f = y.astype(jnp.float32)
        pred = jax.nn.sigmoid(logits[:, 0])
        bce = - (y_f * jnp.log(pred + 1e-8) + (1 - y_f) * jnp.log(1 - pred + 1e-8)).mean()
    else:
        logits2 = jnp.concatenate([jnp.zeros((logits.shape[0], 1)), logits], axis=1)
        y_onehot = jax.nn.one_hot(y, 2)
        bce = -jnp.mean(jnp.sum(y_onehot * jax.nn.log_softmax(logits2), axis=1))
    n_nodes = sum(1 for n in genome.nodes.values() if n.type == "hid")
    n_conns = sum(1 for c in genome.conns.values() if c.enabled)
    comp = complexity_penalty * (n_nodes + 0.5 * n_conns)
    return bce + comp

@partial(jax.jit, static_argnums=(0,))
def sgd_step(genome, theta, X, y, lr):
    grad = jax.grad(loss_fn, argnums=1)(genome, theta, X, y)
    return theta - lr * grad

def train_weights(genome: Genome, X, y, steps=300, lr=0.05):
    n_w, n_b = param_shapes(genome)
    key = jax.random.PRNGKey(0)
    theta = jax.random.normal(key, (n_w + n_b,)) * 0.5
    for _ in range(steps):
        theta = sgd_step(genome, theta, X, y, lr)
    loss = loss_fn(genome, theta, X, y)
    return theta, float(loss)

def fitness_of_genome(genome, X, y, train_steps, lr):
    theta, loss = train_weights(genome, X, y, steps=train_steps, lr=lr)
    return -loss

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="circle", choices=["circle", "spiral", "xor"])
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--train_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--plot_only", type=str, default="")
    args = ap.parse_args()

    X_np, y_np = make_task(args.task)
    # convert to JAX arrays
    X = jnp.array(X_np)
    y = jnp.array(y_np)

    if args.plot_only:
        g = load_genome(args.plot_only)
        from visualize import draw_network
        draw_network(g, fname="arch.png")
        print("Saved arch.png")
        return

    n_in, n_out = 2, 1
    pop = NEATPop(n_in, n_out, pop_size=args.pop, compat_threshold=2.5, diff_only=True)

    best = None
    best_fit = -1e9
    for gen in range(args.generations):
        fits = []
        for g in pop.genomes:
            f = fitness_of_genome(g, X, y, args.train_steps, args.lr)
            fits.append(f)
            if f > best_fit:
                best, best_fit = g, f
        pop = pop._next_generation(fits)
        print(f"Gen {gen} best fitness: {best_fit:.4f}")

    print("Final best fitness:", best_fit)
    save_genome(best, f"best_{args.task}.json")
    from visualize import draw_network
    draw_network(best, fname=f"best_{args.task}.png")
    print(f"Saved best genome JSON + PNG for {args.task}")

if __name__ == "__main__":
    # Print device info
    print("JAX devices:", jax.devices())
    print("Default backend:", jax.default_backend())
    main()
