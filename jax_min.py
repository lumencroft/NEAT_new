import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap, lax
import neat_core
import datasets

ACTS = {
    "relu": jax.nn.relu, "tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid, 
    "id": lambda x: x, "sin": jnp.sin, "abs": jnp.abs, "square": jnp.square
}

def get_predict_fn(g):
    in_ids = sorted([n.id for n in g.nodes.values() if n.type == 'in'])
    
    # CRITICAL FIX: Explicitly process Hidden nodes BEFORE Output nodes
    h_nodes = sorted([n for n in g.nodes.values() if n.type == 'hid'], key=lambda x: x.id)
    o_nodes = sorted([n for n in g.nodes.values() if n.type == 'out'], key=lambda x: x.id)
    exec_order = h_nodes + o_nodes
    
    # Map connections for fast lookup
    conn_map = {}
    for c in g.conns.values():
        if c.enabled:
            conn_map.setdefault(c.out_id, []).append((c.in_id, c.weight))

    def predict(vals, x):
        # Initialize inputs
        activations = {i: x[k] for k, i in enumerate(in_ids)}
        
        for n in exec_order:
            # Sum inputs: bias + sum(weight * input_val)
            total = vals['b'].get(n.id, 0.0)
            if n.id in conn_map:
                total += sum(w * activations.get(src, 0.0) for src, w in conn_map[n.id])
            
            activations[n.id] = ACTS[n.activation](total)
            
        return jnp.array([activations[n.id] for n in o_nodes])
    
    return predict

# 3. High-Performance Training Loop (using lax.scan)
def train_genome(g, X, y, steps, lr):
    # 1. Extract Keys for Trainable Parameters
    # We only care about ENABLED connections
    w_keys = [k for k, c in g.conns.items() if c.enabled]
    if not w_keys: return g, 0.5 # Dead network

    # 2. Create a Fast Lookup: Innovation Number -> Array Index
    # This fixes the error by avoiding c.id and using the dict key instead
    w_key_to_idx = {k: i for i, k in enumerate(w_keys)}
    
    # 3. Extract Biases
    p_b = {n.id: n.bias for n in g.nodes.values() if n.type != 'in'}
    b_keys = list(p_b.keys())
    b_key_to_idx = {k: i for i, k in enumerate(b_keys)}

    # 4. Build Topology Map using the Lookup
    # Map: Node_ID -> List of (Input_Node_ID, Weight_Array_Index)
    topo_map = {}
    for k, c in g.conns.items():
        if c.enabled:
            # Use the key 'k' to find the index, not c.id
            w_idx = w_key_to_idx[k] 
            topo_map.setdefault(c.out_id, []).append((c.in_id, w_idx))

    # 5. JAX Data Prep
    w_init = jnp.array([g.conns[k].weight for k in w_keys])
    b_init = jnp.array([g.nodes[k].bias for k in b_keys])
    
    # Topology execution order
    in_ids = sorted([n.id for n in g.nodes.values() if n.type == 'in'])
    h_nodes = sorted([n for n in g.nodes.values() if n.type == 'hid'], key=lambda x: x.id)
    o_nodes = sorted([n for n in g.nodes.values() if n.type == 'out'], key=lambda x: x.id)
    exec_order = h_nodes + o_nodes

    # 6. The Fast JAX Predict Function
    def fast_predict(params, x):
        ws, bs = params 
        activations = {i: x[k] for k, i in enumerate(in_ids)}
        
        for n in exec_order:
            # Retrieve bias
            b_idx = b_key_to_idx.get(n.id)
            if b_idx is not None:
                total = bs[b_idx]
            else:
                total = 0.0
            
            # Sum connections
            if n.id in topo_map:
                for src, w_idx in topo_map[n.id]:
                    total += ws[w_idx] * activations.get(src, 0.0)
            
            activations[n.id] = ACTS[n.activation](total)
            
        return jnp.array([activations[n.id] for n in o_nodes])

    # 7. Optimizer & Loop
    opt = optax.adam(lr)
    
    @jit
    def run_scan(params_tuple, key):
        opt_state = opt.init(params_tuple)
        
        def update_step(carry, _):
            curr_p, curr_opt, k = carry
            k, subk = jax.random.split(k)
            # Random batch of 32
            idx = jax.random.randint(subk, (32,), 0, len(X))
            bx, by = X[idx], y[idx]
            
            def loss(p_input):
                preds = jnp.squeeze(vmap(fast_predict, (None, 0))(p_input, bx))
                # Safe log to prevent NaN
                return -jnp.mean(by * jnp.log(preds+1e-7) + (1-by) * jnp.log(1-preds+1e-7))
            
            l, grads = jax.value_and_grad(loss)(curr_p)
            updates, new_opt = opt.update(grads, curr_opt)
            new_p = optax.apply_updates(curr_p, updates)
            return (new_p, new_opt, k), None

        (final_p, _, _), _ = lax.scan(update_step, (params_tuple, opt_state, key), None, length=steps)
        return final_p

    # 8. Run Training
    final_ws, final_bs = run_scan((w_init, b_init), jax.random.PRNGKey(0))

    # 9. Write Back to Genome
    for i, k in enumerate(w_keys): g.conns[k].weight = float(final_ws[i])
    for i, k in enumerate(b_keys): g.nodes[k].bias = float(final_bs[i])

    # 10. Final Evaluation
    final_preds = jnp.squeeze(vmap(fast_predict, (None, 0))((final_ws, final_bs), X))
    acc = jnp.mean(jnp.round(final_preds) == jnp.squeeze(y))
    
    return g, float(acc)

# 4. Minimal Main
if __name__ == "__main__":
    X, y = datasets.make_circle(n=1000, noise=0.1)
    # Ensure Float32 for JAX
    X, y = jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32)

    pop = neat_core.NEATPop(2, 1, pop_size=100, compat_threshold=3.0)
    # Force sigmoid output for classification
    for g in pop.genomes: g.nodes[2].activation = "sigmoid"

    for gen in range(20):
        results = []
        for g in pop.genomes:
            g, fit = train_genome(g, X, y, steps=500, lr=0.02)
            # Complexity penalty
            adj_fit = fit - (len(g.conns)*0.01)
            results.append((g, fit, adj_fit))
        
        results.sort(key=lambda x: x[2], reverse=True)
        best, best_fit, _ = results[0]
        print(f"Gen {gen} | Best: {best_fit:.2%}")
        
        if best_fit > 0.99: break
        
        pop._next_generation([r[1] for r in results], elite=0.1, m_add_conn=0.5, m_add_node=0.2)

    print(f"Done. Best: {best_fit:.2%}")
    neat_core.save_genome(best, "best_genome.json")