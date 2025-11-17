import numpy as np
import random
from typing import Dict, Callable, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap, lax
    import optax
except ImportError:
    print("JAX or Optax not installed. Please install with: pip install jax jaxlib optax")
    exit()

try:
    import neat_core
except ImportError:
    print("Error: neat_core.py not found in the same directory.")
    exit()

try:
    import datasets
except ImportError:
    print("Error: datasets.py not found in the same directory.")
    exit()


JAX_ACTS = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "id": lambda x: x,
    "sin": jnp.sin,
    "square": jnp.square,
    "abs": jnp.abs,
}

def build_jax_forward_fn(genome: neat_core.Genome) -> Callable:
    
    nodes_in_order = sorted([n for n in genome.nodes.values() if n.type != 'in'], key=lambda n: n.id)
    input_ids = sorted([n.id for n in genome.nodes.values() if n.type == 'in'])
    output_ids = sorted([n.id for n in genome.nodes.values() if n.type == 'out'])
    
    conn_map = {}
    for innov, conn in genome.conns.items():
        if not conn.enabled:
            continue
        if conn.out_id not in conn_map:
            conn_map[conn.out_id] = []
        conn_map[conn.out_id].append((conn.in_id, innov))

    def predict_fn(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        
        values = {nid: x[i] for i, nid in enumerate(input_ids)}
        
        for node in nodes_in_order:
            nid = node.id
            
            z = 0.0
            if nid in conn_map:
                for in_id, innov in conn_map[nid]:
                    w = params['weights'][innov]
                    v = values.get(in_id, 0.0)
                    z += w * v
            
            z += params['biases'][nid]
            
            act_fn = JAX_ACTS[node.activation]
            values[nid] = act_fn(z)
            
        outs = jnp.array([values[oid] for oid in output_ids])
        return outs
    
    return predict_fn


def train_and_eval_genome(
    genome: neat_core.Genome,
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_steps: int,
    lr: float,
    batch_size: int,
    arch_penalty: float
) -> Tuple[neat_core.Genome, float]:
    
    params = {
        'weights': {innov: c.weight for innov, c in genome.conns.items() if c.enabled},
        'biases': {nid: n.bias for nid, n in genome.nodes.items() if n.type != 'in'}
    }
    
    predict_fn = build_jax_forward_fn(genome)
    batched_predict = vmap(predict_fn, in_axes=(None, 0))

    def loss_fn(params, x_batch, y_batch):
        preds = batched_predict(params, x_batch)
        preds = jnp.squeeze(preds)
        y_batch = jnp.squeeze(y_batch)
        
        bce = -jnp.mean(y_batch * jnp.log(preds + 1e-7) + (1 - y_batch) * jnp.log(1 - preds + 1e-7))
        return bce

    optimizer = optax.adam(lr)
    
    @jit
    def run_training_loop(initial_params, initial_opt_state, initial_key):

        def train_step(params, opt_state, x_batch, y_batch):
            loss_val, grads = jax.value_and_grad(loss_fn)(params, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss_val
        
        n_samples = len(X)
        
        def scan_body(carry, _):
            params, opt_state, key = carry
            
            key, subkey = jax.random.split(key)
            
            def sample_batch(key):
                indices = jax.random.permutation(key, n_samples)[:batch_size]
                return X[indices], y[indices]
            
            def use_full_data(key):
                # --- FIX ---
                # Return a batch-sized sample, but with replacement,
                # to match the shape of sample_batch.
                indices = jax.random.choice(key, n_samples, shape=(batch_size,))
                return X[indices], y[indices]
                # --- END FIX ---

            x_batch, y_batch = lax.cond(
                n_samples > batch_size,
                sample_batch,
                use_full_data,
                operand=subkey
            )
            
            new_params, new_opt_state, loss_val = train_step(params, opt_state, x_batch, y_batch)
            
            return (new_params, new_opt_state, key), loss_val

        (final_params, final_opt_state, _), losses = lax.scan(
            scan_body,
            (initial_params, initial_opt_state, initial_key),
            None,
            length=n_steps
        )
        
        return final_params

    opt_state = optimizer.init(params)
    
    final_params = run_training_loop(params, opt_state, jax.random.PRNGKey(0))

    final_preds = jnp.squeeze(batched_predict(final_params, X))
    accuracy = jnp.mean(jnp.round(final_preds) == jnp.squeeze(y))
    
    n_conns = len([c for c in genome.conns.values() if c.enabled])
    n_nodes = len([n for n in genome.nodes.values() if n.type == 'hid'])
    complexity_penalty = arch_penalty * (n_conns + n_nodes)
    
    fitness = float(accuracy - complexity_penalty)

    trained_genome = genome.copy()
    for innov, w in final_params['weights'].items():
        if innov in trained_genome.conns:
            trained_genome.conns[innov].weight = float(w)
    for nid, b in final_params['biases'].items():
        if nid in trained_genome.nodes:
            trained_genome.nodes[nid].bias = float(b)
            
    return trained_genome, fitness


class BackpropNEATPop(neat_core.NEATPop):
    
    def __init__(self, n_inputs, n_outputs, pop_size=100, compat_threshold=3.0,
                 force_initial_hidden=True): # Added a flag
        
        # 1. Standard init
        super().__init__(n_inputs, n_outputs, pop_size, compat_threshold, diff_only=True)
        
        # 2. Set output activations (as before)
        for g in self.genomes:
            for j in range(self.n_outputs):
                nid = self.n_inputs + j
                if nid in g.nodes:
                    g.nodes[nid].activation = "sigmoid"

        # 3. --- NEW: Force initial structure ---
        if force_initial_hidden:
            # ID for the new hidden node (e.g., inputs 0, 1; output 2; new_node 3)
            hidden_node_id = self.n_inputs + self.n_outputs 
            
            # The next available ID for future mutations
            next_id_for_genomes = hidden_node_id + 1

            for g in self.genomes:
                # --- FIX 1: Use NodeGene ---
                g.nodes[hidden_node_id] = neat_core.NodeGene(
                    id=hidden_node_id, 
                    type='hid', 
                    activation='relu', 
                    bias=0.0
                )
                
                # --- FIX 2: Manually add connections using self.db ---
                
                # Connect all inputs to the new hidden node
                for in_id in range(self.n_inputs):
                    innov = self.db.get(in_id, hidden_node_id)
                    weight = random.uniform(-0.1, 0.1)
                    g.conns[innov] = neat_core.ConnGene(
                        in_id, hidden_node_id, weight, True, innov
                    )
                
                # Connect the new hidden node to all outputs
                for out_idx in range(self.n_outputs):
                    output_node_id = self.n_inputs + out_idx
                    innov = self.db.get(hidden_node_id, output_node_id)
                    weight = random.uniform(-0.1, 0.1)
                    g.conns[innov] = neat_core.ConnGene(
                        hidden_node_id, output_node_id, weight, True, innov
                    )

                # Update the genome's internal node counter
                g.next_node_id = max(g.next_node_id, next_id_for_genomes)
                

    def evolve_bp(self, X_train, y_train, generations=50, n_steps=100, 
                  lr=1e-2, batch_size=32, arch_penalty=0.0, elite=0.1, 
                  m_add_conn=0.3, m_add_node=0.05, m_mutate_act=0.05):
        
        best_genome = None
        best_fitness = -1e9

        for gen in range(generations):
            fitness_scores = []
            
            # --- 🖨️ START DEBUG PRINT ---
            # This dictionary will store the fitness scores for each unique structure
            # Key: (num_hidden_nodes, num_connections), Value: [list_of_fitness_scores]
            struct_counts = {}
            # --- 🖨️ END DEBUG PRINT ---

            for i, genome in enumerate(self.genomes):
                trained_genome, fitness = train_and_eval_genome(
                    genome, X_train, y_train,
                    n_steps=n_steps,
                    lr=lr,
                    batch_size=batch_size,
                    arch_penalty=arch_penalty
                )
                
                self.genomes[i] = trained_genome
                fitness_scores.append(fitness)

                # --- 🖨️ START DEBUG PRINT ---
                # Get structural info for this *trained* genome
                n_hid = len([n for n in trained_genome.nodes.values() if n.type == 'hid'])
                n_conn = len([c for c in trained_genome.conns.values() if c.enabled])
                key = (n_hid, n_conn)
                
                # Store this genome's fitness under its structure
                if key not in struct_counts:
                    struct_counts[key] = []
                struct_counts[key].append(fitness)
                # --- 🖨️ END DEBUG PRINT ---

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_genome = trained_genome.copy()
            
            # --- 🖨️ START DEBUG PRINT ---
            print(f"\n--- Gen {gen} Population Summary (Post-Training) ---")
            for (n_hid, n_conn), fits in struct_counts.items():
                print(f"    - Struct (H-Nodes: {n_hid}, Conns: {n_conn}): {len(fits)} genomes, Avg Fit: {np.mean(fits):.4f}, Max Fit: {np.max(fits):.4f}")
            # --- 🖨️ END DEBUG PRINT ---
            
            # This is your original print
            print(f"--- Gen {gen} (Pre-Selection) ---")
            print(f"Best Fitness: {best_fitness:.4f} (Avg: {np.mean(fitness_scores):.4f})")
            print(f"Best Genome: {len(best_genome.nodes)} nodes, {len(best_genome.conns)} conns")

            self._next_generation(
                fitness_scores,
                elite=elite,
                m_add_conn=m_add_conn,
                m_add_node=m_add_node,
                m_mutate_w=0.0,
                m_mutate_act=m_mutate_act
            )

        return best_genome, best_fitness


if __name__ == "__main__":
    
    # --- GPU/Device Check ---
    print(f"JAX devices: {jax.devices()}")
    
    print("🚀 Starting Backprop NEAT example on 'XOR' dataset...")
    
    X_data, y_data = datasets.make_xor(n=1000, noise=0.0)
    
    X_train = jnp.array(X_data, dtype=jnp.float32)
    y_train = jnp.array(y_data, dtype=jnp.float32).reshape(-1, 1)

    pop = BackpropNEATPop(
        n_inputs=2, 
        n_outputs=1, 
        pop_size=20,
        compat_threshold=1.5
    )
    
    best_genome, best_fitness = pop.evolve_bp(
        X_train, y_train,
        generations=150,
        n_steps=5000,
        lr=0.01,
        elite=0.01,
        batch_size=32,
        arch_penalty=0.0,
        m_add_conn=0.5,
        m_add_node=0.4
    )

    print("\n🎉 Evolution Complete!")
    print(f"Best Fitness Achieved: {best_fitness:.4f}")
    
    neat_core.save_genome(best_genome, "best_genome.json")
    print(f"Saved best genome to 'best_genome.json'")

    print("\n--- Testing Best Genome on Training Data ---")
    
    final_params = {
        'weights': {innov: c.weight for innov, c in best_genome.conns.items() if c.enabled},
        'biases': {nid: n.bias for nid, n in best_genome.nodes.items() if n.type != 'in'}
    }
    final_predict_fn = build_jax_forward_fn(best_genome)
    final_batched_predict = vmap(final_predict_fn, in_axes=(None, 0))
    
    preds = final_batched_predict(final_params, X_train)
    
    final_acc = jnp.mean(jnp.round(preds) == y_train)
    print(f"\nFinal Accuracy on Training Set: {final_acc * 100:.2f}%")

    print("Testing on first 5 examples:")
    for i in range(5):
        target = y_train[i][0]
        pred_val = preds[i][0]
        rounded_pred = jnp.round(pred_val)
        status = "✅" if rounded_pred == target else "❌"
        print(f"Input: {X_train[i]}, Target: {target}, Pred: {pred_val:.4f} ({rounded_pred}) {status}")