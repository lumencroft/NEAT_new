import numpy as np
import random
import time
from typing import Dict, Callable, Tuple
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import optax

import neat_core
import datasets


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
    
    hidden_nodes = sorted([n for n in genome.nodes.values() if n.type == 'hid'], key=lambda n: n.id)
    output_nodes = sorted([n for n in genome.nodes.values() if n.type == 'out'], key=lambda n: n.id)
    nodes_in_order = hidden_nodes + output_nodes

    input_ids = sorted([n for n in genome.nodes.values() if n.type == 'in'], key=lambda n: n.id)
    
    conn_map = {}
    for innov, conn in genome.conns.items():
        if not conn.enabled:
            continue
        if conn.out_id not in conn_map:
            conn_map[conn.out_id] = []
        conn_map[conn.out_id].append((conn.in_id, innov))

    def predict_fn(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
        values = {nid.id: x[i] for i, nid in enumerate(input_ids)}
        for node in nodes_in_order:
            nid = node.id
            z = 0.0
            if nid in conn_map:
                for in_id, innov in conn_map[nid]:
                    w = params['weights'].get(innov)
                    if w is None: continue
                    v = values.get(in_id, 0.0)
                    z += w * v
            
            b = params['biases'].get(nid)
            if b is not None:
                z += b
            
            act_fn = JAX_ACTS[node.activation]
            values[nid] = act_fn(z)
            
        return jnp.array([values[nid.id] for nid in output_nodes])
    
    return predict_fn


def train_and_eval_genome(
    genome: neat_core.Genome,
    X: jnp.ndarray,
    y: jnp.ndarray,
    n_steps: int,
    lr: float,
    batch_size: int
) -> Tuple[neat_core.Genome, float]:
    
    params = {
        'weights': {innov: c.weight for innov, c in genome.conns.items() if c.enabled},
        'biases': {nid: n.bias for nid, n in genome.nodes.items() if n.type != 'in'}
    }
    
    if not params['weights'] or not params['biases']:
        return genome.copy(), 0.5

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
            indices = jax.random.permutation(subkey, n_samples)[:batch_size]
            x_batch, y_batch = X[indices], y[indices]
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
    
    fitness = float(accuracy)

    trained_genome = genome.copy()
    for innov, w in final_params['weights'].items():
        if innov in trained_genome.conns:
            trained_genome.conns[innov].weight = float(w)
    for nid, b in final_params['biases'].items():
        if nid in trained_genome.nodes:
            trained_genome.nodes[nid].bias = float(b)
            
    return trained_genome, fitness


def get_network_text(genome: neat_core.Genome) -> str:
    text = "NODES:\n"
    
    nodes = sorted(genome.nodes.values(), key=lambda n: n.id)
    for n in nodes:
        if n.type == 'in':
            text += f"  [{n.id}] IN\n"
        elif n.type == 'out':
            text += f"  [{n.id}] OUT ({n.activation}) Bias: {n.bias:.2f}\n"
        else:
            text += f"  [{n.id}] HID ({n.activation}) Bias: {n.bias:.2f}\n"
    
    text += "\nCONNECTIONS (Weight):\n"
    conns = sorted(genome.conns.values(), key=lambda c: (c.in_id, c.out_id))
    for c in conns:
        if c.enabled:
            text += f"  {c.in_id} -> {c.out_id} ({c.weight:.2f})\n"
        else:
            text += f"  {c.in_id} -> {c.out_id} (DISABLED)\n"
    
    return text

def plot_decision_boundary(ax, genome, X_data_plot, y_data_plot, title):
    try:
        final_params = {
            'weights': {innov: c.weight for innov, c in genome.conns.items() if c.enabled},
            'biases': {nid: n.bias for nid, n in genome.nodes.items() if n.type != 'in'}
        }
        final_predict_fn = build_jax_forward_fn(genome)
        final_batched_predict = vmap(final_predict_fn, in_axes=(None, 0))

        xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 50), np.linspace(-2.5, 2.5, 50))
        grid_points = jnp.array(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        
        Z = final_batched_predict(final_params, grid_points)
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, levels=[0, 0.5, 1.0], colors=['#483D8B', '#DC143C'], alpha=0.3)
    
    except (ValueError, KeyError, jax.errors.TracerArrayConversionError):
        ax.text(0.5, 0.5, 'Invalid network', ha='center', va='center', transform=ax.transAxes, color='red')

    ax.scatter(X_data_plot[y_data_plot == 0, 0], X_data_plot[y_data_plot == 0, 1], c='#483D8B', label='Class 0', alpha=0.6)
    ax.scatter(X_data_plot[y_data_plot == 1, 0], X_data_plot[y_data_plot == 1, 1], c='#DC143C', label='Class 1', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True)
    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])

def plot_live_update(fig, gen, best_genome_for_plot, second_best_genome_for_plot, random_genome_for_plot, 
                     X_data_plot, y_data_plot, best_hist, avg_hist, best_raw_fitness_for_plot):
    fig.clear()
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax1.plot(best_hist, label="Best Raw Fitness", linewidth=2)
    ax1.plot(avg_hist, label="Average Raw Fitness", linestyle=':', alpha=0.7)
    ax1.axhline(y=0.75, color='orange', linestyle='--', label='~75% Threshold')
    ax1.axhline(y=1.0, color='g', linestyle='--', label='100% Threshold')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Accuracy)")
    ax1.set_title(f"NEAT Fitness (Gen: {gen})")
    ax1.legend(loc='lower right')
    ax1.grid(True)
    ax1.set_ylim(0.45, 1.05)
    if len(best_hist) > 1:
        ax1.set_xlim(0, max(10, len(best_hist)-1))

    n_hid_best = len([n for n in best_genome_for_plot.nodes.values() if n.type == 'hid'])
    n_conn_best = len([c for c in best_genome_for_plot.conns.values() if c.enabled])
    plot_decision_boundary(ax2, best_genome_for_plot, X_data_plot, y_data_plot, 
                           f"BEST Genome (H:{n_hid_best}, C:{n_conn_best}, Fit:{best_raw_fitness_for_plot:.2f})")

    if random_genome_for_plot:
        n_hid_rand = len([n for n in random_genome_for_plot.nodes.values() if n.type == 'hid'])
        n_conn_rand = len([c for c in random_genome_for_plot.conns.values() if c.enabled])
        plot_decision_boundary(ax3, random_genome_for_plot, X_data_plot, y_data_plot, 
                               f"RANDOM Genome (H:{n_hid_rand}, C:{n_conn_rand})")
    else:
        ax3.text(0.5, 0.5, 'No random genome available', ha='center', va='center', transform=ax3.transAxes)

    if second_best_genome_for_plot:
        n_hid_sec = len([n for n in second_best_genome_for_plot.nodes.values() if n.type == 'hid'])
        n_conn_sec = len([c for c in second_best_genome_for_plot.conns.values() if c.enabled])
        plot_decision_boundary(ax4, second_best_genome_for_plot, X_data_plot, y_data_plot, 
                               f"SECOND BEST Genome (H:{n_hid_sec}, C:{n_conn_sec})")
    else:
        ax4.text(0.5, 0.5, 'No second best genome available', ha='center', va='center', transform=ax4.transAxes)

    plt.suptitle(f"Generation {gen} Evolution", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.pause(0.1)


class BackpropNEATPop(neat_core.NEATPop):
    
    def __init__(self, n_inputs, n_outputs, pop_size=100, compat_threshold=3.0):
        
        super().__init__(n_inputs, n_outputs, pop_size, compat_threshold, diff_only=True)
        
        for g in self.genomes:
            for j in range(self.n_outputs):
                nid = self.n_inputs + j
                if nid in g.nodes:
                    g.nodes[nid].activation = "sigmoid"

    def evolve_bp(self, X_train, y_train, X_data_plot, y_data_plot, 
                  generations=50, n_steps=100, 
                  lr=1e-2, batch_size=32, elite=0.1, 
                  m_add_conn=0.3, m_add_node=0.05, m_mutate_act=0.05,
                  complexity_bonus=0.001):
        
        overall_best_genome_by_raw_fit = self.genomes[0].copy()
        overall_best_raw_fitness = -1e9
        
        best_fitness_history = []
        avg_fitness_history = []
        
        fig = plt.figure(figsize=(16, 12))

        for gen in range(generations):
            
            genomes_with_scores = []
            raw_fitness_scores = []
            
            start_time = time.time()

            for genome_idx, genome in enumerate(self.genomes):
                
                trained_genome, raw_fit = train_and_eval_genome(
                    genome, X_train, y_train,
                    n_steps=n_steps,
                    lr=lr,
                    batch_size=batch_size
                )
                
                self.genomes[genome_idx] = trained_genome
                
                genomes_with_scores.append((trained_genome, raw_fit))
                raw_fitness_scores.append(raw_fit)
            
            max_raw_fit_this_gen = np.max(raw_fitness_scores)
            
            best_genome_for_plot = None
            
            if max_raw_fit_this_gen < 0.8:
                best_adjusted_score = -1e9
                for g, raw_fit in genomes_with_scores:
                    complexity = len(g.nodes) + len(g.conns)
                    adjusted_fit = raw_fit + complexity * complexity_bonus
                    
                    if adjusted_fit > best_adjusted_score:
                        best_adjusted_score = adjusted_fit
                        best_genome_for_plot = g
            else:
                for g, raw_fit in genomes_with_scores:
                    if raw_fit == max_raw_fit_this_gen:
                        best_genome_for_plot = g
                        break
            
            if best_genome_for_plot is None:
                best_genome_for_plot = genomes_with_scores[0][0]
            
            best_raw_fit_for_plot = [f for g, f in genomes_with_scores if g == best_genome_for_plot][0]

            genomes_with_scores.sort(key=lambda x: x[1], reverse=True)
            second_best_genome_for_plot = genomes_with_scores[1][0] if len(genomes_with_scores) > 1 else None
            random_genome_for_plot = random.choice(genomes_with_scores)[0]

            if max_raw_fit_this_gen > overall_best_raw_fitness:
                overall_best_raw_fitness = max_raw_fit_this_gen
                overall_best_genome_by_raw_fit = best_genome_for_plot.copy()

            gen_time = time.time() - start_time
            best_fitness_history.append(overall_best_raw_fitness)
            avg_fitness_history.append(np.mean(raw_fitness_scores))
            
            print(f"\n--- Gen {gen} (Took {gen_time:.2f}s) ---")
            print(f"    Best RAW Fitness (Overall): {overall_best_raw_fitness:.4f} (Avg RAW: {np.mean(raw_fitness_scores):.4f})")
            
            plot_live_update(fig, gen, 
                             best_genome_for_plot, 
                             second_best_genome_for_plot, 
                             random_genome_for_plot,
                             X_data_plot, y_data_plot, 
                             best_fitness_history, avg_fitness_history,
                             best_raw_fit_for_plot)
            
            self._next_generation(
                raw_fitness_scores,
                elite=elite,
                m_add_conn=m_add_conn,
                m_add_node=m_add_node,
                m_mutate_w=0.0,
                m_mutate_act=m_mutate_act
            )
            
        print("\n🎉 Evolution Complete! Close the plot window to exit.")
        fig.savefig("final_evolution_plot.png")
        print("Saved final plot to 'final_evolution_plot.png'")
        
        return overall_best_genome_by_raw_fit, overall_best_raw_fitness


if __name__ == "__main__":
    
    print(f"JAX devices: {jax.devices()}")
    
    X_data_plot, y_data_plot = datasets.make_xor(n=1000, noise=0.1)
    
    X_train = jnp.array(X_data_plot, dtype=jnp.float32)
    y_train = jnp.array(y_data_plot, dtype=jnp.float32).reshape(-1, 1)

    plt.ion()

    pop = BackpropNEATPop(
        n_inputs=2, 
        n_outputs=1, 
        pop_size=100,
        compat_threshold=1.5
    )
    
    best_genome, best_fitness = pop.evolve_bp(
        X_train, y_train,
        X_data_plot, y_data_plot,
        generations=75,       
        n_steps=5000,
        lr=0.01,
        batch_size=32,
        elite=0.1,
        m_add_conn=0.5,
        m_add_node=0.5,
        complexity_bonus=0.001
    )

    plt.ioff()
    plt.show()

    print(f"Best Fitness Achieved: {best_fitness:.4f}")
    
    neat_core.save_genome(best_genome, "best_genome.json")
    print(f"Saved best genome to 'best_genome.json'")