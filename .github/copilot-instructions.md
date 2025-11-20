# Copilot / AI agent instructions — NEAT_new

Purpose: quickly orient an AI coding assistant to be productive in this repository.

Key points
- Architecture: This repo implements NEAT (NeuroEvolution of Augmenting Topologies) with JAX-accelerated training. Core responsibilities are split between:
  - `neat_core.py`: data structures and evolutionary logic (Genome, Node, Conn, `NEATPop`, serialization via `save_genome`).
  - `jax_min.py` and `jax_run.py`: JAX-based training loops that convert the object-based genome into flat parameter arrays for fast training (uses `jax`, `jax.numpy`, `optax`, `vmap`, `lax.scan`).
  - `backprop_neat_jax.py` / `backprop_neat_jax_gpu.py`: alternate training workflows (CPU / GPU variants).
  - `datasets.py`: synthetic datasets (e.g., circle, xor) used by training scripts.
  - `train.py`, `play.py`, `visualize.py`, `xorvisual.py`: user-facing runners and visualizers.

Data model conventions
- Genomes are Python objects with `nodes` and `conns` dictionaries (keys are identifiers/innovation numbers). Node objects include `id`, `type` (`'in'|'hid'|'out'`), `activation`, and `bias`. Connection objects include `in_id`, `out_id`, `weight`, and `enabled`.
- When converting to JAX arrays, code expects:
  - `w_keys` (list of enabled connection dict keys) used to build `w_init` (weights array).
  - `w_key_to_idx` mapping from dict key -> index in the weight array (important: use the dict key, not `c.id` in some places).
  - `b_keys` and `b_key_to_idx` for biases.
  - A topology map `topo_map` mapping `out_node_id -> list[(src_node_id, weight_array_index)]` used at inference time.

JAX & training specifics (what to watch for)
- Use `jnp.array(..., dtype=jnp.float32)` for inputs/labels to avoid dtype issues.
- PRNGs: training loops use `jax.random.PRNGKey` and split keys in `lax.scan` — keep deterministic seeding in mind when testing.
- `vmap` and `lax.scan` are used heavily. When editing `fast_predict` / `train_genome`, preserve shapes so `vmap` maps over the batch dimension correctly.
- Optimizers use `optax` (e.g., `optax.adam`). Gradients are computed with `jax.value_and_grad` on functions that accept JAX arrays/tuples (e.g., `(weights, biases)`).
- Watch for the explicit execution order: hidden nodes are processed before output nodes (see `exec_order = h_nodes + o_nodes`) — changing that changes semantics.

Project-specific conventions and pitfalls
- Use the connection dict key (the map key in `g.conns`) for indexing weight arrays — several files convert keys -> indices; avoid using `c.id` in places where the dict key is used as the 'innovation number'.
- Biases are stored per-node and converted into a contiguous `b_init` array using `b_keys`; b_keys order matters for write-back.
- Enabled-only: training pipelines filter `g.conns` to only enabled connections; disabled conns are ignored (and may lead to 'dead' networks if none enabled).
- For classification experiments, some scripts explicitly force the output node activation to `sigmoid` (example: `for g in pop.genomes: g.nodes[2].activation = "sigmoid"`). Use this pattern when adding tests that expect probabilities.

Useful entry points / examples (concrete snippets)
- Train a single genome with the fast JAX trainer (from `jax_min.py`):
  - `g, fit = train_genome(g, X, y, steps=500, lr=0.02)`
  - `final_preds = jnp.squeeze(vmap(fast_predict, (None, 0))((final_ws, final_bs), X))`
- Evolution loop pattern (in `jax_min.py` main):
  - Build `NEATPop`, set output activation, call `train_genome` for each genome, compute adjusted fitness, sort, then `pop._next_generation([r[1] for r in results], elite=0.1, m_add_conn=0.5, m_add_node=0.2)`.
- Serializing: `neat_core.save_genome(best, "best_genome.json")` — inspect these files to understand persisted schema.

Developer workflows
- Run a trainer locally: `python jax_min.py` (or `python train.py` for other experiments). Use the GPU variant if you have JAX with GPU: `python backprop_neat_jax_gpu.py`.
- Quick debug: add concise `print()` statements for small test genomes, or run a minimal example with a single genome to inspect `w_keys`, `w_key_to_idx`, `topo_map` shapes before full training.
- When changing shapes or JAX code, run small, deterministic inputs (e.g., `X, y = datasets.make_circle(n=100, noise=0.1)` and `steps=10`) to catch shape/dtype errors quickly.

Files to inspect when making changes
- `neat_core.py` — genome representation, population evolution, mutation/crossover.
- `jax_min.py` — compact, high-performance training; good reference for correct array mapping and `lax.scan` usage.
- `backprop_neat_jax.py` / `_gpu.py` — alternate training strategies and GPU notes.
- `datasets.py` — data generators used across scripts.

When in doubt
- Follow the existing pattern: convert object graphs -> flat arrays -> index maps -> JAX functions. Changing the indexing logic is the most common source of bugs.

If you want more
- Tell me which area to expand (e.g., detailed doc for `neat_core.Genome` schema, or annotated walkthrough of `train_genome` in `jax_min.py`) and I will extend this file.

---
Generated by an AI agent to help contributors and other assistants onboard rapidly.
