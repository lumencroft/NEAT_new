
import math, random, json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable
import numpy as np

# ---- Activation functions (Part A uses a diverse set; Part B will narrow) ----
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def sigmoid(x):
    x = np.clip(x, -50, 50)   
    return 1.0 / (1.0 + np.exp(-x))
def identity(x): return x
def sin(x): return np.sin(x)
def square(x): return np.square(x)
def abs_act(x): return np.abs(x)

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "id": identity,
    "sin": sin,
    "square": square,
    "abs": abs_act,
}
DIFF_ACTIVATIONS = {"relu": relu, "tanh": tanh, "sigmoid": sigmoid, "id": identity}

# ---- Genes ----
@dataclass
class NodeGene:
    id: int
    type: str  # 'in', 'out', 'hid'
    activation: str = "tanh"
    bias: float = 0.0

@dataclass
class ConnGene:
    in_id: int
    out_id: int
    weight: float
    enabled: bool
    innov: int

# ---- Innovation tracker ----
class InnovationDB:
    def __init__(self):
        self.counter = 0
        self.table = {}  # (in_id,out_id) -> innov

    def get(self, in_id, out_id):
        key = (in_id, out_id)
        if key not in self.table:
            self.table[key] = self.counter
            self.counter += 1
        return self.table[key]

# ---- Genome ----
class Genome:
    def __init__(self, n_inputs, n_outputs, innovation_db: InnovationDB, diff_only=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes: Dict[int, NodeGene] = {}
        self.conns: Dict[int, ConnGene] = {}  # innov -> ConnGene
        self.innovation_db = innovation_db
        self.diff_only = diff_only

        # create input/output nodes
        for i in range(n_inputs):
            self.nodes[i] = NodeGene(i, "in", "id", 0.0)
        for j in range(n_outputs):
            nid = n_inputs + j
            self.nodes[nid] = NodeGene(nid, "out", "tanh", 0.0)

        self.next_node_id = n_inputs + n_outputs

    # ensure acyclic by only allowing connections from lower id to higher id
    def _can_connect(self, a, b):
        return a != b and a < b and self.nodes[a].type != "out" and self.nodes[b].type != "in"

    def add_connection_mutation(self, max_tries=20, w_std=1.0):
        node_ids = list(self.nodes.keys())
        for _ in range(max_tries):
            a, b = random.sample(node_ids, 2)
            if not self._can_connect(a, b):
                continue
            innov = self.innovation_db.get(a, b)
            if innov in self.conns:
                continue
            self.conns[innov] = ConnGene(a, b, random.gauss(0, w_std), True, innov)
            return True
        return False

    def add_node_mutation(self):
        enabled_conns = [c for c in self.conns.values() if c.enabled]
        if not enabled_conns:
            return False
        conn = random.choice(enabled_conns)
        conn.enabled = False

        new_id = self.next_node_id
        self.next_node_id += 1
        act_pool = DIFF_ACTIVATIONS if self.diff_only else ACTIVATIONS
        act = random.choice(list(act_pool.keys()))
        self.nodes[new_id] = NodeGene(new_id, "hid", act, 0.0)

        innov1 = self.innovation_db.get(conn.in_id, new_id)
        innov2 = self.innovation_db.get(new_id, conn.out_id)
        self.conns[innov1] = ConnGene(conn.in_id, new_id, 1.0, True, innov1)
        self.conns[innov2] = ConnGene(new_id, conn.out_id, conn.weight, True, innov2)
        return True

    def mutate_weights(self, sigma=0.5, p_reset=0.1):
        for c in self.conns.values():
            if random.random() < p_reset:
                c.weight = random.gauss(0, 1.0)
            else:
                c.weight += random.gauss(0, sigma)
        for n in self.nodes.values():
            if n.type != "in":
                n.bias += random.gauss(0, 0.1)

    def mutate_activations(self, p=0.05):
        pool = DIFF_ACTIVATIONS if self.diff_only else ACTIVATIONS
        keys = list(pool.keys())
        for n in self.nodes.values():
            if n.type == "hid" and random.random() < p:
                n.activation = random.choice(keys)

    def copy(self):
        g = Genome(self.n_inputs, self.n_outputs, self.innovation_db, self.diff_only)
        g.nodes = {i: NodeGene(**vars(n)) for i, n in self.nodes.items()}
        g.conns = {i: ConnGene(**vars(c)) for i, c in self.conns.items()}
        g.next_node_id = self.next_node_id
        return g

    # Forward pass through an acyclic graph by topological order = node id
    def forward(self, x: np.ndarray) -> np.ndarray:
        # x shape: (n_inputs,)
        values = {i: x[i] for i in range(self.n_inputs)}
        order = sorted(self.nodes.keys())
        for nid in order:
            node = self.nodes[nid]
            if node.type == "in":
                continue
            inc = 0.0
            for c in self.conns.values():
                if c.enabled and c.out_id == nid:
                    inc += values.get(c.in_id, 0.0) * c.weight
            z = inc + node.bias
            act = ACTIVATIONS[node.activation]
            values[nid] = float(act(np.array([z]))[0])
        outs = [values[self.n_inputs + j] for j in range(self.n_outputs)]
        return np.array(outs, dtype=np.float32)

    # Distance for speciation
    def distance(self, other, c1=1.0, c2=1.0, c3=0.4):
        innovs1 = set(self.conns.keys())
        innovs2 = set(other.conns.keys())
        matching = innovs1 & innovs2
        disjoint = (innovs1 ^ innovs2)
        N = max(len(self.conns), len(other.conns), 1)
        if matching:
            w = np.mean([abs(self.conns[i].weight - other.conns[i].weight) for i in matching])
        else:
            w = 0.0
        # Excess vs disjoint not separated for simplicity
        return (c1 * len(disjoint)) / N + c3 * w

def crossover(parent1: Genome, parent2: Genome) -> Genome:
    # assume parent1 is fitter or equal; keep its structure preferentially
    child = parent1.copy()
    child.conns = {}
    for innov in set(parent1.conns.keys()) | set(parent2.conns.keys()):
        gene = None
        if innov in parent1.conns and innov in parent2.conns:
            gene = random.choice([parent1.conns[innov], parent2.conns[innov]])
            gene = ConnGene(**vars(gene))
        elif innov in parent1.conns:
            gene = ConnGene(**vars(parent1.conns[innov]))
        if gene is not None:
            # disabled genes may remain disabled with small prob
            if (not gene.enabled) and random.random() < 0.75:
                gene.enabled = False
            child.conns[innov] = gene
    # ensure all nodes exist
    for c in child.conns.values():
        if c.in_id not in child.nodes or c.out_id not in child.nodes:
            # copy missing nodes from fitter parent
            src = parent1 if c.in_id in parent1.nodes else parent2
            for nid in [c.in_id, c.out_id]:
                if nid not in child.nodes and nid in src.nodes:
                    child.nodes[nid] = NodeGene(**vars(src.nodes[nid]))
    return child

# ---- Population / evolution ----
class Species:
    def __init__(self, rep: Genome):
        self.rep = rep.copy()
        self.members: List[Tuple[Genome, float]] = []
        self.best_fitness = -1e9
        self.stagnant = 0

class NEATPop:
    def __init__(self, n_inputs, n_outputs, pop_size=100, compat_threshold=3.0, diff_only=False):
        self.db = InnovationDB()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.compat_threshold = compat_threshold
        self.diff_only = diff_only
        self.rng = random.Random(0)

        self.genomes = [self._make_minimal() for _ in range(pop_size)]
        for g in self.genomes:
            # connect each output to random input to bootstrap signal
            for _ in range(2):
                g.add_connection_mutation()

    def _make_minimal(self):
        return Genome(self.n_inputs, self.n_outputs, self.db, self.diff_only)

    def speciate(self):
        species: List[Species] = []
        for g in self.genomes:
            placed = False
            for s in species:
                if g.distance(s.rep) < self.compat_threshold:
                    s.members.append([g, None])
                    placed = True
                    break
            if not placed:
                sp = Species(g)
                sp.members.append([g, None])
                species.append(sp)
        return species

    def evolve(self, fitness_fn: Callable[[Genome], float], generations=50, elite=0.1,
               m_add_conn=0.3, m_add_node=0.05, m_mutate_w=0.8, m_mutate_act=0.05):
        best = None
        best_fit = -1e9
        for gen in range(generations):
            # evaluate
            fits = []
            for g in self.genomes:
                f = fitness_fn(g)
                fits.append(f)
                if f > best_fit:
                    best, best_fit = g.copy(), f

            # speciate
            species = self.speciate()

            # compute adjusted fitness per species
            new_genomes = []
            for s in species:
                # fill fitness for members
                for m in s.members:
                    m[1] = fitness_fn(m[0])
                s.members.sort(key=lambda x: x[1], reverse=True)
                s.best_fitness = max(s.best_fitness, s.members[0][1])

                # elitism
                k_elite = max(1, int(elite * len(s.members)))
                elites = [x[0] for x in s.members[:k_elite]]
                new_genomes.extend([e.copy() for e in elites])

                # offspring count proportional to species mean fitness
                total_fit = sum([x[1] for x in s.members]) + 1e-8
                share = max(0, int(self.pop_size * (total_fit / (sum(fits) + 1e-8))))
                offspring = max(0, share - k_elite)

                # produce offspring
                parents = [x[0] for x in s.members[: max(2, int(0.5 * len(s.members)) )]]
                for _ in range(offspring):
                    p1, p2 = random.sample(parents, 2)
                    # prefer fitter as p1
                    if fitness_fn(p2) > fitness_fn(p1):
                        p1, p2 = p2, p1
                    child = crossover(p1, p2)
                    # mutations
                    if random.random() < m_add_conn: child.add_connection_mutation()
                    if random.random() < m_add_node: child.add_node_mutation()
                    if random.random() < m_mutate_w: child.mutate_weights()
                    if random.random() < m_mutate_act: child.mutate_activations()
                    new_genomes.append(child)

            # fill up population if needed
            while len(new_genomes) < self.pop_size:
                g = random.choice(self.genomes).copy()
                g.mutate_weights()
                new_genomes.append(g)
            self.genomes = new_genomes[:self.pop_size]
        return best, best_fit
        
    def _next_generation(self, fits, elite=0.1,
                         m_add_conn=0.3, m_add_node=0.05,
                         m_mutate_w=0.8, m_mutate_act=0.05):
        """Produce the next generation using precomputed fitness values."""
        best_fit = max(fits)
        best_idx = fits.index(best_fit)
        best = self.genomes[best_idx].copy()

        species = self.speciate()

        new_genomes = []
        for s in species:
            # collect members' fitness
            # members = [(g, fits[self.genomes.index(g)]) for g in s.members]
            members = [(m[0], fits[self.genomes.index(m[0])]) for m in s.members]

            members.sort(key=lambda x: x[1], reverse=True)
            s.best_fitness = max(s.best_fitness, members[0][1])

            # elitism
            k_elite = max(1, int(elite * len(members)))
            elites = [x[0] for x in members[:k_elite]]
            new_genomes.extend([e.copy() for e in elites])

            # offspring count proportional to mean fitness
            total_fit = sum(x[1] for x in members) + 1e-8
            share = max(0, int(self.pop_size * (total_fit / (sum(fits) + 1e-8))))
            offspring = max(0, share - k_elite)

            # breed offspring
            parents = [x[0] for x in members[:max(2, int(0.5 * len(members)))]]
            for _ in range(offspring):
                p1, p2 = random.sample(parents, 2)
                if fits[self.genomes.index(p2)] > fits[self.genomes.index(p1)]:
                    p1, p2 = p2, p1
                child = crossover(p1, p2)
                if random.random() < m_add_conn: child.add_connection_mutation()
                if random.random() < m_add_node: child.add_node_mutation()
                if random.random() < m_mutate_w: child.mutate_weights()
                if random.random() < m_mutate_act: child.mutate_activations()
                new_genomes.append(child)

        while len(new_genomes) < self.pop_size:
            g = random.choice(self.genomes).copy()
            g.mutate_weights()
            new_genomes.append(g)

        self.genomes = new_genomes[:self.pop_size]
        return self


# ---- Serialization helpers ----
def save_genome(g: Genome, path: str):
    data = {
        "n_inputs": g.n_inputs, "n_outputs": g.n_outputs, "diff_only": g.diff_only,
        "nodes": [vars(n) for n in g.nodes.values()],
        "conns": [vars(c) for c in g.conns.values()],
    }
    with open(path, "w") as f:
        json.dump(data, f)

def load_genome(path: str, innovation_db: InnovationDB=None) -> Genome:
    with open(path, "r") as f:
        data = json.load(f)
    if innovation_db is None: innovation_db = InnovationDB()
    g = Genome(data["n_inputs"], data["n_outputs"], innovation_db, data.get("diff_only", False))
    g.nodes = {n["id"]: NodeGene(**n) for n in data["nodes"]}
    g.conns = {c["innov"]: ConnGene(**c) for c in data["conns"]}
    g.next_node_id = max(g.nodes.keys()) + 1
    return g
