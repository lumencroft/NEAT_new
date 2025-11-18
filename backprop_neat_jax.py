import argparse, json
import numpy as np

# neat_core.py, datasets.py, visualize.py 파일이 필요합니다.
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
    """(가중치 수, 바이어스 수)를 반환합니다."""
    conns = [c for c in genome.conns.values() if c.enabled]
    n_bias = sum(1 for n in genome.nodes.values() if n.type != "in")
    return len(conns), n_bias

# *** 수정된 함수 (JAX 그래디언트 추적) ***
def jax_forward(genome: Genome, theta, X):
    # 정적인 구조 정보 (Genome)
    conns = [c for c in genome.conns.values() if c.enabled]
    n_w, n_b = param_shapes(genome)
    
    # theta (JAX 배열)를 JAX 슬라이싱으로 분해
    w = theta[:n_w]
    b = theta[n_w : n_w + n_b]

    # 구조 -> 인덱스 매핑 (파이썬 딕셔너리, 정적)
    conn_map = {} # (in_id, out_id) -> w 배열의 인덱스
    for i, c in enumerate(conns):
        conn_map[(c.in_id, c.out_id)] = i
    
    bias_map = {} # node_id -> b 배열의 인덱스
    idx = 0
    node_order = sorted(genome.nodes.keys())
    for nid in node_order:
        n = genome.nodes[nid]
        if n.type != "in":
            bias_map[nid] = idx
            idx += 1
    
    max_node_id = node_order[-1] if node_order else 0

    def single(x):
        # 노드 값 저장을 위해 JAX 배열 사용
        values = jnp.zeros(max_node_id + 1)
        
        # 입력 값 설정
        for i in range(genome.n_inputs):
            values = values.at[i].set(x[i])
        
        # 정렬된 노드 순서대로 계산
        for nid in node_order:
            n = genome.nodes[nid]
            if n.type == "in":
                continue
            
            # 가중치 합산
            inc = 0.0
            for (i, j), w_idx in conn_map.items():
                if j == nid:
                    # values[i] (JAX 트레이서) * w[w_idx] (JAX 트레이서)
                    inc += values[i] * w[w_idx]
                    
            # 바이어스 추가
            b_idx = bias_map[nid]
            z = inc + b[b_idx] # b[b_idx] (JAX 트레이서)
            
            # 활성화 함수
            if n.activation == "relu":
                a = jnp.maximum(0.0, z)
            elif n.activation == "tanh":
                a = jnp.tanh(z)
            elif n.activation == "sigmoid":
                a = 1.0 / (1.0 + jnp.exp(-z))
            else:  # identity
                a = z
                
            # 활성화 값 저장
            values = values.at[nid].set(a)
            
        # 출력 값 수집
        outs = []
        for j in range(genome.n_outputs):
            outs.append(values[genome.n_inputs + j])
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

# *** 수정된 함수 (PRNGKey 인자) ***
def train_weights(key: jax.Array, genome: Genome, X, y, steps=300, lr=0.05):
    n_w, n_b = param_shapes(genome)
    # key = jax.random.PRNGKey(0) # <-- 하드코딩된 시드 삭제
    
    # 전달받은 key 사용
    theta = jax.random.normal(key, (n_w + n_b,)) * 0.5
    
    for _ in range(steps):
        theta = sgd_step(genome, theta, X, y, lr)
    loss = loss_fn(genome, theta, X, y)
    return theta, float(loss)

# *** 수정된 함수 (PRNGKey 인자) ***
def fitness_of_genome(key: jax.Array, genome, X, y, train_steps, lr):
    # key를 train_weights로 전달
    theta, loss = train_weights(key, genome, X, y, steps=train_steps, lr=lr)
    return -loss

# *** 수정된 함수 (PRNGKey 관리) ***
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

    # --- 마스터 키 생성 ---
    main_key = jax.random.PRNGKey(42) # 원하는 시드 값 사용

    best = None
    best_fit = -1e9
    for gen in range(args.generations):
        
        # --- 세대별 키 분할 ---
        main_key, gen_key = jax.random.split(main_key)
        # 인구수(pop_size)만큼 개별 키 생성
        genome_keys = jax.random.split(gen_key, len(pop.genomes))

        fits = []
        # enumerate를 사용하여 각 유전체에 고유 키 할당
        for i, g in enumerate(pop.genomes):
            # 고유 키(genome_keys[i])를 전달
            f = fitness_of_genome(genome_keys[i], g, X, y, args.train_steps, args.lr)
            fits.append(f)
            if f > best_fit:
                best, best_fit = g, f
        
        # --- 수정된 로깅 ---
        fits_arr = np.array(fits)
        print(f"Gen {gen} | Best (so far): {best_fit:.4f} | Gen Max: {fits_arr.max():.4f} | Gen Mean: {fits_arr.mean():.4f}")
        
        pop = pop._next_generation(fits)

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