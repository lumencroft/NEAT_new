import argparse
import sys
import time
import numpy as np
import gym
import slimevolleygym
import multiprocessing as mp
from neat_core import NEATPop, crossover
from visualize import draw_network
import warnings
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def evaluate_saved_genome(genome_path):
    """
    저장된 게놈을 로드하여 SlimeVolley 환경에서 실행합니다.
    """
    print(f"'{genome_path}'에서 게놈을 로드하여 실행합니다...")

    # SlimeVolleyGym은 render() 호출 시 자체 윈도우를 엽니다.
    env = gym.make("SlimeVolley-v0")
    env.render()

    # pickle 파일에서 게놈 로드
    try:
        with open(genome_path, 'rb') as f:
            genome = pickle.load(f)
    except FileNotFoundError:
        print(f"오류: {genome_path} 파일을 찾을 수 없습니다.")
        env.close()
        return
    except Exception as e:
        print(f"게놈 로드 중 오류 발생: {e}")
        env.close()
        return

    obs = env.reset()
    total_reward = 0.0

    while True:
        # 사용자가 렌더링 창을 닫았는지 확인
        if not env.unwrapped.server_process:
            print("렌더링 창이 닫혔습니다.")
            break

        # 게놈 네트워크를 통해 행동 결정
        out = genome.forward(np.asarray(obs, dtype=np.float32))
        action = act_from_output(out)

        # 원본 코드의 evaluate_agent_with_record와 동일하게 4개의 반환 값 사용
        obs, reward, done, _ = env.step(action)

        total_reward += float(reward)
        env.render()  # 렌더링 상태 업데이트
        time.sleep(0.01)  # 사람이 볼 수 있도록 속도 조절

        if done:
            print(f"에피소드 종료. 최종 점수: {total_reward}")
            total_reward = 0.0
            obs = env.reset()

    env.close()

def act_from_output(out):
    left = out[0] < -0.33
    right = out[0] > 0.33
    jump = out[1] > 0.5
    return np.array([left, right, jump], dtype=np.int8)


def evaluate_agent_with_record(args):
    env_id, genome, episodes, max_steps = args
    env = gym.make("SlimeVolley-v0")
    total_score = 0.0
    all_data = []
    all_scores = []

    for ep in range(episodes):
        obs = env.reset()
        g = env.unwrapped.game
        done = False
        steps = 0
        ep_score = 0.0
        data = np.empty((max_steps, 6), dtype=np.float32)

        ai_points = 0
        player_points = 0
        score_history = np.zeros((max_steps, 2), dtype=np.int32)

        while not done and steps < max_steps:
            out = genome.forward(np.asarray(obs, dtype=np.float32))
            action = act_from_output(out)
            step = env.step(action)
            
            obs, reward, done, _ = step

            ep_score += float(reward)

            if reward == 1:
                player_points += 1
            elif reward == -1:
                ai_points += 1

            score_history[steps] = [player_points, ai_points]

            data[steps] = [
                g.ball.x, g.ball.y,
                g.agent_left.x, g.agent_left.y,
                g.agent_right.x, g.agent_right.y,
            ]
            steps += 1

            if ai_points >= 5 or player_points >= 5:
                break

        # if ep_score > 0:
        #     ep_score += 4 - steps / max_steps
        # elif ep_score > -3:
        #     ep_score += 3 + steps / max_steps
        # else:
        #     ep_score = steps / max_steps

        total_score += ep_score
        all_data.append((data[:steps], score_history[:steps]))
        all_scores.append(player_points)

    env.close()
    best_idx = int(np.argmax(all_scores))
    return total_score / episodes, all_data[best_idx]


def evaluate_population_parallel_with_record(genomes, episodes=2, max_steps=2000, n_workers=None):
    if n_workers is None:
        n_workers = mp.cpu_count()
    args_list = [("SlimeVolley-v0", g, episodes, max_steps) for g in genomes]
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(evaluate_agent_with_record, args_list)
    fitnesses = [r[0] for r in results]
    records = [r[1] for r in results]
    return fitnesses, records


def create_animation(record, filename="best_agent_simulation.mp4"):
    data, score_history = record
    ball_xs, ball_ys, ai_xs, ai_ys, player_xs, player_ys = data.T

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-25, 25)
    ax.set_ylim(0, 20)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("SlimeVolley Simulation (AI vs Player)")

    entities = [
        ("Ball", "ro", 25, ball_xs, ball_ys),
        ("AI", "go", 45, ai_xs, ai_ys),
        ("Player", "bo", 45, player_xs, player_ys),
    ]
    plots = [ax.plot([], [], style, markersize=size, label=label)[0]
             for label, style, size, *_ in entities]
    ax.legend()

    score_text = ax.text(0, 19, "0 : 0", ha="center", va="center",
                         fontsize=20, fontweight="bold", color="black")

    def init():
        for plot in plots:
            plot.set_data([], [])
        score_text.set_text("0 : 0")
        return plots + [score_text]

    def animate(i):
        for plot, (_, _, _, xs, ys) in zip(plots, entities):
            plot.set_data([xs[i]], [ys[i]])
        ai_score, player_score = score_history[i]
        score_text.set_text(f"{player_score} : {ai_score}")
        return plots + [score_text]

    ani = animation.FuncAnimation(fig, animate, frames=len(data),
                                  init_func=init, blit=True, interval=20)
    ani.save(filename, writer="ffmpeg", fps=50)
    plt.close(fig)


def evaluate_generation(pop, episodes, max_steps, workers):
    fitnesses, record_data = evaluate_population_parallel_with_record(
        pop.genomes, episodes=episodes, max_steps=max_steps, n_workers=workers
    )
    avg_fit = float(np.mean(fitnesses))
    best_fit_now = float(np.max(fitnesses))
    worst_fit = float(np.min(fitnesses))
    best_idx = int(np.argmax(fitnesses))
    best_genome = pop.genomes[best_idx]
    return fitnesses, record_data, avg_fit, best_fit_now, worst_fit, best_idx, best_genome


def evolve_next_generation(pop, fitnesses, champion):
    species = pop.speciate()
    new_genomes = []

    if champion is not None:
        new_genomes.append(champion.copy())

    fit_cache = {id(g): f for g, f in zip(pop.genomes, fitnesses)}

    for s in species:
        s.members.sort(key=lambda m: fit_cache.get(id(m[0])), reverse=True)
        k_elite = max(1, int(0.20 * len(s.members)))
        elites = [m[0].copy() for m in s.members[:k_elite]]
        new_genomes.extend(elites)

    global_elite_n = max(1, int(0.05 * pop.pop_size))
    sorted_by_fit = sorted(pop.genomes, key=lambda g: fit_cache[id(g)], reverse=True)
    top_elites = [g.copy() for g in sorted_by_fit[:global_elite_n]]
    new_genomes.extend(top_elites)

    rng = np.random.default_rng()
    while len(new_genomes) < pop.pop_size:
        p1, p2 = rng.choice(pop.genomes, 2, replace=True)
        if fit_cache[id(p2)] > fit_cache[id(p1)]:
            p1, p2 = p2, p1
        child = crossover(p1, p2)
        if rng.random() < 0.4: child.add_connection_mutation()
        if rng.random() < 0.1: child.add_node_mutation()
        if rng.random() < 0.7: child.mutate_weights()
        if rng.random() < 0.05: child.mutate_activations()
        new_genomes.append(child)

    pop.genomes = new_genomes[:pop.pop_size]


def run_evolution(pop, episodes, max_steps, workers, target_score):
    start_time = time.time()
    generation = 0
    champion = None
    champion_fit = -1e9

    while True:
        generation += 1
        fitnesses, record_data, avg_fit, best_fit_now, worst_fit, best_idx, best_genome = evaluate_generation(
            pop, episodes, max_steps, workers
        )

        if best_fit_now > champion_fit:
            champion_fit = best_fit_now
            champion = best_genome.copy()
            if champion_fit>5:
                filename = f"best_{champion_fit:.2f}_gen{generation:04d}.mp4"
                create_animation(record_data[best_idx], filename)
                print(f"\nSaved video {filename}")
                draw_network(champion, fname="best_network.png")
                # print(f"Best Fitness: {best_fit:.2f}")
                print("Saved network viz to best_network.png")

                # --- 챔피언 게놈 저장 로직 추가 ---
                save_filename = "champion_genome.pkl"
                with open(save_filename, 'wb') as f:
                    pickle.dump(champion, f)
                print(f"챔피언 게놈을 {save_filename}에 저장했습니다.")

        sys.stdout.write(
            f"\r[GEN {generation:04d}] Avg: {avg_fit:6.2f}  Best(now): {best_fit_now:6.2f}  Best(ever): {champion_fit:6.2f}  Worst: {worst_fit:6.2f}"
        )
        sys.stdout.flush()

        if avg_fit >= target_score:
            print(f"\nTarget achieved! Avg {avg_fit:.2f} ≥ {target_score:.2f}")
            break

        evolve_next_generation(pop, fitnesses, champion)

    elapsed = time.time() - start_time
    print(f"\nEvolution finished in {elapsed:.2f}s")
    return champion, champion_fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=720)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--target_score", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--eval", type=str, default=None,
                    help="훈련 대신 평가할 저장된 게놈 파일(.pkl)의 경로")
    args = ap.parse_args()

    # --- 평가 모드 실행 ---
    if args.eval:
        evaluate_saved_genome(args.eval)
    # --- 기존 훈련 모드 실행 ---
    else:
        env = gym.make("SlimeVolley-v0")
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        n_in = obs.shape[0]
        n_out = 2
        env.close()

        pop = NEATPop(
            n_inputs=n_in,
            n_outputs=n_out,
            pop_size=args.pop,
            compat_threshold=3.0,
            diff_only=False
        )

        print("[INFO] Evolution started (integer score-based)...")
        champion, best_fit = run_evolution(
            pop,
            episodes=args.episodes,
            max_steps=args.max_steps,
            workers=args.workers,
            target_score=args.target_score
        )
        


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
