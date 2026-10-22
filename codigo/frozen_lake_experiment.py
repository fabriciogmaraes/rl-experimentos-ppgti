"""
Experimento Frozen Lake: Q-learning (off-policy) vs SARSA (on-policy).

Baseado no notebook de referência da disciplina (Aula 02, slide 56), com
três mudanças pedidas pelo modelo de relatório:
  1) o MESMO mapa para os dois algoritmos (mapa gerado uma vez, com semente);
  2) várias sementes (repetições independentes) por algoritmo;
  3) a recompensa de cada episódio é salva em CSV.

Hiperparâmetros idênticos aos do notebook de referência.
"""
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from collections import deque
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# ---------------- configuração ----------------
MAP_SIZE = 8
P_FROZEN = 0.9
MAP_SEED = 42                # semente do MAPA (fixa, igual para os dois algoritmos)
SEEDS = list(range(10))      # 10 repetições independentes
ALPHA = 0.5
GAMMA = 0.95
EPS_START = 0.5
EPS_FINAL = 0.1
N_EPISODES = 5000
EPS_DECAY = (EPS_START - EPS_FINAL) / N_EPISODES   # decaimento linear por episódio
TEST_FREQ = 100
TEST_EPISODES = 50

DESC = generate_random_map(size=MAP_SIZE, p=P_FROZEN, seed=MAP_SEED)


def make_env():
    return gym.make("FrozenLake-v1", is_slippery=False, desc=DESC)


def greedy_action(q_row, rng):
    """Ação gulosa com desempate aleatório (igual ao notebook)."""
    ids = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(ids))


def eps_greedy(q_row, eps, rng, n_actions):
    if rng.uniform() < eps:
        return int(rng.integers(n_actions))
    return greedy_action(q_row, rng)


def test_policy(env, Q, rng, n_episodes, max_steps):
    """Taxa de sucesso e comprimento médio (só dos sucessos) com política gulosa."""
    wins, lengths = 0.0, []
    for _ in range(n_episodes):
        s, _ = env.reset()
        for t in range(max_steps):
            a = greedy_action(Q[s], rng)
            s, r, term, trunc, _ = env.step(a)
            if term or trunc:
                if r > 0:
                    wins += 1
                    lengths.append(t + 1)
                break
    return wins / n_episodes, (float(np.mean(lengths)) if lengths else np.nan)


def run(algo, seed):
    assert algo in ("qlearning", "sarsa")
    rng = np.random.default_rng(seed)
    env = make_env()
    max_steps = env.spec.max_episode_steps
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_states, n_actions))
    eps = EPS_START
    train_rows, test_rows = [], []
    first_reset = True

    for ep in range(1, N_EPISODES + 1):
        if first_reset:
            s, _ = env.reset(seed=seed)
            first_reset = False
        else:
            s, _ = env.reset()
        total_r, steps = 0.0, 0
        done = False
        a = eps_greedy(Q[s], eps, rng, n_actions) if algo == "sarsa" else None

        while not done:
            if algo == "qlearning":
                a = eps_greedy(Q[s], eps, rng, n_actions)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if algo == "sarsa":
                a2 = eps_greedy(Q[s2], eps, rng, n_actions)
                target = r if term else r + GAMMA * Q[s2, a2]
            else:
                target = r if term else r + GAMMA * Q[s2].max()
            Q[s, a] += ALPHA * (target - Q[s, a])
            if algo == "sarsa":
                a = a2
            s = s2
            total_r += r
            steps += 1

        eps = max(EPS_FINAL, eps - EPS_DECAY)
        train_rows.append((algo, seed, ep, total_r, steps))

        if ep % TEST_FREQ == 0:
            succ, mean_len = test_policy(env, Q, rng, TEST_EPISODES, max_steps)
            test_rows.append((algo, seed, ep, succ, mean_len))

    env.close()
    return train_rows, test_rows, Q


def shortest_path_len():
    """Menor número de passos de S até G (BFS), para comparar com o que foi aprendido."""
    n = MAP_SIZE
    moves = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # esquerda, baixo, direita, cima
    start = (0, 0)
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        if DESC[r][c] == "G":
            return d
        for dr, dc in moves:
            nr, nc = min(max(r + dr, 0), n - 1), min(max(c + dc, 0), n - 1)
            if DESC[nr][nc] != "H" and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None


if __name__ == "__main__":
    print("Mapa (S=início, F=gelo, H=buraco, G=objetivo):")
    for row in DESC:
        print("  ", row)
    print("Caminho mais curto S->G:", shortest_path_len(), "passos")

    all_train, all_test, q_tables = [], [], {}
    t0 = time.time()
    for algo in ("qlearning", "sarsa"):
        for seed in SEEDS:
            tr, te, Q = run(algo, seed)
            all_train += tr
            all_test += te
            q_tables[(algo, seed)] = Q
            print(f"{algo:9s} seed {seed}: sucesso final no teste = {te[-1][3]:.2f}  ({time.time()-t0:.0f}s)")

    pd.DataFrame(all_train, columns=["algoritmo", "semente", "episodio", "recompensa", "passos"]).to_csv(
        "/mnt/user-data/outputs/frozenlake_treino.csv", index=False)
    pd.DataFrame(all_test, columns=["algoritmo", "semente", "episodio", "taxa_sucesso_teste", "passos_medios_sucesso"]).to_csv(
        "/mnt/user-data/outputs/frozenlake_teste.csv", index=False)
    np.savez("/home/claude/q_tables.npz", **{f"{a}_{s}": Q for (a, s), Q in q_tables.items()})
    print("CSVs salvos.")
