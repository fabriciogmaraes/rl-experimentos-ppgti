"""
Experimentos Toy Text: Q-learning (off-policy) vs SARSA (on-policy)
em Taxi, Cliff Walking e Blackjack (Gymnasium).

Mesma estrutura do experimento do Frozen Lake:
  - várias sementes (10 repetições independentes por algoritmo);
  - política de treino ε-greedy com decaimento linear de ε;
  - avaliação periódica com a política gulosa (desempate aleatório);
  - recompensas salvas em CSV.

Uso:  python toytext_experiments.py taxi | cliff | blackjack
Variáveis de ambiente opcionais (para testes rápidos): N_SEEDS, EP_SCALE.
"""
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import gymnasium as gym

OUT = "/mnt/user-data/outputs/"
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
EP_SCALE = float(os.environ.get("EP_SCALE", "1"))
SEEDS = list(range(N_SEEDS))

CFG = {
    "taxi": dict(id="Taxi-v4", kw={}, n_states=500, episodes=5000, alpha=0.5, gamma=0.95,
                 eps0=0.5, eps1=0.1, decay_frac=1.0, test_freq=100, test_eps=50, bad=-10),
    "cliff": dict(id="CliffWalking-v1", kw={"max_episode_steps": 200}, n_states=48, episodes=2000,
                  alpha=0.5, gamma=0.95, eps0=0.5, eps1=0.1, decay_frac=1.0, test_freq=50,
                  test_eps=20, bad=-100),
    "blackjack": dict(id="Blackjack-v1", kw={}, n_states=32 * 11 * 2, episodes=100000, alpha=0.01,
                      gamma=1.0, eps0=1.0, eps1=0.1, decay_frac=0.5, test_freq=5000, test_eps=2000,
                      bad=None, agg=1000, final_eval=100000),
}


def to_idx(name, obs):
    if name == "blackjack":
        s, d, a = obs
        return (s * 11 + d) * 2 + int(a)
    return int(obs)


def greedy_action(q_row, rng):
    m = q_row.max()
    ids = np.flatnonzero(q_row == m)
    return int(ids[0]) if len(ids) == 1 else int(rng.choice(ids))


def eps_greedy(q_row, eps, rng, n_actions):
    if rng.uniform() < eps:
        return int(rng.integers(n_actions))
    return greedy_action(q_row, rng)


def evaluate(name, env, Q, rng, n_episodes):
    """Política gulosa: retorno médio, taxa de sucesso e passos médios."""
    rets, succ, steps = [], 0, []
    for _ in range(n_episodes):
        o, _ = env.reset()
        s = to_idx(name, o)
        total, n = 0.0, 0
        while True:
            a = greedy_action(Q[s], rng)
            o, r, term, trunc, _ = env.step(a)
            s = to_idx(name, o)
            total += r
            n += 1
            if term or trunc:
                if name == "taxi":
                    succ += int(term and r == 20)
                elif name == "cliff":
                    succ += int(term)
                else:  # blackjack: vitória
                    succ += int(r > 0)
                break
        rets.append(total)
        steps.append(n)
    return float(np.mean(rets)), succ / n_episodes, float(np.mean(steps))


def run(name, algo, seed):
    c = CFG[name]
    E = max(1, int(c["episodes"] * EP_SCALE))
    rng = np.random.default_rng(seed)
    env = gym.make(c["id"], **c["kw"])
    nA = env.action_space.n
    Q = np.zeros((c["n_states"], nA))
    eps = c["eps0"]
    decay = (c["eps0"] - c["eps1"]) / (E * c["decay_frac"])
    agg = c.get("agg")
    train_rows, test_rows = [], []
    b_r = b_steps = b_bad = 0.0
    b_n = 0
    first = True
    test_freq = max(1, int(c["test_freq"] * EP_SCALE)) if EP_SCALE < 1 else c["test_freq"]

    for ep in range(1, E + 1):
        if first:
            o, _ = env.reset(seed=seed)
            first = False
        else:
            o, _ = env.reset()
        s = to_idx(name, o)
        total_r, steps, n_bad = 0.0, 0, 0
        done = False
        a = eps_greedy(Q[s], eps, rng, nA) if algo == "sarsa" else None
        while not done:
            if algo == "qlearning":
                a = eps_greedy(Q[s], eps, rng, nA)
            o2, r, term, trunc, _ = env.step(a)
            s2 = to_idx(name, o2)
            done = term or trunc
            if algo == "sarsa":
                a2 = eps_greedy(Q[s2], eps, rng, nA)
                target = r if term else r + c["gamma"] * Q[s2, a2]
            else:
                target = r if term else r + c["gamma"] * Q[s2].max()
            Q[s, a] += c["alpha"] * (target - Q[s, a])
            if algo == "sarsa":
                a = a2
            s = s2
            total_r += r
            steps += 1
            if c["bad"] is not None and r == c["bad"]:
                n_bad += 1
        eps = max(c["eps1"], eps - decay)

        if agg:
            b_r += total_r; b_steps += steps; b_bad += n_bad; b_n += 1
            if ep % agg == 0:
                train_rows.append((algo, seed, ep, b_r / b_n, b_steps / b_n, b_bad / b_n))
                b_r = b_steps = b_bad = 0.0
                b_n = 0
        else:
            train_rows.append((algo, seed, ep, total_r, steps, n_bad))

        if ep % test_freq == 0:
            ret, succ, st = evaluate(name, env, Q, rng, max(1, int(c["test_eps"])))
            test_rows.append((algo, seed, ep, ret, succ, st))

    final = None
    if c.get("final_eval"):
        ret, succ, st = evaluate(name, env, Q, rng, int(c["final_eval"] * (EP_SCALE if EP_SCALE < 1 else 1)) or 1)
        final = (algo, seed, ret, succ, st)
    env.close()
    return train_rows, test_rows, Q, final


if __name__ == "__main__":
    name = sys.argv[1]
    c = CFG[name]
    print(f"== {name}: {c['id']} | sementes={N_SEEDS} | escala de episódios={EP_SCALE}", flush=True)
    os.makedirs("/home/claude/cache", exist_ok=True)
    tr_all, te_all, fin_all, qs = [], [], [], {}
    t0 = time.time()
    for algo in ("qlearning", "sarsa"):
        for seed in SEEDS:
            cache = f"/home/claude/cache/{name}_{algo}_{seed}.pkl"
            if os.path.exists(cache):
                with open(cache, "rb") as f:
                    tr, te, Q, fin = pickle.load(f)
            else:
                tr, te, Q, fin = run(name, algo, seed)
                with open(cache, "wb") as f:
                    pickle.dump((tr, te, Q, fin), f)
            tr_all += tr
            te_all += te
            qs[f"{algo}_{seed}"] = Q
            if fin:
                fin_all.append(fin)
            print(f"{name} {algo:9s} semente {seed}: retorno de teste final = {te[-1][3]:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    tag = "" if EP_SCALE == 1 else "_rapido"
    cols_tr = ["algoritmo", "semente", "episodio", "recompensa", "passos", "eventos_ruins"]
    if c.get("agg"):
        cols_tr = ["algoritmo", "semente", "episodio_fim_bloco", "recompensa_media", "passos_medios", "eventos_ruins_medios"]
    pd.DataFrame(tr_all, columns=cols_tr).to_csv(f"{OUT}{name}_treino{tag}.csv", index=False)
    pd.DataFrame(te_all, columns=["algoritmo", "semente", "episodio", "retorno_teste", "taxa_sucesso_teste", "passos_medios_teste"]).to_csv(
        f"{OUT}{name}_teste{tag}.csv", index=False)
    if fin_all:
        pd.DataFrame(fin_all, columns=["algoritmo", "semente", "retorno_final", "taxa_vitoria_final", "passos_medios"]).to_csv(
            f"{OUT}{name}_avaliacao_final{tag}.csv", index=False)
    np.savez(f"/home/claude/{name}_q_tables{tag}.npz", **qs)
    print("salvo.", flush=True)
