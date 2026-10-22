import numpy as np
import frozen_lake_experiment as fl

fail_log = []   # (algo, seed_ep_marker, tipo_falha)
current = {"algo": None, "seed": None, "ep": 0}
orig_test = fl.test_policy

def test_policy_diag(env, Q, rng, n_episodes, max_steps):
    # mesma lógica e MESMO consumo do rng que a versão original; só classifica as falhas
    wins, lengths = 0.0, []
    holes = timeouts = 0
    for _ in range(n_episodes):
        s, _ = env.reset()
        for t in range(max_steps):
            a = fl.greedy_action(Q[s], rng)
            s, r, term, trunc, _ = env.step(a)
            if term or trunc:
                if r > 0:
                    wins += 1; lengths.append(t + 1)
                elif term:
                    holes += 1
                else:
                    timeouts += 1
                break
    fail_log.append((current["algo"], current["seed"], holes, timeouts, wins / n_episodes))
    return wins / n_episodes, (float(np.mean(lengths)) if lengths else np.nan)

fl.test_policy = test_policy_diag
for algo in ("qlearning", "sarsa"):
    for seed in fl.SEEDS:
        current.update(algo=algo, seed=seed)
        fl.run(algo, seed)

for algo in ("qlearning", "sarsa"):
    rows = [r for r in fail_log if r[0] == algo]
    total_fail_eps_holes = sum(r[2] for r in rows)
    total_fail_eps_timeouts = sum(r[3] for r in rows)
    print(f"{algo}: episódios de teste que falharam -> buraco: {total_fail_eps_holes}, estouro de passos (ciclo): {total_fail_eps_timeouts}")
