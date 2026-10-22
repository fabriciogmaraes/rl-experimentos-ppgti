"""Linhas de base do Blackjack (sem aprendizado), para dar contexto ao retorno médio dos agentes."""
import numpy as np, pandas as pd, gymnasium as gym

def avaliar(politica, n=100000, seed=12345):
    env = gym.make("Blackjack-v1")
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    rets = []
    for _ in range(n):
        (s, d, a), _ = env.reset()
        while True:
            act = politica(s, d, a, rng)
            (s, d, a), r, term, trunc, _ = env.step(act)
            if term or trunc:
                rets.append(r); break
    rets = np.array(rets)
    return rets.mean(), (rets > 0).mean(), (rets == 0).mean(), (rets < 0).mean()

pols = {
    "pedir até somar 17 (parar com 17 ou mais)": lambda s, d, a, rng: int(s < 17),
    "aleatória (50% pedir)": lambda s, d, a, rng: int(rng.integers(2)),
}
rows = []
for nome, p in pols.items():
    m, w, e, l = avaliar(p)
    rows.append((nome, m, w, e, l))
    print(f"{nome}: retorno médio {m:.4f} | vitórias {w:.3f} | empates {e:.3f} | derrotas {l:.3f}")
pd.DataFrame(rows, columns=["politica", "retorno_medio", "vitorias", "empates", "derrotas"]).to_csv(
    "/mnt/user-data/outputs/blackjack_linhas_de_base.csv", index=False)
