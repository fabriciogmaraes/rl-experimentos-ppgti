"""CartPole (Aula 02): funcionamento base do Gymnasium.
Executa o ciclo reset/step com duas políticas sem aprendizado e registra o que o ambiente devolve."""
import numpy as np, pandas as pd, gymnasium as gym

env = gym.make("CartPole-v1")
print("espaço de observação:", env.observation_space)
print("limites teóricos:", env.observation_space.low, env.observation_space.high)
print("espaço de ações:", env.action_space, "| limite de passos:", env.spec.max_episode_steps)

def rodar(politica, n=100, seed=0):
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    o, _ = env.reset(seed=seed)
    linhas = []
    for ep in range(n):
        if ep: o, _ = env.reset()
        total, passos = 0.0, 0
        while True:
            o, r, term, trunc, _ = env.step(politica(o, rng))
            total += r; passos += 1
            if term or trunc:
                causa = ("limite de passos" if trunc else
                         "posição do carrinho" if abs(o[0]) > 2.4 else "ângulo do poste")
                linhas.append((ep, total, passos, causa)); break
    return pd.DataFrame(linhas, columns=["episodio", "retorno", "passos", "causa_termino"])

pols = {
    "aleatória": lambda o, rng: int(rng.integers(2)),
    "heurística (empurrar para o lado em que o poste cai)": lambda o, rng: int(o[2] + 0.5 * o[3] > 0),
}
saida = []
for nome, p in pols.items():
    df = rodar(p); df.insert(0, "politica", nome); saida.append(df)
    print(f"\n{nome}: retorno médio {df.retorno.mean():.1f} ± {df.retorno.std(ddof=1):.1f} "
          f"(mín {df.retorno.min():.0f}, máx {df.retorno.max():.0f})")
    print(df.causa_termino.value_counts().to_string())
pd.concat(saida).to_csv("/mnt/user-data/outputs/cartpole_base.csv", index=False)
