"""Linha de base sem aprendizado nos jogos Atari: política aleatória (placar médio de 10 partidas)."""
import numpy as np, pandas as pd, gymnasium as gym, ale_py, os, sys
sys.argv = ["x", "pong"]
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
class d:
    @staticmethod
    def make_env(name):
        ids={"pong":"ALE/Pong-v5","spaceinvaders":"ALE/SpaceInvaders-v5"}
        env=gym.make(ids[name],frameskip=1,repeat_action_probability=0.0)
        env=AtariPreprocessing(env,noop_max=30,frame_skip=4,screen_size=84,terminal_on_life_loss=False,grayscale_obs=True)
        return FrameStackObservation(env,4)
rows = []
for name in ("pong", "spaceinvaders"):
    env = d.make_env(name); env.reset(seed=123); rng = np.random.default_rng(123); rets = []
    for _ in range(10):
        o, _ = env.reset(); total, done = 0.0, False
        while not done:
            o, r, term, trunc, _ = env.step(int(rng.integers(env.action_space.n))); total += r; done = term or trunc
        rets.append(total)
    print(name, "aleatória: média", np.mean(rets), "± dp", np.std(rets, ddof=1), flush=True)
    rows.append((name, np.mean(rets), np.std(rets, ddof=1), len(rets)))
pd.DataFrame(rows, columns=["jogo", "placar_medio_aleatoria", "desvio_padrao", "partidas"]).to_csv("/mnt/user-data/outputs/atari_linha_de_base_aleatoria.csv", index=False)
