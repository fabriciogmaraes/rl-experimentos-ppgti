"""
DQN e DDQN em Pong e Space Invaders (Atari / ALE), com ORÇAMENTO REDUZIDO para CPU.

Pré-processamento padrão: frame skip 4, escala de cinza, 84x84, pilha de 4 quadros, até 30 "no-ops"
no início, recompensa de treino limitada a {-1, 0, +1} (o retorno registrado é o placar real).
Rede: a arquitetura do artigo original do DQN (Mnih et al., 2013): conv 16 (8x8, passo 4),
conv 32 (4x4, passo 2), camada densa de 256 e uma saída por ação.

DQN  : uma única rede para escolher e avaliar a ação seguinte.
DDQN : rede online + rede alvo; a online escolhe a ação, a alvo avalia.

Uso: python dqn_atari.py pong | spaceinvaders
Variáveis: ATARI_STEPS (passos do agente por execução), N_SEEDS.
"""
import copy
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)
torch.set_num_threads(1)
OUT = "/mnt/user-data/outputs/"
STEPS = int(os.environ.get("ATARI_STEPS", "100000"))
N_SEEDS = int(os.environ.get("N_SEEDS", "2"))

IDS = {"pong": "ALE/Pong-v5", "spaceinvaders": "ALE/SpaceInvaders-v5"}
HP = dict(buffer=50000, batch=32, lr=2.5e-4, gamma=0.99, learning_starts=int(os.environ.get("LEARN_START", "5000")), train_freq=4,
          eps_end=0.1, eps_frac=0.3, target_update=2000, eval_freq=25000, eval_eps=3, eval_eps_greedy=0.05)


def make_env(name):
    env = gym.make(IDS[name], frameskip=1, repeat_action_probability=0.0)
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84, terminal_on_life_loss=False,
                             grayscale_obs=True)
    return FrameStackObservation(env, 4)


class QNet(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4, 16, 8, 4), nn.ReLU(), nn.Conv2d(16, 32, 4, 2), nn.ReLU())
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 9 * 9, 256), nn.ReLU(), nn.Linear(256, n_out))

    def forward(self, x):
        return self.fc(self.conv(x.float() / 255.0))


def evaluate(name, q, seed, rng):
    env = make_env(name)
    env.reset(seed=20_000 + seed)
    rets = []
    for _ in range(HP["eval_eps"]):
        o, _ = env.reset()
        total, done = 0.0, False
        while not done:
            if rng.uniform() < HP["eval_eps_greedy"]:
                a = int(rng.integers(env.action_space.n))
            else:
                with torch.no_grad():
                    a = int(q(torch.from_numpy(np.asarray(o)).unsqueeze(0)).argmax(1).item())
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    env.close()
    return float(np.mean(rets))


def run(name, algo, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(name)
    n_out = env.action_space.n
    q = QNet(n_out)
    target = copy.deepcopy(q) if algo == "ddqn" else None
    opt = torch.optim.Adam(q.parameters(), lr=HP["lr"])
    N = HP["buffer"]
    O = np.zeros((N, 4, 84, 84), np.uint8)
    A = np.zeros(N, np.int64); R = np.zeros(N, np.float32); D = np.zeros(N, np.float32)
    size = ptr = 0

    o, _ = env.reset(seed=seed)
    o = np.asarray(o)
    ep_ret, train_rows, test_rows = 0.0, [], []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        eps = max(HP["eps_end"], 1.0 - (1.0 - HP["eps_end"]) * step / (HP["eps_frac"] * STEPS))
        if rng.uniform() < eps:
            a = int(rng.integers(n_out))
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(o).unsqueeze(0)).argmax(1).item())
        o2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        O[ptr], A[ptr], R[ptr], D[ptr] = o, a, float(np.sign(r)), float(done)
        ptr = (ptr + 1) % N
        size = min(size + 1, N)
        ep_ret += r
        if done:
            train_rows.append((algo, seed, step, ep_ret))
            ep_ret = 0.0
            o2, _ = env.reset()
        o = np.asarray(o2)

        if step >= HP["learning_starts"] and step % HP["train_freq"] == 0:
            last = (ptr - 1) % N
            idx = rng.integers(0, size, HP["batch"])
            idx = np.where(idx == last, (idx - 1) % N, idx)  # o último não tem "próximo" gravado
            nxt_idx = (idx + 1) % N
            bs = torch.from_numpy(O[idx]); bs2 = torch.from_numpy(O[nxt_idx])
            ba, br, bd = torch.from_numpy(A[idx]), torch.from_numpy(R[idx]), torch.from_numpy(D[idx])
            with torch.no_grad():
                if algo == "dqn":
                    nq = q(bs2).max(1).values
                else:
                    a_star = q(bs2).argmax(1, keepdim=True)
                    nq = target(bs2).gather(1, a_star).squeeze(1)
                y = br + HP["gamma"] * (1.0 - bd) * nq
            pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

        if algo == "ddqn" and step % HP["target_update"] == 0:
            target.load_state_dict(q.state_dict())

        if step % HP["eval_freq"] == 0 or step == STEPS:
            test_rows.append((algo, seed, step, evaluate(name, q, seed, rng)))
            print(f"   {name} {algo} semente {seed} passo {step}: teste = {test_rows[-1][3]:.1f} ({time.time()-t0:.0f}s)", flush=True)
    env.close()
    return train_rows, test_rows, time.time() - t0


if __name__ == "__main__":
    name = sys.argv[1]
    os.makedirs("/home/claude/cache_atari", exist_ok=True)
    tr_all, te_all = [], []
    print(f"== {name}: {IDS[name]} | passos={STEPS} | sementes={N_SEEDS}", flush=True)
    for seed in range(N_SEEDS):
        for algo in ("dqn", "ddqn"):
            cache = f"/home/claude/cache_atari/{name}_{algo}_{seed}_{STEPS}.pkl"
            if os.path.exists(cache):
                with open(cache, "rb") as f:
                    tr, te, dt = pickle.load(f)
            else:
                tr, te, dt = run(name, algo, seed)
                with open(cache, "wb") as f:
                    pickle.dump((tr, te, dt), f)
            tr_all += tr; te_all += te
            print(f"{name} {algo:4s} semente {seed}: teste final = {te[-1][3]:.1f} ({dt:.0f}s)", flush=True)
    tag = "" if STEPS == 100000 else f"_{STEPS}"
    pd.DataFrame(tr_all, columns=["algoritmo", "semente", "passo", "retorno"]).to_csv(f"{OUT}{name}_dqn_treino{tag}.csv", index=False)
    pd.DataFrame(te_all, columns=["algoritmo", "semente", "passo", "retorno_teste"]).to_csv(f"{OUT}{name}_dqn_teste{tag}.csv", index=False)
    print("salvo.", flush=True)
