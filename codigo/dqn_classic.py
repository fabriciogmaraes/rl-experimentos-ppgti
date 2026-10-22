"""
DQN e DDQN em ambientes de Controle Clássico do Gymnasium (CartPole-v1, MountainCar-v0).

- DQN  (Mnih et al., 2013): uma única rede Q para escolher e avaliar a ação do estado seguinte,
  com memória de replay (sem rede alvo), como no pseudocódigo do slide da Aula 03.
- DDQN (van Hasselt, Guez e Silver, 2016): rede online + rede alvo; a rede online ESCOLHE a melhor
  ação do estado seguinte e a rede alvo a AVALIA (desacoplamento que reduz a superestimação).

Os dois usam exatamente os mesmos hiperparâmetros, sementes e ambiente; só muda o alvo.

Uso: python dqn_classic.py cartpole | mountaincar
Variáveis opcionais: N_SEEDS, T_SCALE (para testes rápidos).
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

torch.set_num_threads(1)
OUT = "/mnt/user-data/outputs/"
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
T_SCALE = float(os.environ.get("T_SCALE", "1"))


def mc_scale(o):  # normaliza posição e velocidade do Mountain Car
    return np.array([(o[0] + 0.3) / 0.9, o[1] / 0.07], dtype=np.float32)


CFG = {
    "cartpole": dict(id="CartPole-v1", T=50000, hidden=128, lr=1e-3, gamma=0.99, buffer=50000, batch=64,
                     learning_starts=1000, train_freq=2, grad_steps=1, eps_end=0.05, eps_frac=0.2,
                     target_update=500, eval_freq=2500, eval_eps=10, scale=None),
    "mountaincar": dict(id="MountainCar-v0", T=100000, hidden=256, lr=4e-3, gamma=0.98, buffer=10000, batch=128,
                        learning_starts=1000, train_freq=16, grad_steps=8, eps_end=0.07, eps_frac=0.2,
                        target_update=600, eval_freq=5000, eval_eps=5, scale=mc_scale),
}


class QNet(nn.Module):
    def __init__(self, n_in, n_out, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_out))

    def forward(self, x):
        return self.net(x)


def evaluate(env, q, c, n):
    rets = []
    for _ in range(n):
        o, _ = env.reset()
        done, total = False, 0.0
        while not done:
            x = c["scale"](o) if c["scale"] else o.astype(np.float32)
            with torch.no_grad():
                a = int(q(torch.from_numpy(x).unsqueeze(0)).argmax(1).item())
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets))


def run(name, algo, seed):
    c = CFG[name]
    T = max(1, int(c["T"] * T_SCALE))
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, eval_env = gym.make(c["id"]), gym.make(c["id"])
    n_in = env.observation_space.shape[0]
    n_out = env.action_space.n
    q = QNet(n_in, n_out, c["hidden"])
    target = copy.deepcopy(q) if algo == "ddqn" else None
    opt = torch.optim.Adam(q.parameters(), lr=c["lr"])

    N = c["buffer"]
    S = np.zeros((N, n_in), np.float32); S2 = np.zeros((N, n_in), np.float32)
    A = np.zeros(N, np.int64); R = np.zeros(N, np.float32); D = np.zeros(N, np.float32)
    size = ptr = 0

    def sc(o):
        return c["scale"](o) if c["scale"] else o.astype(np.float32)

    o, _ = env.reset(seed=seed)
    eval_env.reset(seed=10_000 + seed)
    s = sc(o)
    ep_ret, train_rows, test_rows = 0.0, [], []
    t0 = time.time()

    for step in range(1, T + 1):
        eps = max(c["eps_end"], 1.0 - (1.0 - c["eps_end"]) * step / (c["eps_frac"] * T))
        if rng.uniform() < eps:
            a = int(rng.integers(n_out))
        else:
            with torch.no_grad():
                a = int(q(torch.from_numpy(s).unsqueeze(0)).argmax(1).item())
        o2, r, term, trunc, _ = env.step(a)
        s2 = sc(o2)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(term)
        ptr = (ptr + 1) % N
        size = min(size + 1, N)
        ep_ret += r
        s = s2
        if term or trunc:
            train_rows.append((algo, seed, step, ep_ret))
            ep_ret = 0.0
            o, _ = env.reset()
            s = sc(o)

        if step >= c["learning_starts"] and step % c["train_freq"] == 0:
            for _ in range(c["grad_steps"]):
                idx = rng.integers(0, size, c["batch"])
                bs, bs2 = torch.from_numpy(S[idx]), torch.from_numpy(S2[idx])
                ba, br, bd = torch.from_numpy(A[idx]), torch.from_numpy(R[idx]), torch.from_numpy(D[idx])
                with torch.no_grad():
                    if algo == "dqn":
                        nxt = q(bs2).max(1).values
                    else:
                        a_star = q(bs2).argmax(1, keepdim=True)
                        nxt = target(bs2).gather(1, a_star).squeeze(1)
                    y = br + c["gamma"] * (1.0 - bd) * nxt
                pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()

        if algo == "ddqn" and step % c["target_update"] == 0:
            target.load_state_dict(q.state_dict())

        if step % max(1, int(c["eval_freq"] * (T_SCALE if T_SCALE < 1 else 1))) == 0:
            test_rows.append((algo, seed, step, evaluate(eval_env, q, c, c["eval_eps"])))

    env.close(); eval_env.close()
    return train_rows, test_rows, time.time() - t0


if __name__ == "__main__":
    name = sys.argv[1]
    lr_env = os.environ.get("LR")           # opcional: sensibilidade à taxa de aprendizado
    if lr_env:
        CFG[name]["lr"] = float(lr_env)
    lrtag = f"_lr{lr_env}" if lr_env else ""
    os.makedirs("/home/claude/cache_dqn", exist_ok=True)
    tr_all, te_all = [], []
    print(f"== {name}: {CFG[name]['id']} | sementes={N_SEEDS} | T_SCALE={T_SCALE}", flush=True)
    for algo in ("dqn", "ddqn"):
        for seed in range(N_SEEDS):
            cache = f"/home/claude/cache_dqn/{name}_{algo}_{seed}_{T_SCALE}{lrtag}.pkl"
            if os.path.exists(cache):
                with open(cache, "rb") as f:
                    tr, te, dt = pickle.load(f)
            else:
                tr, te, dt = run(name, algo, seed)
                with open(cache, "wb") as f:
                    pickle.dump((tr, te, dt), f)
            tr_all += tr; te_all += te
            print(f"{name} {algo:4s} semente {seed}: retorno de teste final = {te[-1][3]:.1f} ({dt:.0f}s)", flush=True)
    tag = ("" if T_SCALE == 1 else "_rapido") + lrtag
    pd.DataFrame(tr_all, columns=["algoritmo", "semente", "passo", "retorno"]).to_csv(f"{OUT}{name}_dqn_treino{tag}.csv", index=False)
    pd.DataFrame(te_all, columns=["algoritmo", "semente", "passo", "retorno_teste"]).to_csv(f"{OUT}{name}_dqn_teste{tag}.csv", index=False)
    print("salvo.", flush=True)
