"""
DQN e DDQN em NumPy (sem PyTorch) para o Mountain Car. Mesmo algoritmo e mesmos hiperparametros de
dqn_classic.py: MLP 2-256-256-3 com ReLU, Adam, perda de Huber, corte de gradiente em 10, replay, ε-greedy linear.

DQN : alvo = r + γ max_a' Q(s', a')            (mesma rede, sem rede alvo)
DDQN: alvo = r + γ Q_alvo(s', argmax_a' Q(s', a'))  (rede online escolhe, rede alvo avalia)

Uso: python dqn_numpy.py mountaincar      (variaveis: N_SEEDS, T_SCALE)
"""
import os, pickle, sys, time
import numpy as np
import pandas as pd
import gymnasium as gym

OUT = "/mnt/user-data/outputs/"
N_SEEDS = int(os.environ.get("N_SEEDS", "2"))
T_SCALE = float(os.environ.get("T_SCALE", "1"))


def mc_scale(o):
    return np.array([(o[0] + 0.3) / 0.9, o[1] / 0.07], dtype=np.float32)


CFG = {
    "mountaincar": dict(id="MountainCar-v0", T=100000, hidden=256, lr=4e-3, gamma=0.98, buffer=10000, batch=128,
                        learning_starts=1000, train_freq=16, grad_steps=8, eps_end=0.07, eps_frac=0.2,
                        target_update=600, eval_freq=5000, eval_eps=5, scale=mc_scale),
}


class MLP:
    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for i, o in zip(sizes[:-1], sizes[1:]):
            lim = 1.0 / np.sqrt(i)  # mesma inicializacao padrao do PyTorch (Linear)
            self.W.append(rng.uniform(-lim, lim, (i, o)).astype(np.float32))
            self.b.append(rng.uniform(-lim, lim, (o,)).astype(np.float32))
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [x for pair in zip(self.W, self.b) for x in pair]

    def forward(self, x, cache=False):
        acts, h = [x], x
        n = len(self.W)
        for k in range(n):
            z = h @ self.W[k] + self.b[k]
            h = np.maximum(z, 0.0) if k < n - 1 else z
            acts.append(h)
        return (h, acts) if cache else h

    def backward(self, acts, dout):
        grads, d = [], dout
        for k in reversed(range(len(self.W))):
            grads = [acts[k].T @ d, d.sum(0)] + grads
            if k > 0:
                d = (d @ self.W[k].T) * (acts[k] > 0)
        return grads

    def step(self, grads, lr, clip=10.0):
        gn = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
        if gn > clip:
            grads = [g * (clip / (gn + 1e-6)) for g in grads]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self.params(), grads, self.m, self.v):
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)

    def copy_from(self, other):
        for p, q in zip(self.params(), other.params()):
            p[...] = q


def evaluate(env, q, c, n):
    rets = []
    for _ in range(n):
        o, _ = env.reset()
        done, total = False, 0.0
        while not done:
            a = int(np.argmax(q.forward(c["scale"](o)[None])[0]))
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
        rets.append(total)
    return float(np.mean(rets))


def run(name, algo, seed):
    c = CFG[name]
    T = max(1, int(c["T"] * T_SCALE))
    rng = np.random.default_rng(seed)
    env, eval_env = gym.make(c["id"]), gym.make(c["id"])
    n_in, n_out = env.observation_space.shape[0], env.action_space.n
    q = MLP([n_in, c["hidden"], c["hidden"], n_out], rng)
    target = MLP([n_in, c["hidden"], c["hidden"], n_out], rng) if algo == "ddqn" else None
    if target is not None:
        target.copy_from(q)
    N = c["buffer"]
    S = np.zeros((N, n_in), np.float32); S2 = S.copy(); A = np.zeros(N, np.int64)
    R = np.zeros(N, np.float32); D = np.zeros(N, np.float32)
    size = ptr = 0
    o, _ = env.reset(seed=seed); eval_env.reset(seed=10_000 + seed)
    s = c["scale"](o)
    ep_ret, train_rows, test_rows = 0.0, [], []
    t0 = time.time()
    for step in range(1, T + 1):
        eps = max(c["eps_end"], 1.0 - (1.0 - c["eps_end"]) * step / (c["eps_frac"] * T))
        a = int(rng.integers(n_out)) if rng.uniform() < eps else int(np.argmax(q.forward(s[None])[0]))
        o2, r, term, trunc, _ = env.step(a)
        s2 = c["scale"](o2)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(term)
        ptr = (ptr + 1) % N; size = min(size + 1, N)
        ep_ret += r; s = s2
        if term or trunc:
            train_rows.append((algo, seed, step, ep_ret)); ep_ret = 0.0
            o, _ = env.reset(); s = c["scale"](o)
        if step >= c["learning_starts"] and step % c["train_freq"] == 0:
            for _ in range(c["grad_steps"]):
                idx = rng.integers(0, size, c["batch"])
                bs, bs2, ba, br, bd = S[idx], S2[idx], A[idx], R[idx], D[idx]
                if algo == "dqn":
                    nxt = q.forward(bs2).max(1)
                else:
                    a_star = q.forward(bs2).argmax(1)
                    nxt = target.forward(bs2)[np.arange(len(idx)), a_star]
                y = br + c["gamma"] * (1.0 - bd) * nxt
                out, acts = q.forward(bs, cache=True)
                pred = out[np.arange(len(idx)), ba]
                dout = np.zeros_like(out)
                dout[np.arange(len(idx)), ba] = np.clip(pred - y, -1.0, 1.0) / len(idx)  # gradiente da perda de Huber
                q.step(q.backward(acts, dout), c["lr"])
        if algo == "ddqn" and step % c["target_update"] == 0:
            target.copy_from(q)
        if step % max(1, int(c["eval_freq"] * (T_SCALE if T_SCALE < 1 else 1))) == 0:
            test_rows.append((algo, seed, step, evaluate(eval_env, q, c, c["eval_eps"])))
    env.close(); eval_env.close()
    return train_rows, test_rows, time.time() - t0


def gradcheck():
    """Confere o gradiente analitico contra diferencas finitas (rede pequena, float64)."""
    rng = np.random.default_rng(0)
    m = MLP([2, 5, 5, 3], rng)
    for k in range(len(m.W)):
        m.W[k] = m.W[k].astype(np.float64); m.b[k] = m.b[k].astype(np.float64)
    x = rng.normal(size=(7, 2)); ya = rng.integers(0, 3, 7); y = rng.normal(size=7)

    def loss():
        out = m.forward(x)[np.arange(7), ya]
        d = out - y
        return float(np.mean(np.where(np.abs(d) <= 1, 0.5 * d * d, np.abs(d) - 0.5)))
    out, acts = m.forward(x, cache=True)
    dout = np.zeros_like(out); dout[np.arange(7), ya] = np.clip(out[np.arange(7), ya] - y, -1, 1) / 7
    grads = m.backward(acts, dout)
    worst = 0.0
    for p, g in zip(m.params(), grads):
        it = np.nditer(p, flags=["multi_index"], op_flags=["readwrite"])
        for _ in it:
            i = it.multi_index; old = p[i]
            p[i] = old + 1e-6; lp = loss(); p[i] = old - 1e-6; lm = loss(); p[i] = old
            worst = max(worst, abs((lp - lm) / 2e-6 - g[i]))
    return worst


if __name__ == "__main__":
    if sys.argv[1] == "gradcheck":
        print("maior erro absoluto do gradiente (analitico vs diferencas finitas):", gradcheck())
        sys.exit()
    name = sys.argv[1]
    os.makedirs("/home/claude/cache_np", exist_ok=True)
    tr_all, te_all = [], []
    print(f"== {name} (NumPy) | sementes={N_SEEDS} | T_SCALE={T_SCALE}", flush=True)
    for seed in range(N_SEEDS):
        for algo in ("dqn", "ddqn"):
            cache = f"/home/claude/cache_np/{name}_{algo}_{seed}_{T_SCALE}.pkl"
            if os.path.exists(cache):
                tr, te, dt = pickle.load(open(cache, "rb"))
            else:
                tr, te, dt = run(name, algo, seed)
                pickle.dump((tr, te, dt), open(cache, "wb"))
            tr_all += tr; te_all += te
            print(f"{name} {algo:4s} semente {seed}: retorno de teste final = {te[-1][3]:.1f} ({dt:.0f}s)", flush=True)
    tag = "" if T_SCALE == 1 else "_rapido"
    pd.DataFrame(tr_all, columns=["algoritmo", "semente", "passo", "retorno"]).to_csv(f"{OUT}{name}_dqn_treino{tag}.csv", index=False)
    pd.DataFrame(te_all, columns=["algoritmo", "semente", "passo", "retorno_teste"]).to_csv(f"{OUT}{name}_dqn_teste{tag}.csv", index=False)
    print("salvo.", flush=True)
