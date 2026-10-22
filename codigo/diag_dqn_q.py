"""Diagnóstico: evolução dos valores Q no CartPole, DQN (sem rede alvo) vs DDQN (rede alvo + estimador duplo).
Semente 0, 12.000 passos, mesmos hiperparâmetros do experimento principal."""
import copy
import numpy as np, torch, torch.nn as nn, gymnasium as gym
import dqn_classic as d
torch.set_num_threads(1)
c = d.CFG["cartpole"]

def probe(algo, T=12000, seed=0):
    np.random.seed(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    env = gym.make(c["id"]); q = d.QNet(4, 2, c["hidden"])
    tgt = copy.deepcopy(q) if algo == "ddqn" else None
    opt = torch.optim.Adam(q.parameters(), lr=c["lr"])
    N = 20000; S = np.zeros((N, 4), np.float32); S2 = S.copy(); A = np.zeros(N, np.int64)
    R = np.zeros(N, np.float32); D = np.zeros(N, np.float32); size = ptr = 0
    o, _ = env.reset(seed=seed); s = o.astype(np.float32)
    for step in range(1, T + 1):
        eps = max(0.05, 1 - 0.95 * step / (0.2 * 50000))
        a = int(rng.integers(2)) if rng.uniform() < eps else int(q(torch.from_numpy(s)[None]).argmax(1).item())
        o2, r, term, trunc, _ = env.step(a); s2 = o2.astype(np.float32)
        S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = s, a, r, s2, float(term)
        ptr = (ptr + 1) % N; size = min(size + 1, N); s = s2
        if term or trunc:
            o, _ = env.reset(); s = o.astype(np.float32)
        if step >= 1000 and step % 2 == 0:
            idx = rng.integers(0, size, 64)
            bs, bs2 = torch.from_numpy(S[idx]), torch.from_numpy(S2[idx])
            ba, br, bd = torch.from_numpy(A[idx]), torch.from_numpy(R[idx]), torch.from_numpy(D[idx])
            with torch.no_grad():
                nxt = q(bs2).max(1).values if algo == "dqn" else tgt(bs2).gather(1, q(bs2).argmax(1, keepdim=True)).squeeze(1)
                y = br + c["gamma"] * (1 - bd) * nxt
            pred = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
        if algo == "ddqn" and step % 500 == 0:
            tgt.load_state_dict(q.state_dict())
        if step % 2000 == 0:
            with torch.no_grad():
                qq = q(torch.from_numpy(S[:size][np.random.randint(0, size, 512)]))
            print(f"{algo} passo {step}: Q médio {qq.mean():.1f} | Q0-Q1 médio {(qq[:,0]-qq[:,1]).mean():+.2f} | "
                  f"% ação 0 gulosa {(qq.argmax(1)==0).float().mean():.2f} | perda {loss.item():.3f}")

if __name__ == "__main__":
    probe("dqn"); probe("ddqn")
