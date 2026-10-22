"""Análise e gráficos: python analyze_toytext.py cliff | taxi | blackjack"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

OUT = "/mnt/user-data/outputs/"
NAMES = {"qlearning": "Q-learning", "sarsa": "SARSA"}
COL = {"qlearning": "#1f77b4", "sarsa": "#e8710a"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{nd}f} ± {np.nanstd(x, ddof=1):.{nd}f}"


def band(ax, df, col, window=None, clip=None, label=True):
    for a in NAMES:
        piv = df[df.algoritmo == a].pivot(index="episodio", columns="semente", values=col)
        if window:
            piv = piv.rolling(window).mean().dropna()
        m, s = piv.mean(axis=1), piv.std(axis=1, ddof=1)
        lo, hi = m - s, m + s
        if clip:
            lo, hi = lo.clip(*clip), hi.clip(*clip)
        ax.plot(m.index, m.values, color=COL[a], lw=2, label=NAMES[a])
        ax.fill_between(m.index, lo, hi, color=COL[a], alpha=0.2)


def mw(x, y):
    r = mannwhitneyu(x, y)
    return f"U = {r.statistic:.1f}, p = {r.pvalue:.4f}"


def first_ckpt(te, col, thr):
    out = {}
    for a in NAMES:
        v = []
        for s, g in te[te.algoritmo == a].groupby("semente"):
            ok = g[g[col] >= thr].episodio
            v.append(ok.min() if len(ok) else np.nan)
        out[a] = np.array(v, float)
    return out


# ------------------------------------------------------------------ CLIFF
def cliff():
    tr = pd.read_csv(OUT + "cliff_treino.csv")
    te = pd.read_csv(OUT + "cliff_teste.csv")
    last = te.episodio.max()
    print(f"== Cliff Walking: política gulosa no episódio {last} ==")
    for a in NAMES:
        d = te[(te.algoritmo == a) & (te.episodio == last)]
        print(NAMES[a], "retorno", ms(d.retorno_teste, 2), "| sucesso", ms(d.taxa_sucesso_teste, 2),
              "| valores por semente:", sorted(d.retorno_teste.round(0).astype(int).tolist()))
    print("\n== Treino (ε-greedy), últimos 100 episódios ==")
    for a in NAMES:
        d = tr[(tr.algoritmo == a) & (tr.episodio > last - 100)].groupby("semente")[["recompensa", "eventos_ruins", "passos"]].mean()
        print(NAMES[a], "retorno", ms(d.recompensa, 2), "| quedas/episódio", ms(d.eventos_ruins, 3), "| passos", ms(d.passos, 2))
    print("\n== Treino, todo o período ==")
    for a in NAMES:
        d = tr[tr.algoritmo == a].groupby("semente")[["recompensa", "eventos_ruins"]].mean()
        print(NAMES[a], "retorno", ms(d.recompensa, 2), "| quedas/episódio", ms(d.eventos_ruins, 3))
    fc = first_ckpt(te, "taxa_sucesso_teste", 1.0)
    for a in NAMES:
        print(f"primeiro checkpoint com 100% de sucesso — {NAMES[a]}: {np.nanmean(fc[a]):.0f} ± {np.nanstd(fc[a], ddof=1):.0f} "
              f"(em {int(np.sum(~np.isnan(fc[a])))}/10 sementes)")
    x = tr[(tr.algoritmo == "qlearning") & (tr.episodio > last - 100)].groupby("semente").recompensa.mean()
    y = tr[(tr.algoritmo == "sarsa") & (tr.episodio > last - 100)].groupby("semente").recompensa.mean()
    print("Mann-Whitney retorno de treino (últimos 100):", mw(x, y))
    x = tr[(tr.algoritmo == "qlearning") & (tr.episodio > last - 100)].groupby("semente").eventos_ruins.mean()
    y = tr[(tr.algoritmo == "sarsa") & (tr.episodio > last - 100)].groupby("semente").eventos_ruins.mean()
    print("Mann-Whitney quedas/episódio (últimos 100):", mw(x, y))
    # falhas do teste guloso
    for a in NAMES:
        d = te[(te.algoritmo == a)]
        print(NAMES[a], "checkpoints com sucesso < 1 no teste:", int((d.taxa_sucesso_teste < 1).sum()), "de", len(d))

    # figura 1: curvas
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    band(axs[0], tr, "recompensa", window=50, clip=(-250, 0))
    axs[0].set_ylim(-250, 5)
    axs[0].set_title("(a) Retorno durante o treino\n(média móvel de 50 episódios)")
    axs[0].set_xlabel("Episódio de treino"); axs[0].set_ylabel("Retorno por episódio"); axs[0].legend(loc="lower right")
    band(axs[1], te, "retorno_teste", clip=(-210, 0))
    axs[1].set_ylim(-210, 5)
    axs[1].set_title("(b) Retorno da política gulosa no teste")
    axs[1].set_xlabel("Episódio de treino"); axs[1].set_ylabel("Retorno por episódio")
    fig.suptitle("Cliff Walking: média ± desvio padrão em 10 sementes")
    fig.tight_layout(); fig.savefig(OUT + "cliff_fig1_curvas.png", dpi=200); plt.close(fig)

    # figura 2: caminho guloso (primeira semente em que os dois chegam ao objetivo)
    Qs = np.load("/home/claude/cliff_q_tables.npz")
    fin = te[te.episodio == last].pivot(index="semente", columns="algoritmo", values="taxa_sucesso_teste")
    seed = int(fin[(fin.qlearning == 1) & (fin.sarsa == 1)].index[0])
    moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # cima, direita, baixo, esquerda
    fig, axs = plt.subplots(2, 1, figsize=(10, 5.6))
    for ax, a in zip(axs, NAMES):
        Q = Qs[f"{a}_{seed}"]
        s, path = 36, [36]
        while s != 47 and len(path) < 60:
            r, c = divmod(s, 12)
            dr, dc = moves[int(np.argmax(Q[s]))]
            r, c = min(max(r + dr, 0), 3), min(max(c + dc, 0), 11)
            s = r * 12 + c
            path.append(s)
            if s in path[:-1]:
                break
        for r in range(4):
            for c in range(12):
                face = "#2b2b2b" if (r == 3 and 1 <= c <= 10) else ("#aec7e8" if (r, c) == (3, 0) else ("#2ca02c" if (r, c) == (3, 11) else "#f2f6fa"))
                ax.add_patch(Rectangle((c, 3 - r), 1, 1, facecolor=face, edgecolor="#888", lw=0.8))
        ax.text(0.5, 0.5, "S", ha="center", va="center", fontsize=13)
        ax.text(11.5, 0.5, "G", ha="center", va="center", fontsize=13, color="white", fontweight="bold")
        ax.text(5.5, 0.5, "penhasco", ha="center", va="center", fontsize=12, color="white")
        xs = [(p % 12) + 0.5 for p in path]
        ys = [3 - (p // 12) + 0.5 for p in path]
        ax.plot(xs, ys, "-o", color=COL[a], lw=3, ms=5)
        ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{NAMES[a]} (semente {seed}): caminho guloso de {len(path) - 1} passos")
    fig.tight_layout(); fig.savefig(OUT + "cliff_fig2_caminho.png", dpi=200); plt.close(fig)
    print("semente usada na figura do caminho:", seed)


# ------------------------------------------------------------------ TAXI
def taxi_referencia():
    """Retorno médio ótimo (sem desconto) por iteração de valor sobre o modelo P do Taxi."""
    import gymnasium as gym
    env = gym.make("Taxi-v4").unwrapped
    P, nS, nA = env.P, env.observation_space.n, env.action_space.n
    V = np.zeros(nS)
    for _ in range(2000):
        Vn = np.array([max(sum(p * (r + 0.99 * V[s2] * (not t)) for p, s2, r, t in P[s][a]) for a in range(nA)) for s in range(nS)])
        if np.max(np.abs(Vn - V)) < 1e-9:
            break
        V = Vn
    pol = [int(np.argmax([sum(p * (r + 0.99 * V[s2] * (not t)) for p, s2, r, t in P[s][a]) for a in range(nA)])) for s in range(nS)]
    dist = env.initial_state_distrib
    tot = 0.0
    for s0 in np.flatnonzero(dist > 0):
        s, ret = int(s0), 0.0
        for _ in range(200):
            p, s2, r, t = P[s][pol[s]][0]
            ret += r; s = s2
            if t:
                break
        tot += dist[s0] * ret
    return tot


def taxi():
    tr = pd.read_csv(OUT + "taxi_treino.csv")
    te = pd.read_csv(OUT + "taxi_teste.csv")
    last = te.episodio.max()
    ref = taxi_referencia()
    print(f"== Taxi: retorno médio da política ótima (iteração de valor): {ref:.3f}")
    print(f"== Taxi: política gulosa no episódio {last} ==")
    for a in NAMES:
        d = te[(te.algoritmo == a) & (te.episodio == last)]
        print(NAMES[a], "retorno", ms(d.retorno_teste, 2), "| sucesso", ms(d.taxa_sucesso_teste, 3), "| passos", ms(d.passos_medios_teste, 2),
              "| retornos por semente:", sorted(d.retorno_teste.round(2).tolist()))
    print("\n== Treino (ε-greedy), últimos 100 episódios ==")
    for a in NAMES:
        d = tr[(tr.algoritmo == a) & (tr.episodio > last - 100)].groupby("semente")[["recompensa", "eventos_ruins", "passos"]].mean()
        print(NAMES[a], "retorno", ms(d.recompensa, 2), "| ações ilegais/episódio", ms(d.eventos_ruins, 3), "| passos", ms(d.passos, 2))
    print("\n== Treino, todo o período ==")
    for a in NAMES:
        d = tr[tr.algoritmo == a].groupby("semente")[["recompensa", "eventos_ruins"]].mean()
        print(NAMES[a], "retorno", ms(d.recompensa, 2), "| ações ilegais/episódio", ms(d.eventos_ruins, 3))
    for thr in (0.9, 1.0):
        fc = first_ckpt(te, "taxa_sucesso_teste", thr)
        for a in NAMES:
            print(f"primeiro checkpoint com sucesso >= {thr} — {NAMES[a]}: {np.nanmean(fc[a]):.0f} ± {np.nanstd(fc[a], ddof=1):.0f} "
                  f"(em {int(np.sum(~np.isnan(fc[a])))}/10 sementes)")
    x = tr[(tr.algoritmo == "qlearning") & (tr.episodio > last - 100)].groupby("semente").recompensa.mean()
    y = tr[(tr.algoritmo == "sarsa") & (tr.episodio > last - 100)].groupby("semente").recompensa.mean()
    print("Mann-Whitney retorno de treino (últimos 100):", mw(x, y))
    x = te[(te.algoritmo == "qlearning") & (te.episodio == last)].retorno_teste
    y = te[(te.algoritmo == "sarsa") & (te.episodio == last)].retorno_teste
    print("Mann-Whitney retorno de teste final:", mw(x, y))
    for a in NAMES:
        d = te[(te.algoritmo == a) & (te.episodio > 1000)]
        print(NAMES[a], "após ep.1000: checkpoints com sucesso < 1:", int((d.taxa_sucesso_teste < 1).sum()), "de", len(d))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    band(axs[0], tr, "recompensa", window=100, clip=(-300, 25))
    axs[0].set_ylim(-300, 25)
    axs[0].set_title("(a) Retorno durante o treino\n(média móvel de 100 episódios)")
    axs[0].set_xlabel("Episódio de treino"); axs[0].set_ylabel("Retorno por episódio"); axs[0].legend(loc="lower right")
    band(axs[1], te, "retorno_teste", clip=(-300, 25))
    axs[1].axhline(ref, color="gray", ls="--", lw=1.2)
    axs[1].text(last * 0.98, ref - 22, f"ótimo ≈ {ref:.1f}", ha="right", color="gray")
    axs[1].set_ylim(-300, 25)
    axs[1].set_title("(b) Retorno da política gulosa no teste")
    axs[1].set_xlabel("Episódio de treino"); axs[1].set_ylabel("Retorno por episódio")
    fig.suptitle("Taxi: média ± desvio padrão em 10 sementes")
    fig.tight_layout(); fig.savefig(OUT + "taxi_fig1_curvas.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ BLACKJACK
def blackjack():
    tr = pd.read_csv(OUT + "blackjack_treino.csv")
    te = pd.read_csv(OUT + "blackjack_teste.csv")
    fin = pd.read_csv(OUT + "blackjack_avaliacao_final.csv")
    base = pd.read_csv(OUT + "blackjack_linhas_de_base.csv")
    print("== Linhas de base ==\n", base.round(4).to_string(index=False))
    print("\n== Avaliação final (100.000 mãos, política gulosa), 10 sementes ==")
    for a in NAMES:
        d = fin[fin.algoritmo == a]
        print(NAMES[a], "retorno médio", ms(d.retorno_final, 4), "| taxa de vitória", ms(d.taxa_vitoria_final, 4))
    x = fin[fin.algoritmo == "qlearning"].retorno_final
    y = fin[fin.algoritmo == "sarsa"].retorno_final
    print("Mann-Whitney retorno final:", mw(x, y))
    print("\n== Treino: recompensa média por mão nos últimos 10 blocos (90.000–100.000) ==")
    for a in NAMES:
        d = tr[(tr.algoritmo == a) & (tr.episodio_fim_bloco > 90000)].groupby("semente").recompensa_media.mean()
        print(NAMES[a], ms(d, 4))
    print("\n== Teste nos checkpoints (2.000 mãos cada): último checkpoint ==")
    last = te.episodio.max()
    for a in NAMES:
        d = te[(te.algoritmo == a) & (te.episodio == last)]
        print(NAMES[a], ms(d.retorno_teste, 4))

    # concordância entre políticas (soma 12-21, dealer 1-10, ás utilizável 0/1)
    Qs = np.load("/home/claude/blackjack_q_tables.npz")

    def pol(Q):
        out = {}
        for s in range(12, 22):
            for d in range(1, 11):
                for ace in (0, 1):
                    q = Q[(s * 11 + d) * 2 + ace]
                    out[(s, d, ace)] = None if q[0] == q[1] else int(q[1] > q[0])
        return out

    agree = []
    for seed in range(10):
        pq, ps = pol(Qs[f"qlearning_{seed}"]), pol(Qs[f"sarsa_{seed}"])
        ks = [k for k in pq if pq[k] is not None and ps[k] is not None]
        agree.append(np.mean([pq[k] == ps[k] for k in ks]))
    print("\nConcordância Q-learning × SARSA (mesma semente, estados 12–21 × 1–10 × ás):", ms(agree, 3))
    selfagree = {}
    for a in NAMES:
        ps_ = [pol(Qs[f"{a}_{s}"]) for s in range(10)]
        v = []
        for i in range(10):
            for j in range(i + 1, 10):
                ks = [k for k in ps_[i] if ps_[i][k] is not None and ps_[j][k] is not None]
                v.append(np.mean([ps_[i][k] == ps_[j][k] for k in ks]))
        selfagree[a] = np.mean(v)
    print("Concordância média entre sementes do mesmo algoritmo:", {NAMES[k]: round(v, 3) for k, v in selfagree.items()})

    # figura 1: curvas
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for a in NAMES:
        piv = tr[tr.algoritmo == a].pivot(index="episodio_fim_bloco", columns="semente", values="recompensa_media")
        m, s = piv.mean(axis=1), piv.std(axis=1, ddof=1)
        axs[0].plot(m.index, m.values, color=COL[a], lw=2, label=NAMES[a])
        axs[0].fill_between(m.index, m - s, m + s, color=COL[a], alpha=0.2)
    axs[0].set_title("(a) Recompensa média por mão durante o treino\n(blocos de 1.000 mãos)")
    axs[0].set_xlabel("Episódio de treino"); axs[0].set_ylabel("Recompensa média por mão"); axs[0].legend(loc="lower right")
    band(axs[1], te.rename(columns={}), "retorno_teste")
    r17 = base.loc[base.politica.str.startswith("pedir"), "retorno_medio"].iloc[0]
    axs[1].axhline(r17, color="gray", ls="--", lw=1.2)
    axs[1].text(te.episodio.max() * 0.98, r17 - 0.012, f"regra 'parar com 17+': {r17:.3f}", ha="right", color="gray", fontsize=10)
    axs[1].set_title("(b) Retorno da política gulosa no teste\n(2.000 mãos por ponto)")
    axs[1].set_xlabel("Episódio de treino"); axs[1].set_ylabel("Retorno médio por mão")
    fig.suptitle("Blackjack: média ± desvio padrão em 10 sementes")
    fig.tight_layout(); fig.savefig(OUT + "blackjack_fig1_curvas.png", dpi=200); plt.close(fig)

    # figura 2: política aprendida (semente 0)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d9d9d9", "#1f77b4", "#e8710a"])  # sem dado, parar, pedir
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, a in enumerate(NAMES):
        Q = Qs[f"{a}_0"]
        for j, ace in enumerate((0, 1)):
            grid = np.zeros((10, 10))
            for ri, s in enumerate(range(21, 11, -1)):
                for d in range(1, 11):
                    q = Q[(s * 11 + d) * 2 + ace]
                    grid[ri, d - 1] = 0 if q[0] == q[1] else (2 if q[1] > q[0] else 1)
            axs[i, j].imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")
            axs[i, j].set_xticks(range(10)); axs[i, j].set_xticklabels(["A"] + [str(k) for k in range(2, 11)])
            axs[i, j].set_yticks(range(10)); axs[i, j].set_yticklabels(range(21, 11, -1))
            axs[i, j].set_title(f"{NAMES[a]}, {'com' if ace else 'sem'} ás utilizável")
            axs[i, j].set_xlabel("Carta visível do dealer"); axs[i, j].set_ylabel("Soma do jogador")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#1f77b4", label="parar"), Patch(color="#e8710a", label="pedir carta"), Patch(color="#d9d9d9", label="sem dado")],
               loc="lower center", ncol=3)
    fig.suptitle("Política aprendida (semente 0)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(OUT + "blackjack_fig2_politica.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    {"cliff": cliff, "taxi": taxi, "blackjack": blackjack}[sys.argv[1]]()
