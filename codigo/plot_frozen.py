import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from frozen_lake_experiment import DESC, MAP_SIZE

OUT = "/mnt/user-data/outputs/"
tr = pd.read_csv(OUT + "frozenlake_treino.csv")
te = pd.read_csv(OUT + "frozenlake_teste.csv")
names = {"qlearning": "Q-learning", "sarsa": "SARSA"}
colors = {"qlearning": "#1f77b4", "sarsa": "#e8710a"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# ---- Fig 1: recompensa média por episódio (média móvel de 100), média ± desvio entre sementes
W = 100
fig, ax = plt.subplots(figsize=(8, 4.5))
for a in names:
    piv = tr[tr.algoritmo == a].pivot(index="episodio", columns="semente", values="recompensa")
    ma = piv.rolling(W).mean().dropna()
    m, s = ma.mean(axis=1), ma.std(axis=1, ddof=1)
    ax.plot(m.index, m.values, color=colors[a], label=names[a], lw=2)
    ax.fill_between(m.index, m - s, m + s, color=colors[a], alpha=0.2)
ax.set_xlabel("Episódio de treino")
ax.set_ylabel("Recompensa média por episódio\n(média móvel de 100)")
ax.set_ylim(0, 1.05)
ax.set_title("Frozen Lake 8×8: recompensa durante o treino\n(média ± desvio padrão, 10 sementes)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "frozenlake_fig1_recompensa_treino.png", dpi=200); plt.close(fig)

# ---- Fig 2: taxa de sucesso no teste (política gulosa)
fig, ax = plt.subplots(figsize=(8, 4.5))
for a in names:
    piv = te[te.algoritmo == a].pivot(index="episodio", columns="semente", values="taxa_sucesso_teste")
    m, s = piv.mean(axis=1), piv.std(axis=1, ddof=1)
    ax.plot(m.index, m.values, color=colors[a], label=names[a], lw=2)
    ax.fill_between(m.index, (m - s).clip(0, 1), (m + s).clip(0, 1), color=colors[a], alpha=0.2)
ax.set_xlabel("Episódio de treino")
ax.set_ylabel("Taxa de sucesso no teste\n(política gulosa, 50 episódios)")
ax.set_ylim(0, 1.05)
ax.set_title("Frozen Lake 8×8: desempenho da política aprendida\n(média ± desvio padrão, 10 sementes)")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT + "frozenlake_fig2_sucesso_teste.png", dpi=200); plt.close(fig)

# ---- Fig 3: política aprendida (semente 0) sobre o mapa
Qs = np.load("/home/claude/q_tables.npz")
arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
fig, axs = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, a in zip(axs, names):
    Q = Qs[f"{a}_0"]
    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            ch = DESC[r][c]
            face = {"H": "#2b2b2b", "G": "#2ca02c", "S": "#aec7e8"}.get(ch, "#f2f6fa")
            ax.add_patch(Rectangle((c, MAP_SIZE - 1 - r), 1, 1, facecolor=face, edgecolor="#888", lw=0.8))
            s = r * MAP_SIZE + c
            if ch in "SF" and Q[s].max() > 1e-9:
                ax.text(c + .5, MAP_SIZE - 1 - r + .5, arrows[int(np.argmax(Q[s]))], ha="center", va="center", fontsize=18, color="#14213d")
            elif ch == "H":
                ax.text(c + .5, MAP_SIZE - 1 - r + .5, "H", ha="center", va="center", color="white", fontsize=12)
            elif ch == "G":
                ax.text(c + .5, MAP_SIZE - 1 - r + .5, "G", ha="center", va="center", color="white", fontsize=14, fontweight="bold")
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{names[a]} (semente 0)")
fig.suptitle("Melhor ação aprendida em cada célula (H = buraco, G = objetivo, azul = início)")
fig.tight_layout(); fig.savefig(OUT + "frozenlake_fig3_politica.png", dpi=200); plt.close(fig)
print("figuras ok")
