"""Análise e gráficos: python analyze_dqn.py cartpole | mountaincar | atari  [sufixo_lr]"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

OUT = "/mnt/user-data/outputs/"
NAMES = {"dqn": "DQN", "ddqn": "DDQN"}
COL = {"dqn": "#1f77b4", "ddqn": "#e8710a"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def ms(x, nd=1):
    x = np.asarray(x, float)
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}" if len(x) > 1 else f"{x[0]:.{nd}f}"


def band(ax, df, xcol, ycol, ylim=None):
    for a in NAMES:
        piv = df[df.algoritmo == a].pivot(index=xcol, columns="semente", values=ycol)
        m, s = piv.mean(axis=1), piv.std(axis=1, ddof=1)
        ax.plot(m.index, m.values, color=COL[a], lw=2, label=NAMES[a])
        ax.fill_between(m.index, m - s, m + s, color=COL[a], alpha=0.2)
    if ylim:
        ax.set_ylim(*ylim)


def binned_train(tr, bin_size):
    tr = tr.copy()
    tr["bloco"] = ((tr.passo - 1) // bin_size + 1) * bin_size
    return tr.groupby(["algoritmo", "semente", "bloco"]).retorno.mean().reset_index().rename(columns={"bloco": "passo"})


def classic(name, suffix="", title="", thr=None, ylim_eval=None, bin_size=2500, tagfig=""):
    tr = pd.read_csv(f"{OUT}{name}_dqn_treino{suffix}.csv")
    te = pd.read_csv(f"{OUT}{name}_dqn_teste{suffix}.csv")
    last = te.passo.max()
    print(f"== {title}{suffix}: política gulosa no passo {last}, sementes por algoritmo:",
          {a: te[te.algoritmo == a].semente.nunique() for a in NAMES})
    for a in NAMES:
        d = te[(te.algoritmo == a) & (te.passo == last)]
        auc = te[te.algoritmo == a].groupby("semente").retorno_teste.mean()
        best = te[te.algoritmo == a].groupby("semente").retorno_teste.max()
        print(NAMES[a], "retorno final", ms(d.retorno_teste), "| média das avaliações (área sob a curva)", ms(auc),
              "| melhor avaliação", ms(best), "| finais por semente:", d.retorno_teste.round(1).tolist())
        if thr is not None:
            firsts = []
            for s, g in te[te.algoritmo == a].groupby("semente"):
                ok = g[g.retorno_teste >= thr].passo
                firsts.append(ok.min() if len(ok) else np.nan)
            f = np.array(firsts, float)
            print(f"   primeira avaliação com retorno >= {thr}: {np.sum(~np.isnan(f))}/{len(f)} sementes",
                  "| passos:", f.tolist())
    x = te[(te.algoritmo == "dqn") & (te.passo == last)].retorno_teste
    y = te[(te.algoritmo == "ddqn") & (te.passo == last)].retorno_teste
    if len(x) > 1 and len(y) > 1:
        r = mannwhitneyu(x, y)
        print(f"Mann-Whitney retorno final DQN vs DDQN: U = {r.statistic:.1f}, p = {r.pvalue:.4f}")
    for a in NAMES:
        t = tr[(tr.algoritmo == a)]
        n_eps = t.groupby("semente").size()
        k = t.sort_values(["semente", "passo"]).groupby("semente").tail(50).groupby("semente").retorno.mean()
        print(NAMES[a], "retorno de treino (últimos 50 episódios)", ms(k), "| episódios por semente", ms(n_eps, 0))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    band(axs[0], binned_train(tr, bin_size), "passo", "retorno", None)
    axs[0].set_title(f"(a) Retorno durante o treino\n(média por blocos de {bin_size:,} passos)".replace(",", "."))
    axs[0].set_xlabel("Passo de treino"); axs[0].set_ylabel("Retorno por episódio"); axs[0].legend(loc="best")
    band(axs[1], te, "passo", "retorno_teste", ylim_eval)
    axs[1].set_title("(b) Retorno da política gulosa no teste")
    axs[1].set_xlabel("Passo de treino"); axs[1].set_ylabel("Retorno por episódio")
    nseed = te.semente.nunique()
    fig.suptitle(f"{title}: média ± desvio padrão em {nseed} sementes")
    fig.tight_layout(); fig.savefig(f"{OUT}{name}_dqn_fig1_curvas{suffix}{tagfig}.png", dpi=200); plt.close(fig)


def atari():
    base = pd.read_csv(OUT + "atari_linha_de_base_aleatoria.csv").set_index("jogo")
    print("== Linha de base aleatória ==\n", base.round(2).to_string())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (jogo, titulo) in zip(axs, (("pong", "Pong"), ("spaceinvaders", "Space Invaders"))):
        try:
            tr = pd.read_csv(f"{OUT}{jogo}_dqn_treino.csv")
            te = pd.read_csv(f"{OUT}{jogo}_dqn_teste.csv")
        except FileNotFoundError:
            print(jogo, "sem resultados"); continue
        print(f"\n== {titulo}: sementes = {te.semente.nunique()} ==")
        for a in NAMES:
            d = te[(te.algoritmo == a)]
            t = tr[tr.algoritmo == a].sort_values("passo")
            print(NAMES[a], "avaliações (passo: placar):", {int(p): round(v, 1) for p, v in zip(d.passo, d.retorno_teste)},
                  "| treino, últimos 10 episódios:", round(t.retorno.tail(10).mean(), 1), "| episódios:", len(t))
            w = 10
            ma = t.retorno.rolling(w).mean()
            ax.plot(t.passo, ma, color=COL[a], lw=2, label=NAMES[a])
        ax.axhline(base.loc[jogo, "placar_medio_aleatoria"], color="gray", ls="--", lw=1.2, label="aleatória")
        ax.set_title(f"{titulo}"); ax.set_xlabel("Passo de treino"); ax.set_ylabel("Placar por partida (média móvel de 10)")
        ax.legend(loc="best")
    fig.suptitle("Atari (orçamento reduzido, 100.000 passos, 1 semente)")
    fig.tight_layout(); fig.savefig(OUT + "atari_dqn_fig1_curvas.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    what = sys.argv[1]
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    if what == "cartpole":
        classic("cartpole", suffix, "CartPole", thr=475, ylim_eval=(0, 520), tagfig="")
    elif what == "mountaincar":
        classic("mountaincar", suffix, "Mountain Car", thr=-199, ylim_eval=(-205, -80), bin_size=5000)
    else:
        atari()
