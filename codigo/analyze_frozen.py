import numpy as np, pandas as pd
tr = pd.read_csv("/mnt/user-data/outputs/frozenlake_treino.csv")
te = pd.read_csv("/mnt/user-data/outputs/frozenlake_teste.csv")
names = {"qlearning": "Q-learning", "sarsa": "SARSA"}

def ms(x): return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"

print("== Sucesso final no teste (ep. 5000), 50 episódios gulosos, por semente ==")
for a in names:
    d = te[(te.algoritmo==a)&(te.episodio==5000)]
    print(names[a], ms(d.taxa_sucesso_teste), "| passos médios dos sucessos:", ms(d.passos_medios_sucesso.dropna()))

print("\n== Recompensa média de treino nos últimos 100 episódios (4901-5000) ==")
for a in names:
    d = tr[(tr.algoritmo==a)&(tr.episodio>4900)].groupby("semente").recompensa.mean()
    print(names[a], ms(d))

print("\n== Recompensa média de treino em todo o treino (por semente) ==")
for a in names:
    d = tr[tr.algoritmo==a].groupby("semente").recompensa.mean()
    print(names[a], ms(d))

print("\n== Primeiro checkpoint com sucesso de teste >= 0.9 / == 1.0 ==")
for thr in (0.9, 1.0):
    for a in names:
        firsts=[]
        for s,g in te[te.algoritmo==a].groupby("semente"):
            ok=g[g.taxa_sucesso_teste>=thr].episodio
            firsts.append(ok.min() if len(ok) else np.nan)
        firsts=np.array(firsts,float)
        print(f"thr {thr} {names[a]}: {np.nanmean(firsts):.0f} ± {np.nanstd(firsts,ddof=1):.0f}  (alcançou em {np.sum(~np.isnan(firsts))}/10 sementes)  valores={firsts.astype(int).tolist()}")

print("\n== Sucesso de teste médio por checkpoint (primeiros 15) ==")
p = te.groupby(["algoritmo","episodio"]).taxa_sucesso_teste.mean().unstack(0)
print(p.head(15).round(3))

print("\n== Fração de episódios de treino com sucesso em blocos ==")
tr["bloco"]=((tr.episodio-1)//500)*500+500
print(tr.groupby(["algoritmo","bloco"]).recompensa.mean().unstack(0).round(3))

print("\n== Comprimento médio dos episódios de treino (últimos 100) ==")
for a in names:
    d = tr[(tr.algoritmo==a)&(tr.episodio>4900)].groupby("semente").passos.mean()
    print(names[a], ms(d))
try:
    from scipy.stats import mannwhitneyu
    for label,col in (("recomp. treino últimos 100", None),):
        pass
    x = tr[(tr.algoritmo=="qlearning")&(tr.episodio>4900)].groupby("semente").recompensa.mean()
    y = tr[(tr.algoritmo=="sarsa")&(tr.episodio>4900)].groupby("semente").recompensa.mean()
    print("\nMann-Whitney recompensa treino últimos 100:", mannwhitneyu(x,y))
except Exception as e:
    print("scipy indisponível:", e)
