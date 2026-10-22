# Experimentos de Aprendizado por Reforço (PPgTI/UFRN)

Código, dados e figuras do relatório da disciplina **Aprendizado por Reforço** (Prof. Charles Madeira).
Autor: Fabricio Guimarães Alcântara da Silva.

Algoritmos: Q-learning e SARSA (tabulares); DQN e DDQN (redes neurais).
Ambientes (Gymnasium): Frozen Lake 8×8, Cliff Walking, Taxi-v4, Blackjack-v1, CartPole-v1, MountainCar-v0, ALE/Pong-v5 e ALE/SpaceInvaders-v5.

## Estrutura
- `codigo/`: scripts Python.
  - Tabulares: `frozen_lake_experiment.py`, `toytext_experiments.py` (Taxi, Cliff Walking, Blackjack), `blackjack_baseline.py`, `diag_sarsa.py`.
  - Profundos: `dqn_classic.py` (CartPole, PyTorch), `dqn_numpy.py` (Mountain Car, NumPy), `dqn_atari.py`, `atari_random.py`, `diag_dqn_q.py` (diagnóstico dos valores Q).
  - Base e análise: `cartpole_base.py`, `analyze_*.py`, `plot_frozen.py`.
- `resultados/dados/`: CSVs de treino e teste de cada experimento.
- `resultados/figuras/`: figuras usadas no relatório.

## Dependências
Python 3, gymnasium (com toy-text, classic-control e pygame), ale-py, numpy, pandas, matplotlib, seaborn, scipy e PyTorch (CartPole e Atari).

## Observações
- O Mountain Car usou uma implementação própria em NumPy (`dqn_numpy.py`), diferente do PyTorch dos demais.
- O Atari foi treinado por 100.000 passos e 1 semente (cerca de 4% do orçamento do artigo original).
- Alguns scripts usam caminhos de pastas do ambiente em que rodei; ajuste-os se necessário.
