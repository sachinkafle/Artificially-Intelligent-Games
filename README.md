# Artificially-Intelligent-Games
AI Games &amp; RL Algorithms Collection  A clean, well-documented reference repo showcasing classic game AIs and reinforcement learning techniques. It bundles five self-contained projects:  Tic-Tac-Toe (Minimax)  Q-Learning Game (tabular)  Monte Carlo Method (RL)  Blackjack AI via Monte Carlo Control  PacMan with Deep Q-Learning (DQN)  
DQN

A compact, learning-focused GitHub repo that bundles five classic game-AI projects and reinforcement learning techniques. Each project is self-contained, well-commented, and runnable from the command line—perfect for studying fundamentals or using as teaching demos.

What’s inside
1) Tic-Tac-Toe — Minimax (aka “minmax”)

An unbeatable agent for perfect-information play.

Core ideas: game tree search, terminal utilities, minimax backups, optional α-β pruning

Goodies: human-vs-AI CLI, move explanations, depth-limited heuristic mode

2) Q-Learning Game — Tabular Control

A small Gridworld-style environment learned from scratch.

Core ideas: ε-greedy exploration, learning-rate scheduling, reward shaping

Goodies: policy/value heatmaps, convergence curves, checkpointed Q-tables

3) Monte Carlo Method — Prediction & Control

Minimal implementations to demystify episodic Monte Carlo RL.

Core ideas: first-visit prediction, return sampling, exploring starts for control

Goodies: reusable MC utilities, side-by-side comparisons with tabular TD

4) Blackjack AI — Monte Carlo Control

Near-optimal Blackjack play learned from sampled returns.

Core ideas: on-policy improvement, state aggregation (player sum / dealer upcard / usable ace)

Goodies: strategy charts, win-rate tracking, evaluation vs baseline dealers

5) PacMan — Deep Q-Learning (DQN)

A PacMan-like grid environment with a PyTorch DQN agent.

Core ideas: replay buffer, target network, ε-decay, Huber loss, gradient clipping

Goodies: training curves (reward/loss), episodic GIFs, model checkpoints

Tech stack

Python 3.10+

NumPy, matplotlib

Gym/Gymnasium (for Blackjack wrapper)

PyTorch (for DQN)
