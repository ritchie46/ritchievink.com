---
author: Ritchie Vink
date: 2017-11-07T16:14:21+01:00
description: 
draft: true
keywords:
- Reinforcement learning
- Tensorflow
tags:
- Python
- Tensorflow
- Reinforcement learning
- FEM
title: Finding suboptimal structures using reinforcement learning
---

Reinforcement learning is a field of artificial intelligence that has made a lot of progress in solving arbitrarily games designed to be hard for humans. Many of the games that can now be solved by artificial intelligence are based on the foundation of being a Markov Decision Process (MDP). 
MDP's are a mathematical framework which state that in any state the environment is, an agent should be able to choose an optimal action that maximizes the probability of winning the game with the maximum reward. Described differently, when taking a photograph of the game at any timestep, all the information needed to make the best move would be available in that picure i.e. the history of the game is not important for the decision making. 

This made me wonder if I could regard the design of a small bridge as a Markov Decision Process and train a neural network to determine the optimal bridge form given the rules of the environment I provide. 

# Environment
First we need an environment with which the agent could interact. 

