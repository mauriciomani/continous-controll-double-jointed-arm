# Project 2: Continuous Control

This repository includes all the needed information to train a double-jointed arm agent to move to target location, similar to UNITY Machine Learning Agents. The latter, works as enviroments to train intelligent agents. This case a simulation of robotic arm following a target ball.
In this notebook, you will learn how to use the Unity ML-Agents environment for the first project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

### What is this about?
We are going to use **Actor Critic** method to solve this enviroment. Specifically **Deep Deterministic Gradient Policy**. We are using this algorithm since we have a continous action space (that we will go further on that below). Our goal is to find an optimal policy, through selecting desired actions (in a continous space) and use those actions to produce Q-values for more information, visit report.pdf.

### More abour this enviroment

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment from Unity.

This environment consist of a double-jointed arm moving to target locations, for example, following a ball. A **reward of +0.1** is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each **action** is a **vector with four numbers**, corresponding to torque applicable to two joints. Every **entry** in the action vector should be a **number between -1 and 1**. We will only train one agent and we would say it has been done when we get an average score of more than 30 over 100 consectuive episodes.
