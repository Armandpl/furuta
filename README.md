# Furuta Pendulum [![flake8 Lint](https://github.com/Armandpl/furuta/actions/workflows/lint.yml/badge.svg)](https://github.com/Armandpl/furuta/actions/workflows/lint.yml)

In this repository you will find everything you need to build [and train a rotary inverted pendulum](https://www.youtube.com/watch?v=9ZhlEquyQEI) (also known as a furuta pendulum).

![](./assets/pendulum.jpg)
## Table of Content
- Motivations and Goals
- 
## Motivations and Goals
- There are many (hundreds!) videos of rotary inverted pendulum on the internet but I couldn't find good documentation on how to build one and how to control one
- Balancing a pendulum is the RL "hello world"
- Seems like a good platform to run RL experiments  (offline RL? [transfer from sim to robot?](https://www.youtube.com/watch?v=aTDkYFZFWug)?)

## MLOps

## Credits
To build this robot we leveraged existing work.
- The we got the encoder precision and the idea to use a direct drive motor from the Quanser Qube design.
- We also re-used a good chunk of Quanser code. Notably: 
  * their VelocityFilter class to compute the angular speeds
  * their GentlyTerminating wrapper to send a zero command to the robot at the end of each episode
  * their simulation
- The arm assembly is inspired by this YouTube video.
- The visualization we use for the simulation is copy-pasted from https://github.com/angelolovatto/gym-cartpole-swingup
- We use the StableBaselines3 library to train the robot. 
- We implemented tricks from Antonin Raffin's talk at rlvs2021.
  * HistoryWrapper and continuity cost
  * gSDE

