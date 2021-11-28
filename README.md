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
To make this robot work we built on top of existing work.
- We got the encoder precision and the idea to use a direct drive motor from the [Quanser Qube design](https://quanserinc.box.com/shared/static/5wnibclu7rp6xihm7mbxqxincu6dogur.pdf).
- We re-used a bits from [Quanser's code](https://git.ias.informatik.tu-darmstadt.de/quanser/clients/-/tree/master/quanser_robots/qube). Notably: 
  * their VelocityFilter class to compute the angular speeds
  * their GentlyTerminating wrapper to send a zero command to the robot at the end of each episode
  * their rotary inverted pendulum simulation
- The arm assembly is inspired by this [YouTube video](https://www.youtube.com/watch?v=xowrt6ShdCw) by Mack Tang.
- The visualization we use for the simulation is copy-pasted from https://github.com/angelolovatto/gym-cartpole-swingup
- We use the [StableBaselines3](https://github.com/DLR-RM/stable-baselines3) library to train the robot. 
- We implemented tricks from [Antonin Raffin's talk at RLVS 2021](https://www.youtube.com/watch?v=Ikngt0_DXJg).
  * HistoryWrapper and continuity cost
  * gSDE
- We use [code from Federico Bolanos](https://github.com/fbolanos/LS7366R/blob/master/LS7366R.py) to read the encoder counters.

