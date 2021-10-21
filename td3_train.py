# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pickle
from time import sleep

import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os

import furuta_gym
import gym_cartpole_swingup        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TD3 agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Furuta-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--episode-length', type=int, default=0,
                        help='the maximum length of each episode')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="furuta",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=int(1e6),
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--tau', type=float, default=0.005,
                        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=256,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--policy-noise', type=float, default=0.2,
                        help='the scale of policy noise')
    parser.add_argument('--exploration-noise', type=float, default=0.1,
                        help='the scale of exploration noise')
    parser.add_argument('--learning-starts', type=int, default=25e3,
                        help="timestep to start learning")
    parser.add_argument('--policy-frequency', type=int, default=2, # relative to training frequency
                        help="the frequency of training policy (delayed)")
    parser.add_argument('--noise-clip', type=float, default=0.5,
                         help='noise clip parameter of the Target Policy Smoothing Regularization')

    # params to accomodate embedded system
    parser.add_argument('--model-artifact', type=str, default=None,
                        help="the artifact version of the model to load")
    parser.add_argument('--neurons', type=int, default=256,
                         help='hidden layer size for the networks')
    parser.add_argument('--dt', type=float, default=0.01,
                         help='control frequency')
    parser.add_argument('--training-frequency', type=int, default=1500, # = 15 sec at 100hz
                        help="the frequency of training critics/q functions")
    parser.add_argument('--policy-starts', type=int, default=25e3,
                        help="When to start using the learned policy")

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.track:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")


# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
if args.gym_id == "Furuta-v0":
    env = gym.make(args.gym_id, dt=args.dt)
else:
    env = gym.make(args.gym_id)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
# respect the default timelimit
assert isinstance(env.action_space, Box), "only continuous action space is supported"
if args.capture_video:
    env = Monitor(env, f'videos/{experiment_name}')

# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()+np.prod(env.action_space.shape), args.neurons)
        self.fc2 = nn.Linear(args.neurons, args.neurons)
        self.fc3 = nn.Linear(args.neurons, 1)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), args.neurons)
        self.fc2 = nn.Linear(args.neurons, args.neurons)
        self.fc_mu = nn.Linear(args.neurons, np.prod(env.action_space.shape))

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc_mu(x))

def linear_schedule(start_sigma: float, end_sigma: float, duration: int, t: int):
    slope =  (end_sigma - start_sigma) / duration
    return max(slope * t + start_sigma, end_sigma)

max_action = float(env.action_space.high[0])
rb = ReplayBuffer(args.buffer_size)
actor = Actor(env).to(device)
qf1 = QNetwork(env).to(device)
qf2 = QNetwork(env).to(device)

# loading model
if args.model_artifact is not None:
    print(f"loading model from Artifacts, version {args.model_artifact}")
    artifact = wandb.use_artifact(f"td3_model:{args.model_artifact}")
    artifact_dir = artifact.download()
    actor.load_state_dict(torch.load(os.path.join(artifact_dir, "target_actor.pth")))
    qf1.load_state_dict(torch.load(os.path.join(artifact_dir, "qf1.pth")))
    qf2.load_state_dict(torch.load(os.path.join(artifact_dir, "qf2.pth")))

qf1_target = QNetwork(env).to(device)
qf2_target = QNetwork(env).to(device)
target_actor = Actor(env).to(device)
target_actor.load_state_dict(actor.state_dict())
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)
loss_fn = nn.MSELoss()

# TRY NOT TO MODIFY: start the game
print("Experiment starts")
obs = env.reset()
episode_reward = 0

first_action_done = False

try:
    for global_step in range(args.total_timesteps):
        start = time.time()
        # ALGO LOGIC: put action logic here
        if global_step < args.policy_starts:
            action = env.action_space.sample()
        else:
            # doing this bc it seems like the first time we run this it takes forever??
            if not first_action_done:
                print("first action hack")
                tmp_obs = env.reset()
                first_action_done = True
            action = actor.forward(obs.reshape((1,)+obs.shape), device)
            action = (
                action.tolist()[0]
                + np.random.normal(0, max_action * args.exploration_noise, size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action)

        # reset ASAP
        if done:
            tmp_obs = env.reset()

        episode_reward += reward

        # ALGO LOGIC: training.
        rb.put((obs, action, reward, next_obs, done))

        # we use tmp_obs = None to know if we should run env.reset or not
        # tmp_obs = None
        if global_step > args.learning_starts and global_step % args.training_frequency == 0:
            # terminate current episode 
            if not done:
                tmp_obs = env.reset()
                done = True

            # train for the past X timestep
            print("learning")
            for i in range(args.training_frequency):
                s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
                with torch.no_grad():
                    clipped_noise = (
                        torch.randn_like(torch.Tensor(action)) * args.policy_noise
                    ).clamp(-args.noise_clip, args.noise_clip)

                    next_state_actions = (
                        target_actor.forward(s_next_obses, device) + clipped_noise.to(device)
                    ).clamp(env.action_space.low[0], env.action_space.high[0])
                    qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
                    qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions, device)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
                qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
                qf1_loss = loss_fn(qf1_a_values, next_q_value)
                qf2_loss = loss_fn(qf2_a_values, next_q_value)

                # optimize the model
                q_optimizer.zero_grad()
                qf1_loss.backward()
                qf2_loss.backward()
                nn.utils.clip_grad_norm_(list(qf1.parameters())+list(qf2.parameters()), args.max_grad_norm)
                q_optimizer.step()

                if i % args.policy_frequency == 0:
                    actor_loss = -qf1.forward(s_obs, actor.forward(s_obs, device), device).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(list(actor.parameters()), args.max_grad_norm)
                    actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        end = time.time()
        real_dt = end-start

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook 
        obs = next_obs

        # make sure loop time/dt is constant (speed calculation are based on fixed dt)
        dt_diff = args.dt - real_dt
        if dt_diff < 0 or args.gym_id != "Furuta-v0":
            pass # bad news the loop is too slow (or we're in sim). todo throw error or warning
        else:
            sleep(dt_diff)

        wandb.log({"loop_time": real_dt,
                   "dt_diff": dt_diff})


        wandb.log({
            "env/motor_angle": env.state[0],
            "env/pendulum_angle": env.state[1],
            "env/motor_angle_velocity": env.state[0],
            "env/pendulum_angle_velocity": env.state[1],
            "env/action": action
        })

        if done:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            print(f"global_step={global_step}, episode_reward={episode_reward}")
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            if tmp_obs is None:
                tmp_obs = env.reset()
            obs, episode_reward = tmp_obs, 0
except KeyboardInterrupt:
    print("Interrupting training")
    env.close()

# make sure we saved the current global_step to be able to resume training
wandb.run.summary["global_step"]

# save model as artifact
print("Saving model")
model_dir = f"models/{experiment_name}"
os.makedirs(model_dir)
torch.save(qf1_target.state_dict(), os.path.join(model_dir, "qf1.pth"))
torch.save(qf2_target.state_dict(), os.path.join(model_dir, "qf2.pth"))
torch.save(target_actor.state_dict(), os.path.join(model_dir, "target_actor.pth"))
artifact = wandb.Artifact("td3_model", type="model")
artifact.add_dir(model_dir)
wandb.log_artifact(artifact)

print("Saving replay buffer")
rb_dir = f"replay_buffers/{experiment_name}"
os.makedirs(rb_dir)
fpath = os.path.join(rb_dir, "rb.p")
pickle.dump(rb.buffer, open(fpath, 'wb'))
buffer_artifact = wandb.Artifact("td3_replay_buffer", type="replay buffer")
buffer_artifact.add_file(fpath)
wandb.log_artifact(buffer_artifact)

env.close()
writer.close()
