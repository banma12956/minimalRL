# TODO: fuck it doesn't work and I don't know why
import gym
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import normal

buffer_limit = 50000
batch_size = 32
tau = 0.005


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = torch.tanh(self.fc2(x)) * 2
        
        return x

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim+act_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class ReplayBuffer():
    def __init__(self):
        super(ReplayBuffer, self).__init__()
        self.buffer = collections.deque(maxlen=buffer_limit)

    def push(self, tran):
        self.buffer.append(tran)

    def sample(self):
        trans = random.sample(self.buffer, batch_size)
        states = torch.stack([ele[0] for ele in trans])
        next_states = torch.stack([ele[1] for ele in trans])
        actions = torch.stack([ele[2] for ele in trans])
        rewards = torch.stack([ele[3] for ele in trans]).unsqueeze(1)

        return states, next_states, actions, rewards
        

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def evaluate(actor):
    env = gym.make('Pendulum-v0')
    done = False
    episode_reward = 0.0
    obs = env.reset()
    while not done:
        obs = torch.from_numpy(obs).float()
        with torch.no_grad():
            act = actor(obs)
        obs, reward, done, _ = env.step(act)
        episode_reward += reward.item()
    
    print("evaluate, get reward", episode_reward)

def rollout(actor, buffer, ou_noise):
    env = gym.make('Pendulum-v0')
    obs = env.reset()
    done = False
    while not done:
        obs = torch.from_numpy(obs).float()
        with torch.no_grad():
            mu = actor(obs)
        # m = normal.Normal(mu, 0.1)
        # act = m.sample()
        act = mu.item() + ou_noise()[0]
        act = torch.tensor([act]).float()

        next_obs, reward, done, _ = env.step(act)

        buffer.push([obs, torch.from_numpy(next_obs).float(), act, reward/100.0])
        obs = next_obs

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def train():
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    buffer = ReplayBuffer()

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim, act_dim)
    actor_target = Actor(obs_dim, act_dim)
    critic_target = Critic(obs_dim, act_dim)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0005)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for i in range(1001):
        rollout(actor, buffer, ou_noise)

        for j in range(10):
            states, next_states, actions, rewards = buffer.sample()

            # update critic
            with torch.no_grad():
                q_target = rewards + critic_target(next_states, actor_target(next_states))
            critic_loss = F.smooth_l1_loss(critic(states, actions), q_target)
            critic_loss.backward()
            critic_optimizer.step()
            critic_optimizer.zero_grad()

            # update actor
            actor_loss = - critic(states, actor(states)).mean()
            actor_loss.backward()
            actor_optimizer.step()
            actor_optimizer.zero_grad()

            # update target networks
            soft_update(actor, actor_target)
            soft_update(critic, critic_target)
        
        if i % 20 == 0:
            print("epoch", i, end=', ')
            evaluate(actor)

if __name__ == '__main__':
    train()