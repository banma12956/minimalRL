import gym
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

gamma        = 1

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.a = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        logits = self.a(x)
        action_prob = F.softmax(logits, dim=-1)

        return action_prob

class Value(nn.Module):
    def __init__(self, obs_dim):
        super(Value, self).__init__()
        self.fc_s = nn.Linear(obs_dim, 256)
        self.fc_v = nn.Linear(256, 1)

    def forward(self, obs):
        h1 = F.relu(self.fc_s(obs))
        v = F.relu(self.fc_v(h1))
        return v

def evaluate(actor):
    env = gym.make('CartPole-v1')
    s = env.reset()
    done = False
    score = 0.0

    while not done:
        logits = actor(torch.from_numpy(s).float())
        a = torch.argmax(logits)
        s_prime, r, done, info = env.step(a.item())
        score += r
        s = s_prime
    print("eval, reward is", score)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def main():
    env = gym.make('CartPole-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    critic = Value(obs_dim)
    actor = Actor(obs_dim, act_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=0.0005)
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    score = 0.0
    print_interval = 100
    step = 0

    for epoch in range(3001):
        s = env.reset()
        s = torch.from_numpy(s).float()
        done = False 

        while not done:
            logits = actor(s)
            cat = Categorical(logits)
            a = cat.sample()
            s_prime, r, done, _ = env.step(a.item())
            step += 1
            s_prime = torch.from_numpy(s_prime).float()
            score += r

            """ Train """
            actor_optimizer.zero_grad()
            actor_loss = - cat.log_prob(a) * (r + gamma * critic(s_prime) - critic(s))
            actor_loss.backward()
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            pred_v = critic(s)
            target_v = r + gamma * critic(s_prime)
            loss_function = nn.MSELoss()
            critic_loss = loss_function(pred_v, target_v)
            critic_loss.backward()
            critic_optimizer.step()

            s = s_prime 

        if epoch%print_interval==0 and epoch!=0:
            print("# of episode :{}, avg score : {:.1f}".format(epoch, score/print_interval), end=', ')
            evaluate(actor)
            score = 0.0

    env.close()         


if __name__ == '__main__':
    main()

# in most cases, it does not work. lmao
# of episode :100, avg score : 38.4, eval, reward is 75.0
# of episode :200, avg score : 59.8, eval, reward is 107.0
# of episode :300, avg score : 157.3, eval, reward is 500.0
# of episode :400, avg score : 191.5, eval, reward is 500.0
# of episode :500, avg score : 197.0, eval, reward is 500.0