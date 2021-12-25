import gym
import random
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

alpha        = 0.2
LOG_SIG_MAX  = 2
LOG_SIG_MIN  = -20
epsilon      = 1e-6
batch_size   = 32
buffer_limit = 50000
tau          = 0.005

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.mean = nn.Linear(128, act_dim)
        self.log_std = nn.Linear(128, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()

        normal = Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)  # Pendulum scale
        action = y * 2
        log_prob = normal.log_prob(x) - torch.log(2 * (1 - y.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        mean = torch.tanh(mean) * 2

        return mean, action, log_prob

class Value(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Value, self).__init__()
        self.fc_s = nn.Linear(obs_dim, 64)
        self.fc_a = nn.Linear(act_dim,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32,1)

    def forward(self, obs, act):
        h1 = F.relu(self.fc_s(obs))
        h2 = F.relu(self.fc_a(act))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

def train(mu, mu_target, q1, q2, q_target1, q_target2, memory, q1_optimizer, q2_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    
    with torch.no_grad():
        _, target_a, log_prob = mu_target(s_prime)
        next_q1 = q_target1(s_prime, target_a)
        next_q2 = q_target2(s_prime, target_a)
        target = r + torch.min(next_q1, next_q2) - alpha * log_prob       # double q learning

    q1_loss = F.smooth_l1_loss(q1(s,a), target)
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()

    q2_loss = F.smooth_l1_loss(q2(s,a), target)
    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_optimizer.step()
    
    _, action, log_prob = mu(s)
    q1_value = q1(s, action)
    q2_value = q2(s, action)
    mu_loss = - (torch.min(q1_value, q2_value) - alpha * log_prob).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def evaluate(mu):
    env = gym.make('Pendulum-v0')
    s = env.reset()
    done = False
    score = 0.0

    while not done:
        a, _, _ = mu(torch.from_numpy(s).float())
        a = a.item()
        s_prime, r, done, info = env.step([a])
        score += r
        s = s_prime
    print("eval, reward is", score)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
    env = gym.make('Pendulum-v0')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    memory = ReplayBuffer()

    q1, q2 = Value(obs_dim, act_dim), Value(obs_dim, act_dim)
    q_target1, q_target2 = Value(obs_dim, act_dim), Value(obs_dim, act_dim)
    q_target1.load_state_dict(q1.state_dict())
    q_target2.load_state_dict(q2.state_dict())
    mu = Actor(obs_dim, act_dim)
    mu_target = Actor(obs_dim, act_dim)
    mu_target.load_state_dict(mu.state_dict())

    mu_optimizer = optim.Adam(mu.parameters(), lr=0.0005)
    q1_optimizer = optim.Adam(q1.parameters(), lr=0.001)
    q2_optimizer = optim.Adam(q2.parameters(), lr=0.001)

    score = 0.0
    print_interval = 20

    for n_epi in range(3001):
        s = env.reset()
        
        for t in range(300): # maximum length of episode is 200 for Pendulum-v0
            _, a, _ = mu(torch.from_numpy(s).float())
            a = a.item()
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score += r
            s = s_prime

            if done:
                break   

        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q1, q2, q_target1, q_target2, memory, q1_optimizer, q2_optimizer, mu_optimizer)
                soft_update(q1,  q_target1)
                soft_update(q2,  q_target2)
                soft_update(mu,  mu_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval), end=', ')
            evaluate(mu)
            score = 0.0

    env.close()         


if __name__ == '__main__':
    main()

# slow like shit