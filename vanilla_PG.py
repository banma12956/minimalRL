import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.softmax(self.fc2(x), dim=0)
        
        return x

class Value(nn.Module):
    def __init__(self, obs_dim):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = self.fc2(x)

        return x

def main():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim, act_dim)
    value = Value(obs_dim)
    value_loss_f = nn.MSELoss()

    policy_optimizer = optim.Adam(model.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value.parameters(), lr=0.0005)

    for i in range(1000):    # run 1000 episodes
        obs = env.reset()
        reward_to_go = np.array([])
        log_a = []
        state_value = []
        done = 0
        while not done:
            obs = torch.from_numpy(obs).float()
            prob = model(obs)
            s_value = value(obs)
            m = Categorical(prob)
            act = m.sample()
            obs, reward, done, info = env.step(act.item())

            reward_to_go = np.append(reward_to_go, [0.0])
            reward_to_go += reward
            log_a.append(m.log_prob(act))
            state_value.append(s_value)

        for r, l, v in zip(reward_to_go, log_a, state_value):
            loss = - l * (r - v.item())
            loss.backward()
            value_loss = value_loss_f(v, torch.tensor([r]).float())
            value_loss.backward()

        policy_optimizer.step()
        policy_optimizer.zero_grad()
        value_optimizer.step()
        value_optimizer.zero_grad()

        if i % 100 == 0:
            print("episode", i, "step count", len(reward_to_go), "reward", reward_to_go[0])

if __name__ == '__main__':
    main()

# without value function:
# episode 0 step count 59 reward 59.0
# episode 100 step count 75 reward 75.0
# episode 200 step count 318 reward 318.0
# episode 300 step count 286 reward 286.0
# episode 400 step count 225 reward 225.0
# episode 500 step count 500 reward 500.0
# episode 600 step count 171 reward 171.0
# episode 700 step count 494 reward 494.0
# episode 800 step count 500 reward 500.0
# episode 900 step count 500 reward 500.0
# episode 1000 step count 500 reward 500.0
# episode 1100 step count 500 reward 500.0
# episode 1200 step count 500 reward 500.0

# with value function: (quicker!)
# episode 0 step count 31 reward 31.0
# episode 100 step count 93 reward 93.0
# episode 200 step count 272 reward 272.0
# episode 300 step count 429 reward 429.0
# episode 400 step count 500 reward 500.0
# episode 500 step count 343 reward 343.0
# episode 600 step count 332 reward 332.0
# episode 700 step count 500 reward 500.0
# episode 800 step count 500 reward 500.0
# episode 900 step count 500 reward 500.0
