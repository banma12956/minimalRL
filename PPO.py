import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

eps_clip = 0.1


class Model(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = self.fc2(x)
        max_x, _ = x.max(dim=-1, keepdim=True)
        x = F.softmax(x-max_x, dim=0)       # TODO: without max_x, it does not work at all!!!
        
        return x

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = self.fc2(x)

        return x


def update_agent(actions, states, probs, advantages, model):
    policy_optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(3):
        probs_new = model(states)
        probs_a = torch.gather(probs_new, dim=1, index=actions)
        ratio = torch.exp(torch.log(probs_a) - torch.log(probs))

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages
        loss = - torch.min(surr1, surr2)
        loss.mean().backward()
        policy_optimizer.step()
        policy_optimizer.zero_grad()

def main():
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = Model(obs_dim, act_dim)
    critic = Critic(obs_dim)
    value_loss_f = nn.MSELoss()

    value_optimizer = optim.Adam(critic.parameters(), lr=0.0005)

    for i in range(1001):    # run 1000 episodes
        obs = env.reset()
        reward_to_go = np.array([])
        state_value = []
        actions = []
        states = []
        probs = []
        done = 0
        while not done:
            obs = torch.from_numpy(obs).float()
            states.append(obs)
            prob = model(obs)
            s_value = critic(obs)
            m = Categorical(prob)
            act = m.sample()
            obs, reward, done, info = env.step(act.item())

            reward_to_go = np.append(reward_to_go, [0.0])
            reward_to_go += reward
            state_value.append(s_value)
            actions.append(act)
            probs.append(prob[act].detach())
        
        reward_to_go = torch.from_numpy(reward_to_go).double().unsqueeze(1)
        state_value = torch.stack(state_value).double()
        actions = torch.as_tensor(actions).unsqueeze(1)
        states = torch.stack(states, dim=0).float()
        probs = torch.stack(probs).unsqueeze(1)

        value_loss = value_loss_f(state_value, reward_to_go)
        value_loss.backward()

        value_optimizer.step()
        value_optimizer.zero_grad()

        advantages = (reward_to_go - state_value)
        update_agent(actions.detach(), states, probs.detach(), advantages.detach(), model)
        
        if i % 100 == 0:
            print("episode", i, "step count", len(reward_to_go), "reward", reward_to_go[0].item())

if __name__ == '__main__':
    main()

# episode 0 step count 20 reward 20.0
# episode 100 step count 183 reward 183.0
# episode 200 step count 325 reward 325.0
# episode 300 step count 133 reward 133.0
# episode 400 step count 500 reward 500.0
# episode 500 step count 500 reward 500.0