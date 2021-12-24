import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
lr_mu        = 0.0005
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005 # for target network soft update

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

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        return q
      
def train(mu, mu_target, q1, q2, q_target1, q_target2, memory, q1_optimizer, q2_optimizer, mu_optimizer, i):
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    
    target_a = mu_target(s_prime) + torch.randn(1) * 0.1    # smooth target action
    next_q1 = q_target1(s_prime, target_a)
    next_q2 = q_target2(s_prime, target_a)
    target = r + gamma * torch.min(next_q1, next_q2)        # double q learning
    
    q1_loss = F.smooth_l1_loss(q1(s,a), target.detach())
    q1_optimizer.zero_grad()
    q1_loss.backward()
    q1_optimizer.step()

    q2_loss = F.smooth_l1_loss(q2(s,a), target.detach())
    q2_optimizer.zero_grad()
    q2_loss.backward()
    q2_optimizer.step()
    
    if i % 2 == 0:  # delayed update
        mu_loss = -q1(s,mu(s)).mean() # That's all for the policy loss.
        mu_optimizer.zero_grad()
        mu_loss.backward()
        mu_optimizer.step()
    
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)
    
def main():
    env = gym.make('Pendulum-v0')
    memory = ReplayBuffer()

    q1, q2 = QNet(), QNet()
    q_target1, q_target2 = QNet(), QNet()
    q_target1.load_state_dict(q1.state_dict())
    q_target2.load_state_dict(q2.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q1_optimizer  = optim.Adam(q1.parameters(), lr=lr_q)
    q2_optimizer  = optim.Adam(q2.parameters(), lr=lr_q)

    for n_epi in range(10000):
        s = env.reset()
        
        for t in range(300): # maximum length of episode is 200 for Pendulum-v0
            a = mu(torch.from_numpy(s).float()) 
            a = torch.randn(1) * 0.5 + a
            a = a.item()
            s_prime, r, done, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime

            if done:
                break              
                
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q1, q2, q_target1, q_target2, memory, 
                      q1_optimizer, q2_optimizer, mu_optimizer, i)
                if i % 2 == 0:
                    soft_update(mu, mu_target)
                    soft_update(q1,  q_target1)
                    soft_update(q2,  q_target2)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()

# with smooth target action, learning is slower

# of episode :20, avg score : -1480.4
# of episode :40, avg score : -1640.8
# of episode :60, avg score : -1559.1
# of episode :80, avg score : -1525.0
# of episode :100, avg score : -1521.3
# of episode :120, avg score : -1557.8
# of episode :140, avg score : -1544.7
# of episode :160, avg score : -1477.9
# of episode :180, avg score : -1404.7
# of episode :200, avg score : -1521.7
# of episode :220, avg score : -1509.1
# of episode :240, avg score : -1302.5
# of episode :260, avg score : -1412.9
# of episode :280, avg score : -1461.5
# of episode :300, avg score : -1493.8
# of episode :320, avg score : -1511.9
# of episode :340, avg score : -1373.1
# of episode :360, avg score : -1264.6
# of episode :380, avg score : -1303.6
# of episode :400, avg score : -1174.7
# of episode :420, avg score : -1415.5
# of episode :440, avg score : -1262.9
# of episode :460, avg score : -1348.2
# of episode :480, avg score : -1184.8
# of episode :500, avg score : -1161.9
# of episode :520, avg score : -997.6
# of episode :540, avg score : -1397.3
# of episode :560, avg score : -1462.2
# of episode :580, avg score : -1333.7
# of episode :600, avg score : -1434.0
# of episode :620, avg score : -954.3
# of episode :640, avg score : -1058.0
# of episode :660, avg score : -1059.8
# of episode :680, avg score : -853.8
# of episode :700, avg score : -675.7
# of episode :720, avg score : -678.3
# of episode :740, avg score : -800.7
# of episode :760, avg score : -305.3
# of episode :780, avg score : -182.7
# of episode :800, avg score : -277.2
# of episode :820, avg score : -216.7
# of episode :840, avg score : -163.2
# of episode :860, avg score : -153.2
# of episode :880, avg score : -193.2