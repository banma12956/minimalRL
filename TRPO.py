import gymnasium as gym
import torch
import numpy as np
import scipy.optimize

from memory import Memory
from model import Policy, Value
from utils import normalization

wandb_record = True


gamma = 0.995
tau = 0.97
delta = 0.03
l2_reg = 1e-3
batch_size = 1000
# lr = 0.001

if wandb_record:
    import wandb
    wandb.init(project="my_TRPO_hopper")
    wandb.run.name = "simple_trpo-delta_{}-batch_{}".format(delta, batch_size)
wand_step = 0

seed = 0
episode_num = 100
env = gym.make("Hopper-v4")
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

torch.manual_seed(seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

value_opt = torch.optim.Adam(value_net.parameters(), lr=1e-3)

memory = Memory()


def select_action(state, eval_action=False):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(state)

    if eval_action:
        action = action_mean
    else:
        action = torch.normal(action_mean, action_std)
    return action.detach().numpy()

def evaluation(moving_state):
    state, info = env.reset()
    
    reward_episode = 0
    step_num = 0
    while step_num < batch_size:
        state = moving_state(state, p=False)
        state = np.array(state).astype(np.float32)
        action = select_action(state, eval_action=True)
        state, reward, done, truncated, info = env.step(action[0])
        reward_episode += reward
        step_num += 1
        if done:
            break
    if wandb_record:
        wandb.log({"eval_reward": reward_episode/batch_size}, step=int(wand_step))

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

def compute_log_prob(mean, log_std, std, action):
    action_log_probs = -(action - mean).pow(2) / (2 * std.pow(2)) - 0.5 * np.log(2 * np.pi) - log_std
    return action_log_probs.sum(1, keepdim=True)

def trpo_step(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.from_numpy(np.array(batch.state))
    values = value_net(states)

    # GAE
    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = returns

    # update value network
    # for i in range(1000):
    #     values = value_net(states)
    #     value_loss = (values - targets).pow(2).mean()
    #     # print("value_loss: {}".format(value_loss))

    #     value_opt.zero_grad()
    #     value_loss.backward()
    #     value_opt.step()

    #     if wandb_record:
    #         wandb.log({"value_loss": value_loss})
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(states)

        value_loss = (values_ - targets).pow(2).mean()
        # print("value_loss", value_loss)
        # if wandb_record:
        #     wandb.log({"value_loss": value_loss})

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    # update policy network
    def get_kl():
        mean1, log_std1, std1 = policy_net(states)
        mean0 = mean1.detach()
        log_std0 = log_std1.detach()
        std0 = std1.detach()
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def fisher_vector_product(p):
        kl = get_kl().mean()
        grad = torch.autograd.grad(kl, policy_net.parameters(), create_graph=True)
        grad_vector = torch.cat([grad.view(-1) for grad in grad])
        grad_vector_product = torch.sum(grad_vector * p)
        grad_grad = torch.autograd.grad(grad_vector_product, policy_net.parameters())
        grad_grad_vector = torch.cat([grad.contiguous().view(-1) for grad in grad_grad]).data
        return grad_grad_vector + p * 0.1

    def conjugate_gradient(b):
        p = b.clone().detach()
        r = b.clone().detach()
        x = torch.zeros(b.size())
        rdotr = torch.dot(r, r)
        for i in range(10):
            z = fisher_vector_product(p)
            v = rdotr / torch.dot(p, z)
            x += v * p
            r -= v * z
            new_rdotr = torch.dot(r, r)
            mu = new_rdotr / rdotr
            p = r + mu * p
            rdotr = new_rdotr
            if rdotr < 1e-10:
                break
        print("cg iter: {}, rdotr: {}".format(i, rdotr))
        return x

    def line_search(step_dir):
        max_backtracks = 10
        for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
            scaled_dir = stepfrac * step_dir
            if scaled_dir @ (fisher_vector_product(scaled_dir)) < delta:
                print("return in line search, stepfrac", stepfrac)
                return scaled_dir
            else:
                print("not return", scaled_dir @ (fisher_vector_product(scaled_dir)))
        return torch.zeros(step_dir.size())

    mean, log_std, std = policy_net(states)
    log_prob = compute_log_prob(mean, log_std, std, actions)
    fix_log_prob = log_prob.detach()

    advantages = (advantages - advantages.mean()) / advantages.std()
    loss = - (advantages * torch.exp(log_prob - fix_log_prob)).mean()
    grad = torch.autograd.grad(loss, policy_net.parameters())
    flat_grad = torch.cat([grad.view(-1) for grad in grad]).data

    step_dir = conjugate_gradient(flat_grad)
    prev_model_flat = get_flat_params_from(policy_net)
    scaled_step = line_search(step_dir)
    set_flat_params_to(policy_net, prev_model_flat - scaled_step)
    # with torch.no_grad():
    #     log_prob = compute_log_prob(mean, log_std, std, actions)
    #     print("previous loss", loss)
    #     print("after update the trust region, loss", - (advantages * torch.exp(log_prob - fix_log_prob)).mean())

def main():
    moving_state = normalization((num_inputs))
    evaluation(moving_state)

    state, info = env.reset()
    state = moving_state(state)

    for episode in range(episode_num):
        step_num = 0
        reward_episode = 0
        while step_num < batch_size:
            state = np.array(state).astype(np.float32)
            action = select_action(state)
            next_state, reward, done, truncated, info = env.step(action[0])
            next_state = moving_state(next_state)
            memory.push(state, action, 1-done, next_state, reward)

            state = next_state
            reward_episode += reward
            step_num += 1
            # print("step_num: {}".format(step_num))

            if done:
                state, info = env.reset()
                state = moving_state(state)

        print("Episode: {}, reward: {}, step_num: {}".format(episode, reward_episode, step_num))
        if wandb_record:
            global wand_step
            wand_step += 1
            wandb.log({"reward": reward_episode/batch_size}, step=int(wand_step))

        batch = memory.sample()
        trpo_step(batch)

        if episode % 3 == 0:
            evaluation(moving_state)

if __name__ == "__main__":
    main()
