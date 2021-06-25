#!/usr/bin/env python
# Created at 2020/3/25
import torch
from torch.utils.tensorboard import SummaryWriter

import pickle

import numpy as np
import torch
import torch.optim as optim

from from_reference.Policy import Policy
from from_reference.QValue import QValue
from from_reference.Value import Value
from from_reference.Util import get_env_info, Memory, check_path, ZFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT = torch.FloatTensor

def get_flat_params(model: nn.Module):

    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):

    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])

def sac_step(policy_net, value_net, value_net_target, q_net_1, q_net_2, optimizer_policy, optimizer_value,
             optimizer_q_net_1, optimizer_q_net_2, states, actions, rewards, next_states, masks, gamma, polyak,
             update_target=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update qvalue net"""

    q_value_1 = q_net_1(states, actions)
    q_value_2 = q_net_2(states, actions)
    with torch.no_grad():
        target_next_value = rewards + gamma * \
            masks * value_net_target(next_states)

    q_value_loss_1 = nn.MSELoss()(q_value_1, target_next_value)
    optimizer_q_net_1.zero_grad()
    q_value_loss_1.backward()
    optimizer_q_net_1.step()

    q_value_loss_2 = nn.MSELoss()(q_value_2, target_next_value)
    optimizer_q_net_2.zero_grad()
    q_value_loss_2.backward()
    optimizer_q_net_2.step()

    """update policy net"""
    new_actions, log_probs = policy_net.rsample(states)
    min_q = torch.min(
        q_net_1(states, new_actions),
        q_net_2(states, new_actions)
    )
    policy_loss = (log_probs - min_q).mean()
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    """update value net"""
    target_value = (min_q - log_probs).detach()
    value_loss = nn.MSELoss()(value_net(states), target_value)
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    if update_target:
        """ update target value net """
        value_net_target_flat_params = get_flat_params(value_net_target)
        value_net_flat_params = get_flat_params(value_net)

        set_flat_params(value_net_target,
                        (1 - polyak) * value_net_flat_params + polyak * value_net_target_flat_params)

    return {"target_value_loss": value_loss,
            "q_value_loss_1": q_value_loss_1,
            "q_value_loss_2": q_value_loss_2,
            "policy_loss": policy_loss
            }


class SAC:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_v=1e-3,
                 lr_q=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 explore_size=10000,
                 step_per_iter=3000,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 target_update_delay=1,
                 seed=1,
                 model_path=None
                 ):
        self.env_id = env_id
        self.gamma = gamma
        self.polyak = polyak
        self.memory = Memory(memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.lr_q = lr_q
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.target_update_delay = target_update_delay
        self.model_path = model_path
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, self.num_actions = get_env_info(
            self.env_id)
        assert env_continuous, "SAC is only applicable to continuous environment !!!!"

        self.action_low, self.action_high = self.env.action_space.low[
            0], self.env.action_space.high[0]
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Policy(num_states, self.num_actions,
                                 max_action=self.action_high, use_sac=True).to(device)

        self.value_net = Value(num_states).to(device)
        self.value_net_target = Value(num_states).to(device)

        self.q_net_1 = QValue(num_states, self.num_actions).to(device)
        self.q_net_2 = QValue(num_states, self.num_actions).to(device)

        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_sac.p".format(self.env_id))
            self.policy_net, self.value_net, self.q_net_1, self.q_net_2, self.running_state \
                = pickle.load(open('{}/{}_sac.p'.format(self.model_path, self.env_id), "rb"))

        self.value_net_target.load_state_dict(self.value_net.state_dict())

        self.optimizer_p = optim.Adam(
            self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v = optim.Adam(
            self.value_net.parameters(), lr=self.lr_v)
        self.optimizer_q_1 = optim.Adam(
            self.q_net_1.parameters(), lr=self.lr_q)
        self.optimizer_q_2 = optim.Adam(
            self.q_net_2.parameters(), lr=self.lr_q)

    def choose_action(self, state):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _ = self.policy_net.rsample(state)
        action = action.cpu().numpy()[0]
        return action

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            state = self.running_state(state)
            action = self.choose_action(state)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter + 1
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        while num_steps < self.step_per_iter:
            state = self.env.reset()
            state = self.running_state(state)
            episode_reward = 0

            for t in range(10000):

                if self.render:
                    self.env.render()

                if global_steps < self.explore_size:  # explore
                    action = self.env.action_space.sample()
                else:  # action
                    action = self.choose_action(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.running_state(next_state)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, None)

                episode_reward += reward
                global_steps += 1
                num_steps += 1

                if global_steps >= self.min_update_step and global_steps % self.update_step == 0:
                    for k in range(1, self.update_step + 1):
                        batch = self.memory.sample(
                            self.batch_size)  # random sample batch
                        self.update(batch, k)

                if done or num_steps >= self.step_per_iter:
                    break

                state = next_state

            num_episodes += 1
            total_reward += episode_reward
            min_episode_reward = min(episode_reward, min_episode_reward)
            max_episode_reward = max(episode_reward, max_episode_reward)

        self.env.close()

        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_episode_reward'] = max_episode_reward
        log['min_episode_reward'] = min_episode_reward

        print(f"Iter: {i_iter}, num steps: {log['num_steps']}, total reward: {log['total_reward']: .4f}, "
              f"min reward: {log['min_episode_reward']: .4f}, max reward: {log['max_episode_reward']: .4f}, "
              f"average reward: {log['avg_reward']: .4f}")

        # record reward information
        writer.add_scalar("total reward", log['total_reward'], i_iter)
        writer.add_scalar("average reward", log['avg_reward'], i_iter)
        writer.add_scalar("min reward", log['min_episode_reward'], i_iter)
        writer.add_scalar("max reward", log['max_episode_reward'], i_iter)
        writer.add_scalar("num steps", log['num_steps'], i_iter)

    def update(self, batch, k_iter):
        """learn model"""
        batch_state = FLOAT(batch.state).to(device)
        batch_action = FLOAT(batch.action).to(device)
        batch_reward = FLOAT(batch.reward).to(device)
        batch_next_state = FLOAT(batch.next_state).to(device)
        batch_mask = FLOAT(batch.mask).to(device)

        # update by SAC
        alg_step_stats = sac_step(self.policy_net, self.value_net, self.value_net_target, self.q_net_1, self.q_net_2,
                                  self.optimizer_p, self.optimizer_v, self.optimizer_q_1, self.optimizer_q_2, batch_state,
                                  batch_action, batch_reward, batch_next_state, batch_mask, self.gamma, self.polyak,
                                  k_iter % self.target_update_delay == 0)

    def save(self, save_path):
        """save model"""
        check_path(save_path)

        pickle.dump((self.policy_net, self.value_net, self.q_net_1, self.q_net_2, self.running_state),
                    open('{}/{}_sac.p'.format(save_path, self.env_id), 'wb'))



def main(env_id='Swimmer-v2', render=False, num_process=1, lr_p=1e-3, lr_v=1e-3, lr_q=1e-3, gamma=0.99, polyak=0.995,explore_size=1e4, memory_size=1e6,
         step_per_iter=1000, batch_size=256, min_update_step=1000, update_step=50, max_iter=500, eval_iter=50,
         save_iter=50, target_update_delay=1, model_path='trained', log_path='./log/', seed=1):
    base_dir = log_path + env_id + "/SAC_exp{}".format(seed)
    writer = SummaryWriter(base_dir)
    sac = SAC(env_id,
              render=render,
              num_process=num_process,
              memory_size=memory_size,
              lr_p=lr_p,
              lr_v=lr_v,
              lr_q=lr_q,
              gamma=gamma,
              polyak=polyak,
              explore_size=explore_size,
              step_per_iter=step_per_iter,
              batch_size=batch_size,
              min_update_step=min_update_step,
              update_step=update_step,
              target_update_delay=target_update_delay,
              seed=seed)

    for i_iter in range(1, max_iter + 1):
        sac.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            sac.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            sac.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
