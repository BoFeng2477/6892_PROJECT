
import torch
from torch.utils.tensorboard import SummaryWriter

import pickle

import numpy as np
import torch.optim as optim
import torch.nn as nn

from from_reference.Policy_ddpg import Policy
from from_reference.Value_ddpg import Value
from from_reference.Util import get_env_info, Memory, check_path, ZFilter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT = torch.FloatTensor

def get_flat_params(model: nn.Module):

    return torch.cat([param.view(-1) for param in model.parameters()])


def get_flat_grad_params(model: nn.Module):

    return torch.cat(
        [param.grad.view(-1) if param.grad is not None else torch.zeros(param.view(-1).shape) for param in
         model.parameters()])

def td3_step(policy_net, policy_net_target, value_net_1, value_net_target_1, value_net_2, value_net_target_2,
             optimizer_policy, optimizer_value_1, optimizer_value_2, states, actions, rewards, next_states, masks,
             gamma, polyak, target_action_noise_std, target_action_noise_clip, action_high, update_policy=False):
    rewards = rewards.unsqueeze(-1)
    masks = masks.unsqueeze(-1)

    """update critic"""
    with torch.no_grad():
        target_action = policy_net_target(next_states)
        target_action_noise = torch.clamp(torch.randn_like(target_action) * target_action_noise_std,
                                          -target_action_noise_clip, target_action_noise_clip)
        target_action = torch.clamp(
            target_action + target_action_noise, -action_high, action_high)
        target_values = rewards + gamma * masks * torch.min(value_net_target_1(next_states, target_action),
                                                            value_net_target_2(next_states, target_action))

    """update value1 target"""
    values_1 = value_net_1(states, actions)
    value_loss_1 = nn.MSELoss()(target_values, values_1)

    optimizer_value_1.zero_grad()
    value_loss_1.backward()
    optimizer_value_1.step()

    """update value2 target"""
    values_2 = value_net_2(states, actions)
    value_loss_2 = nn.MSELoss()(target_values, values_2)

    optimizer_value_2.zero_grad()
    value_loss_2.backward()
    optimizer_value_2.step()

    policy_loss = None
    if update_policy:
        """update policy"""
        policy_loss = - value_net_1(states, policy_net(states)).mean()

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        """soft update target nets"""
        policy_net_flat_params = get_flat_params(policy_net)
        policy_net_target_flat_params = get_flat_params(policy_net_target)
        set_flat_params(policy_net_target,
                        polyak * policy_net_target_flat_params + (1 - polyak) * policy_net_flat_params)

        value_net_1_flat_params = get_flat_params(value_net_1)
        value_net_1_target_flat_params = get_flat_params(value_net_target_1)
        set_flat_params(value_net_target_1,
                        polyak * value_net_1_target_flat_params + (1 - polyak) * value_net_1_flat_params)

        value_net_2_flat_params = get_flat_params(value_net_2)
        value_net_2_target_flat_params = get_flat_params(value_net_target_2)
        set_flat_params(value_net_target_2,
                        polyak * value_net_2_target_flat_params + (1 - polyak) * value_net_2_flat_params)

    return {"q_value_loss_1": value_loss_1,
            "q_value_loss_2": value_loss_2,
            "policy_loss": policy_loss
            }


class TD3:
    def __init__(self,
                 env_id,
                 render=False,
                 num_process=1,
                 memory_size=1000000,
                 lr_p=1e-3,
                 lr_v=1e-3,
                 gamma=0.99,
                 polyak=0.995,
                 action_noise=0.1,
                 target_action_noise_std=0.2,
                 target_action_noise_clip=0.5,
                 explore_size=10000,
                 step_per_iter=3000,
                 batch_size=100,
                 min_update_step=1000,
                 update_step=50,
                 policy_update_delay=2,
                 seed=1,
                 model_path=None
                 ):
        self.env_id = env_id
        self.gamma = gamma
        self.polyak = polyak
        self.action_noise = action_noise
        self.target_action_noise_std = target_action_noise_std
        self.target_action_noise_clip = target_action_noise_clip
        self.memory = Memory(memory_size)
        self.explore_size = explore_size
        self.step_per_iter = step_per_iter
        self.render = render
        self.num_process = num_process
        self.lr_p = lr_p
        self.lr_v = lr_v
        self.batch_size = batch_size
        self.min_update_step = min_update_step
        self.update_step = update_step
        self.policy_update_delay = policy_update_delay
        self.model_path = model_path
        self.seed = seed

        self._init_model()

    def _init_model(self):
        """init model from parameters"""
        self.env, env_continuous, num_states, self.num_actions = get_env_info(
            self.env_id)
        assert env_continuous, "TD3 is only applicable to continuous environment !!!!"

        self.action_low, self.action_high = self.env.action_space.low[
            0], self.env.action_space.high[0]
        # seeding
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.seed(self.seed)

        self.policy_net = Policy(
            num_states, self.num_actions, self.action_high).to(device)
        self.policy_net_target = Policy(
            num_states, self.num_actions, self.action_high).to(device)

        self.value_net_1 = Value(num_states, self.num_actions).to(device)
        self.value_net_target_1 = Value(
            num_states, self.num_actions).to(device)
        self.value_net_2 = Value(num_states, self.num_actions).to(device)
        self.value_net_target_2 = Value(
            num_states, self.num_actions).to(device)

        self.running_state = ZFilter((num_states,), clip=5)

        if self.model_path:
            print("Loading Saved Model {}_td3.p".format(self.env_id))
            self.policy_net, self.value_net_1, self.value_net_2, self.running_state = pickle.load(
                open('{}/{}_td3.p'.format(self.model_path, self.env_id), "rb"))

        self.policy_net_target.load_state_dict(self.policy_net.state_dict())
        self.value_net_target_1.load_state_dict(self.value_net_1.state_dict())
        self.value_net_target_2.load_state_dict(self.value_net_2.state_dict())

        self.optimizer_p = optim.Adam(
            self.policy_net.parameters(), lr=self.lr_p)
        self.optimizer_v_1 = optim.Adam(
            self.value_net_1.parameters(), lr=self.lr_v)
        self.optimizer_v_2 = optim.Adam(
            self.value_net_2.parameters(), lr=self.lr_v)

    def choose_action(self, state, noise_scale):
        """select action"""
        state = FLOAT(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action, log_prob = self.policy_net.get_action_log_prob(state)
        action = action.cpu().numpy()[0]
        # add noise
        noise = noise_scale * np.random.randn(self.num_actions)
        action += noise
        action = np.clip(action, -self.action_high, self.action_high)
        return action

    def eval(self, i_iter, render=False):
        """evaluate model"""
        state = self.env.reset()
        test_reward = 0
        while True:
            if render:
                self.env.render()
            # state = self.running_state(state)
            action = self.choose_action(state, 0)
            state, reward, done, _ = self.env.step(action)

            test_reward += reward
            if done:
                break
        print(f"Iter: {i_iter}, test Reward: {test_reward}")
        self.env.close()

    def learn(self, writer, i_iter):
        """interact"""
        global_steps = (i_iter - 1) * self.step_per_iter
        log = dict()
        num_steps = 0
        num_episodes = 0
        total_reward = 0
        min_episode_reward = float('inf')
        max_episode_reward = float('-inf')

        while num_steps < self.step_per_iter:
            state = self.env.reset()
            # state = self.running_state(state)
            episode_reward = 0

            for t in range(10000):

                if self.render:
                    self.env.render()

                if global_steps < self.explore_size:  # explore
                    action = self.env.action_space.sample()
                else:  # action with noise
                    action = self.choose_action(state, self.action_noise)

                next_state, reward, done, _ = self.env.step(action)
                # next_state = self.running_state(next_state)
                mask = 0 if done else 1
                # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
                self.memory.push(state, action, reward, next_state, mask, None)

                episode_reward += reward
                global_steps += 1
                num_steps += 1

                if global_steps >= self.min_update_step and global_steps % self.update_step == 0:
                    for k in range(self.update_step):
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

        # update by TD3
        alg_step_stats = td3_step(self.policy_net, self.policy_net_target, self.value_net_1, self.value_net_target_1, self.value_net_2,
                                  self.value_net_target_2, self.optimizer_p, self.optimizer_v_1, self.optimizer_v_2, batch_state,
                                  batch_action, batch_reward, batch_next_state, batch_mask, self.gamma, self.polyak,
                                  self.target_action_noise_std, self.target_action_noise_clip, self.action_high,
                                  k_iter % self.policy_update_delay == 0)

    def save(self, save_path):
        """save model"""
        check_path(save_path)
        pickle.dump((self.policy_net, self.value_net_1, self.value_net_2, self.running_state),
                    open('{}/{}_td3.p'.format(save_path, self.env_id), 'wb'))


def main(env_id='Swimmer-v2', render=False, num_process=1, lr_p=1e-3, lr_v=1e-3, gamma=0.99, polyak=0.995, target_action_noise_std=0.2, target_action_noise_clip=0.5,
         explore_size=1e4, memory_size=1e6, step_per_iter=1000, batch_size=256, min_update_step=1000, update_step=50, max_iter=500, eval_iter=50,
         save_iter=50, action_noise=0.1, policy_update_delay=2, model_path='trained', log_path='./log/', seed=1):
    base_dir = log_path + env_id + "/TD3_exp{}".format(seed)
    writer = SummaryWriter(base_dir)

    td3 = TD3(env_id,
              render=render,
              num_process=num_process,
              memory_size=memory_size,
              lr_p=lr_p,
              lr_v=lr_v,
              gamma=gamma,
              polyak=polyak,
              target_action_noise_std=target_action_noise_std,
              target_action_noise_clip=target_action_noise_clip,
              explore_size=explore_size,
              step_per_iter=step_per_iter,
              batch_size=batch_size,
              min_update_step=min_update_step,
              update_step=update_step,
              action_noise=action_noise,
              policy_update_delay=policy_update_delay,
              seed=seed)

    for i_iter in range(1, max_iter + 1):
        td3.learn(writer, i_iter)

        if i_iter % eval_iter == 0:
            td3.eval(i_iter, render=render)

        if i_iter % save_iter == 0:
            td3.save(model_path)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
