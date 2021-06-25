import gym
from gym.spaces import Discrete
import os
import numpy as np
import math
from torch.multiprocessing import Process, Queue
import time
import torch
import random
from collections import namedtuple, deque

__all__ = ['get_env_info', 'get_env_space']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLOAT = torch.FloatTensor

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob'))

def estimate_advantages(rewards, masks, values, gamma, tau):
    tensor_type = type(rewards)
    deltas = tensor_type(rewards.size(0), 1).to(device)
    advantages = tensor_type(rewards.size(0), 1).to(device)

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

    return advantages, returns


def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")

def get_env_space(env_id):
    env = gym.make(env_id)
    # 解除环境限制
    # env = env.unwrapped
    num_states = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    return env, num_states, num_actions


def get_env_info(env_id, unwrap=False):
    env = gym.make(env_id)
    if unwrap:  # 解除环境限制
        env = env.unwrapped
    num_states = env.observation_space.shape[0]
    env_continuous = False
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]
        env_continuous = True

    return env, env_continuous, num_states, num_actions

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)
        self.fix = False

    def __call__(self, x, update=True):
        if update and not self.fix:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

class Memory(object):
    def __init__(self, size=None):
        self.memory = deque(maxlen=size)

    # save item
    def push(self, *args):
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory.clear()

    def append(self, other):
        self.memory += other.memory

    # sample a mini_batch
    def sample(self, batch_size=None):
        # sample all transitions
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:  # sample with size: batch_size
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def __len__(self):
        return len(self.memory)



def collect_samples(pid, queue, env, policy, render, running_state, custom_reward, min_batch_size):
    torch.set_num_threads(1)
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, 'np_random'):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, 'env') and hasattr(env.env, 'np_random'):
            env.env.np_random.seed(env.env.np_random.randint(5000) * pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    num_episodes = 0

    min_episode_reward = float('inf')
    max_episode_reward = float('-inf')
    total_reward = 0

    while num_steps < min_batch_size:
        state = env.reset()
        episode_reward = 0
        if running_state:
            state = running_state(state)

        for t in range(10000):
            if render:
                env.render()
            state_tensor = FLOAT(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = policy.get_action_log_prob(state_tensor)
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            if custom_reward:
                reward = custom_reward(state, action)
            episode_reward += reward

            if running_state:
                next_state = running_state(next_state)

            mask = 0 if done else 1
            # ('state', 'action', 'reward', 'next_state', 'mask', 'log_prob')
            memory.push(state, action, reward, next_state, mask, log_prob)
            num_steps += 1
            if done or num_steps >= min_batch_size:
                break

            state = next_state

        # num_steps += (t + 1)
        num_episodes += 1
        total_reward += episode_reward
        min_episode_reward = min(episode_reward, min_episode_reward)
        max_episode_reward = max(episode_reward, max_episode_reward)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_episode_reward'] = max_episode_reward
    log['min_episode_reward'] = min_episode_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_episode_reward'] = max(
        [x['max_episode_reward'] for x in log_list])
    log['min_episode_reward'] = min(
        [x['min_episode_reward'] for x in log_list])

    return log


class MemoryCollector:
    def __init__(self, env, policy, render=False, running_state=None, custom_reward=None, num_process=1):
        self.env = env
        self.policy = policy
        self.running_state = running_state
        self.custom_reward = custom_reward
        self.render = render
        self.num_process = num_process

    def collect_samples(self, min_batch_size):
        torch.set_num_threads(1)
        self.policy.to(torch.device('cpu'))
        t_start = time.time()
        process_batch_size = int(math.floor(min_batch_size / self.num_process))
        queue = Queue()
        workers = []

        for i in range(self.num_process - 1):
            # don't render other parallel processes
            worker_args = (i + 1, queue, self.env, self.policy,
                           False, self.running_state, self.custom_reward, process_batch_size)
            p = Process(target=collect_samples, args=worker_args)
            workers.append(p)

        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy,
                                      self.render, self.running_state, self.custom_reward, process_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get(timeout=0.5)
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log

        [worker.join() for worker in workers]

        # concat all memories
        for worker_memory in worker_memories:
            memory.append(worker_memory)

        if self.num_process > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)

        t_end = time.time()
        log['sample_time'] = t_end - t_start

        self.policy.to(device)
        return memory, log


