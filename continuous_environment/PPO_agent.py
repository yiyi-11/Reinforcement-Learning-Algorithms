'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:17:42
LastEditor: John
LastEditTime: 2021-04-28 10:11:09
Discription:
Environment:
'''
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
import torch
from collections import deque
import matplotlib.pyplot as plt


class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class Critic(nn.Module):
    def __init__(self, state_dim,hidden_dim=256):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        batch_step = np.arange(0, len(self.states), self.batch_size)
        indices = np.arange(len(self.states), dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_step]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def push(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class PPO:
    def __init__(self, state_dim, action_dim, gamma=0.95, batch_size=64, lr=0.001):
        self.action_dim = action_dim
        self.action_list = [a for a in range(action_dim)]
        self.gamma = gamma
        self.policy_clip = 0.2
        self.n_epochs = 4  # 每一次update的采样更新次数
        self.gae_lambda = 0.95
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        # self.replay_buffer_bound = 2000  # 经验回放容器的最大容纳个数
        # self.replay_buffer = deque(maxlen=self.replay_buffer_bound)
        # self.batch_size = batch_size
        self.memory = PPOMemory(batch_size)
        self.loss = 0

        self.highest_state = -0.3
        self.lowest_state = -0.7

    def additional_reward(self, reward, new_state):
        # rewrite to fit the RL environment
        if new_state[0] >= 0.3:
            reward += 1
            print('higher!')
        elif new_state[0] >= self.highest_state:
            reward += 1
            self.highest_state = new_state[0]
            print("highhigh!")
        elif new_state[0] <= self.lowest_state:
            reward += 2
            self.lowest_state = new_state[0]
            print("lowlow!")
        elif np.abs(new_state[1]) >= 0.03:
            reward += 1
            print("speed speed!")
        return reward

    def add_replay_memory(self, state, action, prob, val, reward, new_state, is_done):
        # reward = self.additional_reward(reward, new_state)
        self.memory.push(state, action, prob, val, reward, is_done)

    def choose_action(self, observation, epsilon=0.1):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        # print(action)
        if np.random.uniform() < epsilon:
            action = np.random.choice(self.action_list)
            action = torch.tensor(action).to(self.device)
        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def update(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, batches = self.memory.sample()

            values = vals_arr
            ### compute advantage ###
            advantage = np.zeros(len(reward_arr), dtype=np.float32)
            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1] * (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)
            ### SGD ###
            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5*critic_loss
                self.loss = total_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.memory.clear()

    def save_model(self, file_path):
        print('model saved')
        torch.save(self.actor.state_dict(), file_path+'_ppo_actor_checkpoint.pth')
        torch.save(self.critic.state_dict(), file_path + '_ppo_critic_checkpoint.pth')

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + '_ppo_actor_checkpoint.pth'))
        self.critic.load_state_dict(torch.load(path + '_ppo_critic_checkpoint.pth'))


if __name__ == '__main__':
    random_seed = 1
    env_name, state_dim, action_dim = "MountainCar-v0", 2, 3
    # env_name, state_dim, action_dim = "CartPole-v0", 4, 2
    env = gym.make(env_name)
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    episodes = 5000  # 训练1000次
    update_frequency = 20  # 网络更新频率
    score_list = []  # 记录所有分数
    moving_average_reward_list = []  # 记录滑动平均分数
    agent = PPO(state_dim, action_dim, batch_size=5)

    print('Start to train !')
    running_steps = 0
    for i in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            # env.render()
            # if i > 200:
            #     env.render()
            action, prob, val = agent.choose_action(state)
            # print(action)
            state_, reward, done, _ = env.step(action)
            running_steps += 1
            score += reward
            agent.add_replay_memory(state, action, prob, val, reward, state_, done)
            if running_steps % update_frequency == 0:
                agent.update()
            state = state_
        score_list.append(score)
        if moving_average_reward_list:
            moving_average_reward_list.append(
                0.9 * moving_average_reward_list[-1] + 0.1 * score)
        else:
            moving_average_reward_list.append(score)
        print('episode:', i, 'score:', score, 'max:', max(score_list))
    print('Complete training！')

    agent.save_model(env_name)

    plt.figure(1)
    plt.plot(np.arange(len(score_list)), score_list, label="score")
    plt.plot(np.arange(len(moving_average_reward_list)), moving_average_reward_list, label="moving average score")
    plt.ylabel('score')
    plt.xlabel('training episode')
    plt.legend()
    plt.show()
    env.close()

    np.save(file=f'data/{env_name}_ppo_score.npy', arr=np.array(score_list))
    np.save(file=f'data/{env_name}_ppo_ma_reward.npy', arr=np.array(moving_average_reward_list))


