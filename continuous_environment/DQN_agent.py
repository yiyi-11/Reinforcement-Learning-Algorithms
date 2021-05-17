import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

from MLP import MLPdqn


class DQN(object):
    def __init__(self, state_dim, action_dim, gamma=0.95, batch_size=64, update_step_freq=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.step_in_episode = 0
        self.update_step_freq = update_step_freq
        self.replay_buffer_bound = 2000  # 经验回放容器的最大容纳个数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_list = [act for act in range(self.action_dim)]
        self.replay_buffer = deque(maxlen=self.replay_buffer_bound)
        self.batch_size = batch_size
        self.gamma = gamma  # discounted factor

        self.policy_model = MLPdqn(self.state_dim, self.action_dim).to(self.device)
        self.target_model = MLPdqn(self.state_dim, self.action_dim).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)

    def choose_action(self, state, epsilon=0.1):
        # 衰减的epsilon greedy
        if np.random.uniform() < epsilon - self.step_in_episode * 0.0002:
        # if np.random.uniform() < epsilon / (self.step_in_episode + 1):
            return np.random.choice(self.action_list)
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        return np.argmax(self.policy_model.forward(state).cpu().detach().numpy()[0])

    def additional_reward(self, state, action, reward, new_state, is_done):
        # rewrite to fit the RL environment
        # if new_state[0] >= 0.35:
        #     reward += 1
        return reward

    def add_replay_memory(self, state, action, reward, new_state, is_done):
        reward = self.additional_reward(state, action, reward, new_state, is_done)
        self.replay_buffer.append((state, action, reward, new_state, is_done))

    def train_one_step(self):
        self.step_in_episode += 1

        # 确保有足够的数据用来训练
        if len(self.replay_buffer) < self.batch_size:
            return

        # 更新target model
        if self.step_in_episode % self.update_step_freq == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        # 更新policy model的参数
        # sample batch
        replay_batch = random.sample(self.replay_buffer, k=self.batch_size)
        state_batch, action_batch, reward_batch, new_state_batch, is_done_batch = zip(*replay_batch)

        state_batch = torch.tensor(state_batch, device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # 例如tensor([[1],...,[0]])
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float)  # tensor([1., 1.,...,1])
        new_state_batch = torch.tensor(new_state_batch, device=self.device, dtype=torch.float)
        is_done_batch = torch.tensor(np.float32(is_done_batch), device=self.device)

        q_values = self.policy_model.forward(state_batch).gather(dim=1, index=action_batch)
        max_next_q_values = self.target_model.forward(new_state_batch).max(axis=1)[0].detach()

        target_values = reward_batch + self.gamma * max_next_q_values * (1 - is_done_batch)

        loss = nn.MSELoss()(q_values, target_values.unsqueeze(1))
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, file_path):
        print('model saved')
        self.target_model.load_state_dict(self.policy_model.state_dict())
        torch.save(self.target_model.state_dict(), file_path+'_dqn_checkpoint.pth')

    def load_model(self, path):
        self.target_model.load_state_dict(torch.load(path + '_dqn_checkpoint.pth'))
        self.policy_model.load_state_dict(self.target_model.state_dict())
        # for target_param, param in zip(self.target_model.parameters(), self.policy_model.parameters()):
        #     param.data.copy_(target_param.data)


if __name__ == "__main__":
    random_seed = 1
    env_name, state_dim, action_dim = "MountainCar-v0", 2, 3
    # env_name, state_dim, action_dim = "CartPole-v0", 4, 2
    env = gym.make(env_name)
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    episodes = 10000  # 训练1000次
    score_list = []  # 记录所有分数
    moving_average_reward_list = []  # 记录滑动平均分数
    agent = DQN(state_dim, action_dim)
    for i in range(episodes):
        s = env.reset()
        score = 0
        while True:
            # env.render()
            a = agent.choose_action(s)
            next_s, reward, done, _ = env.step(a)
            agent.add_replay_memory(s, a, reward, next_s, done)
            agent.train_one_step()
            score += reward
            s = next_s
            if done:
                score_list.append(score)
                # 计算滑动窗口的reward
                if i == 0:
                    moving_average_reward_list.append(score)
                else:
                    moving_average_reward_list.append(
                        0.9 * moving_average_reward_list[-1] + 0.1 * score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                break
        # 判断学会后，停止并保存模型
        # if np.mean(score_list[-20:]) > -120 and np.all(score_list[-10:]) > -190:
        #     # agent.save_model("MountainCar-v0")
        #     break
    agent.save_model(env_name)

    plt.figure(1)
    plt.plot(np.arange(len(score_list)), score_list, label="score")
    plt.plot(np.arange(len(moving_average_reward_list)), moving_average_reward_list, label="moving average score")
    plt.ylabel('score')
    plt.xlabel('training episode')
    plt.legend()
    plt.show()
    env.close()

    np.save(file=f'data/{env_name}_dqn_score.npy', arr=np.array(score_list))
    np.save(file=f'data/{env_name}_dqn_ma_reward.npy', arr=np.array(moving_average_reward_list))

    agent.load_model(env_name)
    for i in range(20):
        s = env.reset()
        score = 0
        while True:
            env.render()
            a = agent.choose_action(s)
            next_s, reward, done, _ = env.step(a)
            # agent.add_replay_memory(s, a, reward, next_s, done)
            # agent.train_one_step()
            score += reward
            s = next_s
            if done:
                score_list.append(score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                break
