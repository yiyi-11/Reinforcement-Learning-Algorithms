# REINFORCE: Monte-Carlo Policy Gradient algorithm
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from MLP import MLPPG


class PolicyGradient:
    def __init__(self, state_dim, action_dim, gamma=0.95, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_list = [act for act in range(self.action_dim)]
        self.policy_model = MLPPG(state_dim, action_dim).to(self.device)
        # self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(), lr=0.001)
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.step = 0

        self.loss_list = []

    def choose_action(self, state, epsilon=0.1):
        self.step += 1
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice(self.action_list)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        probs = self.policy_model(state)
        action = torch.multinomial(probs, num_samples=1, replacement=False)  # 按概率分布选action
        action = action.cpu().detach().numpy()[0]  # 转为标量
        return action

    def train_one_step(self, reward_pool, state_pool, action_pool):
        total_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add
            # running_add = running_add * self.gamma + reward_pool[i]
            # reward_pool[i] = running_add

        # Normalize reward (使得reward有正有负)
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Decent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = torch.tensor(action_pool[i], device=self.device)
            discount_reward = reward_pool[i]

            state = torch.tensor(state, device=self.device, dtype=torch.float32)
            probs = self.policy_model(state)  # 每个动作对应的预测值
            # print("Gt: ", discount_reward)
            # print("prob action: ", probs[action])
            # probs = probs / torch.sum(probs)  # 计算每个动作的概率
            loss = -torch.log(probs[action]) * discount_reward  # Negative score function x reward
            total_loss += torch.abs(loss)

            # print(loss)
            loss.backward()
        self.optimizer.step()
        self.loss_list.append(total_loss.cpu().detach().numpy())

    def save_model(self, path):
        torch.save(self.policy_model.state_dict(), path + 'pg_checkpoint.pth')

    def load_model(self, path):
        self.policy_model.load_state_dict(torch.load(path + 'pg_checkpoint.pth'))


def additional_reward(reward, new_state):
    # rewrite to fit the RL environment
    # if new_state[0] >= 0.35:
    #     reward += 1
    return reward


if __name__ == "__main__":
    random_seed = 1
    env_name, state_dim, action_dim = "MountainCar-v0", 2, 3
    # env_name, state_dim, action_dim = "CartPole-v0", 4, 2
    env = gym.make(env_name)
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    episodes = 5000  # 训练1000次
    score_list = []  # 记录所有分数
    moving_average_reward_list = []  # 记录滑动平均分数
    state_pool = []
    action_pool = []
    reward_pool = []

    # agent
    agent = PolicyGradient(state_dim, action_dim)

    for i in range(episodes):
        s = env.reset()
        score = 0
        while True:
            # if len(score_list) > 200:
            #     env.render()
            a = agent.choose_action(s)
            next_s, reward, done, _ = env.step(a)
            state_pool.append(s)
            action_pool.append(a)

            score += reward
            reward = additional_reward(reward, next_s)
            reward_pool.append(reward)

            s = next_s
            if done:
                score_list.append(score)
                agent.train_one_step(state_pool=state_pool, action_pool=action_pool, reward_pool=reward_pool)
                state_pool = []  # 每个episode的state
                action_pool = []
                reward_pool = []
                # 计算滑动窗口的reward
                if i == 0:
                    moving_average_reward_list.append(score)
                else:
                    moving_average_reward_list.append(
                        0.9 * moving_average_reward_list[-1] + 0.1 * score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                break
        # # 判断学会后，停止并保存模型
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
    # plt.figure(2)
    # plt.plot(np.arange(len(moving_average_reward_list)), moving_average_reward_list)
    # plt.ylabel('moving average score')
    # plt.xlabel('training episode')
    plt.figure(2)
    plt.plot(np.arange(len(agent.loss_list)), agent.loss_list)
    plt.ylabel("abs loss")
    plt.xlabel("training episode")
    plt.legend()
    plt.show()
    env.close()

    np.save(file=f'data/{env_name}_reinforce_score.npy', arr=np.array(score_list))
    np.save(file=f'data/{env_name}_reinforce_ma_reward.npy', arr=np.array(moving_average_reward_list))

    agent.load_model(env_name)
    for i in range(20):
        s = env.reset()
        score = 0
        while True:
            env.render()
            a = agent.choose_action(s)
            next_s, reward, done, _ = env.step(a)
            score += reward
            s = next_s
            if done:
                score_list.append(score)
                print('episode:', i, 'score:', score, 'max:', max(score_list))
                break

