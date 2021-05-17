import random
import gym
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from MLP import ActorCritic


class A2C(object):
    def __init__(self, state_dim, action_dim, gamma=0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_list = [act for act in range(self.action_dim)]
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.step = 0

        self.loss_list = []
        self.total_loss = 0.0

    def choose_action(self, state, epsilon=0.1):
        if np.random.uniform() < epsilon - self.step * 0.0002:
            return np.random.choice(self.action_list)
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        probs, value = self.model(state)
        action = probs.sample()  # 按概率分布选action
        action = action.cpu().detach().numpy()  # 转为标量
        return action

    def train_one_step(self, state, action, reward, new_state, is_done):
        self.step += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        # print(state)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        # print(reward)
        new_state = torch.tensor(new_state, device=self.device, dtype=torch.float32)
        # print(new_state)
        V_state = self.model.forward_critic(state)
        # print(V_state)
        V_new_state = self.model.forward_critic(new_state)
        # print(V_new_state)
        probs_action = self.model.forward_actor(state)
        entropy = Categorical(probs_action).entropy()
        # print('entropy: ', entropy)
        # print(probs_action)
        # print(action)
        # action_onehot = torch.zeros([self.action_dim]).scatter_(-1, torch.tensor(action), 1).to(self.device)
        log_prob = torch.log(probs_action[action])
        # print(log_prob)

        advantage_function = reward + V_new_state - V_state

        self.optimizer.zero_grad()

        critic_loss = torch.pow(advantage_function, 2)
        actor_loss = -log_prob * advantage_function
        # print('actor_loss: ', actor_loss)
        loss = actor_loss + 0.5*critic_loss - 0.001*entropy
        loss.backward()
        self.total_loss += loss.cpu().detach().numpy()

        self.optimizer.step()
        if is_done:
            self.loss_list.append(self.total_loss)
            self.total_loss = 0.0

    def save_model(self, path):
        torch.save(self.model.state_dict(), path + 'a2c_checkpoint.pth')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path + 'a2c_checkpoint.pth'))


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

    # agent
    agent = A2C(state_dim, action_dim)

    for i in range(episodes):
        s = env.reset()
        score = 0
        while True:
            # env.render()
            a = agent.choose_action(s)
            next_s, reward, done, _ = env.step(a)
            score += reward
            reward = additional_reward(reward, next_s)
            agent.train_one_step(s, a, reward, next_s, done)  # update

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

    plt.figure(2)
    plt.plot(np.arange(len(agent.loss_list)), agent.loss_list)
    plt.ylabel("abs loss")
    plt.xlabel("training episode")
    plt.legend()
    plt.show()
    env.close()

    np.save(file=f'data/{env_name}_a2c_score.npy', arr=np.array(score_list))
    np.save(file=f'data/{env_name}_a2c_ma_reward.npy', arr=np.array(moving_average_reward_list))

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
