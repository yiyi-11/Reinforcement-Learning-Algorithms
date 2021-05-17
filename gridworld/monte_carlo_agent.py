import numpy as np
import matplotlib.pyplot as plt
import random
# from gym import Env

from gridworld import SimpleGridWorld, CliffWalk, SimpleGridWorld_WithWall
from agentgrid import AgentGrid


class MonteCarloAgent(AgentGrid):
    def __init__(self, env):
        super(MonteCarloAgent, self).__init__(env)  # agent环境初始化
        self.reward_episode = []
        self.time_episode = []

    def performPolicy(self, s, episode_num, use_epsilon):
        epsilon = 0.2
        s_x, s_y = self.env._state_to_xy(int(s))
        # print(s_x, s_y)
        left = self.env._xy_to_state(s_x-1, s_y)
        right = self.env._xy_to_state(s_x + 1, s_y)
        up = self.env._xy_to_state(s_x, s_y + 1)
        down = self.env._xy_to_state(s_x, s_y - 1)
        choice_spaces = {0: left, 1: right, 2: up, 3: down}
        # print(choice_spaces)

        # boundary effect
        if s_x == 0:
            choice_spaces.pop(0)
        if s_x >= self.env.n_width - 1:
            choice_spaces.pop(1)
        if s_y == 0:
            choice_spaces.pop(3)
        if s_y >= self.env.n_height - 1:
            choice_spaces.pop(2)

        choice_space = choice_spaces.copy()
        for key, value in choice_spaces.items():
            if self.env.is_state_block(value):
                choice_space.pop(key)
            else:
                choice_space[key] = self.V[str(value)]
        # print(choice_space)

        rand_value = random.random()
        if use_epsilon and rand_value < epsilon - episode_num * 0.0002:
            action = random.choice(list(choice_space))
        else:
            str_act = max(choice_space, key=choice_space.get)
            action = int(str_act)
        return action

    def learning(self, gamma, alpha, max_episode_num, render):
        total_time, time_in_episode, num_episode = 0, 0, 0
        N_state = [0] * self.env.n_width * self.env.n_height
        while num_episode < max_episode_num:  # 设置终止条件
            state_series = []
            reward_series = []
            action_series = []
            self.state = self.env.reset()  # 环境初始化
            s0 = self._get_state_name(self.state)  # 获取个体对于观测的命名
            a0 = None
            s1 = None
            state_series.append(s0)
            if render:
                self.env.render()  # 显示UI界面

            time_in_episode = 0
            reward_in_episode = 0
            is_done = False
            # monte carlo模拟
            while not is_done:  # 针对一个Episode内部
                a0 = self.performPolicy(s0, num_episode, use_epsilon=True)
                action_series.append(a0)
                s1, r1, is_done, info = self.act(a0)  # 执行行为

                reward_in_episode += r1  # 更新total reward
                if render:
                    self.env.render()  # 更新UI界面
                s1 = self._get_state_name(s1)  # 获取个体对于新状态的命名
                self._assert_state_in_V(s1, randomized=True)

                state_series.append(s1)
                reward_series.append(r1)

                s0 = s1
                time_in_episode += 1

            # 更新value
            G = 0
            state_series.reverse()
            reverse_state_series = state_series[1:]
            reward_series.reverse()
            reverse_reward_series = reward_series
            for i in range(len(reverse_state_series)):
                G = gamma * G + reverse_reward_series[i]
                N_state[int(reverse_state_series[i])] += 1
                old_v = self._get_V(reverse_state_series[i])
                new_v = old_v + (G - old_v) / N_state[int(reverse_state_series[i])]
                self._set_V(reverse_state_series[i], new_v)

            print("Episode {0} takes {1} steps, reward {2}.".format(
                num_episode, time_in_episode, reward_in_episode))  # 显示每一个Episode花费了多少步
            total_time += time_in_episode
            num_episode += 1
            if num_episode == max_episode_num:  # 终端显示最后Episode的信息
                print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".format(time_in_episode, s0, a0, s1))
            self.time_episode.append(time_in_episode)
            self.reward_episode.append(reward_in_episode)
            # print(action_series)
        # self.env.render()

    def plot_cost(self):
        plt.figure(1)
        plt.plot(np.arange(len(self.reward_episode)), self.reward_episode)
        plt.ylabel('reward')
        plt.xlabel('training episode')
        plt.figure(2)
        plt.plot(np.arange(len(self.time_episode)), self.time_episode)
        plt.ylabel('steps')
        plt.xlabel('training episode')
        plt.show()


if __name__ == "__main__":
    env_name = "SimpleGridWorld_WithWall"  # CliffWalk, SimpleGridWorld, SimpleGridWorld_WithWall
    env = SimpleGridWorld_WithWall()

    random_seed = 1
    env.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent = MonteCarloAgent(env)
    print("Learning...")
    agent.learning(gamma=0.9,
                   alpha=0.1,
                   max_episode_num=1000,
                   render=False)
    agent.plot_cost()

    np.save(file=f'data/{env_name}_mc_reward.npy', arr=np.array(agent.reward_episode))
    np.save(file=f'data/{env_name}_mc_step.npy', arr=np.array(agent.time_episode))

    # agent.learning(gamma=0.9,
    #                alpha=0.1,
    #                max_episode_num=20,
    #                render=True)
