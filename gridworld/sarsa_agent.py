import random
import numpy as np
import matplotlib.pyplot as plt
from gym import Env

from gridworld import SimpleGridWorld, CliffWalk, SimpleGridWorld_WithWall
from agentgrid import AgentGrid


class SarsaAgent(AgentGrid):
    def __init__(self, env: Env):
        super(SarsaAgent, self).__init__(env)  # agent环境初始化
        self.reward_episode = []
        self.time_episode = []

    def learning(self, gamma, alpha, max_episode_num, render):
        total_time, time_in_episode, num_episode = 0, 0, 0
        while num_episode < max_episode_num:  # 设置终止条件
            self.state = self.env.reset()  # 环境初始化
            s0 = self._get_state_name(self.state)  # 获取个体对于观测的命名
            if render:
                self.env.render()  # 显示UI界面

            a0 = self.performPolicy(s0, num_episode, use_epsilon=True)
            s1 = None

            time_in_episode = 0
            reward_in_episode = 0
            is_done = False
            while not is_done:  # 针对一个Episode内部
                s1, r1, is_done, info = self.act(a0)  # 执行行为
                reward_in_episode += r1  # 更新total reward
                if render:
                    self.env.render()  # 更新UI界面
                s1 = self._get_state_name(s1)  # 获取个体对于新状态的命名
                self._assert_state_in_Q(s1, randomized=True)
                # get next action a'
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                # get new Q(s, a) <- Q(s, a) + alpha*(reward + gamma*Q(s', a') - Q(s, a))
                old_q = self._get_Q(s0, a0)
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                # 更新state, action
                s0 = s1
                a0 = a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps, reward {2}.".format(
                num_episode, time_in_episode, reward_in_episode))  # 显示每一个Episode花费了多少步
            total_time += time_in_episode
            num_episode += 1
            if num_episode == max_episode_num:  # 终端显示最后Episode的信息
                print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".format(time_in_episode, s0, a0, s1))
            self.time_episode.append(time_in_episode)
            self.reward_episode.append(reward_in_episode)

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

    agent = SarsaAgent(env)
    print("Learning...")
    agent.learning(gamma=0.9,
                   alpha=0.1,
                   max_episode_num=1000,
                   render=False)
    agent.plot_cost()

    np.save(file=f'data/{env_name}_sarsa_reward.npy', arr=np.array(agent.reward_episode))
    np.save(file=f'data/{env_name}_sarsa_step.npy', arr=np.array(agent.time_episode))

    # agent.learning(gamma=0.9,
    #                alpha=0.1,
    #                max_episode_num=20,
    #                render=True)
