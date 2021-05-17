from gym import Env
from random import random


class AgentGrid(object):
    """
    General Agent for gridworld. (contains Q table, limited state numbers)
    learning method should be rewrote.
    """
    def __init__(self, env):
        self.env = env  # 个体持有环境的引用
        self.Q = {}  # 个体维护一张行为价值表Q
        self.V = {}  # state value table
        self.state = None  # 个体当前的观测，最好写成obs.
        self.resetAgent()

    def resetAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized=False)
        for i in range(self.env.n_width * self.env.n_height):
            self._assert_state_in_V(str(i), randomized=True)
        # self._assert_state_in_V(s_name, randomized=False)

    def performPolicy(self, s, episode_num, use_epsilon):  # e-greedy policy
        # epsilon = 1.00 / (episode_num + 1)
        epsilon = 0.2
        Q_s = self.Q[s]
        rand_value = random()
        if use_epsilon and rand_value < epsilon - episode_num * 0.0002:
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action

    def act(self, a):  # 执行一个行为
        return self.env.step(a)

    def learning(self, gamma, alpha, max_episode_num, render):
        # override by specific agent
        pass

    def _is_state_in_Q(self, s):
        return self.Q.get(s) is not None

    def _is_state_in_V(self, s):
        return self.V.get(s) is not None

    def _init_state_value(self, s_name, randomized=True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n):
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v
        if not self._is_state_in_V(s_name):
            self.V[s_name] = random() / 10 if randomized is True else 0.0

    def _assert_state_in_Q(self, s, randomized=True):
        # can't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _assert_state_in_V(self, s, randomized=True):
        # can't find the state
        if not self._is_state_in_V(s):
            self._init_state_value(s, randomized)

    def _get_state_name(self, state):
        # 得到状态对应的字符串作为以字典存储的价值函数的键值，应针对不同的状态值单独设计，避免重复
        # 这里仅针对grid world
        return str(state)

    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _get_max_Q(self, s):
        self._assert_state_in_Q(s, randomized=True)
        Q_s = self.Q[s]
        return max(Q_s.values())

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _get_V(self, s):
        self._assert_state_in_V(s, randomized=True)
        return self.V[s]

    def _set_V(self, s, value):
        self._assert_state_in_V(s, randomized=True)
        self.V[s] = value
