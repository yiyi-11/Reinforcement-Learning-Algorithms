from gym import Env
import matplotlib.pyplot as plt
import numpy as np

from gridworld import GridWorldEnv, SimpleGridWorld, CliffWalk, LargeGridWorld, SimpleGridWorld_WithWall
from agentgrid import AgentGrid

# env = GridWorldEnv(n_width=12,  # 水平方向格子数量
#                    n_height=4,  # 垂直方向格子数量
#                    u_size=60,  # 可以根据喜好调整大小
#                    default_reward=-1,  # 默认格子的即时奖励值
#                    default_type=0)  # 默认的格子都是可以进入的
# from gym import spaces  # 导入spaces
#
# env.action_space = spaces.Discrete(4)  # 设置行为空间支持的行为数量
#
# env.start = (0, 0)
# env.ends = [(11, 0)]
#
# for i in range(10):
#     env.rewards.append((i + 1, 0, -100))
#     env.ends.append((i + 1, 0))
#
# env.types = [(5, 1, 1), (5, 2, 1)]
# env.refresh_setting()
# env.reset()
# env.render()
# input("press any key to continue...")


env = SimpleGridWorld_WithWall()
env.reset()
env.render()
input("press any key to continue...")
