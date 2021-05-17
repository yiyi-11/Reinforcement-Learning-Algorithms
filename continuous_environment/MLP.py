import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MLPdqn(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(MLPdqn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MLPPG(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的feature即环境的state数目
            output_dim: 输出的action总个数
        """
        super(MLPPG, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        # x = torch.sigmoid(self.fc3(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc_critic = nn.Linear(hidden_dim, 1)
        self.fc_actor = nn.Linear(hidden_dim, output_dim)

        # V
        self.critic = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc_critic
        )

        # Q
        self.actor = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc_actor,
            nn.Softmax(dim=-1),
        )

    def forward_critic(self, x):
        return self.critic(x)

    def forward_actor(self, x):
        probs = self.actor(x)
        return probs

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value
