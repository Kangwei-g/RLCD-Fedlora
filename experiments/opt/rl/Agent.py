import os
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Double DQN 智能体
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, lr=1e-3, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, memory_size=5000, tau=0.05):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.memory = deque(maxlen=memory_size)

        self.q_network = QNetwork(state_size, action_size, hidden_size)
        self.target_network = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update_target_network(self):
        for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        layer = 0
        if np.random.rand() <= self.epsilon:
            layer = random.randint(0, self.action_size - 1)
        else:
            if len(self.memory) == 0:
                layer = random.randint(0, self.action_size - 1)
                print(f'----------随机生成的layernum----------{layer}')
                return layer
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
                layer = int(torch.argmax(q_values, dim=1).item())
        return layer

    def replay(self):
        if len(self.memory) == 0:
            return

        batch_size = min(len(self.memory), self.batch_size)
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: q_network 选择动作，target_network 评估该动作。
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update_target_network()
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.detach().item()

    def load_models(self, filename_q, filename_target):
        """
        Load the state dictionaries of q_network and target_network from files.
        """
        self.q_network.load_state_dict(torch.load(filename_q, weights_only=True))
        self.target_network.load_state_dict(torch.load(filename_target, weights_only=True))

    def save_models(self, filename_q, filename_target,filename_memory):
        """
        Save the state dictionaries of q_network and target_network.
        """
        torch.save(self.q_network.state_dict(), filename_q)
        torch.save(self.target_network.state_dict(), filename_target)
        with open(filename_memory,'wb')as f:
            pickle.dump(self.memory,f)
    def load_memory(self,filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.memory = pickle.load(f)
        else:
            print(f"Warning: File {filename} does not exist.")

    def get_model_parameters_count(self):
        """
        返回模型的总参数数量。
        """
        return sum(p.numel()*2 for p in self.q_network.parameters())

# 环境参数
# state_size = 4  # (t, q_A, q_B, r)
# action_size = 12  # 动作取值范围为 [1,12]，共 12个动作
# agent = DQNAgent(state_size:3, action_size)
#
# # 模拟训练
# num_episodes = 1000
# for e in range(num_episodes):
#     state = np.random.rand(state_size)  # 随机初始化状态
#     total_reward = 0
#     done = False
#
#     while not done:
#         action = agent.act(state)
#         action_value = action + 1  # 将动作值转换到 [1,12] 范围内
#
#         # 这里使用随机奖励和下一个状态进行示例，实际应根据环境计算
#         reward = random.random()
#         next_state = np.random.rand(state_size)
#
#         agent.remember(state, action, reward, next_state)
#         state = next_state
#         total_reward += reward
#
#         if done:
#             print(f"Episode: {e + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")
#             agent.update_target_network()
#
#     agent.replay()
