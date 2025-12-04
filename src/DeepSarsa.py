from src.QNetwork import QNetwork
import numpy as np
import torch
import torch.nn as nn

class DeepSarsa:
    def __init__(self, learning_rate, discount_factor, environment):
        self.env = environment
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        state_dim, action_dim = self.env.get_dim()
        w, h = self.env.get_img_size()
        self.qnet = QNetwork(state_dim,action_dim,w,h)
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(),)
        self.mse_loss_function = nn.MSELoss()

    def epsilon_greedy_action(self, img, state, epsilon=0.5):
        if np.random.uniform(0, 1) < epsilon:
                return self.env.action_space.sample()
        else:
            network_output_to_numpy = self.qnet(img, state).data.numpy()
            return np.argmax(network_output_to_numpy)

    def update(self, img, state, next_img, next_state, action, next_action, reward, end):
        action_tensor = torch.tensor([action], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float)
        end_tensor = torch.tensor([end], dtype=torch.float)
        next_action_tensor = torch.tensor([next_action], dtype=torch.long)

        q_values = self.qnet(img, state)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        next_q_values = self.qnet(next_img, next_state)
        next_q_value = next_q_values.gather(1, next_action_tensor.unsqueeze(1)).squeeze(1)

        target = reward_tensor + (1-end_tensor) * self.discount_factor * next_q_value
        q_network_loss = self.mse_loss_function(q_value, target.detach())

        self.qnet_optim.zero_grad()
        q_network_loss.backward()
        self.qnet_optim.step()

    def train(self, n_episodes, n_steps):
        for eps in range(n_episodes):
            state_img, state = self.env.reset()
            action = self.epsilon_greedy_action(state_img, state)

            for step in range(n_steps):
                next_state_img, next_state, reward, end = self.env.step(action)
                next_action = self.epsilon_greedy_action(next_state_img,next_state)

                self.update(state_img, state, next_state_img, next_state, action, next_action, reward, end)

                state_img, state = next_state_img, next_state
                action = next_action
                if end:
                    break



