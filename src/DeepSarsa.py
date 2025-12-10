from src.QNetwork import QNetwork
import numpy as np
import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt


def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel("Episodio")
    plt.ylabel("Reward total")
    plt.title("Evolución del reward durante el entrenamiento")
    plt.savefig("reward_plot.png")
    plt.close()


class DeepSarsa:
    def __init__(self, learning_rate, discount_factor, environment):
        self.env = environment
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        state_dim, action_dim = self.env.get_dim()
        w, h = self.env.get_img_size()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNetwork(state_dim,action_dim,w,h).to(self.device)

        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(),lr=self.learning_rate)
        self.mse_loss_function = nn.MSELoss()
        self.SmoothL1_loss_function = nn.SmoothL1Loss()


    def epsilon_greedy_action(self, img, state, epsilon=0.9):
        if np.random.uniform(0, 1) < epsilon:
            idx, action = self.env.sample_action()
            return idx, action
        else:

            img_tensor = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0)
            state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)

            action_q = self.qnet(img_tensor, state_tensor)
            idx = action_q.detach().cpu().numpy().squeeze().argmax()

            return idx, self.env.get_action(idx)


    def update(self, img, state, next_img, next_state, action_idx, next_action_idx, reward, end):
        # Convertir inputs a tensores y enviar a device
        img_tensor = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0)
        next_img_tensor = torch.tensor(next_img, dtype=torch.float, device=self.device).unsqueeze(0)

        state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float, device=self.device)

        # Q-values actuales
        action_q = self.qnet(img_tensor, state_tensor).squeeze(0)
        next_action_q = self.qnet(next_img_tensor, next_state_tensor).squeeze(0)

        target_q = action_q.clone().detach()

        reward_t = torch.tensor(reward, dtype=action_q.dtype, device=self.device)

        ai = int(action_idx)
        nai = int(next_action_idx)

        if not end:
            next_q_val = next_action_q[nai].detach()
            target_q[ai] = reward_t + self.discount_factor * next_q_val
        else:
            target_q[ai] = reward_t


        loss = self.SmoothL1_loss_function(action_q[ai].unsqueeze(0), target_q[ai].unsqueeze(0))

        self.qnet_optim.zero_grad()
        loss.backward()
        self.qnet_optim.step()

    def train(self, n_episodes, n_steps, verbose=0):
        start_time = time.time()
        dts = []
        n_dts = 5
        rewards_per_episode = []
        epsilon = 1
        for eps in range(n_episodes):
            total_reward = 0

            st = time.time()
            state_img, state = self.env.reset()
            action_idx, action = self.epsilon_greedy_action(state_img, state, epsilon=epsilon)

            if verbose == 1:
                print(f"Episodio de prueba {eps} iniciado")
            for step in range(n_steps):
                next_state_img, next_state, reward, end = self.env.step(action, n_steps=n_steps)

                total_reward += reward

                next_action_idx, next_action = self.epsilon_greedy_action(next_state_img,next_state, epsilon=epsilon)

                self.update(state_img, state, next_state_img, next_state, action_idx, next_action_idx, reward, end)

                state_img, state, action = next_state_img, next_state, next_action

                if end:
                    break
                if verbose >= 2:
                    print(f"Episode {eps} Step {step}: Reward {reward:.4f}, Progress {state[1]*100:.3f}%, HP {state[4]:.3f}")

            et = time.time()
            dt = et - st
            dts.append(dt)
            if len(dts) > n_dts:
                dts.pop(0)
            avg_dt = sum(dts) / len(dts)
            remaining_time = ((n_episodes-eps-1)*avg_dt)/60

            if verbose == 1:
                print(f"Episodio de prueba {eps} terminado")
                print(f"Tiempo restante: {remaining_time:.0f} minutos")

            epsilon = max(0.2, epsilon * 0.995)

            rewards_per_episode.append(total_reward)
        end_time = time.time()
        print(f" Tiempo de ejecución: {(end_time-start_time)/60:.0f} minutos")

        plot_rewards(rewards_per_episode)

    def play(self, n_episodes=1, n_steps=2000, verbose=0):
        self.qnet.eval()
        for eps in range(n_episodes):
            total_reward = 0
            state_img, state = self.env.reset()

            for step in range(n_steps):
                with torch.no_grad():
                    action_idx, action = self.epsilon_greedy_action(state_img, state, epsilon=0)
                state_img, state, reward, end = self.env.step(action, n_steps=n_steps)
                print(action)

                total_reward += reward

                if end:
                    break
                if verbose == 1:
                    print(f"Reward {reward:.3f}, Progress {state[0]:.3f}, HP {state[1]:.3f}")
            print(f"Recompensa total del episodio: {total_reward}")

    def save(self, path="qnetwork.pth"):
        torch.save(self.qnet.state_dict(), path)
        print(f"Modelo guardado en {path}")

    def load(self, path="qnetwork.pth"):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.qnet.load_state_dict(checkpoint)
            print(f"Modelo cargado desde {path}")
            return True
        except FileNotFoundError:
            print(f"No se a encontrado un modelo para cargar en {path}")
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False


