from tabnanny import check

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNetwork(state_dim,action_dim,w,h).to(self.device)

        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(),lr=self.learning_rate)
        self.mse_loss_function = nn.MSELoss()


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

        target_q = action_q.clone()

        if not end:
            target_q[int(action_idx)] = reward + self.discount_factor * next_action_q[int(next_action_idx)]

        else:
            target_q[int(action_idx)] = reward


        loss = (self.mse_loss_function(action_q, target_q))

        self.qnet_optim.zero_grad()
        loss.backward()
        self.qnet_optim.step()

    def train(self, n_episodes, n_steps, verbose=0):
        epsilon = 1
        for eps in range(n_episodes):
            state_img, state = self.env.reset()
            action_idx, action = self.epsilon_greedy_action(state_img, state, epsilon=epsilon)

            if verbose == 1:
                print(f"Episodio de prueba {eps} iniciado")
            for step in range(n_steps):
                next_state_img, next_state, reward, end = self.env.step(action)
                next_action_idx, next_action = self.epsilon_greedy_action(next_state_img,next_state, epsilon=epsilon)

                self.update(state_img, state, next_state_img, next_state, action_idx, next_action_idx, reward, end)

                state_img, state, action = next_state_img, next_state, next_action

                if end:
                    break
                if verbose >= 2:
                    print(f"Episode {eps} Step {step}: Reward {reward:.3f}, Progress {state[0]:.3f}, HP {state[1]:.3f}")

            if verbose == 1:
                print(f"Episodio de prueba {eps} terminado")

            if epsilon > 0.2:
                epsilon *= 0.995

            if epsilon <= 0.2:
                epsilon = 0.2

    def play(self, n_episodes=1, verbose=False):
        self.qnet.eval()

        for eps in range(n_episodes):
            total_reward = 0
            state_img, state = self.env.reset()
            end = False

            while not end:

                action = self.epsilon_greedy_action(state_img, state, epsilon=0)
                state_img, state, reward, end = self.env.step(action)

                total_reward += reward

                if end:
                    break
                if verbose:
                    print(f"Reward {reward:.3f}, Progress {state[0]:.3f}, HP {state[1]:.3f}")
            print(f"Recompensa total del episodio: {total_reward}")

    def save(self, path="qnetwork.pth"):
        torch.save(self.qnet.state_dict(), path)
        print(f"Modelo guardado en {path}")

    def load(self, path="qnetwork.pth"):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.qnet.load_state_dict(checkpoint)
            self.qnet.eval()  # modo inferencia
            print(f"Modelo cargado desde {path}")
            return True
        except FileNotFoundError:
            print(f"No se a encontrado un modelo para cargar en {path}")
            return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False
