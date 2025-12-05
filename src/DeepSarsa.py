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
        state_dim, gear_dim, brake_dim, accel_dim, steer_dim = self.env.get_dim()
        w, h = self.env.get_img_size()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.qnet = QNetwork(state_dim,gear_dim, brake_dim, accel_dim, steer_dim,w,h).to(self.device)

        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(),)
        self.mse_loss_function = nn.MSELoss()


    def epsilon_greedy_action(self, img, state, epsilon=0.9):
        if np.random.uniform(0, 1) < epsilon:
            return self.env.sample_action()
        else:

            img_tensor = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0)
            state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)

            gear_q, brake_q, accel_q, steer_q = self.qnet(img_tensor, state_tensor)
            gear_q = gear_q.detach().cpu().numpy().squeeze().argmax()
            brake_q = brake_q.detach().cpu().numpy().squeeze().argmax()
            accel_q = accel_q.detach().cpu().numpy().squeeze().argmax()
            steer_q = steer_q.detach().cpu().numpy().squeeze().argmax()

            action = [
                gear_q,
                brake_q,
                accel_q,
                steer_q
            ]

            return action


    def update(self, img, state, next_img, next_state, action, next_action, reward, end):
        # Convertir inputs a tensores y enviar a device
        img_tensor = torch.tensor(img, dtype=torch.float, device=self.device).unsqueeze(0)
        next_img_tensor = torch.tensor(next_img, dtype=torch.float, device=self.device).unsqueeze(0)

        state_tensor = torch.tensor([state], dtype=torch.float, device=self.device)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float, device=self.device)

        # Q-values actuales
        gear_q, brake_q, accel_q, steer_q = self.qnet(img_tensor, state_tensor)
        gear_q = gear_q.squeeze(0)
        brake_q = brake_q.squeeze(0)
        accel_q = accel_q.squeeze(0)
        steer_q = steer_q.squeeze(0)

        next_gear_q, next_brake_q, next_accel_q, next_steer_q = self.qnet(next_img_tensor, next_state_tensor)
        next_gear_q = next_gear_q.squeeze(0)
        next_brake_q = next_brake_q.squeeze(0)
        next_accel_q = next_accel_q.squeeze(0)
        next_steer_q = next_steer_q.squeeze(0)

        target_gear = gear_q.clone()
        target_brake = brake_q.clone()
        target_accel = accel_q.clone()
        target_steer = steer_q.clone()

        if not end:
            target_gear[int(action[0])] = reward + self.discount_factor * next_gear_q[int(next_action[0])]
            target_brake[int(action[1])] = reward + self.discount_factor * next_brake_q[next_action[1]]
            target_accel[int(action[2])] = reward + self.discount_factor * next_accel_q[next_action[2]]
            target_steer[int(action[3])] = reward + self.discount_factor * next_steer_q[int(next_action[3])]
        else:
            target_gear[int(action[0])] = reward
            target_brake[int(action[1])] = reward
            target_accel[int(action[2])] = reward
            target_steer[int(action[3])] = reward

        loss = (self.mse_loss_function(gear_q, target_gear) +
                self.mse_loss_function(brake_q, target_brake) +
                self.mse_loss_function(accel_q, target_accel) +
                self.mse_loss_function(steer_q, target_steer))

        self.qnet_optim.zero_grad()
        loss.backward()
        self.qnet_optim.step()

    def train(self, n_episodes, n_steps, render=False, verbose=False):
        epsilon = 1
        for eps in range(n_episodes):
            state_img, state = self.env.reset()
            action = self.epsilon_greedy_action(state_img, state, epsilon=epsilon)

            for step in range(n_steps):
                next_state_img, next_state, reward, end = self.env.step(action, render=render)
                next_action = self.epsilon_greedy_action(next_state_img,next_state, epsilon=epsilon)

                self.update(state_img, state, next_state_img, next_state, action, next_action, reward, end)

                state_img, state, action = next_state_img, next_state, next_action

                if end:
                    break
                if verbose:
                    print(f"Step {step}: Reward {reward:.3f}, Progress {state[0]:.3f}, HP {state[1]:.3f}")
            if verbose:
                print("Episodio de prueba terminado")

            if epsilon > 0.2:
                epsilon *= 0.995

            if epsilon <= 0.2:
                epsilon = 0.2

    def play(self, n_episodes=1, render=True, verbose=False):
        self.qnet.eval()

        for eps in range(n_episodes):
            total_reward = 0
            state_img, state = self.env.reset()
            end = False

            while not end:

                action = self.epsilon_greedy_action(state_img, state, epsilon=0)
                state_img, state, reward, end = self.env.step(action, render=render)

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
