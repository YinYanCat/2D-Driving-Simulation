from numpy import random

from src.Circuit import Circuit
from src.Vehicle import Vehicle
from src.Visuals import Visual
from sympy import symbols
from sympy.functions import *
import numpy as np
import itertools


class Environment:

    def __init__(self, max_steps, render=True):
        # Parámetros del circuito
        self.start = -10
        self.end = 10
        self.n_events = 20*abs(self.end-self.start)
        self.max_reward = 1000
        self.check_point_perc = (100 / (self.n_events + 1))
        self.check_point_reward = self.max_reward/self.n_events

        # Variables de estado del episodio
        self.max_steps = max_steps
        self.current_step = 0
        self.progress = None
        self.prev_progress = None
        self.rel_hp = None
        self.prev_rel_hp = None
        self.rel_hp_last_checkpoint = None
        self.combo = None
        self.steps_since_checkpoint = None

        self.render = render

        # Objetos del entorno
        self.visual = None
        self.vehicle = None
        self.circuit = None
        self.check_points = None

        # Espacio de acciones
        max_steer_angle = 0.5*np.pi/1.5
        self.steer_angles = [-max_steer_angle, 0, max_steer_angle]
        self.action_space = {
            "gear_change": [0, 1, 2, 3],
            "pedal": [0, 1, 2],
            "steer": [0, 1, 2, 3]
        }

        self.action_table = list(itertools.product(self.action_space["gear_change"],
                                                    self.action_space["pedal"],
                                                    self.action_space["steer"]))

        self.n_actions = len(self.action_table)

    def sample_action(self):
        idx = random.randint(self.n_actions)
        action = self.action_table[idx]

        return idx, action

    def get_action(self, idx):
        return self.action_table[idx]

    def reset(self):
        self.current_step = 0

        self.progress = 0.0
        self.prev_progress = 0.0
        self.rel_hp = 1.0
        self.prev_rel_hp = 1.0
        self.rel_hp_last_checkpoint = 1.0
        self.combo = 0
        self.steps_since_checkpoint = 0

        self.check_points = np.zeros(self.n_events, dtype=int)

        t = symbols("t")
        self.circuit = Circuit(x_func=t, y_func=sin(t), var_symbol=t, variable_start=self.start, variable_finish=self.end)
        self.vehicle = Vehicle(gear_ratios=[3.5, 1.7, 0.25, -0.25], friction_coef=1, max_velocity=10, max_brake=500,
                               max_force=100, max_hp=self.n_events)

        x, y = self.circuit.get_start()
        self.vehicle.set_pos(x, y)
        self.vehicle.set_heading(self.circuit.get_angle_start())

        self.visual = Visual(render=self.render)
        self.visual.add_vehicle(self.vehicle)
        self.visual.set_circuit(self.circuit)

        image, state = self.get_state()

        return image, state

    def get_dim(self):
        state_img, state = self.get_state()
        state_dim = len(state)
        action_dim = self.n_actions
        return state_dim, action_dim

    def get_img_size(self):
        return 84, 84

    def get_reward_progress(self):

        reward = -0.2 # Inicia con penalización por paso

        checkpoint_idx = int(self.progress * 100 / self.check_point_perc) - 1
        checkpoint_idx = min(checkpoint_idx, self.n_events - 1)

        if checkpoint_idx >= 0:
            # Verificar si llegó a un nuevo checkpoint
            if self.check_points[checkpoint_idx] == 0:
                # Marcar todos los checkpoints intermedios alcanzados
                for i in range(checkpoint_idx+1):
                    if self.check_points[i] == 0:
                        self.check_points[i] = 1

                        combo_multiplier = 1.0 + (self.combo * 1.0 / self.n_events)
                        reward += self.check_point_reward * combo_multiplier

                        if self.rel_hp == self.rel_hp_last_checkpoint:
                            self.combo += 1
                        else:
                            self.combo = 0

                        self.rel_hp_last_checkpoint = self.rel_hp
                        self.steps_since_checkpoint = 0

        self.steps_since_checkpoint += 1
        if self.progress >= 1.0:
            completion_bonus = self.max_reward * 0.1 * self.rel_hp
            reward += completion_bonus

        return reward

    def get_state(self):
        state = []

        x, y = self.vehicle.get_pos()
        # Circuito

        #state.append(self.progress)
        t, self.progress = self.circuit.get_progress(x, y)
        state.append(self.progress)

        # Vehicle
        self.prev_rel_hp = self.rel_hp
        state.append(self.prev_rel_hp)
        self.rel_hp = self.vehicle.get_relative_hp()
        state.append(self.rel_hp)

        ratios = self.vehicle.get_gear_ratios()
        max_ratio = max(ratios) if max(ratios) != 0 else 1
        ratios_norm = [r / max_ratio for r in ratios]
        state.extend(ratios_norm)
        state.append(self.vehicle.get_relative_gear())

        heading = self.vehicle.get_heading()
        state.append(heading)

        vx, vy = self.vehicle.get_relative_velocity()
        state.append(vx)
        state.append(vy)

        fx, fy = self.vehicle.get_relative_force()
        state.append(fx)
        state.append(fy)

        # tiempo

        curr_rel_steps = self.current_step / self.max_steps
        state.append(curr_rel_steps)

        # combo

        combo_norm = self.combo / self.n_events  # Normalizar combo [0-1]
        state.append(combo_norm)


        # Image
        tem_x, tem_y = self.vehicle.get_pos()
        tem_x, tem_y = self.circuit.world_to_screen(tem_x, tem_y)
        tem_scale = self.circuit.get_scale() * 0.6

        self.visual.draw()
        img = self.visual.take_circular_ss(tem_x - tem_scale * 4, tem_y - tem_scale * 4, tem_scale * 8, tem_scale * 8)

        return img, state

    def step(self, action, dt=0.1):
        self.current_step += 1
        self.circuit.cicle_lights_state()
        self.vehicle.change_gear(action[0])
        if action[1] == 0:
            self.vehicle.idle()
        elif action[1] == 1:
            self.vehicle.brake(dt)
        elif action[1] == 2:
            self.vehicle.accelerate()

        if 3 > action[2] >= 0:
            self.vehicle.set_steer_angle(self.steer_angles[action[2]])

        self.vehicle.update(dt, circuit=self.circuit)


        image, state = self.get_state()
        reward = self.get_reward_progress()

        is_finish = self.progress >= 1.0
        is_crashed = self.rel_hp <= 0

        done = is_finish or is_crashed

        return image, state, reward, done