from numpy import random

from src.Circuit import Circuit
from src.Vehicle import Vehicle
from src.Visuals import Visual
from sympy import symbols
from sympy.functions import *
import numpy as np
import itertools


class Environment:

    def __init__(self, render=True):
        self.n_checkpoints = 50
        self.check_point_perc = (100 / self.n_checkpoints)
        self.check_point_reward = 500000/self.n_checkpoints
        self.crash_penalty = -500000
        self.prev_rel_hp = None
        self.rel_hp = None
        self.start_y = None
        self.start_x = None
        self.progress = 0
        self.prev_progress = 0
        self.render = render
        self.visual = None
        self.vehicle = None
        self.circuit = None
        self.check_points = None

        max_steer_angle = 0.5*np.pi/1.5
        self.steer_angles = [-max_steer_angle, 0, max_steer_angle]
        self.action_space = {
            "gear_change": [0, 1, 2, 3],
            "pedal": [0, 1, 2],
            "steer": [0, 1, 2]
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
        self.progress = 0
        self.prev_progress = 0
        self.rel_hp = 1
        self.prev_rel_hp = 1

        self.retro = 0
        self.quieto = 0
        self.frames = 0


        self.check_points = np.zeros(self.n_checkpoints)
        t = symbols("t")
        self.circuit = Circuit(x_func=t, y_func=sin(t), var_symbol=t, variable_start=-10, variable_finish=10)
        self.vehicle = Vehicle(gear_ratios=[3.5, 1.7, 0.25, -0.25], friction_coef=1, max_velocity=10, max_brake=500,
                               max_force=100)
        x, y = self.circuit.get_start()
        self.start_x, self.start_y = x, y
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
        return 180, 180

    def get_reward_progress(self, action, n_steps=2000):
        vx, vy = self.vehicle.get_relative_velocity()
        v = np.hypot(vx, vy)

        reward = -5 # Inicia con penalización por paso

        delta_progress = (self.progress - self.prev_progress)
        delta_hp = (self.rel_hp - self.prev_rel_hp)

        reward += delta_progress * 50000

        checkpoint_idx = int(self.progress * 100 / self.check_point_perc)
        checkpoint_idx = min(checkpoint_idx, self.n_checkpoints - 1)

        if checkpoint_idx > 0 and int(self.progress*100) % self.check_point_perc == 0 and self.check_points[checkpoint_idx] == 0:
            self.check_points[checkpoint_idx] = 1
            reward += self.check_point_reward


        if delta_progress < -0.0001:
            reward -= 100

        if abs(delta_progress) < 0.0001 and v == 0: # Penalización por quedarse quieto
            if delta_hp == 0:
                reward += self.crash_penalty / n_steps

        return reward

    def get_state(self):
        state = []

        x, y = self.vehicle.get_pos()
        # Circuito
        state.append(self.prev_progress)
        self.prev_progress = self.progress
        t, self.progress = self.circuit.get_progress(x, y)
        state.append(self.progress)
        state.append(self.circuit.angle_of_curve(t))
        # Vehicle
        state.append(self.prev_rel_hp)
        self.prev_rel_hp = self.rel_hp
        self.rel_hp = self.vehicle.get_relative_hp()
        state.append(self.rel_hp)
        state.append(self.vehicle.get_weight())
        state.append(self.vehicle.get_width()/self.circuit.get_width())

        state.append(self.vehicle.get_relative_gear())

        ratios = self.vehicle.get_gear_ratios()
        max_ratio = max(ratios) if max(ratios) != 0 else 1
        ratios_norm = [r / max_ratio for r in ratios]
        state.extend(ratios_norm)

        heading_wrapped = self.vehicle.get_heading() % (2*np.pi)
        heading_norm = heading_wrapped / (2*np.pi)
        state.append(heading_norm)

        state.append(self.vehicle.get_steering_angle()/(np.pi/1.5))

        vx, vy = self.vehicle.get_relative_velocity()
        state.append(vx)
        state.append(vy)

        fx, fy = self.vehicle.get_relative_force()
        state.append(fx)
        state.append(fy)

        tem_x, tem_y = self.vehicle.get_pos()
        tem_x, tem_y = self.circuit.world_to_screen(tem_x, tem_y)
        tem_scale = self.circuit.get_scale() * 0.6

        self.visual.draw()
        img = self.visual.take_circular_ss(tem_x - tem_scale * 4, tem_y - tem_scale * 4, tem_scale * 8, tem_scale * 8)

        return img, state

    def step(self, action, n_steps=2000, dt=0.01):
        self.circuit.cicle_lights_state()
        self.vehicle.change_gear(action[0])
        if action[1] == 0:
            self.vehicle.idle()
        elif action[1] == 1:
            self.vehicle.brake(dt)
        elif action[1] == 2:
            self.vehicle.accelerate()

        self.vehicle.set_steer_angle(self.steer_angles[action[2]])

        self.vehicle.update(dt)
        self.vehicle.check_outside(self.circuit)

        self.vehicle.check_collision(self.circuit)

        image, state = self.get_state()
        reward = self.get_reward_progress(action, n_steps=n_steps)

        is_finish = self.progress >= 1
        is_crashed = self.rel_hp <= 0


        if is_crashed:
            reward += self.crash_penalty

        if is_finish:
            reward += 0

        done = is_finish or is_crashed

        return image, state, reward, done