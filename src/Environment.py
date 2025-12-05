from src.Circuit import Circuit
from src.Vehicle import Vehicle
from src.Visuals import Visual
from sympy import symbols
from sympy.functions import *
import numpy as np
import pygame


class Environment:

    def __init__(self):
        self.progress = 0
        self.prev_progress = 0
        self.visual = None
        self.vehicle = None
        self.circuit = None

    def reset(self):
        self.progress = 0
        self.prev_progress = 0
        t = symbols("t")
        self.circuit = Circuit(x_func=t, y_func=sin(t), var_symbol=t, variable_start=-10, variable_finish=10)
        self.vehicle = Vehicle(gear_ratios=[0, 3.5, 1.7, 0.25, -1], friction_coef=1, max_velocity=10, max_brake=500,
                               max_force=100)
        x, y = self.circuit.get_start()
        self.vehicle.set_pos(x, y)
        self.vehicle.set_heading(self.circuit.angle_at_start())

        self.visual = Visual()
        self.visual.add_vehicle(self.vehicle)
        self.visual.set_circuit(self.circuit)

        image, state = self.get_state()

        return image, state

    def step(self, action, dt=0.1, render=False):
        self.vehicle.change_gear(action[0])
        if action[1]:
            self.vehicle.brake(dt)
        elif action[2]:
            self.vehicle.accelerate()
        else:
            self.vehicle.idle()

        max_steer_angle = np.pi/1.5
        self.vehicle.set_steer_angle(max_steer_angle*action[3])

        self.vehicle.update(dt)
        if self.vehicle.check_collision(self.circuit):
            self.vehicle.idle()
            self.vehicle.force_stop()
            self.vehicle.damage_vehicle()
            self.vehicle.bounce_out(self.circuit)

        if render:
            self.visual.draw()

        image, state = self.get_state()
        reward = self.get_reward()
        done = self.progress >= 1 or self.vehicle.get_relative_hp() <= 0
        return image, state, reward, done

    def get_reward(self):
        reward = (self.progress - self.prev_progress) * 10 + self.vehicle.get_relative_hp()
        self.prev_progress = self.progress
        return reward

    def get_state(self):
        state = []

        x, y = self.vehicle.get_pos()
        # Circuito
        self.progress = self.circuit.get_progress(x, y)
        state.append(self.progress)
        # Vehicle
        state.append(self.vehicle.get_relative_hp())
        state.append(self.vehicle.get_weight())
        state.append(self.vehicle.get_width())

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
        img = self.visual.take_circular_ss(tem_x - tem_scale * 4, tem_y - tem_scale * 4, tem_scale * 8, tem_scale * 8)

        return img, state

