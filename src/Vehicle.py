import numpy as np
import pygame.draw

from src.Circuit import Circuit


class Vehicle:
    def __init__(self, gear_ratio, weight=100, friction_coef=0.1, max_force=1, max_brake=1, max_velocity=100, max_hp=3, max_resources=100, steer_reposition=0.05, lateral_grip=8.0, width=0.3, start_x=0, start_y=0):
        self.scale = 1
        self.lateral_grip = lateral_grip
        self.steer_reposition = steer_reposition
        self.weight = weight
        self.friction_coef = friction_coef
        self.max_force = max_force
        self.max_brake = max_brake
        self.max_velocity = max_velocity
        self.n_changes = len(gear_ratio)
        self.gear_ratio = gear_ratio
        self.max_hp = max_hp
        self.max_resources = max_resources

        self.hp = max_hp
        self.resources = max_resources
        self.steer_angle = 0
        self.gear_n = 0

        self.heading = 0

        self.current_x_force = 0
        self.current_y_force = 0

        self.current_x_velocity = 0
        self.current_y_velocity = 0


        self.width = width
        self.x = start_x
        self.y = start_y

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def set_heading(self, angle):
        self.heading = angle

    def set_steering(self, angle):
        self.steer_angle = angle

    def change_gear(self, gear_n):
        self.gear_n = max(0, min(self.n_changes - 1, gear_n))

    def damage_vehicle(self):
        if self.hp >=1:
            self.hp -=1
            print(f"lives: {self.hp}")
        else:
            self.hp = 0

    def brake(self, dt):
        v = np.array([self.current_x_velocity, self.current_y_velocity])
        speed = np.linalg.norm(v)

        if speed < 0.01:
            self.current_x_velocity = 0
            self.current_y_velocity = 0
            self.current_x_force = 0
            self.current_y_force = 0
            return

        brake_dir = -v / speed

        fx = brake_dir[0] * self.max_brake
        fy = brake_dir[1] * self.max_brake

        decel = self.max_brake / self.weight
        speed_loss = decel * dt

        if speed <= speed_loss:
            self.current_x_velocity = 0
            self.current_y_velocity = 0
            self.current_x_force = 0
            self.current_y_force = 0
            return

        self.current_x_force = fx
        self.current_y_force = fy

    def idle(self):
        self.current_x_force = 0
        self.current_y_force = 0

    def accelerate(self):
        if self.hp <= 0:
            self.current_x_force = 0
            self.current_y_force = 0
            return

        ratio = self.gear_ratio[self.gear_n]
        force = self.max_force * ratio

        dir_x = np.cos(self.heading)
        dir_y = np.sin(self.heading)

        self.current_x_force = dir_x * force
        self.current_y_force = dir_y * force

    def calculate_force(self):
        friction_x = -self.friction_coef * self.current_x_velocity
        friction_y = -self.friction_coef * self.current_y_velocity

        v_lat = (self.current_x_velocity * -np.sin(self.heading) +
                 self.current_y_velocity * np.cos(self.heading))

        f_lat = -self.lateral_grip * v_lat * self.weight

        f_lat_x = f_lat * (-np.sin(self.heading))
        f_lat_y = f_lat * (np.cos(self.heading))

        # total
        fx = self.current_x_force + friction_x + f_lat_x
        fy = self.current_y_force + friction_y + f_lat_y

        return fx, fy

    def calculate_speed(self, dt):
        fx, fy = self.calculate_force()

        ax = fx / self.weight
        ay = fy / self.weight

        self.current_x_velocity += ax * dt
        self.current_y_velocity += ay * dt

        speed = (self.current_x_velocity**2 + self.current_y_velocity**2)**0.5

        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.current_x_velocity *= scale
            self.current_y_velocity *= scale

    def calculate_position(self, dt):
        self.x += self.current_x_velocity * dt
        self.y += self.current_y_velocity * dt

    def check_collision(self, circuit:Circuit):
        t, x, y = circuit.nearest_curve_point(self.x, self.y)
        v_wall1, v_wall2 = circuit.off_positions(t)
        v_dist1 = np.array(v_wall1) - np.array([self.x, self.y])
        v_dist2 = np.array(v_wall2) - np.array([self.x, self.y])
        norm1 = np.linalg.norm(v_dist1)
        norm2 = np.linalg.norm(v_dist2)

        u1 = v_dist1 / norm1
        u2 = v_dist2 / norm2

        v_dist1_shrink = v_dist1 - self.width/2 * u1
        v_dist2_shrink = v_dist2 - self.width/2 * u2

        u1_shrink = v_dist1_shrink / np.linalg.norm(v_dist1_shrink)
        u2_shrink = v_dist2_shrink / np.linalg.norm(v_dist2_shrink)

        dot = np.dot(u1_shrink, u2_shrink)
        return dot > 1 - 1e-6


    def set_steer_angle(self, steer_angle):
        self.steer_angle = steer_angle


    def update(self, dt, circuit=None):
        self.calculate_speed(dt)

        L = 2.0
        v = np.hypot(self.current_x_velocity, self.current_y_velocity)
        self.heading += (v / L) * np.tan(self.steer_angle) * dt

        self.calculate_position(dt)

        self.steer_angle *= self.steer_reposition**dt

        if circuit is not None:
            if self.check_collision(circuit):
                self.current_x_velocity = 0
                self.current_y_velocity = 0
                self.current_x_force = 0
                self.current_y_force = 0
                self.damage_vehicle()
                t, x, y = circuit.nearest_curve_point(px=self.x, py=self.y)
                self.set_heading(circuit.angle_of_curve(t))
                self.set_pos(x,y)

    def get_pos(self):
        return self.x, self.y

    def set_scale(self, scale):
        self.scale = scale

    def draw(self, screen, circuit=None):
        if circuit:
            pygame.draw.circle(screen, (255, 0, 0), circuit.world_to_screen(self.x, self.y), 0.3 * self.scale)
        else:
            pygame.draw.circle(screen, (255,0,0),(self.x, self.y),self.width*self.scale)
