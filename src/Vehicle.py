import numpy as np
import pygame.draw

from src.Circuit import Circuit


class Vehicle:
    def __init__(self, gear_ratios, weight=100, friction_coef=0.1, max_force=1, max_brake=1, max_velocity=100, max_hp=30, steer_reposition=0.05, lateral_grip=8.0, width=0.3, start_x=0, start_y=0):
        self.scale = 1
        self.lateral_grip = lateral_grip
        self.steer_reposition = steer_reposition
        self.weight = weight
        self.friction_coef = friction_coef
        self.max_force = max_force
        self.max_brake = max_brake
        self.max_velocity = max_velocity
        self.n_gears = len(gear_ratios)
        self.gear_ratios = gear_ratios
        self.max_hp = max_hp

        self.hp = max_hp
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

    # SETTERS
    # -----------
    def set_pos(self, x, y):
        self.x = x
        self.y = y
    def set_scale(self, scale):
        self.scale = scale
    def set_heading(self, angle):
        self.heading = angle
    def set_steer_angle(self, steer_angle):
        self.steer_angle = steer_angle
    # -----------

    def change_gear(self, gear_n):
        self.gear_n = max(0, min(self.n_gears - 1, gear_n))

    def damage_vehicle(self):
        if self.hp >=1:
            self.hp -=1
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

    def force_stop(self):
        self.current_x_velocity = 0
        self.current_y_velocity = 0

    def bounce_out(self, circuit):
        t, x, y = circuit.nearest_curve_point(px=self.x, py=self.y)
        self.set_pos((self.x+x)/2, (self.y+y)/2)

    def accelerate(self):
        if self.hp <= 0:
            self.current_x_force = 0
            self.current_y_force = 0
            return

        ratio = self.gear_ratios[self.gear_n]
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

        if dot > 1 - 1e-6:
            return True
        return False

    def check_outside(self, circuit:Circuit):
        t, x, y = circuit.nearest_curve_point(self.x, self.y)
        x0, y0 = circuit.get_start()
        xf, yf = circuit.get_finish()

        if x == x0 and y == y0:
            self.idle()
            self.force_stop()
            self.set_pos(x0, y0)
            self.set_heading(circuit.get_angle_start())
        elif x == xf and y == yf:
            self.idle()
            self.force_stop()
            self.set_pos(xf, yf)

    def update(self, dt, circuit=None):
        self.calculate_speed(dt)

        L = 2.0
        v = np.hypot(self.current_x_velocity, self.current_y_velocity)
        self.heading += (v / L) * np.tan(self.steer_angle) * dt

        self.heading = (self.heading + np.pi) % (2*np.pi) - np.pi

        self.calculate_position(dt)

        self.steer_angle *= self.steer_reposition**dt
        if circuit:
            self.check_outside(circuit)
            if self.check_collision(circuit):
                self.idle()
                self.force_stop()
                self.damage_vehicle()
                self.bounce_out(circuit)

    # GETTERS
    # -----------
    def get_pos(self):
        return self.x, self.y
    def get_max_brake(self):
        return self.max_brake
    def get_relative_velocity(self):
        return self.current_x_velocity/self.max_velocity, self.current_y_velocity/self.max_velocity
    def get_heading(self):
        return self.heading
    def get_width(self):
        return self.width
    def get_weight(self):
        return self.weight
    def get_relative_force(self):
        return self.current_x_force/self.max_force, self.current_y_force/self.max_force
    def get_steering_angle(self):
        return self.steer_angle
    def get_relative_hp(self):
        return self.hp/self.max_hp
    def get_gear_ratios(self):
        return self.gear_ratios
    def get_relative_gear(self):
        return self.gear_n/self.n_gears
    # -----------

    # VISUAL
    # -----------
    def draw(self, screen, circuit=None):

        if circuit:
            center_x, center_y = circuit.world_to_screen(self.x, self.y)
            line_length = 2*self.width*self.scale
            end_x, end_y = center_x + line_length * np.cos(-self.heading-self.steer_angle), center_y + line_length * np.sin(-self.heading-self.steer_angle)

            pygame.draw.circle(screen, (0, 0, 255), (center_x, center_y), self.width * self.scale)
            pygame.draw.line(screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), int(0.15*self.scale))
        else:
            center_x, center_y = self.x, self.y
            line_length = 2*self.width * self.scale
            end_x, end_y = center_x + line_length * np.cos(-self.heading-self.steer_angle), center_y + line_length * np.sin(-self.heading-self.steer_angle)

            pygame.draw.circle(screen, (0,0,255),(self.x, self.y),self.width*self.scale)
            pygame.draw.line(screen, (255, 0, 0), (center_x, center_y), (end_x, end_y), int(0.15*self.scale))

    #-----------
