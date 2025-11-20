from src.Circuit import Circuit
from src.Vehicle import Vehicle
from src.Visuals import Visual
from sympy import symbols
from sympy.functions import *
import numpy as np
import pygame

def main():
    t = symbols("t")
    circuit = Circuit(x_func=t, y_func=sin(t), var_symbol=t, variable_start=-10, variable_finish=10)
    vehicle = Vehicle(gear_ratio=[0, 3.5, 1.7, 0.25], friction_coef=1, max_velocity=10, max_brake=500, max_force=100)

    visual = Visual()
    visual.add_vehicle(vehicle)
    visual.set_circuit(circuit)
    x,y = circuit.get_start()
    print(x,y)
    vehicle.set_pos(x,y)
    vehicle.set_heading(circuit.angle_at_start())
    pygame.init()

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000  # segundos por frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                visual.fit_screen(width, height)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_0]:
            vehicle.change_gear(0)
        if keys[pygame.K_1]:
            vehicle.change_gear(1)
        if keys[pygame.K_2]:
            vehicle.change_gear(2)
        if keys[pygame.K_3]:
            vehicle.change_gear(3)
        if keys[pygame.K_SPACE]:
            vehicle.brake(dt)

        elif keys[pygame.K_LSHIFT]:
            vehicle.accelerate()
        else:
            vehicle.idle()
        if keys[pygame.K_a]:
            steer_angle = np.pi/1.5
            if keys[pygame.K_w]:
                steer_angle*=0.5
            vehicle.set_steer_angle(steer_angle*0.5)

        elif keys[pygame.K_d]:
            steer_angle = -np.pi/1.5
            if keys[pygame.K_w]:
                steer_angle*=0.5
            vehicle.set_steer_angle(steer_angle*0.5)

        vehicle.update(dt,circuit)
        visual.draw()

if __name__ == '__main__':
    main()


