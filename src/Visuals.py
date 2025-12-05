from src.Circuit import Circuit
import pygame
import numpy as np

from src.Vehicle import Vehicle


class Visual:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Driving Simulation 2D")
        info = pygame.display.Info()
        self.screen_width, self.screen_height = info.current_w, info.current_h
        self.circuit = None
        self.screen = None
        self.scale = 1
        self.vehicles = []
        self.start()

    def set_circuit(self, circuit:Circuit):
        self.circuit = circuit
        self.circuit.fit(self.screen.get_width(), self.screen.get_height())
        self.scale = self.circuit.get_scale()

        for v in self.vehicles:
            v.set_scale(self.scale)

    def fit_screen(self, width, height):
        self.circuit.fit(width, height)
        self.scale = self.circuit.get_scale()
        self.screen_width = width
        self.screen_height = height
        for v in self.vehicles:
            v.set_scale(self.scale)

    def add_vehicle(self, vehicle:Vehicle):
        vehicle.set_scale(self.scale)
        self.vehicles.append(vehicle)

    def start(self):
        self.screen = pygame.display.set_mode((800, 600))#, pygame.RESIZABLE)

    def get_subsurface(self, x, y, width, height):
        new_x, new_y = x, y
        new_width, new_height = width, height
        resize_x, resize_y = 0, 0

        if x < 0:
            new_x = 0
            new_width = width - abs(x)
            resize_x = abs(x)
        if y < 0:
            new_y = 0
            new_height = height - abs(y)
            resize_y = abs(y)

        if x+width > self.screen.get_width():
            new_width = self.screen.get_width() - x
        if y+height > self.screen.get_height():
            new_height = self.screen.get_height() - y

        resize_surface = self.screen.subsurface(pygame.Rect(new_x, new_y, new_width, new_height))

        sub_surface = pygame.Surface((width, height))
        sub_surface.fill((0,0,0))
        sub_surface.blit(resize_surface, (resize_x, resize_y))

        return sub_surface

    def take_circular_ss(self, x, y, width, height):
        sub_surface = self.get_subsurface(x, y, width, height)

        # Máscara circular
        mask = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.circle(mask, (255, 255, 255), (width // 2, height // 2), min(width, height) // 2)

        # Fondo violeta
        circular_ss = pygame.Surface((width, height))
        circular_ss.fill((255, 0, 255))

        # Aplicar Máscara
        temp = pygame.Surface((width, height), pygame.SRCALPHA)
        temp.blit(sub_surface, (0, 0))
        temp.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        circular_ss.blit(temp,(0,0))
        arr = pygame.surfarray.array3d(circular_ss)
        arr = np.transpose(arr, (2, 1, 0))
        arr = arr.astype(np.float32) / 255.0

        #pygame.image.save(circular_ss, "screenshot_partial.png")
        return arr

    def update(self):
        pass

    def get_screen(self):
        return self.screen

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.circuit.draw(self.screen)
        for v in self.vehicles:
            v.draw(self.screen,self.circuit)
        pygame.display.flip()
