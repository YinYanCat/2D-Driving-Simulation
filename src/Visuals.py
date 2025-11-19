from src.Circuit import Circuit
import pygame

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

    def add_vehicle(self, vehicle:Vehicle):
        vehicle.set_scale(self.scale)
        self.vehicles.append(vehicle)

    def start(self):
        self.screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)

    def update(self):
        pass

    def draw(self):
        #self.screen.fill((255, 255, 255))
        self.circuit.draw(self.screen)
        for v in self.vehicles:
            v.draw(self.screen,self.circuit)
        pygame.display.flip()
