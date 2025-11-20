from sympy import diff, sqrt, symbols
from sympy import lambdify

import numpy as np
import pygame

class Circuit:
    def __init__(self, x_func, y_func, var_symbol, variable_start=0, variable_finish=10, draw_points=500, circuit_width=1):
        self.function = [x_func,y_func]
        self.var = var_symbol
        self.variable_start = variable_start
        self.variable_finish = variable_finish
        self.derivate = [diff(x_func,self.var), diff(y_func, self.var)]
        self.tangent_norm = sqrt(self.derivate[0]**2 + self.derivate[1]**2)
        self.normal_vector = [-self.derivate[1]/self.tangent_norm, self.derivate[0]/self.tangent_norm]

        self.circuit_width = circuit_width

        self.x_off1 = x_func + circuit_width * self.normal_vector[0]
        self.y_off1 = y_func + circuit_width * self.normal_vector[1]
        self.x_off2 = x_func - circuit_width * self.normal_vector[0]
        self.y_off2 = y_func - circuit_width * self.normal_vector[1]

        ts = np.linspace(variable_start, variable_finish, draw_points)
        self.ts = ts
        self.x_vals = np.array([float(x_func.subs(var_symbol, t)) for t in ts])
        self.y_vals = np.array([float(y_func.subs(var_symbol, t)) for t in ts])
        self.x1 = np.array([float(self.x_off1.subs(var_symbol, t)) for t in ts])
        self.y1 = np.array([float(self.y_off1.subs(var_symbol, t)) for t in ts])
        self.x2 = np.array([float(self.x_off2.subs(var_symbol, t)) for t in ts])
        self.y2 = np.array([float(self.y_off2.subs(var_symbol, t)) for t in ts])

        self.scale = 1
        self.x_min = 0
        self.y_min = 0
        self.width = 1
        self.height = 1

    def get_width(self):
        return self.circuit_width

    def get_start(self):
        x0, y0 = float(self.function[0].subs(self.var,self.variable_start)), float(self.function[1].subs(self.var,self.variable_start))
        return x0, y0

    def angle_at_start(self):
        return self.angle_of_curve(self.variable_start)

    def fit(self, width, height, draw_points=500):

        self.width = width
        self.height = height

        # Rango total matemático del circuito + bordes
        xs = np.concatenate([self.x_vals, self.x1, self.x2])
        ys = np.concatenate([self.y_vals, self.y1, self.y2])

        self.x_min = xs.min()
        self.y_min = ys.min()

        range_x = xs.max() - xs.min()
        range_y = ys.max() - ys.min()

        # Escala uniforme
        self.scale = min(width / range_x, height / range_y)

    def nearest_curve_point(self, px, py, max_iter=20, tol=1e-6):
        """
        Computes the nearest point on the curve C(t)=(x(t),y(t))
        using Newton iteration on f(t) = (C(t)-P)·C'(t).

        px, py   : point in WORLD space
        returns  : (t, x(t), y(t))
        """

        # Pre-built lambdas for efficiency (build once)
        if not hasattr(self, "_proj_initialized"):
            t = self.var

            # first derivatives (you already computed these)
            dx = self.derivate[0]
            dy = self.derivate[1]

            # second derivatives
            ddx = dx.diff(t)
            ddy = dy.diff(t)

            # lambdify curve position, first & second derivatives
            self._Cx  = lambdify(t, self.function[0], "numpy")
            self._Cy  = lambdify(t, self.function[1], "numpy")
            self._Cdx = lambdify(t, dx, "numpy")
            self._Cdy = lambdify(t, dy, "numpy")
            self._Cddx = lambdify(t, ddx, "numpy")
            self._Cddy = lambdify(t, ddy, "numpy")

            self._proj_initialized = True

        # Initial guess: nearest t by roughly matching x(t)
        # This is NOT the final result, but a useful starting point.
        t0 = np.clip(px, self.variable_start, self.variable_finish)

        t_curr = t0

        for _ in range(max_iter):
            Cx = self._Cx(t_curr)
            Cy = self._Cy(t_curr)
            Cdx = self._Cdx(t_curr)
            Cdy = self._Cdy(t_curr)
            Cddx = self._Cddx(t_curr)
            Cddy = self._Cddy(t_curr)

            rx = Cx - px
            ry = Cy - py

            # f(t) = (C(t)-P)·C'(t)
            f = rx * Cdx + ry * Cdy

            # f'(t)
            fp = (Cdx*Cdx + Cdy*Cdy) + (rx * Cddx + ry * Cddy)

            if abs(fp) < 1e-12:
                break  # Avoid division by near-zero

            t_new = t_curr - f / fp

            # clamp to valid domain
            t_new = np.clip(t_new, self.variable_start, self.variable_finish)

            if abs(t_new - t_curr) < tol:
                t_curr = t_new
                break

            t_curr = t_new

        # Return nearest point
        return t_curr, float(self._Cx(t_curr)), float(self._Cy(t_curr))


    def angle_of_curve(self, t):
        dx_val = float(self.derivate[0].subs(self.var, t))
        dy_val = float(self.derivate[1].subs(self.var, t))
        return np.arctan2(dy_val,dx_val)

    def off_positions(self, t):
        x1 = float(self.x_off1.subs(self.var, t))
        y1 = float(self.y_off1.subs(self.var, t))
        x2 = float(self.x_off2.subs(self.var, t))
        y2 = float(self.y_off2.subs(self.var, t))

        return (x1, y1), (x2, y2)

    def world_to_screen(self, x, y):
        px = int((x - self.x_min) * self.scale)
        py = int(self.height - (y - self.y_min) * self.scale)
        return px, py

    def get_scale(self):
        return self.scale

    def draw(self, screen:pygame.surface):
        # Curva central
        pts = [self.world_to_screen(x, y) for x, y in zip(self.x_vals, self.y_vals)]
        pygame.draw.lines(screen, (0,0,255), False, pts, 2)

        # Offset + ancho
        pts1 = [self.world_to_screen(x, y) for x, y in zip(self.x1, self.y1)]
        pygame.draw.lines(screen, (255,0,0), False, pts1, 2)

        # Offset - ancho
        pts2 = [self.world_to_screen(x, y) for x, y in zip(self.x2, self.y2)]
        pygame.draw.lines(screen, (255,0,0), False, pts2, 2)

        pygame.display.flip()
