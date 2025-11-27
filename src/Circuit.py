from sympy import diff, sqrt, symbols
from sympy import lambdify

import numpy as np
import pygame

class Circuit:
    def __init__(self, x_func, y_func, var_symbol, variable_start=0, variable_finish=10, circuit_width=1):
        self.function = [x_func,y_func]
        self.var = var_symbol

        self.variable_start = variable_start
        self.variable_finish = variable_finish

        self.dx = diff(x_func,self.var)
        self.dy = diff(y_func, self.var)
        self.ddx = diff(self.dx, self.var)
        self.ddy = diff(self.dy, self.var)
        self.tangent_norm = sqrt(self.dx**2 + self.dy**2)
        self.normal_vector = [-self.dy/self.tangent_norm, self.dx/self.tangent_norm]

        self.circuit_width = circuit_width

        self.x_off1 = x_func + circuit_width * self.normal_vector[0]
        self.y_off1 = y_func + circuit_width * self.normal_vector[1]
        self.x_off2 = x_func - circuit_width * self.normal_vector[0]
        self.y_off2 = y_func - circuit_width * self.normal_vector[1]

        # Lambdify
        self._Cx    = lambdify(self.var, x_func, "numpy")
        self._Cy    = lambdify(self.var, y_func, "numpy")
        self._Cdx   = lambdify(self.var, self.dx, "numpy")
        self._Cdy   = lambdify(self.var, self.dy, "numpy")
        self._Cddx  = lambdify(self.var, self.ddx, "numpy")
        self._Cddy  = lambdify(self.var, self.ddy, "numpy")

        # Lambdify de offsets
        self._x1f = lambdify(self.var, self.x_off1, "numpy")
        self._y1f = lambdify(self.var, self.y_off1, "numpy")
        self._x2f = lambdify(self.var, self.x_off2, "numpy")
        self._y2f = lambdify(self.var, self.y_off2, "numpy")

        self.ts = None
        self.x_vals = None
        self.y_vals = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None

        self.calculate_draw_points()

        self.scale = 1
        self.x_min = 0
        self.y_min = 0
        self.width = 1
        self.height = 1


    def calculate_draw_points(self, delta_s=0.05):
        t_temp = np.linspace(self.variable_start, self.variable_finish, 5000)
        x_temp = np.array([float(self.function[0].subs(self.var, t)) for t in t_temp])
        y_temp = np.array([float(self.function[1].subs(self.var, t)) for t in t_temp])

        dx = np.diff(x_temp)
        dy = np.diff(y_temp)
        ds = np.sqrt(dx ** 2 + dy ** 2)
        L = ds.sum()  # longitud total

        # Cantidad óptima de puntos
        n_samples = max(100, int(L / delta_s))
        # -----------------------------
        # Construir ts definitivo
        # -----------------------------
        ts = np.linspace(self.variable_start, self.variable_finish, n_samples)

        self.ts = ts

        # Valores base de la curva
        self.x_vals = self._Cx(ts)
        self.y_vals = self._Cy(ts)

        # Offsets
        self.x1 = self._x1f(ts)
        self.y1 = self._y1f(ts)
        self.x2 = self._x2f(ts)
        self.y2 = self._y2f(ts)


    def get_width(self):
        return self.circuit_width

    def get_start(self):
        x0, y0 = float(self.function[0].subs(self.var,self.variable_start)), float(self.function[1].subs(self.var,self.variable_start))
        return x0, y0

    def angle_at_start(self):
        return self.angle_of_curve(self.variable_start)

    def fit(self, width, height):

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

    def nearest_curve_point(self, px, py, newton_steps=30, tol=1e-8):
        """
        Encuentra el punto en la curva que minimiza la distancia a (px, py).
        Combina sampling + Newton para mayor estabilidad.

        Retorna: (t_min, x(t_min), y(t_min)).
        """

        ts = self.ts
        xs = self._Cx(ts)
        ys = self._Cy(ts)

        d2 = (xs - px) ** 2 + (ys - py) ** 2
        t_curr = self.ts[np.argmin(d2)]

        # --------------- 2. NEWTON PARA REFINAR ----------------
        for i in range(newton_steps):
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
            fp = (Cdx * Cdx + Cdy * Cdy) + (rx * Cddx + ry * Cddy)

            if abs(fp) < 1e-12:
                break

            t_new = t_curr - f / fp
            t_new = np.clip(t_new, self.variable_start, self.variable_finish)

            if abs(t_new - t_curr) < tol:
                t_curr = t_new
                break

            t_curr = t_new

        # Resultado final
        return t_curr, float(self._Cx(t_curr)), float(self._Cy(t_curr))

    def angle_of_curve(self, t):
        dx_val = float(self.dx.subs(self.var, t))
        dy_val = float(self.dy.subs(self.var, t))
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
